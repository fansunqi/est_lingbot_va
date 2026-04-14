# LingBot-VA Latent 数据设计

## 目标

直接使用 LeRobot 0.5 的元数据，并让 LingBot-VA 在训练时使用预先计算好的 latent，而不在训练过程中解码视频。

## 核心规则

不要再维护第二份 segment 元数据文件。

训练 segment 应在运行时根据 LeRobot 元数据构建。

这样可以避免出现重复的事实来源。

每个数据集只保留两个文件：

- `meta/wan_va_config.json`
- `latents/metadata.json`

`meta/wan_va_config.json` 由用户编辑，可以包含：

- 预处理输入，例如相机键、预定义相机预设、目标 FPS
- 训练期的数据集规则，例如 latent 布局、action 重映射和 action 归一化

`latents/metadata.json` 由预处理阶段固化生成，不应手动编辑。它只包含解释已提取 latent 数据所需的 latent 格式事实。

`latents/metadata.json` 还携带提取状态：

- `extracting`：预处理正在进行，或曾被中断，必须继续恢复
- `complete`：所有 latent 文件与文本嵌入文件都已写入并完成校验

训练阶段：

- 信任 `latents/metadata.json` 作为 latent 解释依据
- 从 `meta/wan_va_config.json` 读取训练期的数据集规则
- 如果 `meta/wan_va_config.json` 中的 preprocess 部分与 `latents/metadata.json` 不再一致，则立即失败
- 如果 `latents/metadata.json` 状态不是 `complete`，则立即失败

## Segment 构建

在每个数据集初始化时构建 segment：

- 只批量读取 `hf_dataset` 中所需的索引列，不要逐帧读取
- 优先使用轻量视图，例如 `episode_index`、`frame_index`、`task_index`，以及可选的 `subtask_index`
- 如果存在 `subtask_index` 且 `meta.subtasks` 可用：
  - 按连续相同的 `subtask_index` 区间切分每个 episode
  - 使用对应的 subtask 文本
- 否则：
  - 一个 episode 对应一个 segment
  - 使用该 episode 的 task 文本

同一套 segment 构建逻辑需要同时用于 latent 提取和训练时加载。

数据集结构检查应发生在预处理阶段，而不是训练热路径中。具体来说，预处理阶段要验证：

- `frame_index` 在每个 episode 内是稠密且从 0 开始的
- `task_index` 存在且未越界
- `subtask_index` 若存在，则未越界
- 当不存在 `subtask_index` 时，每个 episode 内的 `task_index` 必须是常量
- `obs_cam_keys` 的数量和顺序与所选预定义相机预设一致
- 所选相机预设在 latent 空间中的尺寸与配置的 `latent_layout` 兼容

每个运行时 segment 只需要包含：

- `episode_index`
- `start_frame`
- `end_frame`
- `global_from`
- `global_to`
- `task_index`
- `subtask_index`（可选）

`global_from` 和 `global_to` 来自 LeRobot 元数据中该 episode 的全局索引范围，再结合该 segment 的局部帧范围得到。

## Latent 文件命名

latent 文件遵循以下命名约定：

- `episode_{episode_index:06d}_{start_frame}_{end_frame}.pth`

运行时计算出的 segment 边界必须与 latent 提取时使用的边界完全一致。

如果 task 或 subtask 标注发生变化，那么此前提取出的 latent 文件将不再匹配运行时 segment，必须重新生成。

训练阶段应信任预处理后的数据集布局，而不是在初始化时再次检查 latent 覆盖率。

预处理结束前，应做一次低成本的完整性检查：在将 `latents/metadata.json` 标记为 `complete` 之前，确认所有预期的 latent 路径都存在。

## 推荐目录布局

```text
dataset_root/
├── data/                      # LeRobot 原生数据
├── videos/                    # LeRobot 原生视频，训练时不使用
├── meta/
│   ├── info.json
│   ├── wan_va_config.json     # 可编辑：预处理 + 训练数据集配置
│   ├── tasks.parquet
│   ├── subtasks.parquet       # 可选
│   ├── episodes/...
│   └── stats.json
├── latents/
│   ├── metadata.json          # 固化的 latent 格式元数据，不要编辑
│   ├── observation.images.top/
│   └── observation.images.wrist/
└── text_emb/
    ├── task_emb.pth
    ├── subtask_emb.pth        # 可选
    └── empty_emb.pth
```

latent 路径不需要镜像 LeRobot 的 chunk 布局。

更推荐扁平的按相机布局，例如：

- `latents/observation.images.top/episode_000000_0_264.pth`
- `latents/observation.images.wrist/episode_000000_0_264.pth`

## Latent 文件内容

latent 文件应尽量保持精简。

内容包括：

- `latent`
- `frame_ids`
- `latent_num_frames`
- `latent_height`
- `latent_width`
- `fps`

`frame_ids` 记录的是 VAE 编码之前实际采样到的原始帧。它在提取阶段用于验证同一 segment 的所有相机是否具有完全一致的时间采样。训练阶段不会读取它：`frame_stride` 直接由 `preprocess_config.frame_stride` 获取，并结合 `latents/metadata.json` 中固化的 `actual_fps` 使用。

`fps` 记录 latent 提取时的实际采样帧率（`dataset_fps / frame_stride`）。它主要用于人工检查和一致性校验，在训练热路径中不会被直接读取。

主要的空间节省来自于不再在每个 latent 文件里重复存储 `text_emb` 张量。

不要在每个 latent 文件中存储重复的业务元数据。

只有 `meta/wan_va_config.json` 中的 preprocess 子集会被冻结写入 `latents/metadata.json`。

action 重映射、归一化统计、latent 布局和 action 后处理属于可编辑的 training 部分，因为修改它们不需要重新提取 latent。

## 相机预设

与其让用户手写任意的逐相机 resize 值，预处理阶段应暴露一组固定的相机预设，以匹配历史训练配置：

| 预设名 | 相机数量 | Resize 分辨率 | 布局 |
|---|---|---|---|
| `one_primary_one_wrist_256` | 2 | 256×256, 256×256 | `horizontal_concat` |
| `one_primary_one_wrist_128` | 2 | 128×128, 128×128 | `horizontal_concat` |
| `one_primary_two_wrist_224x320` | 3 | 224×320, 224×320, 224×320 | `horizontal_concat` |
| `one_primary_two_wrist_tshape_256x320` | 3 | 256×320, 128×160, 128×160 | `robotwin_tshape` |

相机 0 永远是主视角，后续相机视为 wrist 视角。

每个预设都定义了每个相机在 VAE 编码前应采用的精确 resize，以及与该几何布局兼容的 latent 布局。几何不变量，例如 `horizontal_concat` 需要 latent 高度一致，`robotwin_tshape` 需要宽度求和匹配且 wrist 高度一致，会在模块加载时通过 `CameraPresetSpec.__post_init__` 一次性校验。

## 文本嵌入

为每个数据集单独缓存文本嵌入：

- `task_emb.pth`：按本地 `task_index` 索引的堆叠张量
- `subtask_emb.pth`：按本地 `subtask_index` 索引的堆叠张量
- `empty_emb.pth`：用于 CFG dropout

这些缓存应保持数据集本地化，而不是跨混合数据集全局共享。

训练时：

- 若存在 subtasks，则使用 `subtask_index` 查找 `subtask_emb.pth`
- 否则使用 `task_index` 查找 `task_emb.pth`
- 通过将选中的 embedding 替换为 `empty_emb` 实现 CFG dropout

## 训练路径

不要使用 `LeRobotDataset.__getitem__()` 来构建 segment。

只使用：

- `dataset.meta`（通过 `LeRobotDatasetMetadata` 获取）
- `dataset.hf_dataset`（通过 `load_nested_dataset` 单独加载）

原因：

- `LeRobotDataset.__getitem__()` 在存在视频键时会解码视频帧
- `LeRobotDatasetMetadata` 和 parquet 数据已经足够提供所需元数据和 action 表，而无需视频解码

训练路径应为：

1. 根据 LeRobot 元数据构建 segment
2. 从 `latents/` 加载 latent 张量
3. 从 LeRobot parquet 数据加载 actions
   - 初始化时，仅保留所需列的轻量索引视图
   - 在 `__getitem__` 中，保留一个单独的 tensor 格式 action 视图
   - 使用从 `episodes["dataset_from_index"]` 推导出的 `global_from` 与 `global_to` 读取 action 区间
4. 训练过程中绝不解码视频

训练阶段应信任固化的 latent 元数据，并避免在热路径中重复进行 latent 文件检查。只保留常数成本的漂移检查：

- `meta/wan_va_config.json.preprocess` 仍然匹配 `latents/metadata.json.preprocess`
- 当前数据集的 `codebase_version` 以及推导出的 `actual_fps` 仍然匹配 `latents/metadata.json`
- 混合数据集之间仍然具备 batch 兼容性

## 多相机布局

latent 仍按相机分别加载，而多相机拼接仍然是数据集级规则。

拼接策略应保留在每个数据集自己的配置中，因为不同环境可能使用不同布局。

帧对齐由预处理阶段保证，而不是在训练阶段重复验证。

在预处理阶段，每个 segment 的 latent 写入过程都要验证：同一 segment 下所有相机共享相同的 `frame_ids`，且具有相同的 temporal latent 长度。

布局兼容性，也就是所选布局在 latent 空间中的几何兼容性，由 `CameraPresetSpec.__post_init__` 在模块加载时校验，而不在训练阶段重复检查。

## 多数据集说明

每个数据集都应各自维护：

- segment 构建
- 文本嵌入缓存
- 相机配置
- 到统一训练 action 空间的 action 维度映射
- action 归一化统计与规则

混合训练应在这一层之上组合多个数据集，而不是让 task 或 subtask 索引在全局共享。

跨数据集的采样权重应在多数据集混合层处理，而不是内嵌进单个数据集的 latent 格式中。
