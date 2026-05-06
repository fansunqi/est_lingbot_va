# wan_va_config.json 样本配置

选择一套样本配置复制到数据集 `meta/wan_va_config.json`，按需修改。

```bash
cp src/data/samples/wan_va_config.<场景>.json /path/to/your_dataset/meta/wan_va_config.json
```

> **快速选择**：单臂 → `demo.json`；双臂 → `dual.json`。

---

## preprocess（修改后需重新提取 latent）

| 字段 | 说明 | 必须改？ |
|---|---|---|
| `obs_cam_keys` | 数据集中的相机列名 | **是**，须与 LeRobot 数据集 features 中的视频键一一对应 |
| `frame_stride` | 每隔多少原始帧取一帧送入 VAE | 按需调整，见下方指南 |
| `camera_preset` | 相机预设名称 | 选好场景后一般不需要改 |

- **`obs_cam_keys` 顺序**：首个相机为主视角（top / high），其余为腕部视角；双腕时先左后右。
- **`frame_stride` 选取原则**：使 `actual_fps = dataset_fps / frame_stride` 落在 **5–15 fps** 之间，10 fps 附近为佳。

## training（修改后无需重新提取 latent）

| 字段 | 说明 | 必须改？ |
|---|---|---|
| `latent_layout` | latent 拼接方式，须与 `camera_preset` 匹配 | 不改 |
| `action_transform` | action 变换；非 RoboTwin 固定 `identity` | 不改 |
| `action_dim` | 模型空间宽度，**固定 30** | 不改 |
| `action_keys` | 从数据集读取 action 的字段名列表 | **是** |
| `used_action_channel_ids` | 嵌套列表，每个字段到 30 维模型空间的通道映射 | **是** |
| `action_norm_method` | 归一化方法，目前仅 `quantiles` | 不改 |
| `norm_stat` | `identity` 时**禁止填写**（自动读取 `meta/stats.json`）；非 `identity` 时**必须填写** | 条件 |

---

## action_keys 与 used_action_channel_ids

`action_keys` 指定读取哪些字段；`used_action_channel_ids` 是等长的嵌套列表，每个子列表将对应字段的每个 raw 维度映射到模型空间 `[0, 30)` 的位置。`null` 表示跳过该维度。子列表长度须等于字段 flatten 后的维度数，所有非 null 值全局不可重复。

```jsonc
// 单字段数据集
"action_keys": ["action"],
"used_action_channel_ids": [[0, 1, 2, 3, 4, 5, 28]]

// 多字段数据集
"action_keys": ["actions.end.position", "actions.end.orientation", "actions.effector.position"],
"used_action_channel_ids": [
  [0, 1, 2, 7, 8, 9],              // position  (2,3)→6D
  [3, 4, 5, 6, 10, 11, 12, 13],    // orientation (2,4)→8D
  [28, 29]                          // effector  (2,)→2D
]

// 部分选择 — null 跳过
"used_action_channel_ids": [[0, null, 2, 7, null, 9]]
```

---

## 30 维 action 语义

| 维度 | 内容 |
|---|---|
| 0–6 | 左臂 EEF：x, y, z, qx, qy, qz, qw |
| 7–13 | 右臂 EEF：x, y, z, qx, qy, qz, qw |
| 14–20 | 左臂关节角（本版本留空）|
| 21–27 | 右臂关节角（本版本留空）|
| 28 | 左手夹爪 |
| 29 | 右手夹爪 |

> **关于旋转表示**：部分数据集（如 LIBERO）的 EEF 使用旋转向量（3 维）而非四元数。
> 当前做法是将旋转向量填入 qx, qy, qz 位置，qw 留空。待完成数据清洗，后续版本将提供显式的转换接口。

---

## 相机预设

| 预设名 | 相机数 | 各相机 Resize | 对应 latent_layout |
|---|---|---|---|
| `one_primary_one_wrist_256` | 2 | 256×256, 256×256 | `horizontal_concat` |
| `one_primary_one_wrist_128` | 2 | 128×128, 128×128 | `horizontal_concat` |
| `one_primary_two_wrist_224x320` | 3 | 224×320, 224×320, 224×320 | `horizontal_concat` |
| `one_primary_two_wrist_tshape_256x320` | 3 | 256×320, 128×160, 128×160 | `robotwin_tshape` |

> **快速选择**：单臂 → `one_primary_one_wrist_256` + `horizontal_concat`；双臂 → `one_primary_two_wrist_tshape_256x320` + `robotwin_tshape`。

---

## 完整流程

```bash
# 1. 复制样本配置
cp src/data/samples/wan_va_config.demo.json \
   /path/to/your_dataset/meta/wan_va_config.json

# 2. 编辑 obs_cam_keys、action_keys、used_action_channel_ids
vim /path/to/your_dataset/meta/wan_va_config.json

# 3. 提取 latent
python -m src.data.extract_latents \
    --dataset-root /path/to/your_dataset \
    --model-path   /path/to/pretrained_wan \
    --num-gpus 6
# --num-gpus 0（默认）使用所有可见 GPU。
# 如果发现 CPU 占用率接近 100% 且部分 GPU 利用率为 0%，
# 说明并行 worker 过多，请适当调低 --num-gpus。
# 中断后直接重跑同一命令即可；已完成部分不会重复处理。
# 坏帧解码错误会自动跳过，训练时也会忽略，后续如有新的过滤模式可以添加；跳过很多时请先检查原始数据。

# 4. 开始训练
```
