# wan_va_config.json 样本配置

本目录提供多套样本配置，选择一套复制到数据集根目录下 `meta/wan_va_config.json`，然后按需修改。

```bash
cp wan_va/dataset/samples/wan_va_config.<场景>.json /path/to/your_dataset/meta/wan_va_config.json
```

> **快速选择**：单臂 → `demo.json`；双臂 → `dual.json`。

---

## 你需要修改什么

### preprocess（预处理 — 修改后需重新提取 latent）

| 字段 | 说明 | 必须改？ |
|---|---|---|
| `obs_cam_keys` | 数据集中的相机列名 | **是**，须与 LeRobot 数据集 features 中的视频键一一对应 |
| `frame_stride` | 每隔多少原始帧取一帧送入 VAE | 按需调整，见下方指南 |
| `camera_preset` | 相机预设名称 | 选好场景后一般不需要改 |

- **`obs_cam_keys` 顺序**：首个相机为主视角（top / high），其余为腕部视角；双腕时先左后右。
- **`frame_stride` 选取原则**：使 `actual_fps = dataset_fps / frame_stride` 落在 **5–15 fps** 之间，10 fps 附近为佳。

### training（训练 — 修改后无需重新提取 latent）

| 字段 | 说明 | 必须改？ |
|---|---|---|
| `latent_layout` | latent 拼接方式，须与 `camera_preset` 匹配 | 一般不需要改 |
| `action_transform` | action 变换方式 | 非 RoboTwin 数据集**固定** `identity` |
| `action_dim` | 模型空间 action 宽度 | **固定为 30，不可修改** |
| `used_action_channel_ids` | 有效 action 通道索引 | **是**，按数据集实际 action 语义调整，单臂默认为左臂 |
| `action_norm_method` | 归一化方法 | 目前仅支持 `quantiles` |
| `norm_stat` | 各通道的 1% / 99% 分位数 | 见下方说明 |

> **`norm_stat` 规则**：
> - 当 `action_transform` 为 `"identity"` 时，**不要填写** `norm_stat`（填了会报错）。训练时会自动从数据集的 `meta/stats.json` 读取 `action` 字段的 `q01` / `q99` 分位数。
> - 当 `action_transform` 不是 `"identity"`（如 `"robotwin_relative_pose_bimanual"`）时，**必须手动填写** `norm_stat`，因为 action 变换改变了原始 action 空间，数据集中的统计量不再适用。此时 `q01` 和 `q99` 的长度必须等于 `action_dim`（30），未使用的通道填 `0`。

### 30 维 action 语义约定

| 维度 | 内容 |
|---|---|
| 0–6 | 左臂末端执行器（EEF）：x, y, z, qx, qy, qz, qw |
| 7–13 | 右臂末端执行器（EEF）：x, y, z, qx, qy, qz, qw |
| 14–20 | 左臂关节角 joint 1–7（本版本不使用，强制留空）|
| 21–27 | 右臂关节角 joint 1–7（本版本不使用，强制留空）|
| 28 | 左手夹爪 |
| 29 | 右手夹爪 |

> **关于旋转表示**：部分数据集（如 LIBERO）的 EEF 使用旋转向量（3 维）而非四元数。
> 当前做法是将旋转向量填入 qx, qy, qz 位置，qw 留空。待完成数据清洗，后续版本将提供显式的转换接口。

---

## 可用的相机预设

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
# 1. 复制样本配置到数据集目录
cp wan_va/dataset/samples/wan_va_config.demo.json \
   /path/to/your_dataset/meta/wan_va_config.json

# 2. 按需编辑 obs_cam_keys、used_action_channel_ids 等
#    （identity 场景无需填写 norm_stat，会自动从 meta/stats.json 读取）
vim /path/to/your_dataset/meta/wan_va_config.json

# 3. 提取 latent
python -m wan_va.dataset.extract_latents \
    --dataset-root /path/to/your_dataset \
    --model-path   /path/to/pretrained_wan

# 4. 开始训练（训练代码会自动读取 meta/wan_va_config.json）
```
