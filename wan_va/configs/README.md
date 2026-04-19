# wan_va/configs 配置指南

本目录的 Python 配置文件控制训练超参、推理参数和模型路径。
每个数据集自身的数据规则（相机、action 映射、归一化等）在 `meta/wan_va_config.json` 中配置，详见 [`wan_va/dataset/README.md`](../dataset/README.md)。

---

## 文件结构与适用阶段

配置文件读取链：

```
                                  ┌─▶ *_train_cfg.py ─▶ 训练
shared_config.py ─▶ va_*_cfg.py ──┼───────────────────▶ 推理
                                  └─▶ *_i2va.py ──────▶ i2va
```

| 阶段 | 本目录配置 | 数据集 JSON |
|---|---|---|
| 预处理 (`extract_latents`) | - | `meta/wan_va_config.json`，详见 [`wan_va/dataset/README.md`](../dataset/README.md) |
| 训练 (`train.py`) | `va_*_train_cfg.py` | `meta/wan_va_config.json` |
| 推理 (`wan_va_server.py`) | `va_*_cfg.py` | - |
| i2va (`wan_va_server.py`) | `va_*_i2va.py` | - |

`--config-name` 可用的注册名见 [`__init__.py`](./__init__.py) 中的 `VA_CONFIGS`。
命名规则：`<场景>` = 推理，`<场景>_train` = 训练，`<场景>_i2av` = i2va。

---

## 字段说明

### `shared_config.py`

所有配置的公共基础。`-` 表示该字段在对应阶段不生效，修改无效果。

| 字段 | 训练 | 推理/i2va | 说明 |
|---|:---:|:---:|---|
| `param_dtype` | ✓ | ✓ | 模型精度 |
| `save_root` | ✓ | ✓ | 检查点 / 输出保存路径 |
| `patch_size` | ✓ | ✓ | latent 分块尺寸 |
| `host` | - | 仅推理 | server 绑定地址 |
| `port` | - | 仅推理 | server 绑定端口 |
| `enable_offload` | - | ✓ | 将 VAE 和 text_encoder 卸载到 CPU（节省显存） |

### `va_*_cfg.py` — 基础配置

| 字段 | 训练 | 推理/i2va | 说明 |
|---|:---:|:---:|---|
| `wan22_pretrained_model_name_or_path` | ✓ | ✓ | 预训练模型路径 |
| `snr_shift` | ✓ | ✓ | 视频 FlowMatchScheduler shift |
| `action_snr_shift` | ✓ | ✓ | action FlowMatchScheduler shift |
| `action_dim` | ✓ | ✓ | 模型 action 空间维度，必须与各数据集中的值一致 |
| `infer_mode` | - | 仅推理 | 路由 `server` 模式（i2va 由 `*_i2va.py` 覆写） |
| `attn_window` | - | ✓ | transformer 缓存窗口大小 |
| `frame_chunk_size` | - | ✓ | 每次推理的帧块大小 |
| `env_type` | - | ✓ | 环境类型（`none` / `robotwin_tshape`） |
| `height` / `width` | - | ✓ | 推理视频分辨率 |
| `action_per_frame` | - | ✓ | 推理每帧 action 步数 |
| `obs_cam_keys` | - | ✓ | 推理相机键列表 |
| `guidance_scale` | - | ✓ | 视频 CFG 引导系数 |
| `action_guidance_scale` | - | ✓ | action CFG 引导系数 |
| `num_inference_steps` | - | ✓ | 视频去噪步数 |
| `video_exec_step` | - | ✓ | 视频执行步控制 |
| `action_num_inference_steps` | - | ✓ | action 去噪步数 |
| `used_action_channel_ids` | - | ✓ | 推理时从模型输出中按序选取的活跃 action 通道 |
| `inverse_used_action_channel_ids` | - | ✓ | 推理时将观测 action 映射回模型空间的索引 |
| `action_norm_method` | - | ✓ | 推理归一化方式 |
| `norm_stat` | - | ✓ | 推理 q01/q99 统计量 |

### `va_*_train_cfg.py` — 训练配置

基础配置中的字段生效情况见上方表格。以下为训练专属字段：

| 字段 | 说明 |
|---|---|
| `dataset_path` | 数据集目录，可以是单个数据集根或包含多数据集的父目录（递归搜索） |
| `enable_wandb` | 启用 Weights & Biases 日志 |
| `load_worker` | DataLoader worker 数 |
| `save_interval` | 每 N 步保存检查点 |
| `gc_interval` | 每 N 步触发 GC + 清空 CUDA 缓存 |
| `cfg_prob` | CFG dropout 概率（以此概率将 text embedding 替换为 empty_emb） |
| `learning_rate` | AdamW 学习率 |
| `beta1` / `beta2` | AdamW beta 参数 |
| `weight_decay` | 权重衰减 |
| `warmup_steps` | 学习率预热步数 |
| `batch_size` | 每卡 batch 大小 |
| `gradient_accumulation_steps` | 梯度累积步数 |
| `num_steps` | 总训练步数 |

### `va_*_i2va.py` — i2va 配置

基础配置中的字段生效情况见上方表格。以下为 i2va 专属字段：

| 字段 | 说明 |
|---|---|
| `input_img_path` | 初始观测图片目录（每个相机一张 `{cam_key}.png`） |
| `num_chunks_to_infer` | 自回归推理块数 |
| `prompt` | 任务指令文本 |
| `infer_mode` | 覆写为 `'i2va'`（基础配置中为 `'server'`） |

---

## `wan_va_config.json` 与本目录的关系

训练时，`obs_cam_keys`、`used_action_channel_ids`、`action_norm_method`、`norm_stat` 等数据规则**均从每个数据集的 JSON 读取**，不从本目录的 Python 配置读取。修改 JSON 的 `preprocess` 段后必须重新提取 latent，而 `training` 段可自行修改。详见 [`wan_va/dataset/README.md`](../dataset/README.md)。

---

## 典型流程

```bash
# 1. 准备数据集 JSON 配置 + 提取 latent（详见 wan_va/dataset/README.md）

# 2. 训练
NGPU=8 CONFIG_NAME='robotwin_train' bash script/run_va_posttrain.sh

# 3. 推理
NGPU=1 CONFIG_NAME='robotwin' bash script/run_launch_va_server_sync.sh

# 4. i2va
NGPU=1 CONFIG_NAME='robotwin_i2av' bash script/run_launch_va_server_sync.sh
```
