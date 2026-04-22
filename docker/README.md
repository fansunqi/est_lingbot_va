# lingbot-va Docker 环境

用于在另一台服务器上复刻 evaluation 环境（适配 NVIDIA driver 550 + CUDA 12.4）。

## 环境差异说明

| 组件 | 当前服务器 (g41) | Docker 容器 |
|------|-----------------|-------------|
| CUDA | 12.8 | 12.4 |
| lingbot-va Python | 3.13 | 3.12 |
| torch (server) | 2.9.0+cu128 | 2.9.0+cu124 |
| flash-attn | cu128torch2.9-cp313 | cu124torch2.9-cp312 |
| RoboTwin Python | 3.10 | 3.10 |
| torch (client) | 2.4.1+cu121 | 2.4.1+cu121 |
| 渲染 | 光栅化 (default) | 光栅化 (default) |

## 前提条件

- Docker + NVIDIA Container Toolkit (`nvidia-docker`)
- NVIDIA driver ≥ 550
- RTX 4090 (24GB VRAM)

## 快速开始

### 1. 准备文件

将以下内容拷贝到目标服务器：

```bash
# 代码仓库
scp -r lingbot-va/ remote:/path/to/lingbot-va/
scp -r RoboTwin/ remote:/path/to/RoboTwin/

# 模型权重 (~23GB)
scp -r models/lingbot-va-posttrain-robotwin/ remote:/path/to/models/
```

### 2. 配置环境变量

```bash
cd lingbot-va/docker/
cp .env.example .env
# 编辑 .env，设置 MODEL_PATH 为模型权重的绝对路径
vim .env
```

### 3. 构建镜像

```bash
cd lingbot-va/docker/
docker compose build
```

> ⏱ 首次构建约 30-60 分钟（主要时间在编译 pytorch3d）

### 4. 启动容器

```bash
docker compose up -d
docker exec -it lingbot-eval bash
```

> 首次启动会自动安装 lingbot-va 的 Python 环境（约 5-10 分钟）

### 5. 修改模型路径配置

容器内模型挂载到 `/workspace/models/lingbot-va-posttrain-robotwin`，需要修改配置：

```bash
# 在容器内
sed -i 's|/home/cxy/ocean/models/lingbot-va-posttrain-robotwin/|/workspace/models/lingbot-va-posttrain-robotwin/|' \
    /workspace/lingbot-va/wan_va/configs/va_robotwin_cfg.py
```

### 6. 运行 Evaluation

**方式 A：使用一键脚本**

```bash
bash /workspace/lingbot-va/docker/launch_eval.sh adjust_bottle
```

**方式 B：手动分开运行 Server 和 Client**

终端 1 - Server (lingbot-va):
```bash
cd /workspace/lingbot-va
.venv-docker/bin/python -m torch.distributed.run \
    --nproc_per_node 1 --master_port 29061 \
    wan_va/wan_va_server.py --config-name robotwin \
    --port 29056 --save_root visualization/
```

终端 2 - Client (RoboTwin):
```bash
conda activate RoboTwin
cd /workspace/lingbot-va
export ROBOTWIN_ROOT=/workspace/RoboTwin
bash evaluation/robotwin/launch_client.sh ./results adjust_bottle
```

## 常见问题

### Q: 如何查看 GPU 使用情况？
```bash
nvidia-smi
```

### Q: pytorch3d 编译失败？
pytorch3d 编译需要较多内存和时间。确保构建时有足够内存（建议 ≥16GB RAM）。

### Q: 渲染报错 "cannot create buffer"？
确保容器内 Vulkan 驱动正常：
```bash
vulkaninfo --summary
```

### Q: 如何重新安装 lingbot-va 环境？
```bash
rm -rf /workspace/lingbot-va/.venv-docker
# 重启容器，entrypoint 会自动重新安装
```
