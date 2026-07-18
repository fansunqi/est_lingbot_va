# LingBot-VA RoboTwin2.0 复现笔记 — ja25-36 4090 集群

> 在 8×RTX4090(24GB) K8s 集群上复现 posttrain-robotwin 的 RoboTwin2.0 eval。
> 单任务冒烟已通过(2026-07-18)。ssh: `ssh -i ~/.ssh/id_rsa_thinpad quser@zhehanmo.cn -p <port>`(zhehanmo.cn=25MB/s网关;直连IP 0.05MB/s勿用)。

## 两个 conda 环境(装本地盘 /home/quser/miniconda3,勿放NFS)
- **lingbot**(py3.12):模型 server。torch 2.10+cu128(pypi/清华,勿用被墙的 download.pytorch.org)。ckpt 里 transformer/config.json 的 attn_mode 已是 "torch"(eval需要)。robotwin.yaml 已改:model_name_or_path→本地权重、enable_offload/vae_offload=true。
- **RoboTwin**(py3.10):仿真 client。关键版本(必须一起成立):
  - setuptools<81(否则 sapien 报 no pkg_resources)
  - pip 约束 numpy==1.26.4 + opencv-python==4.10.0.84(否则 pip 疯狂回溯 opencv)
  - torch 2.4.1+cu121 + conda cuda-toolkit=12.1(nvcc,编译用)
  - **curobo v0.7.6**(NVlabs),不是 main(2026重构版会让 RoboTwin 导入 CuroboPlanner 失败)。tarball 安装需 `SETUPTOOLS_SCM_PRETEND_VERSION=0.7.6 pip install -e . --no-build-isolation`
  - **warp-lang==1.0.2**(curobo0.7.6 需 wp.torch;新版warp已移除)
  - conda ffmpeg 二进制(录eval视频;pip的"ffmpeg"包不是二进制)
  - msgpack(websocket client)

## 运行时关键点
- SAPIEN 需 `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`(NVIDIA Vulkan ICD 在此,非默认目录)。
- RoboTwin envs/_base_task.py 的 `set_ray_tracing_denoiser("oidn")` → 改 **"none"**:与12GB模型共卡时 oidn/optix 降噪器都崩(illegal memory / OPTIX_ERROR)。
- **⚠️ GPU布局(扩容关键):server 推理峰值~18-19GB + client(curobo+渲染)~4.6GB,同卡在24GB上会系统性OOM(第2个episode起)。修法:server与client分到不同GPU(每任务2卡),不牺牲精度。→ 每8卡节点=4任务并行,8节点=32路并行。**勿降 spp/attn_window/步数来凑(会改精度)。
- 下载:HF被墙→`HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1`;pip走清华;github慢→ghfast.top tarball;权重也在ModelScope(Robbyant/lingbot-va-posttrain-robotwin)。
- quser uid各节点不同(ja25/26=1101,ja33=1001)→ NFS共享写目录需 chmod 1777。

## 复用脚本(ja26)
- /home/quser/run_servers_sep.sh — 4 server 在 GPU 0/2/4/6,端口 29556-29559
- /home/quser/run_clients_sep.sh — 4 client 渲染在 GPU 1/3/5/7
- server: lingbot环境 python -m src.inference.server --config configs/inference/robotwin.yaml --port <p>
- client: RoboTwin环境 cd /home/quser/RoboTwin, 设PYTHONPATH=仓库+ROBOTWIN_ROOT+VK_ICD_FILENAMES, python -m evaluation.robotwin.eval_polict_client_openpi --task_name <t> --task_config demo_clean --policy_name ACT --port <p>
