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

## 运维教训(踩过的坑)
- **eval 跑完必须显式 kill server(按 pid)**:残留 server 占端口 29556-29559 / master 29661,下一轮启动 EADDRINUSE 崩溃、脚本卡在"等 listening"、client 不启动。收尾要清进程。
- **`pkill -f <pat>` / `pgrep -f` 会自匹配**:在远端 `bash -c "...pat..."` 里执行时,模式匹配到自己的父 shell(命令行含该字符串)→ 杀掉自己的 SSH 会话(exit 255)。对策:按具体 pid 杀;或确认无自匹配再用。
- **节点重启后 IPoIB 的 IP + NFS 挂载都会丢**(不在 fstab、IP 非持久):`sudo ip addr add 172.10.24.N/24 dev ibp194s0 && sudo ip link set ibp194s0 up` + `sudo mount -t nfs 172.10.24.25:/srv/share /mnt/share`。
- **eval 用 2 卡/任务**:server 峰值~18GB+client~5GB 同卡在 24GB 会 OOM;分卡(server 偶数卡、client 奇数卡)彻底解决。
- **多节点复制**:节点间默认不能互 SSH;在 ja26 生成 `~/.ssh/id_ed25519_cluster`、分发公钥到各节点 authorized_keys 后,用 IPoIB(172.10.24.N)tar-stream 分发(ja26 1TB 内存,源进 page cache 后并行很快)。
- **脚本里勿用 `set -u`(nounset)包住 `conda activate`**:conda 的 gcc 激活脚本(cuda-toolkit 带来的)引用未定义变量 `SYS_SYSROOT`,nounset 下报错退出。症状:server 起了但 client 全不启动(脚本在 activate RoboTwin 处死)。对策:不用 set -u,或 `set +u` 后再 activate。
- **NFS 服务器节点(ja25)的共享盘在 `/srv/share`,不是 `/mnt/share`**:假设 `/mnt/share` 的脚本在 ja25 上找不到文件而失败。对策:在 ja25 上 `sudo mount --bind /srv/share /mnt/share`,让路径全集群统一。
- **多机批量启动后必须逐机核对 client 真起来了**(client 日志存在 + 奇数卡有显存),别只看 server;不同节点可能因路径/环境差异部分失败。
- **复制 conda 环境不够,必须连带复制 `~/.local` 和 editable 安装的源码目录**:症状连环 ModuleNotFoundError(requests→typing_extensions→…)。原因:(1) 早期用 `pip install --user`(系统py) 把 requests/huggingface_hub/modelscope 等装进 `~/.local`,conda RoboTwin(py3.10)经 user-site 借用;复制只传了 miniconda3 没传 ~/.local。(2) curobo/pytorch3d 是 `pip install -e`(editable)指向 `/home/quser/{curobo_check,pytorch3d_src}`,复制没带这两个源码目录→import 失败。**正确复制清单**:miniconda3 + RoboTwin + models + **curobo_check + pytorch3d_src + ~/.local**。验证:每节点跑 `python -c "import sapien,curobo,pytorch3d,mplib,pydantic,requests,typing_extensions"` 应全 OK 再启动。更稳做法:环境自包含(装进 env、非 --user、非 -e)。
- **部分节点缺 `/etc/vulkan/icd.d/nvidia_icd.json`**(如 ja33):SAPIEN 报 "failed to find a rendering device" / client 打印 "Render Error"。各节点 Vulkan ICD 状态可能不一致(非集群统一)。修:创建该文件(内容:file_format_version 1.0.1 / library_path libGLX_nvidia.so.0 / api_version 1.4.329),libGLX_nvidia.so.0 各节点都在。⚠️注意 `printf|(echo pw|sudo -S tee)` 会让密码 stdin 和内容冲突→写错;用"先写/tmp再 sudo cp"。铺开前应逐节点验证渲染(Sapien_TEST/最小渲染)。

## 实验结论:8 机一致性(2026-07-18)
turn_switch / demo_randomized / action 5步 / 每台相同 100 seed / denoiser=none / 2卡每任务。8 台 SR:ja25 52%, ja26 50%, ja36 48%, ja34 45%, ja35 ~42%(81集,一个 client rollout 挂住), ja27 37%, ja28 36%, ja33 36%。**极差 16%(52→36),n=100 时 σ≈5% → ≈3σ,是真实机器间差异而非噪声。** 驱动 570/595 在高低档都有,非单一主因。推测主因:SAPIEN 光追(32spp 无降噪)在不同 GPU/驱动微状态下观测图像细微差异 → 5步策略敏感 → 部分 seed 成败翻转;GPU 浮点非确定性叠加。待深挖(逐 seed 翻转 + 观测图像对比 / demo_clean 对照)。

## eval 偶发挂死:SAPIEN 相机渲染 _get_rgba 卡住(2026-07-18)
现象:某 client rollout 卡在 "step N/400" 不动(数十分钟),CPU 低、GPU 0%。py-spy dump 该进程 MainThread 栈:_get_rgba (envs/camera/camera.py:335) ← get_rgba ← get_rgb ← get_obs (_base_task.py:450) ← eval_policy。→ **卡在 SAPIEN 相机图像回读(Vulkan 渲染 fence/readback stall)**,不是 websocket/server/NCCL(配对 server 健康但空闲等请求)。属光追渲染在共享 GPU 上的偶发挂死(与 OIDN/OptiX 崩溃同类,这里是 hang 非 crash)。诊断法:`py-spy dump --pid <client>`(需 sudo)。处理:kill 该 client 重启即可(server 无需重载),多为瞬时非确定性。若高频复发,考虑 get_obs 渲染加超时/重试。
