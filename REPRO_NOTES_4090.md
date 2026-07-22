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

### 最终定表(8×100 全完成)
ja25 52% / ja26 50% / ja36 48% / ja34 45% / ja35 43% / ja27 37% / ja28 36% / ja33 36%。均值 43.4%,极差 16%,跨机 std ~6%(>采样σ 5%)→ 真实机器间差异确认。(ja35 曾因 _get_rgba 渲染挂死卡在81,kill 重启 c1 补齐至100。)

## 根因排查①:渲染不是主因(2026-07-18)
ja25 vs ja33 对相同 seed/场景渲染 3 路相机观测,逐像素对比:平均绝对差 0.002-0.005/255,仅 ~0.1% 像素有微差(差>10 的 ~0.01%)→ 观测 99.9% 一致。**渲染差异被排除**。16% SR 差异来自下游:模型 5 步扩散对 GPU 浮点非确定性敏感 → action 微差 → 400 步 rollout 混沌放大 → 成败翻转。待验证:16% 是"机器固有偏差"还是"非确定 eval 的 run-to-run 噪声"(需同机重跑对照)。

## 根因定论:16% 机器间差异 = eval run-to-run 非确定性噪声(非机器/非渲染/非随机化)
三个对照实验汇合:
1. **渲染排除**:ja25(高)vs ja33(低)同 seed 渲染 3 路相机观测,像素级 99.9% 一致(平均差 0.002/255)。
2. **随机化排除**:demo_clean 8机极差 ~12% ≈ demo_randomized ~16%(同量级)——关掉随机背景/光照/杂物,差异没变小。
3. **非机器属性**:demo_clean 相对 demo_randomized 排名彻底洗牌(ja33 36%→44% 由最低变高档;ja26 50%→32% 由最高变最低)——若是机器硬件偏差排名应稳定。
→ 结论:~16% 是**单次 100-seed eval 的 run-to-run 噪声**(模型5步扩散+物理的 GPU 浮点非确定性,经 400 步 rollout 混沌放大),与机器硬件、渲染、随机化均无关。**实践含义:turn_switch 单跑 100-seed 固有抖动 ~±8%;比较 checkpoint/机器需更多 seed 或多次平均,否则真实差异被噪声淹没。** 诊断挂死用 py-spy(见上)。

## GRPO RL 死锁根因定论:多-rank server 的 split-brain update gate(2026-07-19)
**现象**:8-rank DDP server(gs56→gs48)整夜 `global_update_step=0`(一次 optimizer update 都没成、无 checkpoint),最终 rank0 的 watchdog 报 `WorkNCCL(SeqNum=485, OpType=ALLREDUCE, NumelIn=1) ran for 3600043 ms before timing out` → SIGABRT → torchrun 连坐杀光 8 rank。NCCL 日志:rank1-7 "last enqueued 484, last completed 484",rank0 多入队第 485 个 → "the order of collectives is not same for all ranks"。

**根因(源码级)**:`_run_pending_updates()`(src/rl/server.py:1867)是**纯 per-rank 判断**——`if len(self._pending_ready_groups) < rollout_groups_per_update: return`(跳过),否则落入 `_run_grpo_update()`,其第一行就是 `dist.barrier()`(server.py:1088-1089;NCCL 后端下 barrier 就是 NumelIn=1 的 allreduce = 崩溃那个 op)。进入这个 collective **之前没有任何跨-rank 协商**。而且 `_run_pending_updates` 是 **client 驱动**的:只有某 rank 自己的 `client_id==0` coordinator 发来 `command=="run_pending_updates"`(server.py:2896)才执行 → 各 rank 独立异步触发。
member-sharded 下各 rank 的 ready-group 数天然不同、慢 rank 的 client 掉队 → **脑裂**:快 rank(ready≥阈值)进 `_run_grpo_update`→barrier 等齐 8 rank,慢 rank(ready<阈值 / coordinator 未触发)永远不发 barrier → 快 rank 干等 `NCCL_TIMEOUT=3600`(1h)→ 崩。

**为何降阈值 16→4 无效**:门是 per-rank 的,最慢 rank 连 4 都没到,快 rank 早进 barrier。**为何在 `_run_pending_updates` 里加 all_reduce 也修不了**:未触发的 rank 不在 collective 里,照样脑裂。(另:per-epoch 的 KL early-stop `_allreduce_scalar_max`(server.py:1829,也 NumelIn=1)是第二处潜在非对称,但被 MAX 归约保护、各 rank 一致 break;真正杀手是那个 barrier。)

**能跑通的修法(按上手速度)**:
1. **world_size=1 单-rank server** → 零跨-rank collective → 不可能死锁,可靠出 update+checkpoint(单卡 server 成吞吐瓶颈但能跑;先验证 pipeline 真能训练)。
2. **8 台各跑一个独立单-rank run**(超参 sweep) → 榨满集群且无 DDP。
3. **真重构**:把 update 触发与 client 解耦,让全 8 rank lockstep 做 `all_reduce(MIN)` of ready-count 一致决定 update/skip(有风险,别挂机过夜盲改)。
⇒ 单纯"加大 group_size/batch_size"不解决问题、会原样重演死锁。

## RL server 推理路径 ≠ 生产 VA_Server：A/B 坐实的 code-path bug (2026-07-20)
**现象**：GRPO 的 val-before-train SR 仅 ~10-14%(5步),远低于独立 5 步 eval 的 ~44%。
**受控 A/B**(ja25 GPU5 server/GPU6 client,与 RL run 的 GPU0-4 互不干扰):独立 `src.inference.server` VA_Server(robotwin_rl5) vs RL-server eval 路径,**同 10 seed(10000-10011)、同机/环境/任务(demo_clean)/步数(25 video,5 action)/guidance(video5,action1)**。
**结果**:standalone **5/10=50%** vs RL 路径 **1/10=10%**,standalone 严格压制(RL 成功的 seed standalone 都成功,standalone 另胜 4 个,RL 无一处反超)。
**排除**(两边完全一致):去噪步数、guidance、LoRA(step0 零初始化=identity)、eval 确定性(eval_mode 路径确为 deterministic)、seed、task_config;两 config(robotwin.yaml vs robotwin_rl5.yaml)仅 action_num_inference_steps 50→5 之差,而 grpo config 已 override 成 5。
**根因**:唯一差异是代码路径——RL server 在 `_sample_action_chunk`/`_run_video_prefix`(src/rl/server.py)**重写了去噪**,而非调用 `VA_Server.infer`。`scheduler_transition_mean` 只是转调 `FlowMatchScheduler.step`(同数学),故分歧在**循环结构**。头号嫌疑:RL 用 `F.pad(scheduler.timesteps,(0,1),value=0)` 补 0 并对最后一个真实时刻也 step → `scheduler.step` 命中 `timestep_id+1>=len→sigma_=0` → **多去噪一步到 sigma=0**;而 VA_Server 用 `if not last_step:` 提前一格停(动作停在 sigma(t_last))。video prefix 同样补 0。多这一步很可能把动作推离 action head 预期分布。
**严重性**:rollout 也走同一 `_sample_action_chunk` → **RL 在优化一个被错误积分、且和生产不一致的策略**,修复是 RL 结果有意义的前提。
**修法**:把 RL eval/rollout 去噪循环对齐 `VA_Server.infer`(或直接复用),再跑同一 A/B,预期 RL SR 回到 ~50%。A/B 产物在 /mnt/share/rl_exp/ab_test/。

### 更正 (2026-07-20): 上条"pad-0 多走一步"根因判断错误,继续排查中
进一步核查推翻了 pad-0 假设:**VA_Server(src/inference/server.py ~740-748)同样 `F.pad(timesteps,(0,1),value=0)` 补 0**,video/action 步数、`if not last_step`、guidance 与 RL 完全一致 → pad-0 不是差异。也排除 train/eval 模式(VA_Server 无任何 .eval()/.train() 调用;transformer 无 dropout)。已确认:`GRPOTrainingServer(VA_Server)` 继承 `_prepare_latent_input`/`_repeat_input_for_cfg`/`_encode_obs`/`use_cfg`,仅重写 `_run_video_prefix`+`_sample_action_chunk`。**根因尚未静态定位。** 最强未排除嫌疑:**多会话 KV-cache 复用**——RL server 一卡多路复用多个 client 会话(`_switch_to_session`/`_swap_out`/`_session_store`),VA_Server 单会话从不换;A/B 里 standalone 1 会话=50%,RL val 4 client=4 会话共享=10%。定位实验:跑单-client RL eval(num_clients=1,无会话切换)看 SR 是否回到 ~50%(是→多会话问题;否→采样重写问题,再做数值 dump 对比)。测试树 /mnt/share/rl_exp/sr_singleclient/。

### 定位实验结论 (2026-07-20): 根因 = 多会话 KV-cache 并发污染 [已确认]
单-client 测试(RL server + num_clients=1 单会话, 同 10 eval seed): **1-client RL = 3/10=30%** vs 4-client RL val ~0-10% vs standalone 5/10=50%。4 会话→1 会话把 SR 从 ~0-10% 拉回 30%,并恢复了 10000/10001(standalone 也过、4-client 挂)。n=10 时 σ≈16%,故 30% 与 50% 统计不可区分 → **单会话 RL ≈ standalone,退化只在并发会话时出现**。⇒ RL server 的 per-session KV-cache 存取(`_switch_to_session`/`_swap_out`/`_session_store`,VA_Server 无此多会话机制)在多个 client 会话共享一张 server 卡时**互相污染 conditioning**。影响:sr8(4 client/1 server)及任何"多 client/单 server"配置都在退化的 rollout 上训练/eval。修复:(a) 修正 KV-cache 跨会话的正确隔离存取[真解];(b) 每 server rank 只挂 1 client(无并发但无 rollout 并行,需更多 server 卡);(c) 每次 sample 重建 KV(慢)。

### RL run 两种崩溃 + 修复 (2026-07-22): NCCL 超时 & progress 残留导致 rank 失对齐
1-machine GRPO(ja25 world_size=4, group_size=4=world_size, local_group_size=1;每组=同一 item 在 4 rank 各跑一次,凑齐需 4 rank 都交同一 (seed,item))出现两种崩溃,均已修 + 加了兜底。
- **崩溃 A(client 卡死→NCCL 超时)**:原始 clean run 跑~5 步后一个 rollout client 卡死,server rank 在 lockstep `ALLREDUCE` gate 空等 900s 命中 NCCL 默认超时(`DIST_TIMEOUT_MIN=15`)→ SIGABRT → torchrun 连坐杀全体。watchdog `STALE_S=900` 与 NCCL 900s 相等→竞争,NCCL 先触发。**修**:`DIST_TIMEOUT_MIN=15→60`;watchdog `STALE_S=900→420 TICK=90`(`STALE_S=420 TICK=90 bash watchdog_1m.sh`)。
- **崩溃 B(重启失对齐→incomplete-group 死循环)**:client 靠"每 rank 同确定性排列 `assignment_order_for_pass` + 同 `next_item_idx` 起点"无通信对齐;但 client 把进度持久化到 `rollout/rank_<r>/progress/client_0.json` 并按 scope 续跑。任何**不清 progress** 的重启 → 各 rank 从崩溃前各自的 next_item_idx 续跑(实测 rank0=45 / rank1-3=47)→ 跑不同 item → 跨 rank merge 组不完整 → `_gather_sharded_rewards` 旧逻辑 `raise` → 传回全部 client 崩 → supervisor 重启 → 死循环(实测 8368 次错误、~10h 零进展)。**注意 `--resume-from` 不是主因**(checkpoint 无 rollout_store);纯 clean-server 重启也中招。**修(每次重启必做)**:relaunch 前 `rm -f rollout/rank_*/progress/client_0.json` + 清 `rollout/node_logs/rank*.log`(否则 mtime-based watchdog 会秒杀新 client)。全 rank 从 next_item_idx=0 → 对齐 → 实测 **kept_groups=4/4, drops=0**。
- **加固(改在 live tree,未 git commit)**:`_gather_sharded_rewards`(src/rl/server.py ~2482)不完整组改为**丢弃+打日志**(不 raise);调用方 `_run_grpo_update`(~1192)`if gid not in global_rewards: skipped_groups+=1; continue`。因 merged 各 rank 一致→丢弃集合一致→梯度 all-reduce 对齐;全丢时 `if not episodes: return` 各 rank 一致跳过。实测失对齐时丢 4 留 2 不崩。中途 client 重启只损失几个组、不再崩整机。
- **监控正则坑**:别用 `grep "incomplete group"`(会匹配自己的"dropping incomplete group"日志)或 `Traceback .most recent`(`.`匹配到 `(`,命中良性 `ConnectionClosedError`)。用无歧义信号:checkpoint 文件出现 / `pgrep src.rl.server==0` / `grep 'Check client --world_size'`(旧 raise 串,应恒 0) / step 停滞。
- **确认健康 run**:清 progress 后 09:32/09:35 重启,wandb `8u4how42`;step 1-5 全 kept 4/4 drops 0,~25min/步;`grpo_step_000005.pt`+exports @ 11:27;grad_norm~0.002(在学)、`approx_kl≈0` 是 `update_epochs=1` 预期。**ssh 单跳**:`ssh quser@zhehanmo.cn -p 12522` 直接就是 ja25,/mnt/share 可见,scp 用 `-P 12522` 可本地改远程文件。
- **step5 配对 eval(100 seed,512 harness)**:held-out 84 = 0.476→0.500(+0.024,噪声内);train-16 = 0.500→0.750(+0.25,记忆);full-100 = 0.480→0.540(+0.06,主要靠训练 seed)。5 步太早,看后续 held-out 走势。
