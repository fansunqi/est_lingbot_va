# RoboTwin GRPO RL —— 运行与评估手册 (turn_switch)

> 面向**亲自重跑**这套实验的人。记录的是实测能跑通的流程，含本机特有的坑。
> 实验：`turn_switch` 任务的 GRPO RL 微调（LoRA-only），8-rank 复制式 DDP。
> 最近一次维护：2026-06-24。

---

## 0. 全局约定（务必先读）

| 项 | 值 |
|---|---|
| 代码仓库 | `/apdcephfs_cq8/share_1611098/stevefan/robotics/est/lingbot-va`（branch `grpo_cxy`） |
| **server 端** python | `/root/.venv/bin/python`（py3.13 / torch≥2.6，含 torchrun） |
| **client 端** python | conda `RoboTwin` 环境：`/apdcephfs_tj5/share_303547874/stevefan/miniconda3/envs/RoboTwin/bin/python`（py3.10，含 SAPIEN/RoboTwin） |
| RoboTwin 根目录 | `/apdcephfs_cq8/share_1611098/stevefan/robotics/RoboTwin`（client 启动后会 `chdir` 到这里 → **所有路径用绝对路径**） |
| 大文件输出盘 | `/apdcephfs_tj5/share_303547874/stevefan/est_rl_exp/<实验名>/`（空间充足，checkpoint/结果存这里） |
| 配置文件 | `configs/rl/robotwin_grpo_turn_switch_fast.yaml` |
| GPU | 本机 8×80GB；**共享物理机**，可能有别的容器抢 CPU/带宽（见 §6 排错） |

**架构**：server（持模型、做 GRPO update）和 client（跑 RoboTwin 仿真、采 rollout）是**两个独立进程**，用 websocket+msgpack 通信。8-rank DDP 下每个 rank 绑一个端口 `START_PORT+rank`（29546–29553），梯度在 update 时 all-reduce。

---

## 1. 关键坑（本机特有，不看会浪费几小时）

1. **DDP server 脚本默认用 `uv run --active torchrun` → 在本机会 hang。**
   `scripts/run_robotwin_grpo_server_ddp.sh` 第 94 行用 uv，但本机无 `.venv`、`VIRTUAL_ENV` 为空，uv 会尝试 sync 然后超时卡死。
   **解决**：直接手敲 `/root/.venv/bin/torchrun`（见 §3），绕过 uv。

2. **websocket 走了公司代理 → 503。**
   `websockets`≥16 的 `connect()` 默认 `proxy=True`，会读 `http_proxy/HTTP_PROXY/ALL_PROXY/all_proxy` 全部大小写变体，把 localhost 连接也发去公司代理。
   **解决**：client 端必须 `unset` 所有代理变量 + 设 `no_proxy`。光 `unset http_proxy https_proxy` 不够，要连 `ALL_PROXY/all_proxy` 一起 unset（见 §3 client 脚本）。

3. **client 启动后 `os.chdir(ROBOTWIN_ROOT)` → 相对路径全失效。** 所有 `--assignment` / `--save_root` 等一律用**绝对路径**。

4. **resume 的 weights_only 坑（已修，勿回退）。** torch≥2.6 `torch.load` 默认 `weights_only=True`，会拒绝我们 checkpoint 里的 EasyDict config blob。`src/rl/server.py` 的 `load_checkpoint` 已改成 `weights_only=False`。

5. **server 端用 `/root/.venv`，client 端用 conda RoboTwin —— 两个环境别混。** client 脚本里有个检查：非 `RoboTwin` conda 环境会拒绝启动（除非设 `ALLOW_NON_ROBOTWIN_CONDA=1`）。

6. **进程管理铁律**：只停自己起的进程，**按 PID 停**（先 client launcher 后 server torchrun，靠 launcher 的 trap 级联清理子进程）。**绝不 `pkill -f src.rl.server`**——这是共享机，会误杀别人的任务。

7. **绝不中途 kill client（会触发 DDP barrier 死锁，最坑）。** step0 的 initial eval 和每个 update step 的 eval，都会让 8 个 server rank 在 `src/rl/server.py::_gather_replicated_eval_results` 的 **无超时 `dist.barrier()`** 处汇合——必须 8 个 rank 的 client 全部把本地 eval 跑完才能过 barrier。若此时 kill 掉部分 client（哪怕只为修别的问题再重启），已到 barrier 的 rank 会 **100% CPU 空转永久等待**，且主线程被 barrier 占死、**不再响应新的 websocket 握手**（表现为后续重连固定那几个 rank handshake 超时）。
   **判据**：`ps`/`top` 看到部分 server rank `state=R` + `cpu=100%`、另一部分 `state=S`；用 py-spy 抓栈会看到 `barrier → _gather_replicated_eval_results → infer`。
   **唯一解**：整体重启 server（死锁 rank 对 SIGTERM 可能不响应，按 PID `kill -TERM` 后等几秒，必要时 `kill -KILL` **自己的** rank）。client 起来后**全程不要中途 kill**；要停就按 §3.4 顺序整套停。

8. **warp/curobo JIT 内核缓存冷启动竞争（首次跑 / 缓存被清后）。** 8 个 client 同时 import curobo 会并发编译 warp 内核到共享缓存（`~/.cache/warp/<ver>/`），部分 rank 会加载到残缺模块，报 `Warp CUDA error 500: named symbol not found` / `Failed to load CUDA module 'curobo.geom.transform'` / `Failed to find forward kernel 'linear_interpolate_trajectory_kernel'`。
   **判据**：只有部分 rank 报错（缓存竞争是随机的），且 `~/.cache/warp/<ver>/bin/` 里 `.ptx`/`.hash` 文件刚生成。
   **解决**：缓存一旦建好（含 `wp_curobo.util.warp_interpolation.*.ptx` 等），重启一次 client 即可——这次所有 rank 只**读**已编译好的内核，不再竞争。本机 warp 版本 1.0.2、sm70。

---

## 2. 配置要点（`robotwin_grpo_turn_switch_fast.yaml`）

| 参数 | 值 | 含义 |
|---|---|---|
| `group_size` | 8 | 每个 group 采 8 条带噪轨迹，组内比较算 advantage |
| `rollout_groups_per_update` | 16 | 攒满 16 个 group 才触发一次 update step |
| `update_epochs` | 1 | on-policy，每批数据训 1 epoch |
| `batch_size` | 16 | |
| `sampler` | flow_cps | Flow-CPS 采样器 |
| `lr` | 5e-5 | |
| `checkpoint_interval` | 4 | 每 4 个 update step 落一个 checkpoint |
| `eval_every` | **4** | 每 4 个 update step 做一次 validation（已与 checkpoint 对齐；原为 2） |
| `eval_action_num_inference_steps` | 5 | validation 用的确定性去噪步数（别 fallback 到 base 的 50） |

**rollout 与 eval 解耦（本实验的关键改动）**：
- 训练 rollout 用 **16-seed** assignment（`experiments/robotwin_grpo_turn_switch_fast/assignment.json`，seed 10000–10017 中的 16 个）——保持训练动态不变。
- validation 用独立的 **64-seed** assignment（`--eval_assignment`），噪声更小（σ≈0.06 vs 16-seed 的 0.12），趋势更可信。
- 代码侧：`grpo_rollout_client.py` 支持 `--eval_assignment`（缺省时回退到 rollout assignment，向后兼容）；rollout 主循环完全不受影响。
- ⚠️ 注意：64-seed 里有 15 个和训练 16-seed 重叠（约 77% 是 held-out），是**偏泛化**的混合指标，且与离线 64-seed 评估口径一致，便于 sanity check。

---

## 3. 跑训练（8-rank DDP，从头或 resume）

> 全程在 **tmux** 里跑，便于断连后监控。建议两个 window：window 0 = server，window 1 = client。

### 3.1 准备 assignment（只需一次）

**rollout 的 16-seed assignment** 已存在（`experiments/robotwin_grpo_turn_switch_fast/assignment.json`）。若要重新生成：
```bash
/apdcephfs_tj5/share_303547874/stevefan/miniconda3/envs/RoboTwin/bin/python \
  evaluation/robotwin/make_grpo_assignment.py \
  --task turn_switch --task_config demo_clean \
  --num_groups <N> --start_seed 10000 \
  --output <绝对路径>/assignment.json
# make_grpo_assignment 会预筛掉无效 seed，并带上 episode_info（eval 必需）
```

**validation 的 64-seed assignment**：从已有 100-seed assignment 按固定 64 seed 过滤（零成本）。脚本见 `/tmp/gen_eval64.py`，核心逻辑：
```python
seeds64 = set(json.load(open("experiments/grpo_eval_100seed/fixed_64_seeds.json")))
a100    = json.load(open("experiments/grpo_eval_100seed/assignment_100.json"))  # 带 episode_info
sel = sorted([x for x in a100 if x["seed"] in seeds64], key=lambda x: x["seed"])
json.dump(sel, open("<新盘>/eval_assignment_64.json", "w"), indent=2)
# 校验：len(sel)==64 且 all('episode_info' in x)
```

### 3.2 启动 server（tmux window 0）

8-rank DDP，**手敲 torchrun 绕过 uv**。从头训练就去掉 `--resume-from`；resume 就指向已有 checkpoint。

```bash
REPO=/apdcephfs_cq8/share_1611098/stevefan/robotics/est/lingbot-va
NEW=/apdcephfs_tj5/share_303547874/stevefan/est_rl_exp/turn_switch_ddp
cd $REPO

# wandb（online 模式需走代理）
export WANDB_API_KEY="<your-key>"
export WANDB_BASE_URL="https://api.wandb.ai"
export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"
export no_proxy="localhost,127.0.0.1,0.0.0.0"
unset HF_ENDPOINT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mkdir -p $NEW/server

/root/.venv/bin/torchrun \
  --standalone --nproc_per_node=8 --master_port=29646 \
  -m src.rl.server \
  --config configs/rl/robotwin_grpo_turn_switch_fast.yaml \
  --port 29546 \
  --save-root $NEW/server \
  --resume-from $REPO/experiments/robotwin_grpo_turn_switch_ddp/server/checkpoints/grpo_step_000012.pt
  # ↑ 从头训练时删掉这一行
```

> server args：`--config`（必填）、`--port`（rank0 端口，rank r 实际绑 port+r）、`--save-root`（checkpoint 输出根）、`--resume-from`（恢复 LoRA 权重 + optimizer 状态 + global_update_step）。
> **resume 是无缝的**：恢复权重、optimizer 动量、step 计数三者，等价于训练从未中断（GRPO 采样的随机性不可逐位复刻，但属设计内随机）。

等到 8 个 rank 都打印 `server listening on 0.0.0.0:2954x`（约 1–3 分钟，含 base model 加载）再起 client。

### 3.3 启动 client（tmux window 1）

**代理必须全部 unset**，否则 websocket 503。client 走 conda RoboTwin。

```bash
REPO=/apdcephfs_cq8/share_1611098/stevefan/robotics/est/lingbot-va
NEW=/apdcephfs_tj5/share_303547874/stevefan/est_rl_exp/turn_switch_ddp
cd $REPO

export PATH="/apdcephfs_tj5/share_303547874/stevefan/miniconda3/envs/RoboTwin/bin:$PATH"
export ROBOTWIN_ROOT="/apdcephfs_cq8/share_1611098/stevefan/robotics/RoboTwin"
export CONDA_DEFAULT_ENV=RoboTwin
export PYTHONPATH="$REPO"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy   # ← 关键
export no_proxy="localhost,127.0.0.1,0.0.0.0"

export ASSIGNMENT="$REPO/experiments/robotwin_grpo_turn_switch_fast/assignment.json"   # 16-seed rollout
export EVAL_ASSIGNMENT="$NEW/eval_assignment_64.json"                                  # 64-seed validation
export NUM_SERVERS=8
export START_PORT=29546
export SERVER_GPUS="0 1 2 3 4 5 6 7"
export NUM_CLIENTS_PER_SERVER=1
export GROUP_SIZE=8
export SAVE_ROOT="$NEW"
export BASE_RUN_DIR="$NEW/run_$(date +%Y%m%d_%H%M%S)"
export GROUP_BARRIER=1
export GROUP_BARRIER_TIMEOUT=0
export NUM_PASSES=1000
export SKIP_RENDER_CHECK=1
export PYTHON="/apdcephfs_tj5/share_303547874/stevefan/miniconda3/envs/RoboTwin/bin/python"

bash evaluation/robotwin/launch_grpo_rollout_clients_ddp.sh
```

> DDP launcher 会 fan-out 8 个 client group（每 rank 一个），各自连到对应端口、有独立 RUN_DIR。Ctrl-C 会级联停掉所有 client + RoboTwin 子进程。
> `EVAL_ASSIGNMENT` 透传到每个 client 的 `--eval_assignment`；启动后 client log 会打印
> `[eval] using separate eval assignment: 64 items ... (rollout uses 16 items)`，确认解耦生效。

### 3.4 干净停止（要重启/改配置时）

```bash
# 1. 找到自己的两个根 PID
ps -eo pid,cmd | grep -E 'torchrun.*nproc_per_node=8|launch_grpo_rollout_clients_ddp' | grep -v grep
# 2. 先停 client launcher，等其 trap 级联（约 30s）
kill -TERM <client_launcher_pid>; sleep 10
# 3. 再停 server torchrun
kill -TERM <torchrun_pid>; sleep 10
# 4. 确认子进程清空 + 8 卡归零（绝不 pkill）
ps -eo pid,cmd | grep -E 'torchrun|src\.rl\.server|grpo_rollout_client' | grep -v grep || echo clean
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

---

## 4. 监控（验证 resume / 进度 / 健康）

```bash
# resume 是否从正确 step 接上 —— 看 client log 首条 trigger 的 global_update_step
grep -m1 "'global_update_step'" $NEW/run_*/server_0/logs/grpo_client_0.log

# 已完成的 update 数 / 当前攒到第几个 group
grep -c "'updated': True" <client tee log>          # 完成的 update step 数
grep "GRPO update trigger after item" ... | tail -1 # pending_ready_groups: X/16

# 训练健康指标（看 trigger 行的 status）
#   success_rate（训练 rollout 成功率）、ratio≈1.0、approx_kl≈0、grad_norm 正常、clipfrac
# checkpoint 落盘
ls -la $NEW/server/checkpoints/    # grpo_step_0000NN.pt，每 4 step 一个
```

**时间线**（`eval_every=4`、`checkpoint_interval=4` 已对齐）：每 4 个 update step 同时做一次 64-seed validation + 落一个 checkpoint。例如从 step12 resume → step16 出首个 validation + 首个新 checkpoint。

**速度预期**：单 update step 约 1.6–3.5h（攒满 16 group）。turn_switch 失败 episode 会跑满 400 步上限（单条 5–11 分钟），是主要长尾。共享机 CPU 争用时会更慢（见 §6）。

---

## 5. 离线 eval（评估某个 checkpoint，不进训练）

复用同一套 server+client，但 client 加 `--eval_only`（env `EVAL_ONLY=1`），并把 `--resume-from` 指向要评的 checkpoint。

- **单 checkpoint**：`NUM_CLIENTS_PER_SERVER=1`，避免多 client 分片时 group 成员不足导致 barrier hang。
- **8 卡并行评多个/大 assignment**：把 assignment 切成几份（如 100-seed → 50+50），每份起一套独立 server+client，占不同 GPU。
- eval 是 **deterministic**（关 SDE 噪声，每 seed 跑 1 条），同一 checkpoint 重复评结果一致。
- **样本量与噪声**：16-seed σ≈0.12（噪声大，曾出现假高点）；64-seed σ≈0.06；100-seed σ≈0.05。

**已知离线 64-seed 基线**（turn_switch，固定 64 seed）：base 0.37 → step4 0.39 → step8 0.46 → step12 0.47/0.516。RL 在学（100-seed 也单调上升 → 是技能泛化，非死记 16 个布局）。

---

## 6. 排错速查

| 症状 | 原因 | 处理 |
|---|---|---|
| torchrun 启动卡住不动 | `uv run` 在本机 hang | 用 `/root/.venv/bin/torchrun` 手敲 |
| client 连接 503 | 代理拦截 localhost websocket | unset 全部 `*_proxy/ALL_PROXY` + 设 `no_proxy` |
| **部分 rank 固定 handshake 超时（"timed out while waiting for handshake response"）** | **多半是 DDP barrier 死锁**：之前中途 k'd 过 client，残留 rank 卡在 `dist.barrier()` 占死主线程不再响应握手（见 §1.7）。**先排除代理**：无代理下 TCP 能连但 ws 握手超时、且固定那几个 rank → 不是代理 | 用 py-spy/`top` 确认是否 `state=R cpu=100%` 卡 barrier；是则**整体重启 server**。纯重启 client 救不了 |
| 部分 rank 报 warp `named symbol not found` / `Failed to load CUDA module` | warp/curobo JIT 内核缓存冷启动竞争（见 §1.8） | 缓存已建好后**重启一次 client**即可 |
| `FileNotFoundError` 找 assignment | client chdir 到 RoboTwin 根 | 用绝对路径 |
| resume 报 UnpicklingError | torch≥2.6 weights_only | 已修（`load_checkpoint` weights_only=False），勿回退 |
| 某 rank OOM | eval server 叠在 RL rank 上 | 错开 GPU，或先停 RL 再 eval |
| barrier hang | 多 client 但 group 成员不足 | 单 checkpoint 用 `NUM_CLIENTS_PER_SERVER=1` |
| **训练变慢但 GPU util 正常** | **共享机邻居抢 CPU/内存带宽**（RoboTwin 仿真是 CPU 密集；判据：`infer_ms` 正常、`elapsed_s` 翻倍且随时间波动） | 非自身问题，等邻居负载下降；不要动别人进程 |
| wandb 只有 System 栏 | resume 后还没完成第一个 update step | 正常，train 指标在每个 update step 才 log；等首个 `updated:True` |

**调试卡死 rank 的利器（py-spy）**：本机 server venv 无 pip，用 uv 装独立工具即可：
```bash
uv tool install py-spy          # 装到 ~/.local/bin/py-spy
~/.local/bin/py-spy dump --pid <server_rank_pid>   # 抓 Python 栈，定位卡在哪一行
```
看到栈顶是 `barrier → _gather_replicated_eval_results → infer` 即 §1.7 的 eval barrier 死锁。

**快速验证 8 个 server rank 是否都能正常握手**（起 client 前自检，无代理下应全部秒回）：
```bash
PY=/apdcephfs_tj5/share_303547874/stevefan/miniconda3/envs/RoboTwin/bin/python
for port in 29546 29547 29548 29549 29550 29551 29552 29553; do
  printf "port %s: " "$port"
  env -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy no_proxy=127.0.0.1 timeout 15 $PY - <<PYEOF
import time,websockets.sync.client
t=time.time()
try:
    c=websockets.sync.client.connect("ws://127.0.0.1:$port",compression=None,max_size=None,ping_interval=None,close_timeout=10)
    c.recv(); print(f"OK {time.time()-t:.2f}s"); c.close()
except Exception as e: print(f"FAIL {time.time()-t:.2f}s {type(e).__name__}")
PYEOF
done
```
某端口固定 FAIL（10s 超时）而 TCP 可连 → 那个 rank 卡 barrier 死锁，需整体重启 server。

---

## 7. 相关文件清单

| 文件 | 作用 |
|---|---|
| `configs/rl/robotwin_grpo_turn_switch_fast.yaml` | 训练/eval 配置 |
| `src/rl/server.py` | server（模型 + GRPO update + checkpoint） |
| `evaluation/robotwin/grpo_rollout_client.py` | rollout/eval client（支持 `--eval_assignment`） |
| `evaluation/robotwin/launch_grpo_rollout_clients_ddp.sh` | DDP client fan-out launcher |
| `evaluation/robotwin/launch_grpo_rollout_clients.sh` | 单组 client launcher（被上面调用） |
| `evaluation/robotwin/make_grpo_assignment.py` | 生成 rollout assignment（预筛 seed + episode_info） |
| `scripts/run_robotwin_grpo_server_ddp.sh` | 官方 DDP server 脚本（**注意 uv 坑**，本机改用 torchrun） |
| `experiments/robotwin_grpo_turn_switch_fast/assignment.json` | 16-seed rollout assignment |
| `experiments/grpo_eval_100seed/{assignment_100,fixed_64_seeds}.json` | 100-seed eval + 固定 64 seed 来源 |
