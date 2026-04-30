# LingBot-VA Evaluation Pipeline Analysis

## Overview
The lingbot-va project implements a distributed evaluation system for testing vision-action (VA) models on robotic manipulation tasks. The system uses a **client-server architecture** where:
- **Clients**: Run task environments and conduct inference locally
- **Servers**: Host the large transformer/diffusion models and respond to inference requests via WebSocket

---

## 1. Task Definition and Step Limits

### Task Definition Sources
1. **RoboTwin Environment** (external dependency at `/home/cxy/WAM/RoboTwin`)
   - Tasks are defined as environment classes in the RoboTwin simulator
   - Referenced in eval client script via: `class_decorator(task_name)` function
   - Example tasks in launch scripts: "adjust_bottle", "stack_bowls_three", "place_shoe", etc.

2. **Task Configuration Files** (in `/home/taiji/cxy/lingbot-va/wan_va/configs/`)
   - `va_robotwin_cfg.py` - Configuration for RoboTwin environment
   - `va_libero_cfg.py` - Configuration for LIBERO benchmark
   - Defined environment-specific parameters (image size, action dimensions, inference steps)

3. **Task Groups** (hardcoded in launch scripts)
   - Located in: `/evaluation/robotwin/launch_client_split.sh` and `/evaluation/robotwin/launch_client.sh`
   - Organized into groups for parallel evaluation
   - Example:
     ```bash
     task_groups=(
       "stack_bowls_three handover_block hanging_mug scan_object lift_pot put_object_cabinet stack_blocks_three place_shoe"
       "adjust_bottle place_mouse_pad dump_bin_bigbin move_pillbottle_pad pick_dual_bottles shake_bottle place_fan turn_switch"
       # ... 5 more groups
     )
     ```

### Step Limits and Termination Conditions

#### **RoboTwin Evaluation** (`evaluation/robotwin/eval_polict_client_openpi.py`)
- Step limit defined in the **environment object** (`TASK_ENV.step_lim`)
- Not explicitly set in the evaluation script - inherited from RoboTwin environment class
- **Termination conditions** (line 558-612):
  ```python
  while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
      # Execute actions until step limit or success
      if TASK_ENV.eval_success:
          succ = True
          break
  ```
- Test runs until:
  1. Step limit reached (`TASK_ENV.step_lim`)
  2. Task success condition met (`TASK_ENV.eval_success`)
  3. Total success count reaches `test_num` (default: 100)

#### **LIBERO Evaluation** (`evaluation/libero/client.py`)
- Fixed step limit: **800 timesteps** (line 98)
  ```python
  while cur_env.env.timestep < 800:
  ```
- Terminates on:
  1. Episode completion (done flag)
  2. Step limit reached
  3. All episodes tested

### Configuration Parameters Affecting Step Limits

**From `va_robotwin_cfg.py`:**
```python
va_robotwin_cfg.action_per_frame = 16      # Actions per inference frame
va_robotwin_cfg.num_inference_steps = 25   # Video generation steps
va_robotwin_cfg.action_num_inference_steps = 50  # Action diffusion steps
va_robotwin_cfg.frame_chunk_size = 2       # Frames per inference chunk
```

These parameters affect:
- How many action frames are generated per inference call
- How many diffusion steps to use for video/action generation
- The granularity of execution (16 actions per frame = ~2.67 seconds per chunk at 6 Hz)

---

## 2. Task Distribution Across GPUs (Servers and Clients)

### Distributed Evaluation Architecture

#### **Multi-Server Setup** (`launch_server_split.sh`)
```bash
NUM_SERVERS=${NUM_SERVERS:-6}              # Default 6 servers
SERVER_GPU_START=${SERVER_GPU_START:-0}    # Default start at GPU 0
START_PORT=${START_PORT:-29556}             # Default port 29556

# Each server gets:
for i in $(seq 0 $((NUM_SERVERS - 1))); do
    GPU=$((SERVER_GPU_START + i))           # GPU 0, 1, 2, 3, 4, 5
    CURRENT_PORT=$((START_PORT + i))        # Port 29556, 29557, ...
    CURRENT_MASTER_PORT=$((MASTER_PORT + i))  # Master port incremented per server
done
```

**Resource allocation:**
- Each server runs on a **separate GPU**
- One server per inference node
- Server lifecycle: `python -m torch.distributed.run --nproc_per_node 1 wan_va/wan_va_server.py`

#### **Multi-Client Setup** (`launch_client_split.sh`)
```bash
CLIENT_GPUS=(${CLIENT_GPUS:-6 7 6 7 6 7})  # GPU assignment per client
NUM_CLIENTS=${#CLIENT_GPUS[@]}               # Number of clients = GPU array length

# Client i connects to server at port: START_PORT + i
for i in "${!task_names[@]}"; do
    task_name="${task_names[$i]}"           # Task assigned to client i
    gpu_id="${CLIENT_GPUS[$i]}"             # GPU from CLIENT_GPUS[i]
    port=$(( START_PORT + i ))              # Connects to server i
done
```

**Default configuration:**
- 6 clients distributed across GPUs 6 and 7 (3 per GPU)
- Client i connects to server i on port `START_PORT + i`
- Each client runs one task (1:1 mapping)

#### **Task-to-Server Mapping**
```
Client 0 (GPU 6) -> Task "stack_bowls_three"    -> Server 0 (GPU 0, Port 29556)
Client 1 (GPU 7) -> Task "handover_block"       -> Server 1 (GPU 1, Port 29557)
Client 2 (GPU 6) -> Task "hanging_mug"          -> Server 2 (GPU 2, Port 29558)
Client 3 (GPU 7) -> Task "scan_object"          -> Server 3 (GPU 3, Port 29559)
Client 4 (GPU 6) -> Task "lift_pot"             -> Server 4 (GPU 4, Port 29560)
Client 5 (GPU 7) -> Task "put_object_cabinet"   -> Server 5 (GPU 5, Port 29561)
```

### Communication Protocol
- **WebSocket over TCP**: Custom WebSocket connection
- Server address: `ws://0.0.0.0:CURRENT_PORT` (running on `0.0.0.0` for localhost)
- Client initiates connection via `WebsocketClientPolicy(port=port)`
- Serialization: MessagePack + NumPy support (`msgpack_numpy.py`)

### Server Configuration for Distributed Inference
**From `wan_va_server.py` (line 41-100):**
```python
class VA_Server:
    def __init__(self, job_config):
        self.device = torch.device(f"cuda:{job_config.local_rank}")
        self.enable_offload = True  # VAE/text_encoder to CPU to save VRAM
        self.vae_offload = True     # Keep VAE on CPU during inference
        # Model components:
        # - Transformer: ~18 GB
        # - VAE: ~2.7 GB (when offloaded, saved ~5.4GB swap cost)
        # - Text encoder: moved to CPU when offloaded
```

### GPU Memory Management
- **Single GPU per server**: ~24 GB (RTX 4090)
- **Memory allocation**:
  - Transformer: ~18 GB (stays on GPU)
  - VAE: ~2.7 GB (swapped CPU↔GPU per chunk for memory efficiency)
  - Text encoder: Offloaded to CPU during inference
  - Transient activations: >=1.3 GB
- **Optimization**: `pytorch_cuda_alloc_conf=expandable_segments:True` for dynamic allocation

---

## 3. Evaluation Launch and Orchestration Scripts

### Primary Orchestration Scripts

#### **A. Single Machine Multi-Task Evaluation** (`launch_client_split.sh` + `launch_server_split.sh`)

**Purpose**: Run multiple tasks in parallel on a single machine with multiple GPUs

**Server Launch** (`launch_server_split.sh`):
```bash
# Prerequisite: Set environment variables
NUM_SERVERS=6
SERVER_GPU_START=0
START_PORT=29556
MASTER_PORT=29661

# Each server runs: 
python -u -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port $CURRENT_MASTER_PORT \
    wan_va/wan_va_server.py \
    --config-name robotwin \
    --port $CURRENT_PORT
```

**Client Launch** (`launch_client_split.sh`):
```bash
# Prerequisite: Set environment variables
NUM_CLIENTS=6
CLIENT_GPUS=(6 7 6 7 6 7)
START_PORT=29556

# Command to launch all clients:
./evaluation/robotwin/launch_client_split.sh \
    <save_root>      # e.g., "./results" 
    <task_list_id>   # e.g., 0 (selects task_groups[0])
    <seed>           # e.g., 0
    <test_num>       # e.g., 100 (episodes per task)
```

**Task Groups by ID** (from `launch_client_split.sh` lines 26-34):
```
0: stack_bowls_three handover_block hanging_mug scan_object lift_pot put_object_cabinet stack_blocks_three place_shoe
1: adjust_bottle place_mouse_pad dump_bin_bigbin move_pillbottle_pad pick_dual_bottles shake_bottle place_fan turn_switch
2: shake_bottle_horizontally place_container_plate rotate_qrcode place_object_stand put_bottles_dustbin move_stapler_pad place_burger_fries place_bread_basket
3: pick_diverse_bottles open_microwave beat_block_hammer press_stapler click_bell move_playingcard_away open_laptop move_can_pot
4: stack_bowls_two place_a2b_right stamp_seal place_object_basket handover_mic place_bread_skillet stack_blocks_two place_cans_plasticbox
5: click_alarmclock blocks_ranking_size place_phone_stand place_can_basket place_object_scale place_a2b_left grab_roller place_dual_shoes
6: place_empty_cup blocks_ranking_rgb (plus 4 more repeats)
```

#### **B. Docker Single-Task Evaluation** (`docker/launch_eval.sh`)

**Purpose**: Simple Docker-based evaluation for development/testing

**Workflow**:
```bash
# 1. Start server in tmux session
tmux new-session -d -s lb_server "
    python -m torch.distributed.run \
        --nproc_per_node 1 \
        --master_port 29061 \
        wan_va/wan_va_server.py \
        --config-name robotwin \
        --port 29056
"
sleep 30  # Wait for model loading

# 2. Start client
PYTHONWARNINGS=ignore python -m evaluation.robotwin.eval_polict_client_openpi \
    --config policy/ACT/deploy_policy.yml \
    --task_name ${TASK_NAME} \
    --task_config demo_clean \
    --test_num 100 \
    --port 29056
```

#### **C. Standard Evaluation** (`launch_client.sh` + `launch_server.sh`)

**Single-task evaluation for one server-client pair:**

Server:
```bash
export START_PORT=29056
export MASTER_PORT=29061

python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port $MASTER_PORT \
    wan_va/wan_va_server.py \
    --config-name robotwin \
    --port $START_PORT
```

Client:
```bash
python -m evaluation.robotwin.eval_polict_client_openpi \
    --config policy/ACT/deploy_policy.yml \
    --task_name "adjust_bottle" \
    --task_config demo_clean \
    --test_num 100 \
    --port 29056
```

#### **D. Production Server Sync** (`script/run_launch_va_server_sync.sh`)

**Purpose**: Multi-GPU server with synchronization support

**Features**:
- Multi-GPU support (`--nproc_per_node=${num_gpu}`)
- Lighthouse integration for distributed training/inference coordination
- Configurable via environment variables:
  - `NGPU` - Number of GPUs (default: 8)
  - `MASTER_PORT` - PyTorch distributed master port (default: 29501)
  - `PORT` - Server port (default: 1106)
  - `CONFIG_NAME` - Config name (default: robotwin)
  - `TORCHFT_LIGHTHOUSE` - Lighthouse URL for distributed coordination

### LIBERO Evaluation Scripts

**Server** (`evaluation/libero/launch_server.sh`):
```bash
python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port 29061 \
    wan_va/wan_va_server.py \
    --config-name libero \
    --port 29056
```

**Client** (`evaluation/libero/launch_client.sh`):
```bash
python evaluation/libero/client.py \
    --libero-benchmark libero_10 \
    --task-range 0 10 \
    --port 29056 \
    --test-num 50 \
    --out-dir outputs/libero
```

### Evaluation Client Core Logic (`eval_polict_client_openpi.py`)

**Main evaluation loop** (lines 472-656):
```python
def eval_policy(..., test_num=100, ...):
    succ_seed = 0
    now_seed = st_seed
    
    while succ_seed < test_num:
        # 1. Setup environment with demo
        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        
        # 2. Get task instruction
        instruction = TASK_ENV.get_instruction()
        
        # 3. Initialize policy with reset
        model.infer(dict(reset=True, prompt=instruction))
        
        # 4. Execute episode
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            # Get observation
            obs = TASK_ENV.get_obs()
            
            # Infer actions
            ret = model.infer(dict(obs=obs, prompt=instruction, ...))
            action = ret['action']
            
            # Execute action sequences
            for frame in action.shape[1]:
                for step in action.shape[2]:
                    TASK_ENV.take_action(ee_action, action_type='ee')
            
            # Update KV cache for next inference
            model.infer(dict(obs=key_frames, compute_kv_cache=True, ...))
            
            if TASK_ENV.eval_success:
                break
        
        # 5. Log results
        succ_rate = TASK_ENV.suc / TASK_ENV.test_num
        
        succ_seed += 1
```

### Logging and Results Storage

**Directory structure**:
```
{save_root}/
├── stseed-{st_seed}/
│   ├── visualization/
│   │   └── {task_name}/
│   │       └── {test_num}_{prompt}_{success}.mp4
│   └── metrics/
│       └── {task_name}/
│           └── res.json  # {"succ_num": X, "total_num": Y, "succ_rate": Z}
```

**Result JSON format** (line 646-650):
```json
{
  "succ_num": 23.0,
  "total_num": 100.0,
  "succ_rate": 0.23
}
```

---

## 4. Key Configuration Files

### Task Configuration (`wan_va/configs/va_robotwin_cfg.py`)

```python
va_robotwin_cfg.action_dim = 30             # Total action dimensions
va_robotwin_cfg.action_per_frame = 16      # Actions per frame
va_robotwin_cfg.num_inference_steps = 25   # Video diffusion steps
va_robotwin_cfg.action_num_inference_steps = 50  # Action diffusion steps
va_robotwin_cfg.frame_chunk_size = 2       # Frames per chunk
va_robotwin_cfg.height = 256               # Image height
va_robotwin_cfg.width = 320                # Image width
va_robotwin_cfg.attn_window = 72           # Attention window size
va_robotwin_cfg.guidance_scale = 5         # Video guidance scale
va_robotwin_cfg.action_guidance_scale = 1  # Action guidance scale
```

### Model Paths and Weights

**Default model path** (from `va_robotwin_cfg.py` line 9):
```
/mnt/Shared_06_disk1/cxy/WAM/lingbot-va-posttrain-robotwin/
```

**Components loaded**:
- VAE: `{model_path}/vae/`
- Tokenizer: `{model_path}/tokenizer/`
- Text Encoder: `{model_path}/text_encoder/`
- Transformer: `{model_path}/transformer/`

### Environment Setup

**Docker environment** (`docker/.env`):
- `LINGBOT_VA_PATH` - Path to lingbot-va repository
- `ROBOTWIN_PATH` - Path to RoboTwin repository
- `MODEL_PATH` - Path to model weights directory

**Required Python packages** (from `docker/requirements-robotwin.txt`):
```
robotwin dependencies
LIBERO dependencies
PyTorch + CUDA
diffusers
transformers
websockets
```

---

## 5. WebSocket Communication Protocol

### Server-Client Handshake

1. **Client connects**: `ws://0.0.0.0:{port}`
2. **Server sends metadata**: `unpackb(first_message)`
3. **Client ready**: Connection established

### Inference Request/Response Format

**Request** (`WebsocketClientPolicy.infer()` - line 62-69):
```python
data = {
    'reset': bool,                          # Reset server state
    'prompt': str,                          # Language instruction
    'obs': dict | list,                     # Observations
    'compute_kv_cache': bool,               # Update KV cache
    'video_guidance_scale': float,
    'action_guidance_scale': float,
    'save_visualization': bool,
}
# Packed with msgpack + numpy support
```

**Response**:
```python
{
    'action': np.ndarray,                   # Shape: (B, T, 16) for action_per_frame=16
    'video': np.ndarray (optional),         # Imagined video frames
}
```

### Message Serialization

**Packer** (`msgpack_numpy.py`):
- Uses MessagePack for efficient serialization
- Custom NumPy array handling for binary data
- Handles large arrays efficiently over WebSocket

---

## 6. Summary: Execution Flow

### Typical Multi-Task Evaluation Session

```
1. PREPARATION
   - Set environment: NUM_SERVERS=6, NUM_CLIENTS=6, START_PORT=29556
   - Allocate GPUs: Servers 0-5 on GPU 0-5, Clients distributed on GPU 6-7

2. LAUNCH SERVERS (parallel)
   for i in 0..5:
       python -m torch.distributed.run ... wan_va_server.py --port $((29556+i))
           ↓ (GPU i, ~24GB memory)
           - Load transformer (~18GB) to GPU
           - Keep VAE/text_encoder on CPU for offloading

3. WAIT FOR SERVER READY
   sleep 30s (model loading time)

4. LAUNCH CLIENTS (parallel)
   for i in 0..5:
       CUDA_VISIBLE_DEVICES=$((6+i%2)) python eval_polict_client_openpi \
           --task_name ${task_names[i]}
           --test_num 100
           --port $((29556+i))
       ↓ (GPU 6 or 7, environment simulation)
       
5. EXECUTION PER CLIENT (sequential episodes)
   for episode in 1..test_num:
       a. Setup environment with random seed
       b. Get language instruction
       c. Reset server (send prompt)
       d. Execute episode:
          - for step in 0..step_lim:
              - Get observation
              - Call server.infer(obs) via WebSocket
              - Execute actions
       e. Log success/failure
       f. Save video + metrics

6. RESULTS
   - Each task accumulates success metrics
   - Results saved to: {save_root}/stseed-{seed}/metrics/{task_name}/res.json
   - Videos saved to: {save_root}/stseed-{seed}/visualization/{task_name}/
```

---

## 7. Important Constants and Defaults

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `START_PORT` | 29556 | Starting port for servers |
| `NUM_SERVERS` | 6 | Number of parallel servers |
| `NUM_CLIENTS` | 6 | Number of parallel clients |
| `SERVER_GPU_START` | 0 | First GPU for servers |
| `test_num` | 100 | Episodes per task |
| `num_inference_steps` | 25 | Video generation diffusion steps |
| `action_num_inference_steps` | 50 | Action diffusion steps |
| `action_per_frame` | 16 | Actions per inference frame |
| `frame_chunk_size` | 2 | Frames per inference chunk |
| RoboTwin step_lim | External | Task-dependent (from RoboTwin) |
| LIBERO step_lim | 800 | Fixed for all LIBERO tasks |

---

## 8. External Dependencies

- **RoboTwin**: `/home/cxy/WAM/RoboTwin` - Robotic manipulation simulator
  - Defines tasks, step limits, and simulation environment
  - Must be installed in PYTHONPATH for evaluation
  
- **LIBERO**: Robotic learning benchmark
  - Alternative to RoboTwin for evaluation
  - Pre-defined tasks with 800-step episodes

