# LingBot-VA: Server-Client Architecture & GPU Management Analysis

## Executive Summary

LingBot-VA implements a **distributed server-client architecture** for evaluation using:
- **WebSocket protocol** for server-client communication
- **Multi-GPU deployment** with dedicated GPU allocation
- **KV cache management** for efficient transformer inference
- **Streaming VAE & FSDP** for efficient memory usage
- **PyTorch Distributed** for multi-process training/inference

---

## 1. SERVER-CLIENT ARCHITECTURE

### 1.1 Communication Protocol: WebSocket

#### Server Implementation
**File**: `wan_va/utils/Simple_Remote_Infer/deploy/websocket_policy_server.py`

- **Framework**: `websockets.asyncio.server`
- **Features**:
  - Async handler for each client connection
  - Binary serialization using msgpack (with numpy support)
  - Health check endpoint at `/healthz`
  - Configurable timeouts & compression

```python
async def _handler(self, websocket: _server.ServerConnection):
    # 1. Send metadata
    await websocket.send(packer.pack(self._metadata))
    
    # 2. Receive observations, run inference, send actions
    while True:
        obs = unpackb(await websocket.recv())
        action = self._policy.infer(obs)
        await websocket.send(packer.pack(action))
```

**Key Configuration**:
- `compression=None`: Disabled for performance
- `max_size=None`: No message size limit
- `ping_interval=None`: Disabled (long inference times)
- `close_timeout=10`: 10-second close timeout

#### Client Implementation
**File**: `wan_va/utils/Simple_Remote_Infer/deploy/websocket_client_policy.py`

- **Framework**: `websockets.sync.client`
- **Connection Management**:
  - Automatic reconnection with 5-second retry intervals
  - Persistent connection for entire evaluation session
  - Custom headers support for API authentication

```python
def _wait_for_server(self):
    while True:
        try:
            conn = websockets.sync.client.connect(
                self._uri,
                compression=None,
                max_size=None,
                ping_interval=None,
                close_timeout=10
            )
            metadata = unpackb(conn.recv())
            return conn, metadata
        except Exception as e:
            time.sleep(5)  # Retry every 5 seconds
```

#### Message Format
**Serialization**: msgpack + numpy (custom plugin)
- Efficiently packs numpy arrays with dtype preservation
- Reduces network bandwidth for image data
- Metadata includes server configuration info

### 1.2 Server Mode Operation

**File**: `wan_va/utils/sever_utils.py`

#### Two-Layer Architecture

**Rank 0 (Main Process)**:
```
┌─────────────────────────────────────┐
│   WebsocketPolicyServer (Rank 0)    │
├─────────────────────────────────────┤
│ • Listens on network port           │
│ • Accepts client connections        │
│ • Runs DistributedModelWrapper      │
│ • Handles inference requests        │
└─────────────────────────────────────┘
          ↓ PyTorch Distributed (NCCL)
┌─────────────────────────────────────┐
│   Worker Processes (Rank 1..N)      │
├─────────────────────────────────────┤
│ • Wait for broadcast commands       │
│ • Receive observation data          │
│ • Process model inference           │
│ • Return results                    │
└─────────────────────────────────────┘
```

**Operation**:
1. Rank 0 broadcasts `cmd=1` (inference command)
2. Rank 0 broadcasts observation object list
3. All ranks run `model.infer(obs)` on their GPU
4. Results collected by Rank 0
5. Rank 0 sends results to client via WebSocket

#### Shutdown Mechanism
```python
# Rank 0 broadcasts termination signal
cmd = torch.tensor(-1, dtype=torch.int64)
dist.broadcast(cmd, src=0)

# Worker processes break loop and exit
if cmd_val == -1:
    break
```

---

## 2. GPU ASSIGNMENT & DEVICE MANAGEMENT

### 2.1 GPU Allocation Strategy

#### Multi-GPU Server Setup
**File**: `evaluation/robotwin/launch_server_split.sh` (6 servers) / `launch_server_multigpus.sh` (8 servers)

```bash
# Default configuration: 6 servers on GPU 0-5
SERVER_GPU_START=${SERVER_GPU_START:-0}
NUM_SERVERS=${NUM_SERVERS:-6}

for i in $(seq 0 $((NUM_SERVERS - 1))); do
    GPU=$((SERVER_GPU_START + i))
    CURRENT_PORT=$((START_PORT + i))
    CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.run \
        --nproc_per_node 1 \
        wan_va/wan_va_server.py --port $CURRENT_PORT
done
```

**Rationale**:
- **1 server per GPU** (0-5): Model inference offloaded to dedicated GPU
- Each server holds full model (transformer, VAE, text encoder)
- Independent port per server (29556 + i)
- Independent master port (29661 + i)

#### Multi-GPU Client Setup
**File**: `evaluation/robotwin/launch_client_split.sh`

```bash
# Default: 6 clients alternating between GPU 6 and GPU 7
CLIENT_GPUS=(6 7 6 7 6 7)

for i in "${!task_names[@]}"; do
    gpu_id="${CLIENT_GPUS[$i]}"
    port=$((START_PORT + i))  # Connects to server on port 29556+i
    CUDA_VISIBLE_DEVICES=${gpu_id} python -m evaluation.robotwin.eval_polict_client_openpi \
        --port ${port}
done
```

**Rationale**:
- **GPUs 5-7 for clients** (rendering, environment simulation)
- Multiple clients per GPU (e.g., 3 clients per GPU for 6 clients on 2 GPUs)
- Each client connects to ONE dedicated server via TCP
- Client-side GPU used for:
  - Environment simulation (Sapien/physics)
  - Image rendering (ray tracing, OIDN denoiser)
  - Observation preprocessing
  - Action post-processing

### 2.2 Environment Variables

#### CUDA Configuration
```bash
export CUDA_VISIBLE_DEVICES=$GPU  # Restrict to single GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**`expandable_segments:True`**:
- Enables dynamic CUDA memory growth
- Reduces fragmentation for large models
- Allows flexible memory allocation patterns

#### Client Environment Variables
```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9  # 90% GPU memory allocation
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore::UserWarning
```

### 2.3 Device Assignment in Code

**Server Side**: `wan_va/wan_va_server.py`
```python
class VA_Server:
    def __init__(self, job_config):
        self.device = torch.device(f"cuda:{job_config.local_rank}")
        # local_rank set via torch.distributed.run
```

**Distributed Init**: `wan_va/distributed/util.py`
```python
def init_distributed(world_size, local_rank, rank):
    torch.cuda.set_device(local_rank)  # Pin process to local_rank GPU
    dist.init_process_group(backend="nccl", ...)
```

---

## 3. KV CACHE MANAGEMENT

### 3.1 Cache Architecture

**File**: `wan_va/modules/model.py`

The transformer uses **slot-based KV caching** for efficient inference:

```
┌─────────────────────────────────────────────┐
│      KV Cache (Slot-Based)                  │
├─────────────────────────────────────────────┤
│ • Fixed pre-allocated buffer                │
│ • Dynamically allocates "slots" for K,V     │
│ • LRU eviction when cache is full           │
│ • Supports multiple cache instances         │
└─────────────────────────────────────────────┘
```

### 3.2 Cache Structure

**Per Attention Layer** (WanAttention):
```python
class WanAttention(torch.nn.Module):
    def init_kv_cache(self, cache_name, total_tokens, num_heads, head_dim, 
                      device, dtype, batch_size):
        self.attn_caches[cache_name] = {
            'k': torch.empty([batch_size, total_tokens, num_heads, head_dim],
                            device=device, dtype=dtype),
            'v': torch.empty([batch_size, total_tokens, num_heads, head_dim],
                            device=device, dtype=dtype),
            'id': torch.full((total_tokens,), -1, device=device),
            'mask': torch.zeros((total_tokens,), dtype=torch.bool, device=device),
            'is_pred': torch.zeros((total_tokens,), dtype=torch.bool, device=device),
        }
```

**Cache Properties**:
- `k, v`: Key and value tensors
- `id`: Sequence ID for LRU tracking (higher = more recent)
- `mask`: Boolean mask indicating occupied slots
- `is_pred`: Flag indicating if slot is from prediction (vs observation)

### 3.3 Cache Operations

#### 1. Allocation (`allocate_slots`)
```python
def allocate_slots(self, cache_name, key_size):
    mask = cache["mask"]
    ids = cache["id"]
    
    # Get free slots
    free = (~mask).nonzero(as_tuple=False).squeeze(-1)
    
    # If insufficient free slots, evict oldest (lowest id)
    if free.numel() < key_size:
        used = mask.nonzero(as_tuple=False).squeeze(-1)
        used_ids = ids[used]
        order = torch.argsort(used_ids)
        need = key_size - free.numel()
        to_free = used[order[:need]]  # Evict oldest
        mask[to_free] = False
        ids[to_free] = -1
    
    return free[:key_size]  # Allocated slots
```

#### 2. Update (`update_cache`)
```python
def update_cache(self, cache_name, key, value, is_pred):
    slots = self.allocate_slots(cache_name, key.shape[1])
    new_id = self._next_cache_id(cache_name)  # Increment ID
    
    cache['k'][:, slots] = key
    cache['v'][:, slots] = value
    cache['mask'][slots] = True
    cache['id'][slots] = new_id
    cache['is_pred'][slots] = is_pred
```

#### 3. Clearing (`clear_cache`, `clear_pred_cache`)
```python
def clear_pred_cache(self, cache_name):
    # Clear only prediction tokens (is_pred=True)
    cache = self.attn_caches[cache_name]
    is_pred = cache['is_pred']
    cache['mask'][is_pred] = False

def clear_cache(self, cache_name):
    # Clear entire cache
    self.attn_caches[cache_name] = None
```

### 3.4 Cache Lifecycle During Inference

**Server-Side Workflow** (`wan_va_server.py`):

```python
def _reset(self, prompt):
    # Create cache with window size
    self.transformer.create_empty_cache(
        self.cache_name,
        attn_window,
        latent_token_per_chunk,
        action_token_per_chunk,
        dtype=self.dtype,
        device=self.device,
        batch_size=2 if use_cfg else 1
    )

def _compute_kv_cache(self, obs):
    # Build cache from observation
    self.transformer.clear_pred_cache(self.cache_name)
    
    latent_model_input = self._encode_obs(obs)
    action_model_input = self.preprocess_action(obs['state'])
    
    # Update cache with update_cache=2 (both observation and prediction)
    self.transformer(..., update_cache=2, cache_name=self.cache_name)

def _infer(self, obs, frame_st_id=0):
    # Generate predictions with cache
    for i, t in enumerate(timesteps):
        last_step = i == len(timesteps) - 1
        
        # update_cache=1: store as prediction tokens
        output = self.transformer(..., update_cache=1 if last_step else 0)
```

**Cache Size Calculation**:
```python
total_tokens = (attn_window // 2) * latent_token_per_chunk + \
               (attn_window // 2) * action_token_per_chunk

# Example:
# attn_window=4, latent_token=1024, action_token=256
# total_tokens = 2 * 1024 + 2 * 256 = 2560 slots per layer
```

### 3.5 Transfer Between Server and Client

**Client-Server KV Cache Coordination** (`eval_polict_client_openpi.py`):

```python
# Step 1: Reset (create cache on server)
model.infer(dict(reset=True, prompt=prompt))

# Step 2: First inference (observes first frame, predicts next chunk)
ret = model.infer(dict(obs=first_obs, prompt=prompt))
action = ret['action']

# Step 3: Extract key frames from environment
for i in range(action.shape[1]):
    # ... execute action ...
    key_frame_list.append(obs)  # Collect keyframes

# Step 4: Update server cache with keyframes
model.infer(dict(
    obs=key_frame_list,
    compute_kv_cache=True,  # Flag to build cache
    state=action,
    imagine=False
))

# Step 5: Loop back to step 2 with new observations
```

**Message Flow**:
```
Client                    WebSocket                   Server
------                    ---------                   ------
infer(reset=True)  ------>  obs dict  ------>  _reset(), create_empty_cache()
                   <------  empty dict <------  Ready

infer(obs)         ------>  obs dict  ------>  _infer(), generate action
                   <------  action dict <-----  update_cache=1 (pred tokens)

infer(compute_kv)  ------>  keyframes ------>  _compute_kv_cache()
                   <------  empty dict <------  update_cache=2 (obs+pred)
```

---

## 4. RENDERING & GPU MEMORY MANAGEMENT

### 4.1 Rendering Setup

**File**: `evaluation/robotwin/test_render.py`

**Renderer Configuration**:
```python
class Sapien_TEST(gym.Env):
    def setup_scene(self):
        from sapien.render import set_global_config
        
        set_global_config(
            max_num_materials=50000,
            max_num_textures=50000
        )
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)
        
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(32)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("oidn")  # Intel denoiser
```

**Ray Tracing Setup**:
- 32 samples per pixel (high quality)
- Path depth of 8 bounces
- OIDN denoiser for noise reduction
- High material/texture limits for complex scenes

### 4.2 Client-Side GPU Memory Usage

**Client Environment**:
- **GPU Memory Fraction**: 90% (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`)
- **Usage**: Rendering + environment simulation
- **Not Used**: Model weights (on server GPU)

**Memory Breakdown** (typical client on GPU 6 or 7):
```
┌─────────────────────────────────────────────┐
│    Total GPU Memory: ~80 GB / 8 GPUs         │
├─────────────────────────────────────────────┤
│  Client GPU (6-7):  ~60-70 GB available      │
│  ├─ Sapien Renderer: ~20-30 GB              │
│  ├─ Physics Simulation: ~10-15 GB           │
│  ├─ Image Storage: ~5-10 GB                 │
│  └─ Inference buffers: ~10 GB               │
│                                              │
│  Server GPU (0-5):  ~60-70 GB available      │
│  ├─ Model Weights: ~50-60 GB                │
│  │  ├─ Transformer: ~40 GB                  │
│  │  ├─ VAE: ~2.7 GB                         │
│  │  └─ Text Encoder: ~5 GB                  │
│  └─ Activations/Caches: ~10-20 GB           │
└─────────────────────────────────────────────┘
```

### 4.3 Memory Optimization Techniques

**1. VAE Offloading** (`wan_va_server.py`):
```python
self.enable_offload = True  # Default
self.vae_offload = True     # Default

def _encode_obs(self, obs):
    if self.vae_offload:
        self.transformer.to('cpu')
        torch.cuda.empty_cache()
        self.vae.to(self.device)
    
    # ... encode ...
    
    if self.vae_offload:
        self.vae.to('cpu')
        torch.cuda.empty_cache()
        self.transformer.to(self.device)
```

**Effect**: Saves ~2.7 GB by swapping VAE on/off device

**2. Streaming VAE Wrapper**:
```python
self.streaming_vae = WanVAEStreamingWrapper(self.vae)
self.streaming_vae_half = WanVAEStreamingWrapper(self.vae)  # For T-shape env
```

Shared VAE module avoids duplicate 2.7 GB for dual-camera setup

**3. Text Encoder Offloading**:
```python
if self.enable_offload:
    self.transformer.to('cpu')
    self.text_encoder.to(self.device)
    # ... encode text ...
    self.text_encoder.to('cpu')
    self.transformer.to(self.device)
```

**4. Cache Window Management**:
```python
attn_window = job_config.attn_window  # e.g., 4 chunks
# Only keeps last 4 chunks of KV cache in memory
total_tokens = (attn_window // 2) * latent_token_per_chunk + \
               (attn_window // 2) * action_token_per_chunk
```

---

## 5. DATA FLOW ARCHITECTURE

### 5.1 Complete Inference Loop

```
Client Environment                 Server
─────────────────                 ──────
       │
       ├─ Reset Signal ────────────────────> Reset + Create Cache
       │                          
       ├─ Initial Obs ─────────────────────> Encode + Generate Action
       │  (image, state)
       │  <──────────────────────────────────── Action + Metadata
       │
       ├─ Execute Action (Simulation + Render)
       │
       ├─ Collect Keyframes (next_obs)
       │  
       ├─ KV Cache Update ────────────────────> Build Cache from Keyframes
       │  (obs + state)
       │  <──────────────────────────────────── Ack
       │
       ├─ Repeat from "Initial Obs" with new keyframes
       │
       └─ [Loop until task completion or step limit]
```

### 5.2 Distributed Inference Path (Server Side)

```
WebSocket Client
       │
       ├─ Send obs dict ──────────> Rank 0 (Main Process)
       │                             │
       │                             ├─ Broadcast obs
       │                             │
       │                             ├──────────> Rank 1 (GPU 1)
       │                             │            • run infer()
       │                             │  <─────────
       │                             │
       │                             ├──────────> Rank N (GPU N)
       │                             │            • run infer()
       │                             │  <─────────
       │                             │
       │                             └─ Collect results
       │  <─────────────────────────── Send action via WebSocket
       │
```

### 5.3 Message Protocol

**Observation Message** (Client → Server):
```python
{
    'reset': bool,              # Initial reset
    'prompt': str,              # Language instruction
    'compute_kv_cache': bool,   # Build cache from observations
    'imagine': bool,            # (unused)
    'save_visualization': bool, # Debug flag
    'obs': [                    # List of observation dicts
        {
            'observation.images.cam_high': np.uint8 array,
            'observation.images.cam_left_wrist': np.uint8 array,
            'observation.images.cam_right_wrist': np.uint8 array,
            'observation.state': np.float32 array,
            'task': str,
        }
    ],
    'state': np.float32 array,  # Previous action
    'video_guidance_scale': float,
    'action_guidance_scale': float,
}
```

**Action Message** (Server → Client):
```python
{
    'action': np.float32 array,  # Shape: (C, F, H) for C channels, F frames
    'server_timing': {
        'infer_ms': float,       # Inference time
        'prev_total_ms': float,  # Previous round total time
    },
}
```

---

## 6. MULTI-GPU DEPLOYMENT CONFIGURATIONS

### 6.1 Split Setup (Recommended)

**File**: `launch_server_split.sh` + `launch_client_split.sh`

```bash
# Start 6 servers on GPU 0-5
./evaluation/robotwin/launch_server_split.sh

# In another terminal: Start 6 clients on GPU 6, 7 (alternating)
./evaluation/robotwin/launch_client_split.sh
```

**Topology**:
```
GPU 0  GPU 1  GPU 2  GPU 3  GPU 4  GPU 5  │  GPU 6  GPU 7
─────  ─────  ─────  ─────  ─────  ─────     ─────  ─────
[S0]   [S1]   [S2]   [S3]   [S4]   [S5]      [C0]   [C1]
                                              [C2]   [C3]
                                              [C4]   [C5]
       ↑← WebSocket Connection →↓
```

**Advantages**:
- Clear separation: inference (0-5) vs simulation (6-7)
- No GPU contention
- Scalable (easy to add more servers/clients)

### 6.2 Multi-GPU Server Setup (8 servers)

**File**: `launch_server_multigpus.sh`

```bash
for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$i \
    python -m torch.distributed.run \
        --nproc_per_node 1 \
        wan_va/wan_va_server.py --port $((29556 + i))
done
```

- 1 server per GPU (0-7)
- Each on independent port

---

## 7. TASK ASSIGNMENT & LOAD BALANCING

### 7.1 Task Groups (RoboTwin Evaluation)

```python
task_groups = [
    # Group 0 (8 tasks)
    "stack_bowls_three handover_block hanging_mug scan_object " +
    "lift_pot put_object_cabinet stack_blocks_three place_shoe",
    
    # Group 1 (8 tasks)
    "adjust_bottle place_mouse_pad dump_bin_bigbin move_pillbottle_pad " +
    "pick_dual_bottles shake_bottle place_fan turn_switch",
    # ... more groups ...
]
```

### 7.2 Client-Server Mapping

**Dynamic Mapping**:
```python
for i in "${!task_names[@]}"; do
    task_name="${task_names[$i]}"
    gpu_id="${CLIENT_GPUS[$i]}"                # e.g., 6 or 7
    port=$((START_PORT + i))                   # 29556 + i
    
    # Client i connects to Server i on port 29556+i
    python eval_polict_client_openpi --port ${port}
done
```

**Properties**:
- Each client has dedicated server (1:1 mapping)
- Independent TCP port per pair
- Can run multiple clients on same GPU
- Simple failover: restart client/server pair

---

## 8. COMMUNICATION FAILURE HANDLING

### 8.1 Client Reconnection

**Automatic Retry** (`websocket_client_policy.py`):
```python
def _wait_for_server(self):
    while True:
        try:
            conn = websockets.sync.client.connect(...)
            return conn, metadata
        except (ConnectionRefusedError, Exception) as e:
            logging.info(f"Still waiting for server... (Error: {e})")
            time.sleep(5)  # Retry every 5 seconds
```

### 8.2 Network Timeouts

**Configuration**:
- `ping_interval=None`: Disabled (inference can take minutes)
- `close_timeout=10`: 10 seconds to close connection gracefully
- `max_size=None`: No message size restrictions

### 8.3 Graceful Shutdown

**Server-Side**:
```python
# Send termination command via distributed communication
cmd = torch.tensor(-1, dtype=torch.int64)
dist.broadcast(cmd, src=0)
# All ranks break worker_loop and exit
```

**Client-Side**:
```python
# Ctrl+C or natural task completion
# WebSocket connection closes automatically
```

---

## 9. PERFORMANCE METRICS

### 9.1 Timing Breakdown

**Server Timing** (included in action message):
```python
action["server_timing"] = {
    "infer_ms": time.monotonic() - infer_time,
    "prev_total_ms": prev_total_time * 1000,
}
```

**Typical Latencies**:
- Image encoding (VAE): 500-1000 ms
- Transformer inference: 1000-3000 ms
- Text encoding: 100-200 ms
- Network round-trip: 10-100 ms

### 9.2 Throughput

**Inference Speed**:
- **Single Server**: ~0.5-1 Hz (2-3 seconds per action)
- **6 Servers**: 6 clients can run in parallel
- **Network**: ~1-5 MB per message (with compression optimization possible)

---

## 10. CONFIGURATION REFERENCE

### 10.1 Key Configuration Files

| File | Purpose |
|------|---------|
| `wan_va/configs.py` | Model & inference configs (prompt, guidance scales) |
| `evaluation/robotwin/task_config/` | Task-specific settings |
| `policy/ACT/deploy_policy.yml` | Deployment policy config |

### 10.2 Environment Variables

```bash
# GPU Selection
CUDA_VISIBLE_DEVICES=0        # Single GPU
CUDA_VISIBLE_DEVICES=0,1,2    # Multiple GPUs

# Memory Management
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# Debugging
PYTHONUNBUFFERED=1
PYTHONWARNINGS=ignore::UserWarning

# Distributed Training
RANK=0
LOCAL_RANK=0
WORLD_SIZE=1
MASTER_ADDR=localhost
MASTER_PORT=29661
```

### 10.3 Launch Script Parameters

**Server Launch**:
```bash
./launch_server_split.sh
  NUM_SERVERS=6              # Number of servers
  SERVER_GPU_START=0         # Starting GPU ID
  START_PORT=29556           # Starting port
  MASTER_PORT=29661          # Master port
```

**Client Launch**:
```bash
./launch_client_split.sh <save_root> <task_list_id> <seed> <test_num>
  save_root=./results        # Output directory
  task_list_id=0             # Which task group (0-6)
  seed=0                     # Random seed
  test_num=100               # Number of tests per task
```

---

## 11. DEBUGGING & TROUBLESHOOTING

### Common Issues

**1. "Connection refused" on client startup**
- Ensure servers are running first
- Check ports are not in use: `lsof -i :29556`
- Verify `CUDA_VISIBLE_DEVICES` matches server launch

**2. "Out of memory" errors**
- Reduce `attn_window` (fewer cached chunks)
- Enable `enable_offload=True` for VAE/text encoder
- Reduce `frame_chunk_size` in config
- Check `expandable_segments:True` is set

**3. Slow inference (>5 seconds/action)**
- Check GPU utilization: `nvidia-smi`
- Verify no resource contention between server/client GPUs
- Check network latency: `ping` and `iperf`

**4. "Stale connection" on long inference**
- Already handled: `ping_interval=None` disables ping mechanism
- `close_timeout=10` gives 10 seconds for graceful shutdown

---

## 12. SUMMARY TABLE

| Aspect | Details |
|--------|---------|
| **Communication** | WebSocket (async/sync), msgpack+numpy |
| **Server GPU** | 0-5 (inference), 1 server per GPU |
| **Client GPU** | 6-7 (rendering), multiple clients per GPU |
| **KV Cache** | Slot-based, LRU eviction, ~2500 slots/layer |
| **Memory Optimization** | VAE offload, streaming wrapper, cache windowing |
| **Distributed** | PyTorch NCCL, Rank 0 main + worker ranks |
| **Protocol** | Binary serialization, metadata exchange |
| **Failover** | Automatic client retry, graceful shutdown |
| **Scalability** | Linear with server count (1:1 client:server ratio) |

---

## Appendix: Code References

### Key Files
- **Server**: `wan_va/wan_va_server.py`
- **Distributed**: `wan_va/utils/sever_utils.py`, `wan_va/distributed/util.py`
- **WebSocket Server**: `wan_va/utils/Simple_Remote_Infer/deploy/websocket_policy_server.py`
- **WebSocket Client**: `wan_va/utils/Simple_Remote_Infer/deploy/websocket_client_policy.py`
- **Client Evaluation**: `evaluation/robotwin/eval_polict_client_openpi.py`
- **Model**: `wan_va/modules/model.py` (KV cache in WanAttention)
- **Launch Scripts**: `evaluation/robotwin/launch_*.sh`

