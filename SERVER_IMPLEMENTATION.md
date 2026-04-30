# Lingbot-VA Server-Side Implementation for Evaluation Pipeline

## 1. WebSocket Server Architecture

### 1.1 WebsocketPolicyServer (Core Server Implementation)
**Location:** `wan_va/utils/Simple_Remote_Infer/deploy/websocket_policy_server.py`

#### Key Architecture:
- **Framework:** `websockets.asyncio.server` (Python websockets library)
- **Protocol:** WebSocket with msgpack binary serialization for numpy arrays
- **Concurrency:** Fully async, handles multiple clients simultaneously
- **Message Format:** msgpack-encoded numpy arrays

#### Server Initialization:
```python
class WebsocketPolicyServer:
    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
```

**Parameters:**
- `policy`: The inference model/policy object (wrapped in DistributedModelWrapper for multi-GPU setups)
- `host`: Defaults to "0.0.0.0" (listen on all interfaces)
- `port`: Server port (assigned via launch scripts)
- `metadata`: Optional metadata sent to clients on connection

#### Server Event Loop:
```python
async def run(self):
    async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
            ping_interval=None,
            ping_timeout=None,
    ) as server:
        await server.serve_forever()
```

**Configuration Details:**
- `compression=None`: No compression (raw binary data)
- `max_size=None`: Unlimited message size for large observation/action tensors
- `process_request=_health_check`: Handles `/healthz` endpoint for health checks
- `ping_interval=None, ping_timeout=None`: Disabled to prevent timeout issues during long inference

### 1.2 Per-Client Handler (_handler method)

```python
async def _handler(self, websocket: _server.ServerConnection):
    logger.info(f"Connection from {websocket.remote_address} opened")
    packer = Packer()
    
    # Send server metadata to client
    await websocket.send(packer.pack(self._metadata))
    
    prev_total_time = None
    while True:
        try:
            start_time = time.monotonic()
            obs = unpackb(await websocket.recv())
            
            # Perform inference
            infer_time = time.monotonic()
            action = self._policy.infer(obs)
            infer_time = time.monotonic() - infer_time
            
            # Add timing information
            action["server_timing"] = {
                "infer_ms": infer_time * 1000,
            }
            if prev_total_time is not None:
                action["server_timing"]["prev_total_ms"] = prev_total_time * 1000
            
            # Send response
            await websocket.send(packer.pack(action))
            prev_total_time = time.monotonic() - start_time
```

**Key Features:**
- **Per-Client Connection:** Each client gets its own independent handler coroutine
- **Synchronous Inference:** Calls `self._policy.infer(obs)` within async context
- **Timing Tracking:** Records inference time and round-trip time for metrics
- **Error Handling:** Catches exceptions and sends traceback to client
- **Connection Lifecycle:** Detects `websockets.ConnectionClosed` to gracefully handle disconnects

---

## 2. Multi-GPU Distribution & Session Management

### 2.1 Distributed Model Wrapper
**Location:** `wan_va/utils/sever_utils.py`

```python
class DistributedModelWrapper:
    def __init__(self, model, local_rank):
        self.model = model
        self.local_rank = local_rank
    
    def infer(self, obs):
        return distributed_infer(self.model, obs, self.local_rank)
```

#### Distributed Inference Flow:
```python
def distributed_infer(model, obs, local_rank):
    rank = dist.get_rank()
    assert rank == local_rank, "distributed_infer can only run at rank 0"
    
    # Signal rank 0 command to other ranks
    cmd = torch.tensor(1, dtype=torch.int64, device='cuda' if torch.cuda.is_available() else 'cpu')
    dist.broadcast(cmd, src=0)
    
    # Broadcast observation to all ranks
    obj_list = [obs]
    dist.broadcast_object_list(obj_list, src=0)
    
    # Execute inference on rank 0 (only rank 0 has actual model with data)
    result = model.infer(obs)
    
    return result
```

**Key Design Patterns:**
1. **Rank 0 Only**: Only rank 0 (local_rank=0) runs the WebSocket server
2. **PyTorch Distributed**: Uses `torch.distributed` for inter-rank communication
3. **Broadcast Coordination**: Commands and data are broadcast to all workers
4. **Worker Loop**: Other ranks execute `worker_loop()` waiting for commands

### 2.2 Worker Loop for Multi-GPU
```python
def worker_loop(model, local_rank):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rank = dist.get_rank()
    
    while True:
        # Wait for command broadcast
        cmd = torch.zeros(1, dtype=torch.int64, device=device)
        dist.broadcast(cmd, src=0)
        cmd_val = cmd.item()
        
        if cmd_val == -1:  # Exit signal
            break
        elif cmd_val == 1:  # Inference command
            obj_list = [None]
            dist.broadcast_object_list(obj_list, src=0)
            obs = obj_list[0]
            _ = model.infer(obs)  # Execute inference
        else:
            pass
    
    logger.info(f"[worker_loop] Rank {rank} exiting.")
```

**Why This Design:**
- Allows shared model weights across GPUs (no duplication)
- Rank 0 handles I/O (WebSocket), other ranks pre-compute (optional)
- Future optimization: Other ranks could pre-load next batch

---

## 3. KV Cache Management Per Session

### 3.1 Cache Lifecycle in VA_Server
**Location:** `wan_va/wan_va_server.py`

#### Session Reset & Cache Initialization:
```python
def _reset(self, prompt=None):
    logger.info('Reset.')
    
    # Clear old caches
    self.transformer.clear_cache(self.cache_name)
    self.streaming_vae.clear_cache()
    
    # Create fresh KV cache with configured window size
    self.transformer.create_empty_cache(
        self.cache_name,
        self.job_config.attn_window,
        latent_token_per_chunk,
        action_token_per_chunk,
        dtype=self.dtype,
        device=self.device,
        batch_size=2 if self.use_cfg else 1
    )
```

**Cache Parameters:**
- `cache_name`: Named cache identifier (default: 'pos')
- `attn_window`: Sliding window size for attention (limits history length)
- `latent_token_per_chunk`: Tokens per frame chunk for video latents
- `action_token_per_chunk`: Tokens per action frame
- `batch_size`: 1 for normal inference, 2 for classifier-free guidance (CFG)

#### Cache Structure in Attention Layer:
```python
def init_kv_cache(self, cache_name, total_tolen, num_head, head_dim, device, dtype, batch_size):
    if self.attn_caches is None:
        return
    self.attn_caches[cache_name] = {
        'k': torch.empty([batch_size, total_tolen, num_head, head_dim], device=device, dtype=dtype),
        'v': torch.empty([batch_size, total_tolen, num_head, head_dim], device=device, dtype=dtype),
        'id': torch.full((total_tolen, ), -1, device=device),
        "mask": torch.zeros((total_tolen, ), dtype=torch.bool, device=device),
        "is_pred": torch.zeros((total_tolen, ), dtype=torch.bool, device=device),
    }
```

**Cache Fields:**
- **k, v**: Key and value tensors (pre-allocated for fixed window size)
- **id**: Sequence ID for each cache slot (for eviction policy)
- **mask**: Boolean mask indicating which slots are valid
- **is_pred**: Boolean indicating if slot is from inference vs. training context

### 3.2 Cache Update Mechanism

#### During Inference (update_cache=1):
```python
def update_cache(self, cache_name, key, value, is_pred):
    cache = self.attn_caches[cache_name]
    key_size = key.shape[1]
    
    # Allocate slots (LRU eviction if needed)
    slots = self.allocate_slots(cache_name, key_size)
    new_id = self._next_cache_id(cache_name)
    
    cache['k'][:, slots] = key
    cache['v'][:, slots] = value
    cache['mask'][slots] = True
    cache['id'][slots] = new_id
    cache['is_pred'][slots] = is_pred
    return slots
```

**Slot Allocation Strategy:**
```python
def allocate_slots(self, cache_name, key_size):
    cache = self.attn_caches[cache_name]
    mask = cache["mask"]
    ids = cache["id"]
    
    # Find free slots
    free = (~mask).nonzero(as_tuple=False).squeeze(-1)
    
    if free.numel() < key_size:
        # Need to evict old entries (LRU)
        used = mask.nonzero(as_tuple=False).squeeze(-1)
        used_ids = ids[used]
        order = torch.argsort(used_ids)
        need = key_size - free.numel()
        to_free = used[order[:need]]  # Evict oldest
        mask[to_free] = False
        ids[to_free] = -1
        free = (~mask).nonzero(as_tuple=False).squeeze(-1)
    
    return free[:key_size]
```

### 3.3 Cache Clear & Cleanup

#### Per-Step Cache Clear (for prediction):
```python
def clear_pred_cache(self, cache_name):
    if self.attn_caches is None:
        return
    cache = self.attn_caches[cache_name]
    is_pred = cache['is_pred']
    cache['mask'][is_pred] = False  # Mark prediction slots as invalid
```

#### Full Session Cache Clear:
```python
def clear_cache(self, cache_name):
    if self.attn_caches is None:
        return
    self.attn_caches[cache_name] = None  # Delete entire cache
```

### 3.4 Multi-Session Cache Management

**Scenario:** Multiple clients accessing the server simultaneously

```
Client A:
  1. Sends reset + prompt_A
     -> VA_Server._reset() clears global transformer cache
     -> Creates new cache for Client A
  2. Sends obs batch 1
     -> transformer.infer() with update_cache=2 (compute KV)
     -> KV cache updated for Client A
  3. Sends obs batch 2
     -> transformer.infer() uses accumulated KV cache
     
Client B (arrives while Client A is running):
  1. Sends reset + prompt_B
     -> VA_Server._reset() clears global cache (PROBLEM!)
     -> Wipes Client A's accumulated KV cache
     -> Creates new cache for Client B
```

**IMPORTANT LIMITATION:** The current server implementation is NOT truly multi-session:
- Only ONE session can be active at a time
- New client reset clears all KV cache (affects other clients)
- This is by design for single-worker evaluation

**Solution in Multi-Client Setup:**
- Clients are balanced across MULTIPLE server instances (one per GPU)
- Each server instance has its own model and KV cache
- Load-balancing happens at the launch script level

---

## 4. Server Launch Scripts

### 4.1 Single-GPU Server Launch
**File:** `evaluation/robotwin/launch_server.sh`

```bash
START_PORT=${START_PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}

python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port $MASTER_PORT \
    wan_va/wan_va_server.py \
    --config-name robotwin \
    --port $START_PORT \
    --save_root visualization/
```

**Port Assignment:**
- `START_PORT`: Base port for WebSocket server (default: 29056)
- `MASTER_PORT`: PyTorch distributed master port (default: 29061)

### 4.2 Multi-GPU Server Launch
**File:** `evaluation/robotwin/launch_server_multigpus.sh`

```bash
START_PORT=${START_PORT:-29556}
MASTER_PORT=${MASTER_PORT:-29661}

for i in {0..7}; do  
    CURRENT_PORT=$((START_PORT + i))           # 29556, 29557, ..., 29563
    CURRENT_MASTER_PORT=$((MASTER_PORT + i))  # 29661, 29662, ..., 29668
    
    LOG_FILE="${LOG_DIR}/server_${i}_${batch_time}.log"
    
    CUDA_VISIBLE_DEVICES=$i \
    nohup python -m torch.distributed.run \
        --nproc_per_node 1 \
        --master_port $CURRENT_MASTER_PORT \
        wan_va/wan_va_server.py \
        --config-name robotwin \
        --save_root $save_root \
        --port $CURRENT_PORT > $LOG_FILE 2>&1 &
    sleep 2
done
```

**Key Features:**
- **Per-GPU Isolation:** Each GPU runs its own server instance
- **Port Sequence:** Each server gets incremental port (29556 + GPU_id)
- **Separate Master Ports:** Each distributed process group has its own master port
- **Nohup Daemonization:** Runs in background with output logging

### 4.3 Single-GPU Client Launch
**File:** `evaluation/robotwin/launch_client.sh`

```bash
PORT=29056
TASK=adjust_bottle

python -m evaluation.robotwin.eval_polict_client_openpi \
    --config policy/ACT/deploy_policy.yml \
    --task_name ${TASK} \
    --task_config demo_clean \
    --port ${PORT} \
    --test_num 100 \
    ...
```

### 4.4 Multi-GPU Client Launch
**File:** `evaluation/robotwin/launch_client_multigpus.sh`

```bash
start_port=29556
num_gpus=8

# Load-balanced task distribution
mapfile -t task_groups < <(python -m evaluation.robotwin.balance_tasks \
    --num_clients ${NUM_GROUPS} --verbose)

# Launch one client per task
for i in "${!task_names[@]}"; do
    task_name="${task_names[$i]}"
    gpu_id=$(( i % num_gpus ))
    port=$(( start_port + i ))
    
    CUDA_VISIBLE_DEVICES=${gpu_id} \
    python -m evaluation.robotwin.eval_polict_client_openpi \
        --task_name ${task_name} \
        --port ${port} \
        ... > "$log_file" 2>&1 &
done
```

**Load Balancing Logic:**
- Task-to-client assignment via `balance_tasks.py`
- Port assignment: `base_port + client_index`
- GPU assignment: Round-robin or custom via `CLIENT_GPUS` array

### 4.5 Session-Level Evaluation Pipeline
**File:** `evaluation/robotwin/launch_session_eval.sh`

Three-phase pipeline:

#### Phase 1: Collect Valid Seeds (Parallel)
```bash
for w in $(seq 0 $(( COLLECT_WORKERS - 1 ))); do
    gpu_id="${COLLECT_GPUS[$(( w % ${#COLLECT_GPUS[@]} ))]}"
    
    CUDA_VISIBLE_DEVICES=${gpu_id} \
    python -m evaluation.robotwin.collect_seeds \
        --worker_id ${w} \
        --num_workers ${COLLECT_WORKERS} \
        --resume & 
done
# Merge results -> valid_seeds.json
```

#### Phase 2: Balance Task Assignments
```bash
python -m evaluation.robotwin.balance_tasks \
    --mode session \
    --valid_seeds valid_seeds.json \
    --num_clients ${num_clients} \
    --output_dir task_assignments/
# Output: task_assignments/client_0.json, client_1.json, ...
```

#### Phase 3: Launch Eval Clients
```bash
for i in $(seq 0 $(( num_clients - 1 ))); do
    gpu_id="${CLIENT_GPUS[$(( i % ${#CLIENT_GPUS[@]} ))]}"
    port=$(( START_PORT + i ))
    assignment_file="${ASSIGNMENT_DIR}/client_${i}.json"
    
    CUDA_VISIBLE_DEVICES=${gpu_id} \
    python -m evaluation.robotwin.eval_session_client \
        --assignment "${assignment_file}" \
        --port ${port} \
        --client_id ${i} \
        ... &
done
```

---

## 5. Client-Side WebSocket Implementation

### 5.1 WebsocketClientPolicy
**Location:** `wan_va/utils/Simple_Remote_Infer/deploy/websocket_client_policy.py`

```python
class WebsocketClientPolicy:
    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, 
                 api_key: Optional[str] = None) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()
```

#### Connection Retry Loop:
```python
def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
    logging.info(f"Waiting for server at {self._uri}...")
    while True:
        try:
            headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
            conn = websockets.sync.client.connect(
                self._uri,
                compression=None,
                max_size=None,
                additional_headers=headers,
                ping_interval=None,      # Disabled (long inference times)
                close_timeout=10
            )
            metadata = unpackb(conn.recv())
            return conn, metadata
        except (ConnectionRefusedError, Exception) as e:
            logging.info(f"Still waiting for server... (Error: {e})")
            time.sleep(5)
```

**Configuration:**
- **ping_interval=None**: Disabled to prevent timeouts during long inference
- **close_timeout=10**: 10-second grace period for close handshake
- **retry_interval=5s**: Wait 5 seconds before retrying failed connections

#### Inference Interface:
```python
def infer(self, obs: Dict) -> Dict:
    data = self._packer.pack(obs)
    self._ws.send(data)
    response = self._ws.recv()
    if isinstance(response, str):
        raise RuntimeError(f"Error in inference server:\n{response}")
    return unpackb(response)

def reset(self) -> None:
    self.infer(dict(reset=True))
```

---

## 6. Data Flow Diagram

```
┌─────────────────────────────────────────────────────┐
│                  EVALUATION CLIENT                  │
│                                                     │
│  eval_session_client.py                            │
│  - Reads task assignment JSON                      │
│  - Runs episodes sequentially                      │
│  - Creates WebsocketClientPolicy                   │
└──────────────────────┬──────────────────────────────┘
                       │
                       │ WebSocket (msgpack binary)
                       │ obs = {obs, prompt, reset, ...}
                       │
┌──────────────────────▼──────────────────────────────┐
│                   SERVER (1 per GPU)               │
│  wan_va_server.py + WebsocketPolicyServer          │
│                                                     │
│  Rank 0:  WebSocket listening on port 29556+i     │
│    - Accepts connection                           │
│    - Unpacks msgpack data                         │
│    - Calls DistributedModelWrapper.infer()       │
│    - Packs action response                        │
│    - Sends via WebSocket                          │
│                                                     │
│  Rank 1-7:  worker_loop() (if multi-rank)         │
│    - Receives broadcast from rank 0                │
│    - May pre-load KV cache (future)               │
└──────────────────────┬──────────────────────────────┘
                       │
                       │ Inference on GPU
                       │
┌──────────────────────▼──────────────────────────────┐
│          VA_Server (inference model)               │
│                                                     │
│  _reset(prompt):                                   │
│    - Clear transformer KV cache                   │
│    - Encode prompt via T5 text encoder            │
│    - Create empty KV cache with window size       │
│                                                     │
│  _encode_obs(obs):                                 │
│    - Resize images via VAE encoder                │
│    - Offload transformer to CPU (if enabled)     │
│    - Normalize latents                            │
│                                                     │
│  _compute_kv_cache(obs):                          │
│    - Encode observation to latent                 │
│    - Run transformer with update_cache=2         │
│    - Update KV cache (accumulate history)        │
│                                                     │
│  _infer(obs):                                      │
│    - Run transformer denoising loops              │
│    - Use KV cache (update_cache=0/1/2)           │
│    - Return action tensor                        │
└──────────────────────┬──────────────────────────────┘
                       │
                       │ action = {action, timing}
                       │ (binary msgpack)
                       │
┌──────────────────────▼──────────────────────────────┐
│                  EVALUATION CLIENT                  │
│  - Unpacks action                                   │
│  - Executes in environment (SAPIEN)               │
│  - Collects next observation                      │
│  - Sends to server (loop)                         │
└─────────────────────────────────────────────────────┘
```

---

## 7. Key Configuration Parameters

### 7.1 Server Config (shared_config.py)
```python
va_shared_cfg.host = '0.0.0.0'           # Listen on all interfaces
va_shared_cfg.port = 29536               # Default port (overridden by launch)
va_shared_cfg.enable_offload = True      # Offload VAE/text encoder to CPU
va_shared_cfg.vae_offload = True         # Offload VAE between transformer calls
```

### 7.2 Cache Config (from job_config)
```python
self.job_config.attn_window = 8          # Sliding window size (frames of history)
self.job_config.frame_chunk_size = 4     # Frames per inference step
self.job_config.action_per_frame = 5     # Action dimensions per frame
self.job_config.patch_size = (1, 2, 2)   # Patch division for latents
```

### 7.3 Guidance Config
```python
guidance_scale = 5                       # Video CFG scale
action_guidance_scale = 1                # Action CFG scale
```

---

## 8. Multi-Client Handling Summary

### 8.1 True Multi-Session (Not Implemented)
- ❌ Would require per-client KV cache management
- ❌ Would need session-to-cache mapping
- ❌ Currently: New reset clears all caches

### 8.2 Actual Multi-Client Pattern (Evaluation)
- ✅ **Multiple server instances** (8 servers on 8 GPUs)
- ✅ Each server serves ONE client at a time (sequential episodes)
- ✅ Load balancing via task distribution scripts
- ✅ Port assignment: `base_port + server_id`

### 8.3 Client Connection Lifecycle
1. **Client connects** to `ws://0.0.0.0:{port}`
2. **Server accepts** in async handler
3. **Client sends** `{reset=True, prompt="..."}` 
4. **Server clears** KV cache, encodes prompt
5. **Loop:** Client sends `{obs, prompt}`, server returns `{action, timing}`
6. **Episode end** or timeout -> client disconnects
7. **Next client** can connect to same server port

---

## 9. Timing & Performance Characteristics

### 9.1 Inference Timing
Server tracks and returns:
```python
action["server_timing"] = {
    "infer_ms": infer_time * 1000,           # Time to run inference
    "prev_total_ms": prev_total_time * 1000  # Time from previous step
}
```

### 9.2 VAE Offloading
```
Without offload (vae_offload=False):
  - VAE stays on GPU (~2.7 GB VRAM)
  - Saves 100-200ms per chunk (no swap)
  - Requires more VRAM
  
With offload (vae_offload=True):
  - VAE CPU ↔ GPU swap each step
  - Adds 100-200ms latency
  - Reduces peak VRAM usage
```

### 9.3 Typical Pipeline Latency
```
1. Client sends obs       ~0.1ms
2. Server unpacks         ~1ms
3. VAE encode             ~100-300ms (offload time)
4. Transformer infer      ~1000-3000ms (depends on steps)
5. Server packs action    ~1ms
6. Send to client         ~0.1ms
─────────────────────────────────
Total round-trip          ~1100-3300ms per step
```

---

## 10. Health Check & Monitoring

### 10.1 Health Check Endpoint
```python
def _health_check(connection: _server.ServerConnection,
                  request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
```

**Usage:**
```bash
curl http://localhost:29056/healthz  # Returns "OK\n" if server alive
```

### 10.2 Logging
Server logs to:
- Stdout (if run in foreground)
- `logs/server_${GPU_ID}_${TIMESTAMP}.log` (if run with nohup)
- Client logs: `logs/session_client_${i}_${TIMESTAMP}.log`

---

## 11. Summary Table

| Aspect | Details |
|--------|---------|
| **Protocol** | WebSocket with msgpack serialization |
| **Concurrency** | Async Python (asyncio) |
| **Server Framework** | `websockets.asyncio.server` |
| **KV Cache Type** | Rolling window with LRU eviction |
| **Cache Scope** | Per server instance (not per client session) |
| **Multi-GPU | Separate server per GPU, distributed PyTorch coordination |
| **Port Scheme** | `base_port + server_id` (typically 29556-29563 for 8 servers) |
| **Session Lifecycle** | Triggered by client `{reset=True}` |
| **Cache Clear** | Full clear on reset, partial clear (pred slots) on inference step |
| **Message Format** | Binary msgpack (no compression, unlimited size) |
| **Timeout** | No ping/timeout (long inference times) |
| **Health Check** | `/healthz` HTTP endpoint |
| **Multi-Client Support** | Sequential (reset wipes cache), recommend multiple servers instead |

