# Lingbot-VA Server Architecture Diagrams

## 1. Multi-Server Deployment (8-GPU Setup)

```
┌────────────────────────────────────────────────────────────────────────┐
│                         EVALUATION INFRASTRUCTURE                      │
└────────────────────────────────────────────────────────────────────────┘

                    Task Assignment (Load-Balanced)
                                │
                ┌───────────────┼───────────────┐
                │               │               │
        Client 0        Client 1        ...     Client 7
      (task: A)        (task: B)              (task: H)
           │                │                    │
           │ ws://          │ ws://              │ ws://
           │ 0.0.0.0:29556  │ 0.0.0.0:29557     │ 0.0.0.0:29563
           │                │                    │
    ┌──────▼──┐      ┌──────▼──┐        ┌──────▼──┐
    │ Server0 │      │ Server1 │   ...  │ Server7 │
    │ GPU: 0  │      │ GPU: 1  │        │ GPU: 7  │
    │ Rank 0  │      │ Rank 0  │        │ Rank 0  │
    └─────────┘      └─────────┘        └─────────┘
        │                │                    │
        │          [Optional: Worker ranks]   │
        │          Rank 1, 2, 3, ... (unused) │
        │                │                    │
        └────────────────┼────────────────────┘
                         │
                    Model weights (shared via distributed)
```

**Key Points:**
- One server per GPU (8 servers on 8 GPUs)
- Each server handles its own clients sequentially
- Load balancing at application level (not server level)
- Separate master ports for each PyTorch distributed group

---

## 2. Per-Server Internal Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      WAN_VA_SERVER.py                              │
│                      (Single GPU Instance)                         │
└─────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │  WebsocketPolicyServer                                      │
    │  (async event loop)                                         │
    │                                                             │
    │  listen(host='0.0.0.0', port=29556)                        │
    │  ├─ Health check endpoint: GET /healthz                    │
    │  └─ Handler: async def _handler(websocket)                 │
    │     ├─ Connection tracking (logging)                       │
    │     ├─ Message unpacking (msgpack)                         │
    │     ├─ Timing measurement                                  │
    │     └─ Exception handling (traceback to client)            │
    └─────────────────────────────────────────────────────────────┘
                           │
                    Calls per request
                           │
    ┌─────────────────────▼─────────────────────────────────────┐
    │  DistributedModelWrapper                                  │
    │  (Rank 0 only - handles I/O)                             │
    │                                                           │
    │  def infer(obs) -> dict                                   │
    │  ├─ Broadcast command to other ranks                     │
    │  ├─ Broadcast observation                                │
    │  └─ Execute model.infer(obs) [on Rank 0]                │
    └─────────────────────▼─────────────────────────────────────┘
                           │
    ┌─────────────────────▼─────────────────────────────────────┐
    │  VA_Server (PyTorch Model)                               │
    │                                                           │
    │  Model State (per instance, not per session):            │
    │  ├─ self.transformer (model weights)                     │
    │  ├─ self.vae (VAE encoder)                               │
    │  ├─ self.text_encoder (T5)                               │
    │  ├─ self.prompt_embeds (current session)                │
    │  ├─ self.negative_prompt_embeds (CFG)                   │
    │  └─ self.frame_st_id (current position)                 │
    │                                                           │
    │  Methods:                                                │
    │  ├─ infer(obs: dict) -> dict                            │
    │  │  ├─ reset=True?   → _reset(prompt)                   │
    │  │  ├─ compute_kv_cache=True?  → _compute_kv_cache()   │
    │  │  └─ else → _infer(obs)                               │
    │  ├─ _reset(prompt)                                       │
    │  │  ├─ transformer.clear_cache()                         │
    │  │  ├─ streaming_vae.clear_cache()                      │
    │  │  ├─ transformer.create_empty_cache(...)             │
    │  │  └─ encode_prompt() via T5                           │
    │  ├─ _encode_obs(obs)                                     │
    │  │  ├─ Resize images (bilinear)                         │
    │  │  ├─ VAE encode chunk (streaming)                     │
    │  │  └─ Normalize latents                                │
    │  ├─ _compute_kv_cache(obs)                             │
    │  │  ├─ Encode observation                               │
    │  │  ├─ transformer(..., update_cache=2)                │
    │  │  └─ Update KV cache with context                    │
    │  └─ _infer(obs, frame_st_id)                           │
    │     ├─ Generate random latents & actions               │
    │     ├─ Video diffusion loop (50-100 steps)             │
    │     ├─ Action diffusion loop (10-20 steps)             │
    │     ├─ Use transformer KV cache                         │
    │     └─ Return postprocessed action                      │
    └─────────────────────▼─────────────────────────────────────┘
                           │
                      GPU Memory
                           │
    ┌─────────────────────▼─────────────────────────────────────┐
    │  KV Cache (Attention Layers)                             │
    │                                                           │
    │  struct KVCache {                                        │
    │    k: [batch_size, window_tokens, heads, head_dim]      │
    │    v: [batch_size, window_tokens, heads, head_dim]      │
    │    id: [window_tokens]  (LRU eviction id)              │
    │    mask: [window_tokens] (valid slot mask)             │
    │    is_pred: [window_tokens] (prediction flag)          │
    │  }                                                       │
    │                                                           │
    │  Operations:                                             │
    │  ├─ init_kv_cache(window_size)  [reset]                │
    │  ├─ update_cache(k, v, is_pred)  [inference]           │
    │  ├─ allocate_slots(num_slots)  [LRU eviction]          │
    │  ├─ clear_pred_cache()  [between diffusion steps]      │
    │  └─ clear_cache()  [reset session]                     │
    └──────────────────────────────────────────────────────────┘

    ┌────────────────────────────────────────────────────────────┐
    │  PyTorch Distributed (if nproc_per_node > 1)             │
    │                                                            │
    │  Rank 0: WebSocket server (processes requests)           │
    │  Rank 1-N: worker_loop() (standby, ready for tasks)      │
    │                                                            │
    │  Communication via torch.distributed:                     │
    │  ├─ broadcast(cmd, src=0)  [signal ranks]                │
    │  ├─ broadcast_object_list([obs], src=0)  [share data]    │
    │  └─ Coordinated inference [future optimization]          │
    └────────────────────────────────────────────────────────────┘
```

---

## 3. Session Lifecycle

```
┌──────────────────────────────────────────────────────────────┐
│                    CLIENT CONNECTS                           │
│         ws://0.0.0.0:29556 (WebSocket handshake)            │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│              SERVER._handler() STARTS                        │
│            (Async coroutine for this client)                │
└──────────────┬───────────────────────────────────────────────┘
               │
               ├─→ pack(metadata) → client
               │   (server initialization info)
               │
               ├─ WAIT FOR FIRST MESSAGE
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│              CLIENT SENDS: {reset=True, prompt}             │
│                  (Session initialization)                   │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│         SERVER.infer(obs) → VA_SERVER._reset(prompt)        │
│                                                              │
│  1. self.transformer.clear_cache('pos')                    │
│  2. self.streaming_vae.clear_cache()                       │
│  3. self.streaming_vae_half.clear_cache() [optional]       │
│     → Previous session's KV cache is WIPED                 │
│                                                              │
│  4. self.prompt_embeds = T5.encode(prompt)                │
│  5. self.transformer.create_empty_cache(                  │
│       cache_name='pos',                                    │
│       attn_window=8,                                       │
│       latent_tokens_per_chunk=XXX,                         │
│       action_tokens_per_chunk=XXX,                         │
│       batch_size=2 if CFG else 1,                          │
│       device=cuda, dtype=bfloat16                          │
│     )                                                       │
│     → Fresh empty cache allocated                          │
│                                                              │
│  6. self.frame_st_id = 0                                   │
│  7. self.init_latent = None                                │
└──────────────┬───────────────────────────────────────────────┘
               │
               ├─ RESPOND: pack({}) → client
               │
               ├─ WAIT FOR NEXT MESSAGE
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│     EPISODE LOOP: CLIENT SENDS {obs, prompt, ...}           │
│                                                              │
│  Per step:                                                  │
│  1. CLIENT → SERVER: obs (images, state, prompt)           │
│  2. SERVER: _infer(obs, frame_st_id=0)                    │
│     ├─ Frame 0: Encode obs via VAE → latent               │
│     ├─ Run video diffusion (50 steps)                     │
│     │  Each step: transformer(..., update_cache=1/0)     │
│     │  - update_cache=1: Last step → save to KV cache    │
│     │  - update_cache=0: Other steps → query cache       │
│     ├─ Run action diffusion (10 steps)                    │
│     │  Similar cache pattern                              │
│     ├─ Postprocess action                                 │
│     └─ Return {action, server_timing}                     │
│  3. SERVER → CLIENT: pack({action, timing})               │
│                                                              │
│  After step:                                               │
│  ├─ KV cache accumulates history (rolling window)          │
│  ├─ self.frame_st_id += frames_in_this_step              │
│  └─ If new reset: Go back to clear all caches            │
└──────────────┬───────────────────────────────────────────────┘
               │
               ├─ LOOP (repeat for next episode step)
               │
               ├─ EPISODE ENDS
               │  (environment done or step limit)
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│              CLIENT DISCONNECTS                             │
│     (ConnectionClosed exception in _handler)               │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│     SERVER: Clean up & close connection                     │
│                                                              │
│  - KV cache remains in memory (for next client)             │
│  - If new client: Will be cleared by next reset=True       │
│  - Wait for next connection on same port                   │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. KV Cache Evolution During Inference

```
INITIAL STATE (after reset):
┌─────────────────────────────────────────────────────────────┐
│ KV Cache Structure (attn_window=8)                         │
├─────────────────────────────────────────────────────────────┤
│ Slots:    [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]          │
│ mask:     [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]           │
│ id:       [-1] [-1] [-1] [-1] [-1] [-1] [-1] [-1]           │
│ is_pred:  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]           │
└─────────────────────────────────────────────────────────────┘

AFTER STEP 0 (obs encoding, update_cache=2):
┌─────────────────────────────────────────────────────────────┐
│ KV Cache (context from first observation)                  │
├─────────────────────────────────────────────────────────────┤
│ Slots:    [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]          │
│ mask:     [X]  [X]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]           │ (encoded obs)
│ id:       [0]  [1]  [-1] [-1] [-1] [-1] [-1] [-1]          │
│ is_pred:  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]           │ (not prediction)
└─────────────────────────────────────────────────────────────┘
  frame_st_id: 0 → 2 (advanced by observation frames)

DURING VIDEO DIFFUSION (50 steps):
┌─────────────────────────────────────────────────────────────┐
│ KV Cache (accumulating during denoising)                   │
├─────────────────────────────────────────────────────────────┤
│ Per step:                                                  │
│   if step < 49:                                            │
│     output = transformer(input, update_cache=0)           │
│     → Query cache but DON'T save                           │
│   else:                                                    │
│     output = transformer(input, update_cache=1)           │
│     → Query cache AND save prediction to cache            │
│       (is_pred=True for these slots)                      │
└─────────────────────────────────────────────────────────────┘

AFTER VIDEO DIFFUSION (last step saved):
┌─────────────────────────────────────────────────────────────┐
│ KV Cache (context + generated video latent)               │
├─────────────────────────────────────────────────────────────┤
│ Slots:    [0]  [1]  [2]  [3]  [4]  [5]  [ ]  [ ]          │
│ mask:     [X]  [X]  [X]  [X]  [ ]  [ ]  [ ]  [ ]          │ (video context + pred)
│ id:       [0]  [1]  [2]  [2]  [-1] [-1] [-1] [-1]          │
│ is_pred:  [ ]  [ ]  [X]  [X]  [ ]  [ ]  [ ]  [ ]          │ (slots 2-3 are predictions)
└─────────────────────────────────────────────────────────────┘

ACTION DIFFUSION (similar pattern):
┌─────────────────────────────────────────────────────────────┐
│ KV Cache (after action diffusion)                          │
├─────────────────────────────────────────────────────────────┤
│ Slots:    [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]          │
│ mask:     [X]  [X]  [X]  [X]  [X]  [X]  [ ]  [ ]          │ (more populated)
│ id:       [0]  [1]  [2]  [2]  [3]  [3]  [-1] [-1]          │
│ is_pred:  [ ]  [ ]  [X]  [X]  [X]  [X]  [ ]  [ ]          │ (slots 2-5 are predictions)
└─────────────────────────────────────────────────────────────┘

NEXT EPISODE STEP (action computed, clear pred cache):
┌─────────────────────────────────────────────────────────────┐
│ KV Cache (after clear_pred_cache('pos'))                  │
├─────────────────────────────────────────────────────────────┤
│ Slots:    [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]          │
│ mask:     [X]  [X]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]          │ (predictions cleared)
│ id:       [0]  [1]  [-1] [-1] [-1] [-1] [-1] [-1]          │
│ is_pred:  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]          │ (is_pred flags reset)
└─────────────────────────────────────────────────────────────┘
  Context from observation 0 remains, ready for next step!

NEXT STEP (new observation encoding, update_cache=2):
┌─────────────────────────────────────────────────────────────┐
│ KV Cache (context accumulated)                             │
├─────────────────────────────────────────────────────────────┤
│ Slots:    [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]          │
│ mask:     [X]  [X]  [X]  [X]  [ ]  [ ]  [ ]  [ ]          │ (new frames added)
│ id:       [0]  [1]  [2]  [3]  [-1] [-1] [-1] [-1]          │
│ is_pred:  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]          │ (context only)
└─────────────────────────────────────────────────────────────┘
  frame_st_id: 2 → 6 (cumulative history grows)

... (continue until cache fills up) ...

CACHE FULL - LRU EVICTION:
┌─────────────────────────────────────────────────────────────┐
│ When new update needs slots but all are occupied:          │
│                                                              │
│ Before:                                                     │
│   mask=[T,T,T,T,T,T,T,T]  id=[0,1,2,3,4,5,6,7]           │
│   Need 2 new slots                                          │
│                                                              │
│ LRU eviction (keep newest, evict oldest):                  │
│   Sort by id: [0,1,2,3,4,5,6,7]                           │
│   Evict id[0] and id[1] (oldest)                          │
│   → Free slots 0 and 1                                     │
│                                                              │
│ After:                                                      │
│   mask=[F,F,T,T,T,T,T,T]  id=[-1,-1,2,3,4,5,6,7]         │
│   Insert new: mask=[T,T,T,T,T,T,T,T]  id=[8,9,2,3,4,5,6,7]│
│                                                              │
│ Effect: Sliding window slides forward (oldest history dropped)│
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Client-Server Message Protocol

```
MESSAGE FORMAT: Binary msgpack (numpy-compatible)

INITIAL HANDSHAKE:
┌──────────────────────────────────────┐
│ Client connects to ws://0.0.0.0:port │
└────────────────┬─────────────────────┘
                 │
                 ├─ SERVER: send(msgpack(metadata))
                 │          metadata = {
                 │              "version": "1.0",
                 │              "model": "wan_va",
                 │              ...
                 │          }
                 │
                 └─ CLIENT: recv() unpacks metadata

SESSION RESET:
┌──────────────────────────────────────────────┐
│ CLIENT sends:                               │
│ {                                           │
│   "reset": True,                            │
│   "prompt": "move the cup to the table"    │
│ }                                           │
└────────────┬───────────────────────────────┘
             │
             ├─ SERVER: _reset(prompt)
             │  - Clear KV cache
             │  - Encode prompt via T5
             │  - Create empty KV cache
             │
             └─ SERVER responds: pack({})
                Empty response signals reset complete

INFERENCE REQUEST:
┌──────────────────────────────────────────────────────────────┐
│ CLIENT sends:                                                │
│ {                                                            │
│   "obs": {                                                  │
│       "obs": [frame0, frame1, ...],  # RGB images [H, W, 3]│
│       "state": [joint_pos]  # Proprioceptive state        │
│   },                                                         │
│   "prompt": "move the cup to the table",                    │
│   "video_guidance_scale": 5.0,   # CFG scale              │
│   "action_guidance_scale": 1.0   # Action CFG              │
│ }                                                            │
└────────────┬──────────────────────────────────────────────┘
             │
             ├─ SERVER: _infer(obs, frame_st_id)
             │  1. Encode obs → latents (via VAE)
             │  2. Run video diffusion with KV cache
             │  3. Run action diffusion with KV cache
             │  4. Postprocess action
             │  5. Record timing
             │
             └─ SERVER responds:
                {
                  "action": [...],  # [16,] numpy array
                  "server_timing": {
                    "infer_ms": 2150.5,
                    "prev_total_ms": 2155.2
                  }
                }

ASYNC INFERENCE (special modes):
┌──────────────────────────────────────────────────────────────┐
│ CLIENT sends (optional):                                     │
│ {                                                            │
│   "compute_kv_cache": True,  # Pre-compute KV cache        │
│   "obs": {...},              # Observation data            │
│   "prompt": "..."                                          │
│ }                                                            │
│                                                              │
│ SERVER: _compute_kv_cache(obs)                             │
│   → transformer(..., update_cache=2)                       │
│   → Accumulates KV cache without inference                 │
│   → Returns {}                                              │
│                                                              │
│ BENEFIT: Pre-load context for next inference               │
│   → Reduces next step latency                              │
└──────────────────────────────────────────────────────────────┘

ERROR HANDLING:
┌──────────────────────────────────────────┐
│ If exception during inference:           │
│                                          │
│ SERVER: send(traceback.format_exc())    │
│   (sent as string, not packed binary)    │
│                                          │
│ CLIENT: Detects isinstance(response, str)│
│   → raise RuntimeError(response)         │
│   → Prints full server traceback        │
└──────────────────────────────────────────┘
```

---

## 6. Port Assignment Patterns

```
SINGLE SERVER (1 GPU):
┌─────────────────────────────────────────┐
│ launch_server.sh                        │
│  START_PORT=29056                       │
│  MASTER_PORT=29061                      │
│                                         │
│  Server:  ws://0.0.0.0:29056           │
│  Dist:    master_port=29061             │
└─────────────────────────────────────────┘

MULTI-SERVER (8 GPUs):
┌──────────────────────────────────────────────────────────────┐
│ launch_server_multigpus.sh                                  │
│  START_PORT=29556                                           │
│  MASTER_PORT=29661                                          │
│                                                              │
│  GPU 0: ws://0.0.0.0:29556  (dist:29661)                   │
│  GPU 1: ws://0.0.0.0:29557  (dist:29662)                   │
│  GPU 2: ws://0.0.0.0:29558  (dist:29663)                   │
│  GPU 3: ws://0.0.0.0:29559  (dist:29664)                   │
│  GPU 4: ws://0.0.0.0:29560  (dist:29665)                   │
│  GPU 5: ws://0.0.0.0:29561  (dist:29666)                   │
│  GPU 6: ws://0.0.0.0:29562  (dist:29667)                   │
│  GPU 7: ws://0.0.0.0:29563  (dist:29668)                   │
└──────────────────────────────────────────────────────────────┘

MULTI-CLIENT ASSIGNMENT:
┌──────────────────────────────────────────────────────────────┐
│ balance_tasks.py assigns tasks to ports:                    │
│                                                              │
│  Task A (step_lim: 200) → Client 0 → Port 29556           │
│  Task B (step_lim: 150) → Client 1 → Port 29557           │
│  Task C (step_lim: 250) → Client 2 → Port 29558           │
│  ...                                                         │
│                                                              │
│  PORT ASSIGNMENT: client_id = task_id → port = START_PORT + client_id
│                                                              │
│  This ensures load distribution across servers:             │
│  - Each client connects to different server (port)          │
│  - Each server handles one client at a time                │
│  - Balanced workload via task grouping                      │
└──────────────────────────────────────────────────────────────┘
```

---

## 7. Distributed PyTorch Coordination

```
SINGLE GPU (nproc_per_node=1):
┌──────────────────────────────────────────┐
│ pytorch.distributed.run                 │
│   --nproc_per_node 1                    │
│   --master_port 29561                   │
│                                          │
│ Result: Only Rank 0 (= Local Rank 0)    │
│   - WebSocket listening                 │
│   - Processes all requests               │
└──────────────────────────────────────────┘

MULTI-GPU (nproc_per_node > 1):
┌────────────────────────────────────────────────────────────┐
│ pytorch.distributed.run                                   │
│   --nproc_per_node 4                                      │
│   --master_port 29661                                     │
│                                                            │
│ Result: Rank 0 (= Local Rank 0)                           │
│   ├─ Host WebSocket                                       │
│   ├─ broadcast(cmd, src=0)                               │
│   │  (signal inference to other ranks)                   │
│   └─ Process client requests                             │
│                                                            │
│ Result: Rank 1, 2, 3 (= Local Rank 1, 2, 3)              │
│   ├─ worker_loop()                                        │
│   ├─ recv broadcast(cmd)                                 │
│   ├─ If cmd==1: recv broadcast_object_list(obs)         │
│   │  → model.infer(obs)  [pre-compute, future opt]       │
│   └─ If cmd==-1: break (exit)                            │
│                                                            │
│ Communication Pattern:                                    │
│  Iteration 1:                                             │
│    Rank 0: broadcast(cmd=1, obs)                         │
│    Rank 1-3: recv & execute (optional)                   │
│                                                            │
│  Iteration 2:                                             │
│    Rank 0: broadcast(cmd=1, obs)  [new request]          │
│    Rank 1-3: recv & execute (optional)                   │
│                                                            │
│  Shutdown:                                                │
│    Rank 0: broadcast(cmd=-1)                             │
│    Rank 1-3: recv & exit                                 │
└────────────────────────────────────────────────────────────┘

FUTURE OPTIMIZATION (Not yet implemented):
┌──────────────────────────────────────────────────────────────┐
│ Could use Rank 1+ for:                                      │
│  - Pre-loading KV cache while Rank 0 serves                │
│  - Parallel observation encoding                           │
│  - Pipelining multiple requests                            │
│                                                              │
│ Current: Rank 1+ are idle (worker_loop does nothing)      │
└──────────────────────────────────────────────────────────────┘
```

---

## 8. Complete Request-Response Timeline

```
TIME    CLIENT                          SERVER
────────────────────────────────────────────────────────────────
t=0     CONNECT ws://0.0.0.0:29556      ACCEPT
        ────────────────────────────────→ async _handler()
                                         log: "Connection opened"

t=1                                      SEND: pack(metadata)
        ←────────────────────────────────
        RECV: metadata
        store server_metadata

t=2     SEND: {reset=True,
               prompt="move cup"}
        ────────────────────────────────→ RECV: msgpack
                                         UNPACK: obs
                                         CALL: _reset(prompt)
                                         ├─ clear_cache()
                                         ├─ clear_cache() VAE
                                         ├─ create_empty_cache()
                                         └─ encode_prompt()

t=2.5                                    SEND: pack({})
        ←────────────────────────────────
        RECV: {} (reset ACK)

t=3     (episodes begin, loop starts)

t=3+0   SEND: {obs: {...},
               prompt: "move cup"}
        ────────────────────────────────→ START_TIME
                                         RECV: msgpack
                                         UNPACK: obs
                                         
t=3+50  (inference running on server)    INFER_START: _infer()
                                         ├─ _encode_obs(): 150ms
                                         ├─ Video loop: 1000ms
                                         ├─ Action loop: 200ms
                                         ├─ Postprocess: 50ms
                                         └─ INFER_END
                                         
t=3+1410                                 PACK: action
                                         ADD: server_timing
                                         SEND: msgpack
        ←────────────────────────────────
        RECV: action, timing
        
t=3+1420 PROCESS ACTION
        (Step in environment: 50ms)
        
t=3+1470 SEND: {obs: {...}}
        ────────────────────────────────→ (loop repeats)
        
        ...

t=120000 EPISODE COMPLETE
        SEND: {reset=True, prompt="..."}
        ────────────────────────────────→ (session reset)
        
        ... (more episodes) ...
        
t=300000 DISCONNECT
        ────────────────────────────────→ websockets.ConnectionClosed
                                         log: "Connection closed"
                                         → Ready for next client
```

---

## 9. Error Scenarios

```
SCENARIO 1: Server Not Ready
┌──────────────────────────────────────────┐
│ Client.infer() → WebsocketClientPolicy   │
│                                          │
│ connect() → ConnectionRefusedError       │
│   → logging.info("Still waiting...")    │
│   → sleep(5)                             │
│   → Retry (infinite loop)                │
│                                          │
│ When server starts:                      │
│   → Client succeeds on next retry        │
└──────────────────────────────────────────┘

SCENARIO 2: Inference Timeout (Long Diffusion)
┌──────────────────────────────────────────┐
│ Problem: ping_interval=None               │
│   → No automatic disconnect on timeout   │
│   → Server takes 2-3 seconds to infer    │
│   → Client waits indefinitely            │
│                                          │
│ Feature:                                 │
│   → Works correctly (slow but not error) │
│   → By design for long-running tasks    │
└──────────────────────────────────────────┘

SCENARIO 3: Inference Exception
┌──────────────────────────────────────────┐
│ Server exception during _infer()         │
│   → except Exception caught              │
│   → SEND: traceback.format_exc() [string]│
│   → CLOSE: code=INTERNAL_ERROR           │
│                                          │
│ Client receives:                         │
│   → response = recv()                    │
│   → isinstance(response, str) → True     │
│   → raise RuntimeError(response)         │
│   → Full traceback available to user    │
└──────────────────────────────────────────┘

SCENARIO 4: Client Crashes
┌──────────────────────────────────────────┐
│ Client dies mid-inference                │
│   → Connection drops                     │
│   → Server: websockets.ConnectionClosed  │
│   → except → logger.info("Connection...")│
│   → Clean exit from _handler()           │
│   → KV cache remains in memory           │
│   → Port freed for next connection       │
└──────────────────────────────────────────┘

SCENARIO 5: Simultaneous Connections
┌──────────────────────────────────────────┐
│ Multiple clients connect to same port:   │
│   ERROR: Only one listening socket       │
│                                          │
│ Solution (actual deployment):            │
│   → Use separate server per GPU          │
│   → Each server: unique port             │
│   → Load balance clients across servers  │
│                                          │
│ NOT a true multi-session server          │
└──────────────────────────────────────────┘
```

