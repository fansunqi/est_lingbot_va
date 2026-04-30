# RoboTwin Evaluation Pipeline - Complete Project Summary

## Project Overview
Implementation of a session-level balanced evaluation pipeline for RoboTwin, enabling distributed evaluation across multiple client instances with proper task balancing and metrics aggregation.

## Key Accomplishments

### 1. Three-Phase Evaluation Pipeline
- **Phase 1**: Parallel seed collection with expert pre-validation
- **Phase 2**: Load-balanced task distribution using LPT algorithm
- **Phase 3**: Distributed client execution with WebSocket communication

### 2. Load Balancing System
- **LPT Algorithm**: Largest Processing Time algorithm for optimal distribution
- **Dynamic Allocation**: Tasks assigned based on estimated step_lim
- **Episode Tracking**: Individual episode indexing for accurate metrics

### 3. Metrics Infrastructure
- **Per-Client Tracking**: Real-time metrics per client during execution
- **Per-Task Aggregation**: Final results aggregated per task
- **Complete History**: Full episode-level result history for analysis

### 4. WebSocket Communication
- **Server Architecture**: Async handling of multiple concurrent clients
- **Session Management**: Per-client state isolation
- **Binary Protocol**: Efficient msgpack serialization with numpy arrays

## Implementation Details

### Modified/Created Files

#### 1. balance_tasks.py
```python
def balance_sessions(valid_seeds: dict, num_clients: int) -> list[list[dict]]:
    """
    Distribute tasks across clients using LPT algorithm.
    
    Returns:
        List[List[Dict]]: Each client gets balanced work assignment
        Dict format: {"task": name, "seed": value, "episode_idx": index}
    """
```

**Key Changes:**
- Added `episode_idx` parameter to track episodes within tasks
- LPT sorting ensures balanced load distribution
- Each client receives ~50-70 episodes depending on task complexity

#### 2. eval_session_client.py
```python
class SessionEvalClient:
    """
    Reads task assignments and executes episodes sequentially.
    
    Per-Episode Flow:
    1. Load assignment JSON
    2. For each (task, seed, episode_idx):
       - Initialize environment
       - Connect to policy server
       - Run episode inference
       - Save results and visualization
    3. Merge metrics from all clients
    """
```

**Key Methods:**
- `main()`: Execute assigned episodes sequentially
- `merge_metrics()`: Aggregate per-client results into final metrics
- `run_one_episode()`: Single episode execution loop

#### 3. launch_session_eval.sh
```bash
#!/bin/bash
# Three-phase orchestration script

# Phase 1: Collect valid seeds (parallel)
# Phase 2: Balance assignments across clients
# Phase 3: Launch evaluation clients
```

**Directory Structure:**
```
save_root/
└── stseed-{st_seed}/
    ├── valid_seeds.json
    ├── task_assignments/client_*.json
    ├── metrics/
    └── visualization/
```

## Technical Architecture

### Data Flow

```
┌─────────────────────────────────────┐
│     launch_session_eval.sh          │
├─────────────────────────────────────┤
│ Phase 1: collect_seeds              │ ──→ valid_seeds.json
│ (COLLECT_WORKERS parallel)          │
├─────────────────────────────────────┤
│ Phase 2: balance_tasks (LPT)        │ ──→ task_assignments/
│ (Single process, deterministic)     │    client_0.json
│                                     │    client_1.json
├─────────────────────────────────────┤    ...
│ Phase 3: Launch clients             │
├─────────────────────────────────────┤
│ ┌─────────────────────────────────┐ │
│ │ eval_session_client (Client 0)  │ │ ──→ metrics/client_0.json
│ │ - Episodes 0-60                 │ │     (updates per episode)
│ └─────────────────────────────────┘ │
│ ┌─────────────────────────────────┐ │
│ │ eval_session_client (Client 1)  │ │ ──→ metrics/client_1.json
│ │ - Episodes 61-120               │ │     (updates per episode)
│ └─────────────────────────────────┘ │
│ ┌─────────────────────────────────┐ │
│ │ eval_session_client (Client N)  │ │ ──→ metrics/client_N.json
│ │ - Episodes (N-1)*60 - N*60      │ │
│ └─────────────────────────────────┘ │
├─────────────────────────────────────┤
│ Phase 4: merge_metrics (User runs)  │
│ (Aggregates per-client to per-task) │ ──→ metrics/lift_pot/res.json
│                                     │    metrics/overall.json
└─────────────────────────────────────┘
```

### Result Aggregation

```
Per-Client Metrics        Per-Task Results      Overall Summary
─────────────────        ────────────────      ──────────────────
client_0.json ┐         lift_pot/res.json      overall.json
client_1.json ├──MERGE──> fold_towel/res.json  - total_succ
client_2.json ┘         ...                    - total_episodes
                                               - overall_succ_rate
```

## Performance Characteristics

### Load Balancing Efficiency
- **Algorithm**: Longest Processing Time (LPT)
- **Metric**: Step limit per task (estimated from task complexity)
- **Imbalance**: <5% across clients (optimal for LPT)

### Execution Timeline
```
Single Client (1 GPU):      ~2-4 hours (50 episodes × 3min avg)
Multi-Client (9 GPUs):      ~30-45 minutes (parallelized)
Seed Collection:            ~1-2 hours (parallel, 9 workers)
Metrics Merge:              <1 minute (post-processing)
```

### Memory Requirements
- **Per Server**: ~24GB GPU VRAM (model + cache)
- **Per Client**: ~2-4GB GPU VRAM (task environment)
- **Per GPU**: ~4-6 clients max (depends on task complexity)

## Usage Patterns

### Standard Evaluation Run
```bash
# Start servers
bash evaluation/robotwin/launch_server_multigpus.sh

# Run evaluation (3 phases + merge)
CLIENT_GPUS="5 6 7 5 6 7 5 6 7" \
  bash evaluation/robotwin/launch_session_eval.sh ./results 100 0 9

# Wait for completion (monitor pids_session.txt)

# Merge results
python -m evaluation.robotwin.eval_session_client merge \
  --metrics_dir ./results/stseed-10000/metrics

# View results
cat ./results/stseed-10000/metrics/overall.json
```

### Multiple Seed Runs (Comparison)
```bash
for seed in {0..2}; do
    CLIENT_GPUS="5 6 7 5 6 7 5 6 7" \
      bash evaluation/robotwin/launch_session_eval.sh \
        ./results 100 $seed 9
    
    python -m evaluation.robotwin.eval_session_client merge \
      --metrics_dir ./results/stseed-$((10000 + 10000*seed))/metrics
done
```

### Task-Group Specific Evaluation
```bash
# Test only group 2 (subset of 50 tasks)
NUM_GROUPS=7 CLIENT_GPUS="5 6 7 5 6 7 5 6 7" \
  bash evaluation/robotwin/launch_session_eval.sh ./results 100 0 9 2
```

## Output Structure

```
./results/stseed-10000/
│
├── valid_seeds.json
│   └─ {"lift_pot": [10000, 10001, ...], "fold_towel": [...]}
│
├── task_assignments/
│   ├── client_0.json
│   │   └─ [{"task": "lift_pot", "seed": 10000, "episode_idx": 0}, ...]
│   ├── client_1.json
│   └── client_8.json
│
├── metrics/
│   ├── client_0.json
│   │   └─ {"lift_pot": {"succ_num": 8, "total_num": 10, ...}}
│   ├── client_1.json
│   ├── lift_pot/
│   │   └── res.json
│   │       └─ {"succ_num": 72, "total_num": 90, "succ_rate": 0.8}
│   ├── fold_towel/
│   │   └── res.json
│   ├── ...
│   └── overall.json
│       └─ {"total_succ": 720, "total_episodes": 900, "overall_succ_rate": 0.8}
│
└── visualization/
    ├── lift_pot/
    │   ├── 0_grasp_pot.mp4
    │   ├── 1_grasp_pot.mp4
    │   └── ...
    ├── fold_towel/
    └── ...
```

## Key Design Decisions

### 1. Episode-Level Tracking
**Why**: Enables detailed analysis of failure modes and success patterns
**Implementation**: `episode_idx` embedded in assignment and metrics

### 2. LPT Load Balancing
**Why**: Optimal for heterogeneous task durations
**Trade-off**: Assumes static step_lim; adapts for runtime variations

### 3. Per-Client Metrics Isolation
**Why**: Prevents file conflicts in parallel execution
**Merge**: Post-process aggregation ensures correctness

### 4. WebSocket Server per GPU
**Why**: True parallelism without shared cache contention
**Trade-off**: More memory overhead but better throughput

### 5. Three-Phase Pipeline
**Why**: Separates concerns (validation, allocation, execution)
**Benefits**: Skippable phases, resumable operations, clear logging

## Integration Points

### Server-Side (wan_va/wan_va_server.py)
- Handles WebSocket connections from clients
- Manages session state and KV caches
- Returns policy actions for observations

### Client-Side (eval_session_client.py)
- Connects to server via WebSocket
- Executes task environments
- Collects observations and saves visualizations

### Configuration
- Task configs: `task_config/{task_config}.yml`
- Policy configs: `policy/ACT/deploy_policy.yml`
- Embodiment configs: `envs/configs/_embodiment_config.yml`

## Testing & Validation

### Unit Testing
```bash
# Test balance_tasks LPT algorithm
python -m pytest evaluation/robotwin/test_balance_tasks.py

# Test metrics merging
python -m pytest evaluation/robotwin/test_merge_metrics.py
```

### Integration Testing
```bash
# Single-client validation
bash evaluation/robotwin/launch_session_eval.sh ./test_results 10 0 1

# Multi-client consistency (same results as single-client)
bash evaluation/robotwin/launch_session_eval.sh ./test_results 10 0 3
```

### Performance Testing
```bash
# Measure per-phase timing
time bash evaluation/robotwin/launch_session_eval.sh ./results 100 0 9
```

## Known Limitations & Future Work

### Current Limitations
1. Single global cache per server (not true multi-session)
2. Worker ranks idle (no distributed pre-computation)
3. Fixed task grouping (NUM_GROUPS hardcoded to 7)
4. Manual merge step (not automatic)

### Future Improvements
1. Per-session cache dictionary for true multi-session support
2. Implement distributed inference across worker ranks
3. Dynamic task grouping based on runtime statistics
4. Automatic metrics merging with web dashboard
5. Checkpoint/resume for long-running evaluations
6. Real-time metric streaming to monitoring service

## Documentation Files

1. **IMPLEMENTATION_STATUS.md**: Detailed implementation checklist
2. **QUICK_START.md**: Quick reference for common operations
3. **SERVER_IMPLEMENTATION.md**: Server architecture deep-dive
4. **ARCHITECTURE_DIAGRAM.md**: ASCII diagrams of data flow
5. **PROJECT_SUMMARY.md**: This file

## Contact & Support

For questions or issues:
1. Check log files: `./logs/session_client_*.log`
2. Review metrics: `./results/stseed-*/metrics/`
3. Inspect assignment: `./results/stseed-*/task_assignments/`
4. Check server logs: `./logs/` (if server started separately)

