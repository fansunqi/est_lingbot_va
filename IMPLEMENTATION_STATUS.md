# Session-Level Balanced Evaluation Pipeline - Implementation Status

## Overview
This document summarizes the implementation status of the session-level balanced evaluation pipeline for RoboTwin evaluation.

## Changes Made

### 1. balance_tasks.py - Session-Level Task Balancing
**Location:** `evaluation/robotwin/balance_tasks.py`

**Changes:**
- Modified `balance_sessions()` function to include episode indexing
- Added `episode_idx` tracking for each seed within a task
- Episode index is now part of the work unit tuple: `(step_lim, task_name, seed, episode_idx)`
- Each client assignment now includes `{"task": task_name, "seed": seed, "episode_idx": episode_idx}`

**Impact:**
- Enables proper per-task per-episode result tracking
- Accurate metrics reporting at individual episode level
- Load-balanced distribution across clients (using LPT algorithm)

### 2. eval_session_client.py - Session-Based Client Architecture
**Location:** `evaluation/robotwin/eval_session_client.py`

**Key Additions:**
- New command-line arguments:
  - `--seed`: Seed multiplier for stseed folder naming
  - `--client_id`: Client ID for per-client metrics file naming
  
- Directory Structure:
  - Results organized under `save_root/stseed-{st_seed}/`
  - Per-client metrics: `stseed-{st_seed}/metrics/client_{id}.json`
  - Per-task results: `stseed-{st_seed}/metrics/{task_name}/res.json`
  - Overall summary: `stseed-{st_seed}/metrics/overall.json`

- Episode Tracking:
  - Reads `episode_idx` from assignment JSON
  - Tracks results per episode: `{"seed": seed, "episode_idx": episode_idx, "success": succ}`
  - Saves intermediate metrics after each episode
  
- New `merge_metrics()` Function:
  - Aggregates per-client metrics into final results
  - Called via: `python -m evaluation.robotwin.eval_session_client merge --metrics_dir {metrics_dir}`
  - Outputs:
    - Per-task results: `metrics/{task_name}/res.json`
    - Overall summary: `metrics/overall.json`
  - Merges episode histories from all clients

### 3. launch_session_eval.sh - Updated Launch Script
**Location:** `evaluation/robotwin/launch_session_eval.sh`

**Changes:**
- Updated directory structure to use `stseed-{st_seed}` hierarchy
- Seeds file: `${RUN_DIR}/valid_seeds.json`
- Assignments directory: `${RUN_DIR}/task_assignments`
- Pass new arguments to eval clients:
  - `--seed ${seed}`
  - `--client_id ${i}`
  
- Added post-launch instructions for metrics merging

**Launch Command Example:**
```bash
NUM_GROUPS=7 CLIENT_GPUS="5 6 7 5 6 7 5 6 7" \
  bash evaluation/robotwin/launch_session_eval.sh ./results 100 0 9 2
```

## Three-Phase Pipeline

### Phase 1: Collect Valid Seeds (Parallel)
- Runs `collect_seeds` module with multiple workers
- Each worker processes subset of tasks in parallel
- Output: `valid_seeds.json` with pre-validated seeds

### Phase 2: Balance Task Assignments
- Reads `valid_seeds.json`
- Uses LPT algorithm to balance workload across clients
- Generates per-client assignment files
- Output: `task_assignments/client_0.json` ... `client_{n}.json`

### Phase 3: Launch Evaluation Clients
- Launches N clients with load-balanced assignments
- GPU round-robin assignment
- Port assignment: base_port + client_index
- Each client runs assigned episodes sequentially

## Result Organization

```
./results/
├── stseed-10000/
│   ├── valid_seeds.json
│   ├── task_assignments/
│   │   ├── client_0.json
│   │   ├── client_1.json
│   │   └── ...
│   ├── metrics/
│   │   ├── client_0.json          # Per-client intermediate results
│   │   ├── client_1.json
│   │   ├── lift_pot/
│   │   │   └── res.json           # Final per-task results
│   │   ├── fold_towel/
│   │   │   └── res.json
│   │   ├── overall.json           # Final summary
│   │   └── ...
│   └── visualization/
│       ├── lift_pot/
│       ├── fold_towel/
│       └── ...
```

## Metrics Output Format

### Per-Client Metrics (client_{id}.json)
```json
{
  "lift_pot": {
    "succ_num": 8,
    "total_num": 10,
    "succ_rate": 0.8,
    "episodes": [
      {"seed": 10000, "episode_idx": 0, "success": true},
      {"seed": 10001, "episode_idx": 1, "success": false},
      ...
    ]
  },
  ...
}
```

### Per-Task Final Results ({task_name}/res.json)
```json
{
  "succ_num": 24.0,
  "total_num": 30.0,
  "succ_rate": 0.8,
  "episodes": [
    {"seed": 10000, "episode_idx": 0, "success": true},
    ...
  ]
}
```

### Overall Summary (overall.json)
```json
{
  "total_succ": 240.0,
  "total_episodes": 300.0,
  "overall_succ_rate": 0.8
}
```

## Usage Examples

### Full Pipeline (All 50 Tasks, 9 Clients)
```bash
CLIENT_GPUS="5 6 7 5 6 7 5 6 7" \
  bash evaluation/robotwin/launch_session_eval.sh ./results 100 0 9
```

### Specific Task Group (Group 2 out of 7)
```bash
NUM_GROUPS=7 CLIENT_GPUS="5 6 7 5 6 7 5 6 7" \
  bash evaluation/robotwin/launch_session_eval.sh ./results 100 0 9 2
```

### Merge Results After Clients Finish
```bash
python -m evaluation.robotwin.eval_session_client merge \
  --metrics_dir ./results/stseed-10000/metrics
```

## Implementation Benefits

1. **Accurate Metrics**: Per-episode result tracking with seed and episode_idx
2. **Load Balancing**: LPT algorithm distributes work based on task complexity
3. **Scalability**: Supports multiple concurrent clients with proper result aggregation
4. **Transparency**: Full episode history available for detailed analysis
5. **Organized Output**: Clear hierarchical directory structure
6. **Monitoring**: Real-time per-client metrics updates during evaluation

## Testing Recommendations

1. Verify Phase 1 seed collection completes successfully
2. Check Phase 2 generates balanced assignments
3. Monitor Phase 3 clients for proper connection and execution
4. Validate metrics merging produces correct aggregations
5. Compare single-client vs multi-client results for consistency

## Next Steps (Optional)

1. Add distributed training support (utilize worker_loop ranks)
2. Implement dynamic client scaling based on available GPUs
3. Add real-time metrics streaming to central dashboard
4. Implement checkpointing for long evaluation runs
5. Add per-client resource monitoring (GPU/CPU/memory)
