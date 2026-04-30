# Quick Start Guide - Session-Level Balanced Evaluation

## Prerequisites
- Server instances running on designated GPUs and ports
- Valid task configuration files
- Environment setup with all dependencies

## Step 1: Start the Server(s)

### Single GPU Server
```bash
bash evaluation/robotwin/launch_server.sh
# Server starts on port 29056
```

### Multi-GPU Servers (8 GPUs)
```bash
bash evaluation/robotwin/launch_server_multigpus.sh
# Servers start on ports 29556-29563 (one per GPU)
```

## Step 2: Launch Evaluation Pipeline

### Option A: Full Pipeline (All Tasks)
```bash
# Using GPUs 5, 6, 7 (round-robin for 9 clients)
CLIENT_GPUS="5 6 7 5 6 7 5 6 7" \
  bash evaluation/robotwin/launch_session_eval.sh ./results 100 0 9
```

**Parameters:**
- `save_root`: `./results` - output directory
- `test_num`: `100` - per-task samples
- `seed`: `0` - seed multiplier
- `num_clients`: `9` - number of parallel clients
- `group_id`: (omitted) - evaluates all tasks

### Option B: Specific Task Group
```bash
# Evaluate only group 2 (out of 7 groups)
NUM_GROUPS=7 CLIENT_GPUS="5 6 7 5 6 7 5 6 7" \
  bash evaluation/robotwin/launch_session_eval.sh ./results 100 0 9 2
```

### Option C: Reuse Existing Seeds (Skip Collection)
```bash
# Skip Phase 1 if valid_seeds.json already exists
SKIP_COLLECT=1 CLIENT_GPUS="5 6 7 5 6 7 5 6 7" \
  bash evaluation/robotwin/launch_session_eval.sh ./results 100 0 1 5
```

## Step 3: Monitor Progress

### Watch Client Logs
```bash
# Where YYYYMMDD_HHMMSS is the batch timestamp
tail -f logs/session_client_*_YYYYMMDD_HHMMSS.log
```

### Check Intermediate Results
```bash
# Results update as each client finishes episodes
cat ./results/stseed-10000/metrics/client_0.json
```

### Kill All Clients (if needed)
```bash
kill $(cat pids_session.txt)
```

## Step 4: Merge and View Final Results

After all clients finish, merge metrics:

```bash
python -m evaluation.robotwin.eval_session_client merge \
  --metrics_dir ./results/stseed-10000/metrics
```

### View Final Results
```bash
# Per-task results
cat ./results/stseed-10000/metrics/lift_pot/res.json
cat ./results/stseed-10000/metrics/fold_towel/res.json

# Overall summary
cat ./results/stseed-10000/metrics/overall.json

# Visualization videos
ls ./results/stseed-10000/visualization/*/
```

## Result Files Structure

```
./results/stseed-10000/
├── valid_seeds.json                  # Pre-validated seeds per task
├── task_assignments/
│   ├── client_0.json                # ~50 episodes per client
│   ├── client_1.json
│   └── ...
├── metrics/
│   ├── client_0.json                # Per-client intermediate results
│   ├── client_1.json
│   ├── lift_pot/res.json            # Final per-task results
│   ├── fold_towel/res.json
│   ├── overall.json                 # Final summary
│   └── ...
└── visualization/
    ├── lift_pot/
    │   ├── 0_grasp_pot.mp4
    │   └── ...
    ├── fold_towel/
    └── ...
```

## Understanding Results Format

### Per-Client Metrics (client_0.json)
Shows results from each client as they run:
- `succ_num`: Number of successful episodes
- `total_num`: Total episodes executed
- `succ_rate`: Success rate (0.0-1.0)
- `episodes`: Array of individual episode results

### Final Per-Task Results (lift_pot/res.json)
Aggregated results across all clients for one task:
- `succ_num`: Total successes across all clients
- `total_num`: Total episodes attempted
- `succ_rate`: Final success rate
- `episodes`: Complete episode history

### Overall Summary (overall.json)
Global summary across all tasks and clients:
- `total_succ`: Total successes across all tasks
- `total_episodes`: Total episodes run
- `overall_succ_rate`: Final success rate

## Common Issues

### Server Connection Failed
- Check if servers are running: `netstat -an | grep 29556`
- Verify ports are not blocked by firewall
- Check server logs for errors

### Unbalanced Task Distribution
- Verify `valid_seeds.json` has sufficient seeds per task
- Check `task_assignments/` for balanced distribution
- Review load calculation in `balance_tasks.py`

### Missing Metrics
- Ensure all clients finished successfully
- Check client logs for exceptions
- Verify metrics directory permissions

### GPU Out of Memory
- Reduce `test_num` (samples per task)
- Reduce number of concurrent clients
- Use different GPU assignments

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `CLIENT_GPUS` | "5 6 7 5 6 7 5 6 7" | GPU assignment for clients |
| `START_PORT` | 29556 | Base port for client servers |
| `NUM_GROUPS` | 7 | Number of task groups |
| `SKIP_COLLECT` | 0 | Skip seed collection (1 = skip) |
| `COLLECT_WORKERS` | NUM_CLIENTS | Workers for seed collection |
| `TASK_CONFIG` | demo_clean | Task configuration name |

## Example: Production Run

```bash
#!/bin/bash
# Full evaluation with 18 clients on 8 GPUs, groups of 4 tasks each

export NUM_GROUPS=7
export CLIENT_GPUS="0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1"
export START_PORT=30000

# Run all task groups sequentially
for group_id in {0..6}; do
    echo "========== Group $group_id =========="
    bash evaluation/robotwin/launch_session_eval.sh \
        ./results_production \
        100 \
        0 \
        18 \
        $group_id
    
    # Wait for completion (check pids_session.txt)
    wait
    
    # Optional: merge intermediate results
    python -m evaluation.robotwin.eval_session_client merge \
        --metrics_dir ./results_production/stseed-10000/metrics
done

echo "========== Final Results =========="
cat ./results_production/stseed-10000/metrics/overall.json
```

## Next Steps

1. **Analyze Results**: Use provided notebooks to analyze per-task performance
2. **Debug Failures**: Review `visualization/` videos and error logs
3. **Compare Versions**: Run with different model checkpoints for comparison
4. **Optimize**: Adjust `NUM_GROUPS`, `CLIENT_GPUS` based on performance
