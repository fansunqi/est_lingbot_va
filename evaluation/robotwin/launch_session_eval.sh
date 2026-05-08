#!/bin/bash
# Session-level balanced evaluation pipeline for RoboTwin.
#
# This script orchestrates the full evaluation:
#   Phase 1: Collect valid seeds (expert pre-check)
#   Phase 2: Balance assignment across clients
#   Phase 3: Launch all eval clients
#
# Usage:
#   bash evaluation/robotwin/launch_session_eval.sh [save_root] [test_num] [seed] [num_clients] [group_id]
#
# Example:
#   # Full pipeline - all 50 tasks (9 clients across 3 GPUs):
#   CLIENT_GPUS="5 6 7 5 6 7 5 6 7" \
#     bash evaluation/robotwin/launch_session_eval.sh ./results 100 0 9
#
#   # Only eval tasks in group 2 (use NUM_GROUPS to control how many groups):
#   NUM_GROUPS=7 CLIENT_GPUS="5 6 7 5 6 7 5 6 7" \
#     bash evaluation/robotwin/launch_session_eval.sh ./results 100 0 9 2
#
#   # Skip seed collection (reuse existing valid_seeds.json):
#   SKIP_COLLECT=1 CLIENT_GPUS="5 6 7 5 6 7 5 6 7" \
#     bash evaluation/robotwin/launch_session_eval.sh ./results 100 0 9 2
#
#   # Recollect seeds even if valid_seeds.json already exists:
#   FORCE_COLLECT=1 CLIENT_GPUS="5 6 7 5 6 7 5 6 7" \
#     bash evaluation/robotwin/launch_session_eval.sh ./results 100 0 9 2
#
#   # Disable the built-in progress dashboard and return immediately after launch:
#   MONITOR_PROGRESS=0 bash evaluation/robotwin/launch_session_eval.sh ./results 100 0 9 2

set -e

export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH
unset VK_ICD_FILENAMES
export no_proxy=localhost,127.0.0.1,0.0.0.0
export NO_PROXY=localhost,127.0.0.1,0.0.0.0

# Arguments
save_root=${1:-'./results'}
test_num=${2:-100}
seed=${3:-0}
num_clients=${4:-9}
group_id=${5:-""}  # empty = all tasks; otherwise select a task group
task_config=${TASK_CONFIG:-demo_randomized}
NUM_GROUPS=${NUM_GROUPS:-7}

START_PORT=${START_PORT:-29556}
NUM_SERVERS=${NUM_SERVERS:-5}  # Number of servers running (clients are distributed across them)
CLIENT_GPUS=(${CLIENT_GPUS:-5 6 7 5 6 7 5 6 7})
SKIP_COLLECT=${SKIP_COLLECT:-0}
FORCE_COLLECT=${FORCE_COLLECT:-0}
SEED_VALIDATION_REPLAYS=${SEED_VALIDATION_REPLAYS:-1}
MONITOR_PROGRESS=${MONITOR_PROGRESS:-1}
PROGRESS_INTERVAL=${PROGRESS_INTERVAL:-3}
RESTART_PER_TASK=${RESTART_PER_TASK:-1}

cleanup_eval_clients() {
    local exit_code=${1:-130}
    local pids_file=${pid_file:-pids_session.txt}
    local pid
    trap - INT TERM
    echo ""
    echo -e "\033[33mInterrupted. Terminating eval client process groups from ${pids_file}...\033[0m"
    if [ -f "${pids_file}" ]; then
        while read -r pid; do
            if [ -n "${pid}" ]; then
                kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
            fi
        done < "${pids_file}"
        sleep 2
        while read -r pid; do
            if [ -n "${pid}" ] && kill -0 -- "-${pid}" 2>/dev/null; then
                kill -9 -- "-${pid}" 2>/dev/null || true
            fi
        done < "${pids_file}"
    fi
    exit "${exit_code}"
}

# Determine which tasks to evaluate
if [ -n "${group_id}" ]; then
    # Get task list for this group from balance_tasks (task-level grouping)
    TASKS=$(python -m evaluation.robotwin.balance_tasks --mode task --num_clients ${NUM_GROUPS} --group_id ${group_id})
    if [ $? -ne 0 ]; then
        echo -e "\033[31mError: failed to get tasks for group ${group_id}\033[0m"
        exit 1
    fi
    TASKS_ARG="--tasks \"${TASKS}\""
else
    TASKS=""
    TASKS_ARG=""
fi

# Paths (include seed in filename so different seeds don't conflict)
# Use absolute paths to avoid issues with os.chdir in Python scripts
mkdir -p "${save_root}"
save_root=$(realpath "${save_root}")
run_root="${save_root}/${task_config}"
st_seed=$((10000 * (1 + seed)))
group_label=${group_id:-all}
RUN_NAME="stseed-${st_seed}_group-${group_label}_test-${test_num}"
RUN_DIR="${run_root}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
SEEDS_FILE="${RUN_DIR}/valid_seeds.json"
ASSIGNMENT_DIR="${RUN_DIR}/task_assignments"

policy_name=ACT
log_dir="./logs"
mkdir -p "$log_dir"

echo -e "\033[32m============================================\033[0m"
echo -e "\033[32m  Session-Level Balanced Evaluation\033[0m"
echo -e "\033[32m============================================\033[0m"
echo "  save_root:    ${save_root}"
echo "  run_root:     ${run_root}"
echo "  run_dir:      ${RUN_DIR}"
echo "  test_num:     ${test_num}"
echo "  seed:         ${seed}"
echo "  num_clients:  ${num_clients}"
echo "  num_servers:  ${NUM_SERVERS}"
echo "  task_config:  ${task_config}"
echo "  client_gpus:  ${CLIENT_GPUS[*]}"
echo "  start_port:   ${START_PORT}"
echo "  force_collect:${FORCE_COLLECT}"
echo "  seed_replays: ${SEED_VALIDATION_REPLAYS}"
echo "  monitor:      ${MONITOR_PROGRESS}"
echo "  restart/task: ${RESTART_PER_TASK}"
if [ -n "${group_id}" ]; then
    echo "  group_id:     ${group_id} (of ${NUM_GROUPS} groups)"
    echo "  tasks:        ${TASKS}"
else
    echo "  group_id:     (all tasks)"
fi
echo ""

if [ "${FORCE_COLLECT}" -eq 0 ] && [ "${SKIP_COLLECT}" -eq 0 ] && [ ! -f "${SEEDS_FILE}" ]; then
    REUSABLE_SEEDS_FILE=$(python - "${run_root}" "${st_seed}" "${group_label}" "${test_num}" "${TASKS}" <<'PY'
import json
import re
import sys
from pathlib import Path
from evaluation.robotwin.balance_tasks import ALL_TASKS

run_root = Path(sys.argv[1])
st_seed = sys.argv[2]
group_label = sys.argv[3]
test_num = int(sys.argv[4])
tasks_arg = sys.argv[5]
tasks = tasks_arg.split() if tasks_arg else ALL_TASKS
pattern = re.compile(rf"^stseed-{re.escape(st_seed)}_group-{re.escape(group_label)}_test-(\d+)$")

def usable(path: Path) -> bool:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return False
    for task in tasks:
        entries = data.get(task)
        if not isinstance(entries, list) or len(entries) < test_num:
            return False
        if not all(isinstance(e, dict) and "seed" in e and "episode_info" in e for e in entries[:test_num]):
            return False
    return True

candidates = []
for run_dir in run_root.glob(f"stseed-{st_seed}_group-{group_label}_test-*"):
    match = pattern.match(run_dir.name)
    if match is None:
        continue
    candidate_test_num = int(match.group(1))
    if candidate_test_num < test_num:
        continue
    seeds_file = run_dir / "valid_seeds.json"
    if seeds_file.is_file() and usable(seeds_file):
        candidates.append((candidate_test_num, str(seeds_file)))

if candidates:
    candidates.sort()
    print(candidates[0][1])
PY
)
    if [ -n "${REUSABLE_SEEDS_FILE}" ]; then
        SEEDS_FILE="${REUSABLE_SEEDS_FILE}"
        echo -e "\033[33m[Phase 1] Reusing compatible valid seeds: ${SEEDS_FILE}\033[0m"
        echo ""
    fi
fi

# ============================================================
# Phase 1: Collect valid seeds (parallel across GPUs)
# ============================================================
NEED_COLLECT=0
if [ "${FORCE_COLLECT}" -eq 1 ]; then
    if [ "${SKIP_COLLECT}" -eq 1 ]; then
        echo -e "\033[31mError: FORCE_COLLECT=1 conflicts with SKIP_COLLECT=1\033[0m"
        exit 1
    fi
    echo -e "\033[33m[Phase 1] FORCE_COLLECT=1; recollecting valid seeds.\033[0m"
    rm -f "${SEEDS_FILE}" "${RUN_DIR}"/valid_seeds_worker*.json
    NEED_COLLECT=1
elif [ "${SKIP_COLLECT}" -eq 0 ]; then
    if [ ! -f "${SEEDS_FILE}" ]; then
        NEED_COLLECT=1
    elif ! python - "${SEEDS_FILE}" "${test_num}" "${TASKS}" <<'PY'
import json
import sys
from evaluation.robotwin.balance_tasks import ALL_TASKS

path, test_num, tasks_arg = sys.argv[1], int(sys.argv[2]), sys.argv[3]
data = json.load(open(path))
tasks = tasks_arg.split() if tasks_arg else ALL_TASKS
ok = True
for task in tasks:
    entries = data.get(task)
    if not isinstance(entries, list) or len(entries) < test_num:
        ok = False
        break
    if not all(isinstance(e, dict) and "seed" in e and "episode_info" in e for e in entries[:test_num]):
        ok = False
        break
sys.exit(0 if ok else 1)
PY
    then
        echo -e "\033[33m[Phase 1] Existing ${SEEDS_FILE} is missing enough cached episode_info; collecting.\033[0m"
        NEED_COLLECT=1
    fi
fi

if [ "${NEED_COLLECT}" -eq 1 ]; then
    # Number of parallel workers for seed collection
    COLLECT_WORKERS=${COLLECT_WORKERS:-${#CLIENT_GPUS[@]}}
    COLLECT_GPU_CANDIDATES=(${COLLECT_GPUS:-${CLIENT_GPUS[*]}})
    COLLECT_GPUS=()
    for gpu_id in "${COLLECT_GPU_CANDIDATES[@]}"; do
        already_added=0
        for existing_gpu_id in "${COLLECT_GPUS[@]}"; do
            if [ "${gpu_id}" = "${existing_gpu_id}" ]; then
                already_added=1
                break
            fi
        done
        if [ "${already_added}" -eq 0 ]; then
            COLLECT_GPUS+=("${gpu_id}")
        fi
    done
    if [ "${#COLLECT_GPUS[@]}" -eq 0 ]; then
        echo -e "\033[31mError: no collect GPUs available\033[0m"
        exit 1
    fi

    echo -e "\033[34m[Phase 1] Collecting valid seeds (${COLLECT_WORKERS} workers)...\033[0m"
    echo "  Collect GPUs: ${COLLECT_GPUS[*]}"
    echo "  Output: ${SEEDS_FILE}"
    echo ""

    collect_pids=()
    collect_pid_file="${RUN_DIR}/collect_pids.txt"
    collect_progress_dir="${RUN_DIR}/collect_progress_$(date +%Y%m%d_%H%M%S)"
    : > "${collect_pid_file}"
    mkdir -p "${collect_progress_dir}"

    for w in $(seq 0 $(( COLLECT_WORKERS - 1 ))); do
        gpu_id="${COLLECT_GPUS[$(( w % ${#COLLECT_GPUS[@]} ))]}"
        log_file="${log_dir}/collect_worker_${w}.log"

        collect_cmd="python -m evaluation.robotwin.collect_seeds \
            --test_num ${test_num} \
            --seed ${seed} \
            --task_config ${task_config} \
            --output ${SEEDS_FILE} \
            --max_attempts_ratio 3 \
            --validation_replays ${SEED_VALIDATION_REPLAYS} \
            --progress_dir ${collect_progress_dir} \
            --resume \
            --worker_id ${w} \
            --num_workers ${COLLECT_WORKERS}"

        if [ -n "${TASKS}" ]; then
            collect_cmd="${collect_cmd} --tasks \"${TASKS}\""
        fi

        echo -e "\033[33m  [Worker $w] GPU=${gpu_id} LOG=${log_file}\033[0m"

        CUDA_VISIBLE_DEVICES=${gpu_id} \
        PYTHONWARNINGS=ignore::UserWarning \
        eval ${collect_cmd} > "${log_file}" 2>&1 &

        collect_pid=$!
        collect_pids+=("${collect_pid}")
        echo "${collect_pid}" >> "${collect_pid_file}"
    done

    # Wait for all workers to finish
    echo ""
    if [ "${MONITOR_PROGRESS}" -eq 1 ]; then
        echo -e "\033[34m  Monitoring seed collection progress...\033[0m"
        python - "${collect_progress_dir}" "${collect_pid_file}" "${PROGRESS_INTERVAL}" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

progress_dir = Path(sys.argv[1])
pid_file = Path(sys.argv[2])
interval = float(sys.argv[3])

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    pids = [int(line.strip()) for line in pid_file.read_text().splitlines() if line.strip()]
except FileNotFoundError:
    pids = []

def is_running(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

def read_progress_files():
    records = {}
    for path in sorted(progress_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        task = data.get("task", path.stem)
        records[task] = data
    return records

def done_count(records):
    return sum(int(data.get("valid", 0)) for data in records.values())

def total_count(records):
    return sum(int(data.get("total", 0)) for data in records.values())

if tqdm is None:
    print("tqdm is not installed; falling back to plain collect progress output.")
    while True:
        records = read_progress_files()
        running = sum(1 for pid in pids if is_running(pid))
        print("\033[2J\033[H", end="")
        print(f"collect overall: {done_count(records)}/{total_count(records)} | running workers: {running}/{len(pids)}")
        for task, data in sorted(records.items()):
            print(
                f"{task}: valid/total={data.get('valid', 0)}/{data.get('total', 0)} "
                f"attempts={data.get('attempts', 0)}/{data.get('max_attempts', 0)}"
            )
        if running == 0:
            break
        time.sleep(interval)
else:
    bars = {}
    overall_bar = tqdm(total=1, desc="collect overall", position=0, dynamic_ncols=True, leave=True)
    try:
        while True:
            records = read_progress_files()
            running = sum(1 for pid in pids if is_running(pid))
            overall_total = max(total_count(records), 1)
            overall_valid = min(done_count(records), overall_total)

            overall_bar.total = overall_total
            overall_bar.n = overall_valid
            overall_bar.set_postfix_str(f"valid/total={overall_valid}/{overall_total} running={running}/{len(pids)}")
            overall_bar.refresh()

            for idx, (task, data) in enumerate(sorted(records.items()), start=1):
                if task not in bars:
                    bars[task] = tqdm(
                        total=max(int(data.get("total", 0)), 1),
                        desc=f"{task}",
                        position=idx,
                        dynamic_ncols=True,
                        leave=True,
                    )
                bar = bars[task]
                total = max(int(data.get("total", 0)), 1)
                valid = min(int(data.get("valid", 0)), total)
                bar.total = total
                bar.n = valid
                bar.set_postfix_str(
                    f"valid/total={valid}/{data.get('total', 0)} "
                    f"attempts={data.get('attempts', 0)}/{data.get('max_attempts', 0)}"
                )
                bar.refresh()

            if running == 0:
                break
            time.sleep(interval)
    finally:
        for bar in reversed(list(bars.values())):
            bar.close()
        overall_bar.close()
PY
    else
        echo -e "\033[34m  Waiting for ${COLLECT_WORKERS} workers to finish...\033[0m"
    fi

    for pid in "${collect_pids[@]}"; do
        wait $pid
    done

    # Merge shards
    echo -e "\033[34m  Merging shard files...\033[0m"
    python -m evaluation.robotwin.collect_seeds merge --output "${SEEDS_FILE}"

    echo -e "\033[32m[Phase 1] Done.\033[0m"
    echo ""
else
    if [ "${SKIP_COLLECT}" -eq 1 ]; then
        echo -e "\033[33m[Phase 1] Skipped (SKIP_COLLECT=1). Using existing: ${SEEDS_FILE}\033[0m"
    elif [ -f "${SEEDS_FILE}" ]; then
        echo -e "\033[33m[Phase 1] Skipped (${SEEDS_FILE} already exists with cached episode_info).\033[0m"
    else
        echo -e "\033[33m[Phase 1] Skipped. Using existing: ${SEEDS_FILE}\033[0m"
    fi
    if [ ! -f "${SEEDS_FILE}" ]; then
        echo -e "\033[31mError: ${SEEDS_FILE} not found!\033[0m"
        exit 1
    fi
    echo ""
fi

# ============================================================
# Phase 2: Balance assignment
# ============================================================
echo -e "\033[34m[Phase 2] Balancing task assignments across ${num_clients} clients...\033[0m"

balance_cmd="python -m evaluation.robotwin.balance_tasks \
    --mode session \
    --valid_seeds ${SEEDS_FILE} \
    --num_clients ${num_clients} \
    --test_num ${test_num} \
    --output_dir ${ASSIGNMENT_DIR} \
    --verbose"

if [ -n "${TASKS}" ]; then
    balance_cmd="${balance_cmd} --tasks \"${TASKS}\""
fi

eval ${balance_cmd}

echo -e "\033[32m[Phase 2] Done. Assignments saved to ${ASSIGNMENT_DIR}/\033[0m"
echo ""

# ============================================================
# Phase 3: Launch eval clients
# ============================================================
echo -e "\033[34m[Phase 3] Launching ${num_clients} eval clients...\033[0m"

pid_file="pids_session.txt"
: > "$pid_file"
trap 'cleanup_eval_clients 130' INT
trap 'cleanup_eval_clients 143' TERM

batch_time=$(date +%Y%m%d_%H%M%S)
progress_marker="${RUN_DIR}/.progress_${batch_time}.marker"
mkdir -p "${RUN_DIR}/metrics"
: > "${progress_marker}"

for i in $(seq 0 $(( num_clients - 1 ))); do
    gpu_id="${CLIENT_GPUS[$(( i % ${#CLIENT_GPUS[@]} ))]}"
    # Round-robin assign clients to servers
    port=$(( START_PORT + (i % NUM_SERVERS) ))
    assignment_file="${ASSIGNMENT_DIR}/client_${i}.json"
    log_file="${log_dir}/session_client_${i}_${batch_time}.log"

    echo -e "\033[33m  [Client $i] GPU=${gpu_id} PORT=${port} (server $(( i % NUM_SERVERS ))) Assignment=${assignment_file}\033[0m"

    restart_args=()
    if [ "${RESTART_PER_TASK}" -eq 1 ]; then
        restart_args+=(--restart_per_task)
    fi

    CUDA_VISIBLE_DEVICES=${gpu_id} \
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::UserWarning \
    setsid nohup python -u -m evaluation.robotwin.eval_session_client \
        --config policy/${policy_name}/deploy_policy.yml \
        --assignment "${assignment_file}" \
        --port ${port} \
        --save_root "${run_root}" \
        --run_dir "${RUN_DIR}" \
        --seed ${seed} \
        --client_id ${i} \
        --task_config ${task_config} \
        --video_guidance_scale 5 \
        --action_guidance_scale 1 \
        --policy_name ${policy_name} \
        "${restart_args[@]}" > "$log_file" 2>&1 &

    pid=$!
    echo "${pid}" | tee -a "$pid_file"
done

echo ""
echo -e "\033[32mAll ${num_clients} clients launched. PIDs in ${pid_file}.\033[0m"
echo -e "\033[36mTo terminate all process groups: while read -r pid; do kill -- -\${pid}; done < ${pid_file}\033[0m"
echo -e "\033[36mTo monitor: tail -f logs/session_client_*_${batch_time}.log\033[0m"
echo ""
echo -e "\033[36mAfter all clients finish, merge metrics with:\033[0m"
echo -e "\033[36m  python -m evaluation.robotwin.eval_session_client merge --metrics_dir ${RUN_DIR}/metrics\033[0m"

if [ "${MONITOR_PROGRESS}" -eq 1 ]; then
    echo ""
    echo -e "\033[34m[Phase 4] Monitoring client progress with tqdm every ${PROGRESS_INTERVAL}s...\033[0m"

    python - "${ASSIGNMENT_DIR}" "${RUN_DIR}/metrics" "${progress_marker}" "${num_clients}" "${pid_file}" "${PROGRESS_INTERVAL}" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

assignment_dir = Path(sys.argv[1])
metrics_dir = Path(sys.argv[2])
marker = Path(sys.argv[3])
num_clients = int(sys.argv[4])
pid_file = Path(sys.argv[5])
interval = float(sys.argv[6])
marker_mtime = marker.stat().st_mtime if marker.exists() else 0

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    pids = [int(line.strip()) for line in pid_file.read_text().splitlines() if line.strip()]
except FileNotFoundError:
    pids = []

def is_running(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

def load_assignments():
    client_totals = []
    task_totals = {}
    task_order = []
    for client_id in range(num_clients):
        try:
            assignment = json.loads((assignment_dir / f"client_{client_id}.json").read_text())
        except Exception:
            assignment = []
        client_totals.append(len(assignment))
        for item in assignment:
            task = item.get("task", "unknown")
            if task not in task_totals:
                task_totals[task] = 0
                task_order.append(task)
            task_totals[task] += 1
    return client_totals, task_totals, task_order

def read_progress():
    client_done = [0 for _ in range(num_clients)]
    client_succ = [0 for _ in range(num_clients)]
    task_done = {}
    task_succ = {}

    for client_id in range(num_clients):
        metrics_file = metrics_dir / f"client_{client_id}.json"
        if not (metrics_file.exists() and metrics_file.stat().st_mtime >= marker_mtime):
            continue
        try:
            metrics = json.loads(metrics_file.read_text())
        except Exception:
            continue

        for task, task_metrics in metrics.items():
            done = int(task_metrics.get("total_num", 0))
            succ = int(task_metrics.get("succ_num", 0))
            client_done[client_id] += done
            client_succ[client_id] += succ
            task_done[task] = task_done.get(task, 0) + done
            task_succ[task] = task_succ.get(task, 0) + succ

    return client_done, client_succ, task_done, task_succ

def get_client_status(client_id, total, done):
    pid = pids[client_id] if client_id < len(pids) else None
    alive = is_running(pid) if pid is not None else False
    status = "RUN" if alive else ("DONE" if total > 0 and done >= total else "EXIT")
    return status, alive

client_totals, task_totals, task_order = load_assignments()
overall_total = sum(client_totals)

if tqdm is None:
    print("tqdm is not installed; falling back to plain progress output.")
    while True:
        client_done, client_succ, task_done, task_succ = read_progress()
        overall_done = sum(client_done)
        overall_succ = sum(client_succ)
        running = 0
        client_lines = []
        task_lines = []

        for client_id, total in enumerate(client_totals):
            done = client_done[client_id]
            succ = client_succ[client_id]
            status, alive = get_client_status(client_id, total, done)
            running += int(alive)
            pct = (done / total * 100) if total else 100.0
            client_lines.append(
                f"client {client_id:02d}: succ/done/total={succ}/{done}/{total} ({pct:.1f}%) {status}"
            )

        for task in task_order:
            total = task_totals[task]
            done = task_done.get(task, 0)
            succ = task_succ.get(task, 0)
            pct = (done / total * 100) if total else 100.0
            task_lines.append(f"task {task}: succ/done/total={succ}/{done}/{total} ({pct:.1f}%)")

        overall_pct = (overall_done / overall_total * 100) if overall_total else 100.0
        print("\033[2J\033[H", end="")
        print(
            f"overall: succ/done/total={overall_succ}/{overall_done}/{overall_total} "
            f"({overall_pct:.1f}%) | running clients: {running}/{num_clients}"
        )
        print("\nClients:")
        print("\n".join(client_lines))
        print("\nTasks:")
        print("\n".join(task_lines))
        if overall_done >= overall_total or running == 0:
            break
        time.sleep(interval)
else:
    bars = []
    overall_bar = tqdm(
        total=max(overall_total, 1),
        desc="overall",
        position=0,
        dynamic_ncols=True,
        leave=True,
    )
    bars.append(overall_bar)
    if overall_total == 0:
        overall_bar.n = 1
        overall_bar.set_postfix_str("succ/done/total=0/0/0")
        overall_bar.refresh()

    client_bars = []
    for client_id, total in enumerate(client_totals):
        bar = tqdm(
            total=max(total, 1),
            desc=f"client {client_id:02d}",
            position=client_id + 1,
            dynamic_ncols=True,
            leave=True,
        )
        if total == 0:
            bar.n = 1
            bar.set_postfix_str("succ/done/total=0/0/0 idle")
            bar.refresh()
        client_bars.append(bar)
        bars.append(bar)

    task_bars = {}
    task_position_offset = num_clients + 1
    for task_idx, task in enumerate(task_order):
        total = task_totals[task]
        bar = tqdm(
            total=max(total, 1),
            desc=f"task {task}",
            position=task_position_offset + task_idx,
            dynamic_ncols=True,
            leave=True,
        )
        if total == 0:
            bar.n = 1
            bar.set_postfix_str("succ/done/total=0/0/0")
            bar.refresh()
        task_bars[task] = bar
        bars.append(bar)

    try:
        while True:
            client_done, client_succ, task_done, task_succ = read_progress()
            overall_done = sum(client_done)
            overall_succ = sum(client_succ)
            running = 0

            for client_id, total in enumerate(client_totals):
                done = client_done[client_id]
                succ = client_succ[client_id]
                status, alive = get_client_status(client_id, total, done)
                running += int(alive)

                bar = client_bars[client_id]
                if total > 0:
                    bar.n = min(done, total)
                    bar.set_postfix_str(f"succ/done/total={succ}/{done}/{total} {status}")
                    bar.refresh()

            for task in task_order:
                total = task_totals[task]
                done = task_done.get(task, 0)
                succ = task_succ.get(task, 0)
                bar = task_bars[task]
                if total > 0:
                    bar.n = min(done, total)
                    bar.set_postfix_str(f"succ/done/total={succ}/{done}/{total}")
                    bar.refresh()

            overall_bar.n = min(overall_done, overall_total)
            overall_bar.set_postfix_str(
                f"succ/done/total={overall_succ}/{overall_done}/{overall_total} "
                f"running={running}/{num_clients}"
            )
            overall_bar.refresh()

            if overall_done >= overall_total or running == 0:
                break
            time.sleep(interval)
    finally:
        for bar in reversed(bars):
            bar.close()

print("All eval clients have exited or all assigned episodes are accounted for.")
print("Merge metrics with:")
print(f"  python -m evaluation.robotwin.eval_session_client merge --metrics_dir {metrics_dir}")
PY

fi

trap - INT TERM
