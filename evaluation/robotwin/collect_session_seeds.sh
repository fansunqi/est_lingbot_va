#!/bin/bash
# Collect valid RoboTwin seeds only.
#
# Usage:
#   bash evaluation/robotwin/collect_session_seeds.sh [save_root] [test_num] [seed] [group_id]
#
# Examples:
#   CLIENT_GPUS="5 6 7 5 6 7" \
#     bash evaluation/robotwin/collect_session_seeds.sh ./results 50 0 0
#
#   TASKS="place_fan stamp_seal" COLLECT_GPUS="5 6 7" COLLECT_WORKERS=6 \
#     bash evaluation/robotwin/collect_session_seeds.sh ./results 50 0

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(realpath "${SCRIPT_DIR}/../..")
cd "${REPO_ROOT}"

export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH
unset VK_ICD_FILENAMES
export no_proxy=localhost,127.0.0.1,0.0.0.0
export NO_PROXY=localhost,127.0.0.1,0.0.0.0

save_root=${1:-'./results'}
test_num=${2:-100}
seed=${3:-0}
group_id=${4:-""}

task_config=${TASK_CONFIG:-demo_randomized}
NUM_GROUPS=${NUM_GROUPS:-7}
CLIENT_GPUS=(${CLIENT_GPUS:-5 6 7 5 6 7 5 6 7})
SKIP_COLLECT=${SKIP_COLLECT:-0}
FORCE_COLLECT=${FORCE_COLLECT:-0}
SEED_VALIDATION_REPLAYS=${SEED_VALIDATION_REPLAYS:-1}
MONITOR_PROGRESS=${MONITOR_PROGRESS:-1}
PROGRESS_INTERVAL=${PROGRESS_INTERVAL:-3}

if [ -n "${TASKS:-}" ]; then
    selected_tasks="${TASKS}"
elif [ -n "${group_id}" ]; then
    selected_tasks=$(python -m evaluation.robotwin.balance_tasks --mode task --num_clients "${NUM_GROUPS}" --group_id "${group_id}")
else
    selected_tasks=""
fi

mkdir -p "${save_root}"
save_root=$(realpath "${save_root}")
run_root="${save_root}/${task_config}"
st_seed=$((10000 * (1 + seed)))
group_label=${group_id:-all}
if [ -n "${TASKS:-}" ] && [ -z "${group_id}" ]; then
    group_label="custom"
fi
RUN_NAME="stseed-${st_seed}_group-${group_label}_test-${test_num}"
RUN_DIR="${run_root}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
SEEDS_FILE="${RUN_DIR}/valid_seeds.json"

log_dir="./logs"
mkdir -p "${log_dir}"

echo -e "\033[32m============================================\033[0m"
echo -e "\033[32m  Collect RoboTwin Valid Seeds\033[0m"
echo -e "\033[32m============================================\033[0m"
echo "  run_dir:      ${RUN_DIR}"
echo "  output:       ${SEEDS_FILE}"
echo "  test_num:     ${test_num}"
echo "  seed:         ${seed} (st_seed=${st_seed})"
echo "  task_config:  ${task_config}"
echo "  client_gpus:  ${CLIENT_GPUS[*]}"
echo "  force_collect:${FORCE_COLLECT}"
echo "  seed_replays: ${SEED_VALIDATION_REPLAYS}"
echo "  monitor:      ${MONITOR_PROGRESS}"
if [ -n "${selected_tasks}" ]; then
    echo "  tasks:        ${selected_tasks}"
else
    echo "  tasks:        all"
fi
echo ""

NEED_COLLECT=0
if [ "${FORCE_COLLECT}" -eq 1 ]; then
    if [ "${SKIP_COLLECT}" -eq 1 ]; then
        echo -e "\033[31mError: FORCE_COLLECT=1 conflicts with SKIP_COLLECT=1\033[0m"
        exit 1
    fi
    echo -e "\033[33mFORCE_COLLECT=1; recollecting valid seeds.\033[0m"
    rm -f "${SEEDS_FILE}" "${RUN_DIR}"/valid_seeds_worker*.json
    NEED_COLLECT=1
elif [ "${SKIP_COLLECT}" -eq 0 ]; then
    if [ ! -f "${SEEDS_FILE}" ]; then
        NEED_COLLECT=1
    elif ! python - "${SEEDS_FILE}" "${test_num}" "${selected_tasks}" <<'PY'
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
        echo -e "\033[33mExisting ${SEEDS_FILE} is missing enough cached episode_info; collecting.\033[0m"
        NEED_COLLECT=1
    fi
fi

if [ "${NEED_COLLECT}" -ne 1 ]; then
    if [ "${SKIP_COLLECT}" -eq 1 ]; then
        echo -e "\033[33mSkipped (SKIP_COLLECT=1). Using existing: ${SEEDS_FILE}\033[0m"
    else
        echo -e "\033[33mSkipped (${SEEDS_FILE} already exists with cached episode_info).\033[0m"
    fi
    if [ ! -f "${SEEDS_FILE}" ]; then
        echo -e "\033[31mError: ${SEEDS_FILE} not found!\033[0m"
        exit 1
    fi
    exit 0
fi

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

echo -e "\033[34mCollecting valid seeds (${COLLECT_WORKERS} workers)...\033[0m"
echo "  Collect GPUs: ${COLLECT_GPUS[*]}"
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

    if [ -n "${selected_tasks}" ]; then
        collect_cmd="${collect_cmd} --tasks \"${selected_tasks}\""
    fi

    echo -e "\033[33m  [Worker $w] GPU=${gpu_id} LOG=${log_file}\033[0m"

    CUDA_VISIBLE_DEVICES=${gpu_id} \
    PYTHONWARNINGS=ignore::UserWarning \
    eval ${collect_cmd} > "${log_file}" 2>&1 &

    collect_pid=$!
    collect_pids+=("${collect_pid}")
    echo "${collect_pid}" >> "${collect_pid_file}"
done

if [ "${MONITOR_PROGRESS}" -eq 1 ]; then
    echo ""
    echo -e "\033[34mMonitoring seed collection progress...\033[0m"
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
    echo ""
    echo -e "\033[34mWaiting for ${COLLECT_WORKERS} workers to finish...\033[0m"
fi

for pid in "${collect_pids[@]}"; do
    wait "${pid}"
done

echo -e "\033[34mMerging shard files...\033[0m"
python -m evaluation.robotwin.collect_seeds merge --output "${SEEDS_FILE}"

echo -e "\033[32mDone. Saved valid seeds to ${SEEDS_FILE}\033[0m"
