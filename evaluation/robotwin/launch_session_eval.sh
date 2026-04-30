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
task_config=${TASK_CONFIG:-demo_clean}
NUM_GROUPS=${NUM_GROUPS:-7}

START_PORT=${START_PORT:-29556}
CLIENT_GPUS=(${CLIENT_GPUS:-5 6 7 5 6 7 5 6 7})
SKIP_COLLECT=${SKIP_COLLECT:-0}

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

# Paths
SEEDS_FILE="${save_root}/valid_seeds.json"
ASSIGNMENT_DIR="${save_root}/task_assignments"

policy_name=ACT

echo -e "\033[32m============================================\033[0m"
echo -e "\033[32m  Session-Level Balanced Evaluation\033[0m"
echo -e "\033[32m============================================\033[0m"
echo "  save_root:    ${save_root}"
echo "  test_num:     ${test_num}"
echo "  seed:         ${seed}"
echo "  num_clients:  ${num_clients}"
echo "  task_config:  ${task_config}"
echo "  client_gpus:  ${CLIENT_GPUS[*]}"
echo "  start_port:   ${START_PORT}"
if [ -n "${group_id}" ]; then
    echo "  group_id:     ${group_id} (of ${NUM_GROUPS} groups)"
    echo "  tasks:        ${TASKS}"
else
    echo "  group_id:     (all tasks)"
fi
echo ""

# ============================================================
# Phase 1: Collect valid seeds
# ============================================================
if [ "${SKIP_COLLECT}" -eq 0 ]; then
    echo -e "\033[34m[Phase 1] Collecting valid seeds...\033[0m"
    echo "  Output: ${SEEDS_FILE}"
    echo ""

    # Use first available GPU for seed collection
    COLLECT_GPU=${COLLECT_GPU:-${CLIENT_GPUS[0]}}

    collect_cmd="python -m evaluation.robotwin.collect_seeds \
        --test_num ${test_num} \
        --seed ${seed} \
        --task_config ${task_config} \
        --output ${SEEDS_FILE} \
        --resume"

    if [ -n "${TASKS}" ]; then
        collect_cmd="${collect_cmd} --tasks \"${TASKS}\""
    fi

    CUDA_VISIBLE_DEVICES=${COLLECT_GPU} \
    PYTHONWARNINGS=ignore::UserWarning \
    eval ${collect_cmd}

    echo -e "\033[32m[Phase 1] Done.\033[0m"
    echo ""
else
    echo -e "\033[33m[Phase 1] Skipped (SKIP_COLLECT=1). Using existing: ${SEEDS_FILE}\033[0m"
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

log_dir="./logs"
mkdir -p "$log_dir"
pid_file="pids_session.txt"
: > "$pid_file"

batch_time=$(date +%Y%m%d_%H%M%S)

for i in $(seq 0 $(( num_clients - 1 ))); do
    gpu_id="${CLIENT_GPUS[$(( i % ${#CLIENT_GPUS[@]} ))]}"
    port=$(( START_PORT + i ))
    assignment_file="${ASSIGNMENT_DIR}/client_${i}.json"
    log_file="${log_dir}/session_client_${i}_${batch_time}.log"

    echo -e "\033[33m  [Client $i] GPU=${gpu_id} PORT=${port} Assignment=${assignment_file}\033[0m"

    CUDA_VISIBLE_DEVICES=${gpu_id} \
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::UserWarning \
    nohup python -u -m evaluation.robotwin.eval_session_client \
        --config policy/${policy_name}/deploy_policy.yml \
        --assignment "${assignment_file}" \
        --port ${port} \
        --save_root "${save_root}" \
        --task_config ${task_config} \
        --video_guidance_scale 5 \
        --action_guidance_scale 1 \
        --policy_name ${policy_name} > "$log_file" 2>&1 &

    pid=$!
    echo "${pid}" | tee -a "$pid_file"
done

echo ""
echo -e "\033[32mAll ${num_clients} clients launched. PIDs in ${pid_file}.\033[0m"
echo -e "\033[36mTo terminate all: kill \$(cat ${pid_file})\033[0m"
echo -e "\033[36mTo monitor: tail -f logs/session_client_*_${batch_time}.log\033[0m"
