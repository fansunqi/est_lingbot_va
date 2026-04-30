#!/bin/bash
# Launch N eval clients distributed across CLIENT_GPUS.
# Default: 6 clients alternating between GPU 6 and GPU 7 (3 clients per GPU),
# each connecting to its own server (port = START_PORT + i).

export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH
unset VK_ICD_FILENAMES   # stale path from old driver install would break Vulkan
export no_proxy=localhost,127.0.0.1,0.0.0.0
export NO_PROXY=localhost,127.0.0.1,0.0.0.0

save_root=${1:-'./results'}
task_list_id=${2:-0}
seed=${3:-0}
test_num=${4:-100}

START_PORT=${START_PORT:-29556}
# GPU assignment for each client (must match number of clients we launch)
CLIENT_GPUS=(${CLIENT_GPUS:-6 7 6 7 6 7})
NUM_CLIENTS=${#CLIENT_GPUS[@]}

policy_name=ACT
task_config=demo_clean
train_config_name=0
model_name=0

# Use load-balanced task assignment based on step limits.
# The balance_tasks.py script distributes all tasks across NUM_CLIENTS groups
# such that the total step_lim per group is approximately equal.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Read balanced groups from Python balancer
mapfile -t task_groups < <(python -m evaluation.robotwin.balance_tasks --num_clients ${NUM_CLIENTS} --verbose)

if (( task_list_id < 0 || task_list_id >= ${#task_groups[@]} )); then
  echo "task_list_id out of range: $task_list_id (0..$(( ${#task_groups[@]} - 1 )))" >&2
  exit 1
fi

read -r -a task_names <<< "${task_groups[$task_list_id]}"

echo -e "\033[32mtask_list_id=${task_list_id}, tasks in this group: ${#task_names[@]}\033[0m"
printf 'task_names (%d): %s\n' "${#task_names[@]}" "${task_names[*]}"
printf 'client_gpus: %s\n' "${CLIENT_GPUS[*]}"

log_dir="./logs"
mkdir -p "$log_dir"
pid_file="pids.txt"
: > "$pid_file"

batch_time=$(date +%Y%m%d_%H%M%S)

for i in "${!task_names[@]}"; do
    task_name="${task_names[$i]}"
    gpu_id="${CLIENT_GPUS[$(( i % NUM_CLIENTS ))]}"
    port=$(( START_PORT + i ))
    log_file="${log_dir}/${task_name}_${batch_time}.log"

    echo -e "\033[33m[Client $i] task=${task_name} GPU=${gpu_id} PORT=${port} LOG=${log_file}\033[0m"

    CUDA_VISIBLE_DEVICES=${gpu_id} \
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::UserWarning \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    nohup python -u -m evaluation.robotwin.eval_polict_client_openpi \
        --config policy/$policy_name/deploy_policy.yml \
        --overrides \
        --task_name ${task_name} \
        --task_config ${task_config} \
        --train_config_name ${train_config_name} \
        --model_name ${model_name} \
        --ckpt_setting ${model_name} \
        --seed ${seed} \
        --policy_name ${policy_name} \
        --save_root ${save_root} \
        --video_guidance_scale 5 \
        --action_guidance_scale 1 \
        --test_num ${test_num} \
        --port ${port} > "$log_file" 2>&1 &

    pid=$!
    echo "${pid}" | tee -a "$pid_file"
done

echo -e "\033[32mAll ${NUM_CLIENTS} clients launched. PIDs in ${pid_file}.\033[0m"
echo -e "\033[36mTo terminate all: kill \$(cat ${pid_file})\033[0m"
