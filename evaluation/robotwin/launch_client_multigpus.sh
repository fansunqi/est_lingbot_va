#!/bin/bash
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH


save_root=${1:-'./results'}

# General parameters
policy_name=ACT
task_config=demo_clean
train_config_name=0
model_name=0
seed=${3:-0}
test_num=${4:-100}
start_port=29556 
num_gpus=8

task_list_id=${2:-0}

# Use load-balanced task assignment (groups balanced by step_lim)
NUM_GROUPS=${NUM_GROUPS:-7}
mapfile -t task_groups < <(python -m evaluation.robotwin.balance_tasks --num_clients ${NUM_GROUPS} --verbose)

if (( task_list_id < 0 || task_list_id >= ${#task_groups[@]} )); then
  echo "task_list_id out of range: $task_list_id (0..$(( ${#task_groups[@]} - 1 )))" >&2
  exit 1
fi

read -r -a task_names <<< "${task_groups[$task_list_id]}"

echo "task_list_id=$task_list_id"
printf 'task_names (%d): %s\n' "${#task_names[@]}" "${task_names[*]}"

log_dir="./logs"
mkdir -p "$log_dir"

echo -e "\033[32mLaunching ${#task_names[@]} tasks. GPUs assigned by mod ${num_gpus}, ports starting from ${start_port} incrementing.\033[0m"

pid_file="pids.txt"
> "$pid_file"

batch_time=$(date +%Y%m%d_%H%M%S)

for i in "${!task_names[@]}"; do
    task_name="${task_names[$i]}"
    gpu_id=$(( i % num_gpus ))
    port=$(( start_port + i ))

    export CUDA_VISIBLE_DEVICES=${gpu_id}

    log_file="${log_dir}/${task_name}_${batch_time}.log"

    echo -e "\033[33m[Task $i] Task: ${task_name}, GPU: ${gpu_id}, PORT: ${port}, Log: ${log_file}\033[0m"

    PYTHONWARNINGS=ignore::UserWarning \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python -m evaluation.robotwin.eval_polict_client_openpi --config policy/$policy_name/deploy_policy.yml \
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

echo -e "\033[32mAll tasks launched. PIDs saved to ${pid_file}\033[0m"
echo -e "\033[36mTo terminate all processes, run: kill \$(cat ${pid_file})\033[0m"
