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

task_groups=(
  "stack_bowls_three handover_block hanging_mug scan_object lift_pot put_object_cabinet stack_blocks_three place_shoe"
  "adjust_bottle place_mouse_pad dump_bin_bigbin move_pillbottle_pad pick_dual_bottles shake_bottle place_fan turn_switch"
  "shake_bottle_horizontally place_container_plate rotate_qrcode place_object_stand put_bottles_dustbin move_stapler_pad place_burger_fries place_bread_basket"
  "pick_diverse_bottles open_microwave beat_block_hammer press_stapler click_bell move_playingcard_away open_laptop move_can_pot"
  "stack_bowls_two place_a2b_right stamp_seal place_object_basket handover_mic place_bread_skillet stack_blocks_two place_cans_plasticbox"
  "click_alarmclock blocks_ranking_size place_phone_stand place_can_basket place_object_scale place_a2b_left grab_roller place_dual_shoes"
  "place_empty_cup blocks_ranking_rgb"
)

if (( task_list_id < 0 || task_list_id >= ${#task_groups[@]} )); then
  echo "task_list_id out of range: $task_list_id (0..$(( ${#task_groups[@]} - 1 )))" >&2
  exit 1
fi

read -r -a all_tasks <<< "${task_groups[$task_list_id]}"
# take first NUM_CLIENTS tasks from this group
task_names=("${all_tasks[@]:0:$NUM_CLIENTS}")
if (( ${#task_names[@]} < NUM_CLIENTS )); then
  echo "Warning: only ${#task_names[@]} tasks in group ${task_list_id}, reducing clients." >&2
  NUM_CLIENTS=${#task_names[@]}
fi

echo -e "\033[32mtask_list_id=${task_list_id}\033[0m"
printf 'task_names (%d): %s\n' "${#task_names[@]}" "${task_names[*]}"
printf 'gpu_ids: %s\n' "${CLIENT_GPUS[*]:0:$NUM_CLIENTS}"

log_dir="./logs"
mkdir -p "$log_dir"
pid_file="pids.txt"
: > "$pid_file"

batch_time=$(date +%Y%m%d_%H%M%S)

for i in "${!task_names[@]}"; do
    task_name="${task_names[$i]}"
    gpu_id="${CLIENT_GPUS[$i]}"
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
