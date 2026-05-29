#!/usr/bin/env bash
set -euo pipefail

# Launch parallel rollout clients that cooperatively fill one GRPO group.
# Run this from the RoboTwin conda environment.

ASSIGNMENT=${ASSIGNMENT:-${1:?Set ASSIGNMENT or pass assignment JSON as the first argument}}
PORT=${PORT:-29546}
HOST=${HOST:-127.0.0.1}
SAVE_ROOT=${SAVE_ROOT:-experiments/robotwin_grpo_turn_switch_validation/rollouts}
GROUP_SIZE=${GROUP_SIZE:-16}
NUM_CLIENTS=${NUM_CLIENTS:-4}
CLIENT_GPUS=${CLIENT_GPUS:-"0 0 0 0"}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
SKIP_RENDER_CHECK=${SKIP_RENDER_CHECK:-1}
GROUP_BARRIER=${GROUP_BARRIER:-1}
GROUP_BARRIER_TIMEOUT=${GROUP_BARRIER_TIMEOUT:-0}
SAVE_VISUALIZATION=${SAVE_VISUALIZATION:-none}
SAVE_EVAL_VISUALIZATION=${SAVE_EVAL_VISUALIZATION:-none}
TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/root/WAM/.cache/torch_extensions}

export ASSIGNMENT
export PORT
export HOST
export SAVE_ROOT
export GROUP_SIZE
export NUM_CLIENTS
export CLIENT_GPUS
export TASK_CONFIG
export SKIP_RENDER_CHECK
export GROUP_BARRIER
export GROUP_BARRIER_TIMEOUT
export SAVE_VISUALIZATION
export SAVE_EVAL_VISUALIZATION
export TORCH_EXTENSIONS_DIR
mkdir -p "${TORCH_EXTENSIONS_DIR}"

bash evaluation/robotwin/launch_grpo_rollout_clients.sh
