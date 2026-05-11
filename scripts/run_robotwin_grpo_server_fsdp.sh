#!/usr/bin/env bash
set -euo pipefail

# Launch one logical GRPO server sharded across multiple GPUs with FSDP2.
# Rollout clients should connect to PORT; torchrun workers communicate through
# MASTER_PORT internally.

CONFIG=${CONFIG:-configs/rl/robotwin_grpo_turn_switch_check_fsdp.yaml}
PORT=${PORT:-29546}
SAVE_ROOT=${SAVE_ROOT:-experiments/robotwin_grpo_fsdp}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3,4}
MASTER_PORT=${MASTER_PORT:-29646}
LOG_DIR=${LOG_DIR:-logs}
UV=${UV:-uv}

if [[ "${NPROC_PER_NODE}" -lt 4 ]]; then
  echo "FSDP GRPO on 24GB GPUs should use NPROC_PER_NODE >= 4." >&2
  echo "For smaller runs, set text_encoder_cpu_offload: true in the config." >&2
  exit 1
fi

if ! command -v "${UV}" >/dev/null 2>&1; then
  echo "Could not find uv. Run this script from the LingBot checkout with uv available." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${SAVE_ROOT}"
batch_time=$(date +%Y%m%d_%H%M%S)
log_file="${LOG_DIR}/grpo_server_fsdp_${batch_time}.log"

echo "Launching FSDP GRPO server: GPUs=${CUDA_VISIBLE_DEVICES} NPROC=${NPROC_PER_NODE} PORT=${PORT} LOG=${log_file}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup "${UV}" run torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  -m src.rl.server \
  --config "${CONFIG}" \
  --port "${PORT}" \
  --save-root "${SAVE_ROOT}" \
  > "${log_file}" 2>&1 &

echo $! > "${LOG_DIR}/grpo_server_fsdp.pid"
echo "FSDP GRPO server launched. PID: $(cat "${LOG_DIR}/grpo_server_fsdp.pid")"
