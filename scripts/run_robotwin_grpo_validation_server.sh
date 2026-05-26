#!/usr/bin/env bash
set -euo pipefail

# Launch one GRPO validation server. Run this from the LingBot uv environment.

CONFIG=${CONFIG:-configs/rl/robotwin_grpo_turn_switch_validation.yaml}
PORT=${PORT:-29546}
SERVER_GPU=${SERVER_GPU:-0}
SAVE_ROOT=${SAVE_ROOT:-experiments/robotwin_grpo_turn_switch_validation/server}
LOG_DIR=${LOG_DIR:-logs}
UV=${UV:-uv}

if ! command -v "${UV}" >/dev/null 2>&1; then
  echo "Could not find uv. Run this script from the LingBot checkout with uv available." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${SAVE_ROOT}"
batch_time=$(date +%Y%m%d_%H%M%S)
log_file="${LOG_DIR}/grpo_validation_server_${batch_time}.log"

echo "Launching GRPO validation server: GPU=${SERVER_GPU} PORT=${PORT} SAVE_ROOT=${SAVE_ROOT} LOG=${log_file}"
CUDA_VISIBLE_DEVICES="${SERVER_GPU}" \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup "${UV}" run python -u -m src.rl.server \
  --config "${CONFIG}" \
  --save-root "${SAVE_ROOT}" \
  --port "${PORT}" \
  > "${log_file}" 2>&1 &

echo $! > "${LOG_DIR}/grpo_validation_server.pid"
echo "GRPO validation server launched. PID: $(cat "${LOG_DIR}/grpo_validation_server.pid")"
