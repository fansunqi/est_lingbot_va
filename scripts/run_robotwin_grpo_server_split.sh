#!/usr/bin/env bash
set -euo pipefail

# Launch independent GRPO servers on consecutive GPUs, mirroring
# evaluation/robotwin/launch_server_split.sh. Each process uses the normal
# offload config path, not FSDP. Pair each server port with rollout clients.

CONFIG=${CONFIG:-configs/rl/robotwin_grpo_turn_switch_check.yaml}
START_PORT=${START_PORT:-29546}
NUM_SERVERS=${NUM_SERVERS:-1}
SERVER_GPU_START=${SERVER_GPU_START:-0}
SAVE_ROOT=${SAVE_ROOT:-experiments/robotwin_grpo_split}
LOG_DIR=${LOG_DIR:-logs}
UV=${UV:-uv}

if ! command -v "${UV}" >/dev/null 2>&1; then
  echo "Could not find uv. Run this script from the LingBot checkout with uv available." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${SAVE_ROOT}"
batch_time=$(date +%Y%m%d_%H%M%S)

echo "Launching ${NUM_SERVERS} offloaded GRPO server(s) on GPU ${SERVER_GPU_START}..$((SERVER_GPU_START + NUM_SERVERS - 1))"

for i in $(seq 0 $((NUM_SERVERS - 1))); do
  GPU=$((SERVER_GPU_START + i))
  CURRENT_PORT=$((START_PORT + i))
  CURRENT_SAVE_ROOT="${SAVE_ROOT}/server_${i}"
  LOG_FILE="${LOG_DIR}/grpo_server_split_${i}_${batch_time}.log"
  mkdir -p "${CURRENT_SAVE_ROOT}"

  echo "[GRPO server ${i}] GPU=${GPU} PORT=${CURRENT_PORT} SAVE_ROOT=${CURRENT_SAVE_ROOT} LOG=${LOG_FILE}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup "${UV}" run python -u -m src.rl.server \
    --config "${CONFIG}" \
    --save-root "${CURRENT_SAVE_ROOT}" \
    --port "${CURRENT_PORT}" \
    > "${LOG_FILE}" 2>&1 &
  sleep 2
done

echo "All ${NUM_SERVERS} offloaded GRPO server(s) launched."
wait
