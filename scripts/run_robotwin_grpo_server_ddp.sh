#!/usr/bin/env bash
set -euo pipefail

# Launch replicated GRPO servers via torchrun: each rank holds a full copy
# of the model and binds its own websocket port (PORT, PORT+1, ...). Rollout
# clients pair to each port independently (so rollout throughput scales with
# rank count), and gradients are synchronized across ranks at GRPO update
# time inside src.rl.server._run_grpo_update.
#
# Foreground launcher: the server runs as a child process group of this script.
# Press Ctrl-C to terminate the torchrun parent and every server rank together.
#
# Use this when you want one shared policy trained across multiple GPUs but
# also want each GPU to drive its own rollout clients in parallel. For a
# memory-sharded single logical server use ..._fsdp.sh; for fully independent
# (non-synchronized) servers use ..._split.sh.

CONFIG=${CONFIG:-configs/rl/robotwin_grpo_turn_switch_check.yaml}
PORT=${PORT:-29546}
SAVE_ROOT=${SAVE_ROOT:-experiments/robotwin_grpo_ddp}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
MASTER_PORT=${MASTER_PORT:-29646}
LOG_DIR=${LOG_DIR:-logs}
UV=${UV:-uv}

if [[ "${NPROC_PER_NODE}" -lt 2 ]]; then
  echo "Replicated GRPO needs NPROC_PER_NODE >= 2 (else use the single-GPU script)." >&2
  exit 1
fi

if ! command -v "${UV}" >/dev/null 2>&1; then
  echo "Could not find uv. Run this script from the LingBot checkout with uv available." >&2
  exit 1
fi

# Replicated mode is signaled by the absence of rl.use_fsdp; sanity-check the
# config rather than guessing. grep is good enough -- the YAML is small.
if grep -E "^[[:space:]]*use_fsdp:[[:space:]]*true" "${CONFIG}" >/dev/null 2>&1; then
  echo "Config ${CONFIG} sets rl.use_fsdp=true -- use run_robotwin_grpo_server_fsdp.sh instead." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${SAVE_ROOT}"
batch_time=$(date +%Y%m%d_%H%M%S)
log_file="${LOG_DIR}/grpo_server_ddp_${batch_time}.log"

end_port=$((PORT + NPROC_PER_NODE - 1))
echo "Launching replicated GRPO server: GPUs=${CUDA_VISIBLE_DEVICES} NPROC=${NPROC_PER_NODE} PORTS=${PORT}..${end_port} LOG=${log_file}"
echo "Rank r will bind ${PORT}+r -- pair rollout clients to each port (e.g. PORT=${PORT} and PORT=$((PORT+1)))."

SERVER_PID=""
TAIL_PID=""
SERVER_PGID=""
SHUTTING_DOWN=0

cleanup() {
  if [[ "${SHUTTING_DOWN}" == "1" ]]; then
    return
  fi
  SHUTTING_DOWN=1
  trap - INT TERM HUP EXIT
  echo
  echo "[run_robotwin_grpo_server_ddp] Caught shutdown signal, terminating server group..."

  if [[ -n "${TAIL_PID}" ]] && kill -0 "${TAIL_PID}" 2>/dev/null; then
    kill "${TAIL_PID}" 2>/dev/null || true
  fi

  if [[ -n "${SERVER_PGID}" ]]; then
    kill -TERM -- "-${SERVER_PGID}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      if ! kill -0 -- "-${SERVER_PGID}" 2>/dev/null; then
        break
      fi
      sleep 0.5
    done
    if kill -0 -- "-${SERVER_PGID}" 2>/dev/null; then
      echo "[run_robotwin_grpo_server_ddp] Server group ${SERVER_PGID} still alive after SIGTERM; sending SIGKILL."
      kill -KILL -- "-${SERVER_PGID}" 2>/dev/null || true
    fi
  fi

  wait 2>/dev/null || true
  echo "[run_robotwin_grpo_server_ddp] Shutdown complete."
}

trap cleanup INT TERM HUP
trap cleanup EXIT

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
setsid "${UV}" run torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  -m src.rl.server \
  --config "${CONFIG}" \
  --port "${PORT}" \
  --save-root "${SAVE_ROOT}" \
  > "${log_file}" 2>&1 &
SERVER_PID=$!
SERVER_PGID=${SERVER_PID}

echo "${SERVER_PID}" > "${LOG_DIR}/grpo_server_ddp.pid"
echo "Replicated GRPO server PID: ${SERVER_PID} (process group ${SERVER_PGID})"
echo "Streaming log: ${log_file}"
echo "Press Ctrl-C to stop the server and exit."
echo

tail -n +1 -F "${log_file}" &
TAIL_PID=$!

wait "${SERVER_PID}"
