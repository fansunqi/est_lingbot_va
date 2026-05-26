#!/usr/bin/env bash
set -euo pipefail

# Fast GRPO validation server for turn_switch. Run from the LingBot uv env.
#
# Foreground launcher: the server runs as a child process group of this
# script. Ctrl-C (SIGINT) — or SIGTERM/SIGHUP — propagates to the whole
# group, so the python process, any `uv run` wrappers, Lightning spawned
# children, and the log tail are all terminated together.
#
# After the server prints "GRPO server ready", launch rollout clients in
# another terminal from the RoboTwin conda env with:
#
#   cd /root/WAM/lingbot-va
#   conda activate RoboTwin
#   ASSIGNMENT=experiments/robotwin_grpo_turn_switch_fast/assignment.json \
#   SAVE_ROOT=experiments/robotwin_grpo_turn_switch_fast \
#   GROUP_SIZE=8 NUM_CLIENTS=8 GROUP_BARRIER=1 CLIENT_GPUS=0 \
#   bash evaluation/robotwin/launch_grpo_rollout_clients.sh

CONFIG=${CONFIG:-configs/rl/robotwin_grpo_turn_switch_fast.yaml}
PORT=${PORT:-29546}
SERVER_GPU=${SERVER_GPU:-0}
SAVE_ROOT=${SAVE_ROOT:-experiments/robotwin_grpo_turn_switch_fast/server}
LOG_DIR=${LOG_DIR:-logs}
UV=${UV:-uv}

if ! command -v "${UV}" >/dev/null 2>&1; then
  echo "Could not find uv. Run this script from the LingBot checkout with uv available." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${SAVE_ROOT}"
batch_time=$(date +%Y%m%d_%H%M%S)
log_file="${LOG_DIR}/grpo_fast_server_${batch_time}.log"

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
  echo "[run_robotwin_grpo_turn_switch_fast] Caught shutdown signal, terminating server..."

  if [[ -n "${TAIL_PID}" ]] && kill -0 "${TAIL_PID}" 2>/dev/null; then
    kill "${TAIL_PID}" 2>/dev/null || true
  fi

  if [[ -n "${SERVER_PGID}" ]]; then
    # SIGTERM the whole process group, give it 10s, then SIGKILL.
    kill -TERM -- "-${SERVER_PGID}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      if ! kill -0 -- "-${SERVER_PGID}" 2>/dev/null; then
        break
      fi
      sleep 0.5
    done
    if kill -0 -- "-${SERVER_PGID}" 2>/dev/null; then
      echo "[run_robotwin_grpo_turn_switch_fast] Server group ${SERVER_PGID} still alive after SIGTERM; sending SIGKILL."
      kill -KILL -- "-${SERVER_PGID}" 2>/dev/null || true
    fi
  fi

  wait 2>/dev/null || true
  echo "[run_robotwin_grpo_turn_switch_fast] Shutdown complete."
}

trap cleanup INT TERM HUP
trap cleanup EXIT

echo "Launching GRPO fast server: GPU=${SERVER_GPU} PORT=${PORT} SAVE_ROOT=${SAVE_ROOT} LOG=${log_file}"

# setsid puts the child (and everything it spawns) into a fresh process
# group so we can kill the entire tree in cleanup() with `kill -- -PGID`.
CUDA_VISIBLE_DEVICES="${SERVER_GPU}" \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
setsid "${UV}" run python -u -m src.rl.server \
  --config "${CONFIG}" \
  --save-root "${SAVE_ROOT}" \
  --port "${PORT}" \
  > "${log_file}" 2>&1 &
SERVER_PID=$!
# With setsid, the child is its own session/group leader, so PGID == PID.
SERVER_PGID=${SERVER_PID}

echo "${SERVER_PID}" > "${LOG_DIR}/grpo_fast_server.pid"
echo "GRPO fast server PID: ${SERVER_PID} (process group ${SERVER_PGID})"
echo "Streaming log: ${log_file}"
echo "Press Ctrl-C to stop the server and exit."
echo

# Mirror the log to this terminal. Killed by cleanup() on shutdown.
tail -n +1 -F "${log_file}" &
TAIL_PID=$!

# Wait specifically on the server. If the server exits on its own, cleanup()
# (via the EXIT trap) takes care of the tail and any orphans.
wait "${SERVER_PID}"
