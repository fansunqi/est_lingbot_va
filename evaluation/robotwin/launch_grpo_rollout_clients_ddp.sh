#!/usr/bin/env bash
set -euo pipefail

# One-shot launcher for DDP-mode GRPO rollout clients.
#
# The DDP server (scripts/run_robotwin_grpo_server_ddp.sh) binds one websocket
# per rank — rank r listens on START_PORT+r. This script fans out one client
# group per rank, each with its own RUN_DIR (so file barriers don't collide
# across groups), then waits on all of them. Ctrl-C / SIGTERM tears down every
# group together.
#
# Per server rank r we delegate to launch_grpo_rollout_clients.sh with:
#   PORT = START_PORT + r
#   RUN_DIR = BASE_RUN_DIR/server_r
#   CLIENT_GPUS = SERVER_GPUS[r]   (the same GPU that runs server rank r)
#
# All other knobs (GROUP_SIZE, TASK_CONFIG, GROUP_BARRIER, etc.) are forwarded
# verbatim and apply identically to every group.

ASSIGNMENT=${ASSIGNMENT:?Set ASSIGNMENT to a RoboTwin assignment JSON}

NUM_SERVERS=${NUM_SERVERS:-2}
START_PORT=${START_PORT:-29546}
HOST=${HOST:-127.0.0.1}
SAVE_ROOT=${SAVE_ROOT:-./results/grpo_ddp}
BASE_RUN_DIR=${BASE_RUN_DIR:-${SAVE_ROOT}/$(date +%Y%m%d_%H%M%S)}
# Space-separated list of GPU ids, indexed by server rank. Defaults to "0 1 ...".
SERVER_GPUS_DEFAULT=$(seq -s ' ' 0 $((NUM_SERVERS - 1)))
SERVER_GPUS=(${SERVER_GPUS:-${SERVER_GPUS_DEFAULT}})
# Number of client processes per server rank. Each client gets its own
# CUDA_VISIBLE_DEVICES (defaults to the server's own GPU, repeated).
NUM_CLIENTS_PER_SERVER=${NUM_CLIENTS_PER_SERVER:-1}

# Forwarded to launch_grpo_rollout_clients.sh — leave unset to use that
# script's own defaults.
GROUP_SIZE=${GROUP_SIZE:-2}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
GROUP_BARRIER=${GROUP_BARRIER:-1}
GROUP_BARRIER_TIMEOUT=${GROUP_BARRIER_TIMEOUT:-0}
SAVE_VISUALIZATION=${SAVE_VISUALIZATION:-none}
SAVE_EVAL_VISUALIZATION=${SAVE_EVAL_VISUALIZATION:-none}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-}
NUM_PASSES=${NUM_PASSES:-1}
SHUFFLE_ASSIGNMENT=${SHUFFLE_ASSIGNMENT:-0}
ASSIGNMENT_SHUFFLE_SEED=${ASSIGNMENT_SHUFFLE_SEED:-}
CLIENT_RESTART_EVERY_ITEMS=${CLIENT_RESTART_EVERY_ITEMS:-${GRPO_CLIENT_RESTART_EVERY_ITEMS:-25}}
EVAL_ONLY=${EVAL_ONLY:-0}
SKIP_RENDER_CHECK=${SKIP_RENDER_CHECK:-1}
PYTHON=${PYTHON:-python}
ALLOW_NON_ROBOTWIN_CONDA=${ALLOW_NON_ROBOTWIN_CONDA:-}

if [[ "${#SERVER_GPUS[@]}" -ne "${NUM_SERVERS}" ]]; then
  echo "SERVER_GPUS must have exactly NUM_SERVERS entries (got ${#SERVER_GPUS[@]} for NUM_SERVERS=${NUM_SERVERS})." >&2
  exit 1
fi

ASSIGNMENT=$(realpath "${ASSIGNMENT}")
SAVE_ROOT=$(realpath -m "${SAVE_ROOT}")
BASE_RUN_DIR=$(realpath -m "${BASE_RUN_DIR}")
mkdir -p "${BASE_RUN_DIR}"

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SUB_LAUNCHER="${REPO_ROOT}/evaluation/robotwin/launch_grpo_rollout_clients.sh"
if [[ ! -x "${SUB_LAUNCHER}" ]]; then
  echo "Sub-launcher not found or not executable: ${SUB_LAUNCHER}" >&2
  exit 1
fi

echo "DDP rollout fan-out:"
echo "  NUM_SERVERS=${NUM_SERVERS}"
echo "  START_PORT=${START_PORT} (rank r → port START_PORT+r)"
echo "  SERVER_GPUS=${SERVER_GPUS[*]}"
echo "  NUM_CLIENTS_PER_SERVER=${NUM_CLIENTS_PER_SERVER}"
echo "  BASE_RUN_DIR=${BASE_RUN_DIR}"
echo "  ASSIGNMENT=${ASSIGNMENT}"
echo "  NUM_PASSES=${NUM_PASSES}"
echo "  SHUFFLE_ASSIGNMENT=${SHUFFLE_ASSIGNMENT}"
echo "  ASSIGNMENT_SHUFFLE_SEED=${ASSIGNMENT_SHUFFLE_SEED:-<seed>}"
echo "  CLIENT_RESTART_EVERY_ITEMS=${CLIENT_RESTART_EVERY_ITEMS}"

GROUP_PIDS=()
SHUTTING_DOWN=0

cleanup() {
  if [[ "${SHUTTING_DOWN}" == "1" ]]; then
    return
  fi
  SHUTTING_DOWN=1
  trap - INT TERM HUP EXIT
  echo
  echo "[launch_grpo_rollout_clients_ddp] Shutdown signal — terminating ${#GROUP_PIDS[@]} group(s)..."

  # Sub-launcher runs in its own process group (setsid below), so a SIGTERM
  # to -<pgid> reaches every client and every RoboTwin subprocess.
  for pgid in "${GROUP_PIDS[@]}"; do
    kill -TERM -- "-${pgid}" 2>/dev/null || true
  done

  for _ in $(seq 1 30); do
    any_alive=0
    for pgid in "${GROUP_PIDS[@]}"; do
      if kill -0 -- "-${pgid}" 2>/dev/null; then
        any_alive=1
        break
      fi
    done
    [[ "${any_alive}" == "0" ]] && break
    sleep 0.5
  done

  for pgid in "${GROUP_PIDS[@]}"; do
    if kill -0 -- "-${pgid}" 2>/dev/null; then
      echo "[launch_grpo_rollout_clients_ddp] Group ${pgid} still alive — SIGKILL."
      kill -KILL -- "-${pgid}" 2>/dev/null || true
    fi
  done

  wait 2>/dev/null || true
  echo "[launch_grpo_rollout_clients_ddp] Shutdown complete."
}

trap cleanup INT TERM HUP
trap cleanup EXIT

for SERVER_RANK in $(seq 0 $((NUM_SERVERS - 1))); do
  port=$((START_PORT + SERVER_RANK))
  gpu="${SERVER_GPUS[${SERVER_RANK}]}"
  run_dir="${BASE_RUN_DIR}/server_${SERVER_RANK}"
  mkdir -p "${run_dir}"

  # Build per-group CLIENT_GPUS by repeating this server's GPU once per client.
  client_gpus=""
  for _ in $(seq 1 "${NUM_CLIENTS_PER_SERVER}"); do
    client_gpus+="${gpu} "
  done
  client_gpus="${client_gpus% }"

  echo "[group ${SERVER_RANK}] PORT=${port} GPU=${gpu} CLIENTS=${NUM_CLIENTS_PER_SERVER} RUN_DIR=${run_dir}"

  # setsid puts each sub-launcher in its own process group so cleanup() can
  # signal the whole tree (sub-launcher → clients → RoboTwin/SAPIEN children).
  # WORLD_SIZE/RANK propagate the sharded-group indexing to the sub-launcher,
  # which forwards them to grpo_rollout_client as --world_size/--rank. With
  # NUM_SERVERS=2, GROUP_SIZE=4, NUM_CLIENTS_PER_SERVER=1 each rank's single
  # client owns members [rank, rank+2] within every group_id.
  ASSIGNMENT="${ASSIGNMENT}" \
  PORT="${port}" \
  HOST="${HOST}" \
  SAVE_ROOT="${SAVE_ROOT}" \
  RUN_DIR="${run_dir}" \
  GROUP_SIZE="${GROUP_SIZE}" \
  TASK_CONFIG="${TASK_CONFIG}" \
  GROUP_BARRIER="${GROUP_BARRIER}" \
  GROUP_BARRIER_TIMEOUT="${GROUP_BARRIER_TIMEOUT}" \
  SAVE_VISUALIZATION="${SAVE_VISUALIZATION}" \
  SAVE_EVAL_VISUALIZATION="${SAVE_EVAL_VISUALIZATION}" \
  MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS}" \
  NUM_PASSES="${NUM_PASSES}" \
  SHUFFLE_ASSIGNMENT="${SHUFFLE_ASSIGNMENT}" \
  ASSIGNMENT_SHUFFLE_SEED="${ASSIGNMENT_SHUFFLE_SEED}" \
  CLIENT_RESTART_EVERY_ITEMS="${CLIENT_RESTART_EVERY_ITEMS}" \
  EVAL_ONLY="${EVAL_ONLY}" \
  SKIP_RENDER_CHECK="${SKIP_RENDER_CHECK}" \
  PYTHON="${PYTHON}" \
  ALLOW_NON_ROBOTWIN_CONDA="${ALLOW_NON_ROBOTWIN_CONDA}" \
  CLIENT_GPUS="${client_gpus}" \
  NUM_CLIENTS="${NUM_CLIENTS_PER_SERVER}" \
  WORLD_SIZE="${NUM_SERVERS}" \
  RANK="${SERVER_RANK}" \
  setsid bash "${SUB_LAUNCHER}" \
    > "${run_dir}/sub_launcher.log" 2>&1 &

  pid=$!
  GROUP_PIDS+=("${pid}")
  echo "[group ${SERVER_RANK}] sub-launcher PID=${pid} (log: ${run_dir}/sub_launcher.log)"
done

echo
echo "${#GROUP_PIDS[@]} client group(s) launched. Streaming group 0 client_0 log."
echo "Per-group logs: ${BASE_RUN_DIR}/server_<rank>/logs/"
echo "Press Ctrl-C to stop everything."
echo

# Mirror group 0's first client to the terminal so the user sees activity.
GROUP0_LOG="${BASE_RUN_DIR}/server_0/logs/grpo_client_0.log"
for _ in $(seq 1 60); do
  [[ -f "${GROUP0_LOG}" ]] && break
  sleep 0.5
done
if [[ -f "${GROUP0_LOG}" ]]; then
  tail -n +1 -F "${GROUP0_LOG}" &
  TAIL_PID=$!
else
  echo "(group 0 client_0 log not yet available; see ${BASE_RUN_DIR}/server_*/logs/)"
  TAIL_PID=""
fi

wait "${GROUP_PIDS[@]}" 2>/dev/null || true
if [[ -n "${TAIL_PID}" ]] && kill -0 "${TAIL_PID}" 2>/dev/null; then
  kill "${TAIL_PID}" 2>/dev/null || true
fi
