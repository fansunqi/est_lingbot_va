#!/usr/bin/env bash
set -euo pipefail

# Foreground launcher for GRPO rollout clients. Each client runs in its own
# process group (via setsid) so Ctrl-C / SIGTERM / SIGHUP on this script
# tears down every client (and any RoboTwin/SAPIEN subprocesses they spawn)
# in one shot.

ASSIGNMENT=${ASSIGNMENT:?Set ASSIGNMENT to a RoboTwin assignment JSON}
PORT=${PORT:-29546}
HOST=${HOST:-127.0.0.1}
SAVE_ROOT=${SAVE_ROOT:-./results/grpo}
RUN_DIR=${RUN_DIR:-${SAVE_ROOT}/$(date +%Y%m%d_%H%M%S)}
GROUP_SIZE=${GROUP_SIZE:-2}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
PYTHON=${PYTHON:-python}
SKIP_RENDER_CHECK=${SKIP_RENDER_CHECK:-1}
GROUP_BARRIER=${GROUP_BARRIER:-0}
GROUP_BARRIER_TIMEOUT=${GROUP_BARRIER_TIMEOUT:-0}
SAVE_VISUALIZATION=${SAVE_VISUALIZATION:-none}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-}
NUM_PASSES=${NUM_PASSES:-1}
EVAL_ONLY=${EVAL_ONLY:-0}
CLIENT_GPUS=(${CLIENT_GPUS:-0})
NUM_CLIENTS=${NUM_CLIENTS:-${#CLIENT_GPUS[@]}}
# Sharded-group GRPO indexing. WORLD_SIZE = number of GRPO server ranks the
# rollout pool spans (1 for single-server runs). RANK = which rank this
# launcher binds to. The DDP top-level wrapper sets both per group; standalone
# runs leave them at the defaults.
WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}

ASSIGNMENT=$(realpath "${ASSIGNMENT}")
SAVE_ROOT=$(realpath -m "${SAVE_ROOT}")
RUN_DIR=$(realpath -m "${RUN_DIR}")

export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:${LD_LIBRARY_PATH:-}
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
fi
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0}
unset VK_ICD_FILENAMES
export no_proxy=localhost,127.0.0.1,0.0.0.0
export NO_PROXY=localhost,127.0.0.1,0.0.0.0

# RoboTwin/SAPIEN dependencies live in the conda RoboTwin environment. Do not
# launch these clients with uv; the LingBot model server is a separate process.
if [[ "${CONDA_DEFAULT_ENV:-}" != "RoboTwin" && -z "${ALLOW_NON_ROBOTWIN_CONDA:-}" ]]; then
  echo "Expected conda environment 'RoboTwin' for rollout clients." >&2
  echo "Run: conda activate RoboTwin" >&2
  echo "Set ALLOW_NON_ROBOTWIN_CONDA=1 only if your RoboTwin env has another name." >&2
  exit 1
fi

mkdir -p "${RUN_DIR}/logs"
echo "GRPO rollout settings: GROUP_SIZE=${GROUP_SIZE} NUM_CLIENTS=${NUM_CLIENTS} WORLD_SIZE=${WORLD_SIZE} RANK=${RANK} NUM_PASSES=${NUM_PASSES} EVAL_ONLY=${EVAL_ONLY} CLIENT_GPUS=${CLIENT_GPUS[*]}"

CLIENT_PIDS=()
CLIENT_PGIDS=()
TAIL_PID=""
SHUTTING_DOWN=0

cleanup() {
  if [[ "${SHUTTING_DOWN}" == "1" ]]; then
    return
  fi
  SHUTTING_DOWN=1
  trap - INT TERM HUP EXIT
  echo
  echo "[launch_grpo_rollout_clients] Caught shutdown signal, terminating ${#CLIENT_PGIDS[@]} client(s)..."

  if [[ -n "${TAIL_PID}" ]] && kill -0 "${TAIL_PID}" 2>/dev/null; then
    kill "${TAIL_PID}" 2>/dev/null || true
  fi

  # SIGTERM every client's process group.
  for pgid in "${CLIENT_PGIDS[@]}"; do
    kill -TERM -- "-${pgid}" 2>/dev/null || true
  done

  # Wait up to 10s for graceful exit.
  for _ in $(seq 1 20); do
    any_alive=0
    for pgid in "${CLIENT_PGIDS[@]}"; do
      if kill -0 -- "-${pgid}" 2>/dev/null; then
        any_alive=1
        break
      fi
    done
    if [[ "${any_alive}" == "0" ]]; then
      break
    fi
    sleep 0.5
  done

  # SIGKILL any survivors.
  for pgid in "${CLIENT_PGIDS[@]}"; do
    if kill -0 -- "-${pgid}" 2>/dev/null; then
      echo "[launch_grpo_rollout_clients] Client group ${pgid} still alive after SIGTERM; sending SIGKILL."
      kill -KILL -- "-${pgid}" 2>/dev/null || true
    fi
  done

  wait 2>/dev/null || true
  echo "[launch_grpo_rollout_clients] Shutdown complete."
}

trap cleanup INT TERM HUP
trap cleanup EXIT

for CLIENT_ID in $(seq 0 $((NUM_CLIENTS - 1))); do
  gpu_id="${CLIENT_GPUS[$(( CLIENT_ID % ${#CLIENT_GPUS[@]} ))]}"
  log_file="${RUN_DIR}/logs/grpo_client_${CLIENT_ID}.log"
  echo "[GRPO rollout client ${CLIENT_ID}] GPU=${gpu_id} PORT=${PORT} LOG=${log_file}"

  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  PYTHONUNBUFFERED=1 \
  PYTHONWARNINGS=ignore::UserWarning \
  setsid "${PYTHON}" -u -m evaluation.robotwin.grpo_rollout_client \
    --assignment "${ASSIGNMENT}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --save_root "${SAVE_ROOT}" \
    --run_dir "${RUN_DIR}" \
    --client_id "${CLIENT_ID}" \
    --group_size "${GROUP_SIZE}" \
    --num_clients "${NUM_CLIENTS}" \
    --world_size "${WORLD_SIZE}" \
    --rank "${RANK}" \
    --num_passes "${NUM_PASSES}" \
    $([[ "${EVAL_ONLY}" == "1" ]] && printf '%s' '--eval_only') \
    --task_config "${TASK_CONFIG}" \
    --save_visualization "${SAVE_VISUALIZATION}" \
    $([[ -n "${MAX_EPISODE_STEPS}" ]] && printf -- '--max_episode_steps %s' "${MAX_EPISODE_STEPS}") \
    $([[ "${GROUP_BARRIER}" == "1" ]] && printf '%s' '--group_barrier') \
    --group_barrier_timeout "${GROUP_BARRIER_TIMEOUT}" \
    $([[ "${SKIP_RENDER_CHECK}" == "1" ]] && printf '%s' '--skip_render_check') \
    > "${log_file}" 2>&1 &
  pid=$!
  # setsid makes the child its own session/group leader → PGID == PID.
  CLIENT_PIDS+=("${pid}")
  CLIENT_PGIDS+=("${pid}")
  echo "${pid}" >> "${RUN_DIR}/grpo_client_pids.txt"
done

echo "${#CLIENT_PIDS[@]} GRPO rollout clients launched. Logs: ${RUN_DIR}/logs"
echo "Streaming client_0 log; other clients' logs are in the directory above."
echo "Press Ctrl-C to stop all clients and exit."
echo

# Mirror only client 0 to keep the terminal readable; the others log to disk.
tail -n +1 -F "${RUN_DIR}/logs/grpo_client_0.log" &
TAIL_PID=$!

# Wait until every client exits. If any client exits non-zero we still let
# the others run — the user decides when to Ctrl-C; cleanup() handles the rest.
wait "${CLIENT_PIDS[@]}" 2>/dev/null || true
