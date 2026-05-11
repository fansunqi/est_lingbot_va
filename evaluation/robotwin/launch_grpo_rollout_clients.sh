#!/usr/bin/env bash
set -euo pipefail

ASSIGNMENT=${ASSIGNMENT:?Set ASSIGNMENT to a RoboTwin assignment JSON}
PORT=${PORT:-29546}
HOST=${HOST:-127.0.0.1}
SAVE_ROOT=${SAVE_ROOT:-./results/grpo}
RUN_DIR=${RUN_DIR:-${SAVE_ROOT}/$(date +%Y%m%d_%H%M%S)}
GROUP_SIZE=${GROUP_SIZE:-2}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
PYTHON=${PYTHON:-python}
SKIP_RENDER_CHECK=${SKIP_RENDER_CHECK:-1}
CLIENT_GPUS=(${CLIENT_GPUS:-0})
NUM_CLIENTS=${NUM_CLIENTS:-${#CLIENT_GPUS[@]}}

ASSIGNMENT=$(realpath "${ASSIGNMENT}")
SAVE_ROOT=$(realpath -m "${SAVE_ROOT}")
RUN_DIR=$(realpath -m "${RUN_DIR}")

export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:${LD_LIBRARY_PATH:-}
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

for CLIENT_ID in $(seq 0 $((NUM_CLIENTS - 1))); do
  gpu_id="${CLIENT_GPUS[$(( CLIENT_ID % ${#CLIENT_GPUS[@]} ))]}"
  echo "[GRPO rollout client ${CLIENT_ID}] GPU=${gpu_id} PORT=${PORT}"

  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  PYTHONUNBUFFERED=1 \
  PYTHONWARNINGS=ignore::UserWarning \
  nohup "${PYTHON}" -u -m evaluation.robotwin.grpo_rollout_client \
    --assignment "${ASSIGNMENT}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --save_root "${SAVE_ROOT}" \
    --run_dir "${RUN_DIR}" \
    --client_id "${CLIENT_ID}" \
    --group_size "${GROUP_SIZE}" \
    --task_config "${TASK_CONFIG}" \
    $([[ "${SKIP_RENDER_CHECK}" == "1" ]] && printf '%s' '--skip_render_check') \
    > "${RUN_DIR}/logs/grpo_client_${CLIENT_ID}.log" 2>&1 &
  echo $! >> "${RUN_DIR}/grpo_client_pids.txt"
done

echo "GRPO rollout clients launched. Logs: ${RUN_DIR}/logs"
