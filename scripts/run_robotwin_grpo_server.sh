#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/rl/robotwin_grpo.yaml}
PORT=${PORT:-29546}
SAVE_ROOT=${SAVE_ROOT:-experiments/robotwin_grpo}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
UV=${UV:-uv}

# LingBot model dependencies live in this repository's uv environment. The
# RoboTwin/SAPIEN environment should run only the rollout clients.
if ! command -v "${UV}" >/dev/null 2>&1; then
  echo "Could not find uv. Run this server script from the LingBot uv environment." >&2
  exit 1
fi

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  if [[ "${ALLOW_GRPO_FSDP:-0}" != "1" ]]; then
    echo "GRPO rollout currently defaults to split/offload, not FSDP." >&2
    echo "Use scripts/run_robotwin_grpo_server_split.sh for one offloaded server per GPU." >&2
    echo "Set ALLOW_GRPO_FSDP=1 only when intentionally testing FSDP GRPO." >&2
    exit 1
  fi
  "${UV}" run torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m src.rl.server \
    --config "${CONFIG}" \
    --port "${PORT}" \
    --save-root "${SAVE_ROOT}" "$@"
else
  "${UV}" run python -u -m src.rl.server \
    --config "${CONFIG}" \
    --port "${PORT}" \
    --save-root "${SAVE_ROOT}" "$@"
fi
