#!/usr/bin/bash
#
# Thin launcher for src.run. Lightning's DDPStrategy spawns the worker
# subprocesses by itself (via _SubprocessScriptLauncher), so this script
# just activates the venv, sets a few useful env vars, and execs python.
#
# All training knobs live in the YAML at $TASK. CLI flags can override:
#
#   TASK=configs/tasks/train_test.yaml \
#   bash scripts/run_va_posttrain.sh --devices 1 --wandb-mode disabled
#
# Or directly without this wrapper (after activating the venv):
#
#   python -m src.run --task configs/tasks/train_test.yaml --devices 1
#
# Env vars:
#   VENV_PATH    Python env to activate. Accepts a uv venv (with bin/activate)
#                or a conda env prefix (just bin/python). Defaults to .venv.
#   TASK         Path to a YAML task config (default: train_robotwin.yaml).
#   SAVE_ROOT    Optional override for config['save_root']. Forwarded as
#                --save-root to src.run.
#
# Multi-node (set by run_va_posttrain_multinode.sh):
#   NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT — exported for Lightning's
#   ClusterEnvironment to pick up.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV_PATH="${VENV_PATH:-${SCRIPT_DIR}/../.venv}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_PATH}/bin/activate"
elif [[ -x "${VENV_PATH}/bin/python" ]]; then
    export PATH="${VENV_PATH}/bin:${PATH}"
    export VIRTUAL_ENV="${VENV_PATH}"
else
    echo "[run_va_posttrain] no usable env at ${VENV_PATH}; create one with:" >&2
    echo "    UV_PROJECT_ENVIRONMENT='${VENV_PATH}' uv sync" >&2
    echo "  or point VENV_PATH at a conda env prefix containing bin/python." >&2
    exit 1
fi

TASK=${TASK:-"configs/tasks/train_robotwin.yaml"}

# SAVE_ROOT overrides config['save_root'] via src.run's --save-root. Prefer
# this over editing the YAML when launching multiple jobs in parallel.
overrides=()
if [[ -n "${SAVE_ROOT:-}" ]]; then
    overrides+=(--save-root "${SAVE_ROOT}")
fi
overrides+=("$@")

# WandB credentials — overridden by user shell if pre-exported.
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_TEAM_NAME="${WANDB_TEAM_NAME:-Robotics-FiT}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Disable NVLink SHARP multicast — fails with CUDA error 401 on nodes whose
# Fabric Manager / NVSwitch state isn't NVLS-ready. Override with
# NCCL_NVLS_ENABLE=1 if your cluster supports it.
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

# Multi-node coordinates (Lightning's ClusterEnvironment reads these directly).
# Defaults are sane for single-node runs.
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29501}

set -x
exec python -m src.run --task "${TASK}" "${overrides[@]}"
