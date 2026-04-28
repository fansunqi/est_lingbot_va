#!/usr/bin/bash

set -x

umask 007

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# VENV_PATH defaults to the in-repo .venv; override to a shared-storage venv
# for multi-node runs so the env only has to be built once.
VENV_PATH="${VENV_PATH:-${SCRIPT_DIR}/../.venv}"
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
    echo "[run_va_posttrain] no venv at ${VENV_PATH}; create one with:" >&2
    echo "    UV_PROJECT_ENVIRONMENT='${VENV_PATH}' uv sync" >&2
    exit 1
fi
source "${VENV_PATH}/bin/activate"

NGPU=${NGPU:-"8"}
MASTER_PORT=${MASTER_PORT:-"29501"}
PORT=${PORT:-"1106"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train"} # robotwin_train, libero_train

# Multi-node settings
NNODES=${NNODES:-"1"}
NODE_RANK=${NODE_RANK:-"0"}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

# SAVE_ROOT overrides config.save_root via train.py's --save-root. Prefer
# this over editing shared_config.py when launching multiple jobs in parallel.
if [[ -n "${SAVE_ROOT:-}" ]]; then
    overrides="--save-root ${SAVE_ROOT} ${overrides}"
fi

export WANDB_API_KEY="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"
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_TEAM_NAME="mo-zhehan"
export WANDB_PROJECT="lingbot-new"

## node setting
num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}

## cmd setting
export TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
python -m torch.distributed.run \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --nproc_per_node=${num_gpu} \
    --local-ranks-filter=${log_rank} \
    --master_port ${master_port} \
    --tee 3 \
    -m wan_va.train --config-name ${config_name} $overrides
