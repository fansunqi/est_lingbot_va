#!/bin/bash
# ============================================================
# Launch evaluation inside Docker container
# Usage: ./launch_eval.sh [task_name]
# ============================================================
set -e

TASK_NAME=${1:-"adjust_bottle"}
SAVE_ROOT=${2:-"./results"}

echo "============================================"
echo " lingbot-va Evaluation (Docker)"
echo " Task: $TASK_NAME"
echo "============================================"

# --- Start Server (lingbot-va, uv env) ---
echo "[1/2] Starting server..."

# Use the Docker venv (Python 3.12 + cu124)
export PATH="/workspace/lingbot-va/.venv-docker/bin:$PATH"

cd /workspace/lingbot-va

# Launch server in background via tmux
tmux new-session -d -s lb_server "
cd /workspace/lingbot-va && \
/workspace/lingbot-va/.venv-docker/bin/python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port 29061 \
    -m src.inference.server \
    --config configs/inference/robotwin.yaml \
    --port 29056 \
    --save-root visualization/
"

echo "  Server starting in tmux session 'lb_server'..."
echo "  Waiting 30s for model loading..."
sleep 30

# --- Start Client (RoboTwin, conda env) ---
echo "[2/2] Starting client..."

eval "$(conda shell.bash hook)"
conda activate RoboTwin

export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH
export no_proxy=localhost,127.0.0.1,0.0.0.0
export NO_PROXY=localhost,127.0.0.1,0.0.0.0

# Add RoboTwin to PYTHONPATH
export PYTHONPATH="/workspace/RoboTwin:$PYTHONPATH"
export ROBOTWIN_ROOT="/workspace/RoboTwin"

cd /workspace/lingbot-va

PYTHONWARNINGS=ignore::UserWarning \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python -m evaluation.robotwin.eval_polict_client_openpi \
    --config policy/ACT/deploy_policy.yml \
    --overrides \
    --task_name ${TASK_NAME} \
    --task_config demo_clean \
    --train_config_name 0 \
    --model_name 0 \
    --ckpt_setting 0 \
    --seed 0 \
    --policy_name ACT \
    --save_root ${SAVE_ROOT} \
    --video_guidance_scale 5 \
    --action_guidance_scale 1 \
    --test_num 100 \
    --port 29056

echo "============================================"
echo " Evaluation complete!"
echo "============================================"
