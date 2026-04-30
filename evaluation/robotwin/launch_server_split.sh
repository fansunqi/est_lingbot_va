#!/bin/bash
# Launch N servers on consecutive GPUs starting from SERVER_GPU_START.
# Default: 6 servers on GPU 0..5.
# Each server holds the wan_va model and serves one client over a TCP port.

START_PORT=${START_PORT:-29556}
MASTER_PORT=${MASTER_PORT:-29661}
NUM_SERVERS=${NUM_SERVERS:-5}
SERVER_GPU_START=${SERVER_GPU_START:-0}

LOG_DIR='./logs'
mkdir -p "$LOG_DIR"
save_root='./visualization/'
mkdir -p "$save_root"

batch_time=$(date +%Y%m%d_%H%M%S)

echo -e "\033[32mLaunching ${NUM_SERVERS} servers on GPU ${SERVER_GPU_START}..$((SERVER_GPU_START + NUM_SERVERS - 1)).\033[0m"

for i in $(seq 0 $((NUM_SERVERS - 1))); do
    GPU=$((SERVER_GPU_START + i))
    CURRENT_PORT=$((START_PORT + i))
    CURRENT_MASTER_PORT=$((MASTER_PORT + i))
    LOG_FILE="${LOG_DIR}/server_${i}_${batch_time}.log"

    echo -e "\033[33m[Server ${i}] GPU=${GPU} PORT=${CURRENT_PORT} MASTER_PORT=${CURRENT_MASTER_PORT} LOG=${LOG_FILE}\033[0m"

    CUDA_VISIBLE_DEVICES=$GPU \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup python -u -m torch.distributed.run \
        --nproc_per_node 1 \
        --master_port $CURRENT_MASTER_PORT \
        wan_va/wan_va_server.py \
        --config-name robotwin \
        --save_root "$save_root" \
        --port $CURRENT_PORT > "$LOG_FILE" 2>&1 &
    sleep 2
done

echo -e "\033[32mAll ${NUM_SERVERS} servers launched in background.\033[0m"
wait
