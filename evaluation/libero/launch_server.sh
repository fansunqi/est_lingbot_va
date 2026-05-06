
save_root='visualization/'
mkdir -p $save_root

python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port 29061 \
    -m src.inference.server \
    --config configs/inference/libero.yaml \
    --port 29056 \
    --save-root $save_root