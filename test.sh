export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VENV_PATH=/home/zjp/anaconda3/envs/fa4 \
TASK=configs/tasks/train_test.yaml \
bash scripts/run_va_posttrain.sh
#python -m src.run --task configs/tasks/train_test.yaml