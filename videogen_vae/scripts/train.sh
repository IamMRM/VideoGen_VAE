#!/bin/bash

# Training script for video generation model

# Set environment variables for optimal GPU usage
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export TF_ENABLE_ONEDNN_OPTS=0

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi

# Training configuration
CONFIG_PATH="configs/base_config.yaml"
NUM_GPUS=2

# Create directories
mkdir -p outputs logs checkpoints

# Run training with accelerate (FSDP for better GPU utilization)
accelerate launch \
    --num_processes=$NUM_GPUS \
    --mixed_precision=bf16 \
    --multi_gpu \
    --machine_rank=0 \
    --main_process_port=29500 \
    src/training/train.py \
    --config $CONFIG_PATH

echo "Training completed!" 