#!/bin/bash

# Training script for video generation model

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8

# Training configuration
CONFIG_PATH="configs/base_config.yaml"
NUM_GPUS=2

# Create directories
mkdir -p outputs logs checkpoints

# Run training with accelerate (multi-GPU support)
accelerate launch \
    --num_processes=$NUM_GPUS \
    --mixed_precision=fp16 \
    --multi_gpu \
    src/training/train.py \
    --config $CONFIG_PATH

echo "Training completed!" 