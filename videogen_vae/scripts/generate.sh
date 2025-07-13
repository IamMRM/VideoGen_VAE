#!/bin/bash

# Video generation script

# Configuration
CHECKPOINT_PATH="checkpoints/checkpoint_epoch_0099.pt"
OUTPUT_DIR="generated_videos"
NUM_VIDEOS=2
NUM_INFERENCE_STEPS=1000

# Example prompts
PROMPTS=("")

# Create output directory
mkdir -p $OUTPUT_DIR

# Generate videos
python src/inference/generate.py \
    --checkpoint $CHECKPOINT_PATH \
    --prompts "${PROMPTS[@]}" \
    --num-videos $NUM_VIDEOS \
    --output-dir $OUTPUT_DIR \
    --num-inference-steps $NUM_INFERENCE_STEPS \
    --guidance-scale 7.5 \
    --fps 24 \
    --format mp4 \
    --bf16 \
    --seed 42

echo "Video generation completed! Check $OUTPUT_DIR for results." 