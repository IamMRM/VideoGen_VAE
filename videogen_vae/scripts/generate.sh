#!/bin/bash

# Video generation script

# Configuration
CHECKPOINT_PATH="checkpoints/best_model.pt"
OUTPUT_DIR="generated_videos"
NUM_VIDEOS=4
NUM_INFERENCE_STEPS=50

# Example prompts
PROMPTS=(
    "A serene beach with waves gently crashing on the shore"
    "A bustling city street at night with neon lights"
    "A forest path with sunlight filtering through trees"
    "A cat playing with a ball of yarn"
)

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
    --fp16 \
    --seed 42

echo "Video generation completed! Check $OUTPUT_DIR for results." 