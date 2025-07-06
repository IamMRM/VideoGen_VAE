# Video Generation with Diffusion Transformers

A modern, scalable implementation of video generation using state-of-the-art Diffusion Transformer (DiT) architecture. This codebase is designed for training video generation models from scratch on dual GPU setups (2x 24GB VRAM).

## 🚀 Features

- **Modern Architecture**: Implements Diffusion Transformer (DiT) with spatio-temporal attention
- **Multi-GPU Training**: Optimized for dual 3090/A40 GPUs using Accelerate
- **Flexible Configuration**: YAML-based configuration system
- **Mixed Precision**: FP16/BF16 training support for memory efficiency
- **Advanced Sampling**: DDPM and DDIM sampling methods
- **Text Conditioning**: Optional CLIP text encoder integration
- **Efficient Data Pipeline**: Video caching, bucketing, and augmentation
- **Monitoring**: Integrated Weights & Biases and TensorBoard logging

## 📋 Requirements

- Python 3.8+
- CUDA 11.8+
- 2x GPUs with 24GB VRAM (e.g., RTX 3090, A40)
- ~100GB disk space for datasets and checkpoints

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/videogen_vae.git
cd videogen_vae

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install Flash Attention for better performance
pip install flash-attn --no-build-isolation
```

## 📊 Dataset Preparation

Prepare your video dataset with the following structure:

```
dataset/
├── video_001.mp4
├── video_001.txt  # Caption for video_001
├── video_002.mp4
├── video_002.txt
└── ...
```

Each `.txt` file should contain a text description of the corresponding video.

## 🎯 Training

### Configuration

Edit `configs/base_config.yaml` to customize:

- Model architecture (layers, dimensions, attention heads)
- Training parameters (learning rate, batch size, epochs)
- Data settings (resolution, FPS, duration)
- Hardware optimization settings

Key configuration options:

```yaml
model:
  architecture: "dit"  # Diffusion Transformer
  dim: 512            # Model dimension (configurable)
  depth: 24           # Number of transformer blocks
  num_frames: 480     # 20 seconds at 24 FPS
  frame_size: 512     # Square videos (can be 256 for faster training)

training:
  batch_size: 1       # Per GPU
  gradient_accumulation_steps: 4  # Effective batch size = 8
  num_epochs: 100
  mixed_precision: "fp16"  # or "bf16" for A40
```

### Start Training

```bash
# Single command to start training
bash scripts/train.sh

# Or manually with accelerate
accelerate launch --num_processes=2 --mixed_precision=fp16 \
    src/training/train.py --config configs/base_config.yaml
```

### Memory Optimization Tips

For 20-second videos at 512x512:
- Use gradient checkpointing (enabled by default)
- Set `gradient_accumulation_steps: 4` or higher
- Enable `cache_latents: true` for faster data loading
- Use Flash Attention if available

For lower memory usage:
- Reduce `frame_size` to 256
- Reduce `num_frames` (e.g., 240 for 10 seconds)
- Increase `gradient_accumulation_steps`

## 🎬 Inference

Generate videos using a trained checkpoint:

```bash
# Generate videos with example prompts
bash scripts/generate.sh

# Or manually
python src/inference/generate.py \
    --checkpoint checkpoints/best_model.pt \
    --prompts "A cat playing with yarn" "Ocean waves at sunset" \
    --num-videos 4 \
    --num-inference-steps 50 \
    --guidance-scale 7.5 \
    --output-dir generated_videos
```

### Inference Options

- `--num-inference-steps`: Number of denoising steps (50-100 recommended)
- `--guidance-scale`: Classifier-free guidance strength (7.5 default)
- `--format`: Output format (mp4, gif, webm)
- `--fp16`: Use half precision for faster generation
- `--seed`: Random seed for reproducibility

## 📁 Project Structure

```
videogen_vae/
├── configs/
│   └── base_config.yaml      # Main configuration file
├── src/
│   ├── models/
│   │   ├── dit.py           # Diffusion Transformer architecture
│   │   └── diffusion.py     # Diffusion process and sampling
│   ├── data/
│   │   └── dataset.py       # Video dataset and data loading
│   ├── training/
│   │   └── train.py         # Main training script
│   ├── inference/
│   │   └── generate.py      # Video generation script
│   └── utils/
│       ├── ema.py           # Exponential Moving Average
│       └── metrics.py       # Evaluation metrics
├── scripts/
│   ├── train.sh             # Training launch script
│   └── generate.sh          # Generation launch script
└── requirements.txt         # Python dependencies
```

## 🔧 Architecture Details

### Diffusion Transformer (DiT)

The model uses a transformer-based architecture optimized for video generation:

1. **Patch Embedding**: Videos are divided into spatio-temporal patches
2. **Positional Encoding**: Sinusoidal embeddings for time and spatial positions
3. **Transformer Blocks**: Self-attention with adaptive layer normalization
4. **Spatio-Temporal Attention**: Alternating spatial and temporal attention layers
5. **Output Projection**: Predicts noise or velocity for diffusion process

### Key Improvements

- **Flash Attention**: For efficient attention computation
- **Gradient Checkpointing**: Trade compute for memory
- **Mixed Precision**: FP16/BF16 training for efficiency
- **EMA**: Exponential moving average for stable generation
- **DDIM Sampling**: Fast inference with fewer steps

## 📊 Monitoring

Training progress is logged to:
- **TensorBoard**: `tensorboard --logdir logs`
- **Weights & Biases**: Automatic sync if configured

Metrics tracked:
- Training loss
- Learning rate
- Validation FVD (Fréchet Video Distance)
- Sample videos every N epochs

## 🚀 Performance Optimization

### Multi-GPU Training

The codebase uses Hugging Face Accelerate for efficient multi-GPU training:
- Automatic gradient synchronization
- Mixed precision training
- Gradient accumulation
- Distributed data loading

### Memory Management

- **Gradient Accumulation**: Simulate larger batches
- **CPU Offloading**: Move optimizer states to CPU if needed
- **Activation Checkpointing**: Recompute activations during backward pass
- **Flash Attention**: Reduce attention memory usage

## 🔍 Troubleshooting

### Out of Memory Errors

1. Reduce batch size to 1
2. Increase gradient accumulation steps
3. Reduce video resolution or duration
4. Enable gradient checkpointing
5. Use DeepSpeed ZeRO optimization

### Slow Training

1. Ensure Flash Attention is installed
2. Use cached datasets
3. Increase number of data loader workers
4. Use SSD for dataset storage
5. Profile with PyTorch profiler

## 📚 References

This implementation is inspired by:

- [Diffusion Transformers (DiT)](https://arxiv.org/abs/2212.09748)
- [ViD-GPT](https://arxiv.org/abs/2406.10981)
- [Stable Video Diffusion](https://stability.ai/news/stable-video-diffusion-open-ai-video-model)
- [Flash Attention](https://arxiv.org/abs/2205.14135)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions and support, please open an issue on GitHub. 