FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y python3-pip git

# Install PyTorch with CUDA 11.8
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Verify installation
CMD ["python3", "-c", "import torch; print(torch.cuda.is_available())"]