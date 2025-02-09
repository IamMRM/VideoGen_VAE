FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# Install sudo
USER root
RUN apt-get update && \
    apt-get install -y sudo git

WORKDIR /workspace

# Optional: Create a non-root user with sudo privileges
RUN useradd -ms /bin/bash devuser && \
    echo "devuser:devpassword" | chpasswd && \
    adduser devuser sudo && \
    adduser devuser video

USER devuser
# Copy the requirements.txt and install Python packages
COPY ../diffusion-pipe/requirements.txt /workspace/
RUN pip install -r requirements.txt


# Verify installation
#CMD ["python3", "-c", "import torch; print(torch.cuda.is_available())"]