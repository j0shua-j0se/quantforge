# B6 CVaR Portfolio Optimizer - GPU-Enabled Docker Image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Metadata
LABEL maintainer="joshua.jose2002@gmail.com"
LABEL description="B6 Hybrid RL-Ensemble Portfolio Optimizer"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip3 install torch==2.5.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Install RL libraries
RUN pip3 install stable-baselines3==2.3.2 \
    sb3-contrib==2.3.0 \
    gymnasium==0.29.1 \
    tensorboard==2.18.0

# Copy project files
COPY . .

# Create output directories
RUN mkdir -p outputs logs/tensorboard

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Expose TensorBoard port
EXPOSE 6006

# Default command: run pipeline
CMD ["python3", "pipeline.py"]
