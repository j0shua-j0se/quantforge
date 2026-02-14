# B8 Policy Learning - Docker Image
# GPU-enabled PyTorch + Stable-Baselines3

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch with CUDA 11.8
RUN pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install RL and ML libraries
RUN pip install \
    stable-baselines3[extra]==2.2.1 \
    gymnasium==0.29.1 \
    pandas==2.1.4 \
    numpy==1.26.2 \
    pyyaml==6.0.1 \
    tqdm==4.66.1 \
    rich==13.7.0 \
    pyarrow==14.0.1 \
    tensorboard==2.15.1

# Copy B8 files
COPY config/ /app/config/
COPY models/ /app/models/
COPY train_ppo.py /app/
COPY backtest_b8.py /app/
COPY test_env.py /app/

# Create output directories
RUN mkdir -p /app/outputs /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command: run training
CMD ["python", "train_ppo.py"]
