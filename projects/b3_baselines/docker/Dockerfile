# B3 Baseline Strategies - Docker Image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    pandas==2.2.3 \
    numpy==2.2.6 \
    pyyaml==6.0.2 \
    pyarrow==18.1.0

# Copy application code
COPY core/ /app/core/
COPY projects/b3_baselines/ /app/projects/b3_baselines/

# Create output directory
RUN mkdir -p /app/projects/b3_baselines/outputs

# Default command
CMD ["python", "projects/b3_baselines/pipeline.py"]

# Metadata
LABEL maintainer="joshua.jose2002@gmail.com"
LABEL description="B3 Baseline Strategies - QuantForge"
LABEL version="0.4.0"
