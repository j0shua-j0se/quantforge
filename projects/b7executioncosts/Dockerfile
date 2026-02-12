# B7 Execution Cost Modeling - Docker Image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.3.3 \
    pyyaml==6.0.2 \
    tqdm==4.66.0 \
    pyarrow==18.1.0

# Copy B7 code
COPY . /app/

# Create output directories
RUN mkdir -p outputs logs

# Set environment
ENV PYTHONUNBUFFERED=1

# Run pipeline by default
CMD ["python", "pipeline.py"]
