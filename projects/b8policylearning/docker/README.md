```markdown

\# B8 Docker Deployment Guide



\## Prerequisites



\### 1. Docker Desktop with GPU Support

```bash

\# Windows: Install Docker Desktop with WSL2

\# Enable NVIDIA GPU support in Docker Desktop settings

```



\### 2. NVIDIA Container Toolkit

```bash

\# Verify GPU is accessible

docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

```



---



\## Quick Start



\### Build Image

```bash

cd projects/b8policylearning

docker-compose build

```



\*\*Build time:\*\* ~5-8 minutes (downloads 2.6 GB PyTorch)



\### Run Training (GPU)

```bash

docker-compose up b8-training

```



\*\*Expected output:\*\*

```

b8-training | B8 POLICY LEARNING - PPO TRAINING

b8-training | Device: cuda

b8-training | Training PPO agent...

b8-training | 100% ━━━━━━━━━━━━━━━━━━━━━━━

b8-training | ✅ B8 TRAINING COMPLETE!

```



\### Run Backtest

```bash

docker-compose run --rm b8-backtest

```



\### Test Environment

```bash

docker-compose run --rm b8-training python test\_env.py

```



---



\## Volume Mounts



| Host Path | Container Path | Mode | Purpose |

|-----------|----------------|------|---------|

| `../../data` | `/app/data` | ro | B0-B7 data |

| `./outputs` | `/app/outputs` | rw | Models \& results |

| `./logs` | `/app/logs` | rw | TensorBoard logs |



---



\## GPU Configuration



\### Check GPU Access

```bash

docker-compose run --rm b8-training python -c "import torch; print('CUDA:', torch.cuda.is\_available())"

```



\*\*Expected:\*\* `CUDA: True`



\### Monitor GPU Usage

```bash

\# In another terminal while training

nvidia-smi -l 1

```



---



\## Troubleshooting



\### GPU Not Detected

```bash

\# 1. Enable GPU in Docker Desktop

\# Settings → Resources → WSL Integration → Enable GPU



\# 2. Verify nvidia-docker

docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi



\# 3. Restart Docker Desktop

```



\### Out of Memory

```bash

\# Reduce batch size in config/ppo\_config.yaml

batch\_size: 32  # was 64

```



\### Permission Errors

```bash

\# Windows: Run PowerShell as Administrator

\# Or set folder permissions

icacls outputs /grant Everyone:F

```



---



\## Image Details



\*\*Base Image:\*\* `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`  

\*\*Python:\*\* 3.10  

\*\*PyTorch:\*\* 2.1.0+cu118  

\*\*Image Size:\*\* ~4.5 GB  



---



\## Advanced Usage



\### Custom Config

```bash

docker-compose run --rm b8-training python train\_ppo.py --config /app/config/custom.yaml

```



\### Interactive Shell

```bash

docker-compose run --rm b8-training bash

```



\### TensorBoard

```bash

docker-compose run --rm -p 6006:6006 b8-training tensorboard --logdir /app/logs --host 0.0.0.0

```



Then open: `http://localhost:6006`



---



\## Cleanup



```bash

\# Remove containers

docker-compose down



\# Remove image

docker rmi quantforge/b8-ppo:latest



\# Remove volumes

docker volume prune

```



---



\*\*Status:\*\* ✅ GPU-enabled Docker deployment ready

```



\*\*Save (Ctrl+S) and close\*\*



\*\*\*



\## \*\*Step 16D: Test Docker Build\*\*



```powershell

\# Build the image (takes 5-8 minutes first time)

docker-compose build

```



\*\*Expected output:\*\*

```

\[+] Building 320.5s (15/15) FINISHED

&nbsp;=> \[internal] load build definition

&nbsp;=> => transferring dockerfile: 1.2kB

&nbsp;=> \[internal] load .dockerignore

&nbsp;...

&nbsp;=> exporting to image

&nbsp;=> => naming to docker.io/quantforge/b8-ppo:latest

```

