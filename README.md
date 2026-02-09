\# QuantForge: Execution-Aware Portfolio Policy Learning



\[!\[Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)

\[!\[Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)

\[!\[MLflow](https://img.shields.io/badge/mlflow-tracking-orange.svg)](https://mlflow.org/)

\[!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



> \*\*B0: QuantReproLab\*\* - Reproducible quantitative finance research infrastructure implementing walk-forward backtesting, CVaR risk metrics, and MLflow experiment tracking.



---



\## 🎯 Project Overview



QuantForge is a Master's thesis project (M.Sc. AI) building an execution-aware, regime-conditioned portfolio optimization system across 11 modular sub-projects (B0-B11).



\*\*B0 Status:\*\* ✅ \*\*Complete\*\* - Production-ready backtesting infrastructure with Docker containerization and professional logging.



\### Key Features



\- ✅ \*\*Event-Driven Backtester\*\* - Walk-forward safe, no look-ahead bias

\- ✅ \*\*CVaR Risk Metrics\*\* - Tail risk measurement (95%, 99% CVaR)

\- ✅ \*\*Execution Cost Modeling\*\* - Spread + slippage + market impact

\- ✅ \*\*MLflow Tracking\*\* - Comprehensive experiment logging

\- ✅ \*\*Docker Containerization\*\* - Reproducible environments

\- ✅ \*\*Data Versioning\*\* - Snapshot management with manifests

\- ✅ \*\*100% Reproducible\*\* - Fixed seed (SEED=42) for identical results



---



\## 🚀 Quick Start



\### Prerequisites



\- Python 3.10+

\- Conda (Miniconda/Anaconda)

\- Docker Desktop (for containerized execution)



\### Installation



\#### Option 1: Conda Environment (Recommended for Development)



```bash

\# Clone repository

git clone https://github.com/j0shua-j0se/quantforge.git

cd quantforge



\# Create environment

conda env create -f environment.yml

conda activate quantforge



\# Install package

pip install -e .



\# Run smoke test

python projects/b0\_quantreprolab/smoke\_backtest.py



