"""QuantForge Core: Shared utilities for all B0-B11 projects."""

__version__ = "0.0.1"

from core.backtester import Backtester, BacktestConfig
from core.metrics import portfolio_metrics, compute_cvar, compute_max_drawdown
from core.execution import ExecutionModel
from core.logging import setup_mlflow_logger, log_backtest_results, end_mlflow_run
from core.data import DataManager, load_sample_data

__all__ = [
    "Backtester",
    "BacktestConfig",
    "portfolio_metrics",
    "compute_cvar",
    "compute_max_drawdown",
    "ExecutionModel",
    "setup_mlflow_logger",
    "log_backtest_results",
    "end_mlflow_run",
    "DataManager",
    "load_sample_data",
]
