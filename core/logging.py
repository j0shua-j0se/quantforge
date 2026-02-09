"""MLflow experiment tracking integration."""

import mlflow
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def setup_mlflow_logger(experiment_name: str, run_name: Optional[str] = None):
    """
    Setup MLflow for experiment tracking.
    
    Usage:
        setup_mlflow_logger("B0_Backtester", "momentum_test")
        mlflow.log_param("lookback", 20)
        mlflow.log_metric("sharpe", 1.25)
    """
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=run_name)
    logger.info(f"MLflow run started: {run.info.run_id}")
    return run


def log_backtest_results(config: Dict, metrics: Dict, artifacts_path: Optional[str] = None):
    """
    Log backtest configuration and results to MLflow.
    
    Parameters:
        config: Backtest configuration dict
        metrics: Performance metrics dict
        artifacts_path: Optional path to artifacts (plots, CSVs)
    """
    # Log configuration as parameters
    for key, value in config.items():
        if isinstance(value, (int, float, str, bool)):
            mlflow.log_param(key, value)
    
    # Log metrics
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            mlflow.log_metric(metric_name, metric_value)
    
    # Log artifacts if provided
    if artifacts_path:
        mlflow.log_artifacts(artifacts_path)
    
    logger.info(f"Results logged to MLflow")


def end_mlflow_run():
    """End current MLflow run."""
    mlflow.end_run()
    logger.info("MLflow run ended")
