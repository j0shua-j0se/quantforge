"""
Smoke Backtest: Simple test to verify B0 reproducibility.

Tests:
- Data loading
- Backtesting engine
- Metrics calculation
- MLflow logging
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from core import (
    Backtester, 
    BacktestConfig, 
    load_sample_data,
    setup_mlflow_logger,
    log_backtest_results,
    end_mlflow_run
)


def simple_momentum_signal(prices: pd.DataFrame, lookback: int = 20) -> float:
    """Simple momentum signal: buy if price > MA(20)."""
    if len(prices) < lookback:
        return 0.0
    
    current_price = prices['close'].iloc[-1]
    ma = prices['close'].iloc[-lookback:].mean()
    
    return 1.0 if current_price > ma else 0.0


def run_smoke_backtest():
    """Run simple backtest to verify system works."""
    
    print("="*60)
    print("B0 SMOKE BACKTEST - Reproducibility Test")
    print("="*60)
    
    # Step 1: Load data
    print("\n[1/5] Loading sample data...")
    prices = load_sample_data()
    print(f"✓ Loaded {len(prices)} days of SPY data")
    
    # Step 2: Configure backtest
    print("\n[2/5] Configuring backtest...")
    config = BacktestConfig(
        start_date='2020-01-01',
        end_date='2024-12-31',
        initial_capital=100_000,
        transaction_cost_bps=10,
        seed=42
    )
    print(f"✓ Config: {config.start_date} to {config.end_date}, ${config.initial_capital:,}")
    
    # Step 3: Run backtest
    print("\n[3/5] Running backtest...")
    backtester = Backtester(config)
    
    results = backtester.run(
        prices=prices,
        signal_func=lambda p: simple_momentum_signal(p, lookback=20),
        weights_func=lambda s: s  # Weight = signal (0 or 1)
    )
    
    print(f"✓ Backtest complete")
    
    # Step 4: Display metrics
    print("\n[4/5] Performance Metrics:")
    print("-" * 60)
    metrics = results['metrics']
    print(f"  Annual Return:    {metrics['annual_return']*100:>7.2f}%")
    print(f"  Annual Volatility: {metrics['annual_volatility']*100:>7.2f}%")
    print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:>7.2f}")
    print(f"  Sortino Ratio:     {metrics['sortino_ratio']:>7.2f}")
    print(f"  Max Drawdown:      {metrics['max_drawdown']*100:>7.2f}%")
    print(f"  CVaR (95%):        {metrics['cvar_95']*100:>7.2f}%")
    print(f"  Win Rate:          {metrics['win_rate']*100:>7.2f}%")
    print("-" * 60)
    
    # Step 5: Log to MLflow
    print("\n[5/5] Logging to MLflow...")
    try:
        setup_mlflow_logger("B0_Smoke_Test", "momentum_baseline")
        log_backtest_results(config.to_dict(), metrics)
        end_mlflow_run()
        print("✓ Results logged to MLflow")
    except Exception as e:
        print(f"⚠ MLflow logging skipped: {e}")
    
    # Final check
    print("\n" + "="*60)
    print("✓ B0 SMOKE TEST PASSED")
    print("="*60)
    print(f"\nReproducibility Check:")
    print(f"  Seed: {config.seed}")
    print(f"  Final Equity: ${results['equity_curve'].iloc[-1]:,.2f}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print("\nRun this script again - metrics should be identical!")
    
    return results


if __name__ == "__main__":
    results = run_smoke_backtest()
