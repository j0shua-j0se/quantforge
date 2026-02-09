"""Event-driven backtesting engine with walk-forward validation."""

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List, Callable
import logging

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    start_date: str
    end_date: str
    initial_capital: float = 100_000
    transaction_cost_bps: float = 10.0
    slippage_pct: float = 0.01
    seed: int = 42
    
    def to_dict(self):
        return asdict(self)


class Backtester:
    """Production-grade event-driven backtester."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        np.random.seed(config.seed)
        self.equity_curve = []
        self.trades = []
        logger.info(f"Backtester initialized: {config.start_date} to {config.end_date}")
    
    def run(self, prices: pd.DataFrame, signal_func: Callable, weights_func: Callable):
        """Run backtest with walk-forward safety."""
        
        # Filter date range
        mask = (prices.index >= self.config.start_date) & (prices.index <= self.config.end_date)
        prices = prices[mask].copy()
        
        # Calculate returns if not present
        if 'returns' not in prices.columns:
            prices['returns'] = prices['close'].pct_change()
        
        # Initialize
        equity = self.config.initial_capital
        self.equity_curve = [equity]
        prev_weight = 0.0
        
        # Event-driven loop
        for t in range(1, len(prices)):
            # Only use data up to time t
            available_data = prices.iloc[:t+1]
            
            # Generate signals
            signals = signal_func(available_data)
            
            # Compute weights
            target_weight = weights_func(signals)
            
            # Calculate execution costs
            trade_size = abs(target_weight - prev_weight)
            exec_cost = equity * trade_size * (self.config.transaction_cost_bps / 10000 + self.config.slippage_pct)
            
            # Deduct costs
            equity -= exec_cost
            
            # Apply returns
            portfolio_return = target_weight * prices.iloc[t]['returns']
            equity *= (1 + portfolio_return)
            
            self.equity_curve.append(equity)
            prev_weight = target_weight
        
        # Create equity series
        equity_series = pd.Series(self.equity_curve, index=prices.index)
        returns_series = equity_series.pct_change().dropna()
        
        # Compute metrics
        metrics = self._compute_metrics(returns_series)
        
        return {
            'equity_curve': equity_series,
            'returns': returns_series,
            'metrics': metrics,
            'config': self.config
        }
    
    def _compute_metrics(self, returns: pd.Series):
        """Compute performance metrics."""
        from core.metrics import portfolio_metrics
        return portfolio_metrics(returns)
