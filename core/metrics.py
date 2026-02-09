"""Performance and risk metrics with CVaR focus."""

import numpy as np
import pandas as pd
from typing import Dict


def compute_cvar(returns: np.ndarray, alpha: float = 0.95) -> float:
    """
    Compute Conditional Value-at-Risk (Expected Shortfall).
    
    CVaR = average of worst (1-alpha)% returns
    """
    var_threshold = np.percentile(returns, (1 - alpha) * 100)
    cvar = returns[returns <= var_threshold].mean()
    return cvar


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """Compute maximum drawdown."""
    running_max = equity_curve.expanding().max()
    drawdowns = (equity_curve - running_max) / running_max
    max_dd = drawdowns.min()
    return max_dd


def portfolio_metrics(returns: pd.Series, risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Comprehensive portfolio performance metrics.
    
    Returns dict with: annual_return, sharpe_ratio, cvar_95, max_drawdown, etc.
    """
    # Annualization
    periods_per_year = 252
    
    # Basic metrics
    annual_return = returns.mean() * periods_per_year
    annual_vol = returns.std() * np.sqrt(periods_per_year)
    
    # Sharpe ratio
    excess_return = annual_return - risk_free_rate
    sharpe_ratio = excess_return / annual_vol if annual_vol > 0 else 0
    
    # Sortino ratio
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(periods_per_year)
    sortino_ratio = excess_return / downside_std if downside_std > 0 else 0
    
    # CVaR
    cvar_95 = compute_cvar(returns.values, alpha=0.95)
    cvar_99 = compute_cvar(returns.values, alpha=0.99)
    
    # Maximum drawdown
    equity_curve = (1 + returns).cumprod()
    max_drawdown = compute_max_drawdown(equity_curve)
    
    # Calmar ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Additional metrics
    win_rate = (returns > 0).sum() / len(returns)
    
    return {
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'cvar_95': cvar_95,
        'cvar_99': cvar_99,
        'win_rate': win_rate,
        'total_periods': len(returns)
    }
