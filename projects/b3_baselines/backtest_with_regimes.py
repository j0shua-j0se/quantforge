"""
B3+B4 Integration - Backtest Comparison
Compare momentum with and without regime conditioning
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from regime_strategy import RegimeConditionedMomentum, load_latest_regime_labels

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def simple_backtest(signals: pd.DataFrame, prices: pd.DataFrame, initial_capital=1000000, 
                    commission_pct=0.0005, slippage_pct=0.0003):
    """
    Simple backtester for comparison
    
    Args:
        signals: DataFrame with date, ticker, signal columns
        prices: DataFrame with date, ticker, close columns
        initial_capital: Starting cash
        commission_pct: Commission per trade
        slippage_pct: Slippage per trade
    
    Returns:
        Dictionary with performance metrics
    """
    logger.info("Running backtest...")
    
    # Merge signals with prices
    merged = signals.merge(prices[['date', 'ticker', 'close']], on=['date', 'ticker'], how='inner')
    
    # Calculate daily returns for each position
    merged['returns'] = merged.groupby('ticker')['close'].pct_change()
    
    # Apply transaction costs
    total_cost = commission_pct + slippage_pct
    merged['net_returns'] = merged['returns'] * merged['signal'] - total_cost * abs(merged['signal'])
    
    # Calculate portfolio returns (equal-weighted among active positions)
    daily_returns = merged.groupby('date').apply(
        lambda x: x['net_returns'].mean() if (x['signal'] != 0).any() else 0
    )
    
    # Calculate equity curve
    equity = initial_capital * (1 + daily_returns).cumprod()
    
    # Calculate metrics
    total_return = (equity.iloc[-1] / initial_capital - 1) * 100
    annual_return = ((equity.iloc[-1] / initial_capital) ** (252 / len(equity)) - 1) * 100
    volatility = daily_returns.std() * np.sqrt(252) * 100
    sharpe = annual_return / volatility if volatility > 0 else 0
    
    # Max drawdown
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax * 100
    max_drawdown = drawdown.min()
    
    # Count trades
    n_trades = (merged['signal'] != 0).sum()
    
    results = {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'n_trades': n_trades,
        'equity_curve': equity
    }
    
    return results


def main():
    logger.info("=" * 70)
    logger.info("B3+B4 INTEGRATION - REGIME-CONDITIONED BACKTEST")
    logger.info("=" * 70)
    
    # Load data
    logger.info("\nStep 1: Loading data...")
    features = pd.read_parquet('data/features/features_20260212.parquet')
    logger.info(f"Loaded features: {len(features)} rows")
    
    # Load regime labels
    regimes = load_latest_regime_labels()
    
    # Prepare price data
    prices = features[['date', 'ticker', 'close']].copy()
    
    # Strategy 1: Regular Momentum (no regime filter)
    logger.info("\n" + "=" * 70)
    logger.info("STRATEGY 1: Regular Momentum (No Regime Filter)")
    logger.info("=" * 70)
    
    regular_strategy = RegimeConditionedMomentum(allowed_regimes=[0, 1, 2])  # All regimes
    regular_signals = regular_strategy.generate_signals(features, regimes)
    regular_results = simple_backtest(regular_signals, prices)
    
    # Strategy 2: Regime-Conditioned Momentum (only Regime 0 - Expansion)
    logger.info("\n" + "=" * 70)
    logger.info("STRATEGY 2: Regime-Conditioned Momentum (Regime 0 Only)")
    logger.info("=" * 70)
    
    regime_strategy = RegimeConditionedMomentum(allowed_regimes=[0])  # Only expansion
    regime_signals = regime_strategy.generate_signals(features, regimes)
    regime_results = simple_backtest(regime_signals, prices)
    
    # Comparison
    logger.info("\n" + "=" * 70)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("=" * 70)
    
    comparison = pd.DataFrame({
        'Regular Momentum': [
            f"{regular_results['total_return']:.2f}%",
            f"{regular_results['annual_return']:.2f}%",
            f"{regular_results['volatility']:.2f}%",
            f"{regular_results['sharpe_ratio']:.2f}",
            f"{regular_results['max_drawdown']:.2f}%",
            regular_results['n_trades']
        ],
        'Regime-Conditioned': [
            f"{regime_results['total_return']:.2f}%",
            f"{regime_results['annual_return']:.2f}%",
            f"{regime_results['volatility']:.2f}%",
            f"{regime_results['sharpe_ratio']:.2f}",
            f"{regime_results['max_drawdown']:.2f}%",
            regime_results['n_trades']
        ]
    }, index=['Total Return', 'Annual Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Number of Trades'])
    
    print("\n" + comparison.to_string())
    
    # Calculate improvement
    logger.info("\n" + "=" * 70)
    logger.info("IMPROVEMENT FROM REGIME CONDITIONING")
    logger.info("=" * 70)
    
    sharpe_improvement = regime_results['sharpe_ratio'] - regular_results['sharpe_ratio']
    dd_improvement = regime_results['max_drawdown'] - regular_results['max_drawdown']
    
    logger.info(f"Sharpe Ratio Change: {sharpe_improvement:+.2f}")
    logger.info(f"Max Drawdown Change: {dd_improvement:+.2f}%")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"projects/b3_baselines/outputs/regime_comparison_{timestamp}.csv"
    comparison.to_csv(output_file)
    logger.info(f"\n✓ Results saved to: {output_file}")
    
    logger.info("\n✓ B3+B4 Integration Complete!")


if __name__ == "__main__":
    main()
