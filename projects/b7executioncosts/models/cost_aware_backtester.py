"""
B7 Cost-Aware Backtester
Runs backtest with execution costs deducted from returns
"""

import numpy as np
import pandas as pd
from typing import Dict
from pathlib import Path
import logging
from tqdm import tqdm

from .execution_model import ExecutionCostModel

logger = logging.getLogger(__name__)


class CostAwareBacktester:
    """
    Backtester that deducts execution costs from returns.
    Shows gross vs net performance.
    """
    
    def __init__(self, execution_model: ExecutionCostModel, config: Dict):
        """Initialize backtester."""
        self.cost_model = execution_model
        self.config = config
        
        self.rebalance_freq = config['backtest']['rebalance_frequency']
        self.min_trade_threshold = config['backtest']['min_trade_threshold']
        
        logger.info("CostAwareBacktester initialized")
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load price data and B6 results."""
        logger.info("Loading data...")
        
        # Load price data
        prices_path = Path(self.config['data']['prices'])
        df = pd.read_parquet(prices_path)
        
        # Pivot to wide format (date x ticker)
        prices = df.pivot(index='date', columns='ticker', values='close')
        volumes = df.pivot(index='date', columns='ticker', values='volume')
        
        # Calculate volatilities (21-day rolling)
        returns = prices.pct_change()
        volatilities = returns.rolling(21).std()
        
        logger.info(f"Loaded {len(prices)} days, {len(prices.columns)} tickers")
        
        return {
            'prices': prices,
            'volumes': volumes,
            'volatilities': volatilities
        }
    
    def run_backtest(self, data: Dict[str, pd.DataFrame], 
                    initial_capital: float = 1_000_000.0) -> pd.DataFrame:
        """Run backtest with execution costs."""
        
        prices = data['prices']
        volumes = data['volumes']
        volatilities = data['volatilities']
        
        # Rebalance frequency mapping
        rebal_days = {'daily': 1, 'weekly': 5, 'monthly': 21, 'quarterly': 63}
        rebal_freq = rebal_days.get(self.rebalance_freq, 63)
        
        # Initialize portfolio
        portfolio_value = initial_capital
        num_assets = len(prices.columns)
        current_weights = pd.Series(1.0 / num_assets, index=prices.columns)  # Start equal weight
        
        results = []
        
        logger.info(f"Running backtest: ${initial_capital:,.0f} initial capital")
        logger.info(f"Rebalancing every {rebal_freq} days")
        
        dates = prices.index
        
        for i in tqdm(range(len(dates)), desc="Backtesting"):
            date = dates[i]
            
            today_prices = prices.loc[date]
            today_volumes = volumes.loc[date]
            today_vols = volatilities.loc[date]
            
            # Check if rebalance day
            is_rebalance_day = (i % rebal_freq == 0)
            
            execution_cost = 0.0
            
            if is_rebalance_day and i > 0:
                # Momentum rebalance: Top 20 stocks by 3-month return
                lookback_prices = prices.iloc[max(0, i-63):i]  # 3 months
                momentum = (lookback_prices.iloc[-1] / lookback_prices.iloc[0] - 1).fillna(0)
                top_20 = momentum.nlargest(20).index
                
                new_weights = pd.Series(0.0, index=prices.columns)
                new_weights[top_20] = 1.0 / 20  # Equal weight top 20
                
                # Calculate rebalancing cost
                cost, _ = self.cost_model.compute_portfolio_rebalance_cost(
                    old_weights=current_weights,
                    new_weights=new_weights,
                    prices=today_prices,
                    volumes=today_volumes,
                    volatilities=today_vols,
                    market_caps=pd.Series(1e11, index=today_prices.index),  # Placeholder
                    portfolio_value=portfolio_value
                )
                
                execution_cost = cost
                current_weights = new_weights.copy()
            
            # Calculate daily return
            if i > 0:
                prev_prices = prices.iloc[i-1]
                price_returns = (today_prices / prev_prices - 1).fillna(0)
                gross_daily_return = (current_weights * price_returns).sum()
            else:
                gross_daily_return = 0.0
            
            # Update portfolio value (gross)
            portfolio_value_before = portfolio_value
            portfolio_value = portfolio_value * (1 + gross_daily_return)
            
            # Deduct execution cost
            portfolio_value -= execution_cost
            
            # Net return
            net_daily_return = (portfolio_value / portfolio_value_before) - 1
            
            # Record results
            results.append({
                'date': date,
                'gross_return': gross_daily_return,
                'net_return': net_daily_return,
                'execution_cost': execution_cost,
                'portfolio_value': portfolio_value,
                'is_rebalance': is_rebalance_day
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.set_index('date')
        
        logger.info(f"Backtest complete!")
        logger.info(f"Final value: ${portfolio_value:,.0f}")
        
        return results_df
    
    def compute_metrics(self, results: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics."""
        
        gross_returns = results['gross_return'].values
        net_returns = results['net_return'].values
        
        # Annualized metrics
        gross_annual = gross_returns.mean() * 252
        net_annual = net_returns.mean() * 252
        
        gross_vol = gross_returns.std() * np.sqrt(252)
        net_vol = net_returns.std() * np.sqrt(252)
        
        gross_sharpe = gross_annual / gross_vol if gross_vol > 0 else 0
        net_sharpe = net_annual / net_vol if net_vol > 0 else 0
        
        # Max drawdown
        cumulative = (1 + results['net_return']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Total costs
        total_costs = results['execution_cost'].sum()
        
        return {
            'gross_sharpe': gross_sharpe,
            'net_sharpe': net_sharpe,
            'sharpe_degradation_pct': (gross_sharpe - net_sharpe) / gross_sharpe * 100,
            'gross_annual_return': gross_annual,
            'net_annual_return': net_annual,
            'max_drawdown': max_drawdown,
            'total_execution_costs': total_costs,
            'num_rebalances': results['is_rebalance'].sum()
        }


if __name__ == "__main__":
    print("CostAwareBacktester loaded successfully!")
