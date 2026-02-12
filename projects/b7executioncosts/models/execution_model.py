"""
B7 Execution Cost Model
Calculates trading costs: spread + market impact + fees
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class ExecutionCostModel:
    """
    Calculates execution costs for trading.
    
    Three cost components:
    1. Spread: Half-spread paid on every trade
    2. Market Impact: Price moves when you trade large size
    3. Fees: Broker commissions
    """
    
    def __init__(self, config: Dict):
        """Initialize with config parameters."""
        self.spread_large = config['execution_costs']['spread']['large_cap_bps']
        self.spread_mid = config['execution_costs']['spread']['mid_cap_bps']
        self.spread_small = config['execution_costs']['spread']['small_cap_bps']
        self.spread_default = config['execution_costs']['spread']['default_bps']
        
        self.impact_k = config['execution_costs']['impact']['coefficient_k']
        self.impact_alpha = config['execution_costs']['impact']['exponent_alpha']
        self.impact_vol_mult = config['execution_costs']['impact']['volatility_multiplier']
        
        self.fee_fixed = config['execution_costs']['fees']['fixed_per_trade']
        self.fee_per_share = config['execution_costs']['fees']['per_share']
        self.fee_max_pct = config['execution_costs']['fees']['max_pct']
        
        self.large_cap_min = config['market_cap_thresholds']['large_cap_min']
        self.mid_cap_min = config['market_cap_thresholds']['mid_cap_min']
        
        logger.info("ExecutionCostModel initialized")
    
    def get_spread_bps(self, ticker: str, market_cap: float = None) -> float:
        """Get bid-ask spread in basis points."""
        if market_cap is None:
            return self.spread_default
            
        if market_cap >= self.large_cap_min:
            return self.spread_large
        elif market_cap >= self.mid_cap_min:
            return self.spread_mid
        else:
            return self.spread_small
    
    def compute_spread_cost(self, trade_size: float, price: float, spread_bps: float) -> float:
        """Calculate bid-ask spread cost."""
        spread_fraction = spread_bps / 10000.0
        cost = 0.5 * spread_fraction * abs(trade_size) * price
        return cost
    
    def compute_market_impact(self, trade_size: float, daily_volume: float, 
                             volatility: float, price: float) -> float:
        """Calculate market impact using square-root law."""
        if daily_volume <= 0 or abs(trade_size) < 1:
            return 0.0
            
        participation_rate = abs(trade_size) / daily_volume
        impact_bps = self.impact_k * (participation_rate ** self.impact_alpha) * volatility * self.impact_vol_mult * 10000
        cost = (impact_bps / 10000.0) * abs(trade_size) * price
        return cost
    
    def compute_transaction_fees(self, trade_size: float, price: float) -> float:
        """Calculate transaction fees (broker commissions)."""
        shares = abs(trade_size)
        if shares < 1:
            return 0.0
            
        variable_fee = self.fee_per_share * shares
        fee = max(self.fee_fixed, variable_fee)
        
        trade_value = shares * price
        max_fee = self.fee_max_pct * trade_value
        fee = min(fee, max_fee)
        
        return fee
    
    def compute_total_cost(self, trade_size: float, price: float, daily_volume: float,
                          volatility: float, spread_bps: float) -> Dict[str, float]:
        """Calculate total execution cost with breakdown."""
        spread_cost = self.compute_spread_cost(trade_size, price, spread_bps)
        impact_cost = self.compute_market_impact(trade_size, daily_volume, volatility, price)
        fees = self.compute_transaction_fees(trade_size, price)
        
        total_cost = spread_cost + impact_cost + fees
        
        trade_value = abs(trade_size) * price
        cost_bps = (total_cost / trade_value * 10000) if trade_value > 0 else 0.0
        
        return {
            'spread_cost': spread_cost,
            'impact_cost': impact_cost,
            'fees': fees,
            'total_cost': total_cost,
            'cost_bps': cost_bps
        }
    
    def compute_portfolio_rebalance_cost(self, old_weights: pd.Series, new_weights: pd.Series,
                                        prices: pd.Series, volumes: pd.Series,
                                        volatilities: pd.Series, market_caps: pd.Series,
                                        portfolio_value: float) -> Tuple[float, pd.DataFrame]:
        """Calculate total cost of rebalancing portfolio."""
        tickers = old_weights.index.intersection(new_weights.index)
        
        total_cost = 0.0
        cost_records = []
        
        for ticker in tickers:
            old_w = old_weights.get(ticker, 0.0)
            new_w = new_weights.get(ticker, 0.0)
            
            if abs(new_w - old_w) < 1e-6:
                continue
            
            price = prices.get(ticker, 0.0)
            volume = volumes.get(ticker, 1e9)
            vol = volatilities.get(ticker, 0.02)
            mcap = market_caps.get(ticker, None)
            
            if price <= 0:
                continue
            
            old_shares = (old_w * portfolio_value) / price
            new_shares = (new_w * portfolio_value) / price
            trade_size = new_shares - old_shares
            
            spread_bps = self.get_spread_bps(ticker, mcap)
            
            costs = self.compute_total_cost(
                trade_size=trade_size,
                price=price,
                daily_volume=volume,
                volatility=vol,
                spread_bps=spread_bps
            )
            
            total_cost += costs['total_cost']
            
            cost_records.append({
                'ticker': ticker,
                'old_weight': old_w,
                'new_weight': new_w,
                'trade_size': trade_size,
                'price': price,
                'spread_cost': costs['spread_cost'],
                'impact_cost': costs['impact_cost'],
                'fees': costs['fees'],
                'total_cost': costs['total_cost'],
                'cost_bps': costs['cost_bps']
            })
        
        cost_df = pd.DataFrame(cost_records)
        
        logger.info(f"Rebalance cost: ${total_cost:,.2f}")
        
        return total_cost, cost_df


if __name__ == "__main__":
    print("ExecutionCostModel loaded successfully!")
