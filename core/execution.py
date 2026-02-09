"""Execution cost modeling: slippage, spread, market impact."""

import numpy as np
from typing import Dict


class ExecutionModel:
    """
    Model transaction costs realistically.
    
    Components:
    - Bid-ask spread
    - Slippage (price movement during execution)
    - Market impact (permanent price move)
    """
    
    def __init__(self, 
                 base_spread_bps: float = 10.0,
                 slippage_pct: float = 0.01,
                 market_impact_coeff: float = 0.0001):
        self.base_spread_bps = base_spread_bps
        self.slippage_pct = slippage_pct
        self.market_impact_coeff = market_impact_coeff
    
    def estimate_cost(self, notional: float, volume_pct: float = 0.01) -> float:
        """
        Estimate total execution cost (one-way).
        
        Parameters:
            notional: Dollar amount being traded
            volume_pct: Trade size as % of daily volume
        
        Returns:
            Estimated cost in dollars
        """
        # Bid-ask spread
        spread_cost = notional * (self.base_spread_bps / 10_000)
        
        # Slippage
        slippage_cost = notional * self.slippage_pct
        
        # Market impact (square-root law)
        impact_cost = notional * self.market_impact_coeff * np.sqrt(volume_pct)
        
        total_cost = spread_cost + slippage_cost + impact_cost
        return total_cost
    
    def get_cost_breakdown(self, notional: float, volume_pct: float = 0.01) -> Dict[str, float]:
        """Return detailed cost breakdown."""
        spread_cost = notional * (self.base_spread_bps / 10_000)
        slippage_cost = notional * self.slippage_pct
        impact_cost = notional * self.market_impact_coeff * np.sqrt(volume_pct)
        
        return {
            'spread_cost': spread_cost,
            'slippage_cost': slippage_cost,
            'impact_cost': impact_cost,
            'total_cost': spread_cost + slippage_cost + impact_cost,
            'total_bps': ((spread_cost + slippage_cost + impact_cost) / notional) * 10_000
        }
