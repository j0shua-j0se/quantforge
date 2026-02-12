"""Hybrid: Ensemble + RL + Momentum"""
import numpy as np
import pandas as pd
from ensemble_optimizer import EnsembleOptimizer
from rl_backtester import RLBacktester

class HybridOptimizer:
    def __init__(self, alpha=0.95, risk_penalty=0.10, use_rl=True):
        self.ensemble = EnsembleOptimizer(alpha, risk_penalty)
        self.use_rl = use_rl
        self.alpha = alpha
        
        if use_rl:
            self.rl_agent = RLBacktester()
            if self.rl_agent.available:
                print("✓ Hybrid mode: Ensemble + RL + Momentum")
            else:
                self.use_rl = False
                print("⚠ RL not available, using Ensemble only")
    
    def optimize(self, returns_df, risk_scores, regime, regime_constraints):
        """Hybrid optimization"""
        # 1. Get ensemble weights
        w_ensemble, ensemble_results = self.ensemble.optimize(
            returns_df, risk_scores, regime, regime_constraints
        )
        
        # 2. Get RL weights if available
        if self.use_rl and self.rl_agent.available:
            recent_returns = returns_df.iloc[:, -21:].values  # Last 21 days
            w_rl = self.rl_agent.predict_weights(
                recent_returns, regime, risk_scores.values
            )
            
            if w_rl is not None:
                w_rl = pd.Series(w_rl, index=returns_df.index)
                
                # 3. Combine (regime-dependent weighting)
                regime_names = {0: 'expansion', 1: 'transition', 2: 'crisis'}
                regime_name = regime_names.get(regime, 'transition')
                
                if regime == 0:  # Expansion - trust RL more (it learned momentum)
                    weights = 0.6 * w_rl + 0.4 * w_ensemble
                    rl_weight = 0.6
                elif regime == 2:  # Crisis - trust ensemble CVaR more
                    weights = 0.3 * w_rl + 0.7 * w_ensemble
                    rl_weight = 0.3
                else:  # Transition - balanced
                    weights = 0.5 * w_rl + 0.5 * w_ensemble
                    rl_weight = 0.5
                
                print(f"  Hybrid: RL {rl_weight*100:.0f}%, Ensemble {(1-rl_weight)*100:.0f}% ({regime_name})")
            else:
                weights = w_ensemble
                print("  Using Ensemble only (RL prediction failed)")
        else:
            weights = w_ensemble
        
        # Normalize
        weights = weights / weights.sum()
        
        # Compute metrics
        portfolio_returns = (returns_df.values.T @ weights.values)
        var_95 = np.percentile(portfolio_returns, (1 - self.alpha) * 100)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        sharpe = weights @ returns_df.mean(axis=1) / (np.std(portfolio_returns) + 1e-9)
        
        results = {
            'status': 'OPTIMAL',
            'cvar_95': float(cvar_95),
            'sharpe': float(sharpe),
            'n_assets': int((weights.values > 0.001).sum()),
            'method': 'Hybrid-RL-Ensemble-Momentum' if (self.use_rl and self.rl_agent.available) else 'Ensemble'
        }
        
        print(f"✓ Hybrid complete | CVaR={cvar_95:.4f} | Sharpe={sharpe:.2f}")
        
        return weights, results
