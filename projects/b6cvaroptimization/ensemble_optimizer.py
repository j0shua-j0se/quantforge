"""Ensemble Optimizer - Combines CVaR, Sharpe, and Momentum"""
import numpy as np
import pandas as pd
from optimizer import CVaROptimizer
from scipy.optimize import minimize

class EnsembleOptimizer:
    def __init__(self, alpha=0.95, risk_penalty=0.10):
        self.cvar_optimizer = CVaROptimizer(alpha, risk_penalty)
        self.alpha = alpha
        self.risk_penalty = risk_penalty
        
    def optimize_sharpe_ratio(self, returns_df):
        """Max Sharpe optimization"""
        mean_returns = returns_df.mean(axis=1).values
        cov_matrix = np.cov(returns_df.values)
        
        def neg_sharpe(w):
            ret = w @ mean_returns
            vol = np.sqrt(w @ cov_matrix @ w + 1e-9)
            return -(ret / (vol + 1e-9))
        
        n_assets = len(returns_df)
        constraints = ({'type': 'eq', 'fun': lambda w: w.sum() - 1})
        bounds = tuple((0, 0.15) for _ in range(n_assets))
        
        result = minimize(neg_sharpe, x0=np.ones(n_assets)/n_assets,
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        return pd.Series(result.x, index=returns_df.index)
    
    def add_momentum_tilt(self, returns_df, weights, strength=0.15):
        """Add momentum factor tilt"""
        # 6-month momentum (126 trading days)
        if returns_df.shape[1] < 126:
            return weights
        
        momentum = returns_df.iloc[:, -126:].mean(axis=1).values
        momentum_z = (momentum - momentum.mean()) / (momentum.std() + 1e-9)
        momentum_z = np.clip(momentum_z, -2, 2)
        
        # Tilt weights toward positive momentum
        adjusted = weights.values * (1 + strength * momentum_z)
        adjusted = np.maximum(adjusted, 0)
        adjusted /= adjusted.sum()
        
        return pd.Series(adjusted, index=weights.index)
    
    def optimize(self, returns_df, risk_scores, regime, regime_constraints):
        """Ensemble optimization"""
        print("→ Running ensemble optimization...")
        
        # 1. CVaR weights
        w_cvar, cvar_results = self.cvar_optimizer.optimize(
            returns_df, risk_scores, regime, regime_constraints)
        
        # 2. Max Sharpe weights
        w_sharpe = self.optimize_sharpe_ratio(returns_df)
        
        # 3. Ensemble based on regime
        regime_names = {0: 'expansion', 1: 'transition', 2: 'crisis'}
        regime_name = regime_names.get(regime, 'transition')
        
        if regime == 2:  # Crisis - trust CVaR more
            weights = 0.7 * w_cvar + 0.3 * w_sharpe
            print(f"  Regime: {regime_name} → CVaR 70%, Sharpe 30%")
        elif regime == 0:  # Expansion - trust Sharpe more
            weights = 0.4 * w_cvar + 0.6 * w_sharpe
            print(f"  Regime: {regime_name} → CVaR 40%, Sharpe 60%")
        else:  # Transition - balanced
            weights = 0.5 * w_cvar + 0.5 * w_sharpe
            print(f"  Regime: {regime_name} → CVaR 50%, Sharpe 50%")
        
        # 4. Add momentum tilt
        weights = self.add_momentum_tilt(returns_df, weights, strength=0.15)
        
        # 5. Normalize
        weights = weights / weights.sum()
        
        # 6. Compute metrics
        portfolio_returns = (returns_df.values.T @ weights.values)
        var_95 = np.percentile(portfolio_returns, (1 - self.alpha) * 100)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        sharpe = weights @ returns_df.mean(axis=1) / (np.std(portfolio_returns) + 1e-9)
        
        results = {
            'status': 'OPTIMAL',
            'cvar_95': float(cvar_95),
            'sharpe': float(sharpe),
            'n_assets': int((weights.values > 0.001).sum()),
            'method': 'Ensemble-CVaR-Sharpe-Momentum'
        }
        
        print(f"✓ Ensemble complete | CVaR={cvar_95:.4f} | Sharpe={sharpe:.2f}")
        
        return weights, results
