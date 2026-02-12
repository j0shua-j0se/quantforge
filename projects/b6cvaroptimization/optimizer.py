"""B6 CVaR Portfolio Optimizer"""
import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Tuple, Dict

class CVaROptimizer:
    def __init__(self, alpha=0.95, risk_penalty=0.1):
        self.alpha = alpha
        self.risk_penalty = risk_penalty
        print(f"CVaR Optimizer initialized (alpha={alpha})")
    
    def optimize(self, returns_df, risk_scores, regime, regime_constraints):
        print(f"Optimizing for regime {regime}, {len(returns_df)} assets...")
        
        n_assets = len(returns_df)
        n_samples = returns_df.shape[1]
        returns_matrix = returns_df.values
        
        regime_names = {0: 'expansion', 1: 'transition', 2: 'crisis'}
        regime_name = regime_names.get(regime, 'transition')
        max_weight = regime_constraints[regime_name]['max_weight']
        
        w = cp.Variable(n_assets)
        u = cp.Variable(n_samples)
        zeta = cp.Variable()
        
        cvar_term = zeta + (1 / ((1 - self.alpha) * n_samples)) * cp.sum(u)
        mean_returns = returns_matrix.mean(axis=1)
        expected_return = mean_returns @ w
        risk_penalty_term = self.risk_penalty * (risk_scores.values @ w)
        
        objective = cp.Maximize(expected_return - cvar_term - risk_penalty_term)
        
        constraints = [
            u >= -(returns_matrix.T @ w) - zeta,
            u >= 0,
            cp.sum(w) == 1,
            w >= 0,
            w <= max_weight
        ]
        
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, verbose=False)
        
        optimal_weights = np.maximum(w.value, 0)
        optimal_weights /= optimal_weights.sum()
        
        weights = pd.Series(optimal_weights, index=returns_df.index)
        
        portfolio_returns = returns_matrix.T @ optimal_weights
        var_95 = np.percentile(portfolio_returns, (1 - self.alpha) * 100)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        results = {
            'status': problem.status,
            'cvar_95': float(cvar_95),
            'sharpe': float(expected_return.value / (np.std(portfolio_returns) + 1e-9)),
            'n_assets': int((optimal_weights > 0.001).sum())
        }
        
        print(f"Optimization complete | CVaR={cvar_95:.4f} | Sharpe={results['sharpe']:.2f}")
        
        return weights, results
