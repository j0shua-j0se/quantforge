"""GPU-Accelerated Portfolio Environment for RL"""
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class PortfolioEnv(gym.Env):
    """Custom Environment for Portfolio Optimization"""
    
    def __init__(self, returns_df, risk_scores, regimes_series, window=252):
        super().__init__()
        
        self.returns_df = returns_df.values  # Convert to numpy
        self.risk_scores = risk_scores.values
        self.regimes = regimes_series.values
        self.window = window
        self.n_assets = returns_df.shape[1]
        self.asset_names = returns_df.columns.tolist()
        
        # Action space: portfolio weights (continuous)
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_assets,), dtype=np.float32
        )
        
        # Observation space: [returns_window, regime, risk_scores]
        obs_size = self.n_assets * 21 + 1 + self.n_assets
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(obs_size,), dtype=np.float32
        )
        
        self.current_step = window
        self.max_steps = len(self.returns_df) - window - 1
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = self.window
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get current state"""
        # Last 21 days returns for each asset
        recent_returns = self.returns_df[
            self.current_step-21:self.current_step
        ].flatten()
        
        # Current regime
        regime = np.array([self.regimes[self.current_step]])
        
        # Risk scores
        risk = self.risk_scores
        
        obs = np.concatenate([recent_returns, regime, risk]).astype(np.float32)
        obs = np.nan_to_num(obs, 0)  # Replace NaN with 0
        return obs
    
    def step(self, action):
        """Execute one time step"""
        # Normalize action to valid weights
        weights = action / (action.sum() + 1e-9)
        weights = np.clip(weights, 0, 0.20)  # Max 20% per asset
        weights = weights / (weights.sum() + 1e-9)
        
        # Get next day returns
        next_returns = self.returns_df[self.current_step]
        
        # Portfolio return
        portfolio_return = (weights * next_returns).sum()
        
        # Compute reward (Sharpe-like with CVaR penalty)
        recent_portfolio_returns = self.returns_df[
            self.current_step-21:self.current_step
        ] @ weights
        
        volatility = np.std(recent_portfolio_returns) + 1e-9
        mean_return = np.mean(recent_portfolio_returns)
        
        # Reward = Sharpe + CVaR penalty
        sharpe_reward = mean_return / volatility
        
        # CVaR penalty for tail losses
        var_threshold = np.percentile(recent_portfolio_returns, 5)
        if portfolio_return < var_threshold:
            cvar_penalty = -abs(portfolio_return) * 10  # Strong penalty
        else:
            cvar_penalty = 0
        
        reward = sharpe_reward + cvar_penalty
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        
        info = {
            'portfolio_return': float(portfolio_return),
            'sharpe': float(sharpe_reward),
            'weights': weights
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        pass
