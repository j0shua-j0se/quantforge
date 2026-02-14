"""
B8 Trading Environment for PPO
Integrates B7 execution costs with RL agent
"""
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import sys
import os

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

class QuantForgeTradingEnv(gym.Env):
    """Custom trading environment for reinforcement learning"""
    
    def __init__(self, config):
        super().__init__()
        
        print("📊 Initializing Trading Environment...")
        
        # Load config
        self.config = config
        self.n_assets = config['strategy']['n_assets']
        self.initial_capital = config['strategy']['initial_capital']
        
        # Load data
        print("   Loading features...")
        self.features = pd.read_parquet(config['data']['features'])
        print(f"   ✅ Loaded {len(self.features)} rows of data")
        
        # Get numeric columns only (exclude 'ticker', 'date', etc.)
        numeric_cols = self.features.select_dtypes(include=[np.number]).columns
        self.numeric_features = list(numeric_cols)
        self.n_features = len(self.numeric_features)
        print(f"   ✅ Found {self.n_features} numeric features")
        
        # Action space: portfolio weights for each asset
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.n_assets,), 
            dtype=np.float32
        )
        
        # Observation space: actual numeric features + portfolio state
        # Shape: n_features + 5 portfolio statistics
        obs_dim = self.n_features + 5
        self.observation_space = spaces.Box(
            low=-10.0, 
            high=10.0, 
            shape=(obs_dim,),
            dtype=np.float32
        )
        print(f"   ✅ Observation space: {obs_dim} dimensions")
        
        # State variables
        self.current_step = 0
        self.max_steps = len(self.features) - 1
        self.portfolio_value = self.initial_capital
        self.holdings = np.zeros(self.n_assets)
        
        print("   ✅ Environment initialized!")
        
    def reset(self, seed=None):
        """Reset environment to start"""
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.holdings = np.zeros(self.n_assets)
        
        obs = self._get_observation()
        return obs, {}
    
    def _get_observation(self):
        """Get current market state"""
        try:
            # Get current row
            current_data = self.features.iloc[self.current_step]
            
            # Extract only numeric features
            features = current_data[self.numeric_features].values.astype(np.float32)
            
            # Add portfolio state statistics
            portfolio_state = np.array([
                self.portfolio_value / self.initial_capital,  # Normalized value
                np.sum(self.holdings > 0) / self.n_assets,    # % assets held
                np.std(self.holdings),                         # Position concentration
                self.current_step / self.max_steps,            # Time progress
                np.mean(self.holdings)                         # Average weight
            ], dtype=np.float32)
            
            # Concatenate features and portfolio state
            obs = np.concatenate([features, portfolio_state])
            
            # Handle NaN/Inf values
            obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
            
            # Clip to observation space bounds
            obs = np.clip(obs, -10.0, 10.0)
            
            return obs
            
        except Exception as e:
            print(f"⚠️  Observation error at step {self.current_step}: {e}")
            # Return zeros as safe fallback
            return np.zeros(self.observation_space.shape, dtype=np.float32)
    
    def step(self, action):
        """Execute one trading step"""
        
        # Normalize weights to sum to 1 (long-only portfolio)
        weights = np.abs(action)
        weights = weights / (weights.sum() + 1e-8)
        
        # Simplified returns (placeholder - will integrate B7 costs later)
        # Using random returns for now as proof-of-concept
        returns = np.random.randn(self.n_assets) * 0.01  # 1% daily vol
        
        # Portfolio return
        portfolio_return = np.dot(weights, returns)
        
        # Simple execution cost (0.1% of turnover)
        turnover = np.sum(np.abs(weights - self.holdings))
        execution_cost = 0.001 * turnover
        
        # Net return after costs
        net_return = portfolio_return - execution_cost
        
        # Update portfolio value
        self.portfolio_value *= (1 + net_return)
        self.holdings = weights
        
        # Reward = net return with small risk penalty
        reward = net_return - 0.0001 * np.std(returns)
        
        # Move to next step
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        
        # Get next observation
        obs = self._get_observation()
        
        return obs, reward, terminated, False, {}
