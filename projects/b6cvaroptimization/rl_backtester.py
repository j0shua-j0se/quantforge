"""Backtest using trained RL agent"""
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

class RLBacktester:
    def __init__(self, model_path="outputs/rl_agent_ppo.zip"):
        try:
            self.model = PPO.load(model_path)
            self.available = True
            print(f"✓ RL agent loaded from {model_path}")
        except Exception as e:
            print(f"⚠ RL agent not available: {e}")
            self.available = False
    
    def predict_weights(self, recent_returns, regime, risk_scores):
        """Get weights from RL agent"""
        if not self.available:
            return None
        
        # Format observation (match training format)
        obs = np.concatenate([
            recent_returns.flatten(),
            np.array([regime]),
            risk_scores
        ]).astype(np.float32)
        
        obs = np.nan_to_num(obs, 0)  # Replace NaN with 0
        
        # Get action (weights)
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Normalize to valid portfolio weights
        weights = action / (action.sum() + 1e-9)
        weights = np.clip(weights, 0, 0.20)  # Max 20% per asset
        weights = weights / (weights.sum() + 1e-9)
        
        return weights
