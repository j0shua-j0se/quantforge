"""Train RL Agent with GPU Acceleration"""
import torch
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_portfolio_env import PortfolioEnv
import yaml

def train_rl_agent():
    print("="*60)
    print("TRAINING RL AGENT (GPU-ACCELERATED)")
    print("="*60)
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    
    # Load data
    print("Loading data...")
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    signals = pd.read_csv(config['data']['signals_file'])
    features = pd.read_parquet(config['data']['features_file'])
    regimes = pd.read_csv(config['data']['regimes_file'])
    
    # Prepare data
    investable = signals[signals['signal'].isin(['STRONG_BUY', 'BUY'])]
    tickers = investable['ticker'].tolist()
    
    returns_panel = features[features['ticker'].isin(tickers)].pivot_table(
        index='date', columns='ticker', values='returns'
    ).fillna(0)
    
    risk_scores = investable.set_index('ticker')['risk_score']
    
    # Fix timezone issues
    regimes['date'] = pd.to_datetime(regimes['date']).dt.tz_localize(None)
    returns_panel.index = pd.to_datetime(returns_panel.index).tz_localize(None)
    
    # Create regime series (now both are timezone-naive)
    regime_series = pd.Series(index=returns_panel.index, data=0, name='regime')
    for idx, row in regimes.iterrows():
        regime_series[regime_series.index >= row['date']] = row['regime']
    
    print(f"✓ Data loaded: {len(returns_panel)} days, {len(tickers)} assets\n")
    
    # Create environment
    env = DummyVecEnv([lambda: PortfolioEnv(returns_panel, risk_scores, regime_series)])
    
    # Create PPO agent
    print("Initializing PPO agent...")
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        policy_kwargs={'net_arch': [256, 256, 128]},
        device=device,
        verbose=1,
        tensorboard_log="./logs/tensorboard/"
    )
    
    print("\nTraining agent (30-60 minutes on GPU)...")
    print("Progress will be shown below:\n")
    
    model.learn(total_timesteps=100000, progress_bar=True)
    
    # Save model
    model.save("outputs/rl_agent_ppo")
    print("\n" + "="*60)
    print("✓ Model saved to outputs/rl_agent_ppo.zip")
    print("="*60)

if __name__ == "__main__":
    train_rl_agent()
