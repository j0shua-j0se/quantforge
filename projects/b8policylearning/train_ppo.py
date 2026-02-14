"""
B8 PPO Training Pipeline
Train reinforcement learning agent for portfolio optimization
"""
import yaml
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Add models to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from trading_env import QuantForgeTradingEnv

def main():
    print("="*60)
    print("B8 POLICY LEARNING - PPO TRAINING")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load config
    print("\n1. Loading configuration...")
    with open('config/ppo_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print(f"   Device: {config['ppo']['device']}")
    print(f"   Total timesteps: {config['ppo']['total_timesteps']:,}")
    print(f"   Learning rate: {config['ppo']['learning_rate']}")
    print("   ✅ Config loaded")
    
    # Create environment
    print("\n2. Creating trading environment...")
    env = DummyVecEnv([lambda: QuantForgeTradingEnv(config)])
    print("   ✅ Environment vectorized")
    
    # Create PPO agent
    print("\n3. Initializing PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=config['ppo']['learning_rate'],
        n_steps=config['ppo']['n_steps'],
        batch_size=config['ppo']['batch_size'],
        n_epochs=config['ppo']['n_epochs'],
        gamma=config['ppo']['gamma'],
        gae_lambda=config['ppo']['gae_lambda'],
        clip_range=config['ppo']['clip_range'],
        ent_coef=config['ppo']['ent_coef'],
        vf_coef=config['ppo']['vf_coef'],
        max_grad_norm=config['ppo']['max_grad_norm'],
        seed=config['ppo']['seed'],
        device=config['ppo']['device'],
        tensorboard_log="./logs/"
    )
    print("   ✅ PPO agent initialized")
    print(f"   Policy network: MLP")
    print(f"   Device: {model.device}")
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./outputs/checkpoints/',
        name_prefix='ppo_quantforge'
    )
    
    # Train
    print("\n4. Training PPO agent...")
    print(f"   This will take ~5-10 minutes on GPU")
    print(f"   Progress bar will show below:")
    print("-"*60)
    
    model.learn(
        total_timesteps=config['ppo']['total_timesteps'],
        progress_bar=True,
        callback=checkpoint_callback
    )
    
    print("-"*60)
    print("   ✅ Training complete!")
    
    # Save final model
    print("\n5. Saving trained model...")
    os.makedirs('outputs', exist_ok=True)
    model.save("outputs/ppo_quantforge_final")
    print("   ✅ Saved to: outputs/ppo_quantforge_final.zip")
    
    # Quick evaluation
    print("\n6. Quick evaluation (100 steps)...")
    obs = env.reset()
    total_reward = 0
    for i in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        if done:
            break
    
    print(f"   Average reward: {total_reward/100:.6f}")
    print("   ✅ Evaluation complete")
    
    # Save training info
    training_info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_timesteps': config['ppo']['total_timesteps'],
        'device': str(model.device),
        'avg_reward': float(total_reward/100)
    }
    
    with open('outputs/training_info.txt', 'w') as f:
        for key, value in training_info.items():
            f.write(f"{key}: {value}\n")
    
    print("\n" + "="*60)
    print("✅ B8 TRAINING COMPLETE!")
    print("="*60)
    print(f"\nOutputs saved to:")
    print(f"  - outputs/ppo_quantforge_final.zip")
    print(f"  - outputs/checkpoints/")
    print(f"  - logs/ (TensorBoard)")
    print(f"\nNext: Run backtest to compare vs B7 baseline")

if __name__ == "__main__":
    main()
