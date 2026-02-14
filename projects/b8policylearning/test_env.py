"""
Test B8 Trading Environment
Quick check that everything works
"""
import yaml
import sys
import os

# Add models to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from trading_env import QuantForgeTradingEnv

def test_environment():
    print("="*60)
    print("B8 ENVIRONMENT TEST")
    print("="*60)
    
    # Load config
    print("\n1. Loading configuration...")
    with open('config/ppo_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("   ✅ Config loaded")
    
    # Create environment
    print("\n2. Creating trading environment...")
    env = QuantForgeTradingEnv(config)
    print("   ✅ Environment created")
    
    # Test reset
    print("\n3. Testing reset...")
    obs, info = env.reset()
    print(f"   Observation shape: {obs.shape}")
    print(f"   Expected shape: {env.observation_space.shape}")
    assert obs.shape == env.observation_space.shape, "Observation shape mismatch!"
    print("   ✅ Reset works")
    
    # Test step
    print("\n4. Testing step...")
    action = env.action_space.sample()
    print(f"   Action shape: {action.shape}")
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"   Reward: {reward:.6f}")
    print(f"   Portfolio value: ${env.portfolio_value:,.2f}")
    print("   ✅ Step works")
    
    # Test multiple steps
    print("\n5. Testing 10 steps...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            print(f"   Episode ended at step {i+1}")
            break
    print("   ✅ Multiple steps work")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED - ENVIRONMENT READY!")
    print("="*60)

if __name__ == "__main__":
    test_environment()
