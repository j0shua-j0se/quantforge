"""
B8 Backtest - Compare RL Agent vs B7 Baseline
"""
import yaml
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from stable_baselines3 import PPO
from trading_env import QuantForgeTradingEnv

def backtest_rl_agent():
    print("="*60)
    print("B8 BACKTEST - RL AGENT EVALUATION")
    print("="*60)
    
    # Load config
    print("\n1. Loading configuration...")
    with open('config/ppo_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("   ✅ Config loaded")
    
    # Load trained model
    print("\n2. Loading trained PPO agent...")
    try:
        model = PPO.load("outputs/ppo_quantforge_final", device=config['ppo']['device'])
        print(f"   ✅ Model loaded on {model.device}")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return
    
    # Create environment
    print("\n3. Creating test environment...")
    env = QuantForgeTradingEnv(config)
    print("   ✅ Environment ready")
    
    # Run backtest
    print("\n4. Running backtest...")
    print("   Testing on 2000 days...")
    obs, _ = env.reset()
    
    rewards = []
    portfolio_values = [env.initial_capital]
    actions_log = []
    
    test_steps = min(2000, env.max_steps)
    
    for step in range(test_steps):
        # Get action from trained agent
        action, _ = model.predict(obs, deterministic=True)
        actions_log.append(action)
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        
        rewards.append(reward)
        portfolio_values.append(env.portfolio_value)
        
        if terminated:
            print(f"   Episode ended at step {step+1}")
            break
        
        # Progress updates
        if (step + 1) % 500 == 0:
            current_return = (env.portfolio_value / env.initial_capital - 1) * 100
            print(f"   Step {step+1:4d}: Portfolio = ${env.portfolio_value:>12,.2f} ({current_return:+.2f}%)")
    
    print(f"   ✅ Backtest complete ({len(rewards)} steps)")
    
    # Calculate metrics
    print("\n5. Computing performance metrics...")
    returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1])
    
    # Basic metrics
    total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
    n_years = len(returns) / 252
    annual_return = ((portfolio_values[-1] / portfolio_values[0]) ** (1/n_years) - 1) * 100
    volatility = np.std(returns) * np.sqrt(252) * 100
    
    # Sharpe ratio
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe = (mean_return * 252) / (std_return * np.sqrt(252) + 1e-8)
    
    # Max drawdown
    cummax = np.maximum.accumulate(portfolio_values)
    drawdowns = (np.array(portfolio_values) - cummax) / cummax
    max_dd = np.min(drawdowns) * 100
    
    # Sortino ratio (downside deviation)
    negative_returns = returns[returns < 0]
    downside_std = np.std(negative_returns) if len(negative_returns) > 0 else std_return
    sortino = (mean_return * 252) / (downside_std * np.sqrt(252) + 1e-8)
    
    # Calmar ratio (return / max drawdown)
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
    
    # Print results
    print("\n" + "="*60)
    print("B8 RL AGENT RESULTS")
    print("="*60)
    print(f"\nTotal Return:        {total_return:>8.2f}%")
    print(f"Annual Return:       {annual_return:>8.2f}%")
    print(f"Volatility (ann):    {volatility:>8.2f}%")
    print(f"Sharpe Ratio:        {sharpe:>8.3f}")
    print(f"Sortino Ratio:       {sortino:>8.3f}")
    print(f"Calmar Ratio:        {calmar:>8.3f}")
    print(f"Max Drawdown:        {max_dd:>8.2f}%")
    print(f"Final Portfolio:     ${portfolio_values[-1]:>12,.2f}")
    print(f"Days Traded:         {len(returns):>8d}")
    
    print("\n" + "="*60)
    print("COMPARISON TO B7 BASELINE")
    print("="*60)
    print(f"B7 Net Sharpe:       {0.924:>8.3f}")
    print(f"B8 RL Sharpe:        {sharpe:>8.3f}")
    
    if sharpe > 0.924:
        improvement = ((sharpe / 0.924) - 1) * 100
        print(f"\n✅ IMPROVEMENT:      {improvement:>8.1f}%")
        print("   RL agent outperforms baseline!")
    else:
        degradation = ((0.924 - sharpe) / 0.924) * 100
        print(f"\n⚠️  Below baseline:   {-degradation:>8.1f}%")
        print("   Note: Using simplified features & random returns")
        print("   Full B7 integration needed for production")
    
    print("="*60)
    
    # Save results
    print("\n6. Saving results...")
    results_df = pd.DataFrame({
        'step': range(len(portfolio_values)),
        'portfolio_value': portfolio_values,
        'returns': [0] + list(returns)
    })
    results_df.to_csv('outputs/b8_backtest_results.csv', index=False)
    
    # Save metrics
    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': max_dd,
        'final_value': portfolio_values[-1],
        'days_traded': len(returns)
    }
    
    with open('outputs/b8_metrics.txt', 'w') as f:
        f.write("B8 RL AGENT PERFORMANCE METRICS\n")
        f.write("="*50 + "\n\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    print("   ✅ Results saved to:")
    print("      - outputs/b8_backtest_results.csv")
    print("      - outputs/b8_metrics.txt")
    
    print("\n" + "="*60)
    print("✅ BACKTEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    backtest_rl_agent()
