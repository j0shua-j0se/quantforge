"""B6 CVaR Optimization - Main Pipeline"""
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
from optimizer import CVaROptimizer
from backtester import WalkForwardBacktester

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"✓ Configuration loaded")
    return config

def load_data(config):
    print("\nLoading data files...")
    
    signals = pd.read_csv(config['data']['signals_file'])
    print(f"✓ Signals loaded: {len(signals)} rows")
    
    features = pd.read_parquet(config['data']['features_file'])
    print(f"✓ Features loaded: {features.shape}")
    
    regimes = pd.read_csv(config['data']['regimes_file'])
    print(f"✓ Regimes loaded: {len(regimes)} rows")
    
    return signals, features, regimes

def main():
    print("="*60)
    print("B6 CVAR PORTFOLIO OPTIMIZATION")
    print("="*60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    config = load_config()
    signals, features, regimes = load_data(config)
    
    print("\n" + "="*60)
    print("INITIALIZING OPTIMIZER")
    print("="*60)
    optimizer = CVaROptimizer(
        alpha=config['optimizer']['alpha'],
        risk_penalty=config['optimizer']['risk_penalty']
    )
    
    backtester = WalkForwardBacktester(
        optimizer=optimizer,
        train_window=config['backtest']['train_window'],
        test_window=config['backtest']['test_window']
    )
    
    print("\n" + "="*60)
    print("RUNNING BACKTEST")
    print("="*60)
    results = backtester.backtest(features, signals, regimes, config)
    
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"backtest_results_{timestamp}.csv"
    results.to_csv(results_file, index=False)
    print(f"✓ Results saved: {results_file}")
    
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Average Sharpe:  {results['sharpe'].mean():.2f}")
    print(f"Average Max DD:  {results['max_drawdown'].mean():.2%}")
    print(f"Average CVaR:    {results['cvar_95'].mean():.4f}")
    print(f"Win Rate:        {(results['total_return'] > 0).mean():.1%}")
    
    target_sharpe = config['performance']['target_sharpe']
    target_dd = config['performance']['target_max_dd']
    
    sharpe_met = results['sharpe'].mean() >= target_sharpe
    dd_met = results['max_drawdown'].mean() >= target_dd
    
    print(f"\n{'✓' if sharpe_met else '✗'} Sharpe target: {results['sharpe'].mean():.2f} vs {target_sharpe}")
    print(f"{'✓' if dd_met else '✗'} Max DD target: {results['max_drawdown'].mean():.2%} vs {target_dd:.0%}")
    
    if sharpe_met and dd_met:
        print("\n🎉 ALL PERFORMANCE TARGETS ACHIEVED!")
    else:
        print("\n⚠ Some targets not met - consider parameter tuning")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
