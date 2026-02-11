"""
B3 Main Pipeline - Run Everything
"""

import pandas as pd
import yaml
from datetime import datetime
import sys
sys.path.append('projects/b3_baselines')

from strategies.momentum import MomentumStrategy
from backtest import SimpleBacktester

def main():
    print("\n" + "="*60)
    print("B3 BASELINE STRATEGIES PIPELINE")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Load configuration
    print("\n[1/5] Loading configuration...")
    with open('projects/b3_baselines/config.yaml') as f:
        config = yaml.safe_load(f)
    print("✓ Config loaded")
    
    # 2. Load features from B2
    print("\n[2/5] Loading features from B2...")
    features = pd.read_parquet(config['data']['features_file'])
    print(f"✓ Loaded {len(features):,} records")
    print(f"  Date range: {features['date'].min()} to {features['date'].max()}")
    print(f"  Tickers: {features['ticker'].nunique()}")
    
    # Filter to 2018+ (after SMA_200 warmup)
    print("\n  Filtering to 2018-2024 (post-warmup)...")
    features = features[features['date'] >= '2018-01-01']
    print(f"  ✓ {len(features):,} records after filtering")
    
    # 3. Create strategy
    print("\n[3/5] Initializing Momentum Strategy...")
    strategy = MomentumStrategy(config)
    
    # 4. Generate signals
    print("\n[4/5] Generating trading signals...")
    signals = strategy.generate_signals(features)
    print(f"✓ Generated signals for {len(signals):,} data points")
    
    # 5. Run backtest
    print("\n[5/5] Running backtest...")
    backtester = SimpleBacktester(config)
    results = backtester.run(signals)
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    output_dir = 'projects/b3_baselines/outputs'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save equity curve
    equity_file = f'{output_dir}/equity_curve_{timestamp}.csv'
    results['equity_curve'].to_csv(equity_file, index=False)
    print(f"✓ Equity curve saved: {equity_file}")
    
    # Save metrics
    metrics_file = f'{output_dir}/metrics_{timestamp}.txt'
    with open(metrics_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("B3 MOMENTUM STRATEGY - BACKTEST RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("-"*60 + "\n")
        for key, value in results['metrics'].items():
            if isinstance(value, float):
                if 'return' in key or 'drawdown' in key or 'rate' in key:
                    f.write(f"{key:.<30} {value:>10.2%}\n")
                else:
                    f.write(f"{key:.<30} {value:>10.2f}\n")
            else:
                f.write(f"{key:.<30} {value:>10}\n")
        f.write("\n" + "="*60 + "\n")
    print(f"✓ Metrics saved: {metrics_file}")
    
    # Save signals sample
    signals_file = f'{output_dir}/signals_sample_{timestamp}.csv'
    signals.head(100).to_csv(signals_file, index=False)
    print(f"✓ Signals sample saved: {signals_file}")
    
    print("\n" + "="*60)
    print("✓ B3 PIPELINE COMPLETE!")
    print("="*60)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved in: {output_dir}/")
    print(f"  - {equity_file.split('/')[-1]}")
    print(f"  - {metrics_file.split('/')[-1]}")
    print(f"  - {signals_file.split('/')[-1]}")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
        print("\n🎉 SUCCESS! Check the outputs/ folder for your results.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
