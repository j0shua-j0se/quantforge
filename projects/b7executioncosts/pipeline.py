"""
B7 Main Pipeline
Runs execution cost analysis and shows gross vs net performance
"""

import yaml
import logging
from pathlib import Path
import pandas as pd

from models.execution_model import ExecutionCostModel
from models.cost_aware_backtester import CostAwareBacktester

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/b7_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main execution pipeline for B7."""
    
    print("="*60)
    print("B7 EXECUTION COST MODELING - STARTING")
    print("="*60)
    
    # Load configuration
    config_path = Path('config/execution_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n✅ Loaded config from {config_path}")
    
    # Initialize cost model
    cost_model = ExecutionCostModel(config)
    print("✅ Cost model initialized")
    
    # Initialize backtester
    backtester = CostAwareBacktester(cost_model, config)
    print("✅ Backtester initialized")
    
    # Load data
    print("\n📊 Loading data...")
    data = backtester.load_data()
    print("✅ Data loaded successfully")
    
    # Run backtest
    print("\n🚀 Running backtest with execution costs...")
    results = backtester.run_backtest(
        data,
        initial_capital=1_000_000.0
    )
    print("✅ Backtest complete!")
    
    # Compute metrics
    print("\n📈 Computing performance metrics...")
    metrics = backtester.compute_metrics(results)
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS: GROSS vs NET PERFORMANCE")
    print("="*60)
    
    print(f"\n{'Metric':<30} {'Value':>15}")
    print("-" * 50)
    print(f"{'Gross Sharpe Ratio':<30} {metrics['gross_sharpe']:>15.3f}")
    print(f"{'Net Sharpe Ratio':<30} {metrics['net_sharpe']:>15.3f}")
    print(f"{'Sharpe Degradation':<30} {metrics['sharpe_degradation_pct']:>14.1f}%")
    print()
    print(f"{'Gross Annual Return':<30} {metrics['gross_annual_return']*100:>14.1f}%")
    print(f"{'Net Annual Return':<30} {metrics['net_annual_return']*100:>14.1f}%")
    print()
    print(f"{'Max Drawdown':<30} {metrics['max_drawdown']*100:>14.1f}%")
    print()
    print(f"{'Total Execution Costs':<30} ${metrics['total_execution_costs']:>14,.0f}")
    print(f"{'Number of Rebalances':<30} {metrics['num_rebalances']:>15.0f}")
    
    # Save results
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    results.to_csv(output_dir / 'b7_backtest_results.csv')
    
    with open(output_dir / 'b7_metrics.txt', 'w') as f:
        f.write("B7 Execution Cost Modeling - Performance Metrics\n")
        f.write("="*60 + "\n\n")
        for k, v in metrics.items():
            f.write(f"{k:<30}: {v:.4f}\n")
    
    print(f"\n✅ Results saved to {output_dir}/")
    print("   - b7_backtest_results.csv (daily returns)")
    print("   - b7_metrics.txt (summary metrics)")
    
    print("\n" + "="*60)
    print("B7 PIPELINE COMPLETE ✅")
    print("="*60)
    
    return metrics


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\n❌ Error occurred: {e}")
