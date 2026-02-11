"""
Simple Backtester for B3 Strategies - FIXED VERSION
"""

import pandas as pd
import numpy as np
import yaml
from datetime import datetime

class SimpleBacktester:
    """Backtest a strategy and calculate returns"""
    
    def __init__(self, config):
        self.config = config
        self.initial_cash = config['backtest']['starting_cash']
        self.commission = config['costs']['commission_percent'] / 100
        self.slippage = config['costs']['slippage_percent'] / 100
        
        print(f"✓ Backtester initialized")
        print(f"  Starting cash: ${self.initial_cash:,.0f}")
        print(f"  Commission: {self.commission*100:.2f}%")
    
    def run(self, signals_df):
        """
        Run backtest on signals - SIMPLIFIED VERSION
        
        Args:
            signals_df: DataFrame with columns [date, ticker, close, signal]
        
        Returns:
            Dictionary with results
        """
        print("\n" + "="*60)
        print("RUNNING BACKTEST")
        print("="*60)
        
        df = signals_df.copy()
        df = df.sort_values(['date', 'ticker'])
        
        # Calculate daily returns for each ticker
        df['next_close'] = df.groupby('ticker')['close'].shift(-1)
        df['daily_return'] = (df['next_close'] / df['close']) - 1
        
        # Group by date
        dates = sorted(df['date'].unique())
        print(f"✓ Backtesting {len(dates)} trading days")
        
        # Track portfolio
        equity_curve = []
        current_equity = self.initial_cash
        
        for i, date in enumerate(dates):
            day_data = df[df['date'] == date].copy()
            
            # Count positions
            n_long = (day_data['signal'] == 1).sum()
            n_short = (day_data['signal'] == -1).sum()
            total_positions = n_long + n_short
            
            if total_positions == 0:
                equity_curve.append({
                    'date': date,
                    'equity': current_equity,
                    'return': 0.0
                })
                continue
            
            # Calculate weighted portfolio return
            portfolio_return = 0.0
            
            # Long positions: profit when stocks go up
            if n_long > 0:
                long_stocks = day_data[day_data['signal'] == 1]
                long_returns = long_stocks['daily_return'].dropna()
                if len(long_returns) > 0:
                    avg_long_return = long_returns.mean()
                    portfolio_return += avg_long_return * (n_long / total_positions)
            
            # Short positions: profit when stocks go down
            if n_short > 0:
                short_stocks = day_data[day_data['signal'] == -1]
                short_returns = short_stocks['daily_return'].dropna()
                if len(short_returns) > 0:
                    avg_short_return = -short_returns.mean()  # Negative because we're shorting
                    portfolio_return += avg_short_return * (n_short / total_positions)
            
            # Apply transaction costs
            net_return = portfolio_return - self.commission - self.slippage
            
            # Update equity
            current_equity *= (1 + net_return)
            
            equity_curve.append({
                'date': date,
                'equity': current_equity,
                'return': net_return
            })
            
            # Progress
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(dates)} days...")
        
        # Convert to DataFrame
        equity_df = pd.DataFrame(equity_curve)
        
        # Calculate metrics
        metrics = self.calculate_metrics(equity_df)
        
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Total Return:     {metrics['total_return']:>8.2%}")
        print(f"Annual Return:    {metrics['annual_return']:>8.2%}")
        print(f"Sharpe Ratio:     {metrics['sharpe']:>8.2f}")
        print(f"Max Drawdown:     {metrics['max_drawdown']:>8.2%}")
        print(f"Win Rate:         {metrics['win_rate']:>8.2%}")
        print("="*60)
        
        return {
            'equity_curve': equity_df,
            'metrics': metrics
        }
    
    def calculate_metrics(self, equity_df):
        """Calculate performance metrics"""
        
        returns = equity_df['return'].dropna()
        equity = equity_df['equity']
        
        # Total return
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        
        # Annualized return
        n_years = len(equity) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # Sharpe ratio
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Max drawdown
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'n_days': len(equity)
        }


# Test it!
if __name__ == "__main__":
    import sys
    sys.path.append('projects/b3_baselines')
    from strategies.momentum import MomentumStrategy
    
    print("="*60)
    print("TESTING BACKTESTER - FIXED VERSION")
    print("="*60)
    
    # Load config
    with open('projects/b3_baselines/config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Load features
    print("\n1. Loading features...")
    features = pd.read_parquet(config['data']['features_file'])
    
    # Test on 2024 only (fast test)
    print("\n2. Testing on 2024 data...")
    test_data = features[features['date'] >= '2024-01-01']
    print(f"   ✓ {len(test_data):,} records")
    
    # Generate signals
    print("\n3. Generating signals...")
    strategy = MomentumStrategy(config)
    signals = strategy.generate_signals(test_data)
    
    # Run backtest
    print("\n4. Running backtest...")
    backtester = SimpleBacktester(config)
    results = backtester.run(signals)
    
    # Save results
    print("\n5. Saving results...")
    results['equity_curve'].to_csv('projects/b3_baselines/outputs/equity_curve.csv', index=False)
    print("   ✓ Saved to outputs/equity_curve.csv")
    
    print("\n✓ TEST COMPLETE!")
