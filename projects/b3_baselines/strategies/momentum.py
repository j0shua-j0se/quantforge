"""
Simple Momentum Strategy
Buy stocks going UP, short stocks going DOWN
"""

import pandas as pd
import numpy as np

class MomentumStrategy:
    """Buy winners, sell losers"""
    
    def __init__(self, config):
        self.long_pct = config['strategies']['momentum']['long_top_percent'] / 100
        self.short_pct = config['strategies']['momentum']['short_bottom_percent'] / 100
        print(f"✓ Momentum Strategy loaded")
        print(f"  Long top {self.long_pct*100:.0f}%")
        print(f"  Short bottom {self.short_pct*100:.0f}%")
    
    def generate_signals(self, data):
        """
        Generate BUY/SELL signals
        
        Returns:
            DataFrame with 'signal' column (1=buy, -1=sell, 0=hold)
        """
        df = data.copy()
        
        # Use the momentum_60d feature from B2
        if 'momentum_60d' not in df.columns:
            raise ValueError("Need 'momentum_60d' feature from B2!")
        
        # Remove rows with missing momentum
        df = df.dropna(subset=['momentum_60d'])
        
        # For each date, rank stocks by momentum
        df['momentum_rank'] = df.groupby('date')['momentum_60d'].rank(pct=True)
        
        # Create signals
        df['signal'] = 0  # Default: hold
        
        # Buy top performers (high momentum)
        df.loc[df['momentum_rank'] >= (1 - self.long_pct), 'signal'] = 1
        
        # Short bottom performers (low momentum)
        df.loc[df['momentum_rank'] <= self.short_pct, 'signal'] = -1
        
        # Count signals
        n_long = (df['signal'] == 1).sum()
        n_short = (df['signal'] == -1).sum()
        print(f"✓ Signals generated: {n_long} longs, {n_short} shorts")
        
        return df[['date', 'ticker', 'close', 'signal', 'momentum_60d']]


# Test it!
if __name__ == "__main__":
    import yaml
    
    print("="*60)
    print("TESTING MOMENTUM STRATEGY")
    print("="*60)
    
    # Load config
    with open('projects/b3_baselines/config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Load B2 features
    print("\n1. Loading features from B2...")
    features = pd.read_parquet(config['data']['features_file'])
    print(f"   ✓ Loaded {len(features):,} records")
    
    # Test on one month of data
    print("\n2. Testing strategy on January 2024...")
    test_data = features[
        (features['date'] >= '2024-01-01') & 
        (features['date'] <= '2024-01-31')
    ]
    print(f"   ✓ Test data: {len(test_data):,} records")
    
    # Create strategy
    print("\n3. Creating strategy...")
    strategy = MomentumStrategy(config)
    
    # Generate signals
    print("\n4. Generating signals...")
    signals = strategy.generate_signals(test_data)
    
    # Show results
    print("\n5. Sample signals:")
    print(signals[signals['signal'] != 0].head(10))
    
    print("\n✓ TEST PASSED!")
