"""
Price and Momentum Features Module
Computes walk-forward safe price features.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PriceFeaturesEngine:
    """Compute price-based features with walk-forward safety."""
    
    def __init__(self, config):
        self.config = config
        self.windows = config['features']['price_momentum']['windows']
        self.lag = config['walk_forward']['lag_features_by']
    
    def compute_returns(self, df):
        """Compute returns at multiple horizons."""
        logger.info("  Computing returns...")
        
        # Sort by ticker and date (critical!)
        df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
        
        # Daily returns
        df['returns_1d'] = df.groupby('ticker')['close'].pct_change()
        
        # Weekly returns (5 days)
        df['returns_5d'] = df.groupby('ticker')['close'].pct_change(5)
        
        # Monthly returns (20 days)
        df['returns_20d'] = df.groupby('ticker')['close'].pct_change(20)
        
        # Momentum (60 days = ~3 months)
        df['momentum_60d'] = df.groupby('ticker')['close'].pct_change(60)
        
        return df
    
    def compute_moving_averages(self, df):
        """Compute Simple Moving Averages (SMA)."""
        logger.info("  Computing moving averages...")
        
        for window in self.windows:
            # SMA for each window
            df[f'sma_{window}'] = df.groupby('ticker')['close'].transform(
                lambda x: x.rolling(window=window, min_periods=window).mean()
            )
            
            # Price relative to SMA (mean reversion signal)
            df[f'price_to_sma_{window}'] = df['close'] / df[f'sma_{window}']
        
        return df
    
    def compute_rsi(self, df, period=14):
        """Compute Relative Strength Index (RSI)."""
        logger.info("  Computing RSI...")
        
        def rsi_for_ticker(group):
            delta = group['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period, min_periods=period).mean()
            avg_loss = loss.rolling(window=period, min_periods=period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        df['rsi_14'] = df.groupby('ticker', group_keys=False).apply(rsi_for_ticker).values
        
        return df
    
    def apply_lag(self, df, feature_cols):
        """
        Lag features by 1 day to prevent lookahead bias.
        CRITICAL: Feature at time t uses only data from t-1.
        """
        logger.info(f"  Applying {self.lag}-day lag for walk-forward safety...")
        
        for col in feature_cols:
            if col in df.columns:
                df[col] = df.groupby('ticker')[col].shift(self.lag)
        
        return df
    
    def compute_all(self, df):
        """Master function: compute all price features."""
        logger.info("PRICE FEATURES ENGINE:")
        
        # Track which columns are new features
        initial_cols = set(df.columns)
        
        # Compute features
        df = self.compute_returns(df)
        df = self.compute_moving_averages(df)
        df = self.compute_rsi(df)
        
        # Find new feature columns
        feature_cols = list(set(df.columns) - initial_cols)
        
        # Apply lag for walk-forward safety
        df = self.apply_lag(df, feature_cols)
        
        logger.info(f"  ✓ Computed {len(feature_cols)} price features")
        
        return df
