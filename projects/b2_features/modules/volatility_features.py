"""
Volatility Features Module
Computes risk and volatility metrics.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class VolatilityFeaturesEngine:
    """Compute volatility features for tail-risk awareness."""
    
    def __init__(self, config):
        self.config = config
        self.windows = config['features']['volatility']['windows']
    
    def compute_realized_volatility(self, df):
        """Compute realized volatility (rolling standard deviation)."""
        logger.info("  Computing realized volatility...")
        
        # Need returns first
        if 'returns_1d' not in df.columns:
            df['returns_1d'] = df.groupby('ticker')['close'].pct_change()
        
        # Rolling volatility at multiple windows (annualized)
        for window in self.windows:
            df[f'realized_vol_{window}d'] = df.groupby('ticker')['returns_1d'].transform(
                lambda x: x.rolling(window=window, min_periods=window).std() * np.sqrt(252)
            )
        
        return df
    
    def compute_atr(self, df, period=14):
        """
        Average True Range (ATR) - measures volatility including gaps.
        """
        logger.info("  Computing ATR...")
        
        def atr_for_ticker(group):
            high = group['high']
            low = group['low']
            close_prev = group['close'].shift(1)
            
            # True Range components
            tr1 = high - low
            tr2 = abs(high - close_prev)
            tr3 = abs(low - close_prev)
            
            # True Range = max of the three
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # ATR = moving average of True Range
            atr = true_range.rolling(window=period, min_periods=period).mean()
            
            return atr
        
        df['atr_14'] = df.groupby('ticker', group_keys=False).apply(atr_for_ticker).values
        
        return df
    
    def compute_vol_of_vol(self, df):
        """
        Volatility of volatility - detects regime changes.
        """
        logger.info("  Computing volatility of volatility...")
        
        # Need realized_vol first
        if 'realized_vol_20d' not in df.columns:
            df = self.compute_realized_volatility(df)
        
        # Volatility of the volatility
        df['vol_of_vol'] = df.groupby('ticker')['realized_vol_20d'].transform(
            lambda x: x.rolling(window=20, min_periods=20).std()
        )
        
        return df
    
    def compute_all(self, df):
        """Master function: compute all volatility features."""
        logger.info("VOLATILITY FEATURES ENGINE:")
        
        df = self.compute_realized_volatility(df)
        df = self.compute_atr(df)
        df = self.compute_vol_of_vol(df)
        
        # Count volatility features added
        vol_features = [col for col in df.columns if 'vol' in col.lower() or 'atr' in col.lower()]
        logger.info(f"  ✓ Computed {len(vol_features)} volatility features")
        
        return df
