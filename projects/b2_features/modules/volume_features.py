"""
Volume Features Module
Computes microstructure and liquidity indicators.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class VolumeFeaturesEngine:
    """Compute volume-based features."""
    
    def __init__(self, config):
        self.config = config
        self.windows = config['features']['volume']['windows']
    
    def compute_volume_averages(self, df):
        """Compute volume moving averages."""
        logger.info("  Computing volume averages...")
        
        for window in self.windows:
            # Volume SMA
            df[f'volume_sma_{window}'] = df.groupby('ticker')['volume'].transform(
                lambda x: x.rolling(window=window, min_periods=window).mean()
            )
            
            # Volume ratio (current vs average)
            df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_sma_{window}']
        
        return df
    
    def compute_dollar_volume(self, df):
        """Dollar volume (price × volume) - liquidity proxy."""
        logger.info("  Computing dollar volume...")
        
        df['dollar_volume'] = df['close'] * df['volume']
        
        # Dollar volume moving average
        df['dollar_volume_sma_20'] = df.groupby('ticker')['dollar_volume'].transform(
            lambda x: x.rolling(window=20, min_periods=20).mean()
        )
        
        return df
    
    def compute_obv(self, df):
        """
        On-Balance Volume (OBV) - accumulation/distribution indicator.
        """
        logger.info("  Computing On-Balance Volume...")
        
        def obv_for_ticker(group):
            # OBV increases on up days, decreases on down days
            direction = np.sign(group['close'].diff())
            obv = (direction * group['volume']).fillna(0).cumsum()
            return obv
        
        df['obv'] = df.groupby('ticker', group_keys=False).apply(obv_for_ticker).values
        
        # Normalize OBV (different scales per ticker)
        df['obv_normalized'] = df.groupby('ticker')['obv'].transform(
            lambda x: (x - x.rolling(window=60, min_periods=60).mean()) / 
                      x.rolling(window=60, min_periods=60).std()
        )
        
        return df
    
    def compute_all(self, df):
        """Master function: compute all volume features."""
        logger.info("VOLUME FEATURES ENGINE:")
        
        df = self.compute_volume_averages(df)
        df = self.compute_dollar_volume(df)
        df = self.compute_obv(df)
        
        # Count volume features added
        volume_features = [col for col in df.columns if 'volume' in col.lower() or 'obv' in col.lower()]
        logger.info(f"  ✓ Computed {len(volume_features)} volume features")
        
        return df
