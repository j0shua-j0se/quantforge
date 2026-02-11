"""
B4 Macro Regime Detection - Data Loader
Downloads FRED macro data
"""

import os
import logging
from typing import List
import pandas as pd
import numpy as np
from fredapi import Fred

logger = logging.getLogger(__name__)


class FREDDataLoader:
    """Load FRED macroeconomic data"""
    
    def __init__(self, api_key: str, series_ids: List[str]):
        self.fred = Fred(api_key=api_key)
        self.series_ids = series_ids
        logger.info(f"FRED Data Loader initialized with {len(series_ids)} series")
    
    def download_all(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Download all FRED series"""
        logger.info(f"Downloading {len(self.series_ids)} FRED series...")
        
        data = {}
        for series_id in self.series_ids:
            try:
                series = self.fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
                data[series_id] = series
                logger.info(f"✓ Downloaded {series_id}: {len(series)} observations")
            except Exception as e:
                logger.error(f"✗ Failed to download {series_id}: {e}")
        
        df = pd.DataFrame(data)
        logger.info(f"Downloaded macro data: {df.shape[0]} dates × {df.shape[1]} indicators")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from raw FRED data"""
        logger.info("Engineering features...")
        features = pd.DataFrame(index=df.index)
        
        # VIX level (already informative)
        if 'VIXCLS' in df.columns:
            features['vix_level'] = df['VIXCLS']
        
        # Interest rate level
        if 'GS10' in df.columns:
            features['gs10_level'] = df['GS10']
        
        # Unemployment level
        if 'UNRATE' in df.columns:
            features['unrate_level'] = df['UNRATE']
        
        # Fed Funds level
        if 'FEDFUNDS' in df.columns:
            features['fedfunds_level'] = df['FEDFUNDS']
        
        logger.info(f"Created {features.shape[1]} features")
        return features

    
    def standardize(self, features: pd.DataFrame) -> pd.DataFrame:
        """Standardize features to z-scores"""
        logger.info("Standardizing features...")
        
        # First, resample to monthly frequency to handle mixed frequencies
        logger.info("Resampling to monthly frequency...")
        features_monthly = features.resample('MS').mean()  # MS = Month Start
        logger.info(f"After resampling: {len(features_monthly)} months")
        
        # Standardize
        standardized = (features_monthly - features_monthly.mean()) / features_monthly.std()
        
        # Forward fill missing values (max 3 months)
        standardized = standardized.ffill(limit=3)
        
        # Drop any remaining NaN rows
        standardized = standardized.dropna()
        
        logger.info(f"Standardized: {len(standardized)} observations")
        return standardized


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    api_key = os.getenv('FRED_API_KEY')
    
    loader = FREDDataLoader(api_key=api_key, series_ids=['VIXCLS', 'GS10'])
    raw = loader.download_all('2020-01-01', '2024-12-31')
    features = loader.engineer_features(raw)
    standardized = loader.standardize(features)
    print("\n✓ Data loader test complete!")
    print(standardized.head())
