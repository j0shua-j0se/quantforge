"""B1 Data Cleaning - Handle missing values, duplicates, outliers"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    def __init__(self, market_data: pd.DataFrame, config: dict):
        self.data = market_data.copy()
        self.config = config
        self.cleaning_report = {}
        
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'])
    
    def handle_missing_values(self):
        """Fill small gaps, remove rows with critical missing data"""
        logger.info("🧹 Handling missing values...")
        initial_nulls = self.data.isnull().sum().sum()
        
        # Forward fill by ticker (max 5 days)
        for ticker in self.data['ticker'].unique():
            mask = self.data['ticker'] == ticker
            for col in ['close', 'open', 'high', 'low']:
                if col in self.data.columns:
                    self.data.loc[mask, col] = self.data.loc[mask, col].ffill(limit=5)
        
        # Drop remaining nulls
        self.data = self.data.dropna(subset=['close'])
        final_nulls = self.data.isnull().sum().sum()
        
        self.cleaning_report['missing_values'] = {
            'before': int(initial_nulls),
            'after': int(final_nulls)
        }
        logger.info(f"  ✓ Nulls: {initial_nulls} → {final_nulls}")
        return self
    
    def remove_duplicates(self):
        """Remove duplicate date-ticker entries"""
        logger.info("🧹 Removing duplicates...")
        initial = len(self.data)
        self.data = self.data.drop_duplicates(subset=['date', 'ticker'], keep='last')
        removed = initial - len(self.data)
        
        self.cleaning_report['duplicates'] = int(removed)
        logger.info(f"  ✓ Removed {removed} duplicates")
        return self
    
    def detect_outliers(self):
        """Flag outliers using IQR method"""
        logger.info("🔍 Detecting outliers...")
        self.data = self.data.sort_values(['ticker', 'date'])
        self.data['returns'] = self.data.groupby('ticker')['close'].pct_change()
        
        Q1 = self.data['returns'].quantile(0.25)
        Q3 = self.data['returns'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = (self.data['returns'] < Q1 - 3*IQR) | (self.data['returns'] > Q3 + 3*IQR)
        
        self.data['is_outlier'] = outliers
        self.cleaning_report['outliers'] = {'count': int(outliers.sum())}
        logger.info(f"  ✓ Found {outliers.sum()} outliers")
        return self
    
    def validate_ohlc_relationships(self):
        """Check High >= Low, etc."""
        logger.info("🔍 Validating OHLC...")
        violations = ((self.data['high'] < self.data['low']) | 
                     (self.data['high'] < self.data['close']) |
                     (self.data['low'] > self.data['close'])).sum()
        
        self.cleaning_report['ohlc_violations'] = int(violations)
        logger.info(f"  ✓ OHLC violations: {violations}")
        return self
    
    def get_cleaned_data(self):
        return self.data
    
    def get_report(self):
        return self.cleaning_report
