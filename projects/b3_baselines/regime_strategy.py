"""
B3 + B4 Integration - Regime-Conditioned Momentum Strategy
Only trades momentum when in favorable regimes
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class RegimeConditionedMomentum:
    """Momentum strategy that only trades in favorable regimes"""
    
    def __init__(self, allowed_regimes=[0], top_pct=30, bottom_pct=30):
        """
        Initialize regime-conditioned momentum
        
        Args:
            allowed_regimes: List of regimes where we allow trading (default: [0] = Expansion only)
            top_pct: Top % to go long
            bottom_pct: Bottom % to go short
        """
        self.allowed_regimes = allowed_regimes
        self.top_pct = top_pct
        self.bottom_pct = bottom_pct
        logger.info(f"Regime-Conditioned Momentum: Trade only in regimes {allowed_regimes}")
    
    def generate_signals(self, features: pd.DataFrame, regime_labels: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on momentum and regime
        
        Args:
            features: Feature data with momentum indicators
            regime_labels: Regime predictions from B4
            
        Returns:
            DataFrame with signals
        """
        logger.info(f"Generating regime-conditioned signals for {len(features)} observations...")
        
        # Merge features with regime labels
        merged = features.copy()
        merged['date'] = pd.to_datetime(merged['date'])
        
        # Convert regime date to datetime
        regime_labels['date'] = pd.to_datetime(regime_labels['date'])
        
        # Merge on date (broadcast monthly regimes to daily data)
        merged['year_month'] = merged['date'].dt.to_period('M')
        regime_labels['year_month'] = pd.to_datetime(regime_labels['date']).dt.to_period('M')
        
        merged = merged.merge(
            regime_labels[['year_month', 'regime']], 
            on='year_month', 
            how='left'
        )
        
        # Forward-fill regime for any missing dates
        merged['regime'] = merged['regime'].fillna(method='ffill')
        
        # Calculate momentum signals (using 60-day momentum)
        merged['signal'] = 0.0
        
        for date in merged['date'].unique():
            date_data = merged[merged['date'] == date].copy()
            current_regime = date_data['regime'].iloc[0] if len(date_data) > 0 else None
            
            # Only generate signals in allowed regimes
            if current_regime in self.allowed_regimes:
                # Sort by momentum
                if 'momentum_60d' in date_data.columns:
                    momentum_col = 'momentum_60d'
                elif 'returns_60d' in date_data.columns:
                    momentum_col = 'returns_60d'
                else:
                    logger.warning(f"No momentum column found for date {date}")
                    continue
                
                date_data = date_data.sort_values(momentum_col, ascending=False)
                
                n_long = int(len(date_data) * self.top_pct / 100)
                n_short = int(len(date_data) * self.bottom_pct / 100)
                
                # Long top performers
                top_tickers = date_data.head(n_long)['ticker'].values
                merged.loc[(merged['date'] == date) & (merged['ticker'].isin(top_tickers)), 'signal'] = 1.0
                
                # Short bottom performers
                bottom_tickers = date_data.tail(n_short)['ticker'].values
                merged.loc[(merged['date'] == date) & (merged['ticker'].isin(bottom_tickers)), 'signal'] = -1.0
            else:
                # In unfavorable regimes, go to cash (signal = 0)
                merged.loc[merged['date'] == date, 'signal'] = 0.0
        
        logger.info(f"✓ Generated signals with regime conditioning")
        logger.info(f"  Long signals: {(merged['signal'] == 1.0).sum()}")
        logger.info(f"  Short signals: {(merged['signal'] == -1.0).sum()}")
        logger.info(f"  Cash (regime filter): {(merged['signal'] == 0.0).sum()}")
        
        return merged[['date', 'ticker', 'signal', 'regime']]


def load_latest_regime_labels() -> pd.DataFrame:
    """Load the most recent regime labels from B4"""
    import glob
    
    regime_files = glob.glob('data/regimes/regime_labels_*.csv')
    if not regime_files:
        raise FileNotFoundError("No regime label files found. Run B4 first!")
    
    # Get the most recent file
    latest_file = max(regime_files)
    logger.info(f"Loading regime labels from: {latest_file}")
    
    regime_df = pd.read_csv(latest_file)
    logger.info(f"Loaded {len(regime_df)} regime labels")
    
    return regime_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Test: Load regime labels
    regimes = load_latest_regime_labels()
    print("\nRegime Labels:")
    print(regimes.head(10))
    print(f"\nRegime distribution:")
    print(regimes['regime'].value_counts().sort_index())
    
    print("\n✓ Regime integration test complete!")
