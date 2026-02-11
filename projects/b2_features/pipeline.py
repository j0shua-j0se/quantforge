"""
B2 Feature Store Pipeline
Main orchestrator that runs all feature engines.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import yaml
import json
import logging
from datetime import datetime
from pathlib import Path

# Import our feature engines
from modules.price_features import PriceFeaturesEngine
from modules.volatility_features import VolatilityFeaturesEngine
from modules.volume_features import VolumeFeaturesEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class B2FeaturePipeline:
    """Complete B2 pipeline: load → compute → validate → save."""
    
    def __init__(self, config_path="projects/b2_features/configs/b2_config.yaml"):
        logger.info("Initializing B2 Feature Pipeline...")
        self.config = self._load_config(config_path)
        self.report = {'start_time': datetime.now().isoformat()}
    
    def _load_config(self, config_path):
        """Load YAML configuration."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"✓ Config loaded: {config_path}")
        return config
    
    def load_data(self):
        """Stage 1: Load clean market data from B1."""
        logger.info("\n" + "="*60)
        logger.info("STAGE 1: LOADING DATA")
        logger.info("="*60)
        
        market_path = self.config['input']['market_data']
        logger.info(f"Loading: {market_path}")
        
        df = pd.read_parquet(market_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
        
        logger.info(f"✓ Loaded {len(df):,} records")
        logger.info(f"  Tickers: {df['ticker'].nunique()}")
        logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    def compute_features(self, df):
        """Stage 2: Compute all features."""
        logger.info("\n" + "="*60)
        logger.info("STAGE 2: COMPUTING FEATURES")
        logger.info("="*60)
        
        initial_cols = len(df.columns)
        
        # Price features
        if self.config['features']['price_momentum']['enabled']:
            price_engine = PriceFeaturesEngine(self.config)
            df = price_engine.compute_all(df)
        
        # Volatility features
        if self.config['features']['volatility']['enabled']:
            vol_engine = VolatilityFeaturesEngine(self.config)
            df = vol_engine.compute_all(df)
        
        # Volume features
        if self.config['features']['volume']['enabled']:
            volume_engine = VolumeFeaturesEngine(self.config)
            df = volume_engine.compute_all(df)
        
        features_added = len(df.columns) - initial_cols
        logger.info(f"\n✓ Total features added: {features_added}")
        
        return df, features_added
    
    def validate_features(self, df):
        """Stage 3: Quality checks."""
        logger.info("\n" + "="*60)
        logger.info("STAGE 3: VALIDATING FEATURES")
        logger.info("="*60)
        
        # Get feature columns (exclude original data columns)
        original_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 
                        'volume', 'dividends', 'stock_splits', 'returns', 'is_outlier']
        feature_cols = [col for col in df.columns if col not in original_cols]
        
        logger.info(f"Checking {len(feature_cols)} features...")
        
        # Check null percentages
        max_null_pct = self.config['quality']['max_null_pct']
        failed_features = []
        
        for col in feature_cols:
            null_pct = df[col].isnull().sum() / len(df)
            if null_pct > max_null_pct:
                failed_features.append(f"{col}: {null_pct:.2%}")
        
        if failed_features:
            logger.warning(f"⚠ {len(failed_features)} features exceed {max_null_pct*100}% null threshold")
            for feat in failed_features[:5]:
                logger.warning(f"  {feat}")
        else:
            logger.info(f"✓ All features pass null check (<{max_null_pct*100}%)")
        
        # Sample statistics
        logger.info("\nSample feature statistics:")
        for col in feature_cols[:3]:
            stats = df[col].describe()
            logger.info(f"  {col}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        
        return True
    
    def save_features(self, df, features_added):
        """Stage 4: Save feature matrix."""
        logger.info("\n" + "="*60)
        logger.info("STAGE 4: SAVING FEATURES")
        logger.info("="*60)
        
        # Create output directory
        output_dir = Path(self.config['storage']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save with version date
        version_date = datetime.now().strftime("%Y%m%d")
        output_path = output_dir / f"features_{version_date}.parquet"
        
        df.to_parquet(output_path, compression='snappy')
        file_size_mb = output_path.stat().st_size / 1024 / 1024
        
        logger.info(f"✓ Features saved: {output_path}")
        logger.info(f"  Size: {file_size_mb:.2f} MB")
        logger.info(f"  Records: {len(df):,}")
        logger.info(f"  Columns: {len(df.columns)}")
        
        # Create manifest
        feature_cols = [col for col in df.columns if col not in 
                       ['date', 'ticker', 'open', 'high', 'low', 'close', 
                        'volume', 'dividends', 'stock_splits']]
        
        manifest = {
            'version': f"features_{version_date}",
            'timestamp': datetime.now().isoformat(),
            'records': len(df),
            'tickers': int(df['ticker'].nunique()),
            'features_count': len(feature_cols),
            'features': feature_cols,
            'date_range': {
                'start': str(df['date'].min()),
                'end': str(df['date'].max())
            }
        }
        
        manifest_path = output_dir / f"manifest_{version_date}.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"✓ Manifest saved: {manifest_path}")
        
        return str(output_path)
    
    def run(self):
        """Execute complete pipeline."""
        try:
            logger.info("\n" + "="*60)
            logger.info("B2 FEATURE PIPELINE STARTED")
            logger.info("="*60)
            
            # Stage 1: Load
            df = self.load_data()
            
            # Stage 2: Compute
            df, features_added = self.compute_features(df)
            
            # Stage 3: Validate
            self.validate_features(df)
            
            # Stage 4: Save
            output_path = self.save_features(df, features_added)
            
            # Success summary
            logger.info("\n" + "="*60)
            logger.info("B2 PIPELINE COMPLETED SUCCESSFULLY ✓")
            logger.info("="*60)
            logger.info(f"Features computed: {features_added}")
            logger.info(f"Output: {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"\n❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    pipeline = B2FeaturePipeline()
    success = pipeline.run()
    
    if success:
        print("\n🎉 B2 Feature Store build complete!")
    else:
        print("\n❌ B2 build failed. Check logs above.")
