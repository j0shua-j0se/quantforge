"""
B4 Main Pipeline - Regime Detection
"""

import os
import logging
from datetime import datetime
import yaml
import pandas as pd
from data_loader import FREDDataLoader
from hmm_model import RegimeHMM

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main B4 pipeline"""
    logger.info("=" * 60)
    logger.info("B4 MACRO REGIME DETECTION - STARTING")
    logger.info("=" * 60)
    
    # Load config
    with open('projects/b4_macro_regimes/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get API key
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        raise ValueError("FRED_API_KEY not set")
    
    # Step 1: Download FRED data
    logger.info("\nStep 1: Downloading FRED data...")
    loader = FREDDataLoader(
        api_key=api_key,
        series_ids=config['data']['fred_series']
    )
    
    raw_data = loader.download_all(
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date']
    )
    
    # Step 2: Engineer features
    logger.info("\nStep 2: Engineering features...")
    features = loader.engineer_features(raw_data)
    
    # Step 3: Standardize
    logger.info("\nStep 3: Standardizing features...")
    standardized = loader.standardize(features)
    
    if len(standardized) == 0:
        logger.error("No valid data after standardization!")
        return
    
    # Step 4: Fit HMM
    logger.info("\nStep 4: Fitting HMM...")
    hmm_model = RegimeHMM(
        n_regimes=config['hmm']['n_regimes'],
        n_iter=config['hmm']['n_iter'],
        random_state=config['hmm']['random_state']
    )
    hmm_model.fit(standardized)
    
    # Step 5: Predict regimes
    logger.info("\nStep 5: Predicting regimes...")
    predictions = hmm_model.predict(standardized)
    
    # Step 6: Save results
    logger.info("\nStep 6: Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save regime labels
    output_file = f"data/regimes/regime_labels_{timestamp}.csv"
    predictions.to_csv(output_file, index=False)
    logger.info(f"✓ Saved regime labels: {output_file}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("REGIME DETECTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total observations: {len(predictions)}")
    logger.info(f"Date range: {predictions['date'].min()} to {predictions['date'].max()}")
    logger.info("\nRegime distribution:")
    for regime_id, count in predictions['regime'].value_counts().sort_index().items():
        pct = count / len(predictions) * 100
        logger.info(f"  Regime {regime_id}: {count} observations ({pct:.1f}%)")
    
    logger.info("\n✓ B4 Pipeline Complete!")


if __name__ == "__main__":
    main()
