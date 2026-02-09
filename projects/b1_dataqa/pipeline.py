"""B1 Pipeline - Orchestrates ingestion → cleaning → validation"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from projects.b1_dataqa.ingestion import MarketDataIngestion
from projects.b1_dataqa.cleaning import DataCleaner
from projects.b1_dataqa.qa_checks import DataQAValidator

import pandas as pd
import yaml
import json
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


class B1DataPipeline:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = "projects/b1_dataqa/configs/b1_config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.version_date = datetime.now().strftime('%Y%m%d')
    
    def run(self):
        """Run complete B1 pipeline"""
        try:
            logger.info("=" * 70)
            logger.info("🚀 B1 DATA QA PIPELINE STARTED")
            logger.info("=" * 70)
            
            # Stage 1: Ingestion
            logger.info("\n📥 [1/4] INGESTION")
            market_data, macro_data = self.ingest_data()
            logger.info(f"✓ Got {len(market_data)} market records")
            
            # Stage 2: Cleaning
            logger.info("\n🧹 [2/4] CLEANING")
            market_data, cleaning_report = self.clean_data(market_data)
            logger.info(f"✓ Cleaned to {len(market_data)} records")
            
            # Stage 3: Validation
            logger.info("\n🔍 [3/4] VALIDATION")
            qa_passed, qa_report = self.validate_data(market_data)
            
            # Stage 4: Save
            logger.info("\n💾 [4/4] SAVING")
            self.save_data(market_data, macro_data, cleaning_report, qa_report)
            
            logger.info("\n" + "=" * 70)
            logger.info("✅ B1 PIPELINE COMPLETE")
            logger.info(f"📊 Records: {len(market_data):,}")
            logger.info(f"📅 Range: {market_data['date'].min()} to {market_data['date'].max()}")
            logger.info(f"✓ QA: {'PASS' if qa_passed else 'FAIL'}")
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
            raise
    
    def ingest_data(self):
        """Download data"""
        ingester = MarketDataIngestion(self.config)
        tickers = ingester.get_sp500_tickers(limit=100)
        market_data = ingester.download_market_data(tickers)
        macro_data = ingester.download_macro_data()
        ingester.save_raw_snapshots(market_data, macro_data, self.version_date)
        return market_data, macro_data
    
    def clean_data(self, market_data):
        """Clean data"""
        cleaner = DataCleaner(market_data, self.config)
        cleaner.handle_missing_values() \
               .remove_duplicates() \
               .detect_outliers() \
               .validate_ohlc_relationships()
        return cleaner.get_cleaned_data(), cleaner.get_report()
    
    def validate_data(self, data):
        """Run QA checks"""
        validator = DataQAValidator(data, self.config)
        passed = validator.run_all_checks()
        report = validator.get_report()
        
        # Save QA report
        report_path = f"projects/b1_dataqa/outputs/qa_reports/qa_{self.version_date}.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"📄 QA report: {report_path}")
        
        return passed, report
    
    def save_data(self, market_data, macro_data, cleaning_report, qa_report):
        """Save final data"""
        # Save clean data
        clean_path = f"data/market/market_clean_{self.version_date}.parquet"
        market_data.to_parquet(clean_path, compression='snappy', index=False)
        logger.info(f"✓ Saved: {clean_path}")
        
        # Save manifest
        manifest = {
            'version': f"market_clean_{self.version_date}",
            'timestamp': datetime.now().isoformat(),
            'records': len(market_data),
            'tickers': int(market_data['ticker'].nunique()),
            'qa_status': qa_report.get('overall_status', 'UNKNOWN')
        }
        
        manifest_path = f"data/snapshots/manifest_{self.version_date}.json"
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"✓ Manifest: {manifest_path}")


if __name__ == "__main__":
    pipeline = B1DataPipeline()
    pipeline.run()
