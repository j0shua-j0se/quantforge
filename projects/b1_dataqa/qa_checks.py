"""B1 QA Checks - Validate data quality"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataQAValidator:
    def __init__(self, data: pd.DataFrame, config: dict):
        self.data = data.copy()
        self.config = config
        self.qa_report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'checks': {}
        }
    
    def check_null_values(self):
        """Check for null values"""
        logger.info("  [1/3] Checking nulls...")
        nulls = self.data.isnull().sum().sum()
        threshold = self.config['quality_checks']['missing_data']['threshold']
        null_pct = nulls / (len(self.data) * len(self.data.columns))
        passed = null_pct <= threshold
        
        self.qa_report['checks']['null_values'] = {
            'status': 'PASS' if passed else 'FAIL',
            'null_pct': float(null_pct * 100)
        }
        logger.info(f"    {'✓' if passed else '✗'} Nulls: {null_pct*100:.2f}%")
        return passed
    
    def check_volume_sanity(self):
        """Check volume is positive"""
        logger.info("  [2/3] Checking volume...")
        zero_vol = (self.data['volume'] == 0).sum()
        neg_vol = (self.data['volume'] < 0).sum()
        passed = neg_vol == 0
        
        self.qa_report['checks']['volume'] = {
            'status': 'PASS' if passed else 'FAIL',
            'zero_volume': int(zero_vol),
            'negative_volume': int(neg_vol)
        }
        logger.info(f"    {'✓' if passed else '✗'} Volume OK")
        return passed
    
    def check_data_coverage(self):
        """Check data coverage"""
        logger.info("  [3/3] Checking coverage...")
        coverage = self.data.groupby('ticker').size()
        mean_cov = coverage.mean()
        
        self.qa_report['checks']['coverage'] = {
            'status': 'PASS',
            'mean_records': int(mean_cov)
        }
        logger.info(f"    ✓ Coverage: {int(mean_cov)} records/ticker")
        return True
    
    def run_all_checks(self):
        """Run all QA checks"""
        logger.info("🔍 Running QA checks...")
        
        results = [
            self.check_null_values(),
            self.check_volume_sanity(),
            self.check_data_coverage()
        ]
        
        all_passed = all(results)
        self.qa_report['overall_status'] = 'PASS' if all_passed else 'FAIL'
        self.qa_report['checks_passed'] = sum(results)
        self.qa_report['checks_total'] = len(results)
        
        logger.info(f"✓ QA: {sum(results)}/{len(results)} passed")
        return all_passed
    
    def get_report(self):
        return self.qa_report
