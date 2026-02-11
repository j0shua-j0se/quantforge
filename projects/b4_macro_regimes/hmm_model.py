"""
B4 - Hidden Markov Model for Regime Detection
"""

import logging
import numpy as np
import pandas as pd
from hmmlearn import hmm

logger = logging.getLogger(__name__)


class RegimeHMM:
    """Hidden Markov Model for detecting market regimes"""
    
    def __init__(self, n_regimes=3, n_iter=100, random_state=42):
        self.n_regimes = n_regimes
        self.model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="diag",
            n_iter=n_iter,
            random_state=random_state
        )
        self.is_fitted = False
        logger.info(f"Initialized HMM with {n_regimes} regimes")
    
    def fit(self, features: pd.DataFrame):
        """Fit HMM to features"""
        logger.info(f"Fitting HMM to {len(features)} observations...")
        X = features.values
        self.model.fit(X)
        self.is_fitted = True
        logger.info("✓ HMM fitted successfully")
        return self
    
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict regimes"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        X = features.values
        regimes = self.model.predict(X)
        probs = self.model.predict_proba(X)
        
        result = pd.DataFrame({
            'date': features.index,
            'regime': regimes
        })
        
        for i in range(self.n_regimes):
            result[f'prob_regime_{i}'] = probs[:, i]
        
        logger.info(f"✓ Predicted {len(result)} regime labels")
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with dummy data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    features = pd.DataFrame(
        np.random.randn(100, 3),
        index=dates,
        columns=['feat1', 'feat2', 'feat3']
    )
    
    model = RegimeHMM(n_regimes=3)
    model.fit(features)
    predictions = model.predict(features)
    
    print("\n✓ HMM test complete!")
    print(predictions.head(10))
