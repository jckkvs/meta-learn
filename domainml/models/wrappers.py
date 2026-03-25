import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.isotonic import IsotonicRegression
from domainml.core.metadata import FeatureMetadata

class MonotonicityWrapper(BaseEstimator):
    """
    任意の scikit-learn モデルの出力結果に対して、
    Isotonic Regression を用いて事後的に単調性を持たせるラッパー
    """
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        
    def fit(self, X: np.ndarray, y: np.ndarray, metadata: FeatureMetadata = None, **fit_params):
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X, y, **fit_params)
        
        self.primary_mono_idx_ = -1
        self.primary_mono_direction_ = 'inc'
        
        if metadata is not None:
             for i, mono in enumerate(metadata.monotonicities):
                 if mono in ['inc', 'dec']:
                     self.primary_mono_idx_ = i
                     self.primary_mono_direction_ = mono
                     break 
                     
        if self.primary_mono_idx_ != -1:
             preds = self.estimator_.predict(X)
             increasing = (self.primary_mono_direction_ == 'inc')
             self.calibrator_ = IsotonicRegression(increasing=increasing, out_of_bounds='clip')
             feature_val = X[:, self.primary_mono_idx_]
             self.calibrator_.fit(feature_val, preds)
                     
        return self

    def predict(self, X: np.ndarray, metadata: FeatureMetadata = None) -> np.ndarray:
        preds = self.estimator_.predict(X)
        if hasattr(self, 'calibrator_'):
            feature_val = X[:, self.primary_mono_idx_]
            preds = self.calibrator_.predict(feature_val)
        return preds
