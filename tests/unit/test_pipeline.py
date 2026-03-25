import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from domainml.core.pipeline import MetaPipeline
from domainml.core.metadata import FeatureMetadata

class DummyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None, metadata=None):
        self.metadata_in = metadata
        return self
        
    def transform(self, X, metadata=None):
        return X * 2
        
    def get_metadata_out(self, metadata):
        metadata.extrapolation_sigma = 5.0
        return metadata

class DummyEstimator(BaseEstimator):
    def fit(self, X, y=None, metadata=None):
        self.fitted_metadata_ = metadata
        return self
        
    def predict(self, X, metadata=None):
        self.predicted_metadata_ = metadata
        return np.sum(X, axis=1)

def test_meta_pipeline_propagation():
    meta = FeatureMetadata(['f1'])
    pipeline = MetaPipeline([
        ('transformer', DummyTransformer()),
        ('estimator', DummyEstimator())
    ])
    
    X = np.array([[1.0], [2.0]])
    y = np.array([1.0, 2.0])
    
    pipeline.fit(X, y, metadata=meta)
    
    assert pipeline.named_steps['transformer'].metadata_in is not None
    
    final_estimator = pipeline.named_steps['estimator']
    assert final_estimator.fitted_metadata_ is not None
    assert final_estimator.fitted_metadata_.extrapolation_sigma == 5.0

    preds = pipeline.predict(X, metadata=meta)
    assert final_estimator.predicted_metadata_ is not None
    assert np.allclose(preds, [2.0, 4.0])
