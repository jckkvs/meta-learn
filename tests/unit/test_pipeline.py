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

class StandardTransformer(BaseEstimator, TransformerMixin):
    # Does not accept metadata
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X + 1

class CustomFitTransform(BaseEstimator, TransformerMixin):
    def fit_transform(self, X, y=None, metadata=None, **fit_params):
        self.metadata_in = metadata
        self.param = fit_params.get("myparam", None)
        return X * 3
    def transform(self, X):
        return X * 3

def test_pipeline_with_standard_sklearn_components():
    meta = FeatureMetadata(['f1'])
    from sklearn.linear_model import LinearRegression
    pipeline = MetaPipeline([
        ('transformer', StandardTransformer()),
        ('estimator', LinearRegression())
    ])
    X = np.array([[1.0], [2.0]])
    y = np.array([2.0, 3.0])
    
    pipeline.fit(X, y, metadata=meta) # Should not crash and should bypass metadata
    preds = pipeline.predict(X, metadata=meta)
    assert len(preds) == 2

def test_pipeline_passthrough_and_params():
    meta = FeatureMetadata(['f1'])
    pipeline = MetaPipeline([
        ('passthrough_step', 'passthrough'),
        ('ft_transformer', CustomFitTransform()),
        ('passthrough_est', 'passthrough')
    ])
    X = np.array([[1.0], [2.0]])
    y = np.array([1.0, 2.0])
    
    # Passing fit_params through step routing
    pipeline.fit(X, y, metadata=meta, ft_transformer__myparam=42)
    transformer = pipeline.named_steps['ft_transformer']
    assert transformer.metadata_in is not None
    assert transformer.param == 42
    
    preds = pipeline.predict(X, metadata=meta)
    assert np.allclose(preds, X * 3)
