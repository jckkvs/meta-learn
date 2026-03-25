import numpy as np
import pytest
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from domainml.constraints.engine import MonotonicityEngine
from domainml.core.metadata import FeatureMetadata

def test_engine_no_constraints():
    X = np.random.rand(100, 2)
    y = X[:, 0] + X[:, 1]
    metadata = FeatureMetadata(feature_names=["f1", "f2"], monotonicities=["none", "none"])
    
    engine = MonotonicityEngine(LinearRegression())
    engine.fit(X, y, metadata=metadata)
    
    assert isinstance(engine.model_, LinearRegression)

def test_engine_tree_native():
    X = np.random.rand(100, 2)
    y = X[:, 0] - X[:, 1]
    metadata = FeatureMetadata(feature_names=["f1", "f2"], monotonicities=["inc", "dec"])
    
    engine = MonotonicityEngine(LGBMRegressor(n_estimators=10))
    engine.fit(X, y, metadata=metadata)
    
    cst = getattr(engine.model_, 'monotone_constraints', getattr(engine.model_, 'monotonic_cst', None))
    assert cst is not None
    cst = list(cst)
    assert cst[0] == 1
    assert cst[1] == -1

def test_engine_linear_fallback():
    X = np.random.rand(100, 2)
    y = X[:, 0] + X[:, 1]
    metadata = FeatureMetadata(feature_names=["f1", "f2"], monotonicities=["inc", "inc"])
    
    engine = MonotonicityEngine(LinearRegression())
    engine.fit(X, y, metadata=metadata)
    
    from domainml.constraints.monotonicity import MonotonicLinearRegression
    assert isinstance(engine.model_, MonotonicLinearRegression)

def test_engine_kernel_fallback():
    X = np.random.rand(20, 2)
    y = X[:, 0] + X[:, 1]
    metadata = FeatureMetadata(feature_names=["f1", "f2"], monotonicities=["inc", "inc"])
    
    engine = MonotonicityEngine(SVR(kernel="rbf"))
    engine.fit(X, y, metadata=metadata)
    
    from domainml.constraints.kernel import KernelMonotonicity
    assert isinstance(engine.model_, KernelMonotonicity)

def test_engine_general_fallback():
    X = np.random.rand(20, 2)
    y = X[:, 0] + X[:, 1]
    metadata = FeatureMetadata(feature_names=["f1", "f2"], monotonicities=["inc", "inc"])
    
    engine = MonotonicityEngine(KNeighborsRegressor(n_neighbors=3))
    engine.fit(X, y, metadata=metadata)
    
class DummyTreeRegressor(BaseEstimator, RegressorMixin):
    def set_params(self, **params):
        if 'monotone_constraints' in params or 'monotonic_cst' in params:
            raise ValueError("Not supported")
        return self
    def fit(self, X, y):
        return self
    def predict(self, X):
        return np.sum(X, axis=1)

def test_engine_tree_fallback_and_predict_no_metadata():
    X = np.random.rand(20, 2)
    y = X[:, 0] - X[:, 1]
    metadata = FeatureMetadata(feature_names=["f1", "f2"], monotonicities=["inc", "dec"])
    
    engine = MonotonicityEngine(DummyTreeRegressor())
    engine.fit(X, y, metadata=metadata)
    
    from domainml.models.wrappers import MonotonicityWrapper
    assert isinstance(engine.model_, MonotonicityWrapper)
    
    preds = engine.predict(X)
    assert len(preds) == 20

def test_engine_tree_histgradient_native():
    X = np.random.rand(100, 2)
    y = X[:, 0] - X[:, 1]
    metadata = FeatureMetadata(feature_names=["f1", "f2"], monotonicities=["inc", "dec"])
    
    # HistGradientBoosting uses 'monotonic_cst' parameter instead of 'monotone_constraints'
    engine = MonotonicityEngine(HistGradientBoostingRegressor(max_iter=10))
    engine.fit(X, y, metadata=metadata)
    
    assert getattr(engine.model_, 'monotonic_cst') is not None
    assert engine.model_.monotonic_cst[0] == 1
    
    preds = engine.predict(X)
    assert len(preds) == 100

