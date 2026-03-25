import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from domainml.core.metadata import FeatureMetadata
from domainml.models.uncertainty import UncertaintyEstimator
from domainml.core.cache import LazyConstraintEvaluator
from domainml.analysis.parallel import parallel_check_conflicts

def test_uncertainty_estimator():
    np.random.seed(42)
    X = np.random.rand(50, 2)
    y = X[:, 0] * 3.0 + X[:, 1] + np.random.randn(50) * 0.5
    
    metadata = FeatureMetadata(feature_names=["f1", "f2"], monotonicities=["inc", "inc"])
    
    estimator = UncertaintyEstimator(LinearRegression(), n_estimators=5, random_state=42)
    estimator.fit(X, y, metadata=metadata)
    
    X_test = np.random.rand(10, 2)
    mean_preds, lower, upper = estimator.predict(X_test, metadata=metadata, return_interval=True)
    
    assert len(mean_preds) == 10
    assert len(lower) == 10
    assert len(upper) == 10
    assert np.all(lower <= mean_preds)
    assert np.all(mean_preds <= upper)

def test_lazy_constraint_evaluator():
    LazyConstraintEvaluator.clear_cache()
    
    call_count = 0
    @LazyConstraintEvaluator.cache_evaluation
    def dummy_expensive_check(X: np.ndarray, metadata: FeatureMetadata):
        nonlocal call_count
        call_count += 1
        return np.sum(X)
        
    X = np.ones((10, 2))
    metadata = FeatureMetadata(["f1", "f2"])
    
    # 初回呼び出し
    res1 = dummy_expensive_check(X, metadata)
    assert call_count == 1
    assert res1 == 20.0
    
    # 2回目の呼び出し（キャッシュヒットするはず）
    res2 = dummy_expensive_check(X, metadata)
    assert call_count == 1
    assert res1 == res2
    
def test_parallel_check_conflicts():
    np.random.seed(42)
    X = np.random.rand(50, 2)
    # y increases with f1, decreases with f2
    y = X[:, 0] * 5.0 - X[:, 1] * 5.0
    
    # Both constraints are intentionally set incorrectly to force a conflict
    metadata = FeatureMetadata(
        feature_names=["f1", "f2"],
        monotonicities=["dec", "inc"]
    )
    
    conflicts = parallel_check_conflicts(X, y, metadata, threshold=1.0, n_jobs=1)
    
    assert len(conflicts) == 2
    for c in conflicts:
        assert c['feature_name'] in ["f1", "f2"]
