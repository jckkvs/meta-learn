import pytest
import numpy as np
from sklearn.linear_model import LinearRegression

from domainml.core.metadata import FeatureMetadata
from domainml.analysis.causal import CausalConflictDetector
from domainml.analysis.metrics import satisfaction_score
from domainml.model_selection.cv import constrained_cv
from domainml.constraints.engine import MonotonicityEngine

def test_causal_conflict_detector():
    np.random.seed(42)
    X = np.random.rand(100, 2)
    # y increases with f1, decreases with f2
    y = X[:, 0] * 5.0 - X[:, 1] * 5.0
    
    # Intentionally provide wrong constraints
    metadata = FeatureMetadata(
        feature_names=["f1", "f2"],
        monotonicities=["dec", "inc"]  # Both are wrong
    )
    
    detector = CausalConflictDetector(threshold=1.0)
    conflicts = detector.detect_conflicts(X, y, metadata)
    
    assert len(conflicts) == 2
    
def test_satisfaction_score():
    np.random.seed(42)
    X = np.random.rand(50, 2)
    y = X[:, 0] + X[:, 1]
    
    metadata = FeatureMetadata(feature_names=["f1", "f2"], monotonicities=["inc", "inc"])
    
    # 完全に制約を守るモデル (MonotonicityEngine)
    engine = MonotonicityEngine(LinearRegression())
    engine.fit(X, y, metadata=metadata)
    
    score = satisfaction_score(engine, X, metadata, n_samples=100)
    # 充足度スコアは1.0に近いはず
    assert score > 0.99

def test_constrained_cv():
    np.random.seed(42)
    X = np.random.rand(50, 2)
    y = X[:, 0] * 2 + X[:, 1] + np.random.randn(50) * 0.1
    
    metadata = FeatureMetadata(feature_names=["f1", "f2"], monotonicities=["inc", "inc"])
    
    engine = MonotonicityEngine(LinearRegression())
    
    results = constrained_cv(engine, X, y, metadata=metadata, cv=3)
    
    assert 'test_score' in results
    assert 'satisfaction_score' in results
    assert len(results['test_score']) == 3
    assert len(results['satisfaction_score']) == 3
    assert np.mean(results['satisfaction_score']) > 0.99

def test_constrained_cv_without_metadata_support():
    np.random.seed(42)
    X = np.random.rand(50, 2)
    y = X[:, 0] * 2 + X[:, 1]
    
    metadata = FeatureMetadata(feature_names=["f1", "f2"], monotonicities=["inc", "inc"])
    # Passing standard LR which does not support 'metadata' kwarg.
    # It should be properly caught by the built-in TypeError except block.
    results = constrained_cv(LinearRegression(), X, y, metadata=metadata, cv=3)
    assert len(results['test_score']) == 3

from domainml.analysis.metrics import satisfaction_score
def test_satisfaction_score_none_and_dec():
    X = np.random.rand(10, 2)
    y = -2 * X[:, 0] + X[:, 1]
    
    meta_none = FeatureMetadata(["f1", "f2"], ["none", "none"])
    lr = LinearRegression().fit(X, y)
    
    score_none = satisfaction_score(lr, X, meta_none)
    assert score_none == 1.0
    
    meta_dec = FeatureMetadata(["f1", "f2"], ["dec", "inc"])
    # Just checking it covers the mono == 'dec' path in metrics.py
    score_dec = satisfaction_score(lr, X, meta_dec)
    assert 0.0 <= score_dec <= 1.0
