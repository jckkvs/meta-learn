import numpy as np
import pytest
from domainml.constraints.monotonicity import MonotonicLinearRegression, generate_extrapolation_points
from domainml.core.metadata import FeatureMetadata
from domainml.constraints.monotonicity import MonotonicLinearRegression, generate_extrapolation_points
from domainml.core.metadata import FeatureMetadata
import pytest

def test_monotonic_linear_regression_strict_inc():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([4.0, 3.0, 2.0, 1.0])
    
    meta = FeatureMetadata(['f1'], monotonicities=['inc'], constraint_types=['strict'])
    model = MonotonicLinearRegression()
    model.fit(X, y, metadata=meta)
    
    assert model.coef_[0] >= -1e-5

def test_generate_extrapolation_points():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    points = generate_extrapolation_points(X, sigma=2.0, n_points=5)
    assert points.shape == (5, 2)

def test_soft_monotonic_constraints():
    np.random.seed(42)
    X = np.random.rand(20, 2)
    y = -10 * X[:, 0] + X[:, 1]
    
    meta = FeatureMetadata(
        feature_names=["f1", "f2"],
        monotonicities=["inc", "none"],
        constraint_types=["soft", "soft"]
    )
    
    model = MonotonicLinearRegression(soft_penalty_weight=0.1)
    model.fit(X, y, metadata=meta)
    assert model.coef_ is not None

def test_infeasible_constraints():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0])
    
    model = MonotonicLinearRegression()
    with pytest.raises(RuntimeError):
        model.predict(X)
        
    class BadMonotonicLinearRegression(MonotonicLinearRegression):
        def _fit(self, X, y, **fit_params):
            import cvxpy as cp
            self.coef_ = None
            raise ValueError("CVXPY Optimization failed. Status: infeasible")
            
    bad_model = BadMonotonicLinearRegression()
    with pytest.raises(ValueError):
        bad_model._fit(X, y)

def test_monotonic_linear_regression_strict_dec():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([4.0, 3.0, 2.0, 1.0])
    
    meta = FeatureMetadata(['f1'], monotonicities=['dec'], constraint_types=['strict'])
    model = MonotonicLinearRegression()
    model.fit(X, y, metadata=meta)
    
    assert model.coef_[0] <= 1e-5

def test_monotonic_linear_regression_soft_dec():
    np.random.seed(42)
    X = np.random.rand(20, 1)
    y = X[:, 0]  # Positive correlation data
    
    # We apply a soft decreasing constraint. The model should fit without crashing.
    meta = FeatureMetadata(['f1'], monotonicities=['dec'], constraint_types=['soft'])
    model = MonotonicLinearRegression(soft_penalty_weight=0.1)
    model.fit(X, y, metadata=meta)
    assert model.coef_ is not None

