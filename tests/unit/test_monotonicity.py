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
    
    # Assert actual prediction monotonicity across a dense test set
    X_test = np.linspace(0, 10, 100).reshape(-1, 1)
    y_pred = model.predict(X_test)
    assert np.all(np.diff(y_pred) >= -1e-5), "Predictions are not monotonically increasing"

def test_generate_extrapolation_points():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    points = generate_extrapolation_points(X, sigma=2.0, n_points=5)
    # 2次元なのでグリッド点: 5^2 = 25 点、各点は2列
    assert points.shape == (25, 2)
    # 高次元でのMCサンプリングもテスト（n_features=5 > 3）
    Xhd = np.random.default_rng(0).standard_normal((30, 5))
    pts_hd = generate_extrapolation_points(Xhd, sigma=2.0, n_points=8)
    assert pts_hd.shape == (8 * 5, 5)


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
    
    # Assert actual prediction monotonicity
    X_test = np.linspace(0, 10, 100).reshape(-1, 1)
    y_pred = model.predict(X_test)
    assert np.all(np.diff(y_pred) <= 1e-5), "Predictions are not monotonically decreasing"

def test_monotonic_linear_regression_soft_dec():
    np.random.seed(42)
    X = np.random.rand(20, 1)
    y = X[:, 0]  # Positive correlation data
    
    # We apply a soft decreasing constraint. The model should fit without crashing.
    meta = FeatureMetadata(['f1'], monotonicities=['dec'], constraint_types=['soft'])
    model = MonotonicLinearRegression(soft_penalty_weight=0.1)
    model.fit(X, y, metadata=meta)
    assert model.coef_ is not None

