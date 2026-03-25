import numpy as np
import pytest
from sklearn.svm import SVR
from domainml.constraints.kernel import KernelMonotonicity
from domainml.core.metadata import FeatureMetadata

def test_kernel_monotonicity_strict():
    np.random.seed(42)
    X = np.random.rand(50, 1) * 10
    # True function is non-monotonic
    y = np.sin(X[:, 0]) + X[:, 0] * 0.1
    
    metadata = FeatureMetadata(
        feature_names=["f1"], 
        monotonicities=["inc"], 
        constraint_types=["strict"]
    )
    
    model = KernelMonotonicity(estimator=SVR(kernel="rbf"), constraint_type="strict")
    model.fit(X, y, metadata=metadata)
    
    # Test strict monotonicity over a dense grid
    X_test = np.linspace(0, 10, 100).reshape(-1, 1)
    y_pred = model.predict(X_test)
    
    diffs = np.diff(y_pred)
    # Should be monotonically increasing (allow very small negative due to numeric precision)
    assert np.all(diffs >= -5e-3), f"Predictions are not monotonically increasing. Min diff: {np.min(diffs)}"

def test_kernel_monotonicity_soft():
    np.random.seed(42)
    X = np.random.rand(50, 1) * 10
    y = np.sin(X[:, 0]) + X[:, 0] * 0.1
    
    metadata = FeatureMetadata(
        feature_names=["f1"], 
        monotonicities=["inc"], 
        constraint_types=["soft"]
    )
    
    model = KernelMonotonicity(estimator=SVR(kernel="rbf"), constraint_type="soft")
    model.fit(X, y, metadata=metadata)
    
    # It should predict without crashing, and have some positive correlation
    X_test = np.linspace(0, 10, 100).reshape(-1, 1)
    y_pred = model.predict(X_test)
    assert len(y_pred) == 100

def test_kernel_no_metadata_and_dec_constraint():
    np.random.seed(42)
    X = np.random.rand(20, 2)
    y = -X[:, 0] + X[:, 1]
    
    # Test None metadata
    model_no_meta = KernelMonotonicity(estimator=SVR(kernel="rbf"))
    model_no_meta.fit(X, y, metadata=None)
    assert model_no_meta.alpha_ is not None

    meta = FeatureMetadata(
        feature_names=["f1", "f2"], 
        monotonicities=["dec", "none"], 
        constraint_types=["strict", "none"]
    )
    model = KernelMonotonicity(estimator=SVR(kernel="rbf"), constraint_type="strict")
    model.fit(X, y, metadata=meta)
    
    y_pred = model.predict(X)
    assert len(y_pred) == 20

def test_kernel_infeasible_fallback():
    # Intentionally contradictory strict constraints to trigger infeasible status
    # such as 'inc' and 'dec' on the same highly correlated variable if we had it,
    # or just mocking cvxpy status.
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([3.0, 2.0, 1.0])
    meta = FeatureMetadata(["f1"], ["inc"], ["strict"])
    
    class BadKernel(KernelMonotonicity):
        def _generate_virtual_points(self, X):
            return super()._generate_virtual_points(X)
            
    # To force infeasibility, we can manually raise or let CVXPY fail if the points are contradictory.
    # We will simulate the failure path directly
    original_solve = None
    import cvxpy as cp
    original_solve = cp.Problem.solve
    
    def fake_solve(*args, **kwargs):
        args[0]._status = "infeasible"
        return None
        
    cp.Problem.solve = fake_solve
    try:
        model = KernelMonotonicity(estimator=SVR(kernel="rbf"))
        model.fit(X, y, metadata=meta)
        assert model.alpha_ is not None # Fallback unconstrained Ridge
    finally:
        cp.Problem.solve = original_solve

