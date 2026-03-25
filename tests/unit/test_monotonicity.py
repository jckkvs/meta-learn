import numpy as np
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
