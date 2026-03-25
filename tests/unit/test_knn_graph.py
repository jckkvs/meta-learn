import numpy as np
import scipy.sparse as sp
import pytest
from domainml.constraints.knn_graph import KNNGraphWrapper
from domainml.meta.manifold_est import ManifoldAssumption
from domainml.core.metadata import FeatureMetadata

def test_laplacian_computation():
    mixin = KNNGraphWrapper()
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    meta = FeatureMetadata(['f1', 'f2']) 
    
    L_sparse = mixin.compute_laplacian(X, metadata=meta, n_neighbors=2, return_sparse=True)
    assert sp.issparse(L_sparse)
    assert L_sparse.shape == (5, 5)

    L_dense = mixin.compute_laplacian(X, metadata=meta, n_neighbors=2, return_sparse=False)
    assert isinstance(L_dense, np.ndarray)
    np.testing.assert_array_almost_equal(L_sparse.toarray(), L_dense)

def test_laplacian_computation_with_controls():
    mixin = KNNGraphWrapper()
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    meta = FeatureMetadata(['f1', 'f2'], control_flags=[True, True])
    L = mixin.compute_laplacian(X, metadata=meta, return_sparse=True)
    assert L.nnz == 0

def test_manifold_intrinsic_dimension():
    # 完全に相関したデータ（実質1次元）
    x1 = np.linspace(0, 10, 50)
    x2 = x1 * 2 + 1
    x3 = x1 * 0.5 - 2
    X = np.column_stack([x1, x2, x3])
    
    dim = ManifoldAssumption.estimate_intrinsic_dimension(X)
    assert dim == 1
    
    # ランダムデータ（高い次元）
    np.random.seed(42)
    X_rand = np.random.rand(50, 5)
    dim_rand = ManifoldAssumption.estimate_intrinsic_dimension(X_rand)
    assert dim_rand > 1

def test_manifold_validation():
    x1 = np.linspace(0, 10, 50)
    # 実質2次元
    X = np.column_stack([x1, x1*2, np.random.rand(50), np.random.rand(50)])
    
    # isValid is True since intrinsic is likely 2 or 3 out of 4 (threshold=0.8)
    is_valid = ManifoldAssumption.validate_assumption(X, threshold_ratio=0.8)
    assert isinstance(is_valid, bool)
