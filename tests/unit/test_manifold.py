import numpy as np
from domainml.constraints.manifold import ManifoldIntegrationMixin
from domainml.core.metadata import FeatureMetadata

def test_laplacian_computation():
    mixin = ManifoldIntegrationMixin()
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    meta = FeatureMetadata(['f1', 'f2']) 
    
    L = mixin.compute_laplacian(X, metadata=meta, n_neighbors=2)
    assert L.shape == (5, 5)
    
    meta2 = FeatureMetadata(['f1', 'f2'], control_flags=[True, True])
    L2 = mixin.compute_laplacian(X, metadata=meta2)
    assert np.all(L2 == 0)
