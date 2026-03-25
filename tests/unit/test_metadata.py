import pytest
from domainml.core.metadata import FeatureMetadata

def test_feature_metadata_init():
    meta = FeatureMetadata(feature_names=['f1', 'f2'])
    assert meta.n_features == 2
    assert meta.monotonicities == ['none', 'none']
    
def test_feature_metadata_validation():
    with pytest.raises(ValueError):
        FeatureMetadata(feature_names=['f1', 'f2'], monotonicities=['inc'])
    with pytest.raises(ValueError):
        FeatureMetadata(feature_names=['f1', 'f2'], constraint_types=['strict'])
    with pytest.raises(ValueError):
        FeatureMetadata(feature_names=['f1', 'f2'], groups=[1])
    with pytest.raises(ValueError):
        FeatureMetadata(feature_names=['f1', 'f2'], manifold_flags=[True])
    with pytest.raises(ValueError):
        FeatureMetadata(feature_names=['f1', 'f2'], control_flags=[True])

def test_feature_metadata_slice():
    meta = FeatureMetadata(
        feature_names=['f1', 'f2', 'f3'],
        monotonicities=['inc', 'dec', 'none'],
        constraint_types=['strict', 'soft', 'none']
    )
    m2 = meta.slice([0, 2])
    assert m2.feature_names == ['f1', 'f3']
    assert m2.monotonicities == ['inc', 'none']
    assert m2.constraint_types == ['strict', 'none']

def test_feature_metadata_merge():
    m1 = FeatureMetadata(['f1'], ['inc'])
    m2 = FeatureMetadata(['f2'], ['dec'])
    m3 = m1.merge(m2)
    assert m3.feature_names == ['f1', 'f2']
    assert m3.monotonicities == ['inc', 'dec']
