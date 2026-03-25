import numpy as np
import pytest
from domainml.analysis.conflict_detector import ConstraintConflictDetector
from domainml.core.metadata import FeatureMetadata

def test_conflict_detection():
    X = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    meta = FeatureMetadata(['f1', 'f2'], monotonicities=['inc', 'dec'])
    
    detector = ConstraintConflictDetector(correlation_threshold=0.9)
    with pytest.warns(UserWarning):
        warnings = detector.detect(X, metadata=meta)
    
    assert len(warnings) == 1
    assert "Conflict" in warnings[0]
