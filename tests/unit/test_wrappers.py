import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor
from domainml.models.wrappers import MonotonicityWrapper
from domainml.core.metadata import FeatureMetadata

def test_monotonicity_wrapper():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([1.0, 0.0, 5.0, 2.0, 7.0])
    
    meta = FeatureMetadata(['f1'], monotonicities=['inc'])
    wrapper = MonotonicityWrapper(DecisionTreeRegressor(random_state=42))
    wrapper.fit(X, y, metadata=meta)
    
    preds = wrapper.predict(X, metadata=meta)
    for i in range(1, len(preds)):
        assert preds[i] >= preds[i-1]
