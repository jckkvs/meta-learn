import pytest
import numpy as np
from domainml.models.base import DomainEstimator

def test_domain_estimator_base():
    est = DomainEstimator()
    with pytest.raises(NotImplementedError):
        est.fit(np.array([[1.0]]), np.array([1.0]))
    with pytest.raises(NotImplementedError):
        est._fit(np.array([[1.0]]), np.array([1.0]))
    with pytest.raises(NotImplementedError):
        est._predict(np.array([[1.0]]))
