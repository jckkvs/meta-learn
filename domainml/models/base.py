from typing import Optional, Any
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from domainml.core.metadata import FeatureMetadata

class DomainEstimator(BaseEstimator):
    """
    メタデータを受け取り、各種のドメイン制約（単調性など）を考慮して最適化を実行する
    基本の推定器インターフェース。
    """
    def __init__(self):
        self.metadata_: Optional[FeatureMetadata] = None

    def fit(self, X: np.ndarray, y: np.ndarray, metadata: Optional[FeatureMetadata] = None, **fit_params):
        """
        X, y と メタデータ を受け取りモデルを学習する。
        """
        self.metadata_ = metadata.clone() if metadata is not None else None
        self._fit(X, y, **fit_params)
        return self

    def _fit(self, X: np.ndarray, y: np.ndarray, **fit_params):
        """サブクラスでオーバーライドする学習ロジック"""
        raise NotImplementedError

    def predict(self, X: np.ndarray, metadata: Optional[FeatureMetadata] = None) -> np.ndarray:
        """予測ロジック"""
        return self._predict(X)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """サブクラスでオーバーライドする推論ロジック"""
        raise NotImplementedError
