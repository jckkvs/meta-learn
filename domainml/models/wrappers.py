"""
domainml/models/wrappers.py
MonotonicityWrapper（F-340）

任意の scikit-learn 互換モデルに対して、Isotonic Regression による事後単調性補正を
特徴量ごとに個別に学習・適用する。
"""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.isotonic import IsotonicRegression
from domainml.core.metadata import FeatureMetadata
from domainml.core.logger import logger


class MonotonicityWrapper(BaseEstimator):
    """
    任意のモデルの予測値に Isotonic Regression 補正を適用するラッパー（F-340）

    設計:
      - 単調性制約を持つ特徴量ごとに個別の IsotonicRegression を学習
      - predict 時に各補正を平均アンサンブルしてスカラー予測を返す
      - 複数特徴量がある場合も正しく動作し、`np.diff` で単調性が検証可能

    Parameters
    ----------
    base_estimator : sklearn Estimator
        ラップする基底モデル。
    """

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metadata: FeatureMetadata = None,
        **fit_params,
    ) -> "MonotonicityWrapper":
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X, y, **fit_params)

        self._calibrators: dict = {}      # feat_idx -> IsotonicRegression
        self._mono_directions: dict = {}  # feat_idx -> bool (increasing?)

        if metadata is not None:
            preds = self.estimator_.predict(X)
            for i, mono in enumerate(metadata.monotonicities):
                if mono not in ("inc", "dec"):
                    continue
                increasing = mono == "inc"
                sort_idx = np.argsort(X[:, i])
                feat_vals = X[:, i][sort_idx]
                preds_sorted = preds[sort_idx]

                iso = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
                iso.fit(feat_vals, preds_sorted)
                self._calibrators[i] = iso
                self._mono_directions[i] = increasing
                logger.debug(
                    f"MonotonicityWrapper: fitted IsotonicRegression "
                    f"for feature {i} (increasing={increasing})"
                )

        return self

    def predict(self, X: np.ndarray, metadata: FeatureMetadata = None) -> np.ndarray:
        """
        基底モデルの予測に単調性補正を適用する。

        複数の補正がある場合は平均アンサンブル、補正がない場合は基底予測をそのまま返す。
        """
        base_preds = self.estimator_.predict(X)

        if not self._calibrators:
            return base_preds

        corrections = []
        for feat_idx, iso in self._calibrators.items():
            corrected = iso.predict(X[:, feat_idx])
            corrections.append(corrected)

        return np.mean(corrections, axis=0)
