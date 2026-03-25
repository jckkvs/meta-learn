"""
tests/unit/test_wrapper_advanced.py
MonotonicityWrapper (F-340) の高度テスト

np.diff による予測単調性の直接検証を行う（T-340）
"""
import numpy as np
import pytest
from sklearn.linear_model import Ridge
from domainml.models.wrappers import MonotonicityWrapper
from domainml.core.metadata import FeatureMetadata


@pytest.fixture
def inc_data():
    """単調増加データ（ただし基底モデルで単調性が保証されない場合に使う）"""
    rng = np.random.default_rng(0)
    X = np.sort(rng.uniform(0, 1, (60, 1)), axis=0)  # sorted for clear test
    y = X[:, 0] ** 2 + rng.normal(0, 0.05, 60)  # noisy monotone
    return X, y


@pytest.fixture
def dec_data():
    rng = np.random.default_rng(1)
    X = np.sort(rng.uniform(0, 1, (60, 1)), axis=0)
    y = -X[:, 0] ** 2 + rng.normal(0, 0.05, 60)
    return X, y


class TestMonotonicityWrapperAdvanced:
    def test_predict_increasing_monotone(self, inc_data):
        """
        単調増加制約を適用したラッパーの予測は np.diff で単調増加を示す（T-340）
        """
        X, y = inc_data
        meta = FeatureMetadata(["f0"], monotonicities=["inc"], constraint_types=["soft"])
        wrapper = MonotonicityWrapper(Ridge())
        wrapper.fit(X, y, metadata=meta)

        # ソート済み入力で予測
        X_test = np.linspace(0, 1, 30).reshape(-1, 1)
        preds = wrapper.predict(X_test)
        diffs = np.diff(preds)
        # IsotonicRegression の out_of_bounds='clip' により非減少
        assert np.all(diffs >= -1e-8), f"Expected non-decreasing preds, got diffs min={diffs.min()}"

    def test_predict_decreasing_monotone(self, dec_data):
        """
        単調減少制約を適用したラッパーの予測は np.diff で単調減少を示す（T-340）
        """
        X, y = dec_data
        meta = FeatureMetadata(["f0"], monotonicities=["dec"], constraint_types=["soft"])
        wrapper = MonotonicityWrapper(Ridge())
        wrapper.fit(X, y, metadata=meta)

        X_test = np.linspace(0, 1, 30).reshape(-1, 1)
        preds = wrapper.predict(X_test)
        diffs = np.diff(preds)
        assert np.all(diffs <= 1e-8), f"Expected non-increasing preds, got diffs max={diffs.max()}"

    def test_no_metadata_returns_base_preds(self, inc_data):
        """metadata=None のとき基底モデルの予測がそのまま返る（T-340）"""
        X, y = inc_data
        wrapper = MonotonicityWrapper(Ridge())
        wrapper.fit(X, y)
        preds = wrapper.predict(X)
        base_preds = wrapper.estimator_.predict(X)
        np.testing.assert_array_almost_equal(preds, base_preds)

    def test_multiple_constrained_features_ensemble(self):
        """複数制約特徴量がある場合、予測は各補正の平均アンサンブル（T-340）"""
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (50, 2))
        y = X[:, 0] + X[:, 1] + rng.normal(0, 0.05, 50)
        meta = FeatureMetadata(
            ["f0", "f1"],
            monotonicities=["inc", "inc"],
            constraint_types=["soft", "soft"],
        )
        wrapper = MonotonicityWrapper(Ridge())
        wrapper.fit(X, y, metadata=meta)
        preds = wrapper.predict(X)
        assert preds.shape == (50,)
        assert len(wrapper._calibrators) == 2
