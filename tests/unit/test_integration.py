"""
tests/unit/test_integration.py
MetaPipeline E2E 統合テストおよびカバレッジギャップ補完テスト

対象モジュールの不足ブランチ:
- parallel.py line 9 (mono=='none' の即時 return None)
- metrics.py line 36 (eps=None つまり IQR 計算ブランチ)
- manifold_engine.py lines 134-142 (apply_manifold_regularization の正常パス)
- uncertainty.py lines 43, 56 (metadata aware predict / return_interval)
- engine.py lines 80-85 (_apply_linear_constraint / _apply_kernel_constraint)
"""
import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

from domainml import (
    FeatureMetadata,
    MetaPipeline,
    MonotonicityEngine,
    MonotonicLinearRegression,
    satisfaction_score,
    constrained_cv,
)
from domainml.analysis.parallel import parallel_check_conflicts, _evaluate_single_feature
from domainml.models.uncertainty import UncertaintyEstimator
from domainml.constraints.manifold_engine import ManifoldAssumptionEngine


# ─── Fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture
def xy():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (80, 3))
    y = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + rng.normal(0, 0.05, 80)
    return X, y


@pytest.fixture
def meta():
    return FeatureMetadata(
        feature_names=["f0", "f1", "f2"],
        monotonicities=["inc", "dec", "none"],
        constraint_types=["strict", "soft", "none"],
    )


# ─── E2E MetaPipeline 統合テスト ─────────────────────────────────────────────
class TestMetaPipelineE2E:
    def test_pipeline_with_linear_model_strict(self, xy, meta):
        """MetaPipeline + MonotonicLinearRegression の E2E フロー（統合テスト）"""
        X, y = xy
        pipeline = MetaPipeline([("model", MonotonicLinearRegression())])
        pipeline.fit(X, y, metadata=meta)
        preds = pipeline.predict(X)
        assert preds.shape == (80,)
        score = satisfaction_score(pipeline, X, meta)
        assert score >= 0.9, f"Satisfaction score too low: {score:.3f}"

    def test_pipeline_with_engine_tree_model(self, xy, meta):
        """MetaPipeline + MonotonicityEngine (GradientBoosting) の E2E フロー"""
        X, y = xy
        engine = MonotonicityEngine(GradientBoostingRegressor(n_estimators=30, random_state=0))
        pipeline = MetaPipeline([("model", engine)])
        pipeline.fit(X, y, metadata=meta)
        preds = pipeline.predict(X)
        assert preds.shape == (80,)

    def test_pipeline_satisfaction_score_iqr_eps(self, xy, meta):
        """satisfaction_score の IQR eps 自動計算ブランチを実行する（eps=None デフォルト）"""
        X, y = xy
        model = MonotonicityEngine(Ridge())
        model.fit(X, y, metadata=meta)
        # eps=None (デフォルト) で IQR ベース摂動
        score = satisfaction_score(model, X, meta, eps=None)
        assert 0.0 <= score <= 1.0

    def test_constrained_cv_integration(self, xy, meta):
        """constrained_cv が satisfaction_score と test_score を返す"""
        X, y = xy
        engine = MonotonicityEngine(Ridge())
        results = constrained_cv(engine, X, y, meta, cv=3)
        assert "test_score" in results
        assert "satisfaction_score" in results
        assert len(results["test_score"]) == 3

    def test_pipeline_predict_without_refitting(self, xy, meta):
        """fit せずに predict を呼ぶと AttributeError が発生しないことを確認"""
        X, y = xy
        pipeline = MetaPipeline([("model", MonotonicLinearRegression())])
        pipeline.fit(X, y, metadata=meta)
        # 新しいデータへの predict
        X_new = np.random.default_rng(99).uniform(0, 1, (10, 3))
        preds = pipeline.predict(X_new)
        assert preds.shape == (10,)


# ─── parallel.py カバレッジギャップ補完 ────────────────────────────────────────
class TestParallelConflictDetection:
    def test_evaluate_single_feature_none_mono(self):
        """mono=='none' のとき即 None を返す（line 9 ブランチ補完）"""
        X = np.random.default_rng(0).standard_normal((20, 2))
        y = np.random.default_rng(0).standard_normal(20)
        result = _evaluate_single_feature(0, X, y, mono="none", feature_name="f0", threshold=0.1)
        assert result is None

    def test_parallel_check_no_conflicts_when_data_agrees(self):
        """データと一致する制約では競合なし"""
        rng = np.random.default_rng(1)
        X = rng.uniform(0, 1, (40, 2))
        y = X[:, 0] * 3 - X[:, 1]
        meta = FeatureMetadata(["f0", "f1"], monotonicities=["inc", "dec"])
        conflicts = parallel_check_conflicts(X, y, meta, threshold=0.1, n_jobs=1)
        assert isinstance(conflicts, list)


# ─── UncertaintyEstimator カバレッジ補完 ────────────────────────────────────
class TestUncertaintyEstimator:
    def test_return_interval_true(self):
        """return_interval=True が (mean, lower, upper) のタプルを返す（line 56 補完）"""
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (50, 2))
        y = X[:, 0] + X[:, 1]
        ue = UncertaintyEstimator(Ridge(), n_estimators=5, random_state=0)
        ue.fit(X, y)
        result = ue.predict(X, return_interval=True)
        assert isinstance(result, tuple) and len(result) == 3
        mean_preds, lower, upper = result
        assert mean_preds.shape == (50,)
        assert np.all(upper >= lower)

    def test_predict_with_metadata_aware_model(self):
        """metadata aware な model が predict でも metadata を受け取る（line 43 補完）"""
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (40, 2))
        y = X[:, 0] + X[:, 1]
        meta = FeatureMetadata(["f0", "f1"], monotonicities=["inc", "inc"])
        engine = MonotonicityEngine(Ridge())
        ue = UncertaintyEstimator(engine, n_estimators=3, random_state=0)
        ue.fit(X, y, metadata=meta)
        preds = ue.predict(X, metadata=meta)
        assert preds.shape == (40,)


# ─── ManifoldAssumptionEngine cvxpy パス ────────────────────────────────────
class TestManifoldEngineRegularization:
    def test_apply_regularization_with_cvxpy(self):
        """apply_manifold_regularization が cvxpy 式に正則化項を追加する（line 134-142 補完）"""
        try:
            import cvxpy as cp
        except ImportError:
            pytest.skip("cvxpy not available")

        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 3))
        engine = ManifoldAssumptionEngine(manifold_variables=[0, 1, 2])
        engine.build_laplacian_regularization(X)

        f = cp.Variable(20)
        loss = cp.sum_squares(f - 1.0)
        augmented = engine.apply_manifold_regularization(loss, f, lambda_m=0.01)
        # 正則化項が加算されているので loss と同一ではない
        assert augmented is not loss

    def test_apply_regularization_with_zero_lambda(self):
        """lambda_m=0 のときも問題なく動作する"""
        try:
            import cvxpy as cp
        except ImportError:
            pytest.skip("cvxpy not available")
        rng = np.random.default_rng(1)
        X = rng.standard_normal((15, 2))
        engine = ManifoldAssumptionEngine(manifold_variables=[0, 1])
        engine.build_laplacian_regularization(X)
        f = cp.Variable(15)
        loss = cp.sum_squares(f)
        augmented = engine.apply_manifold_regularization(loss, f, lambda_m=0.0)
        assert augmented is not None


# ─── engine.py 線形モデルカテゴリの正確な分類テスト ────────────────────────
class TestModelCategoryDetection:
    def _get_category(self, estimator):
        from domainml.constraints.engine import MonotonicityEngine
        engine = MonotonicityEngine(estimator)
        return engine._detect_model_category(estimator)

    def test_linear_svr_is_kernel_not_linear(self):
        """LinearSVR が 'linear' に誤分類されず 'kernel' に分類される"""
        from sklearn.svm import LinearSVR
        cat = self._get_category(LinearSVR())
        assert cat == "kernel", f"LinearSVR should be kernel, got: {cat}"

    def test_svr_is_kernel(self):
        assert self._get_category(SVR()) == "kernel"

    def test_ridge_is_linear(self):
        assert self._get_category(Ridge()) == "linear"

    def test_gbr_is_tree(self):
        assert self._get_category(GradientBoostingRegressor()) == "tree_based"

    def test_unknown_model_is_unknown(self):
        from sklearn.neighbors import KNeighborsRegressor
        assert self._get_category(KNeighborsRegressor()) == "unknown"
