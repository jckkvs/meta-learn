"""
tests/unit/test_manifold_regularizer.py
ManifoldRegularizer (F-410) と ManifoldPreprocessor (F-412) のテスト
"""
from __future__ import annotations
import numpy as np
import pytest
from sklearn.linear_model import Ridge
from domainml.constraints.manifold_regularizer import ManifoldRegularizer, ManifoldPreprocessor
from domainml.core.metadata import FeatureMetadata, ManifoldConfig


# ─── Fixtures ────────────────────────────────────────────────────────────────────
@pytest.fixture
def X():
    return np.random.default_rng(0).standard_normal((40, 4))


@pytest.fixture
def meta_with_manifold():
    return FeatureMetadata(
        feature_names=["f0", "f1", "f2", "f3"],
        manifold_flags=[True, True, False, False],
        control_flags=[False, False, False, True],
    )


# ─── ManifoldRegularizer ────────────────────────────────────────────────────────
class TestManifoldRegularizer:
    def test_fit_builds_laplacian(self, X):
        """fit でラプラシアン行列が構築される（F-410）"""
        reg = ManifoldRegularizer(n_neighbors=5)
        reg.fit(X)
        L = reg.get_laplacian()
        assert L is not None
        assert L.shape == (40, 40)

    def test_transform_is_identity(self, X):
        """transform は入力をそのまま返す（F-410）"""
        reg = ManifoldRegularizer()
        reg.fit(X)
        X_out = reg.transform(X)
        np.testing.assert_array_equal(X_out, X)

    def test_manifold_cols_with_metadata(self, X, meta_with_manifold):
        """control_flags=True の列は多様体グラフ構築から除外される（F-410）"""
        reg = ManifoldRegularizer(metadata=meta_with_manifold)
        reg.fit(X)
        # manifold_flags=[True,True,False,False], control_flags=[F,F,F,True]
        # → 有効な多様体列 = [0,1]
        assert reg._manifold_cols == [0, 1]
        assert reg.get_laplacian() is not None

    def test_transform_in_pipeline(self, X):
        """sklearn Pipeline に組み込んで動作する（統合テスト）"""
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([
            ("manifold", ManifoldRegularizer(n_neighbors=5)),
            ("model", Ridge()),
        ])
        y = np.random.default_rng(0).standard_normal(40)
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (40,)

    def test_get_regularization_term_cvxpy(self, X):
        """get_regularization_term が cvxpy 式を返す（F-410）"""
        try:
            import cvxpy as cp
        except ImportError:
            pytest.skip("cvxpy not available")
        reg = ManifoldRegularizer(n_neighbors=5)
        reg.fit(X)
        f = cp.Variable(40)
        term = reg.get_regularization_term(f, lambda_m=0.1)
        assert term is not None

    def test_get_regularization_term_without_fit_warns(self, X, caplog):
        """fit 前に get_regularization_term を呼ぶと警告が出る（F-410）"""
        import logging
        try:
            import cvxpy as cp
        except ImportError:
            pytest.skip("cvxpy not available")
        reg = ManifoldRegularizer()
        f = cp.Variable(10)
        with caplog.at_level(logging.WARNING, logger="domainml"):
            term = reg.get_regularization_term(f)
        assert any("not built" in m.lower() for m in caplog.messages)

    def test_no_manifold_cols_warns(self, X, caplog):
        """manifold_flags がすべて False のとき警告を出す（F-410）"""
        import logging
        meta_no_manifold = FeatureMetadata(
            feature_names=["f0", "f1", "f2", "f3"],
            manifold_flags=[False, False, False, False],
        )
        reg = ManifoldRegularizer(metadata=meta_no_manifold)
        with caplog.at_level(logging.WARNING, logger="domainml"):
            reg.fit(X)
        assert any("no manifold columns" in m.lower() for m in caplog.messages)

    def test_manifold_config_overrides_defaults(self, X):
        """ManifoldConfig のキーがパラメータを上書きする（F-401）"""
        cfg: ManifoldConfig = {"n_neighbors": 3, "local_radius": 0.5, "regularization_weight": 0.2}
        reg = ManifoldRegularizer(manifold_config=cfg)
        reg.fit(X)
        assert reg.get_laplacian() is not None


# ─── ManifoldPreprocessor ────────────────────────────────────────────────────────
class TestManifoldPreprocessor:
    def test_lle_transforms_shape(self, X):
        """LLE で低次元埋め込みが生成される（F-412）"""
        prep = ManifoldPreprocessor(n_components=2, method="lle")
        prep.fit(X)
        X_out = prep.transform(X)
        assert X_out.shape == (40, 2)

    def test_isomap_transforms_shape(self, X):
        """Isomap で低次元埋め込みが生成される（F-412）"""
        prep = ManifoldPreprocessor(n_components=3, method="isomap")
        prep.fit(X)
        X_out = prep.transform(X)
        assert X_out.shape == (40, 3)

    def test_append_mode_concatenates(self, X):
        """append=True のとき元特徴量 + 埋め込みが連結される（F-412）"""
        prep = ManifoldPreprocessor(n_components=2, method="lle", append=True)
        prep.fit(X)
        X_out = prep.transform(X)
        assert X_out.shape == (40, 4 + 2)
        assert prep.out_features_ == 6

    def test_fit_transform_consistency(self, X):
        """fit_transform と fit → transform が同じ結果を返す（F-412）"""
        prep1 = ManifoldPreprocessor(n_components=2, method="lle", random_state=42)
        prep2 = ManifoldPreprocessor(n_components=2, method="lle", random_state=42)
        X1 = prep1.fit_transform(X)
        X2 = prep2.fit(X).transform(X)
        np.testing.assert_array_almost_equal(X1, X2, decimal=10)

    def test_invalid_method_raises(self, X):
        """不正な method 名で ValueError が発生する（F-412）"""
        with pytest.raises(ValueError, match="Unknown method"):
            ManifoldPreprocessor(n_components=2, method="pca").fit(X)

    def test_transform_without_fit_raises(self, X):
        """fit 前に transform を呼ぶと RuntimeError が発生する（F-412）"""
        prep = ManifoldPreprocessor(n_components=2, method="lle")
        with pytest.raises(RuntimeError, match="未学習"):
            prep.transform(X)

    def test_out_features_embed_only(self, X):
        """append=False のとき out_features_ = n_components（F-412）"""
        prep = ManifoldPreprocessor(n_components=3, method="lle")
        prep.fit(X)
        assert prep.out_features_ == 3

    def test_umap_requires_umap_learn(self, X):
        """umap-learn がない場合、UMAP 選択で ImportError が発生する（F-412）"""
        try:
            import umap  # noqa: F401
            pytest.skip("umap-learn is installed; skipping ImportError test")
        except ImportError:
            prep = ManifoldPreprocessor(n_components=2, method="umap")
            with pytest.raises(ImportError, match="umap-learn"):
                prep.fit(X)
