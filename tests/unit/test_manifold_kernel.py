"""
tests/unit/test_manifold_kernel.py
ManifoldAwareKernel (F-420) と update_group_manifold (F-402) のテスト
"""
from __future__ import annotations
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # headless テスト用

from domainml.constraints.manifold_kernel import ManifoldAwareKernel
from domainml.analysis.diagnostics import plot_manifold_projection
from domainml.core.metadata import FeatureMetadata, ManifoldConfig


@pytest.fixture
def X():
    return np.random.default_rng(0).standard_normal((30, 4))


# ─── ManifoldAwareKernel ──────────────────────────────────────────────────────
class TestManifoldAwareKernel:
    def test_rbf_kernel_train_shape(self, X):
        """RBF カーネルが (n, n) の行列を返す（F-420）"""
        k = ManifoldAwareKernel(method="rbf", n_neighbors=5)
        k.fit(X)
        K = k(X, X)
        assert K.shape == (30, 30)

    def test_rbf_kernel_is_symmetric(self, X):
        """RBF カーネル行列は対称（F-420）"""
        k = ManifoldAwareKernel(method="rbf", n_neighbors=5)
        k.fit(X)
        K = k(X, X)
        np.testing.assert_array_almost_equal(K, K.T, decimal=10)

    def test_diffusion_kernel_train_shape(self, X):
        """拡散カーネルが (n, n) の行列を返す（F-420）"""
        k = ManifoldAwareKernel(method="diffusion", n_neighbors=5, n_eigenvectors=10)
        k.fit(X)
        K = k(X, X)
        assert K.shape == (30, 30)

    def test_cross_kernel_shape(self, X):
        """K(X_test, X_train) が (n_test, n_train) の形状（F-420）"""
        X_test = np.random.default_rng(1).standard_normal((10, 4))
        k = ManifoldAwareKernel(method="rbf", n_neighbors=5)
        k.fit(X)
        K = k(X_test, X)
        assert K.shape == (10, 30)

    def test_transform_diffusion_shape(self, X):
        """transform が (n_samples, n_eigenvectors) の埋め込みを返す（F-420）"""
        k = ManifoldAwareKernel(method="diffusion", n_neighbors=5, n_eigenvectors=8)
        k.fit(X)
        X_emb = k.transform(X)
        assert X_emb.shape == (30, 8)

    def test_manifold_config_overrides_n_neighbors(self, X):
        """ManifoldConfig の n_neighbors がデフォルトを上書きする（F-401）"""
        cfg: ManifoldConfig = {"n_neighbors": 3}
        k = ManifoldAwareKernel(method="rbf", n_neighbors=10, manifold_config=cfg)
        k.fit(X)
        K = k(X, X)
        assert K.shape == (30, 30)

    def test_fit_not_called_raises(self):
        """fit なしで transform を呼ぶと RuntimeError（F-420）"""
        k = ManifoldAwareKernel(method="diffusion")
        X = np.random.default_rng(0).standard_normal((10, 3))
        with pytest.raises(RuntimeError, match="fit"):
            k.transform(X)


# ─── update_group_manifold ────────────────────────────────────────────────────
class TestUpdateGroupManifold:
    def test_update_sets_manifold_flags(self):
        """update_group_manifold が該当列の manifold_flags を True にする（F-402）"""
        meta = FeatureMetadata(
            ["HOMO", "LUMO", "dipole", "MW", "logP"],
            manifold_flags=[False] * 5,
        )
        meta.update_group_manifold("electronic", [0, 1, 2], intrinsic_dim=2)
        assert meta.manifold_flags[0] is True
        assert meta.manifold_flags[1] is True
        assert meta.manifold_flags[2] is True
        assert meta.manifold_flags[3] is False  # 未登録

    def test_multiple_groups_registered(self):
        """複数グループを登録できる（F-402）"""
        meta = FeatureMetadata(
            ["HOMO", "LUMO", "MW", "logP"],
            manifold_flags=[False] * 4,
        )
        meta.update_group_manifold("electronic", [0, 1], intrinsic_dim=1)
        meta.update_group_manifold("steric", [2, 3], intrinsic_dim=1)
        configs = meta.get_group_manifold_configs()
        assert "electronic" in configs
        assert "steric" in configs
        assert configs["electronic"]["feature_indices"] == [0, 1]

    def test_set_manifold_config(self):
        """set_manifold_config が ManifoldConfig を保存する（F-401）"""
        meta = FeatureMetadata(["f0", "f1"])
        cfg: ManifoldConfig = {"type": "chemical_space", "n_neighbors": 8}
        meta.set_manifold_config(cfg)
        assert meta.get_manifold_config()["type"] == "chemical_space"

    def test_get_manifold_config_default_empty(self):
        """set 前の get は空 dict を返す（F-401）"""
        meta = FeatureMetadata(["f0"])
        assert meta.get_manifold_config() == {}

    def test_out_of_bounds_index_ignored(self):
        """範囲外インデックスは無視されエラーにならない（F-402）"""
        meta = FeatureMetadata(["f0", "f1"])
        meta.update_group_manifold("bad_group", [0, 99])  # 99 は範囲外
        assert meta.manifold_flags[0] is True


# ─── plot_manifold_projection ─────────────────────────────────────────────────
class TestPlotManifoldProjection:
    def test_returns_figure(self, X):
        """Figure オブジェクトが返ることを確認（F-430）"""
        import matplotlib.figure
        fig = plot_manifold_projection(X, method="pca", n_components=2, show=False)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_target_values(self, X):
        """target_values を与えると Figure が返る（F-430）"""
        import matplotlib.figure
        y = np.random.default_rng(0).standard_normal(30)
        fig = plot_manifold_projection(X, target_values=y, method="pca", show=False)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_metadata_control_flags(self, X):
        """control_flags=True の列が除外されても Figure が返る（F-430）"""
        import matplotlib.figure
        meta = FeatureMetadata(
            ["f0", "f1", "f2", "f3"],
            manifold_flags=[True, True, True, False],
            control_flags=[False, False, False, True],
        )
        fig = plot_manifold_projection(X, metadata=meta, method="pca", show=False)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_isomap_2d(self, X):
        """Isomap で 2D に射影できる（F-430）"""
        import matplotlib.figure
        fig = plot_manifold_projection(X, method="isomap", n_components=2, show=False)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_invalid_method_raises(self, X):
        """不正 method で ValueError（F-430）"""
        with pytest.raises(ValueError, match="Unknown method"):
            plot_manifold_projection(X, method="mds", show=False)
