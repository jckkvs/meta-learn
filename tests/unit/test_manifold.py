"""
tests/unit/test_manifold.py
manifold_embed() ショートカット関数のテスト
"""
from __future__ import annotations
import numpy as np
import pytest
from domainml.manifold import manifold_embed
from domainml.core.metadata import FeatureMetadata

@pytest.fixture
def X():
    return np.random.default_rng(0).standard_normal((30, 4))

class TestManifoldEmbed:
    def test_lle_embed(self, X):
        """LLEで正しく埋め込まれる"""
        X_emb = manifold_embed(X, method="lle", n_components=2)
        assert X_emb.shape == (30, 2)

    def test_pca_embed(self, X):
        """PCAで正しく埋め込まれる"""
        X_emb = manifold_embed(X, method="pca", n_components=3)
        assert X_emb.shape == (30, 3)

    def test_append_mode(self, X):
        """append=True のとき元特徴量と連結される"""
        X_emb = manifold_embed(X, method="pca", n_components=2, append=True)
        assert X_emb.shape == (30, 4 + 2)

    def test_metadata_control_flags_exclusion(self, X):
        """control_flags=True または manifold_flags=False の列が除外される"""
        # 0,1: 多様体適用, 2: 適用しない, 3: 制御変数として除外
        meta = FeatureMetadata(
            ["f0", "f1", "f2", "f3"],
            manifold_flags=[True, True, False, False],
            control_flags=[False, False, False, True],
        )
        X_emb = manifold_embed(X, metadata=meta, method="pca", n_components=2)
        assert X_emb.shape == (30, 2)

    def test_invalid_method(self, X):
        """不正なmethod名でValueError"""
        with pytest.raises(ValueError):
            manifold_embed(X, method="unknown_method")
