"""
tests/unit/test_manifold_engine.py
ManifoldAssumptionEngine (T-320) のテスト
"""
import numpy as np
import pytest
from scipy.sparse import issparse
from domainml.constraints.manifold_engine import ManifoldAssumptionEngine


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    return rng.standard_normal((30, 4))  # 30 samples, 4 features


class TestManifoldAssumptionEngine:
    def test_laplacian_shape(self, X):
        """ラプラシアン行列は (n_samples, n_samples) であること（T-320）"""
        engine = ManifoldAssumptionEngine(manifold_variables=[0, 1, 2])
        L = engine.build_laplacian_regularization(X)
        n = X.shape[0]
        assert L.shape == (n, n)

    def test_laplacian_is_sparse(self, X):
        """ラプラシアン行列は疎行列であること（T-320）"""
        engine = ManifoldAssumptionEngine(manifold_variables=[0, 1])
        L = engine.build_laplacian_regularization(X)
        assert issparse(L)

    def test_control_variable_excluded(self, X):
        """制御変数リストはグラフ構築から除外される（T-320）"""
        # manifold_vars に 2 列しか渡さないので、L のランクは cols 2,3 に左右されない
        engine1 = ManifoldAssumptionEngine(
            manifold_variables=[0, 1],
            non_manifold_variables=[2, 3],
        )
        engine2 = ManifoldAssumptionEngine(
            manifold_variables=[0, 1, 2, 3],
        )
        L1 = engine1.build_laplacian_regularization(X)
        L2 = engine2.build_laplacian_regularization(X)
        # 構築に使用した変数の次元が異なるので L1 ≠ L2
        assert not np.allclose(L1.toarray(), L2.toarray())

    def test_laplacian_diagonal_non_negative(self, X):
        """ラプラシアン対角成分は非負（次数行列）（T-320）"""
        engine = ManifoldAssumptionEngine(manifold_variables=[0, 1, 2])
        L = engine.build_laplacian_regularization(X)
        diag = L.diagonal()
        assert np.all(diag >= -1e-10)

    def test_apply_regularization_without_build_raises_warning(self, X, caplog):
        """build 前に apply_manifold_regularization を呼ぶと警告が出る（T-320）"""
        import logging
        engine = ManifoldAssumptionEngine(manifold_variables=[0])
        try:
            import cvxpy as cp
            f = cp.Variable(X.shape[0])
            loss = cp.sum_squares(f)
            with caplog.at_level(logging.WARNING, logger="domainml"):
                engine.apply_manifold_regularization(loss, f, lambda_m=0.01)
            assert any("Laplacian not built" in m for m in caplog.messages)
        except ImportError:
            pytest.skip("cvxpy not available")

    def test_get_laplacian_before_build_is_none(self):
        """build 前に get_laplacian は None を返す（T-320）"""
        engine = ManifoldAssumptionEngine(manifold_variables=[0])
        assert engine.get_laplacian() is None
