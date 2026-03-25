"""
tests/unit/test_group.py
GroupConstraintEngine (T-310) と GroupStandardScaler (T-311) のテスト
"""
import numpy as np
import pytest
from domainml.constraints.group import GroupConstraintEngine, GroupStandardScaler


# ─── Fixtures ───────────────────────────────────────────────────────────────────
@pytest.fixture
def groups():
    return {0: [0, 1], 1: [2, 3]}  # 4 features, 2 groups


@pytest.fixture
def X():
    rng = np.random.default_rng(42)
    return rng.standard_normal((50, 4))


# ─── GroupConstraintEngine ───────────────────────────────────────────────────────
class TestGroupConstraintEngine:
    def test_zero_lambda_no_shrinkage(self, groups):
        """λ=0 のとき係数は変化しない（T-310）"""
        engine = GroupConstraintEngine(groups)
        coef = np.array([1.0, 2.0, -1.0, 0.5])
        result = engine.apply_group_lasso_penalty(coef, lambda_group=0.0)
        np.testing.assert_array_almost_equal(result, coef)

    def test_large_lambda_shrinks_to_zero(self, groups):
        """λ が非常に大きいとき全係数がゼロに縮小する（T-310）"""
        engine = GroupConstraintEngine(groups)
        coef = np.array([0.01, 0.01, 0.01, 0.01])
        result = engine.apply_group_lasso_penalty(coef, lambda_group=1e6)
        np.testing.assert_array_almost_equal(result, np.zeros(4), decimal=5)

    def test_group_lasso_reduces_norm(self, groups):
        """ペナルティ適用後はグループノルムが元より小さくなる（T-310）"""
        engine = GroupConstraintEngine(groups)
        coef = np.array([3.0, 4.0, 1.0, 2.0])
        result = engine.apply_group_lasso_penalty(coef, lambda_group=1.0)
        # グループ 0: norm = 5.0 → shrink = max(0, 1 - 1/5) = 0.8
        original_norm = np.linalg.norm(coef[[0, 1]])
        result_norm = np.linalg.norm(result[[0, 1]])
        assert result_norm < original_norm

    def test_get_group_norms(self, groups):
        """get_group_norms が辞書を返す（T-310）"""
        engine = GroupConstraintEngine(groups)
        coef = np.array([3.0, 4.0, 0.0, 0.0])
        norms = engine.get_group_norms(coef)
        assert set(norms.keys()) == {0, 1}
        assert abs(norms[0] - 5.0) < 1e-6
        assert abs(norms[1] - 0.0) < 1e-6

    def test_already_zero_coef_no_error(self, groups):
        """ゼロ係数グループでもエラーにならない（T-310）"""
        engine = GroupConstraintEngine(groups)
        coef = np.zeros(4)
        result = engine.apply_group_lasso_penalty(coef, lambda_group=1.0)
        np.testing.assert_array_almost_equal(result, np.zeros(4))


# ─── GroupStandardScaler ─────────────────────────────────────────────────────────
class TestGroupStandardScaler:
    def test_fit_transform_shape(self, groups, X):
        """変換後の形状は元データと同じ（T-311）"""
        scaler = GroupStandardScaler(groups)
        X_out = scaler.fit_transform(X)
        assert X_out.shape == X.shape

    def test_group_mean_near_zero(self, groups, X):
        """変換後のグループ全体平均がほぼゼロ（T-311）"""
        scaler = GroupStandardScaler(groups)
        X_out = scaler.fit_transform(X)
        for gid, idx in groups.items():
            assert abs(X_out[:, idx].mean()) < 0.2

    def test_inverse_transform_roundtrip(self, groups, X):
        """逆変換後に元データが再現される（T-311）"""
        scaler = GroupStandardScaler(groups)
        X_out = scaler.fit_transform(X)
        X_back = scaler.inverse_transform(X_out)
        np.testing.assert_array_almost_equal(X_back, X, decimal=10)

    def test_columns_outside_groups_unchanged(self, X):
        """グループ外の列は変換されない（T-311）"""
        groups_partial = {0: [0, 1]}  # cols 2,3 are outside
        scaler = GroupStandardScaler(groups_partial)
        X_out = scaler.fit_transform(X)
        np.testing.assert_array_equal(X_out[:, 2:], X[:, 2:])
