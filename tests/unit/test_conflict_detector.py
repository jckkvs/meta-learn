"""
tests/unit/test_conflict_detector.py
LinearCoefConflictChecker (T-330) のテスト
"""
import numpy as np
import pytest
from domainml.analysis.coef_checker import LinearCoefConflictChecker
from domainml.core.metadata import FeatureMetadata


@pytest.fixture
def detector():
    return LinearCoefConflictChecker(threshold=0.1, correlation_threshold=0.8)


class TestCoefConflicts:
    def test_no_conflict_when_data_agrees(self, detector):
        """データと制約が一致するとき係数競合は検出されない（T-330）"""
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (60, 2))
        y = X[:, 0] * 2 + X[:, 1] * 3  # both positive
        meta = FeatureMetadata(["f0", "f1"], monotonicities=["inc", "inc"])
        conflicts = detector._detect_coef_conflicts(X, y, meta)
        assert len(conflicts) == 0

    def test_conflict_when_data_disagrees(self, detector):
        """データと逆方向の制約を与えると係数競合が検出される（T-330）"""
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (60, 2))
        y = -X[:, 0] * 5 + X[:, 1]  # f0 strongly negative
        meta = FeatureMetadata(["f0", "f1"], monotonicities=["inc", "none"])
        conflicts = detector._detect_coef_conflicts(X, y, meta)
        assert any(c["type"] == "coef_monotonicity_conflict" for c in conflicts)


class TestStatisticalConflicts:
    def test_high_corr_opposing_mono_detected(self, detector):
        """高相関特徴量に相反する単調性制約がある場合、統計的競合が検出される（T-330）"""
        rng = np.random.default_rng(1)
        base = rng.standard_normal(100)
        X = np.column_stack([base, base + rng.normal(0, 0.01, 100)])  # corr ≈ 1
        y = base
        meta = FeatureMetadata(["f0", "f1"], monotonicities=["inc", "dec"])
        conflicts = detector._detect_statistical_conflicts(X, meta)
        assert any(c["type"] == "statistical_monotonicity_conflict" for c in conflicts)

    def test_low_corr_no_statistical_conflict(self, detector):
        """低相関なら統計的競合は検出されない（T-330）"""
        rng = np.random.default_rng(2)
        X = rng.standard_normal((100, 2))  # uncorrelated
        meta = FeatureMetadata(["f0", "f1"], monotonicities=["inc", "dec"])
        conflicts = detector._detect_statistical_conflicts(X, meta)
        assert len(conflicts) == 0

    def test_same_direction_high_corr_no_conflict(self, detector):
        """高相関でも制約方向が同じなら競合なし（T-330）"""
        rng = np.random.default_rng(3)
        base = rng.standard_normal(100)
        X = np.column_stack([base, base + rng.normal(0, 0.01, 100)])
        meta = FeatureMetadata(["f0", "f1"], monotonicities=["inc", "inc"])
        conflicts = detector._detect_statistical_conflicts(X, meta)
        assert len(conflicts) == 0


class TestMathematicalConflicts:
    def test_feasible_constraints_no_conflict(self, detector):
        """同時充足可能な制約では数学的競合なし（T-330）"""
        meta = FeatureMetadata(["f0", "f1"], monotonicities=["inc", "dec"])
        conflicts = detector._detect_mathematical_conflicts(meta)
        # inc と dec は数学的には共存可能（別の変数）
        assert len(conflicts) == 0

    def test_none_constraint_skipped(self, detector):
        """'none' 制約は数学的チェックに含まれない（T-330）"""
        meta = FeatureMetadata(["f0"], monotonicities=["none"])
        conflicts = detector._detect_mathematical_conflicts(meta)
        assert len(conflicts) == 0

    def test_detect_conflicts_runs_all_checks(self, detector):
        """detect_conflicts が全チェックを実行し結果を統合する（T-330）"""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 2))
        y = rng.standard_normal(60)
        meta = FeatureMetadata(["f0", "f1"], monotonicities=["none", "none"])
        results = detector.detect_conflicts(X, y, meta)
        assert isinstance(results, list)
