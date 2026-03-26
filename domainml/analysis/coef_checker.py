"""
domainml/analysis/coef_checker.py
制約競合検出器（F-330）

LinearCoefConflictChecker:
  - 統計的競合: 高相関特徴量間の相反する単調性制約を検出
  - 数学的競合: 線形計画法で制約の同時充足可能性を確認（F-330）
"""
from __future__ import annotations
import numpy as np
from itertools import combinations
from scipy.optimize import linprog
from sklearn.linear_model import LinearRegression
from domainml.core.metadata import FeatureMetadata
from domainml.core.logger import logger


class LinearCoefConflictChecker:
    """
    ドメイン知識の制約間の競合を検出する（F-330）

    検出タイプ:
      1. **係数競合**: 線形回帰の係数がメタデータの単調性制約と逆行
      2. **統計的競合**: 高相関特徴量ペアに相反する単調性制約が指定されている
      3. **数学的競合**: 線形計画法で制約の同時充足可能性を確認

    Parameters
    ----------
    significance_level : float
        統計的競合検出の有意水準（相関の閾値に使用）。
    threshold : float
        係数競合と判断する線形係数の絶対値閾値。
    correlation_threshold : float
        「高相関」と判断する絶対相関係数の閾値（デフォルト 0.8）。
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        threshold: float = 0.1,
        correlation_threshold: float = 0.8,
    ):
        self.significance_level = significance_level
        self.threshold = threshold
        self.correlation_threshold = correlation_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_conflicts(
        self, X: np.ndarray, y: np.ndarray, metadata: FeatureMetadata
    ) -> list:
        """
        すべての競合タイプを検出し、結果リストを返す。

        Returns
        -------
        conflicts : list of dict
            各辞書は `type`, `features`, `severity`, `recommendation` を含む。
        """
        conflicts = []
        conflicts.extend(self._detect_coef_conflicts(X, y, metadata))
        conflicts.extend(self._detect_statistical_conflicts(X, metadata))
        conflicts.extend(self._detect_mathematical_conflicts(metadata))
        logger.debug(
            f"LinearCoefConflictChecker: {len(conflicts)} conflict(s) detected."
        )
        return conflicts

    # ------------------------------------------------------------------
    # 1. 係数競合（既存ロジックの維持・強化）
    # ------------------------------------------------------------------

    def _detect_coef_conflicts(
        self, X: np.ndarray, y: np.ndarray, metadata: FeatureMetadata
    ) -> list:
        """線形回帰係数がメタデータの単調性制約と逆行する場合を検出。"""
        conflicts = []
        lr = LinearRegression()
        lr.fit(X, y)
        coefs = lr.coef_

        for i in range(metadata.n_features):
            mono = metadata.monotonicities[i]
            if mono == "none":
                continue
            obs = coefs[i]
            fname = metadata.feature_names[i]

            if mono == "inc" and obs < -self.threshold:
                msg = f"Coef conflict on '{fname}': inc constraint, but coef={obs:.3f}"
                logger.warning(msg)
                conflicts.append(
                    {
                        "type": "coef_monotonicity_conflict",
                        "features": [i],
                        "feature_names": [fname],
                        "observed_effect": float(obs),
                        "severity": abs(obs),
                        "recommendation": (
                            f"'{fname}' の単調増加制約は観測データと逆行しています。"
                            "制約を削除するか、soft 制約への変更を検討してください。"
                        ),
                    }
                )
            elif mono == "dec" and obs > self.threshold:
                msg = f"Coef conflict on '{fname}': dec constraint, but coef={obs:.3f}"
                logger.warning(msg)
                conflicts.append(
                    {
                        "type": "coef_monotonicity_conflict",
                        "features": [i],
                        "feature_names": [fname],
                        "observed_effect": float(obs),
                        "severity": abs(obs),
                        "recommendation": (
                            f"'{fname}' の単調減少制約は観測データと逆行しています。"
                            "制約を削除するか、soft 制約への変更を検討してください。"
                        ),
                    }
                )
        return conflicts

    # ------------------------------------------------------------------
    # 2. 統計的競合（高相関 × 相反制約）
    # ------------------------------------------------------------------

    def _detect_statistical_conflicts(
        self, X: np.ndarray, metadata: FeatureMetadata
    ) -> list:
        """
        高相関特徴量ペアに相反する単調性制約が設定されている場合を検出。（F-330）

        |corr(i, j)| > correlation_threshold かつ
        mono[i] × mono[j] == -1 （一方が inc、他方が dec）のケースを報告する。
        """
        conflicts = []
        n = metadata.n_features
        corr = np.corrcoef(X.T) if n > 1 else np.array([[1.0]])

        mono_sign = {
            "inc": 1,
            "dec": -1,
            "none": 0,
        }

        for i, j in combinations(range(n), 2):
            si = mono_sign[metadata.monotonicities[i]]
            sj = mono_sign[metadata.monotonicities[j]]

            if si == 0 or sj == 0:
                continue  # どちらかが制約なしなら競合なし

            if abs(corr[i, j]) >= self.correlation_threshold and si * sj < 0:
                fi = metadata.feature_names[i]
                fj = metadata.feature_names[j]
                msg = (
                    f"Statistical conflict: '{fi}'({metadata.monotonicities[i]}) "
                    f"vs '{fj}'({metadata.monotonicities[j]}), "
                    f"corr={corr[i, j]:.3f}"
                )
                logger.warning(msg)
                conflicts.append(
                    {
                        "type": "statistical_monotonicity_conflict",
                        "features": [i, j],
                        "feature_names": [fi, fj],
                        "correlation": float(corr[i, j]),
                        "severity": abs(corr[i, j]),
                        "recommendation": (
                            f"'{fi}' と '{fj}' は高相関（{corr[i, j]:.2f}）ですが、"
                            "異なる単調性制約が設定されています。"
                            "どちらか一方の制約を削除することを検討してください。"
                        ),
                    }
                )
        return conflicts

    # ------------------------------------------------------------------
    # 3. 数学的充足可能性チェック（線形計画法）
    # ------------------------------------------------------------------

    def _detect_mathematical_conflicts(self, metadata: FeatureMetadata) -> list:
        """
        inc/dec 制約を線形不等式として定式化し、同時充足可能性を確認。（F-330）

        係数 w を変数とし：
          - inc:  w_i >= eps
          - dec: -w_i >= eps
        を制約として linprog で実行可能性を確認する。

        全制約が同時に成立しない（infeasible）場合、競合を報告する。
        """
        constrained = [
            (i, metadata.monotonicities[i])
            for i in range(metadata.n_features)
            if metadata.monotonicities[i] != "none"
        ]
        if len(constrained) < 2:
            return []  # 制約 1 件以下では数学的競合は起きない

        n = metadata.n_features
        eps = 1e-6

        # 各制約を A_ub @ w <= b_ub に変換
        A_ub_rows = []
        b_ub = []
        for i, mono in constrained:
            row = np.zeros(n)
            if mono == "inc":
                row[i] = -1.0    # -w_i <= -eps  ↔  w_i >= eps
                b_ub.append(-eps)
            else:
                row[i] = 1.0    #  w_i <= -eps  ↔  w_i <= -eps
                b_ub.append(-eps)
            A_ub_rows.append(row)

        A_ub = np.array(A_ub_rows)
        b_ub_arr = np.array(b_ub)
        c = np.zeros(n)  # 目的関数なし（実行可能性のみ確認）

        result = linprog(c, A_ub=A_ub, b_ub=b_ub_arr, bounds=[(None, None)] * n)

        if result.status == 2:  # infeasible
            constrained_names = [metadata.feature_names[i] for i, _ in constrained]
            logger.warning(
                f"Mathematical conflict: constraints on {constrained_names} "
                "are simultaneously infeasible."
            )
            return [
                {
                    "type": "mathematical_infeasibility",
                    "features": [i for i, _ in constrained],
                    "feature_names": constrained_names,
                    "severity": 1.0,
                    "recommendation": (
                        "指定された単調性制約は数学的に同時充足不可能です。"
                        "制約の見直しを強く推奨します。"
                    ),
                }
            ]
        return []
