"""
domainml/constraints/group.py
特徴量群制約エンジン（F-310, F-311）

GroupConstraintEngine : Group Lasso ペナルティを係数に適用
GroupStandardScaler   : 物理的関係を保つグループ同時スケーリング
"""
import numpy as np
from __future__ import annotations
from typing import Literal
from domainml.core.logger import logger


class GroupConstraintEngine:
    """
    特徴量群制約エンジン（F-310）

    特徴量グループを定義し、Group Lasso ペナルティを係数ベクトルに適用する。
    グループ内特徴量を同時に選択／除外することで、物理的に意味のある特徴量単位を保持する。

    Parameters
    ----------
    groups : Dict[int, List[int]]
        {group_id: [feature_indices]} の辞書。
    group_scaling : str
        "joint"  : グループ全体を一つの正規化単位として扱う（デフォルト）
        "independent" : 特徴量ごとに独立してスケーリング（通常の L1/L2 同等）
    """

    def __init__(
        self,
        groups: Dict[int, List[int]],
        group_scaling: Literal["joint", "independent"] = "joint",
    ):
        self.groups = groups
        self.group_scaling = group_scaling

    def apply_group_lasso_penalty(
        self, coef: np.ndarray, lambda_group: float
    ) -> np.ndarray:
        """
        Group Lasso 縮小を係数ベクトルに適用する。（F-310）

        各グループ g に対して：
            β_g ← β_g * max(0, 1 - λ / ‖β_g‖₂)

        これにより同一グループ内の係数が同時にゼロへ縮小される。

        Parameters
        ----------
        coef : np.ndarray, shape (n_features,)
        lambda_group : float
            グループ Lasso の正則化強度。

        Returns
        -------
        penalized_coef : np.ndarray
            縮小後の係数ベクトル。
        """
        penalized_coef = coef.copy().astype(float)

        for group_id, feat_indices in self.groups.items():
            group_coef = penalized_coef[feat_indices]
            group_norm = np.linalg.norm(group_coef, 2)

            if group_norm > 0:
                shrinkage = max(0.0, 1.0 - lambda_group / group_norm)
                penalized_coef[feat_indices] = group_coef * shrinkage
                logger.debug(
                    f"GroupLasso group={group_id}: norm={group_norm:.4f}, "
                    f"shrinkage={shrinkage:.4f}"
                )
            else:
                logger.debug(f"GroupLasso group={group_id}: coef already zero.")

        return penalized_coef

    def get_group_norms(self, coef: np.ndarray) -> Dict[int, float]:
        """各グループの L2 ノルムを返す（診断用）。"""
        return {
            gid: float(np.linalg.norm(coef[idx], 2))
            for gid, idx in self.groups.items()
        }


class GroupStandardScaler:
    """
    特徴量群を物理的関係を保ちながら同時スケーリングする変換器（F-311）

    通常の StandardScaler では各特徴量を独立にスケーリングするため、同一グループ内の
    相対的スケール（例: 温度 [K] と 圧力 [Pa] の比）が失われる。
    本クラスはグループ全体の global 平均・標準偏差でスケーリングし、
    物理的な相対スケールを保持する。

    Parameters
    ----------
    groups : Dict[int, List[int]]
        {group_id: [feature_indices]}
    """

    def __init__(self, groups: Dict[int, List[int]]):
        self.groups = groups
        self._group_means: Dict[int, float] = {}
        self._group_stds: Dict[int, float] = {}
        self._n_features: int = 0

    def fit(self, X: np.ndarray) -> "GroupStandardScaler":
        """グループごとの global 平均・標準偏差を計算する。"""
        self._n_features = X.shape[1]
        for group_id, feat_indices in self.groups.items():
            X_group = X[:, feat_indices]
            self._group_means[group_id] = float(X_group.mean())
            std = float(X_group.std())
            self._group_stds[group_id] = std if std > 1e-8 else 1.0
        logger.debug(
            f"GroupStandardScaler.fit: groups={list(self.groups.keys())}"
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """グループ共通スケールで変換する。"""
        X_out = X.astype(float).copy()
        for group_id, feat_indices in self.groups.items():
            mu = self._group_means[group_id]
            sigma = self._group_stds[group_id]
            X_out[:, feat_indices] = (X[:, feat_indices] - mu) / sigma
        return X_out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """逆変換（スケールを元に戻す）。"""
        X_out = X.astype(float).copy()
        for group_id, feat_indices in self.groups.items():
            mu = self._group_means[group_id]
            sigma = self._group_stds[group_id]
            X_out[:, feat_indices] = X[:, feat_indices] * sigma + mu
        return X_out
