"""
domainml/constraints/manifold_engine.py
多様体仮説エンジン（F-320）

ManifoldAssumptionEngine:
  - 制御変数（non_manifold_variables）を除外した観測変数のみで k-NN グラフを構築
  - 疎なグラフラプラシアン行列 L を計算
  - CVXPY 損失関数に正則化項 λ f^T L f を付加するヘルパーを提供
"""
import numpy as np
from typing import List, Optional
from scipy.sparse import csr_matrix, diags
from sklearn.neighbors import NearestNeighbors
from domainml.core.logger import logger

try:
    import cvxpy as cp
    _CVXPY_AVAILABLE = True
except ImportError:
    _CVXPY_AVAILABLE = False


class ManifoldAssumptionEngine:
    """
    多様体仮説エンジン（F-320）

    「観測変数は低次元多様体上に存在し、近傍点は予測値も近傍である」という
    ドメイン仮説を、グラフラプラシアン正則化として損失関数に統合する。

    制御変数（実験設計で操作される変数）は多様体構造を持たないと仮定し、
    グラフ構築から除外できる。

    Parameters
    ----------
    manifold_variables : List[int]
        多様体構造を持つ変数の列インデックス（観測変数）。
    non_manifold_variables : List[int]
        多様体仮説が成立しない変数の列インデックス（制御変数・操作変数）。
    n_neighbors : int
        k-NN グラフのご近所数。
    bandwidth : float
        RBF カーネルのバンド幅（類似度 exp(-d²/bandwidth)）。
    """

    def __init__(
        self,
        manifold_variables: List[int],
        non_manifold_variables: Optional[List[int]] = None,
        n_neighbors: int = 10,
        bandwidth: float = 0.1,
    ):
        self.manifold_variables = manifold_variables
        self.non_manifold_variables = non_manifold_variables or []
        self.n_neighbors = n_neighbors
        self.bandwidth = bandwidth
        self._laplacian_matrix: Optional[csr_matrix] = None
        self._n_samples: int = 0

    def build_laplacian_regularization(self, X: np.ndarray) -> csr_matrix:
        """
        多様体変数だけを使って正規化グラフラプラシアン行列 L を構築する。（F-320）

        手順:
          1. manifold_variables の列のみを抽出
          2. k-NN グラフの隣接行列 W を RBF 重みで構成
          3. 次数行列 D と L = D - W を計算

        Returns
        -------
        L : scipy.sparse.csr_matrix, shape (n_samples, n_samples)
        """
        X_manifold = X[:, self.manifold_variables]
        n_samples = X_manifold.shape[0]
        self._n_samples = n_samples

        k = min(self.n_neighbors, n_samples - 1)

        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="auto")
        nn.fit(X_manifold)
        distances, indices = nn.kneighbors(X_manifold)

        # 自分自身（index 0）を除く
        distances = distances[:, 1:]
        indices = indices[:, 1:]

        row_idx = np.repeat(np.arange(n_samples), k)
        col_idx = indices.flatten()
        weights = np.exp(-(distances.flatten() ** 2) / self.bandwidth)

        W = csr_matrix((weights, (row_idx, col_idx)), shape=(n_samples, n_samples))
        # 対称化
        W = (W + W.T) / 2.0

        degree = np.array(W.sum(axis=1)).flatten()
        D = diags(degree)
        L = D - W

        self._laplacian_matrix = L
        logger.debug(
            f"ManifoldAssumptionEngine: Laplacian built. "
            f"n_samples={n_samples}, k={k}, "
            f"manifold_vars={self.manifold_variables}, "
            f"excluded_vars={self.non_manifold_variables}"
        )
        return L

    def apply_manifold_regularization(
        self, loss, f, lambda_m: float = 1e-3
    ):
        """
        CVXPY の損失式に多様体正則化項 λ * f^T L f を追加する。（F-320）

        Parameters
        ----------
        loss : cvxpy.Expression
            現在の損失式。
        f : cvxpy.Variable, shape (n_samples,)
            予測値変数。
        lambda_m : float
            正則化強度。

        Returns
        -------
        augmented_loss : cvxpy.Expression
            正則化項を加算した損失式。
        """
        if self._laplacian_matrix is None:
            logger.warning(
                "ManifoldAssumptionEngine: Laplacian not built. "
                "Call build_laplacian_regularization(X) first. Returning loss unchanged."
            )
            return loss

        if not _CVXPY_AVAILABLE:
            logger.warning("cvxpy is not available; skipping manifold regularization.")
            return loss

        # 疎行列を密行列に変換して CVXPY で計算
        L_dense = self._laplacian_matrix.toarray()
        manifold_reg = cp.quad_form(f, L_dense)
        logger.debug(f"ManifoldAssumptionEngine: Adding regularization with λ={lambda_m}")
        return loss + lambda_m * manifold_reg

    def get_laplacian(self) -> Optional[csr_matrix]:
        """構築済みラプラシアン行列を返す（診断用）。"""
        return self._laplacian_matrix
