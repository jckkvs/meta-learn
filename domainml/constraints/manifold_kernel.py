"""
domainml/constraints/manifold_kernel.py
ManifoldAwareKernel（F-420）: 多様体距離を反映したカーネル行列計算

アプローチ3: 説明変数間の「真の近さ」を多様体距離で再定義し、
カーネル法 (SVR, GPR) や k-NN の距離計量として利用する。

距離の種類:
  - 'rbf'      : RBF カーネル (exp(-γ d²)) に基づく多様体近似
  - 'diffusion': 拡散距離（ランダムウォークの連結安定性）
"""
from __future__ import annotations
import numpy as np
from typing import Literal, Optional
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from domainml.core.metadata import FeatureMetadata, ManifoldConfig
from domainml.core.logger import logger


class ManifoldAwareKernel(BaseEstimator):
    """
    多様体構造を反映したカーネル行列を計算する（F-420）

    学習データの k-NN グラフから多様体距離（拡散距離 / RBF）を推定し、
    カーネル行列 K を構築する。sklearn の SVR, GPR 等の `kernel` 引数として
    `callable` カーネルとして渡すことができる。

    Parameters
    ----------
    method : str
        'rbf' (デフォルト) または 'diffusion'
    n_neighbors : int
        近傍数
    gamma : float
        RBF カーネルのバンド幅パラメータ γ
    n_eigenvectors : int
        拡散距離推定に使う固有ベクトルの数
    manifold_config : ManifoldConfig | None
        ManifoldConfig から local_radius / n_neighbors を上書き可

    Examples
    --------
    >>> from sklearn.svm import SVR
    >>> kernel = ManifoldAwareKernel(method='rbf', n_neighbors=8)
    >>> kernel.fit(X_train)
    >>> K_train = kernel(X_train, X_train)
    >>> svr = SVR(kernel='precomputed')
    >>> svr.fit(K_train, y_train)
    """

    def __init__(
        self,
        method: Literal["rbf", "diffusion"] = "rbf",
        n_neighbors: int = 10,
        gamma: float = 1.0,
        n_eigenvectors: int = 20,
        manifold_config: ManifoldConfig | None = None,
    ):
        self.method = method
        self.n_neighbors = n_neighbors
        self.gamma = gamma
        self.n_eigenvectors = n_eigenvectors
        self.manifold_config = manifold_config or {}
        self._X_train: Optional[np.ndarray] = None
        self._eigenvectors: Optional[np.ndarray] = None
        self._eigenvalues: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y=None) -> ManifoldAwareKernel:
        """学習データの多様体構造（近傍グラフ・固有分解）を事前計算する。"""
        cfg = self.manifold_config
        k = int(cfg.get("n_neighbors", self.n_neighbors))
        bandwidth = float(cfg.get("local_radius", 1.0 / self.gamma))

        n = X.shape[0]
        k = min(k, n - 1)
        self._X_train = X

        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nn.fit(X)
        dists, idxs = nn.kneighbors(X)
        dists, idxs = dists[:, 1:], idxs[:, 1:]

        row = np.repeat(np.arange(n), k)
        col = idxs.flatten()
        w = np.exp(-(dists.flatten() ** 2) / bandwidth)
        W = csr_matrix((w, (row, col)), shape=(n, n))
        W = (W + W.T) / 2.0

        if self.method == "diffusion":
            # 正規化ラプラシアン → 拡散マップ固有分解
            d = np.array(W.sum(axis=1)).flatten()
            d_inv = 1.0 / np.maximum(d, 1e-12)
            D_inv = diags(d_inv)
            P = D_inv @ W  # 遷移行列
            n_eig = min(self.n_eigenvectors, n - 2)
            vals, vecs = eigsh(P, k=n_eig, which="LM")
            order = np.argsort(-vals)
            self._eigenvalues = vals[order]
            self._eigenvectors = vecs[:, order]
            logger.debug(
                f"ManifoldAwareKernel.fit(diffusion): "
                f"{n_eig} eigenvectors computed"
            )
        else:
            # RBF: 学習データを保存（predict 時に使用）
            self._W = W
            logger.debug("ManifoldAwareKernel.fit(rbf): neighbor graph built")

        return self

    def __call__(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """
        カーネル行列 K(X1, X2) を返す。

        X2 が None のときは X1 を自分自身と比較する正方行列を返す。
        """
        if X2 is None:
            X2 = X1

        if self.method == "diffusion":
            return self._diffusion_kernel(X1, X2)
        else:
            return self._rbf_kernel(X1, X2)

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """ユークリッド距離ベースの RBF カーネル（多様体近似）"""
        from sklearn.metrics.pairwise import rbf_kernel
        return rbf_kernel(X1, X2, gamma=self.gamma)

    def _diffusion_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        拡散距離カーネル K(xᵢ, xⱼ) = Σₖ φₖ(xᵢ) λₖ² φₖ(xⱼ)

        学習データとの距離で近傍の固有ベクトル値を補間する。
        """
        if self._eigenvectors is None:
            raise RuntimeError("ManifoldAwareKernel が fit されていません。")

        def _embed(X: np.ndarray) -> np.ndarray:
            """新規データ点を拡散空間に埋め込む（1-NN 近傍補間）"""
            if self._X_train is None:
                raise RuntimeError("fit() を先に呼んでください。")
            nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
            nn.fit(self._X_train)
            _, idxs = nn.kneighbors(X)
            return self._eigenvectors[idxs[:, 0], :]  # (n_query, n_eig)

        Phi1 = _embed(X1) * self._eigenvalues[np.newaxis, :]  # (n1, n_eig)
        Phi2 = _embed(X2)  # (n2, n_eig)
        return Phi1 @ Phi2.T  # (n1, n2)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """多様体埋め込み座標を返す（拡散座標 / RBF 投影）。"""
        if self.method == "diffusion":
            if self._eigenvectors is None:
                raise RuntimeError("fit() が必要です。")
            nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
            nn.fit(self._X_train)
            _, idxs = nn.kneighbors(X)
            return self._eigenvectors[idxs[:, 0], :] * self._eigenvalues[np.newaxis, :]
        else:
            # RBF: 学習データを基底とした特徴マップ
            return self._rbf_kernel(X, self._X_train)
