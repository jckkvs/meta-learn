"""
domainml/constraints/manifold_regularizer.py
多様体正則化パイプラインステップ（F-410, F-411）

ManifoldRegularizer : sklearn TransformerMixin として
    - fit:      近傍グラフ + ラプラシアン行列を構築
    - transform: 恒等変換（特徴量は変更しない）
    - get_regularization_term: CVXPY 式 f^T L f を返す

ManifoldPreprocessor : 多様体学習による次元削減前処理器（F-412）
    - UMAP / LLE などを使った特徴量変換
    - オプショナル依存（umap-learn）に対応
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Literal
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import diags
from domainml.core.metadata import FeatureMetadata, ManifoldConfig
from domainml.core.logger import logger


# ───────────────────────────────────────────────────────────────────────────────
# ManifoldRegularizer
# ───────────────────────────────────────────────────────────────────────────────

class ManifoldRegularizer(BaseEstimator, TransformerMixin):
    """
    多様体構造に基づく正則化をパイプラインに注入する Transformer（F-410）

    fit 時にグラフラプラシアン L を構築し、transform では恒等変換を行う。
    モデルが制約付き最適化（CVXPY）を使う場合、`get_regularization_term(f)` を
    呼び出すことで損失関数に L_manifold = λ f^T L f を追加できる。

    Parameters
    ----------
    manifold_config : ManifoldConfig | None
        多様体設定。None の場合は metadata.manifold_flags の有効特徴量すべてを対象とする
    metadata : FeatureMetadata | None
        制御変数などの除外情報
    n_neighbors : int
        近傍数（manifold_config['n_neighbors'] で上書き可）
    bandwidth : float
        RBF カーネルバンド幅（manifold_config['local_radius'] で上書き可）
    """

    def __init__(
        self,
        manifold_config: ManifoldConfig | None = None,
        metadata: FeatureMetadata | None = None,
        n_neighbors: int = 10,
        bandwidth: float = 0.1,
    ):
        self.manifold_config = manifold_config or {}
        self.metadata = metadata
        self.n_neighbors = n_neighbors
        self.bandwidth = bandwidth
        self._laplacian: Optional[csr_matrix] = None
        self._manifold_cols: list[int] = []

    def fit(self, X: np.ndarray, y=None) -> ManifoldRegularizer:
        """近傍グラフとグラフラプラシアン L を構築する。"""
        n_samples, n_features = X.shape
        cfg = self.manifold_config

        # 多様体変数の列を決定
        if self.metadata is not None:
            self._manifold_cols = [
                i for i, flag in enumerate(self.metadata.manifold_flags)
                if flag and not self.metadata.control_flags[i]
            ]
        else:
            self._manifold_cols = list(range(n_features))

        if not self._manifold_cols:
            logger.warning("ManifoldRegularizer: no manifold columns found. Skipping Laplacian build.")
            return self

        X_sub = X[:, self._manifold_cols]
        k = min(
            int(cfg.get("n_neighbors", self.n_neighbors)),
            n_samples - 1
        )
        bandwidth = float(cfg.get("local_radius", self.bandwidth))

        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nn.fit(X_sub)
        dists, idxs = nn.kneighbors(X_sub)
        dists, idxs = dists[:, 1:], idxs[:, 1:]  # 自分自身を除外

        row = np.repeat(np.arange(n_samples), k)
        col = idxs.flatten()
        w = np.exp(-(dists.flatten() ** 2) / bandwidth)

        W = csr_matrix((w, (row, col)), shape=(n_samples, n_samples))
        W = (W + W.T) / 2.0
        D = diags(np.array(W.sum(axis=1)).flatten())
        self._laplacian = D - W

        logger.debug(
            f"ManifoldRegularizer.fit: Laplacian built. "
            f"n_samples={n_samples}, k={k}, manifold_cols={self._manifold_cols}"
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """恒等変換（特徴量は変更しない）。Laplacian は get_regularization_term で使う。"""
        return X

    def get_regularization_term(self, f, lambda_m: float | None = None):
        """
        CVXPY 損失式に追加する多様体正則化項 λ f^T L f を返す。（F-410）

        Parameters
        ----------
        f : cp.Variable, shape (n_samples,)
        lambda_m : float | None
            正則化強度。None の場合は manifold_config['regularization_weight'] を使用。
        """
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError(
                "cvxpy が必要です: pip install domainml-meta[convex]"
            )

        if self._laplacian is None:
            logger.warning(
                "ManifoldRegularizer: Laplacian not built. "
                "Call fit(X) before get_regularization_term()."
            )
            return cp.Constant(0)

        lam = lambda_m if lambda_m is not None else float(
            self.manifold_config.get("regularization_weight", 0.1)
        )
        L_dense = self._laplacian.toarray()
        reg_term = cp.quad_form(f, L_dense)
        logger.debug(f"ManifoldRegularizer: regularization term with λ={lam}")
        return lam * reg_term

    def get_laplacian(self) -> Optional[csr_matrix]:
        """構築済みラプラシアン行列を返す（診断・可視化用）。"""
        return self._laplacian


# ───────────────────────────────────────────────────────────────────────────────
# ManifoldPreprocessor
# ───────────────────────────────────────────────────────────────────────────────

class ManifoldPreprocessor(BaseEstimator, TransformerMixin):
    """
    多様体学習による次元削減前処理器（F-412）

    高次元説明変数を低次元多様体座標に変換する。
    変換後の低次元座標を新しい説明変数として、または
    既存特徴量と連結して利用できる。

    Parameters
    ----------
    n_components : int
        埋め込みの次元数
    method : str
        多様体学習手法。 'umap' | 'lle' | 'isomap'
    append : bool
        True の場合は元の特徴量に埋め込み座標を連結して返す。
        False の場合は埋め込み座標のみ返す。
    random_state : int
    """

    def __init__(
        self,
        n_components: int = 10,
        method: Literal["umap", "lle", "isomap"] = "lle",
        append: bool = False,
        random_state: int = 42,
        **kwargs,
    ):
        self.n_components = n_components
        self.method = method
        self.append = append
        self.random_state = random_state
        self.kwargs = kwargs
        self.embedder_ = None

    def fit(self, X: np.ndarray, y=None) -> ManifoldPreprocessor:
        """多様体埋め込みを学習する。"""
        if self.method == "umap":
            try:
                from umap import UMAP
            except ImportError:
                raise ImportError(
                    "umap-learn が必要です: pip install umap-learn"
                )
            self.embedder_ = UMAP(
                n_components=self.n_components,
                random_state=self.random_state,
                **self.kwargs,
            )
        elif self.method == "lle":
            from sklearn.manifold import LocallyLinearEmbedding
            self.embedder_ = LocallyLinearEmbedding(
                n_components=self.n_components,
                random_state=self.random_state,
                **self.kwargs,
            )
        elif self.method == "isomap":
            from sklearn.manifold import Isomap
            self.embedder_ = Isomap(
                n_components=self.n_components,
                **self.kwargs,
            )
        else:
            raise ValueError(f"Unknown method: {self.method!r}. Use 'umap', 'lle', or 'isomap'.")

        self.embedder_.fit(X)
        self.in_features_ = X.shape[1]
        logger.debug(
            f"ManifoldPreprocessor.fit: method={self.method}, "
            f"n_components={self.n_components}, input_dim={self.in_features_}"
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """高次元 X を低次元多様体座標に変換する。"""
        if self.embedder_ is None:
            raise RuntimeError("ManifoldPreprocessor が未学習です。fit() を先に呼んでください。")

        X_embedded = self.embedder_.transform(X)

        if self.append:
            return np.hstack([X, X_embedded])
        return X_embedded

    @property
    def out_features_(self) -> int:
        """変換後の特徴量数。"""
        if self.embedder_ is None:
            raise RuntimeError("ManifoldPreprocessor が未学習です。")
        if self.append:
            return self.in_features_ + self.n_components
        return self.n_components
