"""
domainml/analysis/diagnostics.py
多様体構造の可視化・診断ツール（F-430）

plot_manifold_projection:
    説明変数の多様体構造を2D/3D で可視化する。
    色付けを target_values（目的変数）または satisfaction_scores（制約充足度）で行う。
    返り値は Figure オブジェクトなので、テスト環境では plt.show() を呼ばず検証可能。
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Literal
from domainml.core.metadata import FeatureMetadata
from domainml.core.logger import logger


def plot_manifold_projection(
    X: np.ndarray,
    metadata: FeatureMetadata | None = None,
    target_values: np.ndarray | None = None,
    method: Literal["umap", "lle", "isomap", "pca"] = "lle",
    n_components: Literal[2, 3] = 2,
    title: str = "Manifold Projection",
    figsize: tuple[int, int] = (8, 6),
    show: bool = False,
):
    """
    説明変数の多様体構造を低次元に射影して可視化する（F-430）

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        入力特徴量行列（manifold_flags=True の列のみを使う場合は事前にスライス）
    metadata : FeatureMetadata | None
        manifold_flags / control_flags が設定されている場合、制御変数を自動除外する
    target_values : np.ndarray | None, shape (n_samples,)
        散布図の色付けに使う値（目的変数 y など）。None なら均一色。
    method : str
        'umap' | 'lle' | 'isomap' | 'pca'
    n_components : int
        射影次元 (2 または 3)
    title : str
        グラフタイトル
    figsize : tuple
        Figure サイズ
    show : bool
        True のとき plt.show() を呼ぶ（テスト時は False 推奨）

    Returns
    -------
    fig : matplotlib.figure.Figure
        作成した Figure オブジェクト

    Raises
    ------
    ImportError
        matplotlib がインストールされていない場合
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib が必要です: pip install matplotlib"
        )

    # 制御変数を除外した特徴量列を抽出
    if metadata is not None:
        use_cols = [
            i for i in range(X.shape[1])
            if not metadata.control_flags[i]
               and (metadata.manifold_flags[i] if metadata.manifold_flags else True)
        ]
        if not use_cols:
            use_cols = list(range(X.shape[1]))
        X_sub = X[:, use_cols]
        col_names = [metadata.feature_names[i] for i in use_cols]
    else:
        X_sub = X
        col_names = [f"f{i}" for i in range(X.shape[1])]

    logger.debug(
        f"plot_manifold_projection: method={method}, "
        f"n_components={n_components}, cols={col_names}"
    )

    # 射影
    X_proj = _project(X_sub, method=method, n_components=n_components)

    # 可視化
    fig = plt.figure(figsize=figsize)
    colors = target_values if target_values is not None else np.zeros(X.shape[0])

    if n_components == 3:
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(
            X_proj[:, 0], X_proj[:, 1], X_proj[:, 2],
            c=colors, cmap="viridis", alpha=0.8, s=30
        )
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_zlabel("Dim 3")
    else:
        ax = fig.add_subplot(111)
        sc = ax.scatter(
            X_proj[:, 0], X_proj[:, 1],
            c=colors, cmap="viridis", alpha=0.8, s=30
        )
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

    if target_values is not None:
        fig.colorbar(sc, ax=ax, label="Target value")

    ax.set_title(f"{title}\n(method={method}, features={col_names})")

    if show:
        plt.show()

    return fig


def _project(X: np.ndarray, method: str, n_components: int) -> np.ndarray:
    """内部用: 指定手法で X を低次元に射影する。"""
    if method == "umap":
        try:
            from umap import UMAP
            return UMAP(n_components=n_components, random_state=42).fit_transform(X)
        except ImportError:
            raise ImportError("umap-learn が必要です: pip install umap-learn")
    elif method == "lle":
        from sklearn.manifold import LocallyLinearEmbedding
        return LocallyLinearEmbedding(
            n_components=n_components, random_state=42
        ).fit_transform(X)
    elif method == "isomap":
        from sklearn.manifold import Isomap
        return Isomap(n_components=n_components).fit_transform(X)
    elif method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=n_components, random_state=42).fit_transform(X)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'umap', 'lle', 'isomap', or 'pca'.")
