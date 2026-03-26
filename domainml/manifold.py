"""
domainml/manifold.py
多様体仮説を適用するためのユーザー向け簡易インターフェースモジュール。
"""
from __future__ import annotations
import numpy as np
from typing import Literal
from domainml.core.metadata import FeatureMetadata
from domainml.constraints.manifold_regularizer import ManifoldPreprocessor

def manifold_embed(
    X: np.ndarray,
    metadata: FeatureMetadata | None = None,
    method: Literal["umap", "lle", "isomap", "pca"] = "lle",
    n_components: int = 5,
    append: bool = False,
    random_state: int = 42
) -> np.ndarray:
    """
    1行で多様体埋め込み特徴量を生成する簡易関数。
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    metadata : FeatureMetadata | None
        指定された場合、制御変数 (control_flags=True) や 
        多様体除外対象 (manifold_flags=False) を埋め込み計算から除外します。
    method : str
        多様体学習手法。 'umap' | 'lle' | 'isomap' | 'pca'
    n_components : int
        埋め込み先の次元数
    append : bool
        True の場合は元の特徴量行列 X の末尾に埋め込み次元を追加して返す
    random_state : int
    
    Returns
    -------
    np.ndarray
        変換後の特徴量行列
    """
    from domainml.analysis.diagnostics import _project
    
    if metadata is not None:
        use_cols = [
            i for i in range(X.shape[1])
            if not metadata.control_flags[i]
               and (metadata.manifold_flags[i] if any(metadata.manifold_flags) else True)
        ]
        if not use_cols:
            use_cols = list(range(X.shape[1]))
        X_sub = X[:, use_cols]
    else:
        X_sub = X

    X_embedded = _project(X_sub, method=method, n_components=n_components)
    
    if append:
        return np.hstack([X, X_embedded])
    return X_embedded
