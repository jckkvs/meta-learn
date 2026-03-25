import numpy as np
from sklearn.neighbors import kneighbors_graph
from domainml.core.metadata import FeatureMetadata
from domainml.core.logger import logger

class ManifoldIntegrationMixin:
    """
    多様体仮説（データは低次元多様体に存在する）に基いたグラフ正則化行列を計算するミックスイン。
    制御変数（control_flags が True のもの）は多様体構造の推測（距離計算）から除外する。
    """
    
    def compute_laplacian(self, X: np.ndarray, metadata: FeatureMetadata, n_neighbors: int = 5, mode: str = 'connectivity') -> np.ndarray:
        """
        X からグラフ・ラプラシアン行列 L = D - W を計算する。
        """
        # メタデータに従い、多様体の距離計算に使用する特徴量のみを抽出
        mask = [not flag for flag in metadata.control_flags]
        
        logger.debug(f"Computing Laplacian with {n_neighbors} neighbors. Control mask: {mask}")
        
        if not any(mask):
            logger.debug("All features are control variables. Skipping Manifold.")
            return np.zeros((X.shape[0], X.shape[0]))
            
        X_manifold = X[:, mask]
        
        # 隣接行列 W を計算
        W = kneighbors_graph(X_manifold, n_neighbors=n_neighbors, mode=mode, include_self=False).toarray()
        
        # 対称化
        W = 0.5 * (W + W.T)
        
        # 次数行列 D
        D = np.diag(W.sum(axis=1))
        
        # ラプラシアン行列 L
        L = D - W
        logger.debug(f"Laplacian computed. Shape: {L.shape}")
        return L
