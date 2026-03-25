import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
from domainml.core.metadata import FeatureMetadata
from domainml.core.logger import logger

class ManifoldIntegrationMixin:
    """
    多様体仮説（データは低次元多様体に存在する）に基いたグラフ正則化行列を計算するミックスイン。
    制御変数（control_flags が True のもの）は多様体構造の推測（距離計算）から除外する。
    """
    
    def compute_laplacian(self, X: np.ndarray, metadata: FeatureMetadata, n_neighbors: int = 5, mode: str = 'connectivity', return_sparse: bool = True):
        # メタデータに従い、多様体の距離計算に使用する特徴量のみを抽出
        mask = [not flag for flag in metadata.control_flags]
        
        logger.debug(f"Computing Laplacian with {n_neighbors} neighbors. Control mask: {mask}")
        
        if not any(mask):
            logger.debug("All features are control variables. Skipping Manifold.")
            if return_sparse:
                return sp.csr_matrix((X.shape[0], X.shape[0]))
            return np.zeros((X.shape[0], X.shape[0]))
            
        X_manifold = X[:, mask]
        
        # 隣接行列 W を疎行列で直接取得
        W = kneighbors_graph(X_manifold, n_neighbors=n_neighbors, mode=mode, include_self=False)
        
        # 対称化 (W と W.T の加算は scipy.sparse でも可能)
        W = 0.5 * (W + W.T)
        
        # 次数行列 D (スパース行列として作成)
        D_diag = np.asarray(W.sum(axis=1)).flatten()
        D = sp.diags(D_diag)
        
        # ラプラシアン行列 L
        L = D - W
        logger.debug(f"Laplacian computed. Sparse Shape: {L.shape}")
        
        if return_sparse:
            return L
        return L.toarray()
