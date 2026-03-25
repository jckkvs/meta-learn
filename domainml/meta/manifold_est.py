import numpy as np
from sklearn.decomposition import PCA
from domainml.core.logger import logger

class ManifoldAssumption:
    """
    データの本質的次元（Intrinsic Dimension）を推計し、
    多様体仮説の妥当性を評価する。
    """
    
    @staticmethod
    def estimate_intrinsic_dimension(X: np.ndarray, variance_ratio: float = 0.95) -> int:
        """
        PCAの累積寄与率を用いてデータの真の次元を推定する簡易メソッド。
        将来的に k-NN 距離を用いた MLE (Maximum Likelihood Estimation) 等への
        拡張を想定している。
        """
        n_samples, n_features = X.shape
        if n_samples < 2:
            return n_features
            
        pca = PCA()
        pca.fit(X)
        
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        intrinsic_dim = np.searchsorted(cumulative_variance, variance_ratio) + 1
        
        logger.debug(f"Estimated Intrinsic Dimension: {intrinsic_dim} (Original features: {n_features})")
        return int(intrinsic_dim)
        
    @staticmethod
    def validate_assumption(X: np.ndarray, threshold_ratio: float = 0.8) -> bool:
        """
        データが実際に低次元多様体に存在するかどうかを判定する。
        推定次元が元の特徴量数の threshold_ratio 倍未満であれば、多様体仮説は有効とする。
        """
        intrinsic_dim = ManifoldAssumption.estimate_intrinsic_dimension(X)
        n_features = X.shape[1]
        is_valid = intrinsic_dim <= max(1, n_features * threshold_ratio)
        logger.debug(f"Manifold assumption validity: {is_valid}")
        return is_valid
