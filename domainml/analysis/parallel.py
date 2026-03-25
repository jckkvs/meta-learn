import numpy as np
from joblib import Parallel, delayed
from domainml.core.metadata import FeatureMetadata
from domainml.core.logger import logger
from sklearn.linear_model import LinearRegression

def _evaluate_single_feature(i: int, X: np.ndarray, y: np.ndarray, mono: str, feature_name: str, threshold: float):
    if mono == 'none':
        return None
        
    # 簡易重回帰による部分的な効果推定
    lr = LinearRegression()
    lr.fit(X, y)
    observed_effect = lr.coef_[i]
    
    if mono == 'inc' and observed_effect < -threshold:
        return {
            'feature_index': i,
            'feature_name': feature_name,
            'constraint': 'inc',
            'observed_effect': observed_effect,
            'severity': abs(observed_effect)
        }
    elif mono == 'dec' and observed_effect > threshold:
        return {
            'feature_index': i,
            'feature_name': feature_name,
            'constraint': 'dec',
            'observed_effect': observed_effect,
            'severity': abs(observed_effect)
        }
    return None

def parallel_check_conflicts(X: np.ndarray, y: np.ndarray, metadata: FeatureMetadata, threshold: float = 0.1, n_jobs=-1) -> list:
    """
    LinearCoefConflictChecker のロジックを feature ごとに並列化して実行する。
    """
    logger.debug(f"Running parallel conflict detection on {metadata.n_features} features using n_jobs={n_jobs}")
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_single_feature)(
            i, X, y, metadata.monotonicities[i], metadata.feature_names[i], threshold
        ) for i in range(metadata.n_features)
    )
    
    conflicts = [r for r in results if r is not None]
    return conflicts
