import numpy as np
import pandas as pd
from typing import List
from domainml.core.metadata import FeatureMetadata
from domainml.core.logger import logger
import warnings

class ConstraintConflictDetector:
    """
    データとメタデータを照らし合わせ、制約とデータの統計的性質が矛盾していないかを検出する
    ・強い正相関があるのに片方は増加、片方は減少指定
    ・強い負相関があるのに同じ方向に指定
    """
    def __init__(self, correlation_threshold: float = 0.8):
        self.correlation_threshold = correlation_threshold
        
    def detect(self, X: np.ndarray, metadata: FeatureMetadata) -> List[str]:
        warnings_list = []
        n_features = metadata.n_features
        
        logger.debug(f"ConflictDetector started on {n_features} features with threshold {self.correlation_threshold}")
        
        df = pd.DataFrame(X)
        corr_matrix = df.corr().fillna(0).values
        
        logger.debug("Correlations matrix computed.")
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                # 強い正相関
                if corr_matrix[i, j] > self.correlation_threshold:
                    if (metadata.monotonicities[i] == 'inc' and metadata.monotonicities[j] == 'dec') or \
                       (metadata.monotonicities[i] == 'dec' and metadata.monotonicities[j] == 'inc'):
                        warnings_list.append(
                            f"Conflict: feature {i} ({metadata.feature_names[i]}) and "
                            f"feature {j} ({metadata.feature_names[j]}) are "
                            f"highly correlated ({corr_matrix[i,j]:.2f}) but have opposite monotonicity constraints."
                        )
                        
                # 強い負相関
                elif corr_matrix[i, j] < -self.correlation_threshold:
                    if (metadata.monotonicities[i] == 'inc' and metadata.monotonicities[j] == 'inc') or \
                       (metadata.monotonicities[i] == 'dec' and metadata.monotonicities[j] == 'dec'):
                         warnings_list.append(
                            f"Conflict: feature {i} ({metadata.feature_names[i]}) and "
                            f"feature {j} ({metadata.feature_names[j]}) are "
                            f"highly negatively correlated ({corr_matrix[i,j]:.2f}) but have same monotonicity constraints."
                        )
        
        for w in warnings_list:
            logger.warning(w)
            warnings.warn(w, UserWarning)
            
        return warnings_list
