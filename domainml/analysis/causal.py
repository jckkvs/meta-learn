import numpy as np
from sklearn.linear_model import LinearRegression
from domainml.core.metadata import FeatureMetadata
from domainml.core.logger import logger

class CausalConflictDetector:
    """
    データ駆動および因果推論のアプローチ (線形回帰・偏相関の簡易モデル) 
    を用いて、ユーザーから与えられた単調性制約がデータ上で激しく矛盾 (競合) している
    特徴量ペアや関係性を検出する。
    """
    def __init__(self, significance_level: float = 0.05, threshold: float = 0.1):
        self.significance_level = significance_level
        self.threshold = threshold

    def detect_conflicts(self, X: np.ndarray, y: np.ndarray, metadata: FeatureMetadata) -> list:
        """
        単調性制約が設定された特徴量に対して、データから観測される変数の
        効果方向が制約と逆行している場合、その違反をリストとして返す。
        """
        conflicts = []
        n_features = metadata.n_features
        
        # 簡易的にすべての特徴量を含む重回帰を行い、他の特徴量で条件付けた上での
        # 各特徴量の y への因果的（偏）効果および相関方向を推定
        lr = LinearRegression()
        lr.fit(X, y)
        coefs = lr.coef_
        
        for i in range(n_features):
            mono = metadata.monotonicities[i]
            if mono == 'none':
                continue
                
            observed_effect = coefs[i]
            
            # 強い逆行効果があるかチェック
            if mono == 'inc' and observed_effect < -self.threshold:
                logger.warning(f"Causal conflict detected on {metadata.feature_names[i]}: constraint is 'inc', but observed effect is {observed_effect:.3f}")
                conflicts.append({
                    'feature_index': i,
                    'feature_name': metadata.feature_names[i],
                    'constraint': 'inc',
                    'observed_effect': observed_effect,
                    'severity': abs(observed_effect)
                })
            elif mono == 'dec' and observed_effect > self.threshold:
                logger.warning(f"Causal conflict detected on {metadata.feature_names[i]}: constraint is 'dec', but observed effect is {observed_effect:.3f}")
                conflicts.append({
                    'feature_index': i,
                    'feature_name': metadata.feature_names[i],
                    'constraint': 'dec',
                    'observed_effect': observed_effect,
                    'severity': abs(observed_effect)
                })
                
        return conflicts
