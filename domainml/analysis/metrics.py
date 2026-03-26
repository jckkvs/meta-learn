import numpy as np
from domainml.core.metadata import FeatureMetadata
from domainml.core.logger import logger

def satisfaction_score(estimator, X_val: np.ndarray, metadata: FeatureMetadata, n_samples: int = 100, eps: float | None = None) -> float:
    """
    モデルの予測結果がメタデータの単調性制約をどの程度満たしているかを
    [0.0, 1.0] のスコアで返す。
    
    eps=None の場合，各特徴量の IQR（1Q～3Q）に対する相対摂動幅を自動計算する。
    スケール非依存の摂動により、正規化されていないデータでも誤判定を防ぐ。
    """
    constrained_indices = [i for i, m in enumerate(metadata.monotonicities) if m != 'none']
    if not constrained_indices:
        return 1.0 # 制約なしの場合は常に充足
        
    n_total_checks = 0
    n_satisfied = 0

    # ランダムに n_samples 個のポイントを選択
    rng = np.random.default_rng(seed=42)
    indices = rng.choice(X_val.shape[0], min(n_samples, X_val.shape[0]), replace=False)
    X_sample = X_val[indices]
    
    # 基準予測値
    y_base = estimator.predict(X_sample)
    
    for idx in constrained_indices:
        mono = metadata.monotonicities[idx]
        
        # IQR ベースの摂動幅（レビュー指摘対応: 特徴量スケールに依存しない相対摂動）
        if eps is None:
            q75, q25 = np.percentile(X_sample[:, idx], [75, 25])
            feature_eps = max((q75 - q25) * 1e-3, 1e-6)
        else:
            feature_eps = eps
        
        # 摂動を与えたXを作成
        X_perturbed = X_sample.copy()
        X_perturbed[:, idx] += feature_eps
        
        y_perturbed = estimator.predict(X_perturbed)
        diff = y_perturbed - y_base
        
        n_total_checks += len(y_base)
        
        if mono == 'inc':
            # 単調増加: diff >= 0なら満たす (計算誤差許容)
            n_satisfied += np.sum(diff >= -1e-8)
        elif mono == 'dec':
            # 単調減少: diff <= 0なら満たす
            n_satisfied += np.sum(diff <= 1e-8)
            
    score = n_satisfied / n_total_checks if n_total_checks > 0 else 1.0
    logger.debug(f"Calculated satisfaction score: {score:.3f} ({n_satisfied}/{n_total_checks})")
    
    return float(score)
