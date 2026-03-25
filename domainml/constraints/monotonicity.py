import numpy as np
import cvxpy as cp
from sklearn.base import RegressorMixin
from domainml.core.metadata import FeatureMetadata
from domainml.models.base import DomainEstimator
from domainml.core.logger import logger

def generate_extrapolation_points(X: np.ndarray, sigma: float = 3.0, n_points: int = 100) -> np.ndarray:
    """
    学習データの平均と標準偏差ベクトルを計算し、±sigma までの外挿領域における
    仮想サポートポイントをサンプリングして返す。
    """
    mean_X = np.mean(X, axis=0)
    std_X = np.std(X, axis=0) + 1e-8 # ゼロ除算防止
    
    rng = np.random.default_rng(seed=42)
    points = rng.uniform(
        low=mean_X - sigma * std_X, 
        high=mean_X + sigma * std_X, 
        size=(n_points, X.shape[1])
    )
    return points

class MonotonicLinearRegression(DomainEstimator, RegressorMixin):
    """
    凸最適化（CVXPY）を用いて、メタデータの指定に基づく厳密（strict）および
    軟（soft）単調性制約を適用する線形回帰推定器。
    
    線形モデルにおいては単調性（重みの正負）は全定義域（外挿領域含む）で不変となるが、
    汎用的な制約の枠組みとして実装。
    """
    def __init__(self, fit_intercept: bool = True, soft_penalty_weight: float = 100.0):
        super().__init__()
        self.fit_intercept = fit_intercept
        self.soft_penalty_weight = soft_penalty_weight
        self.coef_ = None
        self.intercept_ = 0.0

    def _fit(self, X: np.ndarray, y: np.ndarray, **fit_params):
        n_samples, n_features = X.shape
        
        w = cp.Variable(n_features)
        b = cp.Variable() if self.fit_intercept else 0.0
        
        # 基本の L2 損失
        loss = cp.sum_squares(X @ w + b - y)
        
        constraints = []
        
        if self.metadata_ is not None:
            for i in range(n_features):
                mono = self.metadata_.monotonicities[i]
                ctype = self.metadata_.constraint_types[i]
                
                if mono == 'inc':
                    if ctype == 'strict':
                         constraints.append(w[i] >= 0)
                    elif ctype == 'soft':
                         # Using sum_squares for better penalty scaling as recommended
                         loss += self.soft_penalty_weight * cp.sum_squares(cp.pos(-w[i]))
                         
                elif mono == 'dec':
                    if ctype == 'strict':
                         constraints.append(w[i] <= 0)
                    elif ctype == 'soft':
                         loss += self.soft_penalty_weight * cp.sum_squares(cp.pos(w[i]))
            
            logger.debug(f"Added monotonic constraints from metadata for {n_features} features. Constraints len: {len(constraints)}")

            # extrapolation_sigma に基づく外挿サポート（線形モデルに実影響はないが、APIとして定義）
            sigma = getattr(self.metadata_, 'extrapolation_sigma', 3.0)
            if sigma > 0:
                points = generate_extrapolation_points(X, sigma=sigma, n_points=10)
                logger.debug(f"Generated {len(points)} extrapolation points at sigma={sigma}")
                
        prob = cp.Problem(cp.Minimize(loss), constraints)
        
        logger.debug(f"CVXPY problem configured. Solving...")
        # 解を求める（OSQP, SCS, ECOS等のデフォルトソルバ利用）
        prob.solve()
        logger.debug(f"CVXPY solve completed with status: {prob.status}")
        
        if prob.status not in ["infeasible", "unbounded", None] and w.value is not None:
            # 最適化成功
            self.coef_ = w.value
            self.intercept_ = b.value if self.fit_intercept else 0.0
        else:
            raise ValueError(f"CVXPY Optimization failed. Status: {prob.status}, w.value: {w.value}")

    def _predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return X @ self.coef_ + self.intercept_

