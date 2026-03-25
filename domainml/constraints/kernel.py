import numpy as np
import cvxpy as cp
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import rbf_kernel
from domainml.core.metadata import FeatureMetadata
from domainml.core.logger import logger

class KernelMonotonicity(BaseEstimator, RegressorMixin):
    """
    カーネル法（GPR, SVR）向けに、仮想制約点(Virtual Constraint Points)を用いた
    CVXPYベースの厳密単調性最適化エンジン。
    """
    def __init__(self, estimator=None, constraint_type="strict", extrapolation_sigma=2.0):
        self.estimator = estimator
        self.constraint_type = constraint_type
        self.extrapolation_sigma = extrapolation_sigma
        
    def fit(self, X: np.ndarray, y: np.ndarray, metadata: FeatureMetadata = None, **fit_params):
        self.X_train_ = X.copy()
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # Gamma setting identical to standard sklearn RBF default
        self.gamma_ = 1.0 / n_features
        K = rbf_kernel(X, X, gamma=self.gamma_)
        
        alpha = cp.Variable(n_samples)
        
        # L2 Regularization on alpha directly to avoid PSD wrap hanging issues
        lambda_reg = 0.1
        loss = cp.sum_squares(K @ alpha - y) + lambda_reg * cp.sum_squares(alpha)
        
        constraints = []
        if metadata is not None and any(m != 'none' for m in metadata.monotonicities):
            # 仮想制約点を生成し、そこでの勾配を制約する
            virtual_points = self._generate_virtual_points(X)
            logger.debug(f"Generated {len(virtual_points)} virtual constraint points for Kernel optimization")
            
            K_v = rbf_kernel(virtual_points, X, gamma=self.gamma_)
            
            for feat_idx, mono in enumerate(metadata.monotonicities):
                if mono == 'none':
                    continue
                constraint_mode = metadata.constraint_types[feat_idx] if metadata.constraint_types[feat_idx] != 'none' else self.constraint_type
                for v_idx, v in enumerate(virtual_points):
                    diff = v[feat_idx] - X[:, feat_idx]  # shape: (n_samples,)
                    # カーネル微分の導出
                    gradient_row = -2.0 * self.gamma_ * diff * K_v[v_idx, :]
                    
                    if mono == 'inc':
                        if constraint_mode == 'strict':
                            constraints.append(gradient_row @ alpha >= 0)
                        else:
                            loss += 1.0 * cp.sum(cp.neg(gradient_row @ alpha)) # soft
                    elif mono == 'dec':
                        if constraint_mode == 'strict':
                            constraints.append(gradient_row @ alpha <= 0)
                        else:
                            loss += 1.0 * cp.sum(cp.pos(gradient_row @ alpha)) # soft

        prob = cp.Problem(cp.Minimize(loss), constraints)
        logger.debug("Solving Kernel Monotonicity optimization problem over virtual constraints...")
        prob.solve()
        
        if prob.status not in ["infeasible", "unbounded", None]:
            self.alpha_ = alpha.value
        else:
            logger.warning("Kernel Monotonicity problem infeasible. Falling back to simple unconstrained Ridge.")
            # Unconstrained solution: alpha = (K + lambda*I)^-1 y
            self.alpha_ = np.linalg.solve(K + lambda_reg * np.eye(n_samples), y)
            
        return self
        
    def _generate_virtual_points(self, X: np.ndarray) -> np.ndarray:
        """
        学習データ分布および外挿領域をカバーする仮想制約点を生成。
        次元の呪いを防ぐため乱数サンプリングを利用。
        """
        X_min = X.mean(axis=0) - self.extrapolation_sigma * X.std(axis=0)
        X_max = X.mean(axis=0) + self.extrapolation_sigma * X.std(axis=0)
        n_virtual = min(X.shape[0] * 2, 100)
        rng = np.random.RandomState(42)
        return rng.uniform(X_min, X_max, size=(n_virtual, X.shape[1]))

    def predict(self, X: np.ndarray, metadata: FeatureMetadata = None) -> np.ndarray:
        K = rbf_kernel(X, self.X_train_, gamma=self.gamma_)
        return K @ self.alpha_
