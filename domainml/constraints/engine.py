import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from typing import Literal
from domainml.core.metadata import FeatureMetadata
from domainml.core.logger import logger
from domainml.constraints.monotonicity import MonotonicLinearRegression
from domainml.models.wrappers import MonotonicityWrapper

class MonotonicityEngine(BaseEstimator, RegressorMixin):
    """
    モデルカテゴリに応じた自動的な単調性制約のファクトリエンジン。
    与えられた estimator の型に応じて最適な制約手法を選択し、学習・推論を代行する。
    """
    def __init__(self, estimator, constraint_type: Literal["strict", "soft"] = "strict",
                 extrapolation_sigma: float = 2.0):
        self.estimator = estimator
        self.constraint_type = constraint_type
        self.extrapolation_sigma = extrapolation_sigma
        
    def fit(self, X: np.ndarray, y: np.ndarray, metadata: FeatureMetadata = None, **fit_params):
        if metadata is None or not any(m != 'none' for m in metadata.monotonicities):
            logger.debug("No metadata constraints provided. Fitting bare estimator.")
            self.model_ = clone(self.estimator)
            self.model_.fit(X, y, **fit_params)
            return self
            
        model_category = self._detect_model_category(self.estimator)
        logger.debug(f"Detected model category: {model_category}")
        
        if model_category == "tree_based":
            self.model_ = self._apply_tree_constraint(X, y, metadata, **fit_params)
        elif model_category == "linear":
            self.model_ = self._apply_linear_constraint(X, y, metadata, **fit_params)
        elif model_category == "kernel":
            self.model_ = self._apply_kernel_constraint(X, y, metadata, **fit_params)
        else:
            self.model_ = self._apply_fallback_constraint(X, y, metadata, **fit_params)
            
        return self
        
    def _detect_model_category(self, estimator) -> str:
        name = estimator.__class__.__name__.lower()
        if "gradientboosting" in name or "forest" in name or "tree" in name or "lgbm" in name or "xgb" in name:
            return "tree_based"
        elif "linear" in name or "ridge" in name or "lasso" in name:
            return "linear"
        elif "kernel" in name or "svr" in name or "gaussian" in name:
            return "kernel"
        else:
            return "unknown"
            
    def _apply_tree_constraint(self, X, y, metadata, **fit_params):
        model = clone(self.estimator)
        logger.debug("Attempting to apply native monotonicity constraints to tree estimator")
        monotonic_cst = np.zeros(metadata.n_features)
        for idx, mono in enumerate(metadata.monotonicities):
            if mono == 'inc':
                monotonic_cst[idx] = 1
            elif mono == 'dec':
                monotonic_cst[idx] = -1
                
        try:
            model.set_params(monotone_constraints=monotonic_cst.tolist())
            model.fit(X, y, **fit_params)
            return model
        except ValueError:
            pass
            
        try:
            model.set_params(monotonic_cst=monotonic_cst.tolist())
            model.fit(X, y, **fit_params)
            return model
        except ValueError:
            pass
            
        # Fallback if tree model lacks monotonic_cst Support
        logger.debug("Tree estimator lacks monotonic constraints. Falling back to wrapper.")
        return self._apply_fallback_constraint(X, y, metadata, **fit_params)

    def _apply_linear_constraint(self, X, y, metadata, **fit_params):
        logger.debug("Applying exact CVXPY monotonic linear regression")
        model = MonotonicLinearRegression()
        model.fit(X, y, metadata=metadata, **fit_params)
        return model
        
    def _apply_kernel_constraint(self, X, y, metadata, **fit_params):
        logger.debug("Applying virtual point constraints for kernel method")
        from domainml.constraints.kernel import KernelMonotonicity
        model = KernelMonotonicity(
            estimator=clone(self.estimator),
            constraint_type=self.constraint_type,
            extrapolation_sigma=self.extrapolation_sigma
        )
        model.fit(X, y, metadata=metadata, **fit_params)
        return model

    def _apply_fallback_constraint(self, X, y, metadata, **fit_params):
        logger.debug("Applying isotonic regression fallback wrapper")
        model = MonotonicityWrapper(base_estimator=clone(self.estimator))
        model.fit(X, y, metadata=metadata, **fit_params)
        return model
        
    def predict(self, X: np.ndarray, metadata: FeatureMetadata = None) -> np.ndarray:
        # 内部モデルがメタデータをサポートする場合は渡す
        if hasattr(self.model_, 'predict') and getattr(self.model_.predict, '__code__', None):
            varnames = self.model_.predict.__code__.co_varnames
            if 'metadata' in varnames:
                return self.model_.predict(X, metadata=metadata)
        return self.model_.predict(X)
