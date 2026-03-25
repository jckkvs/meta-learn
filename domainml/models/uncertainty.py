import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils import resample
from domainml.core.metadata import FeatureMetadata
from domainml.core.logger import logger

class UncertaintyEstimator(BaseEstimator, RegressorMixin):
    """
    Bootstrapサンプリングを用いて、基礎となる制約付き推論器の
    予測不確実性（信頼区間）を推定するメタ・エスティメータ。
    """
    def __init__(self, base_estimator, n_estimators=10, random_state=42):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_ = []

    def fit(self, X: np.ndarray, y: np.ndarray, metadata: FeatureMetadata = None, **fit_params):
        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        
        logger.debug(f"Training {self.n_estimators} bootstrap estimators for UncertaintyEstimator")
        for i in range(self.n_estimators):
            X_resampled, y_resampled = resample(X, y, random_state=rng.integers(10000))
            model = clone(self.base_estimator)
            
            try:
                model.fit(X_resampled, y_resampled, metadata=metadata, **fit_params)
            except TypeError:
                model.fit(X_resampled, y_resampled, **fit_params)
                
            self.estimators_.append(model)
        return self

    def predict(self, X: np.ndarray, metadata: FeatureMetadata = None, return_interval=False, alpha=0.05):
        """
        全モデルの予測の平均を返す。
        return_interval=True の場合は、(平均予測値, 下側信頼限界, 上側信頼限界) をタプルで返す。
        """
        all_preds = []
        for model in self.estimators_:
            if hasattr(model, 'predict') and getattr(model.predict, '__code__', None) and 'metadata' in model.predict.__code__.co_varnames:
                preds = model.predict(X, metadata=metadata)
            else:
                preds = model.predict(X)
            all_preds.append(preds)
            
        all_preds = np.array(all_preds) # shape: (n_estimators, n_samples)
        mean_preds = np.mean(all_preds, axis=0)
        
        if return_interval:
            lower = np.percentile(all_preds, alpha / 2 * 100, axis=0)
            upper = np.percentile(all_preds, (1 - alpha / 2) * 100, axis=0)
            return mean_preds, lower, upper
            
        return mean_preds
