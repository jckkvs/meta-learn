import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.base import clone
from domainml.core.metadata import FeatureMetadata
from domainml.analysis.metrics import satisfaction_score
from domainml.core.logger import logger

def constrained_cv(estimator, X: np.ndarray, y: np.ndarray, metadata: FeatureMetadata, cv=5, scoring=None) -> dict:
    """
    制約付きの交差検証を実行する。
    通常の CV スコアに加え、各Foldにおける制約充足度 (satisfaction_score) を算出し、
    制約がどの程度遵守されているかを評価する。
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    results = {
        'test_score': [],
        'satisfaction_score': []
    }
    
    for train_idx, val_idx in kf.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        model = clone(estimator)
        # try to fit with metadata if supported
        try:
            model.fit(X_train, y_train, metadata=metadata)
        except TypeError:
            model.fit(X_train, y_train)
            
        # 計算
        if hasattr(model, 'predict') and getattr(model.predict, '__code__', None) and 'metadata' in model.predict.__code__.co_varnames:
            preds = model.predict(X_val, metadata=metadata)
        else:
            preds = model.predict(X_val)
            
        score = r2_score(y_val, preds) if scoring is None else scoring(model, X_val, y_val)
        results['test_score'].append(score)
        
        # 充足度計算
        satisfaction = satisfaction_score(model, X_val, metadata, n_samples=50)
        results['satisfaction_score'].append(satisfaction)
        
    logger.debug(f"Constrained CV finished. Mean Satisfaction: {np.mean(results['satisfaction_score']):.3f}, Mean Score: {np.mean(results['test_score']):.3f}")
    return results
