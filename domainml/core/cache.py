import hashlib
import numpy as np
from functools import wraps
from domainml.core.logger import logger

class LazyConstraintEvaluator:
    """
    高コストな制約評価や行列演算（カーネル作成など）を、入力データとメタデータのハッシュに
    基づいてキャッシュ・遅延評価する仕組み。
    """
    _cache = {}

    @staticmethod
    def _compute_hash(*args, **kwargs) -> str:
        """
        numpy配列やFeatureMetadataを文字列化してハッシュを計算する
        """
        hash_obj = hashlib.md5()
        for arg in args:
            if isinstance(arg, np.ndarray):
                # 簡易的に最初の数要素とshapeなどでハッシュ化
                h = f"{arg.shape}_{np.sum(arg.flat[:10])}_{np.sum(arg.flat[-10:])}"
                hash_obj.update(h.encode('utf-8'))
            elif hasattr(arg, 'feature_names'):
                # FeatureMetadata
                h = f"{arg.feature_names}_{arg.monotonicities}_{arg.constraint_types}"
                hash_obj.update(h.encode('utf-8'))
            else:
                hash_obj.update(str(arg).encode('utf-8'))
                
        for k, v in sorted(kwargs.items()):
            hash_obj.update(f"{k}:{v}".encode('utf-8'))
            
        return hash_obj.hexdigest()

    @staticmethod
    def clear_cache():
        LazyConstraintEvaluator._cache.clear()
        
    @staticmethod
    def cache_evaluation(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = LazyConstraintEvaluator._compute_hash(func.__name__, *args, **kwargs)
            if key in LazyConstraintEvaluator._cache:
                logger.debug(f"Cache hit for {func.__name__} (key: {key[:8]}...)")
                return LazyConstraintEvaluator._cache[key]
                
            logger.debug(f"Cache miss for {func.__name__}. Evaluating...")
            result = func(*args, **kwargs)
            LazyConstraintEvaluator._cache[key] = result
            return result
            
        return wrapper
