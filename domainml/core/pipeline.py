import inspect
from typing import Optional
from sklearn.pipeline import Pipeline
from .metadata import FeatureMetadata
from .logger import logger

def _accepts_metadata(func) -> bool:
    """関数の引数にmetadataが含まれているか判定する"""
    try:
        sig = inspect.signature(func)
        return 'metadata' in sig.parameters
    except ValueError:
        return False

class MetaPipeline(Pipeline):
    """
    scikit-learn の Pipeline を拡張し、FeatureMetadata を伝播させるためのクラス。
    """
    def fit(self, X, y=None, metadata: Optional[FeatureMetadata] = None, **fit_params):
        fit_params_steps = {name: {} for name, step in self.steps if step is not None}
        for pname, pval in fit_params.items():
            if '__' in pname:
                step, param = pname.split('__', 1)
                fit_params_steps[step][param] = pval

        Xt = X
        current_metadata = metadata

        logger.debug(f"MetaPipeline.fit started with {len(self.steps)} steps. Initial metadata: {current_metadata}")

        for step_idx, name, transformer in self._iter(with_final=False, filter_passthrough=False):
            if transformer is None or transformer == 'passthrough':
                continue

            logger.debug(f"Processing step {step_idx} ({name}): {transformer.__class__.__name__}")
            fit_kwargs = fit_params_steps[name]
            
            # fit_transformが明示的にmetadataを受け取る場合
            if hasattr(transformer, "fit_transform") and _accepts_metadata(transformer.fit_transform):
                Xt = transformer.fit_transform(Xt, y, metadata=current_metadata, **fit_kwargs)
            else:
                # 継承などで隠れている場合は手動でfitとtransformを分ける
                if _accepts_metadata(transformer.fit):
                    transformer.fit(Xt, y, metadata=current_metadata, **fit_kwargs)
                else:
                    transformer.fit(Xt, y, **fit_kwargs)
                
                if _accepts_metadata(transformer.transform):
                    Xt = transformer.transform(Xt, metadata=current_metadata)
                else:
                    Xt = transformer.transform(Xt)

            if current_metadata is not None and hasattr(transformer, "get_metadata_out"):
                old_meta_repr = repr(current_metadata)
                current_metadata = transformer.get_metadata_out(current_metadata)
                logger.debug(f"Metadata updated by {name}: {old_meta_repr} -> {repr(current_metadata)}")

        if self._final_estimator != 'passthrough':
            logger.debug(f"Fitting final estimator {self.steps[-1][0]}: {self._final_estimator.__class__.__name__} with metadata: {current_metadata}")
            fit_kwargs = fit_params_steps[self.steps[-1][0]]
            if _accepts_metadata(self._final_estimator.fit):
                self._final_estimator.fit(Xt, y, metadata=current_metadata, **fit_kwargs)
            else:
                self._final_estimator.fit(Xt, y, **fit_kwargs)

        return self

    def predict(self, X, metadata: Optional[FeatureMetadata] = None, **predict_params):
        Xt = X
        current_metadata = metadata
        
        for _, name, transform in self._iter(with_final=False, filter_passthrough=False):
            if transform is None or transform == 'passthrough':
                 continue
            
            if _accepts_metadata(transform.transform):
                Xt = transform.transform(Xt, metadata=current_metadata)
            else:
                Xt = transform.transform(Xt)
                 
            if current_metadata is not None and hasattr(transform, "get_metadata_out"):
                 current_metadata = transform.get_metadata_out(current_metadata)
                 
        if self._final_estimator != 'passthrough':
            if _accepts_metadata(self._final_estimator.predict):
                return self._final_estimator.predict(Xt, metadata=current_metadata, **predict_params)
            else:
                return self._final_estimator.predict(Xt, **predict_params)
        return Xt
