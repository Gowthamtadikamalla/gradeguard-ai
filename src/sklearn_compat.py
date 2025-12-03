"""
Compatibility shim for loading scikit-learn 1.0.2 models with scikit-learn 1.1.3+.
Handles:
1. The _gb_losses module that was moved in scikit-learn 1.5.x
2. OneHotEncoder _infrequent_enabled attribute missing in 1.0.2 models
"""
import sys
import importlib
import joblib

def patch_sklearn_compat():
    """Patch scikit-learn to handle models from version 1.0.2."""
    try:
        import sklearn
        import sklearn.ensemble
        
        # Check if _gb_losses exists
        if not hasattr(sklearn.ensemble, '_gb_losses'):
            # In scikit-learn 1.1.3, _gb_losses might be in a different location
            # Try to import it from where it might be
            try:
                # For scikit-learn 1.1.x, try importing from _hist_gradient_boosting
                from sklearn.ensemble._hist_gradient_boosting import _loss
                # Create a compatibility module
                class _gb_losses:
                    """Compatibility shim for _gb_losses module."""
                    pass
                sklearn.ensemble._gb_losses = _gb_losses()
                print("[INFO] Patched scikit-learn _gb_losses for compatibility with 1.0.2 models")
            except ImportError:
                # Try alternative approach - create a minimal mock
                try:
                    import types
                    _gb_losses_module = types.ModuleType('_gb_losses')
                    sys.modules['sklearn.ensemble._gb_losses'] = _gb_losses_module
                    sklearn.ensemble._gb_losses = _gb_losses_module
                    print("[INFO] Created mock _gb_losses module for compatibility")
                except Exception as e2:
                    print(f"[WARNING] Could not create _gb_losses compatibility: {e2}")
    except Exception as e:
        print(f"[WARNING] Could not patch scikit-learn: {e}")

def patch_gradient_boosting(obj):
    """Patch GradientBoostingClassifier to add _loss attribute."""
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        if isinstance(obj, GradientBoostingClassifier):
            # In scikit-learn 1.1.3, _loss is required but may be missing in 1.0.2 models
            if not hasattr(obj, '_loss'):
                # Try to get loss from loss parameter
                loss_name = getattr(obj, 'loss', 'log_loss')
                # Map old loss names to new ones
                if loss_name == 'deviance':
                    loss_name = 'log_loss'
                
                # Get n_classes
                n_classes = getattr(obj, 'n_classes_', 2)
                
                # Try to create it from loss name
                try:
                    from sklearn.ensemble._gb_losses import (
                        BinomialDeviance, MultinomialDeviance, ExponentialLoss
                    )
                    n_classes = getattr(obj, 'n_classes_', 2)
                    
                    if loss_name == 'log_loss' or loss_name == 'deviance':
                        if n_classes == 2:
                            obj._loss = BinomialDeviance(n_classes)
                        else:
                            obj._loss = MultinomialDeviance(n_classes)
                    elif loss_name == 'exponential':
                        obj._loss = ExponentialLoss(n_classes)
                    else:
                        # Default to BinomialDeviance for binary, MultinomialDeviance for multiclass
                        if n_classes == 2:
                            obj._loss = BinomialDeviance(n_classes)
                        else:
                            obj._loss = MultinomialDeviance(n_classes)
                    print(f"[INFO] Patched GradientBoostingClassifier: added _loss ({loss_name}, n_classes={n_classes})")
                except Exception as e:
                    # If we can't create it, try to use loss_ property (deprecated but might work)
                    try:
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            if hasattr(obj, 'loss_'):
                                obj._loss = obj.loss_
                                print(f"[INFO] Patched GradientBoostingClassifier: used deprecated loss_ property")
                            else:
                                raise AttributeError("loss_ not found")
                    except Exception as e2:
                        print(f"[WARNING] Could not patch GradientBoostingClassifier _loss: {e2}")
                        import traceback
                        traceback.print_exc()
    except Exception as e:
        # Silently fail for non-GradientBoostingClassifier objects
        pass

def patch_onehotencoder_after_load(obj):
    """Recursively patch OneHotEncoder instances to add missing attributes for scikit-learn 1.1.3+ compatibility."""
    try:
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        import numpy as np
        import types
        
        # First, patch GradientBoostingClassifier if present
        patch_gradient_boosting(obj)
        
        # If it's a OneHotEncoder, add missing attributes and patch transform method
        if isinstance(obj, OneHotEncoder):
            # Add _infrequent_enabled (introduced in scikit-learn 1.1.0)
            if not hasattr(obj, '_infrequent_enabled'):
                obj._infrequent_enabled = False
            
            # Add _infrequent_indices (required by _compute_n_features_outs in 1.1.3)
            if not hasattr(obj, '_infrequent_indices'):
                # For 1.0.2 models, infrequent categories weren't supported
                # Set to None for each feature (no infrequent categories)
                if hasattr(obj, 'categories_') and obj.categories_ is not None:
                    obj._infrequent_indices = [None] * len(obj.categories_)
                else:
                    obj._infrequent_indices = None
            
            # Add _n_features_outs (renamed/changed in scikit-learn 1.1.0+)
            # In 1.0.2, this was stored as n_features_ or calculated from categories_
            # _n_features_outs is an array where each element is the number of output features
            # for each input categorical feature
            # CRITICAL: Must be a numpy array that can be used in np.cumsum([0] + self._n_features_outs)
            if not hasattr(obj, '_n_features_outs'):
                if hasattr(obj, 'categories_') and obj.categories_ is not None:
                    # Calculate from categories_ (scikit-learn 1.0.2 format)
                    # Each categorical feature produces len(categories) output features
                    n_features_outs = [len(cats) for cats in obj.categories_]
                    # Store as numpy array - scikit-learn 1.1.3 expects this
                    obj._n_features_outs = np.array(n_features_outs, dtype=np.int64)
                elif hasattr(obj, 'n_features_'):
                    # Fallback: if n_features_ exists, use it as total
                    obj._n_features_outs = np.array([obj.n_features_], dtype=np.int64)
                else:
                    # Default: assume single feature with 1 output
                    obj._n_features_outs = np.array([1], dtype=np.int64)
            
            # Replace the transform method with a compatible implementation
            # The issue is that scikit-learn 1.1.3's transform has internal changes
            # that are incompatible with 1.0.2 models. We'll implement a compatible version.
            if not hasattr(obj, '_transform_patched'):
                from scipy import sparse
                from sklearn.utils.validation import check_is_fitted
                
                def compatible_transform(self, X):
                    """Compatible transform implementation for scikit-learn 1.0.2 models."""
                    check_is_fitted(self)
                    
                    # Use the _transform method to get encoded integers
                    X_int, X_mask = self._transform(
                        X,
                        handle_unknown=self.handle_unknown,
                        force_all_finite="allow-nan",
                        warn_on_unknown=False
                    )
                    
                    # Map infrequent categories (should be no-op for 1.0.2 models)
                    self._map_infrequent_categories(X_int, X_mask)
                    
                    n_samples, n_features = X_int.shape
                    
                    # Handle drop_idx_ if present
                    if self.drop_idx_ is not None:
                        to_drop = self.drop_idx_.copy()
                        keep_cells = X_int != to_drop
                        for i, cats in enumerate(self.categories_):
                            if to_drop[i] is None:
                                to_drop[i] = len(cats)
                        to_drop = to_drop.reshape(1, -1)
                        X_int[X_int > to_drop] -= 1
                        X_mask &= keep_cells
                    
                    # Calculate feature_indices correctly
                    # Ensure _n_features_outs is a list/array that can be concatenated
                    n_features_outs = self._n_features_outs
                    if isinstance(n_features_outs, np.ndarray):
                        n_features_outs = n_features_outs.tolist()
                    feature_indices = np.cumsum([0] + n_features_outs)
                    
                    # Create sparse matrix
                    mask = X_mask.ravel()
                    indices = (X_int + feature_indices[:-1]).ravel()[mask]
                    
                    indptr = np.empty(n_samples + 1, dtype=int)
                    indptr[0] = 0
                    np.sum(X_mask, axis=1, out=indptr[1:], dtype=indptr.dtype)
                    np.cumsum(indptr[1:], out=indptr[1:])
                    data = np.ones(indptr[-1])
                    
                    out = sparse.csr_matrix(
                        (data, indices, indptr),
                        shape=(n_samples, feature_indices[-1]),
                        dtype=self.dtype,
                    )
                    
                    if not self.sparse:
                        return out.toarray()
                    else:
                        return out
                
                # Bind the compatible method to the instance
                obj.transform = types.MethodType(compatible_transform, obj)
                obj._transform_patched = True
            
            # Only print once per OneHotEncoder
            if not hasattr(obj, '_compat_patched'):
                obj._compat_patched = True
                print("[INFO] Patched OneHotEncoder: added _infrequent_enabled, _n_features_outs, and transform method")
        
        # Patch GradientBoostingClassifier if present
        patch_gradient_boosting(obj)
        
        # If it's a Pipeline, patch all steps
        if isinstance(obj, Pipeline):
            for name, step in obj.named_steps.items():
                # Patch GradientBoostingClassifier first, then recursively patch
                patch_gradient_boosting(step)
                patch_onehotencoder_after_load(step)
        
        # If it's a ColumnTransformer, patch all transformers
        elif isinstance(obj, ColumnTransformer):
            for name, transformer, cols in obj.transformers_:
                if transformer is not None:
                    patch_onehotencoder_after_load(transformer)
        
        # If it's a dict, patch all values
        elif isinstance(obj, dict):
            for value in obj.values():
                patch_onehotencoder_after_load(value)
        
        # If it's a list or tuple, patch all items
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                patch_onehotencoder_after_load(item)
    except Exception as e:
        # Silently fail - we don't want to break loading if patching fails
        pass

# Store original joblib.load
_original_joblib_load = joblib.load

def patched_joblib_load(filename, mmap_mode=None):
    """Patched joblib.load that fixes OneHotEncoder compatibility after loading."""
    obj = _original_joblib_load(filename, mmap_mode=mmap_mode)
    patch_onehotencoder_after_load(obj)
    return obj

# Replace joblib.load with our patched version
joblib.load = patched_joblib_load

# Auto-patch on import
patch_sklearn_compat()

