"""
NumPy compatibility shim for loading models pickled with older NumPy versions.

This module patches NumPy's random number generator pickle handling to allow
loading models that were pickled with NumPy 1.24.4 (from SageMaker) in environments
with NumPy 1.26.4+ (which has breaking changes in the random API).
"""
import numpy as np

def patch_numpy_random_pickle():
    """Patch NumPy's random pickle module to handle old BitGenerator format."""
    try:
        import numpy.random._pickle as np_random_pickle
        from numpy.random import _mt19937
        
        # NumPy 1.26+ changed how BitGenerators are unpickled
        # The model was pickled with NumPy 1.24.4, but we're loading with 1.26.4
        # We need to register the old class name in the BitGenerators registry
        
        # Store original if it exists
        original_ctor = getattr(np_random_pickle, '__bit_generator_ctor', None)
        
        # Try to add MT19937 to the registry if it's not there
        if hasattr(np_random_pickle, '_BitGenerators'):
            if 'MT19937' not in np_random_pickle._BitGenerators and hasattr(_mt19937, 'MT19937'):
                np_random_pickle._BitGenerators['MT19937'] = _mt19937.MT19937
        
        # Also patch the constructor function
        if original_ctor:
            def patched_ctor(bit_generator_name):
                """Patched constructor that handles both old and new BitGenerator names."""
                try:
                    return original_ctor(bit_generator_name)
                except ValueError:
                    # Try to handle old format
                    if 'MT19937' in str(bit_generator_name) and hasattr(_mt19937, 'MT19937'):
                        return _mt19937.MT19937
                    raise
            
            np_random_pickle.__bit_generator_ctor = patched_ctor
        
        print("[INFO] Patched NumPy random pickle for compatibility with NumPy 1.24.4 models")
    except (ImportError, AttributeError) as e:
        # If patching fails, we'll try loading anyway - the error will be clearer
        print(f"[WARNING] Could not patch NumPy random pickle: {e}")

# Auto-patch on import
try:
    if np.__version__ >= "1.26.0":
        patch_numpy_random_pickle()
except Exception as e:
    print(f"[WARNING] NumPy compatibility patch failed: {e}")

