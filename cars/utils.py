"""
Utility functions for SpectralCARSLib.

This module provides utility functions to support the main CARS algorithm.
"""

import contextlib
import warnings

@contextlib.contextmanager
def suppress_pls_warnings():
    """
    Context manager to suppress expected RuntimeWarnings from sklearn PLS
    
    This is particularly useful for suppressing common warnings like
    "invalid value encountered in divide" that can appear during PLS fitting
    with near-singular matrices.
    
    Example:
    --------
    >>> with suppress_pls_warnings():
    >>>     pls.fit(X, y)
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, 
                               message="invalid value encountered in divide")
        warnings.filterwarnings("ignore", category=UserWarning)
        yield
