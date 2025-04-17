"""
Data preprocessing utilities for SpectralCARSLib.

This module provides functions for different preprocessing methods commonly used
in chemometrics and spectroscopy applications.
"""

import numpy as np

def preprocess_data(X, method, mean=None, scale=None):
    """
    Preprocess data using various methods - vectorized implementation
    
    Parameters:
    -----------
    X : array-like
        Data to preprocess
    method : str
        Preprocessing method: 'center', 'autoscaling', 'pareto', 'minmax', 'robust', 'unilength', or 'none'
    mean : array-like, optional
        Precalculated mean values
    scale : array-like, optional
        Precalculated scale values
        
    Returns:
    --------
    tuple : (preprocessed_data, mean, scale)
    """
    # Ensure X is at least 2D
    X = np.atleast_2d(X)
    if X.shape[0] == 1:
        X = X.T
    
    # Calculate stats if not provided
    if mean is None and scale is None:
        if method == 'autoscaling':
            # Use sklearn's StandardScaler approach
            mean, scale = np.mean(X, axis=0), np.std(X, axis=0, ddof=1)
        elif method == 'center':
            mean, scale = np.mean(X, axis=0), np.ones(X.shape[1])
        elif method == 'unilength':
            mean = np.mean(X, axis=0)
            # Vectorized calculation of norm for each column
            X_centered = X - mean
            scale = np.sqrt(np.sum(X_centered**2, axis=0))
        elif method == 'minmax':
            # Use sklearn's MinMaxScaler approach
            mean, scale = np.min(X, axis=0), np.ptp(X, axis=0)  # ptp = peak to peak (max-min)
        elif method == 'pareto':
            mean, scale = np.mean(X, axis=0), np.sqrt(np.std(X, axis=0, ddof=1))
        elif method == 'robust':
            # Robust scaler using median and interquartile range
            mean = np.median(X, axis=0)
            q1 = np.percentile(X, 25, axis=0)
            q3 = np.percentile(X, 75, axis=0)
            scale = q3 - q1  # Interquartile range (IQR)
        elif method == 'none':
            mean, scale = np.zeros(X.shape[1]), np.ones(X.shape[1])
        else:
            raise ValueError(f'Unknown preprocessing method: {method}')
    
    # Handle zero variance before division
    scale = np.where(scale == 0, 1.0, scale)
    
    # Apply preprocessing
    X_processed = (X - mean) / scale
    
    # Return squeezed output (1D if input was 1D)
    return X_processed.squeeze(), mean, scale
