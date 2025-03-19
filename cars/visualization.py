"""
Visualization functions for CARS algorithm results.

This module provides functions for visualizing the results of the
Competitive Adaptive Reweighted Sampling (CARS) algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_sampling_results(results):
    """
    Plot the results of CARS analysis
    
    Parameters:
    -----------
    results : dict
        The results dictionary from competitive_adaptive_sampling
    """
    weight_matrix = results['weight_matrix']
    cv_errors = results['cross_validation_errors']
    best_iter = results['best_iteration']
    iterations = len(cv_errors)
    
    # Pre-calculate variables - use count_nonzero for boolean operations
    var_counts = np.count_nonzero(weight_matrix != 0, axis=0)
    
    # Create figure once
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot number of selected variables per iteration
    ax1.plot(var_counts, linewidth=2, color='navy')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Number of variables', fontsize=12)
    ax1.set_title('Variables Selected per Iteration', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot cross-validation errors - avoid unnecessary range creation
    ax2.plot(np.arange(iterations), cv_errors, linewidth=2, color='darkgreen')
    ax2.axvline(x=best_iter, color='red', linestyle='--', alpha=0.7,
                label=f'Best iteration: {best_iter+1}')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('RMSECV', fontsize=12)
    ax2.set_title('Cross-Validation Error', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot regression coefficient paths
    ax3.plot(weight_matrix.T, linewidth=1, alpha=0.6)
    ylims = ax3.get_ylim()
    
    # Use vectorized approach for creating points
    y_points = np.linspace(ylims[0], ylims[1], 20)
    ax3.plot(np.full(20, best_iter), y_points, 'r*', linewidth=1, alpha=0.8)
    
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Regression coefficients', fontsize=12)
    ax3.set_title('Coefficient Evolution', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_selected_variables(X, wavelengths, selected_vars, title="Selected Variables"):
    """
    Plot the selected variables, highlighting them in the average spectrum
    
    Parameters:
    -----------
    X : array-like
        The predictor matrix of shape (n_samples, n_features)
    wavelengths : array-like
        The wavelengths or variable indices corresponding to each feature
    selected_vars : array-like
        Indices of selected variables from CARS
    title : str, default="Selected Variables"
        Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate mean spectrum
    mean_spectrum = np.mean(X, axis=0)
    
    # Plot full spectrum
    ax.plot(wavelengths, mean_spectrum, 'b-', alpha=0.5, label='Full spectrum')
    
    # Create a mask for selected variables
    mask = np.zeros(len(wavelengths), dtype=bool)
    mask[selected_vars] = True
    
    # Highlight selected variables
    ax.plot(wavelengths[mask], mean_spectrum[mask], 'ro', label='Selected variables')
    
    ax.set_xlabel('Variable index/wavelength', fontsize=12)
    ax.set_ylabel('Intensity/value', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig