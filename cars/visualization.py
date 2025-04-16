"""
Visualization functions for SpectralCARSLib results.

This module provides functions for visualizing the results of the
Competitive Adaptive Reweighted Sampling (CARS) algorithm variants.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_sampling_results(results):
    """
    Plot the results of CARS analysis
    
    Parameters:
    -----------
    results : dict
        The results dictionary from competitive_adaptive_sampling or competitive_adaptive_reweighted_sampling
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure for further customization
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
    
    # Use vectorized approach for creating points for vertical line
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
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure for further customization
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

def plot_classification_results(results):
    """
    Plot the results of CARS classification analysis

    Parameters:
    -----------
    results : dict
        The results dictionary from competitive_adaptive_sampling_classification
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure for further customization
    """
    weight_matrix = results['weight_matrix']
    metric_values = results['metric_values']
    best_iter = results['best_iteration']
    iterations = len(metric_values)
    metric_name = results['metric'].upper()
    conf_matrix = results['confusion_matrix']
    target_names = results['target_names']

    # Calculate number of variables in each iteration
    var_counts = np.count_nonzero(weight_matrix != 0, axis=0)

    # Create figure
    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.5])

    # Plot number of selected variables per iteration
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(var_counts, linewidth=2, color='navy')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Number of variables', fontsize=12)
    ax1.set_title('Variables Selected per Iteration', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Plot metric values
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(np.arange(iterations), metric_values, linewidth=2, color='darkgreen')
    ax2.axvline(x=best_iter, color='red', linestyle='--', alpha=0.7,
                label=f'Best iteration: {best_iter+1}')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel(f'{metric_name}', fontsize=12)
    ax2.set_title(f'Classification {metric_name}', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot regression coefficient paths
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(weight_matrix.T, linewidth=1, alpha=0.6)
    ylims = ax3.get_ylim()

    # Create points for vertical line at best iteration
    y_points = np.linspace(ylims[0], ylims[1], 20)
    ax3.plot(np.full(20, best_iter), y_points, 'r*', linewidth=1, alpha=0.8)

    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Regression coefficients', fontsize=12)
    ax3.set_title('Coefficient Evolution', fontsize=14)
    ax3.grid(True, alpha=0.3)

    # Plot confusion matrix
    ax4 = fig.add_subplot(gs[2, :])
    im = ax4.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax4.set_title('Confusion Matrix', fontsize=14)
    fig.colorbar(im, ax=ax4)
    
    # Add labels to confusion matrix
    tick_marks = np.arange(len(target_names))
    ax4.set_xticks(tick_marks)
    ax4.set_xticklabels(target_names)
    ax4.set_yticks(tick_marks)
    ax4.set_yticklabels(target_names)
    
    # Add text annotations
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax4.text(j, i, format(conf_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")
                     
    ax4.set_ylabel('True label', fontsize=12)
    ax4.set_xlabel('Predicted label', fontsize=12)

    plt.tight_layout()
    return fig
