import numpy as np
import pandas as pd
from scipy import stats, linalg
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, label_binarize
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
import warnings

# Import from utils and preprocessing
from .utils import suppress_pls_warnings
from .preprocessing import preprocess_data

def fit_pls_with_preprocessing(X, y, n_components, preprocess):
    """Helper function to handle different preprocessing methods for PLS fitting"""
    if preprocess in ['center', 'autoscaling']:
        pls_scale = (preprocess == 'autoscaling')
        pls = PLSRegression(n_components=n_components, scale=pls_scale)
        with suppress_pls_warnings():
            pls.fit(X, y)
    else:
        X_proc, _, _ = preprocess_data(X, preprocess)
        y_proc, _, _ = preprocess_data(y, 'center')
        pls = PLSRegression(n_components=n_components, scale=False)
        with suppress_pls_warnings():
            pls.fit(X_proc, y_proc)
    return pls

def predict_classification(pls, X, encoding='ordinal', n_classes=2):
    """
    Convert PLS regression predictions to class labels
    
    Parameters:
    -----------
    pls : PLSRegression object
        Fitted PLS model
    X : array-like
        Features to predict
    encoding : str, default='ordinal'
        Either 'ordinal' for binary/ordinal classification or 'onehot' for multi-class classification
    n_classes : int, default=2
        Number of classes for multi-class ordinal encoding
    
    Returns:
    --------
    tuple : (class_predictions, probability_estimates)
    """
    # Get raw predictions
    y_pred_raw = pls.predict(X)
    
    if encoding == 'ordinal':
        if n_classes == 2:
            # Binary classification with threshold at 0.5
            probabilities = 1 / (1 + np.exp(-y_pred_raw))  # Apply sigmoid to get probabilities
            predictions = (probabilities > 0.5).astype(int)
            return predictions.ravel(), probabilities.ravel()
        else:
            # Multi-class ordinal encoding
            # For multi-class ordinal, we need to convert to probabilities differently
            # First, scale predictions to be in the appropriate range
            scaled_preds = y_pred_raw.ravel()
            
            # Convert to "soft" probabilities using softmax-like normalization
            # Use a technique similar to ordinal regression
            class_probs = np.zeros((len(scaled_preds), n_classes))
            
            # Map continuous predictions to class probabilities
            for i, pred in enumerate(scaled_preds):
                # Calculate distance from pred to each class center
                distances = np.array([abs(pred - cls) for cls in range(n_classes)])
                # Convert distances to probabilities (closer = higher probability)
                sim = 1 / (1 + distances)
                class_probs[i] = sim / sim.sum()  # Normalize
            
            # Choose class with highest probability
            predictions = np.round(scaled_preds).clip(0, n_classes-1).astype(int)
            
            return predictions, class_probs
    
    elif encoding == 'onehot':
        # Multi-class - predict class with highest score
        # Convert raw predictions to probabilities using softmax
        exp_preds = np.exp(y_pred_raw - np.max(y_pred_raw, axis=1, keepdims=True))  # Numerical stability
        probabilities = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
        predictions = np.argmax(y_pred_raw, axis=1)
        return predictions, probabilities
    
    else:
        raise ValueError(f"Unknown encoding: {encoding}. Use 'ordinal' or 'onehot'")

def evaluate_component_count_classification(X_sel, y_array, n_comp, kf, preprocess, encoding='ordinal', 
                                            target_names=None, metric='accuracy'):
    """
    Evaluate a single component count for cross-validation in classification
    
    Parameters:
    -----------
    X_sel : array-like
        Selected features
    y_array : array-like
        Target values (ordinal or one-hot encoded)
    n_comp : int
        Number of components to evaluate
    kf : KFold object
        Cross-validation splitter
    preprocess : str
        Preprocessing method
    encoding : str, default='ordinal'
        Target encoding ('ordinal' or 'onehot')
    target_names : list, optional
        Class names for multi-class problems
    metric : str, default='accuracy'
        Evaluation metric ('accuracy', 'f1', or 'auc')
    
    Returns:
    --------
    tuple : (n_components, metric_value, confusion_matrix)
    """
    # Determine number of classes and class indices for stratification
    if encoding == 'onehot':
        is_onehot = len(y_array.shape) > 1 and y_array.shape[1] > 1
        
        if is_onehot:
            # Convert one-hot encoding to class indices for stratification
            y_indices = np.argmax(y_array, axis=1)
            n_classes = y_array.shape[1]
        else:
            # If y is already class indices
            y_indices = y_array
            n_classes = len(np.unique(y_indices))
            
        # For prediction comparison
        y_true_indices = y_indices
    else:
        # For ordinal classification
        y_indices = y_array
        y_true_indices = y_array
        n_classes = len(np.unique(y_indices))

    # Initialize predictions
    y_pred = np.zeros_like(y_true_indices)
    
    # For AUC with multi-class, use one-vs-rest approach
    if metric == 'auc' and n_classes > 2:
        y_prob = np.zeros((len(y_true_indices), n_classes))
    elif metric == 'auc':
        y_prob = np.zeros_like(y_true_indices, dtype=float)
    
    # Manual cross-validation
    cv_splits = list(kf.split(X_sel, y_indices))
    
    for train_idx, test_idx in cv_splits:
        X_train, X_test = X_sel[train_idx], X_sel[test_idx]
        
        if encoding == 'onehot' and len(y_array.shape) > 1:
            y_train = y_array[train_idx]
        else:
            y_train = y_array[train_idx].reshape(-1, 1)
            
        # Preprocess
        X_train_proc, mean_X, scale_X = preprocess_data(X_train, preprocess)
        
        # Fit model
        pls = PLSRegression(n_components=n_comp, scale=False)
        with suppress_pls_warnings():
            pls.fit(X_train_proc, y_train)
            
        # Preprocess test data
        X_test_proc = (X_test - mean_X) / scale_X
        
        # Predict
        if encoding == 'onehot':
            predictions, probabilities = predict_classification(pls, X_test_proc, encoding='onehot')
        else:
            predictions, probabilities = predict_classification(pls, X_test_proc, encoding='ordinal', n_classes=n_classes)
            
        y_pred[test_idx] = predictions
        
        # Store probabilities for AUC calculation
        if metric == 'auc':
            if n_classes > 2:
                y_prob[test_idx] = probabilities
            else:
                y_prob[test_idx] = probabilities
    
    # Calculate metrics
    if metric == 'accuracy':
        metric_value = accuracy_score(y_true_indices, y_pred)
    elif metric == 'f1':
        if n_classes > 2:
            # Multi-class F1
            metric_value = f1_score(y_true_indices, y_pred, average='weighted')
        else:
            # Binary F1
            metric_value = f1_score(y_true_indices, y_pred)
    elif metric == 'auc':
        if n_classes > 2:
            # Multi-class AUC using one-vs-rest approach
            try:
                # Convert to binary format for multi-class ROC AUC
                y_bin = label_binarize(y_true_indices, classes=np.arange(n_classes))
                metric_value = roc_auc_score(y_bin, y_prob, multi_class='ovr')
            except ValueError:
                # Fall back to accuracy if AUC fails
                warnings.warn(f"AUC calculation failed. Falling back to accuracy for evaluation.")
                metric_value = accuracy_score(y_true_indices, y_pred)
        else:
            try:
                metric_value = roc_auc_score(y_true_indices, y_prob)
            except ValueError:
                # Fall back to accuracy if AUC fails
                warnings.warn(f"AUC calculation failed. Falling back to accuracy for evaluation.")
                metric_value = accuracy_score(y_true_indices, y_pred)
    else:
        # Default to accuracy for unknown metrics
        metric_value = accuracy_score(y_true_indices, y_pred)
        
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true_indices, y_pred)
        
    return n_comp, metric_value, conf_matrix

def evaluate_iteration_classification(iter_idx, weight_matrix, X_array, y_array, max_components, 
                                     kf, preprocess, encoding='ordinal', target_names=None, 
                                     metric='accuracy'):
    """Evaluate a single iteration (subset) for classification"""
    # Get non-zero weight features for this iteration
    selected_vars = np.where(weight_matrix[:, iter_idx] != 0)[0]
    if len(selected_vars) == 0:  # Skip if no variables selected
        return iter_idx, -np.inf, np.array([]), 1, np.array([])

    # Get selected data
    X_sel = X_array[:, selected_vars]

    # Calculate maximum possible components once
    max_possible_components = min(X_sel.shape[0]-1, X_sel.shape[1], max_components)
    comp_range = range(1, min(max_possible_components+1, len(selected_vars)+1))
    
    # Evaluate each component count
    comp_results = []

    for n_comp in comp_range:
        n_comp, metric_value, conf_matrix = evaluate_component_count_classification(
            X_sel, y_array, n_comp, kf, preprocess, encoding, target_names, metric)
        comp_results.append((n_comp, metric_value, conf_matrix))

    # Find the best component count (maximizing metric)
    comp_results.sort(key=lambda x: x[1], reverse=True)  # Sort by metric (higher is better)
    best_comp, best_metric, best_conf_matrix = comp_results[0]

    return iter_idx, best_metric, selected_vars, best_comp, best_conf_matrix

def competitive_adaptive_sampling_classification(X, y, max_components, folds=5, preprocess='center', 
                                              iterations=50, encoding='ordinal', metric='accuracy',
                                              adaptive_resampling=False, cv_shuffle_mode='none',
                                              best_metric='max', n_jobs=-1, verbose=1):
    """
    Competitive Adaptive Reweighted Sampling for variable selection in PLS Classification

    Parameters:
    -----------
    X : array-like
        The predictor matrix of shape (n_samples, n_features)
    y : array-like
        The response vector for classification, either:
        - For 'ordinal': shape (n_samples,) with class labels
        - For 'onehot': shape (n_samples, n_classes) with one-hot encoded classes
    max_components : int
        The maximum number of PLS components to extract
    folds : int, default=5
        Number of folds for cross-validation
    preprocess : str, default='center'
        Preprocessing method ('center', 'autoscaling', 'pareto', 'minmax', 'robust', 'unilength', 'none')
    iterations : int, default=50
        Number of Monte Carlo sampling runs
    encoding : str, default='ordinal'
        Target encoding type: 'ordinal' for binary/ordinal, 'onehot' for multi-class
    metric : str, default='accuracy'
        Evaluation metric ('accuracy', 'f1', or 'auc')
    adaptive_resampling : bool, default=False
        Whether to use the original version with random sampling (True) or a simplified deterministic version (False)
    cv_shuffle_mode : str, default='none'
        Cross-validation sample ordering mode
    n_jobs : int, default=-1
        Number of parallel jobs for cross-validation
    verbose : int, default=1
        Verbosity level

    Returns:
    --------
    dict : Results containing selected variables and performance metrics
    """
    start_time = time.time()

    # Convert inputs to numpy arrays if needed
    X_array = X.values if isinstance(X, pd.DataFrame) else X
    
    # Handle y based on encoding type
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = y
        
    # Get unique classes for ordinal encoding
    if encoding == 'ordinal':
        unique_classes = np.unique(y_array)
        target_names = [str(c) for c in unique_classes]
        n_classes = len(unique_classes)
    else:
        # For one-hot encoding, if y is 1D, convert it to one-hot
        if len(y_array.shape) == 1:
            unique_classes = np.unique(y_array)
            target_names = [str(c) for c in unique_classes]
            n_classes = len(unique_classes)
            
            # One-hot encode
            encoder = OneHotEncoder(sparse=False)
            y_array = encoder.fit_transform(y_array.reshape(-1, 1))
        else:
            n_classes = y_array.shape[1]
            target_names = [f"Class_{i}" for i in range(n_classes)]

    # Get dimensions and initialize
    n_samples, n_features = X_array.shape
    max_components = min(n_samples, n_features, max_components)

    # Initial sampling parameters
    initial_subset_ratio = 0.9
    decay_rate_start = 1
    decay_rate_end = 2/n_features

    # Variable selection tracking
    selected_features = np.arange(n_features)
    subset_size = int(n_samples * initial_subset_ratio)
    weight_matrix = np.zeros((n_features, iterations))
    subset_ratios = np.zeros(iterations)

    # Pre-allocate arrays for better performance
    feature_weights_buffer = np.zeros(n_features)

    # Calculate decay rate parameters for exponential function
    decay_factor = np.log(decay_rate_start/decay_rate_end) / (iterations-1)
    decay_coefficient = decay_rate_start * np.exp(decay_factor)

    # Set up the cross-validation based on the cv_shuffle_mode
    # Use StratifiedKFold to maintain class distribution
    if cv_shuffle_mode == 'none':
        if encoding == 'onehot' and len(y_array.shape) > 1:
            # For one-hot encoding, get class indices for stratification
            y_indices = np.argmax(y_array, axis=1)
            kf = StratifiedKFold(n_splits=folds, shuffle=False)
        else:
            kf = StratifiedKFold(n_splits=folds, shuffle=False)
    elif cv_shuffle_mode == 'fixed_seed':
        if encoding == 'onehot' and len(y_array.shape) > 1:
            y_indices = np.argmax(y_array, axis=1)
            kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        else:
            kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    elif cv_shuffle_mode == 'random_seed':
        if encoding == 'onehot' and len(y_array.shape) > 1:
            y_indices = np.argmax(y_array, axis=1)
            kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=None)
        else:
            kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=None)
    else:
        raise ValueError(f"Invalid cv_shuffle_mode: {cv_shuffle_mode}")

    # Print initial info
    if verbose >= 1:
        print(f"Starting variable selection with {n_features} initial variables...")
        print(f"Classification problem with {n_classes} classes using {encoding} encoding")
        if verbose == 1:
            print(f"Running {iterations} iterations - progress will be shown at completion")

    # Main sampling loop
    for iter_idx in range(iterations):
        # Either use Monte Carlo sampling or full dataset
        if adaptive_resampling:
            rand_indices = np.random.permutation(n_samples)
            X_subset = X_array[rand_indices[:subset_size]]
            
            if encoding == 'onehot' and len(y_array.shape) > 1:
                y_subset = y_array[rand_indices[:subset_size]]
            else:
                y_subset = y_array[rand_indices[:subset_size]]
        else:
            X_subset = X_array
            y_subset = y_array

        # Skip if no variables selected
        if len(selected_features) == 0:
            if verbose >= 2:
                print(f'Iteration {iter_idx+1}/{iterations} - No variables selected, skipping')
            continue

        # Build PLS model on current feature subset
        X_sel = X_subset[:, selected_features]

        # Calculate maximum possible components for this subset
        max_possible_components = min(X_sel.shape[0]-1, X_sel.shape[1], max_components)

        # Fit PLS model
        pls = fit_pls_with_preprocessing(X_sel, y_subset, max_possible_components, preprocess)

        # Extract coefficients and update weights - use pre-allocated buffer for efficiency
        feature_weights_buffer.fill(0)  # Reset buffer

        # Get coefficients - handle both binary and multi-class
        if encoding == 'onehot' and len(y_array.shape) > 1:
            # For multi-class, use sum of absolute coefficients across all classes
            coefs = np.abs(pls.coef_).sum(axis=0)
        else:
            coefs = pls.coef_
            
        # Note: Classification CARS uses absolute coefficient values (magnitude only)
        # This differs from standard CARS which uses raw coefficients (with direction)
        feature_weights_buffer[selected_features] = np.abs(coefs.flatten())
        weight_matrix[:, iter_idx] = feature_weights_buffer

        # Calculate exponentially decreasing sampling ratio
        current_ratio = decay_coefficient * np.exp(-decay_factor * (iter_idx+1))
        subset_ratios[iter_idx] = current_ratio
        keep_count = max(1, int(n_features * current_ratio))  # Ensure at least 1 variable

        # Zero out weights of variables that won't be included in the top keep_count
        if keep_count < n_features:
            top_indices = np.argpartition(-feature_weights_buffer, keep_count)[:keep_count]
            mask = np.zeros(n_features, dtype=bool)
            mask[top_indices] = True
            feature_weights_buffer *= mask

        # Feature selection for next iteration
        if adaptive_resampling:
            # Use weighted random sampling based on feature weights
            nonzero_indices = np.nonzero(feature_weights_buffer)[0]
            
            if len(nonzero_indices) <= keep_count:
                selected_features = nonzero_indices
            else:
                weights = feature_weights_buffer[nonzero_indices]
                total_weight = np.sum(weights)
                if total_weight > 0:
                    probs = weights / total_weight
                    sampled_indices = np.random.choice(
                        len(nonzero_indices), 
                        size=keep_count,
                        replace=False,
                        p=probs
                    )
                    selected_features = nonzero_indices[sampled_indices]
                else:
                    selected_features = nonzero_indices[:keep_count]
        else:
            # Deterministic selection
            selected_features = np.nonzero(feature_weights_buffer)[0]

        # Print progress based on verbosity level
        if verbose >= 2:
            print(f'Iteration {iter_idx+1}/{iterations} complete - {len(selected_features)} variables selected')
        elif verbose == 1 and (iter_idx+1) % max(1, iterations//10) == 0:
            progress = (iter_idx+1) / iterations * 100
            print(f"Progress: {progress:.0f}% - Currently {len(selected_features)} variables selected")

    if verbose >= 1:
        print(f"Starting parallel cross-validation with {n_jobs} jobs...")

    # Parallelize the cross-validation evaluations
    results_parallel = Parallel(n_jobs=n_jobs, verbose=max(0, verbose-1))(
        delayed(evaluate_iteration_classification)(
            i, weight_matrix, X_array, y_array, max_components, kf, preprocess, 
            encoding, target_names, metric
        ) for i in range(iterations)
    )

    # Unpack the results
    metric_values = np.zeros(iterations)
    optimal_components = np.zeros(iterations, dtype=int)
    selected_vars_list = [None] * iterations
    confusion_matrices = [None] * iterations

    for idx, metric_val, sel_vars, components, conf_matrix in results_parallel:
        metric_values[idx] = metric_val
        selected_vars_list[idx] = sel_vars
        optimal_components[idx] = components
        confusion_matrices[idx] = conf_matrix

    # Determine if metric should be maximized or minimized
    metrics_to_maximize = ['accuracy', 'f1', 'auc', 'precision', 'recall']
    
    if best_metric == 'max' or metric.lower() in metrics_to_maximize:
        # For these metrics, higher is better
        best_subset_idx = np.argmax(metric_values)
    else:
        # For other metrics like error rates, lower is better
        best_subset_idx = np.argmin(metric_values)
        
    best_metric_value = metric_values[best_subset_idx]
    best_conf_matrix = confusion_matrices[best_subset_idx]
    selected_vars = selected_vars_list[best_subset_idx]

    # Compile results
    elapsed_time = time.time() - start_time

    if verbose >= 1:
        print(f"Optimization completed in {elapsed_time:.2f} seconds")
        print(f"Best subset found at iteration {best_subset_idx+1} with {metric} value: {best_metric_value:.4f}")
        print(f"Selected {len(selected_vars)} variables")

    results = {
        'weight_matrix': weight_matrix,
        'computation_time': elapsed_time,
        'metric_values': metric_values,
        'best_metric_value': best_metric_value,
        'best_iteration': best_subset_idx,
        'optimal_components': optimal_components[best_subset_idx],
        'subset_ratios': subset_ratios,
        'selected_variables': selected_vars,
        'confusion_matrix': best_conf_matrix,
        'encoding': encoding,
        'target_names': target_names,
        'metric': metric
    }

    return results

def plot_classification_results(results):
    """
    Plot the results of CARS classification analysis

    Parameters:
    -----------
    results : dict
        The results dictionary from competitive_adaptive_sampling_classification
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
    plt.show()

# Example data generator functions
from sklearn.datasets import make_classification

def generate_binary_classification_data(n_samples=2000, n_features=200, n_informative=20, random_state=42):
    """Generate synthetic binary classification data"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=10,
        n_repeated=0,
        n_classes=2,
        random_state=random_state,
        class_sep=2.0
    )
    
    # Create a pandas DataFrame
    feature_names = [f'Feature_{i+1}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='Target')
    
    return X_df, y_series

def generate_multiclass_classification_data(n_samples=3000, n_features=200, n_informative=15, 
                                          n_classes=4, random_state=42):
    """Generate synthetic multi-class classification data"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=10,
        n_repeated=0,
        n_classes=n_classes,
        random_state=random_state,
        class_sep=1.5
    )
    
    # Create a pandas DataFrame
    feature_names = [f'Feature_{i+1}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='Target')
    
    # One-hot encode the target
    y_onehot = pd.get_dummies(y_series, prefix='Class')
    
    return X_df, y_series, y_onehot
