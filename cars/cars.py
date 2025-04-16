import numpy as np
import pandas as pd
from scipy import stats, linalg
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed

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

def evaluate_component_count(X_sel, y_array, n_comp, kf, preprocess):
    """Evaluate a single component count for cross-validation"""
    # Use scikit-learn's built-in scaling for 'center' and 'autoscaling'
    if preprocess in ['center', 'autoscaling']:
        # Set scale=True for 'autoscaling', scale=False for 'center'
        pls_scale = (preprocess == 'autoscaling')
        pls = PLSRegression(n_components=n_comp, scale=pls_scale)
        with suppress_pls_warnings():
            y_pred = cross_val_predict(pls, X_sel, y_array, cv=kf)
    else:
        # For other preprocessing methods, perform manual CV
        y_pred = np.zeros_like(y_array)
        cv_splits = list(kf.split(X_sel))

        for train_idx, test_idx in cv_splits:
            X_train, X_test = X_sel[train_idx], X_sel[test_idx]
            y_train = y_array[train_idx]

            # Preprocess
            X_train_proc, mean_X, scale_X = preprocess_data(X_train, preprocess)
            y_train_proc, mean_y, scale_y = preprocess_data(y_train, 'center')

            # Fit model
            pls = PLSRegression(n_components=n_comp, scale=False)
            with suppress_pls_warnings():
                pls.fit(X_train_proc, y_train_proc)

            # Preprocess test data with training parameters
            X_test_proc = (X_test - mean_X) / scale_X

            # Predict and back-transform
            pred = pls.predict(X_test_proc).flatten()
            y_pred[test_idx] = pred * scale_y + mean_y

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_array, y_pred))
    r2 = r2_score(y_array, y_pred)

    return n_comp, rmse, r2

def evaluate_iteration(iter_idx, weight_matrix, X_array, y_array, max_components, kf, preprocess):
    """Evaluate a single iteration (subset)"""
    # Get non-zero weight features for this iteration
    selected_vars = np.where(weight_matrix[:, iter_idx] != 0)[0]
    if len(selected_vars) == 0:  # Skip if no variables selected
        return iter_idx, np.inf, -np.inf, 1

    # Get selected data
    X_sel = X_array[:, selected_vars]

    # Calculate maximum possible components once
    max_possible_components = min(X_sel.shape[0]-1, X_sel.shape[1], max_components)
    comp_range = range(1, min(max_possible_components+1, len(selected_vars)+1))
    
    # Evaluate each component count
    comp_results = []

    for n_comp in comp_range:
        n_comp, rmse, r2 = evaluate_component_count(
            X_sel, y_array, n_comp, kf, preprocess)
        comp_results.append((n_comp, rmse, r2))

    # Find the best component count
    comp_results.sort(key=lambda x: x[1])  # Sort by RMSE
    best_comp, best_rmse, best_r2 = comp_results[0]

    return iter_idx, best_rmse, best_r2, best_comp

def competitive_adaptive_sampling(X, y, max_components, folds=5, preprocess='center', iterations=50,
                              adaptive_resampling=False, cv_shuffle_mode='none',
                              n_jobs=-1, verbose=1):
    """
    Optimized Competitive Adaptive Reweighted Sampling for variable selection in PLS

    Parameters:
    -----------
    X : array-like
        The predictor matrix of shape (n_samples, n_features)
    y : array-like
        The response vector of shape (n_samples,)
    max_components : int
        The maximum number of PLS components to extract
    folds : int, default=5
        Number of folds for cross-validation
    preprocess : str, default='center'
        Preprocessing method ('center', 'autoscaling', 'pareto', 'minmax', 'robust', 'unilength', 'none')
    iterations : int, default=50
        Number of Monte Carlo sampling runs
    adaptive_resampling : bool, default=False
        Whether to use the original version with random sampling (True) or a simplified deterministic version (False)
    cv_shuffle_mode : str, default='none'
        Cross-validation sample ordering mode: 
        - 'none': maintain original sample order in cross-validation
        - 'fixed_seed': shuffle samples with a fixed random seed (42)
        - 'random_seed': shuffle samples with a random seed for each run
    n_jobs : int, default=-1
        Number of parallel jobs for cross-validation
    verbose : int, default=1
        Verbosity level (0=silent, 1=summary progress only, 2=detailed per-iteration progress)

    Returns:
    --------
    dict : Results containing selected variables and performance metrics
    """
    start_time = time.time()

    # Convert inputs to numpy arrays if needed
    X_array = X.values if isinstance(X, pd.DataFrame) else X
    y_array = y.values if isinstance(y, pd.Series) else y

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

    # Set up the KFold cross-validation based on the cv_shuffle_mode
    if cv_shuffle_mode == 'none':
        # No shuffling - maintain original order
        kf = KFold(n_splits=folds, shuffle=False)
    elif cv_shuffle_mode == 'fixed_seed':
        # Shuffle with fixed seed (42)
        kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    elif cv_shuffle_mode == 'random_seed':
        # Shuffle with random seed
        kf = KFold(n_splits=folds, shuffle=True, random_state=None)
    else:
        raise ValueError(f"Invalid cv_shuffle_mode: {cv_shuffle_mode}. Must be 'none', 'fixed_seed', or 'random_seed'")

    # Print initial info
    if verbose >= 1:
        print(f"Starting variable selection with {n_features} initial variables...")
        if verbose == 1:
            print(f"Running {iterations} iterations - progress will be shown at completion")

    # Main sampling loop - variable elimination phase
    for iter_idx in range(iterations):
        # Either use Monte Carlo sampling or full dataset
        if adaptive_resampling:
            rand_indices = np.random.permutation(n_samples)
            X_subset = X_array[rand_indices[:subset_size]]
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

        # Fit PLS model using the helper function
        pls = fit_pls_with_preprocessing(X_sel, y_subset, max_possible_components, preprocess)

        # Extract coefficients and update weights - use pre-allocated buffer for efficiency
        feature_weights_buffer.fill(0)  # Reset buffer

        # Get coefficients excluding intercept
        coefs = pls.coef_
        feature_weights_buffer[selected_features] = coefs.flatten()
        weight_matrix[:, iter_idx] = feature_weights_buffer

        # Take absolute value for selection purposes (in place)
        np.abs(feature_weights_buffer, out=feature_weights_buffer)

        # Calculate exponentially decreasing sampling ratio
        current_ratio = decay_coefficient * np.exp(-decay_factor * (iter_idx+1))
        subset_ratios[iter_idx] = current_ratio
        keep_count = max(1, int(n_features * current_ratio))  # Ensure at least 1 variable

        # For both adaptive and deterministic approaches, zero out weights of variables that 
        # won't be included in the top keep_count
        if keep_count < n_features:
            # Use argpartition for better performance
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
            # Deterministic selection - faster than np.where
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
        delayed(evaluate_iteration)(
            i, weight_matrix, X_array, y_array, max_components, kf, preprocess
        ) for i in range(iterations)
    )

    # Unpack the results
    cv_errors = np.zeros(iterations)
    r_squared = np.zeros(iterations)
    optimal_components = np.zeros(iterations, dtype=int)

    for idx, rmse, r2, components in results_parallel:
        cv_errors[idx] = rmse
        r_squared[idx] = r2
        optimal_components[idx] = components

    # Find the optimal subset
    best_subset_idx = np.argmin(cv_errors)
    best_rmsecv = cv_errors[best_subset_idx]
    best_r2 = r_squared[best_subset_idx]
    selected_vars = np.where(weight_matrix[:, best_subset_idx] != 0)[0]

    # Compile results
    elapsed_time = time.time() - start_time

    if verbose >= 1:
        print(f"Optimization completed in {elapsed_time:.2f} seconds")
        print(f"Best subset found at iteration {best_subset_idx+1} with RMSECV: {best_rmsecv:.4f}")
        print(f"Selected {len(selected_vars)} variables")

    results = {
        'weight_matrix': weight_matrix,
        'computation_time': elapsed_time,
        'cross_validation_errors': cv_errors,
        'min_cv_error': best_rmsecv,
        'max_r_squared': best_r2,
        'best_iteration': best_subset_idx,
        'optimal_components': optimal_components[best_subset_idx],
        'subset_ratios': subset_ratios,
        'selected_variables': selected_vars
    }

    return results

# Import visualization functions
from .visualization import plot_sampling_results
