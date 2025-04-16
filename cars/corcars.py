import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import warnings
import time

# Import from utils and preprocessing
from .utils import suppress_pls_warnings
from .preprocessing import preprocess_data
from .visualization import plot_sampling_results

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
    # Use scikit-learn's built-in cross_val_predict for methods with built-in scaling
    if preprocess in ['center', 'autoscaling']:
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
    errors = y_array - y_pred
    press = np.sum(errors**2)
    rmse = np.sqrt(mean_squared_error(y_array, y_pred))
    r2 = r2_score(y_array, y_pred)
    
    return n_comp, press, rmse, r2, errors

def select_optimal_components(press_values, errors_matrix, alpha=0.25, use_correlation=True):
    """
    Select optimal number of components with optional correlation adjustment
    
    Parameters:
    -----------
    press_values : array
        PRESS values for each number of components, indexed from 0 (for 1 component)
    errors_matrix : array
        Matrix of prediction errors for each component count
    alpha : float
        Significance level (default: 0.25)
    use_correlation : bool
        Whether to use correlation adjustment (default: True)
    
    Returns:
    --------
    int : Optimal number of components (1-based)
    """
    n_samples = errors_matrix.shape[0]
    n_components = len(press_values)
    
    # Find minimum PRESS and corresponding component number (1-based)
    min_press_idx = np.argmin(press_values)
    min_press = press_values[min_press_idx]
    h_star = min_press_idx + 1  # Convert to 1-based component number
    
    # Calculate F-ratios for each model compared to best model
    f_ratios = np.zeros(h_star)
    
    for h in range(1, h_star+1):
        h_idx = h-1  # Convert to 0-based index
        # Standard F-ratio
        f_ratio = press_values[h_idx] / min_press
        
        if use_correlation and h < h_star:
            # Calculate correlation between prediction errors
            correlation = np.corrcoef(errors_matrix[:, h_idx], errors_matrix[:, min_press_idx])[0, 1]
            
            # Adjust F-ratio based on correlation
            # Higher correlation means lower effective F-ratio
            adjustment_factor = max(0.5, 1 - abs(correlation))
            adjusted_f_ratio = 1 + (f_ratio - 1) * adjustment_factor
        else:
            # Use standard F-ratio without adjustment
            adjusted_f_ratio = f_ratio
            
        f_ratios[h_idx] = adjusted_f_ratio
    
    # Find simplest model that's not significantly worse
    # Use appropriate degrees of freedom for F-test in PLS context
    # df1 = df2 = n_samples - h (approximate)
    for h in range(1, h_star+1):
        h_idx = h-1
        df1 = df2 = max(1, n_samples - h)  # Ensure df is at least 1
        critical_f = stats.f.ppf(1-alpha, df1, df2)
        
        if f_ratios[h_idx] < critical_f:
            return h
    
    return h_star

def evaluate_iteration(iter_idx, weight_matrix, X_array, y_array, max_components, kf, preprocess, 
                      alpha=0.25, use_correlation=True):
    """Evaluate a single iteration (subset)"""
    # Get non-zero weight features for this iteration
    selected_vars = np.where(weight_matrix[:, iter_idx] != 0)[0]
    if len(selected_vars) == 0:  # Skip if no variables selected
        return iter_idx, np.inf, -np.inf, 1, []

    # Get selected data
    X_sel = X_array[:, selected_vars]

    # Calculate maximum possible components once
    max_possible_components = min(X_sel.shape[0]-1, X_sel.shape[1], max_components)
    comp_range = range(1, min(max_possible_components+1, len(selected_vars)+1))
    
    # Evaluate each component count
    comp_results = []
    all_errors = np.zeros((len(y_array), len(comp_range)))
    
    for i, n_comp in enumerate(comp_range):
        n_comp, press, rmse, r2, errors = evaluate_component_count(
            X_sel, y_array, n_comp, kf, preprocess)
        
        comp_results.append((n_comp, press, rmse, r2))
        all_errors[:, i] = errors
    
    # Extract PRESS values
    press_values = np.array([res[1] for res in comp_results])
    
    # Select optimal component count
    best_comp = select_optimal_components(
        press_values, all_errors, alpha, use_correlation)
    
    # Get metrics for the optimal component count
    best_metrics = [res for res in comp_results if res[0] == best_comp][0]
    _, _, best_rmse, best_r2 = best_metrics
    
    return iter_idx, best_rmse, best_r2, best_comp, selected_vars

def competitive_adaptive_reweighted_sampling(X, y, max_components, folds=5, preprocess='center', 
                                           iterations=50, adaptive_resampling=False, 
                                           cv_shuffle_mode='none', use_correlation=True,
                                           alpha=0.25, n_jobs=-1, verbose=1):
    """
    CorCARS: Correlation-adjusted Competitive Adaptive Reweighted Sampling
    
    This version extends the original CARS algorithm by incorporating correlation 
    information when selecting the optimal number of PLS components. By accounting 
    for the correlation between prediction errors at different complexity levels,
    CorCARS can identify more parsimonious models while maintaining predictive performance.
    
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
    use_correlation : bool, default=True
        Whether to use correlation adjustment in F-test component selection
    alpha : float, default=0.25
        Significance level for F-test
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
    X_array = X.values if isinstance(X, pd.DataFrame) else np.array(X)
    y_array = y.values if isinstance(y, pd.Series) else np.array(y)
    if len(y_array.shape) > 1 and y_array.shape[1] == 1:
        y_array = y_array.ravel()

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
        if use_correlation:
            print("Using correlation-adjusted F-test for component selection")
        else:
            print("Using standard F-test for component selection")

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

    # Calculate adaptive batch size based on dataset dimensions and available jobs
    # Smaller datasets can use larger batches
    data_size_factor = min(1.0, 1000 / (n_samples * n_features))
    batch_size = max(1, min(25, int(iterations * data_size_factor)))
    
    if verbose >= 2:
        print(f"Using batch size of {batch_size} for parallel processing")
    
    # Initialize arrays to store results
    cv_errors = np.full(iterations, np.inf)
    r_squared = np.full(iterations, -np.inf)
    optimal_components = np.zeros(iterations, dtype=int)
    selected_vars_list = [None] * iterations
    
    # Break evaluation into batches to avoid memory issues with large datasets
    for batch_start in range(0, iterations, batch_size):
        batch_end = min(batch_start + batch_size, iterations)
        
        batch_results = Parallel(n_jobs=n_jobs, verbose=max(0, verbose-1))(
            delayed(evaluate_iteration)(
                i, weight_matrix, X_array, y_array, max_components, kf, preprocess, 
                alpha, use_correlation
            ) for i in range(batch_start, batch_end)
        )
        
        # Process results from this batch
        for idx, rmse, r2, components, vars_idx in batch_results:
            if rmse != np.inf:  # Only update if we got valid results
                cv_errors[idx] = rmse
                r_squared[idx] = r2
                optimal_components[idx] = components
                selected_vars_list[idx] = vars_idx

    # Find the optimal subset among valid results
    valid_indices = np.where(cv_errors != np.inf)[0]
    
    if len(valid_indices) == 0:
        raise ValueError("All models failed. Try different parameters.")
        
    best_subset_idx = valid_indices[np.argmin(cv_errors[valid_indices])]
    best_rmse = cv_errors[best_subset_idx]
    best_r2 = r_squared[best_subset_idx]
    selected_vars = selected_vars_list[best_subset_idx]

    # Compile results
    elapsed_time = time.time() - start_time

    if verbose >= 1:
        print(f"Optimization completed in {elapsed_time:.2f} seconds")
        print(f"Best subset found at iteration {best_subset_idx+1} with RMSE: {best_rmse:.4f}")
        print(f"Selected {len(selected_vars)} variables with {optimal_components[best_subset_idx]} components")

    results = {
        'weight_matrix': weight_matrix,
        'computation_time': elapsed_time,
        'cross_validation_errors': cv_errors,
        'min_cv_error': best_rmse,
        'max_r_squared': best_r2,
        'best_iteration': best_subset_idx,
        'optimal_components': optimal_components[best_subset_idx],
        'subset_ratios': subset_ratios,
        'selected_variables': selected_vars,
        'use_correlation': use_correlation,
        'alpha': alpha
    }

    return results

# Alias for CorCARS
corcars = competitive_adaptive_reweighted_sampling

# For backward compatibility with CARS ver4.5
def competitive_adaptive_sampling(*args, **kwargs):
    """
    Alias for competitive_adaptive_reweighted_sampling with correlation adjustment off.
    
    This maintains compatibility with the original CARS algorithm while allowing access
    to the improved implementation.
    """
    # Set use_correlation=False if not explicitly specified
    if 'use_correlation' not in kwargs:
        kwargs['use_correlation'] = False
    return competitive_adaptive_reweighted_sampling(*args, **kwargs)
