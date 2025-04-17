import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
import warnings
import seaborn as sns
from joblib import Parallel, delayed
import contextlib
import sys
import os
import importlib.util

class CARSOptimizer:
    """
    A versatile parameter optimizer for CARS variants: standard CARS, CorCARS, and CARS Classification.
    Each optimization method works independently.
    """
    
    def __init__(self, X, y, cars_variant='standard', component_ranges=None, preprocess_options=None, 
                 folds=5, iterations=100, n_jobs=-1, verbose=1, 
                 cars_func=None, plot_func=None, **variant_kwargs):
        """
        Initialize the optimizer with dataset and parameter search ranges.
        
        Parameters:
        -----------
        X : array-like
            The predictor matrix
        y : array-like
            The response vector
        cars_variant : str
            Which CARS variant to use: 'standard', 'corcars', or 'classification'
        component_ranges : list, optional
            List of max_components values to try. If None, will use [5,10,15,20,25]
        preprocess_options : list, optional
            List of preprocessing methods to try. If None, will use 
            ['center', 'autoscaling', 'pareto', 'minmax']
        folds : int
            Number of folds for cross-validation 
        iterations : int
            Number of CARS iterations 
        n_jobs : int
            Number of parallel jobs
        verbose : int
            Verbosity level
        cars_func : function, optional
            Direct reference to the CARS implementation function
        plot_func : function, optional
            Direct reference to the plotting function
        **variant_kwargs : dict
            Additional keyword arguments specific to the CARS variant:
            - For classification: 'encoding' ('ordinal'/'onehot'), 'metric' ('accuracy'/'f1'/'auc')
            - For CorCARS: 'use_correlation' (bool), 'alpha' (float)
        """
        # Store original X and y
        self.X_orig = X
        self.y_orig = y
        
        # Convert to numpy arrays for internal use
        self.X = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        self.y = y.values if isinstance(y, pd.Series) else np.array(y)
        
        # Ensure y is 1D for regression
        if cars_variant != 'classification' and len(self.y.shape) > 1 and self.y.shape[1] == 1:
            self.y = self.y.ravel()
        
        # Store CARS variant
        self.cars_variant = cars_variant
        self.variant_kwargs = variant_kwargs
        
        # Default parameter ranges if not provided
        self.component_ranges = component_ranges or [5, 10, 15, 20, 25, 30]
        self.preprocess_options = preprocess_options or ['center', 'autoscaling', 'pareto', 'minmax']
        
        # Parameters
        self.folds = folds
        self.iterations = iterations
        
        # Other settings
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Store function references if provided directly
        self.cars_func = cars_func
        self.plot_func = plot_func
        
        # Validate CARS variant
        valid_variants = ['standard', 'corcars', 'classification']
        if self.cars_variant not in valid_variants:
            raise ValueError(f"Invalid CARS variant: {self.cars_variant}. Must be one of {valid_variants}")
            
        # Handle classification-specific parameters
        if self.cars_variant == 'classification':
            self.encoding = variant_kwargs.get('encoding', 'ordinal')
            self.metric = variant_kwargs.get('metric', 'accuracy')
            # Further validations for classification
            if self.encoding not in ['ordinal', 'onehot']:
                raise ValueError(f"Invalid encoding: {self.encoding}. Must be 'ordinal' or 'onehot'")
            if self.metric not in ['accuracy', 'f1', 'auc']:
                warnings.warn(f"Unusual metric: {self.metric}. Common metrics are 'accuracy', 'f1', or 'auc'")
        
        # CorCARS specific parameters
        if self.cars_variant == 'corcars':
            self.use_correlation = variant_kwargs.get('use_correlation', True)
            self.alpha = variant_kwargs.get('alpha', 0.25)
        
        # Find and import appropriate CARS implementation only if functions not provided
        if self.cars_func is None:
            try:
                self._import_cars_implementation()
            except ImportError as e:
                if self.verbose:
                    print(f"Warning: {str(e)}")
                    print("You may need to pass the functions directly using cars_func and plot_func parameters.")
        
        # Calculate baseline metrics
        try:
            self._calculate_baseline_metrics()
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not calculate baseline metrics: {str(e)}")
            # Set default values
            self.baseline_rmse = 1.0
            self.baseline_r2 = 0.0
            self.baseline_accuracy = 0.5  # Default for classification
            self.baseline_f1 = 0.5  # Default for classification
            self.total_vars = self.X.shape[1]
            self.max_possible_components = min(self.X.shape[0], self.X.shape[1])
    
    def _import_cars_implementation(self):
        """Import the appropriate CARS implementation based on variant"""
        # First, try to import as a module
        try:
            if self.cars_variant == 'standard':
                # Try different module names that might contain CARS implementation
                module_names = ['CARS']
                for name in module_names:
                    try:
                        module = importlib.import_module(name)
                        self.cars_func = module.competitive_adaptive_sampling
                        self.plot_func = getattr(module, 'plot_sampling_results', None)
                        if self.verbose:
                            print(f"Successfully imported {self.cars_variant} from module {name}")
                        return
                    except (ImportError, AttributeError):
                        continue
                    
            elif self.cars_variant == 'corcars':
                # Try different module names for CorCARS
                module_names = ['CorCARS']
                for name in module_names:
                    try:
                        module = importlib.import_module(name)
                        self.cars_func = module.competitive_adaptive_reweighted_sampling
                        self.plot_func = getattr(module, 'plot_sampling_results', None)
                        if self.verbose:
                            print(f"Successfully imported {self.cars_variant} from module {name}")
                        return
                    except (ImportError, AttributeError):
                        continue
                    
            elif self.cars_variant == 'classification':
                # Try different module names for CARS Classification
                module_names = ['CARS_classification']
                for name in module_names:
                    try:
                        module = importlib.import_module(name)
                        self.cars_func = module.competitive_adaptive_sampling_classification
                        self.plot_func = getattr(module, 'plot_classification_results', None)
                        if self.verbose:
                            print(f"Successfully imported {self.cars_variant} from module {name}")
                        return
                    except (ImportError, AttributeError):
                        continue
        
        except Exception as e:
            if self.verbose:
                print(f"Could not import module: {str(e)}")
        
        # If we get here, we couldn't import as a module
        # Instead, try to dynamically load from file in a more GitHub-friendly way
        if self.cars_variant == 'standard':
            file_names = ['CARS.py']  # Simplified file names for GitHub compatibility
            func_name = 'competitive_adaptive_sampling'
            plot_name = 'plot_sampling_results'
        elif self.cars_variant == 'corcars':
            file_names = ['CorCARS.py']
            func_name = 'competitive_adaptive_reweighted_sampling'
            plot_name = 'plot_sampling_results'
        else:  # classification
            file_names = ['CARS_classification.py'] 
            func_name = 'competitive_adaptive_sampling_classification'
            plot_name = 'plot_classification_results'
        
        # Try to load from files
        for file_name in file_names:
            try:
                # Check if file exists
                if not os.path.isfile(file_name):
                    continue
                
                # Generate a module name from the file name
                mod_name = os.path.splitext(os.path.basename(file_name))[0].replace(' ', '_')
                
                # Load the module dynamically
                spec = importlib.util.spec_from_file_location(mod_name, file_name)
                module = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = module
                spec.loader.exec_module(module)
                
                # Get the function
                self.cars_func = getattr(module, func_name)
                self.plot_func = getattr(module, plot_name, None)
                
                if self.verbose:
                    print(f"Successfully loaded {self.cars_variant} from file {file_name}")
                return
                
            except Exception as e:
                if self.verbose:
                    print(f"Error loading from {file_name}: {str(e)}")
                continue
        
        # If we get here, we couldn't load the functions
        raise ImportError(f"Could not import {self.cars_variant} implementation. Ensure the files are in the correct location or pass the functions directly.")
    
    def _calculate_baseline_metrics(self):
        """Calculate baseline metrics for normalization"""
        # This is useful for normalizing metrics in multi-objective optimization
        from sklearn.cross_decomposition import PLSRegression
        cv = KFold(n_splits=self.folds, shuffle=True, random_state=42)
        
        # For regression problems (standard, corcars)
        if self.cars_variant in ['standard', 'corcars']:
            full_pls = PLSRegression(n_components=min(5, self.X.shape[1]))
            
            y_pred_list = []
            y_true_list = []
            
            for train_idx, test_idx in cv.split(self.X):
                X_train, X_test = self.X[train_idx], self.X[test_idx]
                y_train, y_test = self.y[train_idx], self.y[test_idx]
                
                full_pls.fit(X_train, y_train)
                y_pred = full_pls.predict(X_test).ravel()
                
                y_pred_list.append(y_pred)
                y_true_list.append(y_test)
            
            # Concatenate all predictions
            y_pred_all = np.concatenate(y_pred_list)
            y_true_all = np.concatenate(y_true_list)
            
            # Calculate baseline metrics
            self.baseline_rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
            self.baseline_r2 = r2_score(y_true_all, y_pred_all)
            
        # For classification problems
        elif self.cars_variant == 'classification':
            # Handle different encoding formats
            if self.encoding == 'onehot' and len(self.y.shape) > 1:
                # Get class indices for stratification in CV
                y_indices = np.argmax(self.y, axis=1)
                n_classes = self.y.shape[1]
            else:
                y_indices = self.y
                n_classes = len(np.unique(y_indices))
            
            # Use stratified CV for classification
            cv = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=42)
            
            # Create PLS model
            full_pls = PLSRegression(n_components=min(5, self.X.shape[1]))
            
            y_pred_list = []
            y_true_list = []
            
            for train_idx, test_idx in cv.split(self.X, y_indices):
                X_train, X_test = self.X[train_idx], self.X[test_idx]
                
                if self.encoding == 'onehot' and len(self.y.shape) > 1:
                    y_train = self.y[train_idx]
                    # For evaluation, use indices
                    y_test = y_indices[test_idx]
                else:
                    y_train = self.y[train_idx].reshape(-1, 1) if len(self.y.shape) == 1 else self.y[train_idx]
                    y_test = y_indices[test_idx]
                
                full_pls.fit(X_train, y_train)
                
                # For one-hot encoding, predict class with highest score
                if self.encoding == 'onehot' and len(self.y.shape) > 1:
                    y_pred_raw = full_pls.predict(X_test)
                    y_pred = np.argmax(y_pred_raw, axis=1)
                else:
                    # For ordinal, round to nearest integer and clip to valid range
                    y_pred_raw = full_pls.predict(X_test).ravel()
                    y_pred = np.round(y_pred_raw).clip(0, n_classes-1).astype(int)
                
                y_pred_list.append(y_pred)
                y_true_list.append(y_test)
            
            # Concatenate all predictions
            y_pred_all = np.concatenate(y_pred_list)
            y_true_all = np.concatenate(y_true_list)
            
            # Calculate classification baseline metrics
            self.baseline_accuracy = accuracy_score(y_true_all, y_pred_all)
            self.baseline_f1 = f1_score(y_true_all, y_pred_all, average='weighted' if n_classes > 2 else 'binary')
        
        # Store other useful values
        self.total_vars = self.X.shape[1]
        self.max_possible_components = min(self.X.shape[0], self.X.shape[1])
        
        if self.verbose:
            if self.cars_variant in ['standard', 'corcars']:
                print(f"Baseline RMSE: {self.baseline_rmse:.4f}")
                print(f"Baseline R²: {self.baseline_r2:.4f}")
            else:
                print(f"Baseline Accuracy: {self.baseline_accuracy:.4f}")
                print(f"Baseline F1 Score: {self.baseline_f1:.4f}")
    
    def _run_cars(self, max_components, preprocess, **extra_kwargs):
        """
        Run the appropriate CARS variant with given parameters
        
        Parameters:
        -----------
        max_components : int
            Maximum number of components to use
        preprocess : str
            Preprocessing method
        extra_kwargs : dict
            Additional keyword arguments specific to the CARS variant
            
        Returns:
        --------
        dict : Results from CARS run
        """
        if self.cars_func is None:
            raise ImportError("CARS implementation not available. Please provide the function using cars_func parameter.")
        
        # Combine default and user-provided kwargs
        call_kwargs = {}
        
        # Common parameters
        call_kwargs.update({
            'X': self.X_orig,
            'y': self.y_orig,
            'max_components': max_components,
            'preprocess': preprocess,
            'folds': self.folds,
            'iterations': self.iterations,
            'n_jobs': self.n_jobs,
            'verbose': max(0, self.verbose-1)
        })
        
        # Add variant-specific parameters
        if self.cars_variant == 'classification':
            call_kwargs.update({
                'encoding': extra_kwargs.pop('encoding', self.encoding),
                'metric': extra_kwargs.pop('metric', self.metric)
            })
        elif self.cars_variant == 'corcars':
            call_kwargs.update({
                'use_correlation': extra_kwargs.pop('use_correlation', self.use_correlation),
                'alpha': extra_kwargs.pop('alpha', self.alpha)
            })
        
        # Add any additional kwargs
        call_kwargs.update(extra_kwargs)
        
        # Call the appropriate CARS function
        return self.cars_func(**call_kwargs)
    
    def staged_parameter_scan(self, alpha=0.6, beta=0.2, gamma=0.2, batch_size=None, **kwargs):
        """
        Find optimal parameters using a two-stage approach:
        1. First scan with fixed preprocess='autoscaling' to find optimal max_components region
        2. Then fine-tune within that region using all preprocessing methods
        
        Parameters:
        -----------
        alpha : float
            Weight for predictive performance (RMSE or classification metric)
        beta : float
            Weight for model parsimony (number of variables)
        gamma : float
            Weight for component complexity
        batch_size : int, optional
            Number of parameter combinations to evaluate in a batch to limit memory usage
        **kwargs : dict
            Additional parameters specific to the CARS variant
            
        Returns:
        --------
        dict : Results of optimization
        """
        start_time = time.time()
        
        if self.verbose:
            print("Starting Staged Parameter Scan...")
            print(f"Weights: Performance={alpha}, Variables={beta}, Components={gamma}")
            print("Stage 1: Coarse scan with autoscaling to find optimal max_components region")
        
        # Stage 1: Coarse scan with fixed preprocess='autoscaling'
        coarse_results = []
        
        # Calculate default batch size based on available memory and dataset size
        if batch_size is None:
            # Heuristic: use smaller batches for larger datasets
            data_size_factor = min(1.0, 1000 / (self.X.shape[0] * self.X.shape[1]))
            batch_size = max(1, min(5, int(len(self.component_ranges) * data_size_factor)))
        
        # Process coarse scan in batches to manage memory
        for batch_start in range(0, len(self.component_ranges), batch_size):
            batch_end = min(batch_start + batch_size, len(self.component_ranges))
            batch_components = self.component_ranges[batch_start:batch_end]
            
            if self.verbose:
                print(f"  Evaluating batch {batch_start//batch_size + 1}: max_components={batch_components}")
            
            for max_comp in batch_components:
                try:
                    result = self._run_cars(max_comp, 'autoscaling', **kwargs)
                    
                    # Handle different metrics based on CARS variant
                    if self.cars_variant in ['standard', 'corcars']:
                        # Regression metrics
                        perf_metric = result['min_cv_error']  # RMSE, lower is better
                        normalized_perf = perf_metric / self.baseline_rmse if hasattr(self, 'baseline_rmse') else perf_metric
                        better_perf = normalized_perf < 1  # Lower is better
                    else:
                        # Classification metrics
                        perf_metric = result['best_metric_value']  # Accuracy, F1, etc., higher is better
                        normalized_perf = 1 - perf_metric  # Convert to error (lower is better)
                        better_perf = normalized_perf < 0.5  # Lower is better
                    
                    # Calculate other metrics
                    n_vars = len(result['selected_variables'])
                    if 'optimal_components' in result:
                        n_components = result['optimal_components']
                    else:
                        n_components = max_comp  # Fallback
                    
                    # Normalize metrics
                    norm_vars = n_vars / self.total_vars
                    norm_comp = n_components / self.max_possible_components
                    
                    # Calculate composite score (lower is better)
                    score = alpha * normalized_perf + beta * norm_vars + gamma * norm_comp
                    
                    # Store enhanced result
                    result['max_components_setting'] = max_comp
                    result['preprocess_method'] = 'autoscaling'
                    result['multi_obj_score'] = score
                    result['norm_perf'] = normalized_perf
                    result['norm_vars'] = norm_vars
                    result['norm_comp'] = norm_comp
                    result['n_selected_vars'] = n_vars
                    
                    coarse_results.append(result)
                    
                except Exception as e:
                    print(f"Error with parameters {max_comp}, 'autoscaling': {str(e)}")
        
        # Find best max_components region
        if not coarse_results:
            print("No valid results in coarse scan. Cannot proceed.")
            return None
        
        # Sort by score (lower is better)
        coarse_results.sort(key=lambda x: x['multi_obj_score'])
        
        # Find the optimal range for fine-tuning
        best_comp = coarse_results[0]['max_components_setting']
        
        # Determine fine scan range around the best component value
        min_comp = max(best_comp - 4, 1)
        max_comp = best_comp + 5  # +5 to include best_comp+4 in range
        
        if self.verbose:
            print(f"Best max_components from coarse scan: {best_comp}")
            print(f"Stage 2: Fine scan in range ({min_comp}-{max_comp-1}) with all preprocessing methods")
        
        # Stage 2: Fine scan with all preprocessing methods
        fine_results = []
        
        # Create list of all parameter combinations for fine scan
        fine_params = []
        for max_comp in range(min_comp, max_comp):
            for prep in self.preprocess_options:
                fine_params.append((max_comp, prep))
        
        # Process fine scan in batches
        for batch_start in range(0, len(fine_params), batch_size):
            batch_end = min(batch_start + batch_size, len(fine_params))
            batch_params = fine_params[batch_start:batch_end]
            
            if self.verbose:
                param_str = ", ".join([f"({p[0]}, {p[1]})" for p in batch_params[:3]])
                if len(batch_params) > 3:
                    param_str += f", ... ({len(batch_params)} total)"
                print(f"  Evaluating batch {batch_start//batch_size + 1}: params={param_str}")
            
            for max_comp, prep in batch_params:
                try:
                    result = self._run_cars(max_comp, prep, **kwargs)
                    
                    # Handle different metrics based on CARS variant
                    if self.cars_variant in ['standard', 'corcars']:
                        # Regression metrics
                        perf_metric = result['min_cv_error']  # RMSE, lower is better
                        normalized_perf = perf_metric / self.baseline_rmse if hasattr(self, 'baseline_rmse') else perf_metric
                    else:
                        # Classification metrics
                        perf_metric = result['best_metric_value']  # Accuracy, F1, etc., higher is better
                        normalized_perf = 1 - perf_metric  # Convert to error (lower is better)
                    
                    # Calculate other metrics
                    n_vars = len(result['selected_variables'])
                    if 'optimal_components' in result:
                        n_components = result['optimal_components']
                    else:
                        n_components = max_comp  # Fallback
                    
                    # Normalize metrics
                    norm_vars = n_vars / self.total_vars
                    norm_comp = n_components / self.max_possible_components
                    
                    # Calculate composite score (lower is better)
                    score = alpha * normalized_perf + beta * norm_vars + gamma * norm_comp
                    
                    # Store enhanced result
                    result['max_components_setting'] = max_comp
                    result['preprocess_method'] = prep
                    result['multi_obj_score'] = score
                    result['norm_perf'] = normalized_perf
                    result['norm_vars'] = norm_vars
                    result['norm_comp'] = norm_comp
                    result['n_selected_vars'] = n_vars
                    
                    fine_results.append(result)
                    
                except Exception as e:
                    print(f"Error with parameters {max_comp}, {prep}: {str(e)}")
        
        # Find best result from fine scan
        if fine_results:
            fine_results.sort(key=lambda x: x['multi_obj_score'])
            best_result = fine_results[0]
        else:
            # Fallback to coarse scan if fine scan failed
            best_result = coarse_results[0]
        
        elapsed_time = time.time() - start_time
        
        if self.verbose:
            print(f"Staged Parameter Scan completed in {elapsed_time:.2f} seconds")
            print(f"Best parameters: max_components={best_result['max_components_setting']}, "
                  f"preprocess='{best_result['preprocess_method']}'")
            print(f"Selected {best_result['n_selected_vars']} variables with "
                  f"{best_result['optimal_components']} components")
            
            if self.cars_variant in ['standard', 'corcars']:
                print(f"RMSE: {best_result['min_cv_error']:.4f}, R²: {best_result['max_r_squared']:.4f}")
            else:
                print(f"{self.metric.upper()}: {best_result['best_metric_value']:.4f}")
        
        # Store results
        optimization_results = {
            'method': 'staged_parameter_scan',
            'best_result': best_result,
            'all_results': coarse_results + fine_results,
            'computation_time': elapsed_time,
            'weights': {'alpha': alpha, 'beta': beta, 'gamma': gamma},
            'scan_details': {
                'coarse_best': best_comp,
                'fine_range': (min_comp, max_comp-1)
            },
            'cars_variant': self.cars_variant
        }
        
        return optimization_results

    def bayesian_optimization(self, n_calls=20, min_components=5, max_components=30, penalization=1e5, **kwargs):
        """
        Find optimal parameters using Bayesian optimization.
        
        Parameters:
        -----------
        n_calls : int
            Number of parameter combinations to evaluate
        min_components : int
            Minimum number of components to try
        max_components : int
            Maximum number of components to try
        penalization : float
            Value to return for failed evaluations (high for minimization)
        **kwargs : dict
            Additional parameters specific to the CARS variant
            
        Returns:
        --------
        dict : Results of optimization
        """
        try:
            # Try to import skopt for Bayesian optimization
            from skopt import gp_minimize
            from skopt.space import Integer, Categorical
            from skopt.plots import plot_convergence
        except ImportError:
            print("Skipping Bayesian optimization as scikit-optimize is not available.")
            print("Install with: pip install scikit-optimize")
            return None
        
        start_time = time.time()
        
        if self.verbose:
            print(f"Starting Bayesian Optimization with {n_calls} evaluations...")
        
        # Define the search space
        # Limit max_components to a reasonable range based on dataset
        max_components = min(max_components, self.X.shape[0] // 2, self.X.shape[1])
        
        search_space = [
            Integer(min_components, max_components, name='max_components', prior='log-uniform'),
            Categorical(self.preprocess_options, name='preprocess')
        ]
        
        # Storage for all results
        all_results = []
        failed_evaluations = []
        
        # Define the objective function
        def objective(params):
            max_components, preprocess = params
            
            if self.verbose:
                print(f"  Evaluating max_components={max_components}, preprocess='{preprocess}'")
            
            try:
                # Run CARS with these parameters
                result = self._run_cars(max_components, preprocess, **kwargs)
                
                # Store enhanced result
                result['max_components_setting'] = max_components
                result['preprocess_method'] = preprocess
                if 'selected_variables' in result:
                    result['n_selected_vars'] = len(result['selected_variables'])
                
                all_results.append(result)
                
                # Return the appropriate metric as the objective to minimize
                if self.cars_variant in ['standard', 'corcars']:
                    return result['min_cv_error']  # RMSE for regression
                else:
                    # For classification, convert to error (1 - metric) since we're minimizing
                    return 1 - result['best_metric_value']
                    
            except Exception as e:
                error_info = {
                    'parameters': {'max_components': max_components, 'preprocess': preprocess},
                    'error': str(e)
                }
                failed_evaluations.append(error_info)
                if self.verbose:
                    print(f"Error with parameters {max_components}, {preprocess}: {str(e)}")
                # Return a penalized value on error
                return penalization
        
        # Run Bayesian optimization
        opt_result = gp_minimize(
            objective, 
            search_space, 
            n_calls=n_calls, 
            random_state=42, 
            verbose=self.verbose
        )
        
        # Check if we have valid results
        if not all_results:
            print("No valid results from Bayesian optimization. Check for errors.")
            if failed_evaluations:
                print(f"Found {len(failed_evaluations)} failed evaluations:")
                for i, failure in enumerate(failed_evaluations[:3]):
                    print(f"  {i+1}. {failure['parameters']} - {failure['error']}")
                if len(failed_evaluations) > 3:
                    print(f"  ... and {len(failed_evaluations) - 3} more.")
            return None
        
        # Find the corresponding CARS result
        if self.cars_variant in ['standard', 'corcars']:
            best_result = min(all_results, key=lambda x: x['min_cv_error'])
        else:
            best_result = max(all_results, key=lambda x: x['best_metric_value'])
        
        elapsed_time = time.time() - start_time
        
        if self.verbose:
            print(f"Bayesian Optimization completed in {elapsed_time:.2f} seconds")
            print(f"Best parameters: max_components={best_result['max_components_setting']}, "
                  f"preprocess='{best_result['preprocess_method']}'")
            print(f"Selected {best_result['n_selected_vars']} variables with "
                  f"{best_result['optimal_components']} components")
            
            if self.cars_variant in ['standard', 'corcars']:
                print(f"RMSE: {best_result['min_cv_error']:.4f}, R²: {best_result['max_r_squared']:.4f}")
            else:
                print(f"{self.metric.upper()}: {best_result['best_metric_value']:.4f}")
            
            if failed_evaluations:
                print(f"Note: {len(failed_evaluations)} evaluations failed during optimization.")
        
        # Store results
        optimization_results = {
            'method': 'bayesian',
            'best_result': best_result,
            'all_results': all_results,
            'failed_evaluations': failed_evaluations,
            'computation_time': elapsed_time,
            'skopt_result': opt_result,
            'cars_variant': self.cars_variant
        }
        
        return optimization_results
    
    def plot_optimization_results(self, results, save_path=None):
        """
        Plot the results of parameter optimization.
        
        Parameters:
        -----------
        results : dict
            The results dictionary from an optimization method
        save_path : str, optional
            Path to save the plot. If None, the plot will be displayed.
        """
        # Create figure
        plt.figure(figsize=(16, 12))
        
        # Different plot based on optimization method
        method = results.get('method')
        
        if method == 'staged_parameter_scan':
            self._plot_staged_scan_results(results)
        elif method == 'bayesian':
            self._plot_bayesian_results(results)
        else:
            print(f"No specific plotting function for method: {method}")
            return
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def _plot_staged_scan_results(self, results):
        """Plot results from staged parameter scan"""
        all_results = results.get('all_results', [])
        
        if not all_results:
            plt.text(0.5, 0.5, "No results to plot", ha='center', va='center', fontsize=14)
            return
            
        # Separate results by stage
        scan_details = results.get('scan_details', {})
        fine_range = scan_details.get('fine_range', (0, 0))
        
        coarse_results = []
        fine_results = []
        
        for result in all_results:
            max_comp = result.get('max_components_setting')
            prep = result.get('preprocess_method')
            
            if prep == 'autoscaling' and max_comp not in range(fine_range[0], fine_range[1]+1):
                coarse_results.append(result)
            else:
                fine_results.append(result)
        
        # Subplot 1: Parameter heatmap
        plt.subplot(2, 2, 1)
        self._plot_parameter_heatmap(all_results, best_result=results['best_result'])
        
        # Subplot 2: Performance vs components by preprocessing
        plt.subplot(2, 2, 2)
        self._plot_performance_by_components(coarse_results, fine_results, best_result=results['best_result'])
        
        # Subplot 3: Multi-objective score components
        plt.subplot(2, 2, 3)
        self._plot_score_components(results['best_result'])
        
        # Subplot 4: Variable count vs components
        plt.subplot(2, 2, 4)
        self._plot_variable_count(all_results, best_result=results['best_result'])
    
    def _plot_bayesian_results(self, results):
        """Plot results from Bayesian optimization"""
        # Try to get the skopt result
        skopt_result = results.get('skopt_result')
        all_results = results.get('all_results', [])
        
        if not all_results:
            plt.text(0.5, 0.5, "No results to plot", ha='center', va='center', fontsize=14)
            return
            
        # Subplot 1: Parameter effect heatmap
        plt.subplot(2, 2, 1)
        self._plot_parameter_heatmap(all_results, best_result=results['best_result'])
        
        # Subplot 2: Convergence plot
        plt.subplot(2, 2, 2)
        if skopt_result:
            try:
                from skopt.plots import plot_convergence
                plot_convergence(skopt_result)
                plt.title("Bayesian Optimization Convergence", fontsize=14)
            except Exception as e:
                plt.text(0.5, 0.5, f"Could not plot convergence: {str(e)}", 
                       ha='center', va='center', fontsize=12)
                plt.axis('off')
        else:
            plt.text(0.5, 0.5, "No convergence data available", 
                   ha='center', va='center', fontsize=12)
            plt.axis('off')
        
        # Subplot 3: Performance by iteration
        plt.subplot(2, 2, 3)
        self._plot_performance_by_iteration(all_results)
        
        # Subplot 4: Variable count
        plt.subplot(2, 2, 4)
        self._plot_variable_count(all_results, best_result=results['best_result'])
    
    def _plot_parameter_heatmap(self, results, best_result=None):
        """Create a heatmap of parameters vs performance"""
        # Extract parameters and performance metrics
        data = []
        
        for result in results:
            max_comp = result.get('max_components_setting')
            prep = result.get('preprocess_method')
            
            # Get appropriate performance metric
            if self.cars_variant in ['standard', 'corcars']:
                perf = result.get('min_cv_error')  # Lower is better
            else:
                perf = result.get('best_metric_value')  # Higher is better
            
            if max_comp is not None and prep is not None and perf is not None:
                data.append((max_comp, prep, perf))
        
        if not data:
            plt.text(0.5, 0.5, "No valid parameter data", ha='center', va='center', fontsize=12)
            return
            
        # Convert to dataframe for heatmap
        df = pd.DataFrame(data, columns=['max_components', 'preprocess', 'performance'])
        pivot = df.pivot_table(index='max_components', columns='preprocess', values='performance')
        
        # Choose colormap based on metric (lower or higher is better)
        cmap = 'YlGnBu_r' if self.cars_variant in ['standard', 'corcars'] else 'YlGnBu'
        
        # Plot heatmap
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap)
        
        # Highlight best parameters if available
        if best_result:
            best_comp = best_result.get('max_components_setting')
            best_prep = best_result.get('preprocess_method')
            
            # Try to find position in the heatmap
            try:
                i = pivot.index.tolist().index(best_comp)
                j = pivot.columns.tolist().index(best_prep)
                plt.gca().add_patch(plt.Rectangle((j, i), 1, 1, fill=False, 
                                               edgecolor='red', lw=2, alpha=0.7))
            except (ValueError, IndexError):
                pass
        
        plt.title('Parameter Performance Heatmap', fontsize=14)
        plt.xlabel('Preprocessing Method')
        plt.ylabel('Max Components')
    
    def _plot_performance_by_components(self, coarse_results, fine_results, best_result=None):
        """Plot performance vs component count by preprocessing method"""
        if not coarse_results and not fine_results:
            plt.text(0.5, 0.5, "No results to plot", ha='center', va='center', fontsize=12)
            return
        
        # Determine whether metrics are minimized (RMSE) or maximized (accuracy, F1)
        is_minimization = self.cars_variant in ['standard', 'corcars']
        metric_name = 'RMSE' if is_minimization else self.metric.upper()
        
        # Plot coarse scan results
        if coarse_results:
            coarse_x = [r.get('max_components_setting') for r in coarse_results]
            
            if is_minimization:
                coarse_y = [r.get('min_cv_error') for r in coarse_results]
            else:
                coarse_y = [r.get('best_metric_value') for r in coarse_results]
                
            plt.plot(coarse_x, coarse_y, 'o-', color='blue', label='Coarse scan (autoscaling)')
        
        # Plot fine scan results by preprocessing method
        if fine_results:
            prep_methods = set(r.get('preprocess_method') for r in fine_results)
            
            for prep in prep_methods:
                prep_results = [r for r in fine_results if r.get('preprocess_method') == prep]
                
                if prep_results:
                    prep_x = [r.get('max_components_setting') for r in prep_results]
                    
                    if is_minimization:
                        prep_y = [r.get('min_cv_error') for r in prep_results]
                    else:
                        prep_y = [r.get('best_metric_value') for r in prep_results]
                    
                    plt.plot(prep_x, prep_y, 'o--', label=f'Fine scan ({prep})')
        
        # Plot best result
        if best_result:
            best_x = best_result.get('max_components_setting')
            
            if is_minimization:
                best_y = best_result.get('min_cv_error')
            else:
                best_y = best_result.get('best_metric_value')
                
            plt.plot(best_x, best_y, 'r*', markersize=15, label='Best result')
        
        plt.xlabel('Max Components')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} vs Component Count', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    def _plot_score_components(self, best_result):
        """Plot the components of the multi-objective score"""
        if not best_result:
            plt.text(0.5, 0.5, "No result to plot", ha='center', va='center', fontsize=12)
            return
            
        # Get score components
        components = ['Performance', 'Variables', 'Complexity']
        values = [
            best_result.get('norm_perf', 0),
            best_result.get('norm_vars', 0),
            best_result.get('norm_comp', 0)
        ]
        
        bars = plt.bar(components, values)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.xlabel('Score Component')
        plt.ylabel('Normalized Value')
        plt.title('Multi-Objective Score Components', fontsize=14)
        plt.ylim(0, max(values) * 1.2)
    
    def _plot_variable_count(self, results, best_result=None):
        """Plot variable count vs component count"""
        # Extract data
        data = []
        
        for result in results:
            max_comp = result.get('max_components_setting')
            n_vars = result.get('n_selected_vars')
            prep = result.get('preprocess_method')
            
            if max_comp is not None and n_vars is not None and prep is not None:
                data.append((max_comp, prep, n_vars))
        
        if not data:
            plt.text(0.5, 0.5, "No valid variable count data", ha='center', va='center', fontsize=12)
            return
            
        # Plot by preprocessing method
        prep_methods = set(d[1] for d in data)
        
        for prep in prep_methods:
            prep_data = [(d[0], d[2]) for d in data if d[1] == prep]
            x = [d[0] for d in prep_data]
            y = [d[1] for d in prep_data]
            plt.plot(x, y, 'o-', label=f'{prep}')
        
        # Highlight best result
        if best_result:
            best_x = best_result.get('max_components_setting')
            best_y = best_result.get('n_selected_vars')
            plt.plot(best_x, best_y, 'r*', markersize=15, label='Best result')
        
        plt.xlabel('Max Components')
        plt.ylabel('Number of Selected Variables')
        plt.title('Variables Selected vs Component Count', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    def _plot_performance_by_iteration(self, results):
        """Plot performance metric by iteration"""
        # Sort by index
        iterations = range(len(results))
        
        # Get performance metrics based on CARS variant
        if self.cars_variant in ['standard', 'corcars']:
            perf = [r.get('min_cv_error') for r in results]
            plt.plot(iterations, perf, 'o-')
            plt.ylabel('RMSE')
            plt.title('RMSE by Evaluation', fontsize=14)
        else:
            perf = [r.get('best_metric_value') for r in results]
            plt.plot(iterations, perf, 'o-')
            plt.ylabel(self.metric.upper())
            plt.title(f'{self.metric.upper()} by Evaluation', fontsize=14)
        
        plt.xlabel('Evaluation')
        plt.grid(True, alpha=0.3)
        
        # Add best point
        if self.cars_variant in ['standard', 'corcars']:
            best_idx = np.argmin(perf)
        else:
            best_idx = np.argmax(perf)
            
        plt.plot(best_idx, perf[best_idx], 'r*', markersize=15, label='Best result')
        plt.legend()
    
    def plot_best_cars_results(self, result, save_path=None):
        """
        Plot the detailed results from the best CARS run.
        
        Parameters:
        -----------
        result : dict
            The result dictionary from optimization
        save_path : str, optional
            Path to save the plot. If None, the plot will be displayed.
        """
        best_result = result.get('best_result')
        
        if not best_result:
            print("No best result to plot")
            return
            
        if self.plot_func is None:
            # Implement our own plotting function
            self._plot_custom_cars_results(best_result)
        else:
            # Use the built-in plotting function from the CARS implementation
            try:
                # Create a figure that will be saved
                plt.figure(figsize=(12, 15))
                self.plot_func(best_result)
                
                # Save if requested
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    plt.close()
                return
            except Exception as e:
                print(f"Error using built-in plot function: {str(e)}")
                # Fall back to our custom plotting
                print("Falling back to custom plotting...")
                self._plot_custom_cars_results(best_result)
        
        # Save plot if path is provided (for custom plotting)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_custom_cars_results(self, result):
        """
        Custom plotting function for CARS results when built-in function is not available.
        
        Parameters:
        -----------
        result : dict
            The CARS result dictionary
        """
        if 'weight_matrix' not in result:
            print("Cannot plot: weight matrix not found in results")
            return
            
        # Extract data
        weight_matrix = result['weight_matrix']
        if self.cars_variant in ['standard', 'corcars']:
            metric_values = result['cross_validation_errors']
            metric_name = 'RMSE'
        else:
            metric_values = result['metric_values']
            metric_name = result.get('metric', self.metric).upper()
            
        best_iter = result['best_iteration']
        iterations = len(metric_values)
        
        # Calculate number of variables in each iteration
        var_counts = np.count_nonzero(weight_matrix != 0, axis=0)
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot number of selected variables per iteration
        ax1.plot(var_counts, linewidth=2, color='navy')
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Number of variables', fontsize=12)
        ax1.set_title('Variables Selected per Iteration', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Plot cross-validation errors or metric values
        ax2.plot(np.arange(iterations), metric_values, linewidth=2, color='darkgreen')
        ax2.axvline(x=best_iter, color='red', linestyle='--', alpha=0.7,
                    label=f'Best iteration: {best_iter+1}')
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel(metric_name, fontsize=12)
        
        if self.cars_variant in ['standard', 'corcars']:
            ax2.set_title('Cross-Validation Error', fontsize=14)
        else:
            ax2.set_title(f'Classification {metric_name}', fontsize=14)
            
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot regression coefficient paths
        ax3.plot(weight_matrix.T, linewidth=1, alpha=0.6)
        ylims = ax3.get_ylim()
        
        # Create points for vertical line at best iteration
        y_points = np.linspace(ylims[0], ylims[1], 20)
        ax3.plot(np.full(20, best_iter), y_points, 'r*', linewidth=1, alpha=0.8)
        
        ax3.set_xlabel('Iteration', fontsize=12)
        ax3.set_ylabel('Regression coefficients', fontsize=12)
        ax3.set_title('Coefficient Evolution', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # Add confusion matrix for classification if available
        if self.cars_variant == 'classification' and 'confusion_matrix' in result:
            # Create a fourth subplot for confusion matrix
            fig.set_size_inches(12, 20)  # Make figure taller
            ax4 = fig.add_subplot(4, 1, 4)
            
            conf_matrix = result['confusion_matrix']
            target_names = result.get('target_names', [f'Class {i}' for i in range(conf_matrix.shape[0])])
            
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


def example_usage():
    """Example usage of the CARSOptimizer class"""
    # Generate some example data (replace this with your actual data)
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=100, n_features=50, n_informative=10, random_state=42)
    
    # For standard CARS
    optimizer = CARSOptimizer(
        X=X, 
        y=y, 
        cars_variant='standard',
        component_ranges=[5, 10, 15, 20],
        preprocess_options=['center', 'autoscaling', 'pareto'],
        folds=5,
        iterations=50,
        verbose=1
    )
    
    # Run staged parameter scan
    results = optimizer.staged_parameter_scan(alpha=0.6, beta=0.2, gamma=0.2)
    
    # Plot optimization results
    optimizer.plot_optimization_results(results, save_path="standard_cars_optimization.png")
    
    # Plot detailed CARS results for the best parameters
    optimizer.plot_best_cars_results(results, save_path="standard_cars_best_result.png")
    
    # For classification
    # Generate classification data (replace with your actual data)
    from sklearn.datasets import make_classification
    X_cls, y_cls = make_classification(n_samples=100, n_features=50, n_informative=10, n_classes=3, random_state=42)
    
    # Create one-hot encoded target
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(sparse=False)
    y_onehot = enc.fit_transform(y_cls.reshape(-1, 1))
    
    # Create optimizer for classification with one-hot encoding
    cls_optimizer = CARSOptimizer(
        X=X_cls,
        y=y_onehot,  # Use one-hot encoded target
        cars_variant='classification',
        component_ranges=[5, 10, 15],
        preprocess_options=['center', 'autoscaling'],
        folds=5,
        iterations=50,
        verbose=1,
        encoding='onehot',  # Specify encoding
        metric='f1'  # Specify metric
    )
    
    # Run Bayesian optimization
    cls_results = cls_optimizer.bayesian_optimization(n_calls=15)
    
    # Plot results
    if cls_results:
        cls_optimizer.plot_optimization_results(cls_results, save_path="classification_cars_optimization.png")
        cls_optimizer.plot_best_cars_results(cls_results, save_path="classification_cars_best_result.png")
    
    # For CorCARS
    corcars_optimizer = CARSOptimizer(
        X=X,
        y=y,
        cars_variant='corcars',
        component_ranges=[5, 10, 15, 20],
        preprocess_options=['center', 'autoscaling', 'pareto'],
        folds=5,
        iterations=50,
        verbose=1,
        use_correlation=True,  # CorCARS specific parameter
        alpha=0.25  # CorCARS specific parameter
    )
    
    # Run staged parameter scan
    corcars_results = corcars_optimizer.staged_parameter_scan()
    
    # Plot results
    corcars_optimizer.plot_optimization_results(corcars_results)
    
    return optimizer, cls_optimizer, corcars_optimizer

# Run the example if this file is executed directly
if __name__ == "__main__":
    example_usage()
