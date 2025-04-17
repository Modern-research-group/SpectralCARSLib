import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import sys
import importlib
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from joblib import Parallel, delayed
import warnings
import traceback


class SimpleCARSOptimizer:
    """
    A user-friendly all-in-one optimizer and model builder for CARS variable selection.
    
    This optimizer provides:
    1. Simple API with sensible defaults
    2. Pre-configured recipes for common use cases
    3. Clear error messages and recovery strategies
    4. Progress reporting with time estimates
    5. All-in-one workflow from optimization to final model
    
    Examples:
    ---------
    >>> # Basic usage with auto-configuration
    >>> optimizer = SimpleCARSOptimizer(X, y)
    >>> result = optimizer.run()  # All-in-one optimization and model building
    >>> selected_vars = result['selected_variables']  # Get selected variables
    >>> final_model = result['model']  # Get the final fitted model
    >>> 
    >>> # Make predictions with the final model
    >>> y_pred = optimizer.predict(X_new)
    >>> 
    >>> # Specify a recipe for specific scenarios
    >>> optimizer = SimpleCARSOptimizer(X, y)
    >>> result = optimizer.run(recipe="fast")  # Quick results
    >>> result = optimizer.run(recipe="thorough")  # Comprehensive search
    >>> 
    >>> # Classification with one-hot encoding
    >>> optimizer = SimpleCARSOptimizer(X, y_onehot, task="classification", encoding="onehot")
    >>> result = optimizer.run()
    """
    
    # Define recipes - pre-configured parameter sets for common scenarios
    RECIPES = {
        "fast": {
            "description": "Quick optimization with minimal computation",
            "config": {
                "component_ranges": [5, 10],
                "preprocess_options": ["autoscaling"],
                "iterations": 30,
                "n_jobs": -1,
                "use_correlation": False,
                "perform_stability": False
            }
        },
        "default": {
            "description": "Balanced optimization with reasonable computation time",
            "config": {
                "component_ranges": [5, 10, 15, 20],
                "preprocess_options": ["center", "autoscaling"],
                "iterations": 50,
                "n_jobs": -1,
                "use_correlation": True,
                "perform_stability": False
            }
        },
        "thorough": {
            "description": "Comprehensive optimization exploring many parameters",
            "config": {
                "component_ranges": [5, 10, 15, 20, 25, 30],
                "preprocess_options": ["center", "autoscaling", "pareto", "minmax"],
                "iterations": 100,
                "n_jobs": -1,
                "use_correlation": True,
                "perform_stability": True,
                "n_bootstrap": 5
            }
        },
        "robust": {
            "description": "Stability-focused optimization resistant to overfitting",
            "config": {
                "component_ranges": [5, 10, 15, 20],
                "preprocess_options": ["autoscaling", "pareto"],
                "iterations": 50,
                "n_jobs": -1,
                "use_correlation": True,
                "perform_stability": True,
                "n_bootstrap": 10,
                "stability_threshold": 0.7
            }
        },
        "classification": {
            "description": "Optimized for classification tasks",
            "config": {
                "component_ranges": [3, 5, 8, 10, 15],
                "preprocess_options": ["autoscaling"],
                "iterations": 50,
                "n_jobs": -1,
                "use_correlation": True,
                "encoding": "ordinal",
                "metric": "f1"
            }
        }
    }
    
    def __init__(self, X, y, task="auto", verbose=1, encoding=None, metric=None):
        """
        Initialize the optimizer.
        
        Parameters:
        -----------
        X : array-like
            The predictor matrix
        y : array-like
            Target variable (vector for regression, 
                            vector or matrix for classification)
        task : str, default="auto"
            Task type: "auto", "regression" or "classification".
            If "auto", will try to detect based on y structure.
        verbose : int, default=1
            Verbosity level (0=silent, 1=normal, 2=detailed)
        encoding : str, optional
            For classification, "ordinal" or "onehot"
        metric : str, optional
            For classification, "accuracy", "f1", or "auc"
        """
        self.X_original = X
        self.y_original = y
        self.verbose = verbose
        
        # Convert to numpy arrays if needed
        self.X = X.values if hasattr(X, 'values') else np.array(X)
        self.y = y.values if hasattr(y, 'values') else np.array(y)
        
        # Auto-detect task if needed
        if task.lower() == "auto":
            # Check if y looks like classification data
            if len(self.y.shape) > 1 and self.y.shape[1] > 1:
                # Multi-column y is typically one-hot encoded classification
                self.task = "classification"
                self._encoding = "onehot"
                if self.verbose:
                    print("Detected classification task with one-hot encoding")
            elif np.issubdtype(self.y.dtype, np.integer) or (np.unique(self.y).size / len(self.y) < 0.05):
                # Integer y or few unique values relative to size suggests classification
                self.task = "classification"
                self._encoding = "ordinal"
                if self.verbose:
                    print("Detected classification task with ordinal encoding")
            else:
                # Default to regression
                self.task = "regression"
                if self.verbose:
                    print("Detected regression task")
        else:
            # Use specified task
            self.task = task.lower()
        
        # Make sure task is valid
        if self.task not in ["regression", "classification"]:
            raise ValueError(f"Task '{self.task}' not supported. Use 'regression' or 'classification'.")
        
        # Set classification-specific parameters
        if self.task == "classification":
            self._encoding = encoding or ("onehot" if len(self.y.shape) > 1 and self.y.shape[1] > 1 else "ordinal")
            self._metric = metric or ("f1" if self._encoding == "onehot" else "accuracy")
        
        # Validate data
        self._validate_data()
        
        # Store original dimensions
        self.n_samples, self.n_features = self.X.shape
        
        # Initialize result storage
        self.last_result = None
        self.best_result = None
        self.all_results = []
        self.final_model = None
        self.preprocessor = None
        
        # Try to import CARS implementations
        self.cars_modules = self._import_cars_implementations()
    
    def _validate_data(self):
        """Validate input data for basic issues"""
        # Check for NaN values
        if np.isnan(self.X).any():
            raise ValueError("Input X contains NaN values. Please handle missing data before optimization.")
        
        if np.isnan(self.y).any():
            raise ValueError("Target y contains NaN values. Please handle missing data before optimization.")
        
        # Check for inf values
        if np.isinf(self.X).any():
            raise ValueError("Input X contains infinite values. Please handle these values before optimization.")
            
        if np.isinf(self.y).any():
            raise ValueError("Target y contains infinite values. Please handle these values before optimization.")
        
        # Check dimensions
        if len(self.X.shape) != 2:
            raise ValueError(f"Input X must be 2-dimensional, got shape {self.X.shape}")
        
        if len(self.y.shape) == 0:
            raise ValueError("Target y must not be a scalar value")
            
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(f"X and y must have the same number of samples. X has {self.X.shape[0]}, y has {self.y.shape[0]}")
            
        # Check for classification-specific issues
        if self.task == "classification":
            if len(self.y.shape) == 1:
                # Check for ordinal class values
                unique_classes = np.unique(self.y)
                if len(unique_classes) < 2:
                    raise ValueError(f"Classification task requires at least 2 classes, found {len(unique_classes)}")
                
                # Warn for sparse classes
                for cls in unique_classes:
                    count = np.sum(self.y == cls)
                    if count < 5:
                        warnings.warn(f"Class {cls} has only {count} samples, which may be too few for reliable model building.")
            
            elif len(self.y.shape) == 2:
                # Check for one-hot encoding
                row_sums = np.sum(self.y, axis=1)
                if not np.allclose(row_sums, 1.0):
                    warnings.warn("One-hot encoded y should have exactly one 1 per row. Your data may not be properly one-hot encoded.")
    
    def _import_cars_implementations(self):
        """Try to import CARS implementations and return available modules"""
        # This function tries multiple import strategies
        
        cars_modules = {
            "standard": None,
            "corcars": None,
            "classification": None
        }
        
        # First try: direct import from SpectralCARSLib
        try:
            from SpectralCARSLib import (
                competitive_adaptive_sampling, 
                competitive_adaptive_reweighted_sampling,
                competitive_adaptive_sampling_classification
            )
            cars_modules["standard"] = competitive_adaptive_sampling
            cars_modules["corcars"] = competitive_adaptive_reweighted_sampling
            cars_modules["classification"] = competitive_adaptive_sampling_classification
            
            if self.verbose:
                print("Successfully imported CARS implementations from SpectralCARSLib")
            return cars_modules
            
        except ImportError:
            if self.verbose:
                print("SpectralCARSLib direct import failed, trying alternative methods...")
        
        # Second try: import by individual module name
        module_names = {
            "standard": ["cars", "CARS", "spectralcarslib.cars"],
            "corcars": ["corcars", "CorCARS", "spectralcarslib.corcars"],
            "classification": ["classification", "CARS_classification", "spectralcarslib.classification"]
        }
        
        function_names = {
            "standard": "competitive_adaptive_sampling",
            "corcars": "competitive_adaptive_reweighted_sampling",
            "classification": "competitive_adaptive_sampling_classification"
        }
        
        for variant, names in module_names.items():
            for name in names:
                try:
                    module = importlib.import_module(name)
                    func_name = function_names[variant]
                    if hasattr(module, func_name):
                        cars_modules[variant] = getattr(module, func_name)
                        if self.verbose:
                            print(f"Successfully imported {variant} CARS from module {name}")
                        break
                except (ImportError, AttributeError):
                    continue
        
        # Check if we found any implementation
        if all(v is None for v in cars_modules.values()):
            warnings.warn(
                "Could not import any CARS implementations. "
                "Make sure SpectralCARSLib is installed or provide implementation functions directly."
            )
            
        return cars_modules
    
    def run(self, recipe=None, build_final_model=True, **kwargs):
        """
        All-in-one function to optimize CARS parameters and build the final model.
        
        Parameters:
        -----------
        recipe : str, optional
            Name of pre-configured recipe: "fast", "default", "thorough", "robust", "classification"
            If None, will auto-select based on dataset characteristics
        build_final_model : bool, default=True
            Whether to build the final model with optimized parameters
        **kwargs : dict
            Custom parameters that override recipe defaults
            
        Returns:
        --------
        dict : Complete results including:
            - All optimization results
            - Selected variables
            - Final model (if build_final_model=True)
            - Preprocessor (if build_final_model=True)
        """
        # Step 1: Run optimization
        optimization_result = self.optimize(recipe=recipe, **kwargs)
        
        # Step 2: Build final model if requested
        if build_final_model:
            if self.verbose:
                print("\nBuilding final model with optimized parameters...")
            
            self.build_final_model(optimization_result)
            
            # Add model to result
            optimization_result['model'] = self.final_model
            optimization_result['preprocessor'] = self.preprocessor
            
            if self.verbose:
                print("Final model built successfully.")
        
        return optimization_result
    
    def get_recommended_recipe(self):
        """Get the recommended recipe based on dataset characteristics"""
        n_samples, n_features = self.X.shape
        
        if n_samples < 50 or n_features < 50:
            # Small dataset can use thorough recipe
            return "thorough"
        elif n_samples > 1000 or n_features > 500:
            # Large dataset needs fast recipe
            return "fast"
        elif self.task == "classification":
            # Classification tasks use classification recipe
            return "classification"
        else:
            # Default for medium datasets
            return "default"
    
    def print_available_recipes(self):
        """Print information about all available optimization recipes"""
        print("Available optimization recipes:")
        print("------------------------------")
        
        for name, recipe in self.RECIPES.items():
            print(f"\n{name}: {recipe['description']}")
            print("  Configuration:")
            for key, value in recipe['config'].items():
                print(f"    {key}: {value}")
    
    def optimize(self, recipe=None, **kwargs):
        """
        Optimize CARS parameters using the specified recipe or custom parameters.
        
        Parameters:
        -----------
        recipe : str, optional
            Name of pre-configured recipe: "fast", "default", "thorough", "robust", "classification"
            If None, will auto-select based on dataset characteristics
        **kwargs : dict
            Custom parameters that override recipe defaults
            
        Returns:
        --------
        dict : Best optimization result
        """
        start_time = time.time()
        
        # Get recipe configuration
        config = self._get_configuration(recipe, **kwargs)
        
        if self.verbose:
            self._print_optimization_header(config)
        
        # Define CARS variant based on task
        if self.task == "regression":
            if config["use_correlation"]:
                cars_variant = "corcars"
            else:
                cars_variant = "standard"
        else:  # classification
            cars_variant = "classification"
        
        # Check if we have the required implementation
        if self.cars_modules[cars_variant] is None:
            raise ImportError(
                f"Could not find implementation for {cars_variant} CARS. "
                "Make sure SpectralCARSLib is installed correctly."
            )
        
        # Create optimization stages based on configuration
        if config.get("perform_stability", False):
            stages = self._create_stability_based_stages(config, cars_variant)
        else:
            stages = self._create_standard_stages(config, cars_variant)
        
        # Execute optimization stages
        try:
            all_results = []
            
            for stage_idx, stage in enumerate(stages):
                if self.verbose:
                    print(f"\nStage {stage_idx+1}/{len(stages)}: {stage['name']}")
                
                # Process the stage
                stage_results = self._process_stage(stage, config, cars_variant)
                all_results.extend(stage_results)
                
                if self.verbose and len(stage_results) > 0:
                    # Report best result from this stage
                    best_stage_result = self._get_best_result(stage_results, cars_variant)
                    self._print_stage_result(best_stage_result, cars_variant)
                
                # Early stopping if a clearly superior result is found
                if self._check_early_stopping(stage_results, all_results, cars_variant):
                    if self.verbose:
                        print("\nEarly stopping: Found clearly superior parameter set")
                    break
            
            # Find overall best result
            if all_results:
                best_result = self._get_best_result(all_results, cars_variant)
                self.best_result = best_result
                self.all_results = all_results
                
                # Final optimization report
                if self.verbose:
                    self._print_final_result(best_result, cars_variant, time.time() - start_time)
                
                return best_result
            else:
                raise RuntimeError("No valid results from optimization. All parameter combinations failed.")
                
        except KeyboardInterrupt:
            # Handle user interruption gracefully
            if self.verbose:
                print("\nOptimization interrupted by user.")
            
            # Return best result so far if available
            if all_results:
                best_result = self._get_best_result(all_results, cars_variant)
                self.best_result = best_result
                self.all_results = all_results
                
                if self.verbose:
                    print("Returning best result found so far:")
                    self._print_result_summary(best_result, cars_variant)
                
                return best_result
            else:
                print("No valid results obtained before interruption.")
                return None
    
    def build_final_model(self, optimization_result=None):
        """
        Build the final model using the optimized parameters.
        
        Parameters:
        -----------
        optimization_result : dict, optional
            Result from optimization. If None, uses stored best_result.
            
        Returns:
        --------
        tuple : (model, preprocessor)
            The final fitted model and preprocessor
        """
        # Get optimization result
        if optimization_result is None:
            if self.best_result is None:
                raise ValueError("No optimization results available. Run optimize() first.")
            optimization_result = self.best_result
        
        # Extract key information
        selected_vars = optimization_result['selected_variables']
        optimal_components = optimization_result['optimal_components']
        preprocess_method = optimization_result['preprocess_method']
        
        if len(selected_vars) == 0:
            raise ValueError("No variables were selected in the optimization result.")
        
        # Create preprocessor based on method
        if preprocess_method == 'autoscaling':
            preprocessor = StandardScaler()
        elif preprocess_method == 'center':
            # Only center, no scaling
            preprocessor = StandardScaler(with_std=False)
        else:
            # For other methods, we'll create a custom preprocessor
            preprocessor = self._create_custom_preprocessor(preprocess_method)
        
        # Select variables and preprocess data
        X_selected = self.X[:, selected_vars]
        X_processed = preprocessor.fit_transform(X_selected)
        
        # Create and fit model
        if self.task == "regression":
            model = PLSRegression(n_components=optimal_components)
            model.fit(X_processed, self.y)
        else:  # classification
            model = PLSRegression(n_components=optimal_components)
            
            # Handle different encodings
            if self._encoding == "onehot":
                # Fit with one-hot encoded targets
                model.fit(X_processed, self.y)
            else:  # ordinal
                # Convert to 2D array for sklearn
                y_2d = self.y.reshape(-1, 1) if len(self.y.shape) == 1 else self.y
                model.fit(X_processed, y_2d)
        
        # Store model and preprocessor
        self.final_model = model
        self.preprocessor = preprocessor
        
        # Store information needed for prediction
        self._model_info = {
            'selected_vars': selected_vars,
            'optimal_components': optimal_components,
            'preprocess_method': preprocess_method,
            'task': self.task,
            'encoding': self._encoding if self.task == "classification" else None
        }
        
        return model, preprocessor
    
    def predict(self, X_new):
        """
        Make predictions using the final model.
        
        Parameters:
        -----------
        X_new : array-like
            New data to predict on
            
        Returns:
        --------
        array : Predictions
        """
        if self.final_model is None or self.preprocessor is None:
            raise ValueError(
                "No final model available. Run build_final_model() or run(build_final_model=True) first."
            )
        
        # Convert to numpy array if needed
        X_new = X_new.values if hasattr(X_new, 'values') else np.array(X_new)
        
        # Check dimensions
        if X_new.shape[1] != self.X.shape[1]:
            raise ValueError(
                f"Input X has {X_new.shape[1]} features but the model was trained with {self.X.shape[1]} features."
            )
        
        # Select variables and preprocess
        selected_vars = self._model_info['selected_vars']
        X_selected = X_new[:, selected_vars]
        X_processed = self.preprocessor.transform(X_selected)
        
        # Make predictions
        raw_predictions = self.final_model.predict(X_processed)
        
        # Process predictions based on task
        if self.task == "regression":
            # Return raw predictions
            # Check dimensionality of predictions
            if len(raw_predictions.shape) == 1:
                # Already 1D, return as is
                return raw_predictions
            else:
                # It's 2D, check if we should flatten it
                return raw_predictions.ravel() if raw_predictions.shape[1] == 1 else raw_predictions
        else:  # classification
            if self._encoding == "onehot":
                # Return class with highest probability
                return np.argmax(raw_predictions, axis=1)
            else:  # ordinal
                # Round to nearest class and ensure valid range
                n_classes = len(np.unique(self.y))
                return np.round(raw_predictions).clip(0, n_classes-1).astype(int).ravel()
    
    def predict_proba(self, X_new):
        """
        For classification, predict class probabilities.
        
        Parameters:
        -----------
        X_new : array-like
            New data to predict on
            
        Returns:
        --------
        array : Class probabilities
        """
        if self.task != "classification":
            raise ValueError("predict_proba() is only available for classification tasks.")
            
        if self.final_model is None or self.preprocessor is None:
            raise ValueError(
                "No final model available. Run build_final_model() or run(build_final_model=True) first."
            )
        
        # Convert to numpy array if needed
        X_new = X_new.values if hasattr(X_new, 'values') else np.array(X_new)
        
        # Check dimensions
        if X_new.shape[1] != self.X.shape[1]:
            raise ValueError(
                f"Input X has {X_new.shape[1]} features but the model was trained with {self.X.shape[1]} features."
            )
        
        # Select variables and preprocess
        selected_vars = self._model_info['selected_vars']
        X_selected = X_new[:, selected_vars]
        X_processed = self.preprocessor.transform(X_selected)
        
        # Make predictions
        raw_predictions = self.final_model.predict(X_processed)
        
        # Convert to probabilities
        if self._encoding == "onehot":
            # For one-hot encoding, convert to probabilities using softmax
            exp_preds = np.exp(raw_predictions - np.max(raw_predictions, axis=1, keepdims=True))
            probabilities = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
            return probabilities
        else:  # ordinal
            # For binary classification, convert to probability using sigmoid
            if len(np.unique(self.y)) == 2:
                probabilities = 1 / (1 + np.exp(-raw_predictions))
                return np.column_stack((1 - probabilities, probabilities)).ravel()
            else:
                # For multi-class ordinal, convert to "soft" probabilities
                n_classes = len(np.unique(self.y))
                probabilities = np.zeros((len(raw_predictions), n_classes))
                raw_preds = raw_predictions.ravel()
                
                # Calculate "distances" to each class
                for i, pred in enumerate(raw_preds):
                    # Calculate distance from pred to each class center
                    distances = np.array([abs(pred - cls) for cls in range(n_classes)])
                    # Convert distances to probabilities (closer = higher probability)
                    sim = 1 / (1 + distances)
                    probabilities[i] = sim / sim.sum()  # Normalize
                
                return probabilities
    
    def _create_custom_preprocessor(self, method):
        """Create a custom preprocessor for non-standard preprocessing methods"""
        if method not in ['pareto', 'minmax', 'robust', 'unilength', 'none']:
            raise ValueError(f"Unsupported preprocessing method: {method}")
        
        # Create a custom preprocessor class
        class CustomPreprocessor:
            def __init__(self, method):
                self.method = method
                self.mean_ = None
                self.scale_ = None
            
            def fit(self, X, y=None):
                if self.method == 'pareto':
                    self.mean_ = np.mean(X, axis=0)
                    self.scale_ = np.sqrt(np.std(X, axis=0, ddof=1))
                elif self.method == 'minmax':
                    self.mean_ = np.min(X, axis=0)
                    self.scale_ = np.ptp(X, axis=0)  # peak to peak (max-min)
                elif self.method == 'robust':
                    self.mean_ = np.median(X, axis=0)
                    q1 = np.percentile(X, 25, axis=0)
                    q3 = np.percentile(X, 75, axis=0)
                    self.scale_ = q3 - q1  # Interquartile range (IQR)
                elif self.method == 'unilength':
                    self.mean_ = np.mean(X, axis=0)
                    X_centered = X - self.mean_
                    self.scale_ = np.sqrt(np.sum(X_centered**2, axis=0))
                else:  # 'none'
                    self.mean_ = np.zeros(X.shape[1])
                    self.scale_ = np.ones(X.shape[1])
                
                # Handle zero variance
                self.scale_[self.scale_ == 0] = 1.0
                
                return self
            
            def transform(self, X):
                return (X - self.mean_) / self.scale_
            
            def fit_transform(self, X, y=None):
                return self.fit(X).transform(X)
        
        return CustomPreprocessor(method)
    
    def _get_configuration(self, recipe=None, **kwargs):
        """Get configuration from recipe and override with custom parameters"""
        # Auto-select recipe if not specified
        if recipe is None:
            recipe = self.get_recommended_recipe()
            if self.verbose:
                print(f"Auto-selected recipe: '{recipe}'")
        
        # Check if recipe exists
        if recipe not in self.RECIPES:
            raise ValueError(
                f"Recipe '{recipe}' not found. Available recipes: {', '.join(self.RECIPES.keys())}"
            )
        
        # Get base configuration from recipe
        config = self.RECIPES[recipe]["config"].copy()
        
        # Override with custom parameters
        config.update(kwargs)
        
        # Set defaults for classification task
        if self.task == "classification":
            config["encoding"] = config.get("encoding", self._encoding)
            config["metric"] = config.get("metric", self._metric)
        
        # Validate configuration
        self._validate_configuration(config)
        
        return config
    
    def _validate_configuration(self, config):
        """Validate configuration parameters"""
        # Check component ranges
        if "component_ranges" in config:
            if not isinstance(config["component_ranges"], (list, tuple)) or not all(isinstance(c, int) for c in config["component_ranges"]):
                raise ValueError("component_ranges must be a list of integers")
            
            # Check if component ranges are reasonable for the dataset
            max_possible = min(self.n_samples, self.n_features)
            if any(c > max_possible for c in config["component_ranges"]):
                warnings.warn(
                    f"Some component values exceed the maximum possible ({max_possible}) for this dataset. "
                    "These values will be limited during optimization."
                )
        
        # Check preprocessing options
        if "preprocess_options" in config:
            valid_preprocess = ["center", "autoscaling", "pareto", "minmax", "robust", "unilength", "none"]
            invalid_options = [p for p in config["preprocess_options"] if p not in valid_preprocess]
            if invalid_options:
                raise ValueError(
                    f"Invalid preprocessing option(s): {', '.join(invalid_options)}. "
                    f"Valid options are: {', '.join(valid_preprocess)}"
                )
        
        # Check iterations
        if "iterations" in config and (not isinstance(config["iterations"], int) or config["iterations"] <= 0):
            raise ValueError("iterations must be a positive integer")
        
        # Check n_jobs
        if "n_jobs" in config and not isinstance(config["n_jobs"], int):
            raise ValueError("n_jobs must be an integer")
        
        # Check classification-specific parameters
        if self.task == "classification":
            if "encoding" in config and config["encoding"] not in ["ordinal", "onehot"]:
                raise ValueError("encoding must be 'ordinal' or 'onehot'")
            
            if "metric" in config and config["metric"] not in ["accuracy", "f1", "auc"]:
                raise ValueError("metric must be 'accuracy', 'f1', or 'auc'")
    
    def _create_standard_stages(self, config, cars_variant):
        """Create standard optimization stages"""
        # Stage 1: Coarse scan with fixed preprocessing
        stage1 = {
            "name": "Coarse parameter scan",
            "description": "Evaluating different max_components values with fixed preprocessing",
            "parameters": []
        }
        
        # Use autoscaling as default preprocessing for first stage
        default_preprocess = "autoscaling"
        if default_preprocess not in config["preprocess_options"]:
            default_preprocess = config["preprocess_options"][0]
        
        # Add parameter combinations for stage 1
        for max_comp in config["component_ranges"]:
            stage1["parameters"].append({
                "max_components": max_comp,
                "preprocess": default_preprocess
            })
        
        # Stage 2: Fine-tuned scan with different preprocessing options
        stage2 = {
            "name": "Fine-tuned parameter scan",
            "description": "Evaluating different preprocessing methods with promising component values",
            "parameters": []
        }
        
        # For stage 2, we'll use a subset of component values and all preprocessing options
        # This will be filled after stage 1 completes
        
        # For now, return just stage 1
        return [stage1, stage2]
    
    def _create_stability_based_stages(self, config, cars_variant):
        """Create stability-based optimization stages"""
        # Stage 1: Component stability assessment
        stage1 = {
            "name": "Component stability assessment",
            "description": "Assessing component stability with bootstrap sampling",
            "parameters": []
        }
        
        # For stability assessment, we use a single preprocessing method first
        default_preprocess = "autoscaling"
        if default_preprocess not in config["preprocess_options"]:
            default_preprocess = config["preprocess_options"][0]
        
        # Create a special parameter entry for stability assessment
        stage1["parameters"].append({
            "stability_assessment": True,
            "component_ranges": config["component_ranges"],
            "preprocess": default_preprocess,
            "n_bootstrap": config.get("n_bootstrap", 5),
            "stability_threshold": config.get("stability_threshold", 0.6)
        })
        
        # Stage 2: Fine-tuned scan with stable components and different preprocessing
        stage2 = {
            "name": "Fine-tuned parameter scan with stable components",
            "description": "Evaluating different preprocessing methods with stable component values",
            "parameters": []
        }
        
        # For stage 2, we'll use stable component values and all preprocessing options
        # This will be filled after stage 1 completes
        
        return [stage1, stage2]
    
    def _process_stage(self, stage, config, cars_variant):
        """Process an optimization stage"""
        stage_results = []
        
        # Handle special case for stability assessment
        if len(stage["parameters"]) == 1 and "stability_assessment" in stage["parameters"][0]:
            stability_params = stage["parameters"][0]
            stage_results = self._perform_stability_assessment(
                stability_params, 
                config,
                cars_variant
            )
            return stage_results
        
        # Normal parameter grid evaluation
        for param_idx, params in enumerate(stage["parameters"]):
            if self.verbose:
                self._print_parameter_progress(param_idx, len(stage["parameters"]), params)
            
            try:
                result = self._run_cars(params, config, cars_variant)
                
                # Add parameter info to result
                result.update({
                    "max_components_setting": params["max_components"],
                    "preprocess_method": params["preprocess"],
                    "n_selected_vars": len(result["selected_variables"])
                })
                
                stage_results.append(result)
                self.last_result = result
                
            except Exception as e:
                if self.verbose:
                    print(f"Error with parameters {params}: {str(e)}")
                    # Print more details in verbose mode
                    if self.verbose > 1:
                        print(traceback.format_exc())
        
        # If this is stage 1, update stage 2 with promising component values
        if stage["name"] == "Coarse parameter scan" and len(stage_results) > 0:
            next_stage = self._get_next_stage(stage)
            if next_stage:
                self._update_stage2_parameters(next_stage, stage_results, config, cars_variant)
        
        return stage_results
    
    def _perform_stability_assessment(self, stability_params, config, cars_variant):
        """Perform component stability assessment with bootstrap sampling"""
        if self.verbose:
            print(f"Performing bootstrap stability assessment with {stability_params['n_bootstrap']} samples")
        
        # Extract parameters
        component_ranges = stability_params["component_ranges"]
        preprocess = stability_params["preprocess"]
        n_bootstrap = stability_params["n_bootstrap"]
        stability_threshold = stability_params["stability_threshold"]
        
        # Storage for results
        bootstrap_results = []
        
        # Function to run for each bootstrap sample
        def run_bootstrap(boot_idx, seed):
            # Create bootstrap sample
            np.random.seed(seed)
            indices = np.random.choice(self.n_samples, self.n_samples, replace=True)
            
            # Extract bootstrap data
            X_boot = self.X[indices]
            if len(self.y.shape) == 1:
                y_boot = self.y[indices]
            else:
                y_boot = self.y[indices, :]
            
            boot_results = []
            
            # Run CARS for each component value
            for max_comp in component_ranges:
                try:
                    # Prepare parameters
                    boot_params = {
                        "max_components": max_comp,
                        "preprocess": preprocess
                    }
                    
                    # Update with other necessary config
                    for key in ["iterations", "encoding", "metric"]:
                        if key in config:
                            boot_params[key] = config[key]
                    
                    # Set n_jobs=1 to avoid nested parallelism issues
                    boot_params["n_jobs"] = 1
                    boot_params["verbose"] = 0  # Silence individual runs
                    
                    # Run CARS
                    result = self._run_cars_function(boot_params, cars_variant)
                    
                    # Record key information
                    boot_results.append({
                        "bootstrap_idx": boot_idx,
                        "max_components": max_comp,
                        "optimal_components": result["optimal_components"],
                        "min_cv_error": result.get("min_cv_error"),
                        "best_metric_value": result.get("best_metric_value"),
                        "n_selected_vars": len(result["selected_variables"])
                    })
                    
                except Exception as e:
                    if self.verbose > 1:
                        print(f"Bootstrap {boot_idx}, max_comp={max_comp} failed: {str(e)}")
            
            return boot_results
        
        # Run bootstrap samples (sequentially if n_jobs=1, otherwise parallel)
        if config.get("n_jobs", -1) == 1:
            for boot_idx in range(n_bootstrap):
                if self.verbose:
                    print(f"  Running bootstrap sample {boot_idx+1}/{n_bootstrap}")
                results = run_bootstrap(boot_idx, seed=boot_idx+42)
                bootstrap_results.extend(results)
        else:
            # Run in parallel
            bootstrap_samples = Parallel(n_jobs=config.get("n_jobs", -1), verbose=self.verbose)(
                delayed(run_bootstrap)(i, seed=i+42) for i in range(n_bootstrap)
            )
            
            # Flatten results
            for results in bootstrap_samples:
                bootstrap_results.extend(results)
        
        # Analyze bootstrap results for stability
        stability_analysis = self._analyze_component_stability(
            bootstrap_results, 
            stability_threshold
        )
        
        # Get stage 2 and update with stable component values
        next_stage = self._get_next_stage({"name": "Component stability assessment"})
        if next_stage:
            self._update_stage2_with_stable_components(
                next_stage, 
                stability_analysis, 
                config
            )
        
        # Run a final CARS with the most stable component value
        if stability_analysis["stable_components"]:
            best_max_comp = stability_analysis["most_stable_component"]
            
            if self.verbose:
                print(f"Running final CARS with most stable max_components={best_max_comp}")
            
            # Run with most stable component value
            final_params = {
                "max_components": best_max_comp,
                "preprocess": preprocess
            }
            
            try:
                result = self._run_cars(final_params, config, cars_variant)
                
                # Add stability information to result
                result.update({
                    "max_components_setting": best_max_comp,
                    "preprocess_method": preprocess,
                    "n_selected_vars": len(result["selected_variables"]),
                    "stability_analysis": stability_analysis
                })
                
                self.last_result = result
                return [result]
                
            except Exception as e:
                if self.verbose:
                    print(f"Error with stable component value {best_max_comp}: {str(e)}")
                    if self.verbose > 1:
                        print(traceback.format_exc())
                
                # Return empty list, next stage will try more combinations
                return []
        else:
            if self.verbose:
                print("No stable component values found. Moving to standard optimization.")
            
            # Return empty list, next stage will try more combinations
            return []
    
    def _analyze_component_stability(self, bootstrap_results, stability_threshold):
        """Analyze bootstrap results for component stability"""
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(bootstrap_results)
        
        if len(df) == 0:
            return {
                "stable_components": False,
                "most_stable_component": None,
                "stability_scores": {}
            }
        
        # Group by max_components
        stability_scores = {}
        
        for max_comp in df["max_components"].unique():
            # Get results for this max_components value
            comp_df = df[df["max_components"] == max_comp]
            
            # Count frequency of each optimal component value
            opt_counts = comp_df["optimal_components"].value_counts(normalize=True)
            
            if len(opt_counts) > 0:
                # Get most frequent optimal component
                most_frequent = opt_counts.index[0]
                frequency = opt_counts.iloc[0]
                
                stability_scores[max_comp] = {
                    "most_frequent_component": most_frequent,
                    "frequency": frequency,
                    "stable": frequency >= stability_threshold
                }
        
        # Find most stable component
        stable_max_comps = [max_comp for max_comp, info in stability_scores.items() 
                           if info["stable"]]
        
        if stable_max_comps:
            # Sort by stability score (higher is better)
            sorted_stable = sorted(
                stable_max_comps,
                key=lambda x: stability_scores[x]["frequency"],
                reverse=True
            )
            most_stable = sorted_stable[0]
        else:
            # No stable components found, use highest stability score
            if stability_scores:
                sorted_by_stability = sorted(
                    stability_scores.keys(),
                    key=lambda x: stability_scores[x]["frequency"],
                    reverse=True
                )
                most_stable = sorted_by_stability[0]
            else:
                most_stable = None
        
        return {
            "stable_components": bool(stable_max_comps),
            "most_stable_component": most_stable,
            "stability_scores": stability_scores
        }
    
    def _update_stage2_parameters(self, stage2, stage1_results, config, cars_variant):
        """Update stage 2 parameters based on stage 1 results"""
        if not stage1_results:
            # If stage 1 had no valid results, use all parameter combinations
            for max_comp in config["component_ranges"]:
                for prep in config["preprocess_options"]:
                    stage2["parameters"].append({
                        "max_components": max_comp,
                        "preprocess": prep
                    })
            return
        
        # Find promising component values (top 2)
        sorted_results = self._sort_results(stage1_results, cars_variant)
        best_result = sorted_results[0]
        
        # Get the best max_components value
        best_max_comp = best_result["max_components_setting"]
        
        # Define range around best max_components
        min_comp = max(best_max_comp - 5, min(config["component_ranges"]))
        max_comp = min(best_max_comp + 5, max(config["component_ranges"]))
        
        # Ensure we have at least 3 values if possible
        if max_comp - min_comp < 2 and len(config["component_ranges"]) > 2:
            # Expand search space
            all_comps = sorted(config["component_ranges"])
            idx = all_comps.index(best_max_comp)
            
            # Try to include one value below and one above
            if idx > 0:
                min_comp = all_comps[idx-1]
            if idx < len(all_comps) - 1:
                max_comp = all_comps[idx+1]
        
        # Get component values in range
        comp_values = [c for c in config["component_ranges"] if min_comp <= c <= max_comp]
        
        # Create parameter combinations for stage 2
        for max_comp in comp_values:
            for prep in config["preprocess_options"]:
                stage2["parameters"].append({
                    "max_components": max_comp,
                    "preprocess": prep
                })
    
    def _update_stage2_with_stable_components(self, stage2, stability_analysis, config):
        """Update stage 2 parameters with stable component values"""
        # If we found stable components, use only those
        if stability_analysis["stable_components"]:
            # Get stable component values
            stable_comps = [max_comp for max_comp, info in stability_analysis["stability_scores"].items() 
                           if info["stable"]]
            
            # Add parameter combinations for all preprocessing methods
            for max_comp in stable_comps:
                for prep in config["preprocess_options"]:
                    stage2["parameters"].append({
                        "max_components": max_comp,
                        "preprocess": prep
                    })
        else:
            # No stable components, use component with highest stability score
            if stability_analysis["most_stable_component"] is not None:
                best_max_comp = stability_analysis["most_stable_component"]
                
                # Add with all preprocessing methods
                for prep in config["preprocess_options"]:
                    stage2["parameters"].append({
                        "max_components": best_max_comp,
                        "preprocess": prep
                    })
            else:
                # Fallback to all combinations
                for max_comp in config["component_ranges"]:
                    for prep in config["preprocess_options"]:
                        stage2["parameters"].append({
                            "max_components": max_comp,
                            "preprocess": prep
                        })
    
    def _get_next_stage(self, current_stage):
        """Get the next stage after the current one"""
        if current_stage["name"] == "Coarse parameter scan":
            return {"name": "Fine-tuned parameter scan", "parameters": []}
        elif current_stage["name"] == "Component stability assessment":
            return {"name": "Fine-tuned parameter scan with stable components", "parameters": []}
        return None
    
    def _check_early_stopping(self, stage_results, all_results, cars_variant):
        """Check if we can stop early due to a clearly superior result"""
        # Need at least some results to check
        if not stage_results or len(stage_results) < 2:
            return False
        
        # Get best result from current stage
        best_stage = self._get_best_result(stage_results, cars_variant)
        
        # Get previous best result
        prev_best = None
        if len(all_results) > len(stage_results):
            prev_results = all_results[:-len(stage_results)]
            if prev_results:
                prev_best = self._get_best_result(prev_results, cars_variant)
        
        # If no previous results, can't stop early
        if prev_best is None:
            return False
        
        # Check if current best is clearly better than previous best
        if cars_variant in ["standard", "corcars"]:
            # For regression, lower RMSE is better
            curr_metric = best_stage["min_cv_error"]
            prev_metric = prev_best["min_cv_error"]
            
            # Is current clearly better (15% improvement)?
            if curr_metric < prev_metric * 0.85:
                return True
        else:
            # For classification, higher metric is better
            curr_metric = best_stage["best_metric_value"]
            prev_metric = prev_best["best_metric_value"]
            
            # Is current clearly better (15% improvement)?
            if curr_metric > prev_metric * 1.15:
                return True
        
        return False
    
    def _get_best_result(self, results, cars_variant):
        """Get the best result from a list of results"""
        if not results:
            return None
        
        # Sort results and return the best one
        sorted_results = self._sort_results(results, cars_variant)
        return sorted_results[0]
    
    def _sort_results(self, results, cars_variant):
        """Sort results by quality (best first)"""
        if cars_variant in ["standard", "corcars"]:
            # For regression, sort by min_cv_error (lower is better)
            return sorted(results, key=lambda x: x["min_cv_error"])
        else:
            # For classification, sort by metric value (higher is better)
            return sorted(results, key=lambda x: x["best_metric_value"], reverse=True)
    
    def _run_cars(self, params, config, cars_variant):
        """Run CARS with the given parameters and configuration"""
        # Copy parameters to avoid modifying the original
        all_params = params.copy()
        
        # Add common parameters from config
        for key in ["iterations", "n_jobs", "verbose"]:
            if key in config:
                all_params[key] = config[key]
        
        # Add classification-specific parameters
        if cars_variant == "classification":
            for key in ["encoding", "metric"]:
                if key in config:
                    all_params[key] = config[key]
        
        # Add correlation parameters for CorCARS
        if cars_variant == "corcars":
            for key in ["use_correlation", "alpha"]:
                if key in config:
                    all_params[key] = config[key]
        
        # Keep track of what we're running
        if self.verbose > 1:
            print(f"  Running {cars_variant} CARS with parameters: {all_params}")
        
        # Run CARS function
        return self._run_cars_function(all_params, cars_variant)
    
    def _run_cars_function(self, params, cars_variant):
        """Call the appropriate CARS function with error handling"""
        # Get the function
        cars_func = self.cars_modules[cars_variant]
        if cars_func is None:
            raise ImportError(f"No implementation found for {cars_variant} CARS")
        
        # Make sure parameter max_components is valid
        max_components = params.get("max_components", 10)
        max_possible = min(self.n_samples, self.n_features)
        if max_components > max_possible:
            if self.verbose:
                print(f"Warning: Limiting max_components from {max_components} to {max_possible}")
            params["max_components"] = max_possible
        
        # Prepare extra arguments
        try:
            # Copy parameters, removing any that aren't expected by the function
            func_params = {
                "X": self.X_original,
                "y": self.y_original
            }
            
            # Add all other parameters
            for key, value in params.items():
                func_params[key] = value
            
            # Set reasonable verbosity
            if "verbose" not in func_params:
                func_params["verbose"] = max(0, self.verbose - 1)
            
            # Call the function
            return cars_func(**func_params)
            
        except Exception as e:
            error_msg = str(e)
            
            # Check for common errors and provide more helpful messages
            if "NaN" in error_msg or "Infinity" in error_msg:
                raise ValueError(
                    f"Error in CARS: {error_msg}\n"
                    "This may be caused by numerical instability. "
                    "Try a different preprocessing method or reduce max_components."
                )
            elif "memory" in error_msg.lower():
                raise MemoryError(
                    f"Memory error in CARS: {error_msg}\n"
                    "Try reducing max_components, iterations, or n_jobs."
                )
            else:
                # Re-raise with context
                raise RuntimeError(f"Error running {cars_variant} CARS: {error_msg}")
    
    def _print_optimization_header(self, config):
        """Print header information for optimization"""
        print("\n" + "="*60)
        print(f"Starting CARS Optimization for {self.task.capitalize()} Task")
        print("="*60)
        
        # Print dataset info
        print(f"\nDataset: {self.n_samples} samples, {self.n_features} features")
        
        # Print task-specific info
        if self.task == "classification":
            if len(self.y.shape) == 1:
                n_classes = len(np.unique(self.y))
            else:
                n_classes = self.y.shape[1]
            
            print(f"Classification with {n_classes} classes, {config['encoding']} encoding")
            print(f"Evaluation metric: {config['metric']}")
        else:
            variant = "CorCARS" if config.get("use_correlation", False) else "standard CARS"
            print(f"Regression with {variant}")
        
        # Print configuration summary
        print("\nConfiguration:")
        print(f"  Component ranges: {config['component_ranges']}")
        print(f"  Preprocessing methods: {config['preprocess_options']}")
        print(f"  CARS iterations: {config['iterations']}")
        print(f"  Parallel jobs: {config['n_jobs']}")
        
        if config.get("perform_stability", False):
            print(f"  Stability assessment: {config['n_bootstrap']} bootstrap samples")
            print(f"  Stability threshold: {config['stability_threshold']}")
        
        print(f"\nRunning optimization with multi-stage approach...")
    
    def _print_parameter_progress(self, current, total, params):
        """Print progress information for parameter evaluation"""
        print(f"  Evaluating parameter set {current+1}/{total}: " + 
              f"max_components={params['max_components']}, preprocess='{params['preprocess']}'")
    
    def _print_stage_result(self, result, cars_variant):
        """Print result summary for a stage"""
        print("\nBest result from this stage:")
        self._print_result_summary(result, cars_variant)
    
    def _print_final_result(self, result, cars_variant, elapsed_time):
        """Print final optimization result"""
        print("\n" + "="*60)
        print(f"Optimization Completed in {elapsed_time:.1f} seconds")
        print("="*60)
        print("\nBest Parameters:")
        print(f"  max_components: {result['max_components_setting']}")
        print(f"  preprocess: '{result['preprocess_method']}'")
        print(f"  optimal components: {result['optimal_components']}")
        print(f"  selected variables: {result['n_selected_vars']}")
        
        print("\nPerformance:")
        if cars_variant in ["standard", "corcars"]:
            print(f"  RMSE: {result['min_cv_error']:.4f}")
            print(f"  R²: {result['max_r_squared']:.4f}")
        else:
            metric = result.get('metric', self._metric).upper()
            print(f"  {metric}: {result['best_metric_value']:.4f}")
    
    def _print_result_summary(self, result, cars_variant):
        """Print a summary of a result"""
        if result is None:
            print("  No valid result available")
            return
        
        print(f"  Parameters: max_components={result['max_components_setting']}, preprocess='{result['preprocess_method']}'")
        print(f"  Selected {result['n_selected_vars']} variables with {result['optimal_components']} components")
        
        if cars_variant in ["standard", "corcars"]:
            print(f"  RMSE: {result['min_cv_error']:.4f}, R²: {result['max_r_squared']:.4f}")
        else:
            metric = result.get('metric', self._metric).upper()
            print(f"  {metric}: {result['best_metric_value']:.4f}")
    
    def evaluate(self, X_test=None, y_test=None):
        """
        Evaluate the model on test data or using cross-validation.
        
        Parameters:
        -----------
        X_test : array-like, optional
            Test data. If None, uses cross-validation metrics from optimization.
        y_test : array-like, optional
            Test targets. Required if X_test is provided.
            
        Returns:
        --------
        dict : Evaluation metrics
        """
        # Check if we have a final model
        if self.final_model is None:
            if self.best_result is None:
                raise ValueError("No model or optimization results available. Run run() first.")
            
            # Use cross-validation metrics from optimization
            if self.verbose:
                print("No final model available. Using cross-validation metrics from optimization.")
            
            result = self.best_result
            
            if self.task == "regression":
                metrics = {
                    "rmse": result["min_cv_error"],
                    "r2": result["max_r_squared"],
                    "n_components": result["optimal_components"],
                    "n_selected_vars": result["n_selected_vars"],
                    "evaluation_type": "cross-validation"
                }
            else:  # classification
                metric_name = result.get("metric", self._metric)
                
                metrics = {
                    metric_name: result["best_metric_value"],
                    "n_components": result["optimal_components"],
                    "n_selected_vars": result["n_selected_vars"],
                    "evaluation_type": "cross-validation"
                }
            
            if self.verbose:
                print("\nCross-Validation Metrics:")
                for name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"  {name}: {value:.4f}")
                    else:
                        print(f"  {name}: {value}")
            
            return metrics
        
        # If test data is provided, use it for evaluation
        if X_test is not None:
            if y_test is None:
                raise ValueError("y_test must be provided with X_test.")
            
            # Make predictions
            if self.task == "regression":
                y_pred = self.predict(X_test)
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                metrics = {
                    "rmse": rmse,
                    "r2": r2,
                    "n_components": self._model_info["optimal_components"],
                    "n_selected_vars": len(self._model_info["selected_vars"]),
                    "evaluation_type": "test_set"
                }
                
                if self.verbose:
                    print("\nTest Set Metrics:")
                    print(f"  RMSE: {rmse:.4f}")
                    print(f"  R²: {r2:.4f}")
                
                return metrics
            
            else:  # classification
                y_pred = self.predict(X_test)
                
                # Convert y_test to class indices if one-hot encoded
                y_test_cls = y_test
                if len(y_test.shape) > 1 and y_test.shape[1] > 1:
                    y_test_cls = np.argmax(y_test, axis=1)
                
                # Calculate metrics
                from sklearn.metrics import accuracy_score, f1_score
                accuracy = accuracy_score(y_test_cls, y_pred)
                
                # Calculate F1 based on number of classes
                if len(np.unique(y_test_cls)) > 2:
                    f1 = f1_score(y_test_cls, y_pred, average='weighted')
                else:
                    f1 = f1_score(y_test_cls, y_pred)
                
                metrics = {
                    "accuracy": accuracy,
                    "f1": f1,
                    "n_components": self._model_info["optimal_components"],
                    "n_selected_vars": len(self._model_info["selected_vars"]),
                    "evaluation_type": "test_set"
                }
                
                if self.verbose:
                    print("\nTest Set Metrics:")
                    print(f"  Accuracy: {accuracy:.4f}")
                    print(f"  F1 Score: {f1:.4f}")
                
                return metrics
        
        # If no test data and we have a model, run cross-validation on the original data
        if self.verbose:
            print("No test data provided. Using cross-validation on original data.")
        
        # Extract parameters from model info
        selected_vars = self._model_info["selected_vars"]
        n_components = self._model_info["optimal_components"]
        
        # Set up cross-validation
        if self.task == "classification":
            if len(self.y.shape) == 1:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                splits = list(cv.split(self.X, self.y))
            else:
                # For one-hot encoding, use class indices for stratification
                y_indices = np.argmax(self.y, axis=1)
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                splits = list(cv.split(self.X, y_indices))
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            splits = list(cv.split(self.X))
        
        # Select variables
        X_sel = self.X[:, selected_vars]
        
        # Perform cross-validation
        if self.task == "regression":
            y_pred = np.zeros_like(self.y)
            
            for train_idx, test_idx in splits:
                X_train, X_test = X_sel[train_idx], X_sel[test_idx]
                y_train, y_test = self.y[train_idx], self.y[test_idx]
                
                # Preprocess
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Fit model
                model = PLSRegression(n_components=n_components)
                model.fit(X_train_scaled, y_train)
                
                # Predict
                y_pred[test_idx] = model.predict(X_test_scaled).ravel()
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(self.y, y_pred))
            r2 = r2_score(self.y, y_pred)
            
            metrics = {
                "rmse": rmse,
                "r2": r2,
                "n_components": n_components,
                "n_selected_vars": len(selected_vars),
                "evaluation_type": "cross-validation"
            }
            
            if self.verbose:
                print("\nCross-Validation Metrics:")
                print(f"  RMSE: {rmse:.4f}")
                print(f"  R²: {r2:.4f}")
            
            return metrics
            
        else:  # classification
            if self._encoding == "onehot":
                y_pred = np.zeros(self.n_samples, dtype=int)
                
                for train_idx, test_idx in splits:
                    X_train, X_test = X_sel[train_idx], X_sel[test_idx]
                    y_train = self.y[train_idx]
                    
                    # Preprocess
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Fit model
                    model = PLSRegression(n_components=n_components)
                    model.fit(X_train_scaled, y_train)
                    
                    # Predict
                    y_pred_raw = model.predict(X_test_scaled)
                    y_pred[test_idx] = np.argmax(y_pred_raw, axis=1)
                
                # For evaluation, use class indices
                y_true = np.argmax(self.y, axis=1)
                
            else:  # ordinal
                y_pred = np.zeros_like(self.y)
                
                for train_idx, test_idx in splits:
                    X_train, X_test = X_sel[train_idx], X_sel[test_idx]
                    y_train, y_test = self.y[train_idx], self.y[test_idx]
                    
                    # Preprocess
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Fit model
                    model = PLSRegression(n_components=n_components)
                    model.fit(X_train_scaled, y_train.reshape(-1, 1))
                    
                    # Predict
                    y_pred_raw = model.predict(X_test_scaled).ravel()
                    
                    # Round to nearest class
                    n_classes = len(np.unique(self.y))
                    y_pred[test_idx] = np.round(y_pred_raw).clip(0, n_classes-1).astype(int)
                
                y_true = self.y
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, f1_score
            accuracy = accuracy_score(y_true, y_pred)
            
            # Calculate F1 based on number of classes
            if len(np.unique(y_true)) > 2:
                f1 = f1_score(y_true, y_pred, average='weighted')
            else:
                f1 = f1_score(y_true, y_pred)
            
            metrics = {
                "accuracy": accuracy,
                "f1": f1,
                "n_components": n_components,
                "n_selected_vars": len(selected_vars),
                "evaluation_type": "cross-validation"
            }
            
            if self.verbose:
                print("\nCross-Validation Metrics:")
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  F1 Score: {f1:.4f}")
            
            return metrics
    
    def plot_results(self, save_path=None):
        """
        Create a comprehensive plot of optimization results and model performance.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot. If None, the plot will be displayed.
            
        Returns:
        --------
        fig : matplotlib figure
            The figure object for further customization
        """
        if not self.all_results and not self.best_result:
            raise ValueError("No optimization results available. Run run() first.")
        
        # Create figure
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Determine CARS variant
        if self.task == "regression":
            cars_variant = "standard"  # Default to standard for plotting purposes
        else:
            cars_variant = "classification"
        
        # Plot 1: Selected variables
        ax = axs[0, 0]
        
        if self.best_result and "selected_variables" in self.best_result:
            selected_vars = self.best_result["selected_variables"]
            
            if len(selected_vars) > 0:
                # Create dummy wavelength indices if spectral data
                dummy_wavelengths = np.arange(self.X.shape[1])
                
                # Calculate mean spectra
                mean_spectrum = np.mean(self.X, axis=0)
                
                # Plot the full spectrum
                ax.plot(dummy_wavelengths, mean_spectrum, 'b-', alpha=0.5, label='Full data')
                
                # Create a mask for selected variables
                mask = np.zeros(self.X.shape[1], dtype=bool)
                mask[selected_vars] = True
                
                # Highlight selected variables
                ax.plot(dummy_wavelengths[mask], mean_spectrum[mask], 'ro', label='Selected variables')
                
                # Add labels
                ax.set_title(f"Selected Variables ({len(selected_vars)} of {self.X.shape[1]})", fontsize=12)
                ax.set_xlabel("Variable Index", fontsize=10)
                ax.set_ylabel("Mean Value", fontsize=10)
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, "No variables selected", ha='center', va='center', fontsize=12)
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, "No selected variables data available", ha='center', va='center', fontsize=12)
            ax.axis('off')
        
        # Plot 2: Optimization progress (if available)
        ax = axs[0, 1]
        
        if self.all_results:
            # Extract data for plot
            data = []
            for result in self.all_results:
                if "max_components_setting" in result and "preprocess_method" in result:
                    data_point = {
                        "max_components": result["max_components_setting"],
                        "preprocess": result["preprocess_method"],
                        "optimal_components": result["optimal_components"],
                        "n_selected_vars": len(result["selected_variables"])
                    }
                    
                    if self.task == "regression":
                        data_point["metric"] = result["min_cv_error"]
                    else:
                        data_point["metric"] = result["best_metric_value"]
                    
                    data.append(data_point)
            
            # Convert to DataFrame
            if data:
                df = pd.DataFrame(data)
                
                # Group by max_components and preprocess
                for prep in df["preprocess"].unique():
                    subset = df[df["preprocess"] == prep]
                    if len(subset) > 1:  # Need at least 2 points to draw a line
                        ax.plot(subset["max_components"], subset["metric"], 'o-', label=prep)
                
                # Highlight best result
                if self.best_result:
                    best_max_comp = self.best_result["max_components_setting"]
                    best_prep = self.best_result["preprocess_method"]
                    
                    best_subset = df[(df["max_components"] == best_max_comp) & 
                                   (df["preprocess"] == best_prep)]
                    
                    if len(best_subset) > 0:
                        best_metric = best_subset["metric"].iloc[0]
                        ax.plot(best_max_comp, best_metric, 'r*', 
                               markersize=15, label='Best')
                
                # Add labels
                metric_name = "RMSE" if self.task == "regression" else self._metric.upper()
                ax.set_title(f"{metric_name} by Parameters", fontsize=12)
                ax.set_xlabel("Max Components", fontsize=10)
                ax.set_ylabel(metric_name, fontsize=10)
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, "No optimization data available", ha='center', va='center', fontsize=12)
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, "No optimization data available", ha='center', va='center', fontsize=12)
            ax.axis('off')
        
        # Plot 3: Component distribution (if available)
        ax = axs[1, 0]
        
        if self.all_results:
            # Extract data for component counts
            if data:
                df = pd.DataFrame(data)
                
                # Count frequency of optimal component values
                comp_counts = df["optimal_components"].value_counts().sort_index()
                
                if len(comp_counts) > 0:
                    # Plot as bar chart
                    ax.bar(comp_counts.index, comp_counts.values)
                    
                    # Highlight optimal component count from best result
                    if self.best_result:
                        best_comp = self.best_result["optimal_components"]
                        if best_comp in comp_counts:
                            best_idx = list(comp_counts.index).index(best_comp)
                            ax.bar([best_comp], [comp_counts[best_comp]], color='green', 
                                 label=f'Best: {best_comp}')
                    
                    # Add labels
                    ax.set_title("Optimal Component Distribution", fontsize=12)
                    ax.set_xlabel("Number of Components", fontsize=10)
                    ax.set_ylabel("Frequency", fontsize=10)
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                else:
                    ax.text(0.5, 0.5, "No component data available", ha='center', va='center', fontsize=12)
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, "No component data available", ha='center', va='center', fontsize=12)
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, "No component data available", ha='center', va='center', fontsize=12)
            ax.axis('off')
        
        # Plot 4: Summary of best result
        ax = axs[1, 1]
        
        if self.best_result:
            # Create a text summary
            if self.task == "regression":
                text = (
                    f"Best Result Summary:\n\n"
                    f"Parameters:\n"
                    f"- max_components: {self.best_result['max_components_setting']}\n"
                    f"- preprocess: {self.best_result['preprocess_method']}\n\n"
                    f"Model:\n"
                    f"- Optimal components: {self.best_result['optimal_components']}\n"
                    f"- Selected variables: {len(self.best_result['selected_variables'])}\n\n"
                    f"Performance:\n"
                    f"- RMSE: {self.best_result['min_cv_error']:.4f}\n"
                    f"- R²: {self.best_result['max_r_squared']:.4f}\n"
                )
            else:
                metric_name = self.best_result.get("metric", self._metric).upper()
                text = (
                    f"Best Result Summary:\n\n"
                    f"Parameters:\n"
                    f"- max_components: {self.best_result['max_components_setting']}\n"
                    f"- preprocess: {self.best_result['preprocess_method']}\n"
                    f"- encoding: {self.best_result.get('encoding', self._encoding)}\n\n"
                    f"Model:\n"
                    f"- Optimal components: {self.best_result['optimal_components']}\n"
                    f"- Selected variables: {len(self.best_result['selected_variables'])}\n\n"
                    f"Performance:\n"
                    f"- {metric_name}: {self.best_result['best_metric_value']:.4f}\n"
                )
            
            # Add text about final model
            if self.final_model is not None:
                text += "\nFinal model built and ready for prediction."
            
            # Add text to plot
            ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=10, 
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, "No best result available", ha='center', va='center', fontsize=12)
            ax.axis('off')
        
        # Add main title
        task_title = "Regression" if self.task == "regression" else "Classification"
        plt.suptitle(f"SimpleCARSOptimizer Results - {task_title}", fontsize=14)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Save or display
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return fig

    def get_selected_variables(self):
        """
        Get the indices of selected variables.
        
        Returns:
        --------
        array : Indices of selected variables
        """
        if self.best_result is None:
            raise ValueError("No optimization results available. Run run() first.")
        
        return self.best_result["selected_variables"]
    
    def get_selected_data(self, X=None):
        """
        Get the data with only selected variables.
        
        Parameters:
        -----------
        X : array-like, optional
            Data to select from. If None, uses the original training data.
            
        Returns:
        --------
        array : Data with only selected variables
        """
        selected_vars = self.get_selected_variables()
        
        if X is None:
            return self.X[:, selected_vars]
        else:
            # Convert to numpy array if needed
            X_array = X.values if hasattr(X, 'values') else np.array(X)
            
            # Check dimensions
            if X_array.shape[1] != self.X.shape[1]:
                raise ValueError(
                    f"Input X has {X_array.shape[1]} features but the model was trained with {self.X.shape[1]} features."
                )
                
            return X_array[:, selected_vars]


# Example usage
def example_usage():
    """Example usage of the SimpleCARSOptimizer"""
    import numpy as np
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    
    # Generate synthetic data
    X, y = make_regression(
        n_samples=100,
        n_features=200,
        n_informative=20,
        random_state=42
    )
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create optimizer with auto task detection
    optimizer = SimpleCARSOptimizer(X_train, y_train, task="auto", verbose=1)
    
    # Run all-in-one optimization and model building
    result = optimizer.run(recipe="fast")
    
    # Get selected variables
    selected_vars = result['selected_variables']
    print(f"Selected {len(selected_vars)} variables")
    
    # Make predictions with the final model
    y_pred = optimizer.predict(X_test)
    
    # Evaluate on test data
    metrics = optimizer.evaluate(X_test, y_test)
    
    # Plot results
    optimizer.plot_results()
    
    return optimizer, result, metrics

if __name__ == "__main__":
    example_usage()
