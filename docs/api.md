# SpectralCARSLib API Documentation

## Core Module: CARS

### `competitive_adaptive_sampling`

```python
SpectralCARSLib.competitive_adaptive_sampling(X, y, max_components, folds=5, preprocess='center', 
                              iterations=50, adaptive_resampling=False, 
                              cv_shuffle_mode='none', n_jobs=-1, 
                              verbose=1)
```

**Description**: Performs variable selection using the Competitive Adaptive Reweighted Sampling algorithm.

**Parameters**:

- `X` : array-like of shape (n_samples, n_features)
  - The predictor matrix.
- `y` : array-like of shape (n_samples,)
  - The response vector.
- `max_components` : int
  - The maximum number of PLS components to extract.
- `folds` : int, default=5
  - Number of folds for cross-validation.
- `preprocess` : str, default='center'
  - Preprocessing method. Options are:
    - 'center': Mean centering
    - 'autoscaling': Mean centering and unit variance scaling (standardization)
    - 'pareto': Mean centering and scaling by square root of standard deviation
    - 'minmax': Min-max scaling
    - 'robust': Robust scaling using median and IQR
    - 'unilength': Scale to unit vector length
    - 'none': No preprocessing
- `iterations` : int, default=50
  - Number of Monte Carlo sampling runs.
- `adaptive_resampling` : bool, default=False
  - Whether to use random sampling (True) or deterministic version (False).
- `cv_shuffle_mode` : str, default='none'
  - Cross-validation sample ordering mode:
    - 'none': Maintain original sample order
    - 'fixed_seed': Shuffle samples with fixed random seed
    - 'random_seed': Shuffle samples with random seed for each run
- `n_jobs` : int, default=-1
  - Number of parallel jobs for cross-validation. -1 means using all processors.
- `verbose` : int, default=1
  - Verbosity level (0=silent, 1=summary progress only, 2=detailed per-iteration progress).

**Returns**:

- `dict` : Results dictionary containing:
  - 'weight_matrix': Coefficient evolution matrix (n_features, iterations)
  - 'computation_time': Total computation time in seconds
  - 'cross_validation_errors': RMSE for each iteration
  - 'min_cv_error': Minimum cross-validation error
  - 'max_r_squared': Maximum R² value
  - 'best_iteration': Index of the best iteration
  - 'optimal_components': Optimal number of PLS components
  - 'subset_ratios': Sampling ratios for each iteration
  - 'selected_variables': Indices of selected variables

**Example**:

```python
import numpy as np
from SpectralCARSLib import competitive_adaptive_sampling

# Create synthetic data
X = np.random.normal(0, 1, (100, 200))
y = X[:, :10].sum(axis=1) + np.random.normal(0, 0.1, 100)

# Run CARS
results = competitive_adaptive_sampling(X, y, max_components=10, iterations=50)

# Get selected variables
selected_vars = results['selected_variables']
print(f"Selected {len(selected_vars)} variables")
```

## CorCARS Module

### `competitive_adaptive_reweighted_sampling`

```python
SpectralCARSLib.competitive_adaptive_reweighted_sampling(X, y, max_components, folds=5, 
                                       preprocess='center', iterations=50,
                                       adaptive_resampling=False, cv_shuffle_mode='none', 
                                       use_correlation=True, alpha=0.25, 
                                       n_jobs=-1, verbose=1)
```

**Description**: CorCARS (Correlation-adjusted CARS) extends the original algorithm with correlation-based component selection.

**Parameters**: Same as `competitive_adaptive_sampling` with these additions:

- `use_correlation` : bool, default=True
  - Whether to use correlation adjustment in F-test component selection.
- `alpha` : float, default=0.25
  - Significance level for F-test.

**Returns**: Same as `competitive_adaptive_sampling` with these additions:
  - 'use_correlation': Whether correlation adjustment was used
  - 'alpha': Significance level used for F-test

**Example**:

```python
from SpectralCARSLib import competitive_adaptive_reweighted_sampling

# Run CorCARS with correlation-adjusted component selection
results = competitive_adaptive_reweighted_sampling(
    X=X, y=y,
    max_components=10, 
    preprocess='autoscaling',
    use_correlation=True,
    alpha=0.25
)

# CorCARS often selects more parsimonious models
print(f"Selected {len(results['selected_variables'])} variables")
print(f"Optimal components: {results['optimal_components']}")
```

### `corcars`

```python
SpectralCARSLib.corcars(X, y, max_components, folds=5, preprocess='center', iterations=50,
                        adaptive_resampling=False, cv_shuffle_mode='none', 
                        use_correlation=True, alpha=0.25, n_jobs=-1, verbose=1)
```

**Description**: Alias for `competitive_adaptive_reweighted_sampling`.

## Classification Module

### `competitive_adaptive_sampling_classification`

```python
SpectralCARSLib.competitive_adaptive_sampling_classification(X, y, max_components, 
                                          folds=5, preprocess='center', 
                                          iterations=50, encoding='ordinal', 
                                          metric='accuracy', adaptive_resampling=False, 
                                          cv_shuffle_mode='none', best_metric='max',
                                          n_jobs=-1, verbose=1)
```

**Description**: CARS algorithm extended for classification problems (binary and multi-class).

**Parameters**: Similar to standard CARS with these additions:

- `encoding` : str, default='ordinal'
  - Target encoding type: 'ordinal' for binary/ordinal, 'onehot' for multi-class.
- `metric` : str, default='accuracy'
  - Evaluation metric ('accuracy', 'f1', or 'auc').
- `best_metric` : str, default='max'
  - Whether to maximize ('max') or minimize ('min') the metric. Most classification metrics should be maximized.

**Returns**:

- `dict` : Results dictionary containing:
  - 'weight_matrix': Coefficient evolution matrix
  - 'computation_time': Total computation time in seconds
  - 'metric_values': Values of the specified metric for each iteration
  - 'best_metric_value': Best value of the specified metric
  - 'best_iteration': Index of the best iteration
  - 'optimal_components': Optimal number of PLS components
  - 'selected_variables': Indices of selected variables
  - 'confusion_matrix': Confusion matrix for the best model
  - 'encoding': Encoding used ('ordinal' or 'onehot')
  - 'target_names': Names for the target classes
  - 'metric': Metric used for evaluation
  - 'subset_ratios': Sampling ratios for each iteration

**Example**:

```python
from SpectralCARSLib import competitive_adaptive_sampling_classification
from sklearn.datasets import make_classification

# Generate classification data
X, y = make_classification(n_samples=100, n_features=200, n_classes=3, random_state=42)

# Run CARS for classification
results = competitive_adaptive_sampling_classification(
    X=X, y=y,
    max_components=10,
    encoding='ordinal',
    metric='f1'
)

# Get selected variables
selected_vars = results['selected_variables']
print(f"Selected {len(selected_vars)} variables")
print(f"Best {results['metric']} value: {results['best_metric_value']:.4f}")
```

### `generate_binary_classification_data`

```python
SpectralCARSLib.generate_binary_classification_data(n_samples=2000, n_features=200, 
                                                 n_informative=20, random_state=42)
```

**Description**: Generates synthetic binary classification data for testing and examples.

**Parameters**:
- `n_samples` : int, default=2000
  - Number of samples to generate.
- `n_features` : int, default=200
  - Number of features to generate.
- `n_informative` : int, default=20
  - Number of informative features.
- `random_state` : int, default=42
  - Random seed for reproducibility.

**Returns**:
- `tuple` : (X_df, y_series)
  - X_df: pandas DataFrame with the generated features
  - y_series: pandas Series with the target values (0 or 1)

**Example**:
```python
from SpectralCARSLib import generate_binary_classification_data

# Generate binary classification data
X, y = generate_binary_classification_data(n_samples=500, n_features=100)
print(f"Generated data: {X.shape[0]} samples, {X.shape[1]} features")
```

### `generate_multiclass_classification_data`

```python
SpectralCARSLib.generate_multiclass_classification_data(n_samples=3000, n_features=200, 
                                                     n_informative=15, n_classes=4, 
                                                     random_state=42)
```

**Description**: Generates synthetic multi-class classification data for testing and examples.

**Parameters**:
- `n_samples` : int, default=3000
  - Number of samples to generate.
- `n_features` : int, default=200
  - Number of features to generate.
- `n_informative` : int, default=15
  - Number of informative features.
- `n_classes` : int, default=4
  - Number of classes.
- `random_state` : int, default=42
  - Random seed for reproducibility.

**Returns**:
- `tuple` : (X_df, y_series, y_onehot)
  - X_df: pandas DataFrame with the generated features
  - y_series: pandas Series with the ordinal target values
  - y_onehot: pandas DataFrame with one-hot encoded target values

**Example**:
```python
from SpectralCARSLib import generate_multiclass_classification_data

# Generate multi-class data
X, y, y_onehot = generate_multiclass_classification_data(n_samples=500, n_classes=3)
print(f"Generated data: {X.shape[0]} samples, {X.shape[1]} features, {y_onehot.shape[1]} classes")
```

## CARSOptimizer Module

### `CARSOptimizer`

```python
from SpectralCARSLib import CARSOptimizer

optimizer = CARSOptimizer(X, y, cars_variant='standard', component_ranges=None, 
                        preprocess_options=None, folds=5, iterations=100, 
                        n_jobs=-1, verbose=1, cars_func=None, 
                        plot_func=None, **variant_kwargs)
```

**Description**: A versatile parameter optimizer for CARS variants: standard CARS, CorCARS, and CARS Classification.

**Parameters**:

- `X` : array-like
  - The predictor matrix.
- `y` : array-like
  - The response vector.
- `cars_variant` : str
  - Which CARS variant to use: 'standard', 'corcars', or 'classification'.
- `component_ranges` : list, optional
  - List of max_components values to try. Default: [5,10,15,20,25,30]
- `preprocess_options` : list, optional
  - List of preprocessing methods to try. Default: ['center', 'autoscaling', 'pareto', 'minmax']
- `folds` : int
  - Number of folds for cross-validation.
- `iterations` : int, default=100
  - Number of CARS iterations.
- `n_jobs` : int
  - Number of parallel jobs.
- `verbose` : int
  - Verbosity level.
- `cars_func` : function, optional
  - Direct reference to the CARS implementation function if it can't be automatically found.
- `plot_func` : function, optional
  - Direct reference to the plotting function if it can't be automatically found.
- `**variant_kwargs` : dict
  - Additional keyword arguments specific to the CARS variant:
    - For classification: 'encoding' ('ordinal'/'onehot'), 'metric' ('accuracy'/'f1'/'auc')
    - For CorCARS: 'use_correlation' (bool), 'alpha' (float)

**Methods**:

#### `staged_parameter_scan`

```python
optimizer.staged_parameter_scan(alpha=0.6, beta=0.2, gamma=0.2, batch_size=None, **kwargs)
```

**Description**: Find optimal parameters using a two-stage approach with multi-objective optimization.

**Parameters**:
- `alpha` : float
  - Weight for predictive performance (RMSE or classification metric).
- `beta` : float
  - Weight for model parsimony (number of variables).
- `gamma` : float
  - Weight for component complexity.
- `batch_size` : int, optional
  - Number of parameter combinations to evaluate in a batch to limit memory usage.
- `**kwargs` : dict
  - Additional parameters passed to the CARS function.

**Returns**: 
- `dict` : Results of optimization containing:
  - 'method': Optimization method used ('staged_parameter_scan')
  - 'best_result': The best CARS result dict
  - 'all_results': List of all results
  - 'computation_time': Total time taken
  - 'weights': Dict of weights used (alpha, beta, gamma)
  - 'scan_details': Dict with details about the optimization stages
  - 'cars_variant': Which CARS variant was used

#### `bayesian_optimization`

```python
optimizer.bayesian_optimization(n_calls=20, min_components=5, max_components=30, 
                               penalization=1e5, **kwargs)
```

**Description**: Find optimal parameters using Bayesian optimization (requires scikit-optimize).

**Parameters**:
- `n_calls` : int
  - Number of parameter combinations to evaluate.
- `min_components`, `max_components` : int
  - Range for component search.
- `penalization` : float, default=1e5
  - Value to return for failed evaluations (high for minimization).
- `**kwargs` : dict
  - Additional parameters passed to the CARS function.

**Returns**: 
- `dict` : Results of optimization containing:
  - 'method': Optimization method used ('bayesian')
  - 'best_result': The best CARS result dict
  - 'all_results': List of all results
  - 'failed_evaluations': Details about any failed evaluations
  - 'computation_time': Total time taken
  - 'skopt_result': The scikit-optimize result object
  - 'cars_variant': Which CARS variant was used

#### `plot_optimization_results`

```python
optimizer.plot_optimization_results(results, save_path=None)
```

**Description**: Plot the results of parameter optimization.

**Parameters**:
- `results` : dict
  - The results dictionary from an optimization method.
- `save_path` : str, optional
  - Path to save the plot. If None, the plot will be displayed.

#### `plot_best_cars_results`

```python
optimizer.plot_best_cars_results(result, save_path=None)
```

**Description**: Plot the detailed results from the best CARS run.

**Parameters**:
- `result` : dict
  - The result dictionary from optimization.
- `save_path` : str, optional
  - Path to save the plot. If None, the plot will be displayed.

**Example**:

```python
from SpectralCARSLib import CARSOptimizer

# Create optimizer
optimizer = CARSOptimizer(
    X=X, y=y,
    cars_variant='standard',
    component_ranges=[5, 10, 15, 20],
    preprocess_options=['center', 'autoscaling', 'pareto'],
    iterations=50
)

# Run optimization
results = optimizer.staged_parameter_scan(alpha=0.6, beta=0.2, gamma=0.2)

# Get best result
best_result = results['best_result']
print(f"Best parameters: max_components={best_result['max_components_setting']}, "
      f"preprocess='{best_result['preprocess_method']}'")

# Plot results
optimizer.plot_optimization_results(results)
optimizer.plot_best_cars_results(results)
```

## SimpleCARSOptimizer

### `SimpleCARSOptimizer`

```python
from SpectralCARSLib import SimpleCARSOptimizer

optimizer = SimpleCARSOptimizer(X, y, task="auto", verbose=1, encoding=None, metric=None)
```

**Description**: A user-friendly all-in-one optimizer and model builder for CARS variable selection. This class provides a simplified workflow for parameter optimization, model building, and prediction with sensible defaults and pre-configured recipes for common use cases.

**Parameters**:

- `X` : array-like
  - The predictor matrix
- `y` : array-like
  - Target variable (vector for regression, vector or matrix for classification)
- `task` : str, default="auto"
  - Task type: "auto", "regression" or "classification".
  - If "auto", will try to detect based on y structure.
- `verbose` : int, default=1
  - Verbosity level (0=silent, 1=normal, 2=detailed)
- `encoding` : str, optional
  - For classification, "ordinal" or "onehot"
- `metric` : str, optional
  - For classification, "accuracy", "f1", or "auc"

**Available Recipes**:

SimpleCARSOptimizer comes with pre-configured recipes for common scenarios:

- "fast": Quick optimization with minimal computation
- "default": Balanced optimization with reasonable computation time
- "thorough": Comprehensive optimization exploring many parameters
- "robust": Stability-focused optimization resistant to overfitting
- "classification": Optimized for classification tasks

**Methods**:

#### `run`

```python
optimizer.run(recipe=None, build_final_model=True, **kwargs)
```

**Description**: All-in-one function to optimize CARS parameters and build the final model.

**Parameters**:
- `recipe` : str, optional
  - Name of pre-configured recipe: "fast", "default", "thorough", "robust", "classification"
  - If None, will auto-select based on dataset characteristics
- `build_final_model` : bool, default=True
  - Whether to build the final model with optimized parameters
- `**kwargs` : dict
  - Custom parameters that override recipe defaults

**Returns**:
- `dict` : Complete results including:
  - All optimization results
  - Selected variables
  - Final model (if build_final_model=True)
  - Preprocessor (if build_final_model=True)

#### `optimize`

```python
optimizer.optimize(recipe=None, **kwargs)
```

**Description**: Optimize CARS parameters using the specified recipe or custom parameters.

**Parameters**:
- `recipe` : str, optional
  - Name of pre-configured recipe
- `**kwargs` : dict
  - Custom parameters that override recipe defaults

**Returns**:
- `dict` : Best optimization result

#### `build_final_model`

```python
optimizer.build_final_model(optimization_result=None)
```

**Description**: Build the final model using the optimized parameters.

**Parameters**:
- `optimization_result` : dict, optional
  - Result from optimization. If None, uses stored best_result.

**Returns**:
- `tuple` : (model, preprocessor)
  - The final fitted model and preprocessor

#### `predict`

```python
optimizer.predict(X_new)
```

**Description**: Make predictions using the final model.

**Parameters**:
- `X_new` : array-like
  - New data to predict on

**Returns**:
- `array` : Predictions

#### `predict_proba`

```python
optimizer.predict_proba(X_new)
```

**Description**: For classification, predict class probabilities.

**Parameters**:
- `X_new` : array-like
  - New data to predict on

**Returns**:
- `array` : Class probabilities

#### `evaluate`

```python
optimizer.evaluate(X_test=None, y_test=None)
```

**Description**: Evaluate the model on test data or using cross-validation.

**Parameters**:
- `X_test` : array-like, optional
  - Test data. If None, uses cross-validation metrics from optimization.
- `y_test` : array-like, optional
  - Test targets. Required if X_test is provided.

**Returns**:
- `dict` : Evaluation metrics

#### `plot_results`

```python
optimizer.plot_results(save_path=None)
```

**Description**: Create a comprehensive plot of optimization results and model performance.

**Parameters**:
- `save_path` : str, optional
  - Path to save the plot. If None, the plot will be displayed.

**Returns**:
- `matplotlib.figure.Figure` : The figure object for further customization

#### `get_selected_variables`

```python
optimizer.get_selected_variables()
```

**Description**: Get the indices of selected variables.

**Returns**:
- `array` : Indices of selected variables

#### `get_selected_data`

```python
optimizer.get_selected_data(X=None)
```

**Description**: Get the data with only selected variables.

**Parameters**:
- `X` : array-like, optional
  - Data to select from. If None, uses the original training data.

**Returns**:
- `array` : Data with only selected variables

#### `print_available_recipes`

```python
optimizer.print_available_recipes()
```

**Description**: Print information about all available optimization recipes.

**Example**:

```python
from SpectralCARSLib import SimpleCARSOptimizer

# Initialize with auto task detection
optimizer = SimpleCARSOptimizer(X, y)

# Run all-in-one optimization and model building
result = optimizer.run(recipe="fast")

# Get selected variables and model
selected_vars = result['selected_variables']
model = result['model']

# Make predictions with the final model
y_pred = optimizer.predict(X_new)

# For classification, get probabilities
if optimizer.task == "classification":
    y_proba = optimizer.predict_proba(X_new)

# Evaluate performance
metrics = optimizer.evaluate(X_test, y_test)

# Visualize results
optimizer.plot_results()
```

## Preprocessing Module

### `preprocess_data`

```python
SpectralCARSLib.preprocessing.preprocess_data(X, method, mean=None, scale=None)
```

**Description**: Preprocesses data using various methods commonly used in chemometrics.

**Parameters**:

- `X` : array-like
  - Data to preprocess
- `method` : str
  - Preprocessing method:
    - 'center': Mean centering
    - 'autoscaling': Standardization (mean=0, std=1)
    - 'pareto': Pareto scaling (divide by sqrt of std)
    - 'minmax': Min-max scaling
    - 'robust': Robust scaling using median and IQR
    - 'unilength': Scale to unit vector length
    - 'none': No preprocessing
- `mean` : array-like, optional
  - Precalculated mean values
- `scale` : array-like, optional
  - Precalculated scale values

**Returns**:

- `tuple` : (preprocessed_data, mean, scale)

## Visualization Module

### `plot_sampling_results`

```python
SpectralCARSLib.visualization.plot_sampling_results(results)
```

**Description**: Plots the results of CARS analysis.

**Parameters**:

- `results` : dict
  - The results dictionary from competitive_adaptive_sampling

**Returns**:

- `matplotlib.figure.Figure` : The created figure

### `plot_selected_variables`

```python
SpectralCARSLib.visualization.plot_selected_variables(X, wavelengths, selected_vars, title="Selected Variables")
```

**Description**: Plots the selected variables, highlighting them in the average spectrum.

**Parameters**:

- `X` : array-like
  - The predictor matrix of shape (n_samples, n_features)
- `wavelengths` : array-like
  - The wavelengths or variable indices corresponding to each feature
- `selected_vars` : array-like
  - Indices of selected variables from CARS
- `title` : str, default="Selected Variables"
  - Plot title

**Returns**:

- `matplotlib.figure.Figure` : The created figure

### `plot_classification_results`

```python
SpectralCARSLib.visualization.plot_classification_results(results)
```

**Description**: Plots the results of CARS classification analysis.

**Parameters**:

- `results` : dict
  - The results dictionary from competitive_adaptive_sampling_classification

**Returns**:

- `matplotlib.figure.Figure` : The created figure

## Utilities Module

### `suppress_pls_warnings`

```python
from SpectralCARSLib.utils import suppress_pls_warnings

with suppress_pls_warnings():
    # PLS operations that might trigger warnings
```

**Description**: Context manager to suppress expected RuntimeWarnings from sklearn PLS.
