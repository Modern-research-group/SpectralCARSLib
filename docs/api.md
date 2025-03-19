# SpectralCARSLib API Documentation

## Core Module

The core module contains the main implementation of the Competitive Adaptive Reweighted Sampling (CARS) algorithm.

### `competitive_adaptive_sampling`

```python
SpectralCARSLib.competitive_adaptive_sampling(X, y, max_components, folds=5, preprocess='center', 
                                  iterations=50, adaptive_resampling=False, 
                                  shuffle_sample_order=False, n_jobs=-1, 
                                  use_gpu=False, verbose=1)
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
- `iterations` : int, default=50
  - Number of Monte Carlo sampling runs.
- `adaptive_resampling` : bool, default=False
  - Whether to use the original version with random sampling (True) or a simplified deterministic version (False).
- `shuffle_sample_order` : bool, default=False
  - Whether to randomize sample order for cross-validation.
- `n_jobs` : int, default=-1
  - Number of parallel jobs for cross-validation. -1 means using all processors.
- `use_gpu` : bool, default=False
  - Whether to use GPU acceleration if available (requires cuPy and cuML).
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

## Preprocessing Module

The preprocessing module contains functions for data preprocessing.

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
    - 'unilength': Scale to unit vector length
    - 'none': No preprocessing
- `mean` : array-like, optional
  - Precalculated mean values
- `scale` : array-like, optional
  - Precalculated scale values

**Returns**:

- `tuple` : (preprocessed_data, mean, scale)

## Visualization Module

The visualization module contains functions for plotting CARS results.

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

## Utilities Module

The utilities module contains helper functions for the CARS algorithm.

### `suppress_pls_warnings`

```python
from SpectralCARSLib.utils import suppress_pls_warnings

with suppress_pls_warnings():
    # PLS operations that might trigger warnings
```

**Description**: Context manager to suppress expected RuntimeWarnings from sklearn PLS.
