# SpectralCARSLib: Competitive Adaptive Reweighted Sampling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance implementation of Competitive Adaptive Reweighted Sampling (CARS) family of algorithms for variable selection in PLS regression and classification models, with a focus on spectroscopy applications.

## Overview

The CARS family encompasses several algorithms for variable selection in chemometrics and spectroscopy data analysis:

- **Standard CARS**: The original competitive adaptive reweighted sampling algorithm for regression problems
- **CorCARS**: Correlation-adjusted CARS that utilizes error correlation structure for more parsimonious models
- **CARS Classification**: Extended CARS for classification problems with support for binary and multi-class datasets
- **CARSOptimizer**: Advanced parameter optimization tools for the CARS family
- **SimpleCARSOptimizer**: User-friendly all-in-one optimizer with recipes for common use cases

This implementation offers:

- **High Performance**: Optimized implementation with parallel processing support
- **Flexibility**: Multiple preprocessing options (center, autoscaling, pareto, minmax)
- **Visualization**: Built-in plotting functions for all variants
- **Comprehensive API**: Easy-to-use interface with extensive documentation
- **User-Friendly Workflows**: Simplified interfaces for quick and reliable results

CARS works by competitively eliminating variables with small regression coefficients through a Monte Carlo sampling process, enabling effective identification of the most informative variables for PLS regression models.

## Installation
The minimum version is Python>=3.9.0

Install SpectralCARSLib:
```bash
# Install from PyPI
pip install SpectralCARSLib

# For development extras (testing, documentation)
pip install SpectralCARSLib[dev]

# For optimizer functionality (requires scikit-optimize)
pip install SpectralCARSLib[optimizer]
```

Clone the repository locally and install with:
```bash
git clone https://github.com/Ginnovation-lab/SpectralCARSLib.git
cd SpectralCARSLib
pip install -e .
```

## Quick Start

### Standard CARS for Regression

```python
from SpectralCARSLib import competitive_adaptive_sampling
import numpy as np

# Generate synthetic data (500 variables, 20 relevant)
np.random.seed(42)
n_samples, n_features = 200, 500
X = np.random.normal(0, 1, (n_samples, n_features))
true_coef = np.zeros(n_features)
true_coef[:20] = np.random.normal(0, 5, 20)
y = X.dot(true_coef) + np.random.normal(0, 1, n_samples)

# Run CARS variable selection
cars_results = competitive_adaptive_sampling(
    X=X,
    y=y,
    max_components=10,
    folds=5,
    preprocess='center',
    iterations=50,
    adaptive_resampling=False,
    verbose=1
)

# Get selected variables
selected_vars = cars_results['selected_variables']
print(f"Selected {len(selected_vars)} out of {n_features} variables")

# Plot results
from SpectralCARSLib import plot_sampling_results
plot_sampling_results(cars_results)
```

### CorCARS for Improved Component Selection

```python
from SpectralCARSLib import competitive_adaptive_reweighted_sampling

# Run CorCARS with correlation-adjusted component selection
corcars_results = competitive_adaptive_reweighted_sampling(
    X=X,
    y=y,
    max_components=10,
    folds=5,
    preprocess='center',
    iterations=50,
    use_correlation=True,  # Enable correlation adjustment
    alpha=0.25,            # Significance level for F-test
    verbose=1
)

# Get selected variables
selected_vars = corcars_results['selected_variables']
print(f"Selected {len(selected_vars)} out of {n_features} variables")
```

### CARS for Classification

```python
from SpectralCARSLib import competitive_adaptive_sampling_classification
from sklearn.datasets import make_classification

# Generate classification data
X_cls, y_cls = make_classification(n_samples=100, n_features=50, 
                                 n_informative=10, n_classes=3, 
                                 random_state=42)

# Run CARS for classification
cls_results = competitive_adaptive_sampling_classification(
    X=X_cls,
    y=y_cls,
    max_components=10,
    preprocess='autoscaling',
    iterations=50,
    encoding='ordinal',  # Use 'onehot' for one-hot encoded targets
    metric='f1',         # Options: 'accuracy', 'f1', 'auc'
    verbose=1
)

# Get selected variables and plot results
selected_vars = cls_results['selected_variables']
print(f"Selected {len(selected_vars)} out of {X_cls.shape[1]} variables")

from SpectralCARSLib import plot_classification_results
plot_classification_results(cls_results)
```

### Parameter Optimization with CARSOptimizer

```python
from SpectralCARSLib import CARSOptimizer

# Create optimizer
optimizer = CARSOptimizer(
    X=X, 
    y=y,
    cars_variant='standard',  # Use 'standard', 'corcars', or 'classification'
    component_ranges=[5, 10, 15, 20],
    preprocess_options=['center', 'autoscaling', 'pareto'],
    folds=5,
    iterations=50,
    verbose=1
)

# Run optimization
results = optimizer.staged_parameter_scan(alpha=0.6, beta=0.2, gamma=0.2)

# Show best parameters
best_result = results['best_result']
print(f"Best parameters: max_components={best_result['max_components_setting']}, "
      f"preprocess='{best_result['preprocess_method']}'")
print(f"Selected {best_result['n_selected_vars']} variables with "
      f"{best_result['optimal_components']} components")
print(f"RMSE: {best_result['min_cv_error']:.4f}, R²: {best_result['max_r_squared']:.4f}")

# Plot optimization results
optimizer.plot_optimization_results(results)
```

### Simple All-in-One Workflow with SimpleCARSOptimizer

```python
from SpectralCARSLib import SimpleCARSOptimizer
from sklearn.model_selection import train_test_split

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create SimpleCARSOptimizer with automatic task detection
optimizer = SimpleCARSOptimizer(X_train, y_train)

# Run all-in-one optimization and model building with a predefined recipe
result = optimizer.run(recipe="fast")

# Get selected variables and the model
selected_vars = result['selected_variables']
model = result['model']
print(f"Selected {len(selected_vars)} out of {X_train.shape[1]} variables")

# Make predictions with the final model
y_pred = optimizer.predict(X_test)

# Evaluate model performance
metrics = optimizer.evaluate(X_test, y_test)
print(f"Test set RMSE: {metrics['rmse']:.4f}")
print(f"Test set R²: {metrics['r2']:.4f}")

# Visualize results
optimizer.plot_results()
```

## Documentation

For detailed API documentation and examples, please see the [docs](docs/) directory.

## Citing

If you use this implementation in your research, please cite:

```
@article{
  title={Rapid maize seed vigor classification using deep learning and hyperspectral imaging techniques},
  author={Wongchaisuwat, Papis and Chakranon, Pongsan and Yinpin, Achitpon and Onwimol, Damrongvudhi and Wonggasem, Kris},
  journal={Smart Agricultural Technology},
  volume={10},
  year={2025},
  pages={100820},
  publisher={Elsevier},
  doi={10.1016/j.atech.2025.100820}
}
```

For the original CARS method, please also cite:

```
@article{
  title={Variable selection in visible and near-infrared spectral analysis for noninvasive blood glucose concentration prediction},
  author={Li, Hongdong and Liang, Yizeng and Xu, Qingsong and Cao, Dongsheng},
  journal={Analytica Chimica Acta},
  volume={648},
  number={1},
  pages={77--84},
  year={2009},
  publisher={Elsevier}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
