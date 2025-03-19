# SpectralCARSLib: Competitive Adaptive Reweighted Sampling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance implementation of Competitive Adaptive Reweighted Sampling (CARS) for variable selection in PLS regression models, with a focus on spectroscopy applications.

## Overview

CARS is a powerful variable selection method commonly used in chemometrics and spectroscopy data analysis. This implementation offers:

- **High Performance**: Optimized implementation with parallel processing support
- **Flexibility**: Multiple preprocessing options (center, autoscaling, pareto, minmax)
- **GPU Support**: Optional GPU acceleration via cuPy and cuML (when available)
- **Visualization**: Built-in plotting functions for CARS results
- **Comprehensive API**: Easy-to-use interface with extensive documentation

CARS works by competitively eliminating variables with small regression coefficients through a Monte Carlo sampling process, enabling effective identification of the most informative variables for PLS regression models.

## Installation
The minimum version is Python>=3.9.0

Install SpectralCARSLib:
```bash
# Install from PyPI
pip install SpectralCARSLib

# Install with GPU support
pip install SpectralCARSLib[gpu]
```
Clone the repository locally and install with
```bash
git clone https://github.com/Ginnovation-lab/SpectralCARSLib.git
cd cars
pip install -e .
```

## Quick Start

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
from SpectralCARSLib.visualization import plot_sampling_results
plot_sampling_results(cars_results)
```

## Documentation

For detailed API documentation and examples, please see the [docs](docs/) directory.

## Citing

If you use this implementation in your research, please cite:

```
@article{li2009variable,
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
