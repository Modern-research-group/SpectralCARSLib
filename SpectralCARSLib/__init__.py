"""
SpectralCARSLib: Competitive Adaptive Reweighted Sampling for variable selection in PLS regression.

This package provides optimized implementations of the CARS algorithm family
for variable selection in PLS regression models, particularly for spectroscopy applications.
"""

__version__ = '4.5.0'
__author__ = 'Special Research Unit of Big Data Analytics in Food, Agriculture and Health, Kasetsart University'

# Import preprocessing utilities
from .preprocessing import preprocess_data

# Import utility functions
from .utils import suppress_pls_warnings

# Import visualization functions
from .visualization import (
    plot_sampling_results,
    plot_selected_variables,
    plot_classification_results
)

# Import main functions from standard CARS
from .cars import competitive_adaptive_sampling

# Import functions from CorCARS
from .corcars import (
    competitive_adaptive_reweighted_sampling,
    corcars  # Alias for competitive_adaptive_reweighted_sampling
)

# Import functions from CARS Classification
from .classification import (
    competitive_adaptive_sampling_classification,
    generate_binary_classification_data,
    generate_multiclass_classification_data
)

# Import optimizer
from .optimizer import CARSOptimizer
from .simple_optimizer import SimpleCARSOptimizer