"""
SpectralCARSLib: Competitive Adaptive Reweighted Sampling for variable selection in PLS regression.

This package provides an optimized implementation of the CARS algorithm
for variable selection in PLS regression models, particularly for spectroscopy applications.
"""

__version__ = '4.3.0'
__author__ = 'Department of Industrial Engineering and Department of Agronomy, Kasetsart University'

# Import main functions for convenient access
from .core import competitive_adaptive_sampling
from .preprocessing import preprocess_data
from .visualization import plot_sampling_results, plot_selected_variables
