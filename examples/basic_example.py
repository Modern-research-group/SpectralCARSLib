"""
Basic example demonstrating CARS variable selection on synthetic data.

This example shows how to use the CARS algorithm on a simple synthetic dataset
with 500 variables where only 20 are truly relevant.
"""

import numpy as np
import matplotlib.pyplot as plt
from SpectralCARSLib import competitive_adaptive_sampling, plot_sampling_results

def run_basic_example():
    """Run a basic example of CARS on synthetic data"""
    # Generate synthetic data
    np.random.seed(42)
    n_samples, n_features = 200, 500
    X = np.random.normal(0, 1, (n_samples, n_features))
    
    # Create true coefficients (only first 20 variables are relevant)
    true_coef = np.zeros(n_features)
    true_coef[:20] = np.random.normal(0, 5, 20)
    
    # Generate response with noise
    y = X.dot(true_coef) + np.random.normal(0, 1, n_samples)
    
    print("Running CARS on synthetic data...")
    print(f"Dataset: {n_samples} samples, {n_features} variables")
    print(f"True model: Only first 20 variables are relevant\n")
    
    # Run CARS
    cars_results = competitive_adaptive_sampling(
        X=X,
        y=y,
        max_components=10,
        folds=5,
        preprocess='center',
        iterations=50,  # Reduced for example
        adaptive_resampling=False,
        shuffle_sample_order=False,
        verbose=1
    )
    
    # Get selected variables
    selected_vars = cars_results['selected_variables']
    
    # Calculate performance metrics
    true_positives = np.sum(selected_vars < 20)
    false_positives = np.sum(selected_vars >= 20)
    
    print("\nResults:")
    print(f"Selected {len(selected_vars)} out of {n_features} variables")
    print(f"True positives: {true_positives}/{20}")
    print(f"False positives: {false_positives}/{n_features-20}")
    
    # Plot results
    plot_sampling_results(cars_results)
    plt.show()
    
    return cars_results

if __name__ == "__main__":
    run_basic_example()
