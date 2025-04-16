"""
Example demonstrating CorCARS (Correlation-adjusted CARS) for variable selection.

This example compares standard CARS with CorCARS on synthetic spectral data,
highlighting the benefits of correlation-adjusted component selection.
"""

import numpy as np
import matplotlib.pyplot as plt
from SpectralCARSLib import competitive_adaptive_sampling, competitive_adaptive_reweighted_sampling
from SpectralCARSLib import plot_sampling_results, plot_selected_variables

def generate_spectral_data(n_samples=200, n_wavelengths=300, n_informative=20, noise_level=0.1, random_state=42):
    """
    Generate synthetic spectral data with correlated variables.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_wavelengths : int
        Number of wavelengths (variables)
    n_informative : int
        Number of informative variables
    noise_level : float
        Amount of noise to add
    random_state : int
        Random seed
        
    Returns:
    --------
    X : array
        Spectral data matrix
    y : array
        Target values
    true_variables : array
        Indices of the true informative variables
    """
    np.random.seed(random_state)
    
    # Create wavelength axis
    wavelengths = np.linspace(1000, 2500, n_wavelengths)
    
    # Create empty data matrix
    X = np.zeros((n_samples, n_wavelengths))
    
    # Generate baseline for each sample
    for i in range(n_samples):
        # Random baseline with smooth variation
        baseline = 0.1 + 0.05 * np.sin(wavelengths/300) + 0.02 * np.cos(wavelengths/500)
        baseline += 0.01 * np.random.randn(n_wavelengths)  # Small random noise
        
        # Add baseline to spectrum
        X[i, :] = baseline
    
    # Select indices for informative variables (in clusters to simulate peaks)
    np.random.seed(random_state)
    
    # Create clusters of informative variables (to simulate peaks)
    n_clusters = 4
    variables_per_cluster = n_informative // n_clusters
    
    # Place cluster centers with minimum distance
    min_distance = n_wavelengths // (n_clusters * 2)
    possible_positions = np.arange(min_distance, n_wavelengths - min_distance)
    cluster_centers = np.sort(np.random.choice(possible_positions, size=n_clusters, replace=False))
    
    # Generate clusters around centers
    true_variables = []
    for center in cluster_centers:
        # Create a small cluster around each center
        cluster_size = variables_per_cluster
        cluster_range = np.arange(center - cluster_size//2, center + cluster_size//2)
        # Ensure all indices are within bounds
        cluster_range = cluster_range[(cluster_range >= 0) & (cluster_range < n_wavelengths)]
        true_variables.extend(cluster_range)
    
    # Ensure we have exactly n_informative variables
    if len(true_variables) < n_informative:
        # Add more variables if needed
        additional = np.random.choice(
            [i for i in range(n_wavelengths) if i not in true_variables],
            size=n_informative - len(true_variables),
            replace=False
        )
        true_variables.extend(additional)
    elif len(true_variables) > n_informative:
        # Remove excess variables if needed
        true_variables = true_variables[:n_informative]
    
    true_variables = np.array(sorted(true_variables))
    
    # Generate coefficients for the true variables (with correlation structure)
    true_coef = np.zeros(n_wavelengths)
    
    # For each cluster, assign similar coefficient values
    for center in cluster_centers:
        # Define cluster range
        cluster_range = np.arange(center - variables_per_cluster//2, center + variables_per_cluster//2)
        cluster_range = cluster_range[(cluster_range >= 0) & (cluster_range < n_wavelengths)]
        
        # Generate base coefficient for this cluster
        base_coef = np.random.uniform(-4, 4)
        
        # Assign slightly varying coefficients to variables in this cluster
        for idx in cluster_range:
            if idx in true_variables:
                distance = abs(idx - center)
                # Coefficient decreases with distance from center
                true_coef[idx] = base_coef * (1 - 0.2 * distance/(variables_per_cluster//2))
    
    # Add peaks to the spectra based on the true variables
    for i in range(n_samples):
        for var_idx in true_variables:
            # Add a peak with slight random variation
            peak_width = np.random.randint(5, 15)
            peak_height = 1.0 + 0.2 * np.random.randn()
            
            # Create a Gaussian peak
            for j in range(max(0, var_idx - peak_width), min(n_wavelengths, var_idx + peak_width + 1)):
                X[i, j] += peak_height * np.exp(-0.5 * ((j - var_idx) / (peak_width/2.5))**2)
    
    # Generate target values based on true coefficients
    y = X.dot(true_coef)
    
    # Add noise to target
    y += noise_level * np.std(y) * np.random.randn(n_samples)
    
    return X, y, true_variables, wavelengths

def visualize_data(X, wavelengths, true_variables):
    """Visualize the synthetic spectral data"""
    plt.figure(figsize=(10, 6))
    
    # Plot sample spectra
    for i in range(min(5, X.shape[0])):
        plt.plot(wavelengths, X[i], alpha=0.7)
    
    # Highlight true variables
    y_min, y_max = plt.ylim()
    for var_idx in true_variables:
        plt.axvline(x=wavelengths[var_idx], color='r', alpha=0.2, linestyle=':')
    
    plt.title("Sample Synthetic Spectral Data")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorbance")
    plt.tight_layout()
    plt.show()

def compare_cars_corcars(X, y, wavelengths, true_variables):
    """Compare standard CARS with CorCARS"""
    # Run standard CARS
    print("Running standard CARS...")
    cars_results = competitive_adaptive_sampling(
        X=X,
        y=y,
        max_components=15,
        folds=5,
        preprocess='autoscaling',
        iterations=100,
        adaptive_resampling=False,
        verbose=1
    )
    
    # Run CorCARS
    print("\nRunning CorCARS with correlation-adjusted component selection...")
    corcars_results = competitive_adaptive_reweighted_sampling(
        X=X,
        y=y,
        max_components=15,
        folds=5,
        preprocess='autoscaling',
        iterations=100,
        adaptive_resampling=False,
        use_correlation=True,
        alpha=0.25,
        verbose=1
    )
    
    # Evaluate results
    cars_selected = cars_results['selected_variables']
    corcars_selected = corcars_results['selected_variables']
    
    # Calculate true positive rate (how many of the selected variables are truly informative)
    cars_tp = len(set(cars_selected).intersection(set(true_variables)))
    corcars_tp = len(set(corcars_selected).intersection(set(true_variables)))
    
    cars_tpr = cars_tp / len(cars_selected) if cars_selected.size > 0 else 0
    corcars_tpr = corcars_tp / len(corcars_selected) if corcars_selected.size > 0 else 0
    
    print("\nResults Comparison:")
    print(f"CARS: {len(cars_selected)} variables selected, {cars_tp} true positives, TPR: {cars_tpr:.3f}")
    print(f"CorCARS: {len(corcars_selected)} variables selected, {corcars_tp} true positives, TPR: {corcars_tpr:.3f}")
    print(f"CARS RMSE: {cars_results['min_cv_error']:.4f}, Components: {cars_results['optimal_components']}")
    print(f"CorCARS RMSE: {corcars_results['min_cv_error']:.4f}, Components: {corcars_results['optimal_components']}")
    
    # Visualize CARS results
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plot_selected_variables(X, wavelengths, cars_selected, title="Standard CARS: Selected Variables")
    
    # Highlight true variables
    y_min, y_max = plt.ylim()
    for var_idx in true_variables:
        plt.axvline(x=wavelengths[var_idx], color='g', alpha=0.3, linestyle=':')
    plt.legend(['Mean spectrum', 'Selected variables', 'True variables'])
    
    # Visualize CorCARS results
    plt.subplot(2, 1, 2)
    plot_selected_variables(X, wavelengths, corcars_selected, title="CorCARS: Selected Variables")
    
    # Highlight true variables
    y_min, y_max = plt.ylim()
    for var_idx in true_variables:
        plt.axvline(x=wavelengths[var_idx], color='g', alpha=0.3, linestyle=':')
    plt.legend(['Mean spectrum', 'Selected variables', 'True variables'])
    
    plt.tight_layout()
    plt.show()
    
    # Plot CARS results
    plot_sampling_results(cars_results)
    plt.suptitle("Standard CARS Results", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    
    # Plot CorCARS results
    plot_sampling_results(corcars_results)
    plt.suptitle("CorCARS Results", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    
    return cars_results, corcars_results

def main():
    """Main function to run the example"""
    # Generate synthetic spectral data
    print("Generating synthetic spectral data...")
    X, y, true_variables, wavelengths = generate_spectral_data(
        n_samples=200,
        n_wavelengths=300,
        n_informative=20,
        noise_level=0.1
    )
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} wavelengths")
    print(f"True variables: {len(true_variables)} informative wavelengths")
    
    # Visualize dataset
    visualize_data(X, wavelengths, true_variables)
    
    # Compare CARS and CorCARS
    cars_results, corcars_results = compare_cars_corcars(X, y, wavelengths, true_variables)
    
    return X, y, wavelengths, true_variables, cars_results, corcars_results

if __name__ == "__main__":
    main()
