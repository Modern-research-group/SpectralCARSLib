"""
Example demonstrating CARS variable selection on synthetic spectral data.

This example shows how to use the CARS algorithm on synthetic spectral data,
which is more representative of real-world applications in spectroscopy.
"""

import numpy as np
import matplotlib.pyplot as plt
from SpectralCARSLib import competitive_adaptive_sampling, plot_sampling_results, plot_selected_variables

def generate_simple_spectral_data(n_samples=100, n_wavelengths=200, snr=10):
    """
    Generate simple synthetic spectral data with:
    - Smooth baseline
    - Three active peaks that contribute to y
    - Correlated neighboring variables (wavelengths)
    - Controlled noise level
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_wavelengths : int
        Number of wavelengths (variables)
    snr : float
        Signal-to-noise ratio
    
    Returns:
    --------
    X : array, shape (n_samples, n_wavelengths)
        Spectral data
    y : array, shape (n_samples,)
        Target values
    """
    # Create wavelength axis (like a typical NIR spectrum)
    wavelengths = np.linspace(1000, 2500, n_wavelengths)
    
    # Create empty arrays for data
    X = np.zeros((n_samples, n_wavelengths))
    y = np.zeros(n_samples)
    
    # Define peak centers for active regions
    peak_centers = [n_wavelengths//4, n_wavelengths//2, 3*n_wavelengths//4]
    peak_widths = [15, 10, 20]
    
    # Coefficients for y contribution
    peak_coefficients = [2.5, -1.8, 3.0]
    
    # Create samples with baseline and peaks
    for i in range(n_samples):
        # Simple baseline with slight random variation
        baseline = 0.2 + 0.05 * np.sin(wavelengths/400)
        baseline += np.random.normal(0, 0.02, n_wavelengths)
        
        # Smooth baseline to ensure correlation between adjacent points
        baseline = np.convolve(baseline, np.ones(5)/5, mode='same')
        
        # Add baseline to spectrum
        X[i, :] = baseline
        
        # Add the active peaks with varying intensities
        for center, width, coef in zip(peak_centers, peak_widths, peak_coefficients):
            # Random intensity for this sample's peak
            intensity = np.random.uniform(0.7, 1.3)
            
            # Create gaussian peak
            peak = intensity * np.exp(-0.5 * ((np.arange(n_wavelengths) - center) / (width/2.5))**2)
            
            # Add peak to spectrum
            X[i, :] += peak
            
            # Add weighted contribution to y
            y[i] += coef * intensity
        
        # Add some realistic measurement noise
        signal_power = np.var(X[i, :])
        noise_power = signal_power / snr
        X[i, :] += np.random.normal(0, np.sqrt(noise_power), n_wavelengths)
    
    # Add some noise to y
    y += np.random.normal(0, 0.1 * np.std(y), n_samples)
    
    # Create true coefficients vector for reference
    true_coef = np.zeros(n_wavelengths)
    for center, width, coef in zip(peak_centers, peak_widths, peak_coefficients):
        region = np.arange(max(0, center-width), min(n_wavelengths, center+width+1))
        importance = np.exp(-0.5 * ((region - center) / (width/2.5))**2)
        true_coef[region] = coef * importance / np.max(importance)
    
    return X, y, wavelengths, true_coef, peak_centers, peak_widths

def run_spectral_example():
    """Run CARS on synthetic spectral data"""
    # Generate synthetic spectral data
    np.random.seed(42)
    X, y, wavelengths, true_coef, peak_centers, peak_widths = generate_simple_spectral_data(
        n_samples=200, 
        n_wavelengths=200,
        snr=20
    )
    
    # Plot example spectrum and true coefficients
    plt.figure(figsize=(10, 6))
    plt.subplot(211)
    plt.plot(wavelengths, X[0], 'b-')
    plt.title('Example Synthetic Spectrum')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Absorbance')

    plt.subplot(212)
    plt.plot(wavelengths, true_coef, 'r-')
    plt.title('True Variable Importance')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Coefficient')
    plt.tight_layout()
    plt.show()
    
    print("Running CARS on synthetic spectral data...")
    
    # Run CARS
    cars_results = competitive_adaptive_sampling(
        X=X,
        y=y,
        max_components=10,
        folds=5,
        preprocess='center',
        iterations=50,  # Reduced for example
        adaptive_resampling=False,
        verbose=1
    )
    
    # Get selected variables
    selected_vars = cars_results['selected_variables']
    
    # Check if CARS selected the important regions
    true_positives = 0
    for var in selected_vars:
        for center, width in zip(peak_centers, peak_widths):
            if center-width <= var <= center+width:
                true_positives += 1
                break
    
    # Define important variable regions
    total_important_vars = sum(2*width + 1 for width in peak_widths)
    
    # Calculate performance metrics
    if len(selected_vars) > 0:
        true_positive_rate = true_positives / min(len(selected_vars), total_important_vars)
    else:
        true_positive_rate = 0
    
    print("\nResults:")
    print(f"Selected {len(selected_vars)} out of {X.shape[1]} variables")
    print(f"True positive rate: {true_positive_rate:.2f}")
    
    # Plot results
    plot_sampling_results(cars_results)
    plt.figure()
    plot_selected_variables(X, wavelengths, selected_vars, 
                           title="Selected Variables in Spectral Data")
    plt.show()
    
    return cars_results, wavelengths, X, selected_vars

if __name__ == "__main__":
    run_spectral_example()
