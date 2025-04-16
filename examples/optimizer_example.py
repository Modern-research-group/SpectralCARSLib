"""
Example demonstrating CARSOptimizer for parameter optimization.

This example shows how to use the CARSOptimizer to find optimal parameters
for CARS, CorCARS, and CARS Classification algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder

from SpectralCARSLib import (
    competitive_adaptive_sampling,
    competitive_adaptive_reweighted_sampling,
    competitive_adaptive_sampling_classification,
    CARSOptimizer,
    plot_sampling_results,
    plot_classification_results
)

def generate_regression_data(n_samples=200, n_features=200, n_informative=20, 
                           noise=0.2, random_state=42):
    """Generate synthetic regression data"""
    X, y, true_coef = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        coef=True,
        random_state=random_state
    )
    
    # Return data and indices of non-zero coefficients
    true_variables = np.nonzero(true_coef)[0]
    return X, y, true_variables

def generate_classification_data(n_samples=200, n_features=200, n_informative=20,
                              n_classes=3, random_state=42):
    """Generate synthetic classification data"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=10,
        n_repeated=0,
        n_classes=n_classes,
        class_sep=1.5,
        random_state=random_state
    )
    
    # Create one-hot encoded target for multi-class
    if n_classes > 2:
        encoder = OneHotEncoder(sparse_output=False)
        y_onehot = encoder.fit_transform(y.reshape(-1, 1))
    else:
        y_onehot = None
    
    return X, y, y_onehot

def evaluate_variable_selection(true_variables, selected_variables, n_features):
    """Evaluate the quality of variable selection"""
    true_set = set(true_variables)
    selected_set = set(selected_variables)
    
    # Calculate metrics
    true_positives = len(true_set.intersection(selected_set))
    false_positives = len(selected_set - true_set)
    false_negatives = len(true_set - selected_set)
    
    # Precision, recall, F1
    precision = true_positives / len(selected_set) if len(selected_set) > 0 else 0
    recall = true_positives / len(true_set) if len(true_set) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Other metrics
    accuracy = (n_features - false_positives - false_negatives) / n_features
    jaccard = len(true_set.intersection(selected_set)) / len(true_set.union(selected_set)) if len(true_set.union(selected_set)) > 0 else 0
    
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'jaccard': jaccard
    }

def optimize_standard_cars():
    """Example of optimizing standard CARS for regression"""
    print("\n=== Optimizing Standard CARS for Regression ===\n")
    
    # Generate synthetic data
    X, y, true_variables = generate_regression_data(
        n_samples=200, n_features=200, n_informative=20, noise=0.2
    )
    
    print(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"True informative variables: {len(true_variables)}")
    
    # Create train/test split for later evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create optimizer with explicit function references
    optimizer = CARSOptimizer(
        X=X_train,
        y=y_train,
        cars_variant='standard',
        component_ranges=[5, 10, 15, 20, 25],
        preprocess_options=['center', 'autoscaling', 'pareto', 'minmax'],
        folds=5,
        iterations=50,
        verbose=1,
        cars_func=competitive_adaptive_sampling,
        plot_func=plot_sampling_results
    )
    
    # Run staged parameter scan
    print("\nRunning staged parameter scan...")
    scan_results = optimizer.staged_parameter_scan(
        alpha=0.6,  # Weight for RMSE
        beta=0.2,   # Weight for number of variables
        gamma=0.2   # Weight for number of components
    )
    
    # Get best parameters and model
    best_result = scan_results['best_result']
    best_vars = best_result['selected_variables']
    
    print(f"\nBest parameters: max_components={best_result['max_components_setting']}, "
          f"preprocess='{best_result['preprocess_method']}'")
    print(f"Selected {len(best_vars)} variables with {best_result['optimal_components']} components")
    
    # Evaluate variable selection quality
    var_metrics = evaluate_variable_selection(true_variables, best_vars, X.shape[1])
    
    print("\nVariable Selection Quality:")
    print(f"True Positives: {var_metrics['true_positives']} / {len(true_variables)}")
    print(f"False Positives: {var_metrics['false_positives']}")
    print(f"Precision: {var_metrics['precision']:.4f}")
    print(f"Recall: {var_metrics['recall']:.4f}")
    print(f"F1 Score: {var_metrics['f1']:.4f}")
    print(f"Jaccard Similarity: {var_metrics['jaccard']:.4f}")
    
    # Try to run Bayesian optimization if scikit-optimize is available
    try:
        print("\nRunning Bayesian optimization...")
        bayes_results = optimizer.bayesian_optimization(n_calls=15)
        
        if bayes_results:
            bayes_best = bayes_results['best_result']
            bayes_vars = bayes_best['selected_variables']
            
            print(f"\nBayesian best parameters: max_components={bayes_best['max_components_setting']}, "
                  f"preprocess='{bayes_best['preprocess_method']}'")
            print(f"Selected {len(bayes_vars)} variables with {bayes_best['optimal_components']} components")
            
            # Compare variable selection with staged scan
            overlap = np.intersect1d(best_vars, bayes_vars)
            
            print(f"\nOverlap between optimization methods: {len(overlap)} variables")
            print(f"Jaccard similarity: {len(overlap) / len(np.union1d(best_vars, bayes_vars)):.4f}")
    except ImportError:
        print("\nSkipped Bayesian optimization as scikit-optimize is not available.")
        bayes_results = None
    
    # Plot optimization results
    plt.figure(figsize=(16, 12))
    optimizer.plot_optimization_results(scan_results)
    plt.tight_layout()
    plt.show()
    
    # Plot best CARS results
    plt.figure(figsize=(12, 15))
    optimizer.plot_best_cars_results(scan_results)
    plt.tight_layout()
    plt.show()
    
    # Return results
    return {
        'optimizer': optimizer,
        'scan_results': scan_results,
        'bayes_results': bayes_results,
        'true_variables': true_variables,
        'var_metrics': var_metrics
    }

def optimize_corcars():
    """Example of optimizing CorCARS for regression with correlated data"""
    print("\n=== Optimizing CorCARS for Regression ===\n")
    
    # Generate synthetic data with correlation structure
    # Note: For a real example, you might want to generate data with 
    # actual correlation structure between variables
    X, y, true_variables = generate_regression_data(
        n_samples=200, n_features=200, n_informative=20, noise=0.2
    )
    
    print(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"True informative variables: {len(true_variables)}")
    
    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create optimizer for CorCARS with explicit function references
    optimizer = CARSOptimizer(
        X=X_train,
        y=y_train,
        cars_variant='corcars',
        component_ranges=[5, 10, 15, 20, 25],
        preprocess_options=['center', 'autoscaling', 'pareto'],
        folds=5,
        iterations=50,
        use_correlation=True,  # Enable correlation adjustment
        alpha=0.25,            # Significance level for F-test
        verbose=1,
        cars_func=competitive_adaptive_reweighted_sampling,
        plot_func=plot_sampling_results
    )
    
    # Run optimization
    print("\nRunning parameter optimization for CorCARS...")
    results = optimizer.staged_parameter_scan(
        alpha=0.6,  # Weight for RMSE
        beta=0.2,   # Weight for number of variables
        gamma=0.2   # Weight for number of components
    )
    
    # Get best parameters
    best_result = results['best_result']
    best_vars = best_result['selected_variables']
    
    print(f"\nBest parameters: max_components={best_result['max_components_setting']}, "
          f"preprocess='{best_result['preprocess_method']}'")
    print(f"Selected {best_result['n_selected_vars']} variables with "
          f"{best_result['optimal_components']} components")
    
    # Evaluate variable selection
    var_metrics = evaluate_variable_selection(true_variables, best_vars, X.shape[1])
    
    print("\nVariable Selection Quality:")
    print(f"True Positives: {var_metrics['true_positives']} / {len(true_variables)}")
    print(f"Precision: {var_metrics['precision']:.4f}")
    print(f"Recall: {var_metrics['recall']:.4f}")
    print(f"F1 Score: {var_metrics['f1']:.4f}")
    
    # Compare with standard CARS
    print("\nComparing with standard CARS...")
    standard_optimizer = CARSOptimizer(
        X=X_train,
        y=y_train,
        cars_variant='standard',
        component_ranges=[best_result['max_components_setting']],  # Use same max_components
        preprocess_options=[best_result['preprocess_method']],     # Use same preprocessing
        folds=5,
        iterations=50,
        verbose=1,
        cars_func=competitive_adaptive_sampling,
        plot_func=plot_sampling_results
    )
    
    standard_results = standard_optimizer.staged_parameter_scan()
    standard_best = standard_results['best_result']
    standard_vars = standard_best['selected_variables']
    
    print(f"\nStandard CARS results:")
    print(f"Selected {len(standard_vars)} variables with {standard_best['optimal_components']} components")
    print(f"RMSE: {standard_best['min_cv_error']:.4f}")
    
    # Compare models
    print(f"\nComparison:")
    print(f"CorCARS: {len(best_vars)} variables, {best_result['optimal_components']} components, RMSE: {best_result['min_cv_error']:.4f}")
    print(f"Standard CARS: {len(standard_vars)} variables, {standard_best['optimal_components']} components, RMSE: {standard_best['min_cv_error']:.4f}")
    
    # Variable selection overlap
    overlap = np.intersect1d(best_vars, standard_vars)
    print(f"Overlap: {len(overlap)} variables")
    print(f"Jaccard similarity: {len(overlap) / len(np.union1d(best_vars, standard_vars)):.4f}")
    
    # Plot optimization results
    plt.figure(figsize=(16, 12))
    optimizer.plot_optimization_results(results)
    plt.tight_layout()
    plt.show()
    
    # Plot best CARS results
    plt.figure(figsize=(12, 15))
    optimizer.plot_best_cars_results(results)
    plt.tight_layout()
    plt.show()
    
    # Return results
    return {
        'corcars_optimizer': optimizer,
        'corcars_results': results,
        'standard_optimizer': standard_optimizer,
        'standard_results': standard_results,
        'true_variables': true_variables,
        'var_metrics': var_metrics
    }

def optimize_cars_classification():
    """Example of optimizing CARS Classification"""
    print("\n=== Optimizing CARS Classification ===\n")
    
    # Generate multi-class classification data
    X, y, y_onehot = generate_classification_data(
        n_samples=300, n_features=200, n_informative=20, n_classes=3
    )
    
    print(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    
    # Create train/test split
    X_train, X_test, y_train, y_test, y_onehot_train, y_onehot_test = train_test_split(
        X, y, y_onehot, test_size=0.3, random_state=42, stratify=y
    )
    
    # Create optimizer for classification with one-hot encoding
    optimizer = CARSOptimizer(
        X=X_train,
        y=y_onehot_train,  # Use one-hot encoded target
        cars_variant='classification',
        component_ranges=[5, 10, 15, 20],
        preprocess_options=['center', 'autoscaling', 'pareto'],
        folds=5,
        iterations=50,
        encoding='onehot',  # Specify encoding
        metric='f1',        # Specify metric
        verbose=1,
        cars_func=competitive_adaptive_sampling_classification,
        plot_func=plot_classification_results
    )
    
    # Run optimization
    print("\nRunning parameter optimization for CARS Classification...")
    results = optimizer.staged_parameter_scan(
        alpha=0.6,  # Weight for classification metric
        beta=0.2,   # Weight for number of variables
        gamma=0.2   # Weight for number of components
    )
    
    # Get best parameters
    best_result = results['best_result']
    best_vars = best_result['selected_variables']
    
    print(f"\nBest parameters: max_components={best_result['max_components_setting']}, "
          f"preprocess='{best_result['preprocess_method']}'")
    print(f"Selected {best_result['n_selected_vars']} variables with "
          f"{best_result['optimal_components']} components")
    print(f"F1 Score: {best_result['best_metric_value']:.4f}")
    
    # Plot optimization results for one-hot encoding
    plt.figure(figsize=(16, 12))
    optimizer.plot_optimization_results(results)
    plt.tight_layout()
    plt.show()
    
    # Plot best CARS results for one-hot encoding
    plt.figure(figsize=(15, 15))
    optimizer.plot_best_cars_results(results)
    plt.suptitle('One-hot Encoding Results', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    
    print("\nComparison with ordinal encoding:")
    print("For brevity in this example, we'll just use the one-hot encoding results")
    print("You can uncomment the ordinal encoding code if needed")
    
    # Note: In a real application, you would compare with ordinal encoding
    # as shown in the commented code below
    
    '''
    # Create optimizer for classification with ordinal encoding
    optimizer_ordinal = CARSOptimizer(
        X=X_train,
        y=y_train,  # Use ordinal encoded target
        cars_variant='classification',
        component_ranges=[5, 10, 15, 20],
        preprocess_options=['center', 'autoscaling'],
        folds=5,
        iterations=50,
        encoding='ordinal',
        metric='accuracy',
        verbose=1,
        cars_func=competitive_adaptive_sampling_classification,
        plot_func=plot_classification_results
    )
    
    # Run optimization
    ordinal_results = optimizer_ordinal.staged_parameter_scan(
        alpha=0.6, beta=0.2, gamma=0.2
    )
    
    # Get best parameters
    ordinal_best = ordinal_results['best_result']
    ordinal_vars = ordinal_best['selected_variables']
    
    print(f"\nBest parameters (ordinal): max_components={ordinal_best['max_components_setting']}, "
          f"preprocess='{ordinal_best['preprocess_method']}'")
    print(f"Selected {ordinal_best['n_selected_vars']} variables with "
          f"{ordinal_best['optimal_components']} components")
    print(f"Accuracy: {ordinal_best['best_metric_value']:.4f}")
    
    # Compare variable selection
    overlap = np.intersect1d(best_vars, ordinal_vars)
    
    print(f"\nVariable selection overlap: {len(overlap)} variables")
    print(f"Jaccard similarity: {len(overlap) / len(np.union1d(best_vars, ordinal_vars)):.4f}")
    
    # Plot optimization results for ordinal encoding
    plt.figure(figsize=(16, 12))
    optimizer_ordinal.plot_optimization_results(ordinal_results)
    plt.tight_layout()
    plt.show()
    
    # Plot best CARS results for ordinal encoding
    plt.figure(figsize=(15, 15))
    optimizer_ordinal.plot_best_cars_results(ordinal_results)
    plt.suptitle('Ordinal Encoding Results', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    '''
    
    # For this example, create a simulated ordinal result for the return value
    simulated_ordinal = {
        'optimizer': None,
        'results': None,
    }
    
    # Return results with just one-hot encoding
    return {
        'onehot_optimizer': optimizer,
        'onehot_results': results,
        'ordinal_optimizer': simulated_ordinal['optimizer'],
        'ordinal_results': simulated_ordinal['results']
    }

def main():
    """Run all optimizer examples"""
    print("SpectralCARSLib - Optimizer Examples")
    print("====================================")
    
    # Optimize standard CARS
    cars_results = optimize_standard_cars()
    
    # Optimize CorCARS
    corcars_results = optimize_corcars()
    
    # Optimize CARS Classification
    cls_results = optimize_cars_classification()
    
    print("\n=== Optimization Results Summary ===")
    
    # Standard CARS summary
    cars_best = cars_results['scan_results']['best_result']
    print("\nStandard CARS:")
    print(f"  Best parameters: max_components={cars_best['max_components_setting']}, "
          f"preprocess='{cars_best['preprocess_method']}'")
    print(f"  Selected variables: {cars_best['n_selected_vars']}")
    print(f"  Components: {cars_best['optimal_components']}")
    print(f"  RMSE: {cars_best['min_cv_error']:.4f}")
    print(f"  Variable selection - F1 Score: {cars_results['var_metrics']['f1']:.4f}")
    
    # CorCARS summary
    corcars_best = corcars_results['corcars_results']['best_result']
    print("\nCorCARS:")
    print(f"  Best parameters: max_components={corcars_best['max_components_setting']}, "
          f"preprocess='{corcars_best['preprocess_method']}'")
    print(f"  Selected variables: {corcars_best['n_selected_vars']}")
    print(f"  Components: {corcars_best['optimal_components']}")
    print(f"  RMSE: {corcars_best['min_cv_error']:.4f}")
    print(f"  Variable selection - F1 Score: {corcars_results['var_metrics']['f1']:.4f}")
    
    # Classification summary
    onehot_best = cls_results['onehot_results']['best_result']
    
    print("\nCARS Classification (one-hot):")
    print(f"  Best parameters: max_components={onehot_best['max_components_setting']}, "
          f"preprocess='{onehot_best['preprocess_method']}'")
    print(f"  Selected variables: {onehot_best['n_selected_vars']}")
    print(f"  Components: {onehot_best['optimal_components']}")
    print(f"  F1 Score: {onehot_best['best_metric_value']:.4f}")
    
    print("\nCARS Classification (ordinal):")
    print("  Results skipped in this example to avoid import issues")
    
    return {
        'cars': cars_results,
        'corcars': corcars_results,
        'classification': cls_results
    }

if __name__ == "__main__":
    main()
