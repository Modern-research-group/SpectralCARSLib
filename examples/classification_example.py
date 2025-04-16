"""
Example demonstrating CARS Classification for variable selection in classification problems.

This example shows how to use the CARS Classification algorithm with both
binary and multi-class datasets, using different encodings and metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.cross_decomposition import PLSRegression

from SpectralCARSLib import competitive_adaptive_sampling_classification
from SpectralCARSLib import plot_classification_results

def generate_binary_classification_data(n_samples=200, n_features=200, n_informative=20, random_state=42):
    """Generate synthetic binary classification data with informative features"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=10,
        n_repeated=0,
        n_classes=2,
        class_sep=2.0,
        random_state=random_state
    )
    
    return X, y

def generate_multiclass_classification_data(n_samples=300, n_features=200, n_informative=20, 
                                          n_classes=4, random_state=42):
    """Generate synthetic multi-class classification data"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=10,
        n_repeated=0,
        n_classes=n_classes,
        random_state=random_state,
        class_sep=1.5
    )
    
    # Create one-hot encoded version of the target
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y.reshape(-1, 1))
    
    return X, y, y_onehot

def evaluate_model(X, y, selected_variables, n_components=None, test_size=0.3, 
                 encoding='ordinal', random_state=42):
    """
    Evaluate selected variables on test data
    
    Parameters:
    -----------
    X : array
        Feature matrix
    y : array
        Target values
    selected_variables : array
        Indices of selected variables
    n_components : int, optional
        Number of PLS components to use (defaults to min(10, len(selected_variables)))
    test_size : float
        Proportion of data to use for testing
    encoding : str
        Target encoding ('ordinal' or 'onehot')
    random_state : int
        Random seed for train-test split
    
    Returns:
    --------
    dict : Dictionary of evaluation metrics
    """
    # Handle empty variable selection
    if len(selected_variables) == 0:
        return {
            'accuracy': 0,
            'f1_score': 0,
            'confusion_matrix': None,
            'n_variables': 0,
            'n_components': 0
        }
    
    # Default number of components if not specified
    if n_components is None:
        n_components = min(10, len(selected_variables))
    
    # Split data
    X_selected = X[:, selected_variables]
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=test_size, random_state=random_state, stratify=y if encoding == 'ordinal' else None
    )
    
    # Train PLS model
    pls = PLSRegression(n_components=n_components)
    
    # Handle different encodings
    if encoding == 'ordinal':
        # For ordinal encoding, reshape y_train if needed
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
        
        # Fit the model
        pls.fit(X_train, y_train)
        
        # Make predictions
        y_pred_raw = pls.predict(X_test)
        
        # Get unique classes
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        
        if n_classes == 2:
            # Binary classification with threshold at 0.5
            y_pred = (y_pred_raw > 0.5).astype(int)
        else:
            # Multi-class - round to nearest class
            y_pred = np.round(y_pred_raw).clip(0, n_classes-1).astype(int)
        
    else:  # onehot encoding
        # Fit the model (y_train is already one-hot encoded)
        pls.fit(X_train, y_train)
        
        # Make predictions
        y_pred_raw = pls.predict(X_test)
        
        # Check if y_pred_raw is 1D (when only one component is used or output is flattened)
        if len(y_pred_raw.shape) == 1:
            # Handle 1D case - reshape to have second dimension
            y_pred_raw = y_pred_raw.reshape(-1, 1)
        
        # Get class with highest score
        y_pred = np.argmax(y_pred_raw, axis=1)
        
        # Convert y_test from one-hot to class indices if needed
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_test = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    if len(np.unique(y_test)) > 2:
        f1 = f1_score(y_test, y_pred, average='weighted')
    else:
        f1 = f1_score(y_test, y_pred)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': acc,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'n_variables': len(selected_variables),
        'n_components': n_components
    }

def run_binary_classification_example():
    """Run example for binary classification"""
    print("\n=== Binary Classification Example ===\n")
    
    # Generate binary classification data
    X, y = generate_binary_classification_data(n_samples=200, n_features=200, n_informative=20)
    
    print(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")
    unique, counts = np.unique(y, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    print(f"Class distribution: {class_distribution}")
    
    # Run CARS for binary classification
    print("\nRunning CARS for binary classification...")
    binary_results = competitive_adaptive_sampling_classification(
        X=X,
        y=y,
        max_components=10,
        folds=5,
        preprocess='autoscaling',
        iterations=50,
        encoding='ordinal',
        metric='f1',
        adaptive_resampling=False,
        verbose=1
    )
    
    # Evaluate model with selected variables
    selected_vars = binary_results['selected_variables']
    print(f"\nSelected {len(selected_vars)} out of {X.shape[1]} variables")
    print(f"Optimal components: {binary_results['optimal_components']}")
    
    # Test model performance
    eval_results = evaluate_model(
        X=X, 
        y=y, 
        selected_variables=selected_vars,
        n_components=binary_results['optimal_components'],
        encoding='ordinal'
    )
    
    print("\nTest set performance:")
    print(f"Accuracy: {eval_results['accuracy']:.4f}")
    print(f"F1 Score: {eval_results['f1_score']:.4f}")
    
    # Plot results
    plot_classification_results(binary_results)
    
    return X, y, binary_results, eval_results

def run_multiclass_classification_example():
    """Run example for multi-class classification with both encodings"""
    print("\n=== Multi-class Classification Example ===\n")
    
    # Generate multi-class data
    X, y, y_onehot = generate_multiclass_classification_data(
        n_samples=300, 
        n_features=200, 
        n_informative=20, 
        n_classes=4
    )
    
    print(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    unique, counts = np.unique(y, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    print(f"Class distribution: {class_distribution}")
    
    # Run with ordinal encoding
    print("\nRunning CARS with ordinal encoding...")
    ordinal_results = competitive_adaptive_sampling_classification(
        X=X,
        y=y,  # Use ordinal encoding
        max_components=15,
        folds=5,
        preprocess='autoscaling',
        iterations=50,
        encoding='ordinal',
        metric='accuracy',
        adaptive_resampling=False,
        verbose=1
    )
    
    # Run with one-hot encoding
    print("\nRunning CARS with one-hot encoding...")
    onehot_results = competitive_adaptive_sampling_classification(
        X=X,
        y=y_onehot,  # Use one-hot encoding
        max_components=15,
        folds=5,
        preprocess='autoscaling',
        iterations=50,
        encoding='onehot',
        metric='f1',
        adaptive_resampling=False,
        verbose=1
    )
    
    # Evaluate and compare results
    ordinal_vars = ordinal_results['selected_variables']
    onehot_vars = onehot_results['selected_variables']
    
    print("\nResults Comparison:")
    print(f"Ordinal encoding: {len(ordinal_vars)} variables, {ordinal_results['optimal_components']} components")
    print(f"One-hot encoding: {len(onehot_vars)} variables, {onehot_results['optimal_components']} components")
    
    # Evaluate models
    ordinal_eval = evaluate_model(
        X=X, 
        y=y, 
        selected_variables=ordinal_vars,
        n_components=ordinal_results['optimal_components'],
        encoding='ordinal'
    )
    
    onehot_eval = evaluate_model(
        X=X, 
        y=y, 
        selected_variables=onehot_vars,
        n_components=onehot_results['optimal_components'],
        encoding='onehot'
    )
    
    print("\nTest Set Performance:")
    print(f"Ordinal encoding - Accuracy: {ordinal_eval['accuracy']:.4f}, F1: {ordinal_eval['f1_score']:.4f}")
    print(f"One-hot encoding - Accuracy: {onehot_eval['accuracy']:.4f}, F1: {onehot_eval['f1_score']:.4f}")
    
    # Compare variable selection
    overlap = np.intersect1d(ordinal_vars, onehot_vars)
    print(f"\nOverlap between encodings: {len(overlap)} variables")
    
    # Variable selection similarity
    jaccard = len(overlap) / len(np.union1d(ordinal_vars, onehot_vars)) if len(np.union1d(ordinal_vars, onehot_vars)) > 0 else 0
    print(f"Jaccard similarity: {jaccard:.4f}")
    
    # Plot results
    plot_classification_results(ordinal_results)
    plt.suptitle("Ordinal Encoding Results", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    
    plot_classification_results(onehot_results)
    plt.suptitle("One-hot Encoding Results", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    
    return X, y, y_onehot, ordinal_results, onehot_results, ordinal_eval, onehot_eval

def main():
    """Run the full classification example suite"""
    # Run binary classification example
    binary_X, binary_y, binary_results, binary_eval = run_binary_classification_example()
    
    # Run multi-class classification example
    multi_X, multi_y, multi_y_onehot, ordinal_results, onehot_results, ordinal_eval, onehot_eval = run_multiclass_classification_example()
    
    print("\n=== Summary ===")
    print("Binary classification:")
    print(f"  Selected {len(binary_results['selected_variables'])} variables")
    print(f"  Test accuracy: {binary_eval['accuracy']:.4f}")
    
    print("\nMulti-class with ordinal encoding:")
    print(f"  Selected {len(ordinal_results['selected_variables'])} variables")
    print(f"  Test accuracy: {ordinal_eval['accuracy']:.4f}")
    
    print("\nMulti-class with one-hot encoding:")
    print(f"  Selected {len(onehot_results['selected_variables'])} variables")
    print(f"  Test accuracy: {onehot_eval['accuracy']:.4f}")
    
    return {
        'binary': {
            'X': binary_X,
            'y': binary_y,
            'results': binary_results,
            'eval': binary_eval
        },
        'multiclass': {
            'X': multi_X,
            'y': multi_y,
            'y_onehot': multi_y_onehot,
            'ordinal_results': ordinal_results,
            'onehot_results': onehot_results,
            'ordinal_eval': ordinal_eval,
            'onehot_eval': onehot_eval
        }
    }

if __name__ == "__main__":
    main()
