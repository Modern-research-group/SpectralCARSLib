With this content:

```python
"""
Example demonstrating SimpleCARSOptimizer for all-in-one variable selection and model building.

This example shows how to use the SimpleCARSOptimizer to perform variable selection,
build the final model, and make predictions in a single workflow.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score

from SpectralCARSLib import SimpleCARSOptimizer

def regression_example():
    """Example of SimpleCARSOptimizer for regression task"""
    print("\n=== Regression Example ===\n")
    
    # Generate synthetic regression data
    X, y, coef = make_regression(
        n_samples=200, 
        n_features=200, 
        n_informative=20,
        coef=True,
        random_state=42
    )
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"Generated dataset: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples, {X.shape[1]} features")
    print(f"True informative features: 20")
    
    # Create optimizer with auto task detection
    optimizer = SimpleCARSOptimizer(X_train, y_train, verbose=1)
    
    # Run all-in-one optimization and model building
    print("\nRunning SimpleCARSOptimizer with 'fast' recipe...")
    result = optimizer.run(recipe="fast")
    
    # Get selected variables
    selected_vars = result['selected_variables']
    print(f"\nSelected {len(selected_vars)} variables")
    print(f"True positive rate: {np.sum(selected_vars < 20) / len(selected_vars):.2f}")
    
    # Make predictions with the final model
    y_pred = optimizer.predict(X_test)
    
    # Calculate metrics manually
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nTest set performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Evaluate with built-in method
    metrics = optimizer.evaluate(X_test, y_test)
    
    # Plot results
    optimizer.plot_results()
    
    return optimizer, result

def classification_example():
    """Example of SimpleCARSOptimizer for classification task"""
    print("\n=== Classification Example ===\n")
    
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=200,
        n_features=100,
        n_informative=20,
        n_classes=3,
        random_state=42
    )
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print(f"Generated dataset: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples, {X.shape[1]} features")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Create optimizer with auto task detection
    optimizer = SimpleCARSOptimizer(X_train, y_train, verbose=1)
    
    # Run all-in-one optimization and model building
    print("\nRunning SimpleCARSOptimizer with 'classification' recipe...")
    result = optimizer.run(recipe="classification")
    
    # Get selected variables
    selected_vars = result['selected_variables']
    print(f"\nSelected {len(selected_vars)} variables")
    print(f"True positive rate: {np.sum(selected_vars < 20) / len(selected_vars):.2f}")
    
    # Make predictions with the final model
    y_pred = optimizer.predict(X_test)
    
    # Calculate metrics manually
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nTest set performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Evaluate with built-in method
    metrics = optimizer.evaluate(X_test, y_test)
    
    # Plot results
    optimizer.plot_results()
    
    return optimizer, result

def main():
    """Run all examples"""
    print("SimpleCARSOptimizer Examples")
    print("===========================\n")
    
    # Run regression example
    reg_optimizer, reg_result = regression_example()
    
    # Run classification example
    cls_optimizer, cls_result = classification_example()
    
    print("\nExamples completed successfully!")
    
    return {
        'regression': (reg_optimizer, reg_result),
        'classification': (cls_optimizer, cls_result)
    }

if __name__ == "__main__":
    main()