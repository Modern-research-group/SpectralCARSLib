# Recommended Metrics for CARS Classification

## Binary Classification
For binary classification problems (2 classes):

| Metric      | Description                                                 | Best for                                            |
|-------------|-------------------------------------------------------------|-----------------------------------------------------|
| `accuracy`  | Proportion of correctly classified instances                | Balanced datasets                                   |
| `f1`        | Harmonic mean of precision and recall                       | Imbalanced datasets                                 |
| `auc`       | Area Under the ROC Curve                                    | When ranking performance is important               |

**Recommended default**: `f1` for imbalanced data, `accuracy` for balanced data


## Multi-class Classification

### With One-Hot Encoding
For multi-class problems using one-hot encoding:

| Metric      | Description                                                 | Best for                                            |
|-------------|-------------------------------------------------------------|-----------------------------------------------------|
| `f1`        | Weighted F1 score across all classes                        | General purpose, handles class imbalance            |
| `accuracy`  | Proportion of correctly classified instances                | Balanced datasets                                   |

**Recommended default**: `f1` (weighted)

### With Ordinal Encoding
For multi-class problems using ordinal encoding:

| Metric      | Description                                                 | Best for                                            |
|-------------|-------------------------------------------------------------|-----------------------------------------------------|
| `accuracy`  | Proportion of correctly classified instances                | Most multi-class ordinal problems                   |
| `f1`        | Weighted F1 score across all classes                        | When class imbalance exists                         |

**Recommended default**: `accuracy`

## Example Usage

```python
# Binary classification with imbalanced data
result = competitive_adaptive_sampling_classification(
    X=X, y=y,
    encoding='ordinal',
    metric='f1'
)

# Multi-class with one-hot encoding
result = competitive_adaptive_sampling_classification(
    X=X, y=y_onehot,
    encoding='onehot',
    metric='f1'
)

# Multi-class with ordinal encoding
result = competitive_adaptive_sampling_classification(
    X=X, y=y,
    encoding='ordinal',
    metric='accuracy'
)
```

## Performance Comparison

For best results when choosing between encoding methods and metrics:

1. Run the algorithm with different configurations
2. Compare the selected variables and performance results
3. Choose the configuration that provides the best cross-validation performance
4. Validate on an independent test set
