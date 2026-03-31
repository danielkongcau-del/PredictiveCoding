from __future__ import annotations

from collections.abc import Callable

import numpy as np

MetricFn = Callable[[np.ndarray, np.ndarray], float]
BaselineMetricFn = Callable[[np.ndarray], float]


def regression_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Return mean squared error for arrays shaped (batch, features)."""
    return float(np.mean((predictions - targets) ** 2))


def regression_mean_baseline_mse(targets: np.ndarray) -> float:
    """Return the MSE of predicting the per-feature mean target value."""
    mean_prediction = np.mean(targets, axis=0, keepdims=True)
    return float(np.mean((targets - mean_prediction) ** 2))


def classification_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Return argmax accuracy for predictions and one-hot targets shaped (batch, classes)."""
    predicted_labels = np.argmax(predictions, axis=1)
    target_labels = np.argmax(targets, axis=1)
    return float(np.mean(predicted_labels == target_labels))


def majority_class_baseline_accuracy(targets: np.ndarray) -> float:
    """Return the accuracy of always predicting the majority class from one-hot targets."""
    target_labels = np.argmax(targets, axis=1)
    class_counts = np.bincount(target_labels, minlength=targets.shape[1])
    return float(np.max(class_counts) / target_labels.shape[0])


def metric_higher_is_better(metric_name: str) -> bool:
    """Return whether larger metric values indicate better model performance."""
    mapping = {
        "accuracy": True,
        "mse": False,
    }
    try:
        return mapping[metric_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported metric '{metric_name}'.") from exc
