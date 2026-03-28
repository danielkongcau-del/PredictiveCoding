from __future__ import annotations

import numpy as np

from pc.metrics import (
    classification_accuracy,
    majority_class_baseline_accuracy,
    regression_mean_baseline_mse,
    regression_mse,
)


def test_regression_mse_matches_manual_value() -> None:
    predictions = np.array([[1.0], [3.0]], dtype=np.float64)
    targets = np.array([[2.0], [1.0]], dtype=np.float64)
    assert regression_mse(predictions, targets) == 2.5


def test_regression_mean_baseline_mse_matches_manual_value() -> None:
    targets = np.array([[0.0], [2.0], [4.0]], dtype=np.float64)
    assert regression_mean_baseline_mse(targets) == (8.0 / 3.0)


def test_classification_accuracy_uses_argmax() -> None:
    predictions = np.array(
        [
            [0.1, 0.7, 0.2],
            [0.8, 0.1, 0.1],
            [0.2, 0.2, 0.6],
        ],
        dtype=np.float64,
    )
    targets = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    assert classification_accuracy(predictions, targets) == (2.0 / 3.0)


def test_majority_class_baseline_accuracy_matches_class_frequency() -> None:
    targets = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    assert majority_class_baseline_accuracy(targets) == 0.75
