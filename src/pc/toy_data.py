from __future__ import annotations

import numpy as np

from .utils import make_rng


def make_linear_regression_data(num_points: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic 1D linear regression data shaped (batch, 1)."""
    x = np.linspace(-1.0, 1.0, num_points, dtype=np.float64).reshape(-1, 1)
    y = (0.75 * x) - 0.1
    return x, y


def make_sine_regression_data(num_points: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic 1D nonlinear regression data shaped (batch, 1)."""
    x = np.linspace(-np.pi, np.pi, num_points, dtype=np.float64).reshape(-1, 1)
    y = np.sin(x)
    return x, y


def make_blobs_classification_data(
    seed: int = 7,
    points_per_class: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic 2D three-class blob data with one-hot targets."""
    rng = make_rng(seed)
    centers = np.array(
        [
            [-1.5, -0.5],
            [1.5, -0.25],
            [0.0, 1.75],
        ],
        dtype=np.float64,
    )
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for class_index, center in enumerate(centers):
        class_features = center + rng.normal(
            loc=0.0,
            scale=0.35,
            size=(points_per_class, 2),
        )
        class_labels = np.full((points_per_class,), class_index, dtype=np.int64)
        features.append(class_features.astype(np.float64, copy=False))
        labels.append(class_labels)

    x = np.vstack(features)
    class_ids = np.concatenate(labels)
    permutation = rng.permutation(x.shape[0])
    x = x[permutation]
    class_ids = class_ids[permutation]

    y = np.zeros((x.shape[0], centers.shape[0]), dtype=np.float64)
    y[np.arange(x.shape[0]), class_ids] = 1.0
    return x, y
