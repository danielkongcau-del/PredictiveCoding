from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .utils import make_rng


@dataclass(frozen=True)
class SupervisedDataSplit:
    """Deterministic train/val/test split with batch-first arrays.

    Shapes:
    - `x_train`: `(batch_train, features)`
    - `y_train`: `(batch_train, targets)`
    - `x_val`: `(batch_val, features)`
    - `y_val`: `(batch_val, targets)`
    - `x_test`: `(batch_test, features)`
    - `y_test`: `(batch_test, targets)`
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def x_eval(self) -> np.ndarray:
        """Backward-compatible alias for validation features shaped (batch, features)."""
        return self.x_val

    @property
    def y_eval(self) -> np.ndarray:
        """Backward-compatible alias for validation targets shaped (batch, targets)."""
        return self.y_val


def _make_offset_grid(
    start: float,
    stop: float,
    num_points: int,
    *,
    offset_fraction: float,
) -> np.ndarray:
    """Return a dense 1D grid with a fractional offset from the interval boundaries."""
    if num_points <= 0:
        raise ValueError("num_points must be positive.")
    if not (0.0 < offset_fraction < 1.0):
        raise ValueError("offset_fraction must lie strictly between 0 and 1.")
    step = (stop - start) / float(num_points)
    first = start + (offset_fraction * step)
    last = stop - ((1.0 - offset_fraction) * step)
    return np.linspace(first, last, num_points, dtype=np.float64).reshape(-1, 1)


def make_linear_regression_data(
    num_points: int = 16,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic 1D linear regression data shaped (batch, 1)."""
    _ = seed
    x = np.linspace(-1.0, 1.0, num_points, dtype=np.float64).reshape(-1, 1)
    y = (0.75 * x) - 0.1
    return x, y


def make_linear_regression_split(
    num_points: int = 16,
    val_num_points: int = 129,
    test_num_points: int = 129,
    seed: int | None = None,
) -> SupervisedDataSplit:
    """Create deterministic train/val/test regression arrays with distinct x locations."""
    x_train, y_train = make_linear_regression_data(num_points=num_points, seed=seed)
    x_val = _make_offset_grid(-1.0, 1.0, val_num_points, offset_fraction=0.5)
    y_val = (0.75 * x_val) - 0.1
    x_test = _make_offset_grid(-1.0, 1.0, test_num_points, offset_fraction=0.25)
    y_test = (0.75 * x_test) - 0.1
    return SupervisedDataSplit(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        metadata={
            "evaluation_protocol": "dense_offset_val_test_grids",
            "train_num_points": int(num_points),
            "val_num_points": int(val_num_points),
            "test_num_points": int(test_num_points),
            "val_grid_offset_fraction": 0.5,
            "test_grid_offset_fraction": 0.25,
        },
    )


def make_sine_regression_data(
    num_points: int = 32,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic 1D nonlinear regression data shaped (batch, 1)."""
    _ = seed
    x = np.linspace(-np.pi, np.pi, num_points, dtype=np.float64).reshape(-1, 1)
    y = np.sin(x)
    return x, y


def make_sine_regression_split(
    num_points: int = 32,
    val_num_points: int = 257,
    test_num_points: int = 257,
    seed: int | None = None,
) -> SupervisedDataSplit:
    """Create deterministic train/val/test sine-regression arrays with distinct x locations."""
    x_train, y_train = make_sine_regression_data(num_points=num_points, seed=seed)
    x_val = _make_offset_grid(-np.pi, np.pi, val_num_points, offset_fraction=0.5)
    y_val = np.sin(x_val)
    x_test = _make_offset_grid(-np.pi, np.pi, test_num_points, offset_fraction=0.25)
    y_test = np.sin(x_test)
    return SupervisedDataSplit(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        metadata={
            "evaluation_protocol": "dense_offset_val_test_grids",
            "train_num_points": int(num_points),
            "val_num_points": int(val_num_points),
            "test_num_points": int(test_num_points),
            "val_grid_offset_fraction": 0.5,
            "test_grid_offset_fraction": 0.25,
        },
    )


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


def make_blobs_classification_split(
    seed: int = 7,
    points_per_class: int = 24,
    val_points_per_class: int = 48,
    test_points_per_class: int = 48,
) -> SupervisedDataSplit:
    """Create deterministic train/val/test blobs data with separately sampled held-out points."""
    x_train, y_train = make_blobs_classification_data(seed=seed, points_per_class=points_per_class)
    val_seed = int(seed) + 1
    test_seed = int(seed) + 2
    x_val, y_val = make_blobs_classification_data(
        seed=val_seed,
        points_per_class=val_points_per_class,
    )
    x_test, y_test = make_blobs_classification_data(
        seed=test_seed,
        points_per_class=test_points_per_class,
    )
    return SupervisedDataSplit(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        metadata={
            "evaluation_protocol": "separately_sampled_val_test_sets",
            "train_points_per_class": int(points_per_class),
            "val_points_per_class": int(val_points_per_class),
            "test_points_per_class": int(test_points_per_class),
            "train_data_seed": int(seed),
            "val_data_seed": val_seed,
            "test_data_seed": test_seed,
        },
    )
