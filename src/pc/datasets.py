from __future__ import annotations

from typing import Any

import numpy as np

from .toy_data import SupervisedDataSplit


def _validate_split_fractions(
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> None:
    """Validate requested split fractions."""
    fractions = (train_fraction, val_fraction, test_fraction)
    if any(fraction <= 0.0 for fraction in fractions):
        raise ValueError("train_fraction, val_fraction, and test_fraction must all be positive.")
    total = train_fraction + val_fraction + test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError("train_fraction + val_fraction + test_fraction must sum to 1.0.")


def _one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Return one-hot targets shaped (batch, num_classes) with dtype float64."""
    return np.eye(num_classes, dtype=np.float64)[labels]


def _class_counts(labels: np.ndarray, num_classes: int) -> list[int]:
    """Return per-class counts as a length-num_classes Python list."""
    return np.bincount(labels, minlength=num_classes).astype(int).tolist()


def load_digits_split(
    *,
    split_seed: int = 0,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> SupervisedDataSplit:
    """Load sklearn digits as a deterministic stratified train/val/test split.

    Shapes:
    - `x_train`: `(batch_train, 64)`
    - `y_train`: `(batch_train, 10)`
    - `x_val`: `(batch_val, 64)`
    - `y_val`: `(batch_val, 10)`
    - `x_test`: `(batch_test, 64)`
    - `y_test`: `(batch_test, 10)`
    """
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    _validate_split_fractions(train_fraction, val_fraction, test_fraction)

    dataset = load_digits()
    x_all = np.asarray(dataset.data, dtype=np.float64) / 16.0
    labels = np.asarray(dataset.target, dtype=np.int64)
    indices = np.arange(labels.shape[0], dtype=np.int64)

    holdout_fraction = val_fraction + test_fraction
    train_indices, holdout_indices = train_test_split(
        indices,
        test_size=holdout_fraction,
        random_state=split_seed,
        stratify=labels,
    )

    holdout_labels = labels[holdout_indices]
    test_fraction_within_holdout = test_fraction / holdout_fraction
    val_indices, test_indices = train_test_split(
        holdout_indices,
        test_size=test_fraction_within_holdout,
        random_state=split_seed,
        stratify=holdout_labels,
    )

    train_indices = np.sort(train_indices)
    val_indices = np.sort(val_indices)
    test_indices = np.sort(test_indices)

    num_classes = int(np.max(labels)) + 1

    x_train = x_all[train_indices]
    x_val = x_all[val_indices]
    x_test = x_all[test_indices]

    train_labels = labels[train_indices]
    val_labels = labels[val_indices]
    test_labels = labels[test_indices]

    y_train = _one_hot_encode(train_labels, num_classes)
    y_val = _one_hot_encode(val_labels, num_classes)
    y_test = _one_hot_encode(test_labels, num_classes)

    metadata: dict[str, Any] = {
        "dataset_name": "digits",
        "dataset_loader": "sklearn.datasets.load_digits",
        "total_size": int(x_all.shape[0]),
        "input_dim": int(x_all.shape[1]),
        "target_dim": num_classes,
        "num_classes": num_classes,
        "split_fractions": {
            "train": float(train_fraction),
            "val": float(val_fraction),
            "test": float(test_fraction),
        },
        "split_seed": int(split_seed),
        "split_strategy": "stratified_train_val_test",
        "normalization": {
            "dtype": "float64",
            "formula": "x.astype(float64) / 16.0",
            "scale_divisor": 16.0,
        },
        "train_size": int(train_indices.shape[0]),
        "val_size": int(val_indices.shape[0]),
        "test_size": int(test_indices.shape[0]),
        "train_class_counts": _class_counts(train_labels, num_classes),
        "val_class_counts": _class_counts(val_labels, num_classes),
        "test_class_counts": _class_counts(test_labels, num_classes),
        "train_indices": train_indices.astype(int).tolist(),
        "val_indices": val_indices.astype(int).tolist(),
        "test_indices": test_indices.astype(int).tolist(),
    }

    return SupervisedDataSplit(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        metadata=metadata,
    )
