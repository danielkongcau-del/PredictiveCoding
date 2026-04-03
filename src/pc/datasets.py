from __future__ import annotations

from pathlib import Path
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


def _build_stratified_split_indices(
    labels: np.ndarray,
    *,
    split_seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic stratified train/val/test indices."""
    from sklearn.model_selection import train_test_split

    _validate_split_fractions(train_fraction, val_fraction, test_fraction)

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

    return (
        np.sort(train_indices),
        np.sort(val_indices),
        np.sort(test_indices),
    )


def _build_supervised_split(
    *,
    x_all: np.ndarray,
    labels: np.ndarray,
    dataset_name: str,
    dataset_loader: str,
    split_seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    normalization: dict[str, Any],
    extra_metadata: dict[str, Any] | None = None,
) -> SupervisedDataSplit:
    """Build a deterministic stratified split with batch-first float64 arrays."""
    x_array = np.asarray(x_all, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    if x_array.ndim != 2:
        raise ValueError("x_all must be shaped (batch, features).")
    if label_array.ndim != 1:
        raise ValueError("labels must be shaped (batch,).")
    if x_array.shape[0] != label_array.shape[0]:
        raise ValueError("x_all and labels must have the same number of samples.")

    train_indices, val_indices, test_indices = _build_stratified_split_indices(
        label_array,
        split_seed=split_seed,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )

    num_classes = int(np.max(label_array)) + 1

    x_train = x_array[train_indices]
    x_val = x_array[val_indices]
    x_test = x_array[test_indices]

    train_labels = label_array[train_indices]
    val_labels = label_array[val_indices]
    test_labels = label_array[test_indices]

    y_train = _one_hot_encode(train_labels, num_classes)
    y_val = _one_hot_encode(val_labels, num_classes)
    y_test = _one_hot_encode(test_labels, num_classes)

    metadata: dict[str, Any] = {
        "dataset_name": dataset_name,
        "dataset_loader": dataset_loader,
        "total_size": int(x_array.shape[0]),
        "input_dim": int(x_array.shape[1]),
        "target_dim": num_classes,
        "num_classes": num_classes,
        "split_fractions": {
            "train": float(train_fraction),
            "val": float(val_fraction),
            "test": float(test_fraction),
        },
        "split_seed": int(split_seed),
        "split_strategy": "stratified_train_val_test",
        "normalization": dict(normalization),
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
    if extra_metadata is not None:
        metadata.update(extra_metadata)

    return SupervisedDataSplit(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        metadata=metadata,
    )


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

    dataset = load_digits()
    x_all = np.asarray(dataset.data, dtype=np.float64) / 16.0
    labels = np.asarray(dataset.target, dtype=np.int64)
    return _build_supervised_split(
        x_all=x_all,
        labels=labels,
        dataset_name="digits",
        dataset_loader="sklearn.datasets.load_digits",
        split_seed=split_seed,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        normalization={
            "dtype": "float64",
            "formula": "x.astype(float64) / 16.0",
            "scale_divisor": 16.0,
        },
    )


def load_fashion_mnist_split(
    *,
    split_seed: int = 0,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    data_home: str | Path | None = None,
    cache: bool = True,
) -> SupervisedDataSplit:
    """Load Fashion-MNIST as a deterministic stratified train/val/test split.

    Shapes:
    - `x_train`: `(batch_train, 784)`
    - `y_train`: `(batch_train, 10)`
    - `x_val`: `(batch_val, 784)`
    - `y_val`: `(batch_val, 10)`
    - `x_test`: `(batch_test, 784)`
    - `y_test`: `(batch_test, 10)`
    """
    from sklearn.datasets import fetch_openml

    dataset = fetch_openml(
        name="Fashion-MNIST",
        version=1,
        as_frame=False,
        parser="auto",
        data_home=None if data_home is None else str(data_home),
        cache=cache,
    )
    x_all = np.asarray(dataset.data, dtype=np.float64) / 255.0
    labels = np.asarray(dataset.target, dtype=np.int64)
    return _build_supervised_split(
        x_all=x_all,
        labels=labels,
        dataset_name="fashion_mnist",
        dataset_loader="sklearn.datasets.fetch_openml",
        split_seed=split_seed,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        normalization={
            "dtype": "float64",
            "formula": "x.astype(float64) / 255.0",
            "scale_divisor": 255.0,
        },
        extra_metadata={
            "openml_dataset_name": "Fashion-MNIST",
            "openml_version": 1,
            "data_home": None if data_home is None else str(data_home),
            "cache": bool(cache),
        },
    )
