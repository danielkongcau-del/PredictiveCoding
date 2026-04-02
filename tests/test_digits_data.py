from __future__ import annotations

import numpy as np

from pc.datasets import load_digits_split


def test_load_digits_split_shapes_dtypes_and_one_hot_targets() -> None:
    split = load_digits_split(split_seed=17)

    assert split.x_train.ndim == 2
    assert split.x_val.ndim == 2
    assert split.x_test.ndim == 2
    assert split.y_train.ndim == 2
    assert split.y_val.ndim == 2
    assert split.y_test.ndim == 2

    assert split.x_train.shape[1] == 64
    assert split.x_val.shape[1] == 64
    assert split.x_test.shape[1] == 64
    assert split.y_train.shape[1] == 10
    assert split.y_val.shape[1] == 10
    assert split.y_test.shape[1] == 10

    assert split.x_train.dtype == np.float64
    assert split.x_val.dtype == np.float64
    assert split.x_test.dtype == np.float64
    assert split.y_train.dtype == np.float64
    assert split.y_val.dtype == np.float64
    assert split.y_test.dtype == np.float64

    for targets in (split.y_train, split.y_val, split.y_test):
        assert np.all(np.logical_or(targets == 0.0, targets == 1.0))
        assert np.allclose(targets.sum(axis=1), 1.0)

    assert float(split.x_train.min()) >= 0.0
    assert float(split.x_train.max()) <= 1.0
    assert float(split.x_val.min()) >= 0.0
    assert float(split.x_val.max()) <= 1.0
    assert float(split.x_test.min()) >= 0.0
    assert float(split.x_test.max()) <= 1.0


def test_load_digits_split_is_reproducible_under_fixed_seed() -> None:
    first = load_digits_split(split_seed=23)
    second = load_digits_split(split_seed=23)

    assert np.array_equal(first.x_train, second.x_train)
    assert np.array_equal(first.y_train, second.y_train)
    assert np.array_equal(first.x_val, second.x_val)
    assert np.array_equal(first.y_val, second.y_val)
    assert np.array_equal(first.x_test, second.x_test)
    assert np.array_equal(first.y_test, second.y_test)
    assert first.metadata == second.metadata


def test_load_digits_split_uses_non_overlapping_stratified_splits() -> None:
    split = load_digits_split(split_seed=31)
    metadata = split.metadata

    train_indices = set(metadata["train_indices"])
    val_indices = set(metadata["val_indices"])
    test_indices = set(metadata["test_indices"])

    assert train_indices.isdisjoint(val_indices)
    assert train_indices.isdisjoint(test_indices)
    assert val_indices.isdisjoint(test_indices)

    union_indices = train_indices | val_indices | test_indices
    assert len(union_indices) == metadata["total_size"]

    for targets, class_counts in (
        (split.y_train, metadata["train_class_counts"]),
        (split.y_val, metadata["val_class_counts"]),
        (split.y_test, metadata["test_class_counts"]),
    ):
        recovered_labels = np.argmax(targets, axis=1)
        recovered_counts = np.bincount(recovered_labels, minlength=10).astype(int).tolist()
        assert recovered_counts == class_counts
        assert all(count > 0 for count in class_counts)


def test_load_digits_split_metadata_is_complete_and_self_consistent() -> None:
    split = load_digits_split(split_seed=41)
    metadata = split.metadata

    required_keys = {
        "dataset_name",
        "dataset_loader",
        "total_size",
        "input_dim",
        "target_dim",
        "num_classes",
        "split_fractions",
        "split_seed",
        "split_strategy",
        "normalization",
        "train_size",
        "val_size",
        "test_size",
        "train_class_counts",
        "val_class_counts",
        "test_class_counts",
        "train_indices",
        "val_indices",
        "test_indices",
    }
    assert required_keys.issubset(metadata.keys())

    assert metadata["dataset_name"] == "digits"
    assert metadata["dataset_loader"] == "sklearn.datasets.load_digits"
    assert metadata["input_dim"] == 64
    assert metadata["target_dim"] == 10
    assert metadata["num_classes"] == 10
    assert metadata["split_seed"] == 41
    assert metadata["split_strategy"] == "stratified_train_val_test"
    assert metadata["normalization"]["dtype"] == "float64"
    assert metadata["normalization"]["scale_divisor"] == 16.0

    fractions = metadata["split_fractions"]
    assert fractions == {"train": 0.7, "val": 0.15, "test": 0.15}

    assert metadata["train_size"] == split.x_train.shape[0] == split.y_train.shape[0]
    assert metadata["val_size"] == split.x_val.shape[0] == split.y_val.shape[0]
    assert metadata["test_size"] == split.x_test.shape[0] == split.y_test.shape[0]
    assert metadata["total_size"] == (
        metadata["train_size"] + metadata["val_size"] + metadata["test_size"]
    )

    assert len(metadata["train_class_counts"]) == 10
    assert len(metadata["val_class_counts"]) == 10
    assert len(metadata["test_class_counts"]) == 10
    assert sum(metadata["train_class_counts"]) == metadata["train_size"]
    assert sum(metadata["val_class_counts"]) == metadata["val_size"]
    assert sum(metadata["test_class_counts"]) == metadata["test_size"]
