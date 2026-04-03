from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pc.datasets import load_fashion_mnist_split
from pc.real_pc import RealPCConfig, run_real_pc_experiment
from pc.toy_data import SupervisedDataSplit

ROOT = Path(__file__).resolve().parents[1]


def _make_fake_fashion_mnist_dataset(samples_per_class: int = 20) -> SimpleNamespace:
    labels = np.repeat(np.arange(10, dtype=np.int64), samples_per_class)
    features = (
        (labels[:, None] * 17 + np.arange(784, dtype=np.int64)[None, :]) % 256
    ).astype(np.float64)
    targets = labels.astype(str)
    return SimpleNamespace(data=features, target=targets)


def test_load_fashion_mnist_split_shapes_dtypes_one_hot_and_normalization_range(
    monkeypatch,
) -> None:
    def fake_fetch_openml(*, name, version, as_frame, parser, data_home, cache):
        assert name == "Fashion-MNIST"
        assert version == 1
        assert as_frame is False
        _ = (parser, data_home, cache)
        return _make_fake_fashion_mnist_dataset()

    monkeypatch.setattr("sklearn.datasets.fetch_openml", fake_fetch_openml)

    split = load_fashion_mnist_split(split_seed=17)

    assert split.x_train.shape[1] == 784
    assert split.x_val.shape[1] == 784
    assert split.x_test.shape[1] == 784
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

    for features in (split.x_train, split.x_val, split.x_test):
        assert float(features.min()) >= 0.0
        assert float(features.max()) <= 1.0


def test_load_fashion_mnist_split_is_reproducible_under_fixed_seed(monkeypatch) -> None:
    monkeypatch.setattr(
        "sklearn.datasets.fetch_openml",
        lambda **_: _make_fake_fashion_mnist_dataset(),
    )

    first = load_fashion_mnist_split(split_seed=23)
    second = load_fashion_mnist_split(split_seed=23)

    assert np.array_equal(first.x_train, second.x_train)
    assert np.array_equal(first.y_train, second.y_train)
    assert np.array_equal(first.x_val, second.x_val)
    assert np.array_equal(first.y_val, second.y_val)
    assert np.array_equal(first.x_test, second.x_test)
    assert np.array_equal(first.y_test, second.y_test)
    assert first.metadata == second.metadata


def test_load_fashion_mnist_split_uses_non_overlapping_stratified_splits_and_complete_metadata(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "sklearn.datasets.fetch_openml",
        lambda **_: _make_fake_fashion_mnist_dataset(samples_per_class=24),
    )

    split = load_fashion_mnist_split(split_seed=31, data_home=Path("fashion_cache"))
    metadata = split.metadata

    train_indices = set(metadata["train_indices"])
    val_indices = set(metadata["val_indices"])
    test_indices = set(metadata["test_indices"])

    assert train_indices.isdisjoint(val_indices)
    assert train_indices.isdisjoint(test_indices)
    assert val_indices.isdisjoint(test_indices)
    assert len(train_indices | val_indices | test_indices) == metadata["total_size"]

    assert metadata["dataset_name"] == "fashion_mnist"
    assert metadata["dataset_loader"] == "sklearn.datasets.fetch_openml"
    assert metadata["input_dim"] == 784
    assert metadata["target_dim"] == 10
    assert metadata["num_classes"] == 10
    assert metadata["split_seed"] == 31
    assert metadata["split_strategy"] == "stratified_train_val_test"
    assert metadata["normalization"]["dtype"] == "float64"
    assert metadata["normalization"]["formula"] == "x.astype(float64) / 255.0"
    assert metadata["normalization"]["scale_divisor"] == 255.0
    assert metadata["openml_dataset_name"] == "Fashion-MNIST"
    assert metadata["openml_version"] == 1
    assert metadata["data_home"] == "fashion_cache"
    assert metadata["cache"] is True

    for targets, class_counts in (
        (split.y_train, metadata["train_class_counts"]),
        (split.y_val, metadata["val_class_counts"]),
        (split.y_test, metadata["test_class_counts"]),
    ):
        recovered_labels = np.argmax(targets, axis=1)
        recovered_counts = np.bincount(recovered_labels, minlength=10).astype(int).tolist()
        assert recovered_counts == class_counts
        assert all(count > 0 for count in class_counts)


def test_real_pc_can_route_to_fashion_mnist_split_and_write_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    x_train = np.linspace(0.0, 1.0, 40 * 784, dtype=np.float64).reshape(40, 784)
    train_labels = np.arange(40, dtype=np.int64) % 10
    x_val = np.linspace(0.1, 0.9, 20 * 784, dtype=np.float64).reshape(20, 784)
    val_labels = np.arange(20, dtype=np.int64) % 10
    x_test = np.linspace(0.2, 0.8, 20 * 784, dtype=np.float64).reshape(20, 784)
    test_labels = np.arange(20, dtype=np.int64) % 10
    eye = np.eye(10, dtype=np.float64)
    split = SupervisedDataSplit(
        x_train=x_train,
        y_train=eye[train_labels],
        x_val=x_val,
        y_val=eye[val_labels],
        x_test=x_test,
        y_test=eye[test_labels],
        metadata={
            "dataset_name": "fashion_mnist",
            "dataset_loader": "stub",
            "total_size": 80,
            "input_dim": 784,
            "target_dim": 10,
            "num_classes": 10,
        },
    )

    monkeypatch.setattr("pc.real_pc.load_fashion_mnist_split", lambda **_: split)

    result = run_real_pc_experiment(
        RealPCConfig(
            experiment_name="fashion_mnist_pc_test",
            dataset_name="fashion_mnist",
            output_root=tmp_path,
            run_id="fashion_smoke",
            layer_dims=(784, 16, 10),
            epochs=1,
            batch_size=20,
            train_steps=1,
            eval_steps=1,
        )
    )

    assert result.run_dir == tmp_path / "fashion_mnist_pc_test"
    assert result.summary["dataset_name"] == "fashion_mnist"
    assert (result.run_dir / "config.json").exists()
    assert (result.run_dir / "epoch_metrics.csv").exists()
    assert (result.run_dir / "summary.json").exists()


def test_fashion_mnist_pc_script_exposes_runnable_entrypoint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    eye = np.eye(10, dtype=np.float64)
    split = SupervisedDataSplit(
        x_train=np.zeros((20, 784), dtype=np.float64),
        y_train=eye[np.arange(20) % 10],
        x_val=np.zeros((10, 784), dtype=np.float64),
        y_val=eye[np.arange(10) % 10],
        x_test=np.zeros((10, 784), dtype=np.float64),
        y_test=eye[np.arange(10) % 10],
        metadata={
            "dataset_name": "fashion_mnist",
            "dataset_loader": "stub",
            "total_size": 40,
            "input_dim": 784,
            "target_dim": 10,
            "num_classes": 10,
        },
    )

    monkeypatch.setattr("pc.real_pc.load_fashion_mnist_split", lambda **_: split)

    module = runpy.run_path(str(ROOT / "experiments" / "fashion_mnist_pc.py"))
    run = module["run"]
    result = run(output_root=tmp_path, run_id="fashion_script_smoke", plot_curves=False)

    assert result.run_dir == tmp_path / "fashion_mnist_pc"
    assert result.summary["dataset_name"] == "fashion_mnist"
