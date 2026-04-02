from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .datasets import load_digits_split
from .metrics import classification_accuracy, majority_class_baseline_accuracy
from .minibatch import iter_minibatches
from .mlp_baseline import MLPNetwork, init_mlp_baseline_layers
from .utils import set_seed

OutputLayout = Literal["single_dir", "run_id_subdir"]


@dataclass
class RealMLPConfig:
    """Configuration for a Phase 3 real-data MLP experiment."""

    experiment_name: str = "digits_mlp"
    dataset_name: str = "digits"
    task_name: str = "classification"
    run_seed: int = 0
    data_seed: int = 0
    model_init_seed: int = 0
    batch_order_seed: int = 0
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: OutputLayout = "single_dir"
    plot_curves: bool = False
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    layer_dims: tuple[int, ...] = (64, 64, 10)
    hidden_activation: str = "tanh"
    output_activation: str = "identity"
    weight_scale: float = 0.05
    eta_w: float = 0.05
    eta_b: float | None = None
    epochs: int = 100
    batch_size: int = 64
    shuffle_batches: bool = True
    logging: dict[str, Any] = field(default_factory=dict)

    def resolved_run_id(self) -> str:
        """Return an explicit run_id or a timestamp-based default."""
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_{self.run_seed}"


@dataclass
class RealMLPRunResult:
    """Materialized outputs of a Phase 3 real-data MLP run."""

    run_dir: Path
    config: dict[str, Any]
    epoch_metrics: list[dict[str, Any]]
    summary: dict[str, Any]


def _resolve_run_dir(
    output_root: str | Path,
    experiment_name: str,
    run_id: str,
    output_layout: OutputLayout,
) -> Path:
    if output_layout == "single_dir":
        return Path(output_root) / experiment_name
    if output_layout == "run_id_subdir":
        return Path(output_root) / experiment_name / run_id
    raise ValueError(f"Unsupported output_layout '{output_layout}'.")


def _prepare_run_dir(run_dir: Path) -> Path:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, indent=2)


def _write_epoch_metrics(path: Path, rows: list[dict[str, Any]], num_layers: int) -> None:
    fieldnames = [
        "epoch",
        "batch_size",
        "batches_per_epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "train_baseline_accuracy",
        "val_baseline_accuracy",
        "accuracy",
        "baseline_accuracy",
    ]
    fieldnames.extend([f"weight_norm_l{layer_index}" for layer_index in range(1, num_layers + 1)])
    fieldnames.extend([f"bias_norm_l{layer_index}" for layer_index in range(1, num_layers + 1)])

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _evaluate_model(
    model: MLPNetwork,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    """Return MSE loss and argmax accuracy for batch-first arrays."""
    predictions = model.predict(np.asarray(x, dtype=np.float64))
    targets = np.asarray(y, dtype=np.float64)
    loss = float(np.mean((predictions - targets) ** 2))
    accuracy = classification_accuracy(predictions, targets)
    return loss, accuracy


def _snapshot_parameters(model: MLPNetwork) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(layer.weight.copy(), layer.bias.copy()) for layer in model.layers]


def _restore_parameters(model: MLPNetwork, snapshot: list[tuple[np.ndarray, np.ndarray]]) -> None:
    if len(snapshot) != len(model.layers):
        raise ValueError("Parameter snapshot must align with model layers.")
    for layer, (weight, bias) in zip(model.layers, snapshot, strict=True):
        layer.weight = weight.copy()
        layer.bias = bias.copy()


def _config_payload(
    config: RealMLPConfig,
    run_id: str,
    data_metadata: dict[str, Any],
    batches_per_epoch: int,
) -> dict[str, Any]:
    return {
        "experiment_name": config.experiment_name,
        "run_id": run_id,
        "seed": config.run_seed,
        "seeds": {
            "run_seed": config.run_seed,
            "data_seed": config.data_seed,
            "model_init_seed": config.model_init_seed,
            "batch_order_seed": config.batch_order_seed,
        },
        "task": {
            "name": config.task_name,
            "primary_metric_name": "accuracy",
            "primary_metric_higher_is_better": True,
            "selection_metric_source": "val_metric",
            "report_metric_source": "test_metric",
        },
        "data": dict(data_metadata),
        "model": {
            "model_family": "mlp",
            "layer_dims": list(config.layer_dims),
            "hidden_activation": config.hidden_activation,
            "output_activation": config.output_activation,
            "weight_scale": config.weight_scale,
            "model_init_seed": config.model_init_seed,
        },
        "training": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "batches_per_epoch": batches_per_epoch,
            "eta_w": config.eta_w,
            "eta_b": config.eta_w if config.eta_b is None else config.eta_b,
            "shuffle_batches": config.shuffle_batches,
            "batch_order_seed": config.batch_order_seed,
        },
        "logging": {
            "output_root": str(config.output_root),
            "output_layout": config.output_layout,
            "plot_curves": config.plot_curves,
            **config.logging,
        },
    }


def _plot_learning_curves(run_dir: Path, epoch_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "plot_curves=True requires matplotlib to be installed in the current environment."
        ) from exc

    epochs = np.array([row["epoch"] for row in epoch_rows], dtype=np.int64)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    figure = plt.figure()
    axis = figure.add_subplot(1, 1, 1)
    axis.plot(epochs, [row["train_loss"] for row in epoch_rows], label="train_loss")
    axis.plot(epochs, [row["val_loss"] for row in epoch_rows], label="val_loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE loss")
    axis.set_title("Digits MLP Loss Curves")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots_dir / "loss_curves.png")
    plt.close(figure)

    figure = plt.figure()
    axis = figure.add_subplot(1, 1, 1)
    axis.plot(epochs, [row["train_accuracy"] for row in epoch_rows], label="train_accuracy")
    axis.plot(epochs, [row["val_accuracy"] for row in epoch_rows], label="val_accuracy")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy")
    axis.set_title("Digits MLP Accuracy Curves")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots_dir / "accuracy_curves.png")
    plt.close(figure)


def run_digits_mlp_experiment(config: RealMLPConfig) -> RealMLPRunResult:
    """Run a deterministic Phase 3a digits MLP baseline experiment."""
    if config.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    set_seed(config.run_seed)
    split = load_digits_split(
        split_seed=config.data_seed,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
    )

    model = MLPNetwork(
        layers=init_mlp_baseline_layers(
            config.layer_dims,
            hidden_activation=config.hidden_activation,
            output_activation=config.output_activation,
            weight_scale=config.weight_scale,
            seed=config.model_init_seed,
            dtype=np.float64,
        ),
        eta_w=config.eta_w,
        eta_b=config.eta_b,
    )

    train_baseline_accuracy = majority_class_baseline_accuracy(split.y_train)
    val_baseline_accuracy = majority_class_baseline_accuracy(split.y_val)
    test_baseline_accuracy = majority_class_baseline_accuracy(split.y_test)

    batches_per_epoch = len(
        list(
            iter_minibatches(
                split.x_train,
                split.y_train,
                config.batch_size,
                shuffle=config.shuffle_batches,
                seed=config.batch_order_seed,
            )
        )
    )

    run_id = config.resolved_run_id()
    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, run_id, config.output_layout)
    )

    config_payload = _config_payload(
        config=config,
        run_id=run_id,
        data_metadata=split.metadata,
        batches_per_epoch=batches_per_epoch,
    )
    _write_json(run_dir / "config.json", config_payload)

    best_snapshot: list[tuple[np.ndarray, np.ndarray]] | None = None
    best_epoch = 0
    best_val_accuracy = -np.inf
    best_val_loss = np.inf
    epoch_rows: list[dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        minibatches = iter_minibatches(
            split.x_train,
            split.y_train,
            config.batch_size,
            shuffle=config.shuffle_batches,
            seed=config.batch_order_seed + (epoch - 1),
        )
        for x_batch, y_batch in minibatches:
            model.train_batch(x_batch, y_batch)

        train_loss, train_accuracy = _evaluate_model(model, split.x_train, split.y_train)
        val_loss, val_accuracy = _evaluate_model(model, split.x_val, split.y_val)

        row: dict[str, Any] = {
            "epoch": epoch,
            "batch_size": config.batch_size,
            "batches_per_epoch": batches_per_epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "train_baseline_accuracy": train_baseline_accuracy,
            "val_baseline_accuracy": val_baseline_accuracy,
            "accuracy": val_accuracy,
            "baseline_accuracy": val_baseline_accuracy,
        }
        for layer_index, layer in enumerate(model.layers, start=1):
            row[f"weight_norm_l{layer_index}"] = float(np.linalg.norm(layer.weight))
            row[f"bias_norm_l{layer_index}"] = float(np.linalg.norm(layer.bias))
        epoch_rows.append(row)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            best_epoch = epoch
            best_snapshot = _snapshot_parameters(model)

    if best_snapshot is None:
        raise RuntimeError("No best model snapshot was recorded.")
    _restore_parameters(model, best_snapshot)

    train_loss, train_accuracy = _evaluate_model(model, split.x_train, split.y_train)
    val_loss, val_accuracy = _evaluate_model(model, split.x_val, split.y_val)
    test_loss, test_accuracy = _evaluate_model(model, split.x_test, split.y_test)

    summary = {
        "experiment_name": config.experiment_name,
        "run_id": run_id,
        "phase": "Phase 3",
        "math_version": "phase0-baseline",
        "model_family": "mlp",
        "dataset_name": config.dataset_name,
        "task_name": config.task_name,
        "seed": config.run_seed,
        "seeds": {
            "run_seed": config.run_seed,
            "data_seed": config.data_seed,
            "model_init_seed": config.model_init_seed,
            "batch_order_seed": config.batch_order_seed,
        },
        "metric_name": "accuracy",
        "primary_metric_name": "accuracy",
        "metric_higher_is_better": True,
        "primary_metric_higher_is_better": True,
        "selection_metric_source": "val_metric",
        "selection_metric_value": val_accuracy,
        "report_metric_source": "test_metric",
        "report_metric_value": test_accuracy,
        "train_metric": train_accuracy,
        "val_metric": val_accuracy,
        "test_metric": test_accuracy,
        "primary_metric_value": test_accuracy,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "baseline_metric_name": "baseline_accuracy",
        "train_baseline_metric": train_baseline_accuracy,
        "val_baseline_metric": val_baseline_accuracy,
        "test_baseline_metric": test_baseline_accuracy,
        "baseline_metric_value": test_baseline_accuracy,
        "best_epoch": best_epoch,
        "best_val_metric": best_val_accuracy,
        "best_val_loss": best_val_loss,
        "batch_size": config.batch_size,
        "batches_per_epoch": batches_per_epoch,
        "epochs": config.epochs,
    }

    _write_epoch_metrics(run_dir / "epoch_metrics.csv", epoch_rows, num_layers=len(model.layers))
    _write_json(run_dir / "summary.json", summary)

    if config.plot_curves:
        _plot_learning_curves(run_dir, epoch_rows)

    return RealMLPRunResult(
        run_dir=run_dir,
        config=config_payload,
        epoch_metrics=epoch_rows,
        summary=summary,
    )
