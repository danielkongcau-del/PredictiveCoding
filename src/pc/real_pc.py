from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np

from .datasets import load_digits_split, load_fashion_mnist_split
from .inference import build_clamped_mask, initialize_states, run_inference, run_teacher_inference_export
from .layers import init_mlp_layers
from .metrics import (
    classification_accuracy,
    majority_class_baseline_accuracy,
    summarize_teacher_reference_metrics,
)
from .minibatch import iter_minibatches
from .models import PCNetwork
from .state_io import flatten_hidden_states
from .toy_data import SupervisedDataSplit
from .training import TrainBatchResult
from .utils import set_seed

OutputLayout = Literal["single_dir", "run_id_subdir"]
_TEACHER_REFERENCE_DISABLE_REASON = (
    "Standalone real-data evaluation omits teacher-reference metrics by default because "
    "predict-mode candidate-vs-teacher comparisons are often trivial under forward initialization "
    "and do not match FMPC training-time target-clamped teacher semantics."
)


@dataclass
class RealPCConfig:
    """Configuration for a Phase 3 real-data predictive-coding experiment."""

    experiment_name: str = "digits_pc"
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
    sigma2: float | tuple[float, ...] = 1.0
    eta_x: float = 0.10
    eta_w: float = 0.02
    eta_b: float | None = 0.02
    train_steps: int = 30
    eval_steps: int | None = 30
    inference_backend: Literal["pc_euler", "pc_rk2", "fmpc"] = "pc_euler"
    inference_method: Literal["euler", "rk2"] | None = None
    teacher_reference_backend: Literal["pc_euler", "pc_rk2"] | None = None
    teacher_reference_eval_steps: int | None = None
    state_init: str = "forward"
    epochs: int = 60
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
class RealPCRunResult:
    """Materialized outputs of a Phase 3 real-data PC run."""

    run_dir: Path
    config: dict[str, Any]
    epoch_metrics: list[dict[str, Any]]
    summary: dict[str, Any]
    model: PCNetwork | None = None
    split: SupervisedDataSplit | None = None


@dataclass
class _PCSplitEvaluation:
    """Evaluation outputs for one batch-first split under the current PC protocol."""

    loss: float
    accuracy: float
    final_energy: float
    candidate_wall_time_seconds: float
    teacher_reference: dict[str, Any] | None


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
        "train_steps",
        "eval_steps",
        "inference_backend",
        "inference_method",
        "state_init",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "train_baseline_accuracy",
        "val_baseline_accuracy",
        "accuracy",
        "baseline_accuracy",
        "train_mean_pre_update_energy",
        "train_mean_post_update_energy",
        "train_mean_energy_delta",
    ]
    fieldnames.extend([f"weight_norm_l{layer_index}" for layer_index in range(1, num_layers + 1)])
    fieldnames.extend([f"bias_norm_l{layer_index}" for layer_index in range(1, num_layers + 1)])

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _evaluate_pc_split(
    model: PCNetwork,
    x: np.ndarray,
    y: np.ndarray,
    *,
    teacher_reference_backend: Literal["pc_euler", "pc_rk2"] | None,
    teacher_reference_steps: int,
) -> _PCSplitEvaluation:
    """Evaluate one split and optionally compare it against an explicit slow-teacher reference.

    Shape contract:
    - `x`: `(batch, input_dim)`
    - `y`: `(batch, target_dim)`
    """
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    clamped_mask = build_clamped_mask(len(model.layers) + 1, mode="predict")

    candidate_initial_states = initialize_states(
        model.layers,
        x_array,
        init=model.state_init,
        mode="predict",
    )
    candidate_start = perf_counter()
    candidate_result = run_inference(
        candidate_initial_states,
        model.layers,
        clamped_mask,
        eta_x=model.eta_x,
        steps=model.eval_steps,
        backend=model.inference_backend,
        record_trace=False,
    )
    candidate_wall_time_seconds = float(perf_counter() - candidate_start)

    predictions = candidate_result.states[-1]
    loss = float(np.mean((predictions - y_array) ** 2))
    accuracy = classification_accuracy(predictions, y_array)

    teacher_reference: dict[str, Any] | None = None
    if teacher_reference_backend is not None:
        teacher_start = perf_counter()
        teacher_export = run_teacher_inference_export(
            model.layers,
            x_array,
            init=model.state_init,
            mode="predict",
            eta_x=model.eta_x,
            steps=teacher_reference_steps,
            backend=teacher_reference_backend,
            record_trace=True,
            record_trajectory=False,
        )
        teacher_wall_time_seconds = float(perf_counter() - teacher_start)
        candidate_z0 = flatten_hidden_states(candidate_initial_states, clamped_mask)
        candidate_z_terminal = flatten_hidden_states(candidate_result.states, clamped_mask)
        teacher_reference = {
            "candidate_backend": model.inference_backend,
            "candidate_inference_method": model.inference_method,
            "candidate_steps": int(model.eval_steps),
            "candidate_inference_wall_time_seconds": candidate_wall_time_seconds,
            "teacher_backend": teacher_export.inference_backend,
            "teacher_inference_method": teacher_export.inference_method,
            "teacher_steps": teacher_export.steps,
            "teacher_inference_wall_time_seconds": teacher_wall_time_seconds,
            **summarize_teacher_reference_metrics(
                candidate_z0=candidate_z0,
                candidate_z_terminal=candidate_z_terminal,
                candidate_final_energy=candidate_result.final_energy,
                teacher_z0=teacher_export.z0,
                teacher_z_terminal=teacher_export.z_star,
                teacher_final_energy=teacher_export.energy_trace[-1],
            ),
        }

    return _PCSplitEvaluation(
        loss=loss,
        accuracy=accuracy,
        final_energy=float(candidate_result.final_energy),
        candidate_wall_time_seconds=candidate_wall_time_seconds,
        teacher_reference=teacher_reference,
    )


def _snapshot_parameters(model: PCNetwork) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(layer.weight.copy(), layer.bias.copy()) for layer in model.layers]


def _restore_parameters(model: PCNetwork, snapshot: list[tuple[np.ndarray, np.ndarray]]) -> None:
    if len(snapshot) != len(model.layers):
        raise ValueError("Parameter snapshot must align with model layers.")
    for layer, (weight, bias) in zip(model.layers, snapshot, strict=True):
        layer.weight = weight.copy()
        layer.bias = bias.copy()


def _sigma2_payload(sigma2: float | tuple[float, ...]) -> float | list[float]:
    if isinstance(sigma2, tuple):
        return [float(value) for value in sigma2]
    return float(sigma2)


def _config_payload(
    config: RealPCConfig,
    run_id: str,
    data_metadata: dict[str, Any],
    batches_per_epoch: int,
    model: PCNetwork,
) -> dict[str, Any]:
    return {
        "experiment_name": config.experiment_name,
        "run_id": run_id,
        "seed": config.run_seed,
        "run_seed": config.run_seed,
        "data_seed": config.data_seed,
        "model_init_seed": config.model_init_seed,
        "batch_order_seed": config.batch_order_seed,
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
            "model_family": "pc",
            "layer_dims": list(config.layer_dims),
            "hidden_activation": config.hidden_activation,
            "output_activation": config.output_activation,
            "weight_scale": config.weight_scale,
            "sigma2": _sigma2_payload(config.sigma2),
            "model_init_seed": config.model_init_seed,
        },
        "training": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "batches_per_epoch": batches_per_epoch,
            "eta_x": config.eta_x,
            "eta_w": config.eta_w,
            "eta_b": model.eta_b,
            "train_steps": config.train_steps,
            "eval_steps": model.eval_steps,
            "inference_backend": model.inference_backend,
            "inference_method": model.inference_method,
            "state_init": config.state_init,
            "shuffle_batches": config.shuffle_batches,
            "batch_order_seed": config.batch_order_seed,
        },
        "evaluation": {
            "teacher_reference_backend": config.teacher_reference_backend,
            "teacher_reference_eval_steps": config.teacher_reference_eval_steps,
            "teacher_reference_metrics_enabled": False,
            "teacher_reference_disable_reason": _TEACHER_REFERENCE_DISABLE_REASON,
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
    axis.set_title("Digits PC Loss Curves")
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
    axis.set_title("Digits PC Accuracy Curves")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots_dir / "accuracy_curves.png")
    plt.close(figure)


def _mean_batch_energy(train_batch_results: list[TrainBatchResult]) -> tuple[float, float, float]:
    pre = np.array([result.pre_update_energy for result in train_batch_results], dtype=np.float64)
    post = np.array(
        [0.0 if result.post_update_energy is None else result.post_update_energy for result in train_batch_results],
        dtype=np.float64,
    )
    return float(np.mean(pre)), float(np.mean(post)), float(np.mean(post - pre))


def _load_split_for_config(config: RealPCConfig):
    if config.dataset_name == "digits":
        return load_digits_split(
            split_seed=config.data_seed,
            train_fraction=config.train_fraction,
            val_fraction=config.val_fraction,
            test_fraction=config.test_fraction,
        )
    if config.dataset_name == "fashion_mnist":
        return load_fashion_mnist_split(
            split_seed=config.data_seed,
            train_fraction=config.train_fraction,
            val_fraction=config.val_fraction,
            test_fraction=config.test_fraction,
        )
    raise ValueError(f"Unsupported dataset_name '{config.dataset_name}'.")


def run_real_pc_experiment(
    config: RealPCConfig,
    *,
    return_runtime_objects: bool = False,
) -> RealPCRunResult:
    """Run a deterministic real-data predictive-coding experiment."""
    if config.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.train_steps < 0:
        raise ValueError("train_steps must be non-negative.")
    if config.eval_steps is not None and config.eval_steps < 0:
        raise ValueError("eval_steps must be non-negative when provided.")
    if config.teacher_reference_eval_steps is not None and config.teacher_reference_eval_steps < 0:
        raise ValueError("teacher_reference_eval_steps must be non-negative when provided.")

    set_seed(config.run_seed)
    split = _load_split_for_config(config)
    input_dim = int(split.metadata.get("input_dim", split.x_train.shape[1]))
    target_dim = int(split.metadata.get("target_dim", split.y_train.shape[1]))
    if config.layer_dims[0] != input_dim:
        raise ValueError(
            f"layer_dims[0] must match dataset input_dim {input_dim}, "
            f"received {config.layer_dims[0]}."
        )
    if config.layer_dims[-1] != target_dim:
        raise ValueError(
            f"layer_dims[-1] must match dataset target_dim {target_dim}, "
            f"received {config.layer_dims[-1]}."
        )

    model = PCNetwork(
        layers=init_mlp_layers(
            config.layer_dims,
            hidden_activation=config.hidden_activation,
            output_activation=config.output_activation,
            weight_scale=config.weight_scale,
            sigma2=config.sigma2,
            seed=config.model_init_seed,
            dtype=np.float64,
        ),
        eta_x=config.eta_x,
        eta_w=config.eta_w,
        eta_b=config.eta_b,
        train_steps=config.train_steps,
        eval_steps=config.eval_steps,
        inference_backend=config.inference_backend,
        inference_method=config.inference_method,
        state_init=config.state_init,
    )

    train_baseline_accuracy = majority_class_baseline_accuracy(split.y_train)
    val_baseline_accuracy = majority_class_baseline_accuracy(split.y_val)
    test_baseline_accuracy = majority_class_baseline_accuracy(split.y_test)

    batches_per_epoch = sum(
        1
        for _ in iter_minibatches(
            split.x_train,
            split.y_train,
            config.batch_size,
            shuffle=config.shuffle_batches,
            seed=config.batch_order_seed,
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
        model=model,
    )
    _write_json(run_dir / "config.json", config_payload)

    best_snapshot: list[tuple[np.ndarray, np.ndarray]] | None = None
    best_epoch = 0
    best_val_accuracy = -np.inf
    best_val_loss = np.inf
    epoch_rows: list[dict[str, Any]] = []
    train_wall_time_start = perf_counter()

    for epoch in range(1, config.epochs + 1):
        batch_results: list[TrainBatchResult] = []
        minibatches = iter_minibatches(
            split.x_train,
            split.y_train,
            config.batch_size,
            shuffle=config.shuffle_batches,
            seed=config.batch_order_seed + (epoch - 1),
        )
        for x_batch, y_batch in minibatches:
            batch_results.append(model.train_batch(x_batch, y_batch, compute_post_update_energy=True))

        train_eval = _evaluate_pc_split(
            model,
            split.x_train,
            split.y_train,
            teacher_reference_backend=None,
            teacher_reference_steps=0,
        )
        val_eval = _evaluate_pc_split(
            model,
            split.x_val,
            split.y_val,
            teacher_reference_backend=None,
            teacher_reference_steps=0,
        )
        mean_pre_update_energy, mean_post_update_energy, mean_energy_delta = _mean_batch_energy(batch_results)

        row: dict[str, Any] = {
            "epoch": epoch,
            "batch_size": config.batch_size,
            "batches_per_epoch": batches_per_epoch,
            "train_steps": config.train_steps,
            "eval_steps": model.eval_steps,
            "inference_backend": model.inference_backend,
            "inference_method": model.inference_method,
            "state_init": config.state_init,
            "train_loss": train_eval.loss,
            "train_accuracy": train_eval.accuracy,
            "val_loss": val_eval.loss,
            "val_accuracy": val_eval.accuracy,
            "train_baseline_accuracy": train_baseline_accuracy,
            "val_baseline_accuracy": val_baseline_accuracy,
            "accuracy": val_eval.accuracy,
            "baseline_accuracy": val_baseline_accuracy,
            "train_mean_pre_update_energy": mean_pre_update_energy,
            "train_mean_post_update_energy": mean_post_update_energy,
            "train_mean_energy_delta": mean_energy_delta,
        }
        for layer_index, layer in enumerate(model.layers, start=1):
            row[f"weight_norm_l{layer_index}"] = float(np.linalg.norm(layer.weight))
            row[f"bias_norm_l{layer_index}"] = float(np.linalg.norm(layer.bias))
        epoch_rows.append(row)

        if val_eval.accuracy > best_val_accuracy:
            best_val_accuracy = val_eval.accuracy
            best_val_loss = val_eval.loss
            best_epoch = epoch
            best_snapshot = _snapshot_parameters(model)

    train_wall_time_seconds = float(perf_counter() - train_wall_time_start)

    if best_snapshot is None:
        raise RuntimeError("No best model snapshot was recorded.")
    _restore_parameters(model, best_snapshot)

    train_eval = _evaluate_pc_split(
        model,
        split.x_train,
        split.y_train,
        teacher_reference_backend=None,
        teacher_reference_steps=0,
    )
    val_eval = _evaluate_pc_split(
        model,
        split.x_val,
        split.y_val,
        teacher_reference_backend=None,
        teacher_reference_steps=0,
    )
    test_eval = _evaluate_pc_split(
        model,
        split.x_test,
        split.y_test,
        teacher_reference_backend=None,
        teacher_reference_steps=0,
    )

    teacher_reference_summary = {
        "enabled": False,
        "reason": _TEACHER_REFERENCE_DISABLE_REASON,
        "requested_teacher_backend": config.teacher_reference_backend,
        "requested_teacher_eval_steps": config.teacher_reference_eval_steps,
    }

    final_eval_wall_time_seconds = float(
        train_eval.candidate_wall_time_seconds
        + val_eval.candidate_wall_time_seconds
        + test_eval.candidate_wall_time_seconds
    )
    teacher_reference_wall_time_seconds = None

    summary = {
        "experiment_name": config.experiment_name,
        "run_id": run_id,
        "phase": "Phase 3",
        "math_version": "phase0-baseline",
        "model_family": "pc",
        "dataset_name": config.dataset_name,
        "task_name": config.task_name,
        "seed": config.run_seed,
        "run_seed": config.run_seed,
        "data_seed": config.data_seed,
        "model_init_seed": config.model_init_seed,
        "batch_order_seed": config.batch_order_seed,
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
        "selection_metric_value": val_eval.accuracy,
        "report_metric_source": "test_metric",
        "report_metric_value": test_eval.accuracy,
        "train_metric": train_eval.accuracy,
        "val_metric": val_eval.accuracy,
        "test_metric": test_eval.accuracy,
        "primary_metric_value": test_eval.accuracy,
        "train_loss": train_eval.loss,
        "val_loss": val_eval.loss,
        "test_loss": test_eval.loss,
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
        "eta_x": config.eta_x,
        "eta_w": config.eta_w,
        "eta_b": model.eta_b,
        "train_steps": config.train_steps,
        "eval_steps": model.eval_steps,
        "inference_backend": model.inference_backend,
        "inference_method": model.inference_method,
        "state_init": config.state_init,
        "hidden_activation": config.hidden_activation,
        "output_activation": config.output_activation,
        "timing": {
            "train_wall_time_seconds": train_wall_time_seconds,
            "final_evaluation_wall_time_seconds": final_eval_wall_time_seconds,
            "teacher_reference_wall_time_seconds": teacher_reference_wall_time_seconds,
        },
        "teacher_reference": teacher_reference_summary,
    }

    _write_epoch_metrics(run_dir / "epoch_metrics.csv", epoch_rows, num_layers=len(model.layers))
    _write_json(run_dir / "summary.json", summary)

    if config.plot_curves:
        _plot_learning_curves(run_dir, epoch_rows)

    return RealPCRunResult(
        run_dir=run_dir,
        config=config_payload,
        epoch_metrics=epoch_rows,
        summary=summary,
        model=model if return_runtime_objects else None,
        split=split if return_runtime_objects else None,
    )


def run_digits_pc_experiment(
    config: RealPCConfig,
    *,
    return_runtime_objects: bool = False,
) -> RealPCRunResult:
    """Run the canonical deterministic digits predictive-coding baseline experiment."""
    if config.dataset_name != "digits":
        raise ValueError("run_digits_pc_experiment requires config.dataset_name == 'digits'.")
    return run_real_pc_experiment(config, return_runtime_objects=return_runtime_objects)
