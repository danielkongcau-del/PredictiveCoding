from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np

from .models import PCNetwork
from .utils import set_seed

MetricFn = Callable[[np.ndarray, np.ndarray], float]
BaselineMetricFn = Callable[[np.ndarray], float]
OutputLayout = Literal["single_dir", "run_id_subdir"]


@dataclass
class ExperimentConfig:
    """Configuration for a Phase 1 experiment run and its saved artifacts."""

    experiment_name: str
    seed: int
    epochs: int
    data_seed: int | None = None
    model_init_seed: int | None = None
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: OutputLayout = "single_dir"
    plot_energy: bool = False
    trace_policy: str | Sequence[int] = "default"
    task: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    logging: dict[str, Any] = field(default_factory=dict)

    def resolved_run_id(self) -> str:
        """Return the explicit run_id or the default timestamp-based run_id."""
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_{self.seed}"


@dataclass
class ExperimentRunResult:
    """Materialized outputs of a structured experiment run."""

    run_dir: Path
    config: dict[str, Any]
    epoch_metrics: list[dict[str, Any]]
    summary: dict[str, Any]
    trace_manifest: list[dict[str, Any]]
    trace_arrays: dict[str, np.ndarray]


def _resolve_run_dir(
    output_root: str | Path,
    experiment_name: str,
    run_id: str,
    output_layout: OutputLayout,
) -> Path:
    """Resolve the output path for the selected layout without changing semantics elsewhere."""
    if output_layout == "single_dir":
        return Path(output_root) / experiment_name
    if output_layout == "run_id_subdir":
        return Path(output_root) / experiment_name / run_id
    raise ValueError(f"Unsupported output_layout '{output_layout}'.")


def _prepare_run_dir(run_dir: Path) -> Path:
    """Create a fresh run directory at the already-resolved output path."""
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _stringify_trace_policy(trace_policy: str | Sequence[int]) -> str | list[int]:
    if isinstance(trace_policy, str):
        return trace_policy
    return [int(epoch) for epoch in trace_policy]


def _config_dict(config: ExperimentConfig, run_id: str) -> dict[str, Any]:
    return {
        "experiment_name": config.experiment_name,
        "run_id": run_id,
        "seed": config.seed,
        "seeds": {
            "run_seed": config.seed,
            "data_seed": config.data_seed,
            "model_init_seed": config.model_init_seed,
        },
        "task": dict(config.task),
        "data": dict(config.data),
        "model": dict(config.model),
        "training": dict(config.training),
        "logging": {
            "output_root": str(config.output_root),
            "output_layout": config.output_layout,
            "plot_energy": config.plot_energy,
            "trace_policy": _stringify_trace_policy(config.trace_policy),
            **config.logging,
        },
    }


def _should_save_trace(epoch: int, epochs: int, trace_policy: str | Sequence[int]) -> bool:
    if isinstance(trace_policy, str):
        if trace_policy == "default":
            return epoch in {1, epochs}
        if trace_policy == "all":
            return True
        raise ValueError(f"Unsupported trace_policy '{trace_policy}'.")
    explicit_epochs = {int(value) for value in trace_policy}
    return epoch in explicit_epochs


def _trace_key(epoch: int, epochs: int, trace_policy: str | Sequence[int]) -> str:
    if isinstance(trace_policy, str) and trace_policy == "default":
        if epoch == 1:
            return "train_initial_pre_update"
        if epoch == epochs:
            return "train_epoch_final_pre_update"
    return f"train_epoch_{epoch:03d}_pre_update"


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, indent=2)


def _write_epoch_metrics(path: Path, rows: list[dict[str, Any]], num_layers: int, task_name: str) -> None:
    fieldnames = [
        "epoch",
        "train_steps",
        "pre_update_energy",
        "post_update_energy",
        "energy_delta",
        "trace_saved",
    ]
    fieldnames.extend([f"weight_norm_l{layer_index}" for layer_index in range(1, num_layers + 1)])
    fieldnames.extend([f"bias_norm_l{layer_index}" for layer_index in range(1, num_layers + 1)])
    if task_name == "regression":
        fieldnames.extend(
            [
                "train_mse",
                "val_mse",
                "train_baseline_mse",
                "val_baseline_mse",
                "mse",
                "baseline_mse",
            ]
        )
    elif task_name == "classification":
        fieldnames.extend(
            [
                "train_accuracy",
                "val_accuracy",
                "train_baseline_accuracy",
                "val_baseline_accuracy",
                "accuracy",
                "baseline_accuracy",
            ]
        )
    else:
        raise ValueError(f"Unsupported task '{task_name}'.")

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_traces(run_dir: Path, trace_arrays: dict[str, np.ndarray]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "plot_energy=True requires matplotlib to be installed in the current environment."
        ) from exc

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for key, trace in trace_arrays.items():
        figure = plt.figure()
        axis = figure.add_subplot(1, 1, 1)
        axis.plot(np.arange(trace.shape[0]), trace)
        axis.set_xlabel("Inference step")
        axis.set_ylabel("Energy")
        axis.set_title(key)
        figure.tight_layout()
        figure.savefig(plots_dir / f"{key}.png")
        plt.close(figure)


def run_supervised_experiment(
    *,
    config: ExperimentConfig,
    model: PCNetwork,
    x: np.ndarray,
    y: np.ndarray,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    x_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    x_eval: np.ndarray | None = None,
    y_eval: np.ndarray | None = None,
    task_name: str,
    primary_metric_name: str,
    primary_metric_higher_is_better: bool,
    primary_metric_fn: MetricFn,
    baseline_metric_name: str,
    baseline_metric_fn: BaselineMetricFn,
) -> ExperimentRunResult:
    """Run a supervised experiment with train/val/test metrics.

    Shapes:
    - `x`, `y`: training arrays shaped `(batch_train, features)` / `(batch_train, targets)`
    - `x_val`, `y_val`: validation arrays shaped `(batch_val, features)` / `(batch_val, targets)`
    - `x_test`, `y_test`: test arrays shaped `(batch_test, features)` / `(batch_test, targets)`
    """
    if config.epochs <= 0:
        raise ValueError("config.epochs must be positive.")
    if x_val is None or y_val is None:
        x_val = x_eval
        y_val = y_eval
    if x_val is None or y_val is None:
        x_val = x
        y_val = y
    if x_test is None or y_test is None:
        x_test = x_val
        y_test = y_val

    set_seed(config.seed)
    run_id = config.resolved_run_id()
    run_dir = _prepare_run_dir(
        _resolve_run_dir(
            config.output_root,
            config.experiment_name,
            run_id,
            config.output_layout,
        )
    )

    config_payload = _config_dict(config, run_id)
    _write_json(run_dir / "config.json", config_payload)

    train_baseline_metric_value = baseline_metric_fn(y)
    val_baseline_metric_value = baseline_metric_fn(y_val)
    test_baseline_metric_value = baseline_metric_fn(y_test)
    epoch_rows: list[dict[str, Any]] = []
    trace_arrays: dict[str, np.ndarray] = {}
    trace_manifest: list[dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        batch_result = model.train_batch(x, y, compute_post_update_energy=True)
        train_predictions = model.predict(x)
        val_predictions = model.predict(x_val)
        train_metric_value = primary_metric_fn(train_predictions, y)
        val_metric_value = primary_metric_fn(val_predictions, y_val)
        trace_saved = _should_save_trace(epoch, config.epochs, config.trace_policy)

        if trace_saved:
            key = _trace_key(epoch, config.epochs, config.trace_policy)
            trace_arrays[key] = np.asarray(batch_result.energy_trace, dtype=np.float64)
            trace_manifest.append(
                {
                    "key": key,
                    "epoch": 0 if key == "train_initial_pre_update" else epoch,
                    "stage": "pre_update_inference",
                    "mode": "train",
                    "steps": model.train_steps,
                    "eta_x": model.eta_x,
                    "state_init": model.state_init,
                    "trace_length": len(batch_result.energy_trace),
                }
            )

        row: dict[str, Any] = {
            "epoch": epoch,
            "train_steps": batch_result.train_steps,
            "pre_update_energy": batch_result.pre_update_energy,
            "post_update_energy": batch_result.post_update_energy,
            "energy_delta": batch_result.post_update_energy - batch_result.pre_update_energy,
            "trace_saved": trace_saved,
        }
        for layer_index, value in enumerate(batch_result.parameter_norms["weight_norms"], start=1):
            row[f"weight_norm_l{layer_index}"] = value
        for layer_index, value in enumerate(batch_result.parameter_norms["bias_norms"], start=1):
            row[f"bias_norm_l{layer_index}"] = value

        if task_name == "regression":
            row["train_mse"] = train_metric_value
            row["val_mse"] = val_metric_value
            row["train_baseline_mse"] = train_baseline_metric_value
            row["val_baseline_mse"] = val_baseline_metric_value
            row["mse"] = val_metric_value
            row["baseline_mse"] = val_baseline_metric_value
        elif task_name == "classification":
            row["train_accuracy"] = train_metric_value
            row["val_accuracy"] = val_metric_value
            row["train_baseline_accuracy"] = train_baseline_metric_value
            row["val_baseline_accuracy"] = val_baseline_metric_value
            row["accuracy"] = val_metric_value
            row["baseline_accuracy"] = val_baseline_metric_value
        else:
            raise ValueError(f"Unsupported task '{task_name}'.")

        epoch_rows.append(row)

    best_row = (
        max(epoch_rows, key=lambda row: row[f"val_{primary_metric_name}"])
        if primary_metric_higher_is_better
        else min(epoch_rows, key=lambda row: row[f"val_{primary_metric_name}"])
    )
    final_row = epoch_rows[-1]
    test_predictions = model.predict(x_test)
    test_metric_value = primary_metric_fn(test_predictions, y_test)
    summary = {
        "experiment_name": config.experiment_name,
        "run_id": run_id,
        "phase": "Phase 1",
        "math_version": "phase0-baseline",
        "seed": config.seed,
        "final_epoch": final_row["epoch"],
        "final_pre_update_energy": final_row["pre_update_energy"],
        "final_post_update_energy": final_row["post_update_energy"],
        "metric_name": primary_metric_name,
        "metric_higher_is_better": primary_metric_higher_is_better,
        "train_metric": final_row[f"train_{primary_metric_name}"],
        "val_metric": final_row[f"val_{primary_metric_name}"],
        "test_metric": test_metric_value,
        "eval_metric": final_row[f"val_{primary_metric_name}"],
        "primary_metric_name": primary_metric_name,
        "primary_metric_value": test_metric_value,
        "primary_metric_higher_is_better": primary_metric_higher_is_better,
        "selection_metric_source": "val_metric",
        "selection_metric_value": final_row[f"val_{primary_metric_name}"],
        "report_metric_source": "test_metric",
        "report_metric_value": test_metric_value,
        "baseline_metric_name": baseline_metric_name,
        "train_baseline_metric": train_baseline_metric_value,
        "val_baseline_metric": val_baseline_metric_value,
        "test_baseline_metric": test_baseline_metric_value,
        "eval_baseline_metric": val_baseline_metric_value,
        "baseline_metric_value": test_baseline_metric_value,
        "best_epoch": best_row["epoch"],
        "best_val_metric": best_row[f"val_{primary_metric_name}"],
        "saved_trace_keys": sorted(trace_arrays.keys()),
    }

    _write_epoch_metrics(
        run_dir / "epoch_metrics.csv",
        epoch_rows,
        num_layers=len(model.layers),
        task_name=task_name,
    )
    np.savez(run_dir / "energy_traces.npz", **trace_arrays)
    _write_json(run_dir / "energy_traces_manifest.json", trace_manifest)
    _write_json(run_dir / "summary.json", summary)

    if config.plot_energy:
        _plot_traces(run_dir, trace_arrays)

    return ExperimentRunResult(
        run_dir=run_dir,
        config=config_payload,
        epoch_metrics=epoch_rows,
        summary=summary,
        trace_manifest=trace_manifest,
        trace_arrays=trace_arrays,
    )
