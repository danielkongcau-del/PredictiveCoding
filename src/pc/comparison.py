from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .benchmark_specs import ToyBenchmarkSpec, get_benchmark_spec, run_pc_benchmark
from .experiment import ExperimentRunResult, OutputLayout
from .metrics import metric_higher_is_better
from .utils import set_seed

WINNER_TOLERANCE_RTOL = 1.0e-12
WINNER_TOLERANCE_ATOL = 1.0e-12


@dataclass
class ComparisonRunResult:
    """Materialized outputs of one Phase 2 benchmark comparison run."""

    run_dir: Path
    pc_result: ExperimentRunResult
    mlp_run_dir: Path
    mlp_epoch_metrics: list[dict[str, Any]]
    mlp_summary: dict[str, Any]
    comparison_summary: dict[str, Any]


def _resolve_run_id(run_id: str | None) -> str:
    if run_id is not None:
        return run_id
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_comparison_root(
    output_root: str | Path,
    benchmark_name: str,
    run_id: str,
    output_layout: OutputLayout,
) -> Path:
    comparison_name = f"compare_{benchmark_name}"
    if output_layout == "single_dir":
        return Path(output_root) / comparison_name
    if output_layout == "run_id_subdir":
        return Path(output_root) / comparison_name / run_id
    raise ValueError(f"Unsupported output_layout '{output_layout}'.")


def _prepare_run_dir(run_dir: Path) -> Path:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, indent=2)


def _write_mlp_epoch_metrics(
    path: Path,
    rows: list[dict[str, Any]],
    num_layers: int,
    task_name: str,
) -> None:
    fieldnames = ["epoch", "loss"]
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


def select_comparison_winner(
    metric_name: str,
    pc_value: float,
    mlp_value: float,
) -> tuple[str, str]:
    """Return the winner label and a fixed human-readable reason string."""
    if np.isclose(
        pc_value,
        mlp_value,
        rtol=WINNER_TOLERANCE_RTOL,
        atol=WINNER_TOLERANCE_ATOL,
    ):
        return "tie", "tie: primary metric values are equal within tolerance"

    if metric_higher_is_better(metric_name):
        if mlp_value > pc_value:
            return "mlp", "higher_is_better: mlp_primary_metric_value > pc_primary_metric_value"
        return "pc", "higher_is_better: pc_primary_metric_value > mlp_primary_metric_value"

    if mlp_value < pc_value:
        return "mlp", "lower_is_better: mlp_primary_metric_value < pc_primary_metric_value"
    return "pc", "lower_is_better: pc_primary_metric_value < mlp_primary_metric_value"


def _mlp_config_payload(
    spec: ToyBenchmarkSpec,
    run_id: str,
    output_root: str | Path,
    output_layout: OutputLayout,
    split,
) -> dict[str, Any]:
    return {
        "experiment_name": "mlp",
        "run_id": run_id,
        "seed": spec.run_seed,
        "run_seed": spec.run_seed,
        "data_seed": spec.data_seed,
        "model_init_seed": spec.model_init_seed,
        "seeds": {
            "run_seed": spec.run_seed,
            "data_seed": spec.data_seed,
            "model_init_seed": spec.model_init_seed,
        },
        "task": spec.task_config(),
        "data": spec.data_config(split),
        "model": spec.mlp_model_config(),
        "training": {
            "epochs": spec.epochs,
            "eta_w": spec.mlp_training.eta_w,
            "eta_b": spec.mlp_training.eta_b,
            "loss_name": "mse",
            "run_seed": spec.run_seed,
        },
        "logging": {
            "output_root": str(output_root),
            "output_layout": output_layout,
            "plot_energy": False,
            "trace_policy": "not_applicable",
        },
    }


def _run_mlp_experiment(
    spec: ToyBenchmarkSpec,
    *,
    output_root: Path,
    run_id: str,
    output_layout: OutputLayout,
    split,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    mlp_run_dir = output_root / "mlp"
    mlp_run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(spec.run_seed)
    model = spec.build_mlp_model()
    config_payload = _mlp_config_payload(
        spec,
        run_id,
        output_root,
        output_layout,
        split,
    )
    _write_json(mlp_run_dir / "config.json", config_payload)

    x_train = split.x_train
    y_train = split.y_train
    x_val = split.x_val
    y_val = split.y_val
    x_test = split.x_test
    y_test = split.y_test
    train_baseline_metric_value = spec.baseline_metric_fn(y_train)
    val_baseline_metric_value = spec.baseline_metric_fn(y_val)
    test_baseline_metric_value = spec.baseline_metric_fn(y_test)
    epoch_rows: list[dict[str, Any]] = []

    for epoch in range(1, spec.epochs + 1):
        batch_result = model.train_batch(x_train, y_train)
        train_predictions = model.predict(x_train)
        val_predictions = model.predict(x_val)
        train_metric_value = spec.primary_metric_fn(train_predictions, y_train)
        val_metric_value = spec.primary_metric_fn(val_predictions, y_val)
        row: dict[str, Any] = {
            "epoch": epoch,
            "loss": batch_result.loss,
        }
        for layer_index, value in enumerate(batch_result.parameter_norms["weight_norms"], start=1):
            row[f"weight_norm_l{layer_index}"] = value
        for layer_index, value in enumerate(batch_result.parameter_norms["bias_norms"], start=1):
            row[f"bias_norm_l{layer_index}"] = value

        if spec.task_name == "regression":
            row["train_mse"] = train_metric_value
            row["val_mse"] = val_metric_value
            row["train_baseline_mse"] = train_baseline_metric_value
            row["val_baseline_mse"] = val_baseline_metric_value
            row["mse"] = val_metric_value
            row["baseline_mse"] = val_baseline_metric_value
        elif spec.task_name == "classification":
            row["train_accuracy"] = train_metric_value
            row["val_accuracy"] = val_metric_value
            row["train_baseline_accuracy"] = train_baseline_metric_value
            row["val_baseline_accuracy"] = val_baseline_metric_value
            row["accuracy"] = val_metric_value
            row["baseline_accuracy"] = val_baseline_metric_value
        else:
            raise ValueError(f"Unsupported task '{spec.task_name}'.")

        epoch_rows.append(row)

    test_predictions = model.predict(x_test)
    test_metric_value = spec.primary_metric_fn(test_predictions, y_test)
    best_row = (
        max(epoch_rows, key=lambda row: row[spec.primary_metric_name])
        if spec.primary_metric_higher_is_better
        else min(epoch_rows, key=lambda row: row[spec.primary_metric_name])
    )
    final_row = epoch_rows[-1]
    summary = {
        "experiment_name": "mlp",
        "run_id": run_id,
        "phase": "Phase 2",
        "model_family": "mlp",
        "seed": spec.run_seed,
        "run_seed": spec.run_seed,
        "data_seed": spec.data_seed,
        "model_init_seed": spec.model_init_seed,
        "seeds": {
            "run_seed": spec.run_seed,
            "data_seed": spec.data_seed,
            "model_init_seed": spec.model_init_seed,
        },
        "final_epoch": final_row["epoch"],
        "final_loss": final_row["loss"],
        "metric_name": spec.primary_metric_name,
        "metric_higher_is_better": spec.primary_metric_higher_is_better,
        "train_metric": final_row[f"train_{spec.primary_metric_name}"],
        "val_metric": final_row[f"val_{spec.primary_metric_name}"],
        "test_metric": test_metric_value,
        "eval_metric": final_row[f"val_{spec.primary_metric_name}"],
        "primary_metric_name": spec.primary_metric_name,
        "primary_metric_value": test_metric_value,
        "primary_metric_higher_is_better": spec.primary_metric_higher_is_better,
        "selection_metric_source": "val_metric",
        "selection_metric_value": final_row[f"val_{spec.primary_metric_name}"],
        "report_metric_source": "test_metric",
        "report_metric_value": test_metric_value,
        "baseline_metric_name": spec.baseline_metric_name,
        "train_baseline_metric": train_baseline_metric_value,
        "val_baseline_metric": val_baseline_metric_value,
        "test_baseline_metric": test_baseline_metric_value,
        "eval_baseline_metric": val_baseline_metric_value,
        "baseline_metric_value": test_baseline_metric_value,
        "best_epoch": best_row["epoch"],
        "best_val_metric": best_row[f"val_{spec.primary_metric_name}"],
    }

    _write_mlp_epoch_metrics(
        mlp_run_dir / "epoch_metrics.csv",
        epoch_rows,
        num_layers=len(model.layers),
        task_name=spec.task_name,
    )
    _write_json(mlp_run_dir / "summary.json", summary)
    return mlp_run_dir, epoch_rows, summary


def run_benchmark_comparison(
    benchmark_name: str,
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    output_layout: OutputLayout = "single_dir",
    plot_energy: bool = False,
) -> ComparisonRunResult:
    """Run one PC-vs-MLP comparison for an existing toy benchmark."""
    spec = get_benchmark_spec(benchmark_name)
    split = spec.make_dataset_split()
    resolved_run_id = _resolve_run_id(run_id)
    comparison_root = _prepare_run_dir(
        _resolve_comparison_root(
            output_root,
            benchmark_name,
            resolved_run_id,
            output_layout,
        )
    )

    pc_result = run_pc_benchmark(
        spec,
        output_root=comparison_root,
        run_id=resolved_run_id,
        plot_energy=plot_energy,
        output_layout="single_dir",
        experiment_name="pc",
        split=split,
    )
    mlp_run_dir, mlp_epoch_metrics, mlp_summary = _run_mlp_experiment(
        spec,
        output_root=comparison_root,
        run_id=resolved_run_id,
        output_layout=output_layout,
        split=split,
    )

    pc_metric = float(pc_result.summary["test_metric"])
    mlp_metric = float(mlp_summary["test_metric"])
    winner, winner_reason = select_comparison_winner(
        spec.primary_metric_name,
        pc_metric,
        mlp_metric,
    )
    comparison_summary = {
        "experiment_name": f"compare_{benchmark_name}",
        "run_id": resolved_run_id,
        "phase": "Phase 2",
        "benchmark_name": benchmark_name,
        "task_name": spec.task_name,
        "primary_metric_name": spec.primary_metric_name,
        "primary_metric_higher_is_better": spec.primary_metric_higher_is_better,
        "baseline_metric_name": spec.baseline_metric_name,
        "metric_name": spec.primary_metric_name,
        "metric_higher_is_better": spec.primary_metric_higher_is_better,
        "pc_train_metric_value": float(pc_result.summary["train_metric"]),
        "pc_val_metric_value": float(pc_result.summary["val_metric"]),
        "pc_test_metric_value": float(pc_result.summary["test_metric"]),
        "mlp_train_metric_value": float(mlp_summary["train_metric"]),
        "mlp_val_metric_value": float(mlp_summary["val_metric"]),
        "mlp_test_metric_value": float(mlp_summary["test_metric"]),
        "baseline_metric_value": float(pc_result.summary["test_baseline_metric"]),
        "pc_primary_metric_value": pc_metric,
        "mlp_primary_metric_value": mlp_metric,
        "primary_metric_difference_mlp_minus_pc": mlp_metric - pc_metric,
        "winner": winner,
        "metric_winner_reason": winner_reason,
        "winner_tolerance_rtol": WINNER_TOLERANCE_RTOL,
        "winner_tolerance_atol": WINNER_TOLERANCE_ATOL,
        "pc_summary_path": (Path("pc") / "summary.json").as_posix(),
        "mlp_summary_path": (Path("mlp") / "summary.json").as_posix(),
        "pc_math_version": pc_result.summary["math_version"],
    }
    _write_json(comparison_root / "comparison_summary.json", comparison_summary)

    return ComparisonRunResult(
        run_dir=comparison_root,
        pc_result=pc_result,
        mlp_run_dir=mlp_run_dir,
        mlp_epoch_metrics=mlp_epoch_metrics,
        mlp_summary=mlp_summary,
        comparison_summary=comparison_summary,
    )
