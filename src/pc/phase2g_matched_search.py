from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .benchmark_specs import (
    MLPTrainingSpec,
    PCTrainingSpec,
    ToyBenchmarkSpec,
    get_benchmark_spec,
    run_pc_benchmark,
)
from .comparison import (
    WINNER_TOLERANCE_ATOL,
    WINNER_TOLERANCE_RTOL,
    select_comparison_winner,
)
from .experiment import OutputLayout
from .metrics import metric_higher_is_better
from .utils import set_seed

PHASE2G_BENCHMARK_NAMES = (
    "toy_regression",
    "toy_sine_regression",
)


@dataclass(frozen=True)
class PCMatchedSearchTrial:
    """One deterministic Phase 2g predictive-coding configuration."""

    config_id: str
    eta_x: float
    eta_w: float
    train_steps: int
    epochs: int
    state_init: str

    @property
    def eta_b(self) -> float:
        return self.eta_w

    @property
    def eval_steps(self) -> int:
        return self.train_steps

    def to_pc_training_spec(self) -> PCTrainingSpec:
        return PCTrainingSpec(
            eta_x=self.eta_x,
            eta_w=self.eta_w,
            eta_b=self.eta_b,
            train_steps=self.train_steps,
            eval_steps=self.eval_steps,
            state_init=self.state_init,
        )


@dataclass(frozen=True)
class MLPMatchedSearchTrial:
    """One deterministic Phase 2g standard-MLP configuration."""

    config_id: str
    eta_w: float
    epochs: int

    @property
    def eta_b(self) -> float:
        return self.eta_w

    def to_mlp_training_spec(self) -> MLPTrainingSpec:
        return MLPTrainingSpec(
            eta_w=self.eta_w,
            eta_b=self.eta_b,
        )


@dataclass
class Phase2GMatchedSearchRunResult:
    """Materialized outputs of one Phase 2g matched search run."""

    run_dir: Path
    study_config: dict[str, Any]
    pc_search_rows: list[dict[str, Any]]
    mlp_search_rows: list[dict[str, Any]]
    aggregate_summary: dict[str, Any]
    best_pc_config_summary: dict[str, Any]
    best_mlp_config_summary: dict[str, Any]


def _resolve_run_id(run_id: str | None) -> str:
    if run_id is not None:
        return run_id
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_search_root(
    output_root: str | Path,
    benchmark_name: str,
    run_id: str,
    output_layout: OutputLayout,
) -> Path:
    base_dir = Path(output_root) / "phase2g_matched_search" / benchmark_name
    if output_layout == "single_dir":
        return base_dir
    if output_layout == "run_id_subdir":
        return base_dir / run_id
    raise ValueError(f"Unsupported output_layout '{output_layout}'.")


def _prepare_run_dir(run_dir: Path) -> Path:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, indent=2)


def _write_search_results(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "config_id",
        "model_family",
        "eta_x",
        "eta_w",
        "eta_b",
        "train_steps",
        "eval_steps",
        "epochs",
        "state_init",
        "status",
        "failure_reason",
        "metric_name",
        "metric_higher_is_better",
        "train_metric",
        "val_metric",
        "test_metric",
        "selection_metric_source",
        "selection_metric_value",
        "report_metric_source",
        "report_metric_value",
        "selection_metric_rank",
        "selection_metric_delta_vs_best",
        "train_baseline_metric",
        "val_baseline_metric",
        "test_baseline_metric",
        "beats_val_baseline",
        "best_epoch",
        "final_pre_update_energy",
        "final_post_update_energy",
        "final_loss",
        "summary_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def _clean_failed_trial_dir(trial_dir: Path) -> None:
    if trial_dir.exists():
        shutil.rmtree(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)


def _trial_failure_reason(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _default_pc_search_space(spec: ToyBenchmarkSpec) -> dict[str, list[float] | list[int]]:
    if spec.benchmark_name == "toy_regression":
        return {
            "eta_x": [0.05, 0.1, 0.2],
            "eta_w": [0.1, 0.2, 0.4],
            "train_steps": [25, 50, 100],
            "epochs": [60, 120, 240],
        }
    if spec.benchmark_name == "toy_sine_regression":
        return {
            "eta_x": [0.05, 0.075, 0.1, 0.15],
            "eta_w": [0.06, 0.1, 0.2],
            "train_steps": [30, 60, 120],
            "epochs": [80, 160, 320],
        }
    raise ValueError(f"Unsupported Phase 2g benchmark '{spec.benchmark_name}'.")


def _default_mlp_search_space(spec: ToyBenchmarkSpec) -> dict[str, list[float] | list[int]]:
    if spec.benchmark_name not in PHASE2G_BENCHMARK_NAMES:
        raise ValueError(f"Unsupported Phase 2g benchmark '{spec.benchmark_name}'.")
    return {
        "eta_w": [0.03, 0.05, 0.1, 0.2],
        "epochs": [80, 160, 320],
    }


def _resolve_search_space(
    default_space: dict[str, list[float] | list[int]],
    override: dict[str, list[float] | list[int]] | None,
) -> dict[str, list[float] | list[int]]:
    resolved = {key: list(values) for key, values in default_space.items()}
    if override is None:
        return resolved
    for key, values in override.items():
        if key not in resolved:
            raise ValueError(f"Unsupported search-space key '{key}'.")
        if len(values) == 0:
            raise ValueError(f"Search-space key '{key}' must contain at least one value.")
        resolved[key] = list(values)
    return resolved


def resolve_pc_search_space(
    spec: ToyBenchmarkSpec,
    override: dict[str, list[float] | list[int]] | None = None,
) -> dict[str, list[float] | list[int]]:
    """Return the deterministic Phase 2g PC search space for one benchmark."""
    return _resolve_search_space(_default_pc_search_space(spec), override)


def resolve_mlp_search_space(
    spec: ToyBenchmarkSpec,
    override: dict[str, list[float] | list[int]] | None = None,
) -> dict[str, list[float] | list[int]]:
    """Return the deterministic Phase 2g MLP search space for one benchmark."""
    return _resolve_search_space(_default_mlp_search_space(spec), override)


def build_pc_search_trials(
    spec: ToyBenchmarkSpec,
    override: dict[str, list[float] | list[int]] | None = None,
) -> tuple[list[PCMatchedSearchTrial], dict[str, list[float] | list[int]]]:
    """Return the deterministic Cartesian product for Phase 2g PC search."""
    search_space = resolve_pc_search_space(spec, override)
    trials: list[PCMatchedSearchTrial] = []
    config_index = 1
    for eta_x in search_space["eta_x"]:
        for eta_w in search_space["eta_w"]:
            for train_steps in search_space["train_steps"]:
                for epochs in search_space["epochs"]:
                    trials.append(
                        PCMatchedSearchTrial(
                            config_id=f"cfg_{config_index:03d}",
                            eta_x=float(eta_x),
                            eta_w=float(eta_w),
                            train_steps=int(train_steps),
                            epochs=int(epochs),
                            state_init=spec.pc_training.state_init,
                        )
                    )
                    config_index += 1
    return trials, search_space


def build_mlp_search_trials(
    spec: ToyBenchmarkSpec,
    override: dict[str, list[float] | list[int]] | None = None,
) -> tuple[list[MLPMatchedSearchTrial], dict[str, list[float] | list[int]]]:
    """Return the deterministic Cartesian product for Phase 2g MLP search."""
    search_space = resolve_mlp_search_space(spec, override)
    trials: list[MLPMatchedSearchTrial] = []
    config_index = 1
    for eta_w in search_space["eta_w"]:
        for epochs in search_space["epochs"]:
            trials.append(
                MLPMatchedSearchTrial(
                    config_id=f"cfg_{config_index:03d}",
                    eta_w=float(eta_w),
                    epochs=int(epochs),
                )
            )
            config_index += 1
    return trials, search_space


def metric_value_is_better(metric_name: str, candidate_value: float, reference_value: float) -> bool:
    """Return whether a candidate metric strictly beats a reference metric."""
    if np.isclose(
        candidate_value,
        reference_value,
        rtol=WINNER_TOLERANCE_RTOL,
        atol=WINNER_TOLERANCE_ATOL,
    ):
        return False
    if metric_higher_is_better(metric_name):
        return candidate_value > reference_value
    return candidate_value < reference_value


def rank_search_rows(
    rows: list[dict[str, Any]],
    metric_name: str,
) -> list[dict[str, Any]]:
    """Return rows ranked strictly by selection_metric_value, which is validation-only."""
    successful_rows = [row for row in rows if row["status"] == "ok"]
    if not successful_rows:
        return [dict(row, selection_metric_rank=None, selection_metric_delta_vs_best=None) for row in rows]

    if metric_higher_is_better(metric_name):
        ranked_successful = sorted(
            successful_rows,
            key=lambda row: (-float(row["selection_metric_value"]), str(row["config_id"])),
        )
    else:
        ranked_successful = sorted(
            successful_rows,
            key=lambda row: (float(row["selection_metric_value"]), str(row["config_id"])),
        )

    rank_mapping = {row["config_id"]: rank for rank, row in enumerate(ranked_successful, start=1)}
    best_selection_metric = float(ranked_successful[0]["selection_metric_value"])

    ranked_rows: list[dict[str, Any]] = []
    for row in rows:
        ranked_row = dict(row)
        if row["status"] != "ok":
            ranked_row["selection_metric_rank"] = None
            ranked_row["selection_metric_delta_vs_best"] = None
        else:
            ranked_row["selection_metric_rank"] = rank_mapping[row["config_id"]]
            ranked_row["selection_metric_delta_vs_best"] = (
                float(row["selection_metric_value"]) - best_selection_metric
            )
        ranked_rows.append(ranked_row)
    return ranked_rows


def _select_best_row(rows: list[dict[str, Any]], metric_name: str) -> dict[str, Any]:
    successful_rows = [row for row in rows if row["status"] == "ok"]
    if not successful_rows:
        raise ValueError("At least one successful configuration is required.")
    ranked_rows = rank_search_rows(successful_rows, metric_name)
    best_ranked_row = min(ranked_rows, key=lambda row: int(row["selection_metric_rank"]))
    for row in successful_rows:
        if row["config_id"] == best_ranked_row["config_id"]:
            return row
    raise RuntimeError("Best row lookup failed after ranking.")


def _selection_reason(metric_name: str, successful_count: int) -> str:
    if metric_higher_is_better(metric_name):
        return f"selected highest val_{metric_name} across {successful_count} successful configurations"
    return f"selected lowest val_{metric_name} across {successful_count} successful configurations"


def _top_ranked_configs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "config_id": row["config_id"],
            "val_metric": row["val_metric"],
            "test_metric": row["test_metric"],
            "train_metric": row["train_metric"],
            "selection_metric_rank": row["selection_metric_rank"],
        }
        for row in sorted(
            [row for row in rows if row["status"] == "ok"],
            key=lambda item: int(item["selection_metric_rank"]),
        )[:5]
    ]


def _pc_training_dict(training: PCTrainingSpec, *, epochs: int) -> dict[str, Any]:
    return {
        "eta_x": training.eta_x,
        "eta_w": training.eta_w,
        "eta_b": training.eta_b,
        "train_steps": training.train_steps,
        "eval_steps": training.eval_steps,
        "epochs": epochs,
        "state_init": training.state_init,
    }


def _mlp_training_dict(training: MLPTrainingSpec, *, epochs: int) -> dict[str, Any]:
    return {
        "eta_w": training.eta_w,
        "eta_b": training.eta_b,
        "epochs": epochs,
    }


def _study_config_payload(
    spec: ToyBenchmarkSpec,
    *,
    run_id: str,
    output_layout: OutputLayout,
    split,
    pc_search_space: dict[str, list[float] | list[int]],
    mlp_search_space: dict[str, list[float] | list[int]],
    pc_trial_count: int,
    mlp_trial_count: int,
) -> dict[str, Any]:
    return {
        "experiment_name": f"phase2g_matched_search_{spec.benchmark_name}",
        "run_id": run_id,
        "phase": "Phase 2g",
        "benchmark_name": spec.benchmark_name,
        "task_name": spec.task_name,
        "search_target": "matched_pc_and_mlp",
        "metric_name": spec.primary_metric_name,
        "metric_higher_is_better": spec.primary_metric_higher_is_better,
        "selected_by_metric_source": "val_metric",
        "final_report_metric_source": "test_metric",
        "test_metric_used_for_selection": False,
        "seeds": {
            "run_seed": spec.run_seed,
            "data_seed": spec.data_seed,
            "model_init_seed": spec.model_init_seed,
        },
        "data": spec.data_config(split),
        "pc_base_training": _pc_training_dict(spec.pc_training, epochs=spec.epochs),
        "pc_search_space": pc_search_space,
        "pc_search_space_size": pc_trial_count,
        "pc_fixed_fields": {
            "state_init": spec.pc_training.state_init,
            "eta_b_equals_eta_w": True,
            "eval_steps_equals_train_steps": True,
        },
        "mlp_base_training": _mlp_training_dict(spec.mlp_training, epochs=spec.epochs),
        "mlp_search_space": mlp_search_space,
        "mlp_search_space_size": mlp_trial_count,
        "mlp_fixed_fields": {
            "architecture_matches_benchmark_spec": True,
            "eta_b_equals_eta_w": True,
        },
        "output_layout": output_layout,
        "notes": {
            "runtime_budget": "Phase 2g uses the full small Cartesian products documented in the study config; no silent pruning is applied.",
            "selection_rule": "Configuration ranking is always based on validation metric only.",
            "report_rule": "Final method comparison is always based on held-out test metric.",
        },
    }


def _mlp_config_payload(
    spec: ToyBenchmarkSpec,
    trial: MLPMatchedSearchTrial,
    *,
    run_id: str,
    search_root: Path,
    output_layout: OutputLayout,
    split,
) -> dict[str, Any]:
    return {
        "experiment_name": trial.config_id,
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
            "epochs": trial.epochs,
            "eta_w": trial.eta_w,
            "eta_b": trial.eta_b,
            "loss_name": "mse",
            "run_seed": spec.run_seed,
        },
        "logging": {
            "output_root": str(search_root),
            "output_layout": output_layout,
            "plot_energy": False,
            "trace_policy": "not_applicable",
        },
    }


def _run_mlp_trial(
    spec: ToyBenchmarkSpec,
    trial: MLPMatchedSearchTrial,
    *,
    trial_root: Path,
    run_id: str,
    output_layout: OutputLayout,
    split,
) -> dict[str, Any]:
    trial_root.mkdir(parents=True, exist_ok=True)
    set_seed(spec.run_seed)
    model = spec.build_mlp_model()
    _write_json(
        trial_root / "config.json",
        _mlp_config_payload(
            spec,
            trial,
            run_id=run_id,
            search_root=trial_root.parent,
            output_layout=output_layout,
            split=split,
        ),
    )

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
        "experiment_name": trial.config_id,
        "run_id": run_id,
        "phase": "Phase 2g",
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
        trial_root / "epoch_metrics.csv",
        epoch_rows,
        num_layers=len(model.layers),
        task_name=spec.task_name,
    )
    _write_json(trial_root / "summary.json", summary)
    return summary


def _pc_row_from_summary(
    trial: PCMatchedSearchTrial,
    summary: dict[str, Any],
    *,
    summary_path: str,
) -> dict[str, Any]:
    return {
        "config_id": trial.config_id,
        "model_family": "predictive_coding",
        "eta_x": trial.eta_x,
        "eta_w": trial.eta_w,
        "eta_b": trial.eta_b,
        "train_steps": trial.train_steps,
        "eval_steps": trial.eval_steps,
        "epochs": trial.epochs,
        "state_init": trial.state_init,
        "status": "ok",
        "failure_reason": "",
        "metric_name": summary["metric_name"],
        "metric_higher_is_better": summary["metric_higher_is_better"],
        "train_metric": float(summary["train_metric"]),
        "val_metric": float(summary["val_metric"]),
        "test_metric": float(summary["test_metric"]),
        "selection_metric_source": "val_metric",
        "selection_metric_value": float(summary["val_metric"]),
        "report_metric_source": "test_metric",
        "report_metric_value": float(summary["test_metric"]),
        "selection_metric_rank": None,
        "selection_metric_delta_vs_best": None,
        "train_baseline_metric": float(summary["train_baseline_metric"]),
        "val_baseline_metric": float(summary["val_baseline_metric"]),
        "test_baseline_metric": float(summary["test_baseline_metric"]),
        "beats_val_baseline": metric_value_is_better(
            str(summary["metric_name"]),
            float(summary["val_metric"]),
            float(summary["val_baseline_metric"]),
        ),
        "best_epoch": int(summary["best_epoch"]),
        "final_pre_update_energy": float(summary["final_pre_update_energy"]),
        "final_post_update_energy": float(summary["final_post_update_energy"]),
        "final_loss": None,
        "summary_path": summary_path,
    }


def _mlp_row_from_summary(
    trial: MLPMatchedSearchTrial,
    summary: dict[str, Any],
    *,
    summary_path: str,
) -> dict[str, Any]:
    return {
        "config_id": trial.config_id,
        "model_family": "mlp",
        "eta_x": None,
        "eta_w": trial.eta_w,
        "eta_b": trial.eta_b,
        "train_steps": None,
        "eval_steps": None,
        "epochs": trial.epochs,
        "state_init": None,
        "status": "ok",
        "failure_reason": "",
        "metric_name": summary["metric_name"],
        "metric_higher_is_better": summary["metric_higher_is_better"],
        "train_metric": float(summary["train_metric"]),
        "val_metric": float(summary["val_metric"]),
        "test_metric": float(summary["test_metric"]),
        "selection_metric_source": "val_metric",
        "selection_metric_value": float(summary["val_metric"]),
        "report_metric_source": "test_metric",
        "report_metric_value": float(summary["test_metric"]),
        "selection_metric_rank": None,
        "selection_metric_delta_vs_best": None,
        "train_baseline_metric": float(summary["train_baseline_metric"]),
        "val_baseline_metric": float(summary["val_baseline_metric"]),
        "test_baseline_metric": float(summary["test_baseline_metric"]),
        "beats_val_baseline": metric_value_is_better(
            str(summary["metric_name"]),
            float(summary["val_metric"]),
            float(summary["val_baseline_metric"]),
        ),
        "best_epoch": int(summary["best_epoch"]),
        "final_pre_update_energy": None,
        "final_post_update_energy": None,
        "final_loss": float(summary["final_loss"]),
        "summary_path": summary_path,
    }


def _failed_pc_row(
    trial: PCMatchedSearchTrial,
    spec: ToyBenchmarkSpec,
    split,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "config_id": trial.config_id,
        "model_family": "predictive_coding",
        "eta_x": trial.eta_x,
        "eta_w": trial.eta_w,
        "eta_b": trial.eta_b,
        "train_steps": trial.train_steps,
        "eval_steps": trial.eval_steps,
        "epochs": trial.epochs,
        "state_init": trial.state_init,
        "status": "failed",
        "failure_reason": _trial_failure_reason(exc),
        "metric_name": spec.primary_metric_name,
        "metric_higher_is_better": spec.primary_metric_higher_is_better,
        "train_metric": None,
        "val_metric": None,
        "test_metric": None,
        "selection_metric_source": "val_metric",
        "selection_metric_value": None,
        "report_metric_source": "test_metric",
        "report_metric_value": None,
        "selection_metric_rank": None,
        "selection_metric_delta_vs_best": None,
        "train_baseline_metric": spec.baseline_metric_fn(split.y_train),
        "val_baseline_metric": spec.baseline_metric_fn(split.y_val),
        "test_baseline_metric": spec.baseline_metric_fn(split.y_test),
        "beats_val_baseline": None,
        "best_epoch": None,
        "final_pre_update_energy": None,
        "final_post_update_energy": None,
        "final_loss": None,
        "summary_path": "",
    }


def _failed_mlp_row(
    trial: MLPMatchedSearchTrial,
    spec: ToyBenchmarkSpec,
    split,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "config_id": trial.config_id,
        "model_family": "mlp",
        "eta_x": None,
        "eta_w": trial.eta_w,
        "eta_b": trial.eta_b,
        "train_steps": None,
        "eval_steps": None,
        "epochs": trial.epochs,
        "state_init": None,
        "status": "failed",
        "failure_reason": _trial_failure_reason(exc),
        "metric_name": spec.primary_metric_name,
        "metric_higher_is_better": spec.primary_metric_higher_is_better,
        "train_metric": None,
        "val_metric": None,
        "test_metric": None,
        "selection_metric_source": "val_metric",
        "selection_metric_value": None,
        "report_metric_source": "test_metric",
        "report_metric_value": None,
        "selection_metric_rank": None,
        "selection_metric_delta_vs_best": None,
        "train_baseline_metric": spec.baseline_metric_fn(split.y_train),
        "val_baseline_metric": spec.baseline_metric_fn(split.y_val),
        "test_baseline_metric": spec.baseline_metric_fn(split.y_test),
        "beats_val_baseline": None,
        "best_epoch": None,
        "final_pre_update_energy": None,
        "final_post_update_energy": None,
        "final_loss": None,
        "summary_path": "",
    }


def _family_best_summary(
    *,
    benchmark_name: str,
    run_id: str,
    task_name: str,
    metric_name: str,
    model_family: str,
    best_row: dict[str, Any],
    search_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    successful_count = len([row for row in search_rows if row["status"] == "ok"])
    best_rank = next(
        int(row["selection_metric_rank"])
        for row in search_rows
        if row["config_id"] == best_row["config_id"]
    )
    if model_family == "predictive_coding":
        config = {
            "eta_x": best_row["eta_x"],
            "eta_w": best_row["eta_w"],
            "eta_b": best_row["eta_b"],
            "train_steps": best_row["train_steps"],
            "eval_steps": best_row["eval_steps"],
            "epochs": best_row["epochs"],
            "state_init": best_row["state_init"],
        }
    else:
        config = {
            "eta_w": best_row["eta_w"],
            "eta_b": best_row["eta_b"],
            "epochs": best_row["epochs"],
        }
    return {
        "experiment_name": f"phase2g_matched_search_{benchmark_name}",
        "run_id": run_id,
        "phase": "Phase 2g",
        "benchmark_name": benchmark_name,
        "task_name": task_name,
        "model_family": model_family,
        "metric_name": metric_name,
        "metric_higher_is_better": metric_higher_is_better(metric_name),
        "selected_by_metric_source": "val_metric",
        "final_report_metric_source": "test_metric",
        "best_config_id": best_row["config_id"],
        "best_config_rank": best_rank,
        "best_config": config,
        "train_metric": best_row["train_metric"],
        "val_metric": best_row["val_metric"],
        "test_metric": best_row["test_metric"],
        "selection_metric_value": best_row["selection_metric_value"],
        "report_metric_value": best_row["report_metric_value"],
        "train_baseline_metric": best_row["train_baseline_metric"],
        "val_baseline_metric": best_row["val_baseline_metric"],
        "test_baseline_metric": best_row["test_baseline_metric"],
        "best_epoch": best_row["best_epoch"],
        "final_pre_update_energy": best_row["final_pre_update_energy"],
        "final_post_update_energy": best_row["final_post_update_energy"],
        "final_loss": best_row["final_loss"],
        "summary_path": best_row["summary_path"],
        "selection_reason": _selection_reason(metric_name, successful_count),
    }


def _build_aggregate_summary(
    *,
    benchmark_name: str,
    run_id: str,
    task_name: str,
    metric_name: str,
    pc_search_rows: list[dict[str, Any]],
    mlp_search_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    best_pc_row = _select_best_row(pc_search_rows, metric_name)
    best_mlp_row = _select_best_row(mlp_search_rows, metric_name)
    pc_test_metric = float(best_pc_row["test_metric"])
    mlp_test_metric = float(best_mlp_row["test_metric"])
    winner, winner_reason = select_comparison_winner(metric_name, pc_test_metric, mlp_test_metric)

    return {
        "experiment_name": f"phase2g_matched_search_{benchmark_name}",
        "run_id": run_id,
        "phase": "Phase 2g",
        "benchmark_name": benchmark_name,
        "task_name": task_name,
        "search_target": "matched_pc_and_mlp",
        "metric_name": metric_name,
        "metric_higher_is_better": metric_higher_is_better(metric_name),
        "selected_by_metric_source": "val_metric",
        "final_report_metric_source": "test_metric",
        "test_metric_used_for_selection": False,
        "pc_trial_count": len(pc_search_rows),
        "pc_successful_trial_count": len([row for row in pc_search_rows if row["status"] == "ok"]),
        "pc_failed_trial_count": len([row for row in pc_search_rows if row["status"] == "failed"]),
        "pc_failed_config_ids": [row["config_id"] for row in pc_search_rows if row["status"] == "failed"],
        "mlp_trial_count": len(mlp_search_rows),
        "mlp_successful_trial_count": len([row for row in mlp_search_rows if row["status"] == "ok"]),
        "mlp_failed_trial_count": len([row for row in mlp_search_rows if row["status"] == "failed"]),
        "mlp_failed_config_ids": [row["config_id"] for row in mlp_search_rows if row["status"] == "failed"],
        "best_pc_config_id": best_pc_row["config_id"],
        "best_pc_config": {
            "eta_x": best_pc_row["eta_x"],
            "eta_w": best_pc_row["eta_w"],
            "eta_b": best_pc_row["eta_b"],
            "train_steps": best_pc_row["train_steps"],
            "eval_steps": best_pc_row["eval_steps"],
            "epochs": best_pc_row["epochs"],
            "state_init": best_pc_row["state_init"],
        },
        "best_pc_train_metric": best_pc_row["train_metric"],
        "best_pc_val_metric": best_pc_row["val_metric"],
        "best_pc_test_metric": best_pc_row["test_metric"],
        "best_pc_summary_path": best_pc_row["summary_path"],
        "best_mlp_config_id": best_mlp_row["config_id"],
        "best_mlp_config": {
            "eta_w": best_mlp_row["eta_w"],
            "eta_b": best_mlp_row["eta_b"],
            "epochs": best_mlp_row["epochs"],
        },
        "best_mlp_train_metric": best_mlp_row["train_metric"],
        "best_mlp_val_metric": best_mlp_row["val_metric"],
        "best_mlp_test_metric": best_mlp_row["test_metric"],
        "best_mlp_summary_path": best_mlp_row["summary_path"],
        "test_winner": winner,
        "test_winner_reason": winner_reason,
        "pc_beats_mlp_on_test": winner == "pc",
        "test_metric_difference_mlp_minus_pc": mlp_test_metric - pc_test_metric,
        "winner_tolerance_rtol": WINNER_TOLERANCE_RTOL,
        "winner_tolerance_atol": WINNER_TOLERANCE_ATOL,
        "pc_selection_reason": _selection_reason(
            metric_name,
            len([row for row in pc_search_rows if row["status"] == "ok"]),
        ),
        "mlp_selection_reason": _selection_reason(
            metric_name,
            len([row for row in mlp_search_rows if row["status"] == "ok"]),
        ),
        "top_ranked_pc_configs": _top_ranked_configs(pc_search_rows),
        "top_ranked_mlp_configs": _top_ranked_configs(mlp_search_rows),
    }


def run_phase2g_matched_search(
    benchmark_name: str,
    *,
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    output_layout: OutputLayout = "single_dir",
    plot_energy: bool = False,
    pc_search_space_override: dict[str, list[float] | list[int]] | None = None,
    mlp_search_space_override: dict[str, list[float] | list[int]] | None = None,
) -> Phase2GMatchedSearchRunResult:
    """Run the deterministic Phase 2g matched PC+MLP search ranked by validation metric."""
    if benchmark_name not in PHASE2G_BENCHMARK_NAMES:
        raise ValueError(
            f"Phase 2g currently supports only {PHASE2G_BENCHMARK_NAMES}, got '{benchmark_name}'."
        )

    spec = get_benchmark_spec(benchmark_name)
    split = spec.make_dataset_split()
    resolved_run_id = _resolve_run_id(run_id)
    pc_trials, pc_search_space = build_pc_search_trials(spec, pc_search_space_override)
    mlp_trials, mlp_search_space = build_mlp_search_trials(spec, mlp_search_space_override)
    run_dir = _prepare_run_dir(
        _resolve_search_root(output_root, benchmark_name, resolved_run_id, output_layout)
    )

    study_config = _study_config_payload(
        spec,
        run_id=resolved_run_id,
        output_layout=output_layout,
        split=split,
        pc_search_space=pc_search_space,
        mlp_search_space=mlp_search_space,
        pc_trial_count=len(pc_trials),
        mlp_trial_count=len(mlp_trials),
    )
    _write_json(run_dir / "study_config.json", study_config)

    pc_search_rows: list[dict[str, Any]] = []
    for trial in pc_trials:
        trial_spec = replace(
            spec,
            epochs=trial.epochs,
            pc_training=trial.to_pc_training_spec(),
        )
        trial_dir = run_dir / "pc_trials" / trial.config_id
        try:
            result = run_pc_benchmark(
                trial_spec,
                output_root=run_dir / "pc_trials",
                run_id=resolved_run_id,
                plot_energy=plot_energy,
                output_layout="single_dir",
                experiment_name=trial.config_id,
                split=split,
            )
            pc_search_rows.append(
                _pc_row_from_summary(
                    trial,
                    result.summary,
                    summary_path=(Path("pc_trials") / trial.config_id / "summary.json").as_posix(),
                )
            )
        except Exception as exc:
            _clean_failed_trial_dir(trial_dir)
            pc_search_rows.append(_failed_pc_row(trial, spec, split, exc))

    mlp_search_rows: list[dict[str, Any]] = []
    for trial in mlp_trials:
        trial_spec = replace(
            spec,
            epochs=trial.epochs,
            mlp_training=trial.to_mlp_training_spec(),
        )
        trial_dir = run_dir / "mlp_trials" / trial.config_id
        try:
            summary = _run_mlp_trial(
                trial_spec,
                trial,
                trial_root=trial_dir,
                run_id=resolved_run_id,
                output_layout="single_dir",
                split=split,
            )
            mlp_search_rows.append(
                _mlp_row_from_summary(
                    trial,
                    summary,
                    summary_path=(Path("mlp_trials") / trial.config_id / "summary.json").as_posix(),
                )
            )
        except Exception as exc:
            _clean_failed_trial_dir(trial_dir)
            mlp_search_rows.append(_failed_mlp_row(trial, spec, split, exc))

    pc_search_rows = rank_search_rows(pc_search_rows, spec.primary_metric_name)
    mlp_search_rows = rank_search_rows(mlp_search_rows, spec.primary_metric_name)
    _write_search_results(run_dir / "pc_search_results.csv", pc_search_rows)
    _write_search_results(run_dir / "mlp_search_results.csv", mlp_search_rows)

    best_pc_row = _select_best_row(pc_search_rows, spec.primary_metric_name)
    best_mlp_row = _select_best_row(mlp_search_rows, spec.primary_metric_name)
    best_pc_config_summary = _family_best_summary(
        benchmark_name=benchmark_name,
        run_id=resolved_run_id,
        task_name=spec.task_name,
        metric_name=spec.primary_metric_name,
        model_family="predictive_coding",
        best_row=best_pc_row,
        search_rows=pc_search_rows,
    )
    best_mlp_config_summary = _family_best_summary(
        benchmark_name=benchmark_name,
        run_id=resolved_run_id,
        task_name=spec.task_name,
        metric_name=spec.primary_metric_name,
        model_family="mlp",
        best_row=best_mlp_row,
        search_rows=mlp_search_rows,
    )
    aggregate_summary = _build_aggregate_summary(
        benchmark_name=benchmark_name,
        run_id=resolved_run_id,
        task_name=spec.task_name,
        metric_name=spec.primary_metric_name,
        pc_search_rows=pc_search_rows,
        mlp_search_rows=mlp_search_rows,
    )

    _write_json(run_dir / "best_pc_config_summary.json", best_pc_config_summary)
    _write_json(run_dir / "best_mlp_config_summary.json", best_mlp_config_summary)
    _write_json(run_dir / "aggregate_summary.json", aggregate_summary)

    return Phase2GMatchedSearchRunResult(
        run_dir=run_dir,
        study_config=study_config,
        pc_search_rows=pc_search_rows,
        mlp_search_rows=mlp_search_rows,
        aggregate_summary=aggregate_summary,
        best_pc_config_summary=best_pc_config_summary,
        best_mlp_config_summary=best_mlp_config_summary,
    )
