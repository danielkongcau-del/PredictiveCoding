from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .benchmark_specs import PCTrainingSpec, ToyBenchmarkSpec, get_benchmark_spec, run_pc_benchmark
from .experiment import OutputLayout
from .metrics import metric_higher_is_better
from .utils import set_seed

PHASE2F_BENCHMARK_NAMES = (
    "toy_regression",
    "toy_sine_regression",
)
JOINT_SEARCH_TOLERANCE_RTOL = 1.0e-12
JOINT_SEARCH_TOLERANCE_ATOL = 1.0e-12


@dataclass(frozen=True)
class PCJointSearchTrial:
    """One deterministic Phase 2f predictive-coding configuration."""

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


@dataclass
class PCJointSearchRunResult:
    """Materialized outputs of one Phase 2f joint search run."""

    run_dir: Path
    study_config: dict[str, Any]
    search_rows: list[dict[str, Any]]
    aggregate_summary: dict[str, Any]
    best_config_summary: dict[str, Any]
    mlp_summary: dict[str, Any]


def _resolve_run_id(run_id: str | None) -> str:
    if run_id is not None:
        return run_id
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_joint_search_root(
    output_root: str | Path,
    benchmark_name: str,
    run_id: str,
    output_layout: OutputLayout,
) -> Path:
    base_dir = Path(output_root) / "phase2f_joint_search" / benchmark_name
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
        "val_metric_rank",
        "val_metric_delta_vs_best",
        "train_baseline_metric",
        "val_baseline_metric",
        "test_baseline_metric",
        "beats_val_baseline",
        "best_epoch",
        "final_pre_update_energy",
        "final_post_update_energy",
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


def _default_search_space(spec: ToyBenchmarkSpec) -> dict[str, list[float] | list[int]]:
    if spec.benchmark_name == "toy_regression":
        return {
            "eta_x": [0.05, 0.1, 0.2],
            "eta_w": [0.1, 0.2, 0.4],
            "train_steps": [25, 50, 100],
            "epochs": [spec.epochs],
        }
    if spec.benchmark_name == "toy_sine_regression":
        return {
            "eta_x": [0.05, 0.075, 0.1, 0.15],
            "eta_w": [0.06, 0.1, 0.2],
            "train_steps": [30, 60, 120],
            "epochs": [spec.epochs],
        }
    raise ValueError(f"Unsupported Phase 2f benchmark '{spec.benchmark_name}'.")


def resolve_joint_search_space(
    spec: ToyBenchmarkSpec,
    search_space_override: dict[str, list[float] | list[int]] | None = None,
) -> dict[str, list[float] | list[int]]:
    """Return the deterministic Phase 2f search space for one regression benchmark."""
    search_space = _default_search_space(spec)
    if search_space_override is not None:
        for key, values in search_space_override.items():
            if key not in search_space:
                raise ValueError(f"Unsupported search-space key '{key}'.")
            if len(values) == 0:
                raise ValueError(f"Search-space key '{key}' must contain at least one value.")
            search_space[key] = list(values)
    return search_space


def build_joint_search_trials(
    spec: ToyBenchmarkSpec,
    search_space_override: dict[str, list[float] | list[int]] | None = None,
) -> tuple[list[PCJointSearchTrial], dict[str, list[float] | list[int]]]:
    """Return the deterministic Cartesian product for one Phase 2f benchmark."""
    search_space = resolve_joint_search_space(spec, search_space_override)
    trials: list[PCJointSearchTrial] = []
    config_index = 1

    for eta_x in search_space["eta_x"]:
        for eta_w in search_space["eta_w"]:
            for train_steps in search_space["train_steps"]:
                for epochs in search_space["epochs"]:
                    trials.append(
                        PCJointSearchTrial(
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


def _mlp_config_payload(
    spec: ToyBenchmarkSpec,
    run_id: str,
    search_root: Path,
    output_layout: OutputLayout,
    split,
) -> dict[str, Any]:
    return {
        "experiment_name": "mlp_reference",
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
            "output_root": str(search_root),
            "output_layout": output_layout,
            "plot_energy": False,
            "trace_policy": "not_applicable",
        },
    }


def _run_mlp_reference(
    spec: ToyBenchmarkSpec,
    *,
    search_root: Path,
    run_id: str,
    output_layout: OutputLayout,
    split,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    mlp_reference_dir = search_root / "mlp_reference"
    mlp_reference_dir.mkdir(parents=True, exist_ok=True)
    set_seed(spec.run_seed)
    model = spec.build_mlp_model()
    _write_json(
        mlp_reference_dir / "config.json",
        _mlp_config_payload(spec, run_id, search_root, output_layout, split),
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
        "experiment_name": "mlp_reference",
        "run_id": run_id,
        "phase": "Phase 2f",
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
        mlp_reference_dir / "epoch_metrics.csv",
        epoch_rows,
        num_layers=len(model.layers),
        task_name=spec.task_name,
    )
    _write_json(mlp_reference_dir / "summary.json", summary)
    return mlp_reference_dir, epoch_rows, summary


def metric_value_is_better(
    metric_name: str,
    candidate_value: float,
    reference_value: float,
) -> bool:
    """Return whether a candidate metric strictly beats a reference metric."""
    if np.isclose(
        candidate_value,
        reference_value,
        rtol=JOINT_SEARCH_TOLERANCE_RTOL,
        atol=JOINT_SEARCH_TOLERANCE_ATOL,
    ):
        return False

    if metric_higher_is_better(metric_name):
        return candidate_value > reference_value
    return candidate_value < reference_value


def rank_joint_search_rows(
    rows: list[dict[str, Any]],
    metric_name: str,
) -> list[dict[str, Any]]:
    """Return rows with deterministic rank fields based on val_metric only."""
    successful_rows = [row for row in rows if row["status"] == "ok"]
    if not successful_rows:
        return [dict(row, val_metric_rank=None, val_metric_delta_vs_best=None) for row in rows]

    if metric_higher_is_better(metric_name):
        ranked_successful = sorted(
            successful_rows,
            key=lambda row: (-float(row["val_metric"]), str(row["config_id"])),
        )
    else:
        ranked_successful = sorted(
            successful_rows,
            key=lambda row: (float(row["val_metric"]), str(row["config_id"])),
        )
    rank_mapping = {row["config_id"]: rank for rank, row in enumerate(ranked_successful, start=1)}
    best_val_metric = float(ranked_successful[0]["val_metric"])

    ranked_rows: list[dict[str, Any]] = []
    for row in rows:
        ranked_row = dict(row)
        if row["status"] != "ok":
            ranked_row["val_metric_rank"] = None
            ranked_row["val_metric_delta_vs_best"] = None
        else:
            ranked_row["val_metric_rank"] = rank_mapping[row["config_id"]]
            ranked_row["val_metric_delta_vs_best"] = float(row["val_metric"]) - best_val_metric
        ranked_rows.append(ranked_row)
    return ranked_rows


def _select_best_row(rows: list[dict[str, Any]], metric_name: str) -> dict[str, Any]:
    successful_rows = [row for row in rows if row["status"] == "ok"]
    if not successful_rows:
        raise ValueError("At least one successful configuration is required.")
    ranked_rows = rank_joint_search_rows(successful_rows, metric_name)
    best_ranked_row = min(ranked_rows, key=lambda row: int(row["val_metric_rank"]))
    for row in successful_rows:
        if row["config_id"] == best_ranked_row["config_id"]:
            return row
    raise RuntimeError("Best row lookup failed after ranking.")


def _selection_reason(metric_name: str, successful_count: int) -> str:
    if metric_higher_is_better(metric_name):
        return f"selected highest val_{metric_name} across {successful_count} successful configurations"
    return f"selected lowest val_{metric_name} across {successful_count} successful configurations"


def _build_aggregate_summary(
    *,
    benchmark_name: str,
    run_id: str,
    task_name: str,
    metric_name: str,
    search_rows: list[dict[str, Any]],
    mlp_summary: dict[str, Any],
) -> dict[str, Any]:
    successful_rows = [row for row in search_rows if row["status"] == "ok"]
    if not successful_rows:
        raise ValueError("At least one successful configuration is required.")

    ranked_rows = rank_joint_search_rows(search_rows, metric_name)
    best_row = _select_best_row(ranked_rows, metric_name)
    best_rank = next(
        int(row["val_metric_rank"])
        for row in ranked_rows
        if row["config_id"] == best_row["config_id"]
    )
    top_ranked_configs = [
        {
            "config_id": row["config_id"],
            "val_metric": row["val_metric"],
            "test_metric": row["test_metric"],
            "train_metric": row["train_metric"],
            "val_metric_rank": row["val_metric_rank"],
        }
        for row in sorted(
            [row for row in ranked_rows if row["status"] == "ok"],
            key=lambda row: int(row["val_metric_rank"]),
        )[:5]
    ]

    best_test_metric = float(best_row["test_metric"])
    mlp_test_metric = float(mlp_summary["test_metric"])

    return {
        "experiment_name": f"phase2f_joint_search_{benchmark_name}",
        "run_id": run_id,
        "phase": "Phase 2f",
        "benchmark_name": benchmark_name,
        "task_name": task_name,
        "search_target": "predictive_coding",
        "ranking_metric_name": metric_name,
        "ranking_metric_source": "val_metric",
        "ranking_metric_higher_is_better": metric_higher_is_better(metric_name),
        "trial_count": len(search_rows),
        "successful_trial_count": len(successful_rows),
        "failed_trial_count": len(search_rows) - len(successful_rows),
        "failed_config_ids": [
            row["config_id"] for row in search_rows if row["status"] == "failed"
        ],
        "best_config_id": best_row["config_id"],
        "best_config_rank": best_rank,
        "best_config": {
            "eta_x": best_row["eta_x"],
            "eta_w": best_row["eta_w"],
            "eta_b": best_row["eta_b"],
            "train_steps": best_row["train_steps"],
            "eval_steps": best_row["eval_steps"],
            "epochs": best_row["epochs"],
            "state_init": best_row["state_init"],
        },
        "best_train_metric": best_row["train_metric"],
        "best_val_metric": best_row["val_metric"],
        "best_test_metric": best_row["test_metric"],
        "best_val_baseline_metric": best_row["val_baseline_metric"],
        "best_test_baseline_metric": best_row["test_baseline_metric"],
        "best_epoch": best_row["best_epoch"],
        "best_summary_path": best_row["summary_path"],
        "best_pc_beats_mlp_reference": metric_value_is_better(
            metric_name,
            best_test_metric,
            mlp_test_metric,
        ),
        "mlp_reference_train_metric": mlp_summary["train_metric"],
        "mlp_reference_val_metric": mlp_summary["val_metric"],
        "mlp_reference_test_metric": mlp_summary["test_metric"],
        "selection_reason": _selection_reason(metric_name, len(successful_rows)),
        "top_ranked_configs": top_ranked_configs,
    }


def _build_best_config_summary(
    aggregate_summary: dict[str, Any],
    best_row: dict[str, Any],
    mlp_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment_name": aggregate_summary["experiment_name"],
        "run_id": aggregate_summary["run_id"],
        "phase": "Phase 2f",
        "benchmark_name": aggregate_summary["benchmark_name"],
        "task_name": aggregate_summary["task_name"],
        "ranking_metric_name": aggregate_summary["ranking_metric_name"],
        "ranking_metric_source": "val_metric",
        "best_config_id": aggregate_summary["best_config_id"],
        "best_config_rank": aggregate_summary["best_config_rank"],
        "best_config": dict(aggregate_summary["best_config"]),
        "train_metric": best_row["train_metric"],
        "val_metric": best_row["val_metric"],
        "test_metric": best_row["test_metric"],
        "eval_metric": best_row["val_metric"],
        "val_baseline_metric": best_row["val_baseline_metric"],
        "test_baseline_metric": best_row["test_baseline_metric"],
        "best_epoch": best_row["best_epoch"],
        "final_pre_update_energy": best_row["final_pre_update_energy"],
        "final_post_update_energy": best_row["final_post_update_energy"],
        "summary_path": best_row["summary_path"],
        "selection_metric_source": "val_metric",
        "selection_metric_value": best_row["val_metric"],
        "report_metric_source": "test_metric",
        "report_metric_value": best_row["test_metric"],
        "mlp_reference_val_metric": mlp_summary["val_metric"],
        "mlp_reference_test_metric": mlp_summary["test_metric"],
        "best_pc_beats_mlp_reference": aggregate_summary["best_pc_beats_mlp_reference"],
        "selection_reason": aggregate_summary["selection_reason"],
    }


def _study_config_payload(
    spec: ToyBenchmarkSpec,
    *,
    run_id: str,
    output_layout: OutputLayout,
    split,
    search_space: dict[str, list[float] | list[int]],
    trial_count: int,
) -> dict[str, Any]:
    return {
        "experiment_name": f"phase2f_joint_search_{spec.benchmark_name}",
        "run_id": run_id,
        "phase": "Phase 2f",
        "benchmark_name": spec.benchmark_name,
        "task_name": spec.task_name,
        "search_target": "predictive_coding",
        "ranking_metric_name": spec.primary_metric_name,
        "ranking_metric_source": "val_metric",
        "ranking_metric_higher_is_better": spec.primary_metric_higher_is_better,
        "seeds": {
            "run_seed": spec.run_seed,
            "data_seed": spec.data_seed,
            "model_init_seed": spec.model_init_seed,
        },
        "data": spec.data_config(split),
        "base_pc_training": {
            "eta_x": spec.pc_training.eta_x,
            "eta_w": spec.pc_training.eta_w,
            "eta_b": spec.pc_training.eta_b,
            "train_steps": spec.pc_training.train_steps,
            "eval_steps": spec.pc_training.eval_steps,
            "state_init": spec.pc_training.state_init,
            "epochs": spec.epochs,
        },
        "search_space": search_space,
        "total_search_space_size": trial_count,
        "fixed_fields": {
            "state_init": spec.pc_training.state_init,
            "eval_steps_equals_train_steps": True,
            "eta_b_equals_eta_w": True,
        },
        "mlp_training": {
            "eta_w": spec.mlp_training.eta_w,
            "eta_b": spec.mlp_training.eta_b,
            "epochs": spec.epochs,
        },
        "output_layout": output_layout,
        "notes": {
            "epochs": (
                "Epochs are fixed by default in Phase 2f to keep the initial joint search "
                "runtime modest; pass search_space_override['epochs'] to widen the search."
            ),
        },
    }


def run_pc_joint_search(
    benchmark_name: str,
    *,
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    output_layout: OutputLayout = "single_dir",
    plot_energy: bool = False,
    search_space_override: dict[str, list[float] | list[int]] | None = None,
) -> PCJointSearchRunResult:
    """Run the Phase 2f deterministic PC joint search ranked by validation metric."""
    if benchmark_name not in PHASE2F_BENCHMARK_NAMES:
        raise ValueError(
            f"Phase 2f currently supports only {PHASE2F_BENCHMARK_NAMES}, got '{benchmark_name}'."
        )

    spec = get_benchmark_spec(benchmark_name)
    split = spec.make_dataset_split()
    resolved_run_id = _resolve_run_id(run_id)
    trials, search_space = build_joint_search_trials(spec, search_space_override)
    run_dir = _prepare_run_dir(
        _resolve_joint_search_root(output_root, benchmark_name, resolved_run_id, output_layout)
    )

    study_config = _study_config_payload(
        spec,
        run_id=resolved_run_id,
        output_layout=output_layout,
        split=split,
        search_space=search_space,
        trial_count=len(trials),
    )
    _write_json(run_dir / "study_config.json", study_config)

    _, _, mlp_summary = _run_mlp_reference(
        spec,
        search_root=run_dir,
        run_id=resolved_run_id,
        output_layout=output_layout,
        split=split,
    )

    search_rows: list[dict[str, Any]] = []
    for trial in trials:
        trial_spec = replace(
            spec,
            epochs=trial.epochs,
            pc_training=trial.to_pc_training_spec(),
        )
        trial_dir = run_dir / "trials" / trial.config_id
        try:
            result = run_pc_benchmark(
                trial_spec,
                output_root=run_dir / "trials",
                run_id=resolved_run_id,
                plot_energy=plot_energy,
                output_layout="single_dir",
                experiment_name=trial.config_id,
                split=split,
            )
            summary = result.summary
            search_rows.append(
                {
                    "config_id": trial.config_id,
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
                    "val_metric_rank": None,
                    "val_metric_delta_vs_best": None,
                    "train_baseline_metric": float(summary["train_baseline_metric"]),
                    "val_baseline_metric": float(summary["val_baseline_metric"]),
                    "test_baseline_metric": float(summary["test_baseline_metric"]),
                    "beats_val_baseline": metric_value_is_better(
                        spec.primary_metric_name,
                        float(summary["val_metric"]),
                        float(summary["val_baseline_metric"]),
                    ),
                    "best_epoch": int(summary["best_epoch"]),
                    "final_pre_update_energy": float(summary["final_pre_update_energy"]),
                    "final_post_update_energy": float(summary["final_post_update_energy"]),
                    "summary_path": (Path("trials") / trial.config_id / "summary.json").as_posix(),
                }
            )
        except Exception as exc:
            _clean_failed_trial_dir(trial_dir)
            search_rows.append(
                {
                    "config_id": trial.config_id,
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
                    "val_metric_rank": None,
                    "val_metric_delta_vs_best": None,
                    "train_baseline_metric": spec.baseline_metric_fn(split.y_train),
                    "val_baseline_metric": spec.baseline_metric_fn(split.y_val),
                    "test_baseline_metric": spec.baseline_metric_fn(split.y_test),
                    "beats_val_baseline": None,
                    "best_epoch": None,
                    "final_pre_update_energy": None,
                    "final_post_update_energy": None,
                    "summary_path": "",
                }
            )

    search_rows = rank_joint_search_rows(search_rows, spec.primary_metric_name)
    _write_search_results(run_dir / "search_results.csv", search_rows)

    aggregate_summary = _build_aggregate_summary(
        benchmark_name=benchmark_name,
        run_id=resolved_run_id,
        task_name=spec.task_name,
        metric_name=spec.primary_metric_name,
        search_rows=search_rows,
        mlp_summary=mlp_summary,
    )
    _write_json(run_dir / "aggregate_summary.json", aggregate_summary)

    best_row = next(
        row
        for row in search_rows
        if row["config_id"] == aggregate_summary["best_config_id"]
    )
    best_config_summary = _build_best_config_summary(
        aggregate_summary,
        best_row,
        mlp_summary,
    )
    _write_json(run_dir / "best_config_summary.json", best_config_summary)

    return PCJointSearchRunResult(
        run_dir=run_dir,
        study_config=study_config,
        search_rows=search_rows,
        aggregate_summary=aggregate_summary,
        best_config_summary=best_config_summary,
        mlp_summary=mlp_summary,
    )
