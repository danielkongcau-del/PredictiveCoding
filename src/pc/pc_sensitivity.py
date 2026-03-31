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

PHASE2B_BENCHMARK_NAMES = (
    "toy_regression",
    "toy_sine_regression",
)
SENSITIVITY_TOLERANCE_RTOL = 1.0e-12
SENSITIVITY_TOLERANCE_ATOL = 1.0e-12


@dataclass(frozen=True)
class PCSensitivityTrial:
    """Single one-at-a-time predictive-coding sensitivity trial."""

    trial_id: str
    parameter_group: str
    eta_x: float
    eta_w: float
    train_steps: int
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
class PCSensitivityRunResult:
    """Materialized outputs of one Phase 2b sensitivity study."""

    run_dir: Path
    candidate_grid: dict[str, Any]
    trial_rows: list[dict[str, Any]]
    aggregate_summary: dict[str, Any]
    mlp_summary: dict[str, Any]


def _resolve_run_id(run_id: str | None) -> str:
    if run_id is not None:
        return run_id
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_sensitivity_root(
    output_root: str | Path,
    benchmark_name: str,
    run_id: str,
    output_layout: OutputLayout,
) -> Path:
    experiment_name = f"pc_sensitivity_{benchmark_name}"
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


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, indent=2)


def _write_trial_table(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "trial_id",
        "parameter_group",
        "eta_x",
        "eta_w",
        "eta_b",
        "train_steps",
        "eval_steps",
        "state_init",
        "status",
        "failure_reason",
        "primary_metric_name",
        "primary_metric_value",
        "primary_metric_delta_vs_default",
        "baseline_metric_name",
        "baseline_metric_value",
        "beats_task_baseline",
        "final_pre_update_energy",
        "final_pre_update_energy_delta_vs_default",
        "final_post_update_energy",
        "best_epoch",
        "trial_summary_path",
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
        fieldnames.extend(["mse", "baseline_mse"])
    elif task_name == "classification":
        fieldnames.extend(["accuracy", "baseline_accuracy"])
    else:
        raise ValueError(f"Unsupported task '{task_name}'.")

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def metric_value_is_better(
    metric_name: str,
    candidate_value: float,
    reference_value: float,
) -> bool:
    """Return whether a candidate metric strictly beats a reference metric."""
    if np.isclose(
        candidate_value,
        reference_value,
        rtol=SENSITIVITY_TOLERANCE_RTOL,
        atol=SENSITIVITY_TOLERANCE_ATOL,
    ):
        return False

    if metric_higher_is_better(metric_name):
        return candidate_value > reference_value
    return candidate_value < reference_value


def select_sensitivity_winner(
    metric_name: str,
    left_winner_key: str,
    left_label: str,
    left_value: float,
    right_winner_key: str,
    right_label: str,
    right_value: float,
) -> tuple[str, str]:
    """Return a winner key and a human-readable reason for a metric comparison."""
    if np.isclose(
        left_value,
        right_value,
        rtol=SENSITIVITY_TOLERANCE_RTOL,
        atol=SENSITIVITY_TOLERANCE_ATOL,
    ):
        return "tie", f"tie: {left_label} equals {right_label} within tolerance"

    if metric_higher_is_better(metric_name):
        if left_value > right_value:
            return left_winner_key, f"higher_is_better: {left_label} > {right_label}"
        return right_winner_key, f"higher_is_better: {right_label} > {left_label}"

    if left_value < right_value:
        return left_winner_key, f"lower_is_better: {left_label} < {right_label}"
    return right_winner_key, f"lower_is_better: {right_label} < {left_label}"


def _candidate_values(benchmark_name: str) -> dict[str, list[float] | list[str]]:
    mapping: dict[str, dict[str, list[float] | list[str]]] = {
        "toy_regression": {
            "eta_x": [0.1, 0.2, 0.4],
            "eta_w": [0.025, 0.05, 0.1],
            "train_steps": [12, 25, 50],
            "state_init": ["forward", "zeros"],
        },
        "toy_sine_regression": {
            "eta_x": [0.075, 0.15, 0.3],
            "eta_w": [0.015, 0.03, 0.06],
            "train_steps": [15, 30, 60],
            "state_init": ["forward", "zeros"],
        },
    }
    try:
        return mapping[benchmark_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported Phase 2b benchmark '{benchmark_name}'.") from exc


def build_sensitivity_trials(spec: ToyBenchmarkSpec) -> list[PCSensitivityTrial]:
    """Return the fixed one-at-a-time trial set for one Phase 2b benchmark."""
    candidates = _candidate_values(spec.benchmark_name)
    base = spec.pc_training
    return [
        PCSensitivityTrial(
            trial_id="default",
            parameter_group="default",
            eta_x=base.eta_x,
            eta_w=base.eta_w,
            train_steps=base.train_steps,
            state_init=base.state_init,
        ),
        PCSensitivityTrial(
            trial_id="eta_x_half",
            parameter_group="eta_x",
            eta_x=float(candidates["eta_x"][0]),
            eta_w=base.eta_w,
            train_steps=base.train_steps,
            state_init=base.state_init,
        ),
        PCSensitivityTrial(
            trial_id="eta_x_double",
            parameter_group="eta_x",
            eta_x=float(candidates["eta_x"][2]),
            eta_w=base.eta_w,
            train_steps=base.train_steps,
            state_init=base.state_init,
        ),
        PCSensitivityTrial(
            trial_id="eta_w_half",
            parameter_group="eta_w",
            eta_x=base.eta_x,
            eta_w=float(candidates["eta_w"][0]),
            train_steps=base.train_steps,
            state_init=base.state_init,
        ),
        PCSensitivityTrial(
            trial_id="eta_w_double",
            parameter_group="eta_w",
            eta_x=base.eta_x,
            eta_w=float(candidates["eta_w"][2]),
            train_steps=base.train_steps,
            state_init=base.state_init,
        ),
        PCSensitivityTrial(
            trial_id="train_steps_half",
            parameter_group="train_steps",
            eta_x=base.eta_x,
            eta_w=base.eta_w,
            train_steps=int(candidates["train_steps"][0]),
            state_init=base.state_init,
        ),
        PCSensitivityTrial(
            trial_id="train_steps_double",
            parameter_group="train_steps",
            eta_x=base.eta_x,
            eta_w=base.eta_w,
            train_steps=int(candidates["train_steps"][2]),
            state_init=base.state_init,
        ),
        PCSensitivityTrial(
            trial_id="state_init_zeros",
            parameter_group="state_init",
            eta_x=base.eta_x,
            eta_w=base.eta_w,
            train_steps=base.train_steps,
            state_init=str(candidates["state_init"][1]),
        ),
    ]


def _replace_pc_training(spec: ToyBenchmarkSpec, trial: PCSensitivityTrial) -> ToyBenchmarkSpec:
    return replace(spec, pc_training=trial.to_pc_training_spec())


def _mlp_config_payload(
    spec: ToyBenchmarkSpec,
    run_id: str,
    output_root: str | Path,
    output_layout: OutputLayout,
    x: np.ndarray,
    y: np.ndarray,
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
        "data": spec.data_config(x, y),
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


def _base_pc_config_payload(
    spec: ToyBenchmarkSpec,
    run_id: str,
    output_root: str | Path,
    plot_energy: bool,
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    return {
        "experiment_name": spec.benchmark_name,
        "run_id": run_id,
        "seed": spec.run_seed,
        "seeds": {
            "run_seed": spec.run_seed,
            "data_seed": spec.data_seed,
            "model_init_seed": spec.model_init_seed,
        },
        "task": spec.task_config(),
        "data": spec.data_config(x, y),
        "model": spec.pc_model_config(),
        "training": {
            "epochs": spec.epochs,
            "eta_x": spec.pc_training.eta_x,
            "eta_w": spec.pc_training.eta_w,
            "eta_b": spec.pc_training.eta_b,
            "train_steps": spec.pc_training.train_steps,
            "eval_steps": spec.pc_training.eval_steps,
            "run_seed": spec.run_seed,
        },
        "logging": {
            "output_root": str(output_root),
            "output_layout": "single_dir",
            "plot_energy": plot_energy,
            "trace_policy": "default",
        },
    }


def _run_mlp_reference(
    spec: ToyBenchmarkSpec,
    *,
    sensitivity_root: Path,
    run_id: str,
    output_layout: OutputLayout,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    mlp_reference_dir = sensitivity_root / "mlp_reference"
    mlp_reference_dir.mkdir(parents=True, exist_ok=True)
    set_seed(spec.run_seed)
    model = spec.build_mlp_model()
    config_payload = _mlp_config_payload(
        spec,
        run_id,
        sensitivity_root,
        output_layout,
        x,
        y,
    )
    _write_json(mlp_reference_dir / "config.json", config_payload)

    baseline_metric_value = spec.baseline_metric_fn(y)
    epoch_rows: list[dict[str, Any]] = []

    for epoch in range(1, spec.epochs + 1):
        batch_result = model.train_batch(x, y)
        predictions = model.predict(x)
        primary_metric_value = spec.primary_metric_fn(predictions, y)
        row: dict[str, Any] = {
            "epoch": epoch,
            "loss": batch_result.loss,
        }
        for layer_index, value in enumerate(batch_result.parameter_norms["weight_norms"], start=1):
            row[f"weight_norm_l{layer_index}"] = value
        for layer_index, value in enumerate(batch_result.parameter_norms["bias_norms"], start=1):
            row[f"bias_norm_l{layer_index}"] = value

        if spec.task_name == "regression":
            row["mse"] = primary_metric_value
            row["baseline_mse"] = baseline_metric_value
        elif spec.task_name == "classification":
            row["accuracy"] = primary_metric_value
            row["baseline_accuracy"] = baseline_metric_value
        else:
            raise ValueError(f"Unsupported task '{spec.task_name}'.")

        epoch_rows.append(row)

    best_row = (
        max(epoch_rows, key=lambda row: row[spec.primary_metric_name])
        if spec.primary_metric_higher_is_better
        else min(epoch_rows, key=lambda row: row[spec.primary_metric_name])
    )
    final_row = epoch_rows[-1]
    summary = {
        "experiment_name": "mlp_reference",
        "run_id": run_id,
        "phase": "Phase 2b",
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
        "primary_metric_name": spec.primary_metric_name,
        "primary_metric_value": final_row[spec.primary_metric_name],
        "primary_metric_higher_is_better": spec.primary_metric_higher_is_better,
        "baseline_metric_name": spec.baseline_metric_name,
        "baseline_metric_value": baseline_metric_value,
        "best_epoch": best_row["epoch"],
    }

    _write_mlp_epoch_metrics(
        mlp_reference_dir / "epoch_metrics.csv",
        epoch_rows,
        num_layers=len(model.layers),
        task_name=spec.task_name,
    )
    _write_json(mlp_reference_dir / "summary.json", summary)
    return mlp_reference_dir, epoch_rows, summary


def _clean_failed_trial_dir(trial_dir: Path) -> None:
    if trial_dir.exists():
        shutil.rmtree(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)


def _trial_failure_reason(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _successful_trial_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["status"] == "ok"]


def _select_best_trial_row(rows: list[dict[str, Any]], metric_name: str) -> dict[str, Any]:
    if not rows:
        raise ValueError("At least one successful trial row is required.")
    if metric_higher_is_better(metric_name):
        return max(rows, key=lambda row: row["primary_metric_value"])
    return min(rows, key=lambda row: row["primary_metric_value"])


def _apply_default_deltas(
    rows: list[dict[str, Any]],
    default_metric_value: float | None,
    default_final_pre_update_energy: float | None,
) -> list[dict[str, Any]]:
    updated_rows: list[dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        if row["status"] != "ok":
            new_row["primary_metric_delta_vs_default"] = None
            new_row["final_pre_update_energy_delta_vs_default"] = None
        elif default_metric_value is None or default_final_pre_update_energy is None:
            new_row["primary_metric_delta_vs_default"] = None
            new_row["final_pre_update_energy_delta_vs_default"] = None
        elif row["trial_id"] == "default":
            new_row["primary_metric_delta_vs_default"] = 0.0
            new_row["final_pre_update_energy_delta_vs_default"] = 0.0
        else:
            new_row["primary_metric_delta_vs_default"] = (
                float(row["primary_metric_value"]) - default_metric_value
            )
            new_row["final_pre_update_energy_delta_vs_default"] = (
                float(row["final_pre_update_energy"]) - default_final_pre_update_energy
            )
        updated_rows.append(new_row)
    return updated_rows


def _parameter_group_summaries(
    rows: list[dict[str, Any]],
    metric_name: str,
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for parameter_group in ("eta_x", "eta_w", "train_steps", "state_init"):
        group_rows = [
            row
            for row in rows
            if row["status"] == "ok" and row["parameter_group"] == parameter_group
        ]
        if not group_rows:
            summaries[parameter_group] = {
                "best_trial_id": None,
                "primary_metric_value": None,
                "primary_metric_delta_vs_default": None,
                "final_pre_update_energy_delta_vs_default": None,
            }
            continue

        best_row = _select_best_trial_row(group_rows, metric_name)
        summaries[parameter_group] = {
            "best_trial_id": best_row["trial_id"],
            "primary_metric_value": best_row["primary_metric_value"],
            "primary_metric_delta_vs_default": best_row["primary_metric_delta_vs_default"],
            "final_pre_update_energy_delta_vs_default": best_row[
                "final_pre_update_energy_delta_vs_default"
            ],
        }
    return summaries


def _build_aggregate_summary(
    *,
    benchmark_name: str,
    run_id: str,
    task_name: str,
    metric_name: str,
    baseline_metric_name: str,
    trial_rows: list[dict[str, Any]],
    mlp_reference_summary: dict[str, Any],
) -> dict[str, Any]:
    successful_rows = _successful_trial_rows(trial_rows)
    if not successful_rows:
        raise ValueError("At least one successful trial is required to build an aggregate summary.")

    default_row = next(
        (
            row
            for row in trial_rows
            if row["trial_id"] == "default" and row["status"] == "ok"
        ),
        None,
    )
    best_row = _select_best_trial_row(successful_rows, metric_name)
    baseline_metric_value = float(best_row["baseline_metric_value"])
    best_metric_value = float(best_row["primary_metric_value"])
    mlp_metric_value = float(mlp_reference_summary["primary_metric_value"])

    if default_row is None:
        default_metric_value = None
        default_reason = "unavailable: default trial failed"
    else:
        default_metric_value = float(default_row["primary_metric_value"])
        _, default_reason = select_sensitivity_winner(
            metric_name,
            "best_pc",
            "best_pc_primary_metric_value",
            best_metric_value,
            "default",
            "default_pc_primary_metric_value",
            default_metric_value,
        )

    mlp_winner, mlp_reason = select_sensitivity_winner(
        metric_name,
        "best_pc",
        "best_pc_primary_metric_value",
        best_metric_value,
        "mlp_reference",
        "mlp_reference_primary_metric_value",
        mlp_metric_value,
    )

    return {
        "experiment_name": f"pc_sensitivity_{benchmark_name}",
        "run_id": run_id,
        "phase": "Phase 2b",
        "benchmark_name": benchmark_name,
        "task_name": task_name,
        "trial_design": "one_at_a_time",
        "trial_count": len(trial_rows),
        "successful_trial_count": len(successful_rows),
        "failed_trial_count": len(trial_rows) - len(successful_rows),
        "failed_trial_ids": [
            row["trial_id"] for row in trial_rows if row["status"] == "failed"
        ],
        "primary_metric_name": metric_name,
        "primary_metric_higher_is_better": metric_higher_is_better(metric_name),
        "baseline_metric_name": baseline_metric_name,
        "default_pc_trial_id": "default",
        "default_pc_primary_metric_value": default_metric_value,
        "best_pc_trial_id": best_row["trial_id"],
        "best_pc_primary_metric_value": best_metric_value,
        "best_pc_beats_task_baseline": metric_value_is_better(
            metric_name,
            best_metric_value,
            baseline_metric_value,
        ),
        "best_pc_beats_mlp_reference": mlp_winner == "best_pc",
        "best_pc_vs_default_reason": default_reason,
        "best_pc_vs_mlp_reference_reason": mlp_reason,
        "mlp_reference_primary_metric_value": mlp_metric_value,
        "best_pc_vs_mlp_reference_winner": mlp_winner,
        "best_pc_trial_summary_path": best_row["trial_summary_path"],
        "mlp_reference_summary_path": (Path("mlp_reference") / "summary.json").as_posix(),
        "parameter_group_summaries": _parameter_group_summaries(trial_rows, metric_name),
    }


def _plot_primary_metric_summary(
    run_dir: Path,
    trial_rows: list[dict[str, Any]],
    aggregate_summary: dict[str, Any],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "plot_summary=True requires matplotlib to be installed in the current environment."
        ) from exc

    successful_rows = _successful_trial_rows(trial_rows)
    labels = [str(row["trial_id"]) for row in successful_rows]
    values = [float(row["primary_metric_value"]) for row in successful_rows]

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(10, 4))
    axis = figure.add_subplot(1, 1, 1)
    axis.bar(labels, values)
    if aggregate_summary["default_pc_primary_metric_value"] is not None:
        axis.axhline(
            float(aggregate_summary["default_pc_primary_metric_value"]),
            color="tab:orange",
            linestyle="--",
            label="default_pc",
        )
    axis.axhline(
        float(aggregate_summary["mlp_reference_primary_metric_value"]),
        color="tab:green",
        linestyle=":",
        label="mlp_reference",
    )
    axis.set_ylabel(str(aggregate_summary["primary_metric_name"]))
    axis.set_title(str(aggregate_summary["benchmark_name"]))
    axis.tick_params(axis="x", rotation=45)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots_dir / "primary_metric_by_trial.png")
    plt.close(figure)


def _candidate_grid_payload(
    spec: ToyBenchmarkSpec,
    run_id: str,
    output_layout: OutputLayout,
    trials: list[PCSensitivityTrial],
) -> dict[str, Any]:
    base = spec.pc_training
    candidates = _candidate_values(spec.benchmark_name)
    return {
        "experiment_name": f"pc_sensitivity_{spec.benchmark_name}",
        "run_id": run_id,
        "benchmark_name": spec.benchmark_name,
        "task_name": spec.task_name,
        "output_layout": output_layout,
        "seeds": {
            "run_seed": spec.run_seed,
            "data_seed": spec.data_seed,
            "model_init_seed": spec.model_init_seed,
        },
        "fixed_defaults": {
            "eta_x": base.eta_x,
            "eta_w": base.eta_w,
            "eta_b": base.eta_b,
            "train_steps": base.train_steps,
            "eval_steps": base.eval_steps,
            "state_init": base.state_init,
            "epochs": spec.epochs,
        },
        "trial_design": "one_at_a_time",
        "coupling_rules": {
            "eta_b_equals_eta_w": True,
            "eval_steps_equals_train_steps": True,
        },
        "candidates": candidates,
        "trials": [
            {
                "trial_id": trial.trial_id,
                "parameter_group": trial.parameter_group,
                "eta_x": trial.eta_x,
                "eta_w": trial.eta_w,
                "eta_b": trial.eta_b,
                "train_steps": trial.train_steps,
                "eval_steps": trial.eval_steps,
                "state_init": trial.state_init,
            }
            for trial in trials
        ],
        "notes": {
            "state_init": (
                "state_init affects the full predictive-coding run path, including "
                "training-state initialization and prediction/eval-state initialization."
            ),
        },
    }


def run_pc_sensitivity_study(
    benchmark_name: str,
    *,
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    output_layout: OutputLayout = "single_dir",
    plot_energy: bool = False,
    plot_summary: bool = False,
) -> PCSensitivityRunResult:
    """Run the fixed Phase 2b PC sensitivity study for one regression benchmark."""
    if benchmark_name not in PHASE2B_BENCHMARK_NAMES:
        raise ValueError(
            f"Phase 2b currently supports only {PHASE2B_BENCHMARK_NAMES}, got '{benchmark_name}'."
        )

    spec = get_benchmark_spec(benchmark_name)
    resolved_run_id = _resolve_run_id(run_id)
    run_dir = _prepare_run_dir(
        _resolve_sensitivity_root(output_root, benchmark_name, resolved_run_id, output_layout)
    )
    x, y = spec.make_data()
    trials = build_sensitivity_trials(spec)
    candidate_grid = _candidate_grid_payload(spec, resolved_run_id, output_layout, trials)
    _write_json(run_dir / "candidate_grid.json", candidate_grid)

    _, _, mlp_summary = _run_mlp_reference(
        spec,
        sensitivity_root=run_dir,
        run_id=resolved_run_id,
        output_layout=output_layout,
        x=x,
        y=y,
    )
    baseline_metric_value = spec.baseline_metric_fn(y)

    default_metric_value: float | None = None
    default_final_pre_update_energy: float | None = None
    trial_rows: list[dict[str, Any]] = []
    _write_json(
        run_dir / "base_pc_config.json",
        _base_pc_config_payload(
            spec,
            resolved_run_id,
            run_dir / "trials",
            plot_energy,
            x,
            y,
        ),
    )

    for trial in trials:
        trial_spec = _replace_pc_training(spec, trial)
        trial_dir = run_dir / "trials" / trial.trial_id
        try:
            result = run_pc_benchmark(
                trial_spec,
                output_root=run_dir / "trials",
                run_id=resolved_run_id,
                plot_energy=plot_energy,
                output_layout="single_dir",
                experiment_name=trial.trial_id,
                x=x,
                y=y,
            )
            summary = result.summary
            row: dict[str, Any] = {
                "trial_id": trial.trial_id,
                "parameter_group": trial.parameter_group,
                "eta_x": trial.eta_x,
                "eta_w": trial.eta_w,
                "eta_b": trial.eta_b,
                "train_steps": trial.train_steps,
                "eval_steps": trial.eval_steps,
                "state_init": trial.state_init,
                "status": "ok",
                "failure_reason": "",
                "primary_metric_name": spec.primary_metric_name,
                "primary_metric_value": float(summary["primary_metric_value"]),
                "baseline_metric_name": spec.baseline_metric_name,
                "baseline_metric_value": float(summary["baseline_metric_value"]),
                "beats_task_baseline": metric_value_is_better(
                    spec.primary_metric_name,
                    float(summary["primary_metric_value"]),
                    float(summary["baseline_metric_value"]),
                ),
                "final_pre_update_energy": float(summary["final_pre_update_energy"]),
                "final_post_update_energy": float(summary["final_post_update_energy"]),
                "best_epoch": int(summary["best_epoch"]),
                "trial_summary_path": (Path("trials") / trial.trial_id / "summary.json").as_posix(),
            }
            trial_rows.append(row)

            if trial.trial_id == "default":
                default_metric_value = float(summary["primary_metric_value"])
                default_final_pre_update_energy = float(summary["final_pre_update_energy"])
        except Exception as exc:
            _clean_failed_trial_dir(trial_dir)
            trial_rows.append(
                {
                    "trial_id": trial.trial_id,
                    "parameter_group": trial.parameter_group,
                    "eta_x": trial.eta_x,
                    "eta_w": trial.eta_w,
                    "eta_b": trial.eta_b,
                    "train_steps": trial.train_steps,
                    "eval_steps": trial.eval_steps,
                    "state_init": trial.state_init,
                    "status": "failed",
                    "failure_reason": _trial_failure_reason(exc),
                    "primary_metric_name": spec.primary_metric_name,
                    "primary_metric_value": None,
                    "baseline_metric_name": spec.baseline_metric_name,
                    "baseline_metric_value": baseline_metric_value,
                    "beats_task_baseline": None,
                    "final_pre_update_energy": None,
                    "final_post_update_energy": None,
                    "best_epoch": None,
                    "trial_summary_path": "",
                }
            )

    trial_rows = _apply_default_deltas(
        trial_rows,
        default_metric_value,
        default_final_pre_update_energy,
    )
    _write_trial_table(run_dir / "trial_table.csv", trial_rows)

    aggregate_summary = _build_aggregate_summary(
        benchmark_name=benchmark_name,
        run_id=resolved_run_id,
        task_name=spec.task_name,
        metric_name=spec.primary_metric_name,
        baseline_metric_name=spec.baseline_metric_name,
        trial_rows=trial_rows,
        mlp_reference_summary=mlp_summary,
    )
    _write_json(run_dir / "aggregate_summary.json", aggregate_summary)

    if plot_summary:
        _plot_primary_metric_summary(run_dir, trial_rows, aggregate_summary)

    return PCSensitivityRunResult(
        run_dir=run_dir,
        candidate_grid=candidate_grid,
        trial_rows=trial_rows,
        aggregate_summary=aggregate_summary,
        mlp_summary=mlp_summary,
    )
