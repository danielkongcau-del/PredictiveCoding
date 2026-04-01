from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .benchmark_specs import PCTrainingSpec, ToyBenchmarkSpec, get_benchmark_spec, run_pc_benchmark
from .experiment import OutputLayout
from .metrics import metric_higher_is_better
from .pc_multiseed import default_seed_values_for_benchmark, get_phase2c_tuned_pc_training
from .utils import set_seed

PHASE2D_BENCHMARK_NAMES = (
    "toy_regression",
    "toy_sine_regression",
)
PHASE2D_VARIANT_GROUPS = {
    "default_pc": "main_pc",
    "tuned_pc": "main_pc",
    "tuned_pc_budget2x": "budget_check",
    "mlp": "mlp_reference",
}
PHASE2D_VARIANT_ORDER = (
    "default_pc",
    "tuned_pc",
    "tuned_pc_budget2x",
    "mlp",
)
PHASE2D_BUDGET_DIAGNOSTIC_VARIANT = "tuned_pc_budget2x"
TIE_RTOL = 1.0e-12
TIE_ATOL = 1.0e-12


@dataclass
class PCDiagnosticsRunResult:
    """Materialized outputs of one Phase 2d diagnostic study."""

    run_dir: Path
    study_config: dict[str, Any]
    seed_records: list[dict[str, Any]]
    epoch_records: list[dict[str, Any]]
    epoch_summary: list[dict[str, Any]]
    diagnostic_summary: dict[str, Any]


def _resolve_run_id(run_id: str | None) -> str:
    if run_id is not None:
        return run_id
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_diagnostics_root(
    output_root: str | Path,
    benchmark_name: str,
    run_id: str,
    output_layout: OutputLayout,
) -> Path:
    experiment_name = f"pc_diagnostics_{benchmark_name}"
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


def _write_seed_records(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "seed_index",
        "run_seed",
        "data_seed",
        "model_init_seed",
        "default_pc_status",
        "default_pc_failure_reason",
        "default_pc_primary_metric_value",
        "default_pc_best_epoch",
        "default_pc_final_minus_best_metric",
        "default_pc_final_pre_update_energy",
        "default_pc_summary_path",
        "tuned_pc_status",
        "tuned_pc_failure_reason",
        "tuned_pc_primary_metric_value",
        "tuned_pc_best_epoch",
        "tuned_pc_final_minus_best_metric",
        "tuned_pc_final_pre_update_energy",
        "tuned_pc_summary_path",
        "mlp_status",
        "mlp_failure_reason",
        "mlp_primary_metric_value",
        "mlp_best_epoch",
        "mlp_final_minus_best_metric",
        "mlp_final_loss",
        "mlp_summary_path",
        "tuned_pc_budget2x_status",
        "tuned_pc_budget2x_failure_reason",
        "tuned_pc_budget2x_primary_metric_value",
        "tuned_pc_budget2x_best_epoch",
        "tuned_pc_budget2x_final_minus_best_metric",
        "tuned_pc_budget2x_final_pre_update_energy",
        "tuned_pc_budget2x_summary_path",
        "primary_metric_delta_tuned_pc_minus_default_pc",
        "primary_metric_delta_mlp_minus_tuned_pc",
        "primary_metric_delta_budget2x_minus_tuned_pc",
        "tuned_pc_beats_default_pc",
        "tuned_pc_beats_mlp",
        "budget2x_beats_tuned",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_epoch_records(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "seed_index",
        "run_seed",
        "variant",
        "variant_group",
        "epoch",
        "primary_metric_name",
        "primary_metric_value",
        "pre_update_energy",
        "post_update_energy",
        "loss",
        "train_steps",
        "summary_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_epoch_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "variant",
        "variant_group",
        "epoch",
        "seed_count",
        "primary_metric_mean",
        "primary_metric_std",
        "pre_update_energy_mean",
        "pre_update_energy_std",
        "post_update_energy_mean",
        "post_update_energy_std",
        "loss_mean",
        "loss_std",
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


def _training_dict(training: PCTrainingSpec, *, epochs: int) -> dict[str, Any]:
    return {
        "epochs": epochs,
        "eta_x": training.eta_x,
        "eta_w": training.eta_w,
        "eta_b": training.eta_b,
        "train_steps": training.train_steps,
        "eval_steps": training.eval_steps,
        "state_init": training.state_init,
    }


def _budget_diagnostic_training(tuned_training: PCTrainingSpec) -> PCTrainingSpec:
    return PCTrainingSpec(
        eta_x=tuned_training.eta_x,
        eta_w=tuned_training.eta_w,
        eta_b=tuned_training.eta_b,
        train_steps=2 * tuned_training.train_steps,
        eval_steps=2 * tuned_training.eval_steps,
        state_init=tuned_training.state_init,
    )


def _seeded_spec(
    base_spec: ToyBenchmarkSpec,
    *,
    seed: int,
    pc_training: PCTrainingSpec | None = None,
) -> ToyBenchmarkSpec:
    return replace(
        base_spec,
        run_seed=seed,
        model_init_seed=seed,
        data_seed=base_spec.data_seed,
        pc_training=base_spec.pc_training if pc_training is None else pc_training,
    )


def _metric_value_is_better(metric_name: str, left_value: float, right_value: float) -> bool:
    if np.isclose(left_value, right_value, rtol=TIE_RTOL, atol=TIE_ATOL):
        return False
    if metric_higher_is_better(metric_name):
        return left_value > right_value
    return left_value < right_value


def _compare_metric_values(metric_name: str, left_value: float, right_value: float) -> str:
    if np.isclose(left_value, right_value, rtol=TIE_RTOL, atol=TIE_ATOL):
        return "tie"
    if metric_higher_is_better(metric_name):
        return "left" if left_value > right_value else "right"
    return "left" if left_value < right_value else "right"


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _std_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=0))


def _resolve_variant_group(variant: str) -> str:
    try:
        return PHASE2D_VARIANT_GROUPS[variant]
    except KeyError as exc:
        raise ValueError(f"Unsupported diagnostic variant '{variant}'.") from exc


def _mlp_config_payload(
    spec: ToyBenchmarkSpec,
    run_id: str,
    output_root: str | Path,
    output_layout: OutputLayout,
    x: np.ndarray,
    y: np.ndarray,
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


def _run_mlp_variant(
    spec: ToyBenchmarkSpec,
    *,
    variant_dir: Path,
    run_id: str,
    output_layout: OutputLayout,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    variant_dir.mkdir(parents=True, exist_ok=True)
    set_seed(spec.run_seed)
    model = spec.build_mlp_model()
    config_payload = _mlp_config_payload(spec, run_id, variant_dir.parent, output_layout, x, y)
    _write_json(variant_dir / "config.json", config_payload)

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
        "experiment_name": "mlp",
        "run_id": run_id,
        "phase": "Phase 2d",
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
        variant_dir / "epoch_metrics.csv",
        epoch_rows,
        num_layers=len(model.layers),
        task_name=spec.task_name,
    )
    _write_json(variant_dir / "summary.json", summary)
    return variant_dir, epoch_rows, summary


def _clean_failed_variant_dir(variant_dir: Path) -> None:
    if variant_dir.exists():
        shutil.rmtree(variant_dir)
    variant_dir.mkdir(parents=True, exist_ok=True)


def _failure_reason(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _best_epoch_and_metric(
    rows: list[dict[str, Any]],
    metric_name: str,
    *,
    higher_is_better: bool,
) -> tuple[int, float, float]:
    best_row = max(rows, key=lambda row: row[metric_name]) if higher_is_better else min(
        rows,
        key=lambda row: row[metric_name],
    )
    final_row = rows[-1]
    return int(best_row["epoch"]), float(best_row[metric_name]), float(final_row[metric_name])


def _summary_path_for_seed_variant(seed: int, variant: str) -> str:
    return (Path("seeds") / f"seed_{int(seed):04d}" / variant / "summary.json").as_posix()


def _append_pc_epoch_records(
    epoch_records: list[dict[str, Any]],
    *,
    seed_index: int,
    seed: int,
    variant: str,
    summary_path: str,
    result_rows: list[dict[str, Any]],
    primary_metric_name: str,
) -> None:
    variant_group = _resolve_variant_group(variant)
    for row in result_rows:
        epoch_records.append(
            {
                "seed_index": seed_index,
                "run_seed": int(seed),
                "variant": variant,
                "variant_group": variant_group,
                "epoch": int(row["epoch"]),
                "primary_metric_name": primary_metric_name,
                "primary_metric_value": float(row[primary_metric_name]),
                "pre_update_energy": float(row["pre_update_energy"]),
                "post_update_energy": float(row["post_update_energy"]),
                "loss": None,
                "train_steps": int(row["train_steps"]),
                "summary_path": summary_path,
            }
        )


def _append_mlp_epoch_records(
    epoch_records: list[dict[str, Any]],
    *,
    seed_index: int,
    seed: int,
    summary_path: str,
    result_rows: list[dict[str, Any]],
    primary_metric_name: str,
) -> None:
    variant = "mlp"
    variant_group = _resolve_variant_group(variant)
    for row in result_rows:
        epoch_records.append(
            {
                "seed_index": seed_index,
                "run_seed": int(seed),
                "variant": variant,
                "variant_group": variant_group,
                "epoch": int(row["epoch"]),
                "primary_metric_name": primary_metric_name,
                "primary_metric_value": float(row[primary_metric_name]),
                "pre_update_energy": None,
                "post_update_energy": None,
                "loss": float(row["loss"]),
                "train_steps": None,
                "summary_path": summary_path,
            }
        )


def _group_values(rows: list[dict[str, Any]], field_name: str) -> list[float]:
    return [float(row[field_name]) for row in rows if row[field_name] is not None]


def _build_epoch_summary_rows(epoch_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in epoch_records:
        key = (str(row["variant"]), int(row["epoch"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    variant_order = {name: index for index, name in enumerate(PHASE2D_VARIANT_ORDER)}
    for (variant, epoch), rows in sorted(
        grouped.items(),
        key=lambda item: (variant_order[item[0][0]], item[0][1]),
    ):
        summary_rows.append(
            {
                "variant": variant,
                "variant_group": _resolve_variant_group(variant),
                "epoch": epoch,
                "seed_count": len(rows),
                "primary_metric_mean": _mean_or_none(_group_values(rows, "primary_metric_value")),
                "primary_metric_std": _std_or_none(_group_values(rows, "primary_metric_value")),
                "pre_update_energy_mean": _mean_or_none(_group_values(rows, "pre_update_energy")),
                "pre_update_energy_std": _std_or_none(_group_values(rows, "pre_update_energy")),
                "post_update_energy_mean": _mean_or_none(_group_values(rows, "post_update_energy")),
                "post_update_energy_std": _std_or_none(_group_values(rows, "post_update_energy")),
                "loss_mean": _mean_or_none(_group_values(rows, "loss")),
                "loss_std": _std_or_none(_group_values(rows, "loss")),
            }
        )
    return summary_rows


def _correlation_from_epoch_mean_curve(
    epoch_summary_rows: list[dict[str, Any]],
    *,
    variant: str,
    energy_field: str,
) -> tuple[float | None, int]:
    filtered_rows = [
        row
        for row in epoch_summary_rows
        if row["variant"] == variant
        and row[energy_field] is not None
        and row["primary_metric_mean"] is not None
    ]
    energy_values = np.asarray([float(row[energy_field]) for row in filtered_rows], dtype=np.float64)
    metric_values = np.asarray(
        [float(row["primary_metric_mean"]) for row in filtered_rows],
        dtype=np.float64,
    )
    sample_count = int(energy_values.shape[0])
    if sample_count < 2:
        return None, sample_count
    if np.isclose(np.std(energy_values, ddof=0), 0.0) or np.isclose(
        np.std(metric_values, ddof=0),
        0.0,
    ):
        return None, sample_count
    correlation = float(np.corrcoef(energy_values, metric_values)[0, 1])
    if np.isnan(correlation):
        return None, sample_count
    return correlation, sample_count


def _variant_value_list(seed_records: list[dict[str, Any]], field_name: str) -> list[float]:
    return [float(row[field_name]) for row in seed_records if row[field_name] is not None]


def _paired_rows(
    seed_records: list[dict[str, Any]],
    left_status_field: str,
    right_status_field: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in seed_records
        if row[left_status_field] == "ok" and row[right_status_field] == "ok"
    ]


def _count_pairwise_outcomes(
    seed_records: list[dict[str, Any]],
    *,
    metric_name: str,
    left_value_field: str,
    right_value_field: str,
    left_status_field: str,
    right_status_field: str,
) -> tuple[int, int, int]:
    left_wins = 0
    right_wins = 0
    ties = 0
    for row in _paired_rows(seed_records, left_status_field, right_status_field):
        comparison = _compare_metric_values(
            metric_name,
            float(row[left_value_field]),
            float(row[right_value_field]),
        )
        if comparison == "left":
            left_wins += 1
        elif comparison == "right":
            right_wins += 1
        else:
            ties += 1
    return left_wins, right_wins, ties


def _build_study_config(
    base_spec: ToyBenchmarkSpec,
    *,
    run_id: str,
    output_layout: OutputLayout,
    seed_values: Sequence[int],
) -> dict[str, Any]:
    tuned_training = get_phase2c_tuned_pc_training(base_spec.benchmark_name)
    budget_training = _budget_diagnostic_training(tuned_training)
    return {
        "experiment_name": f"pc_diagnostics_{base_spec.benchmark_name}",
        "run_id": run_id,
        "phase": "Phase 2d",
        "benchmark_name": base_spec.benchmark_name,
        "task_name": base_spec.task_name,
        "seed_values": [int(seed) for seed in seed_values],
        "seed_semantics": {
            "data_seed": "fixed",
            "run_seed": "varies_with_seed",
            "model_init_seed": "varies_with_seed",
            "notes": "This phase mainly measures initialization stability, not dataset sampling variability.",
        },
        "data_seed_fixed": base_spec.data_seed,
        "default_pc_training": _training_dict(base_spec.pc_training, epochs=base_spec.epochs),
        "tuned_pc_training": _training_dict(tuned_training, epochs=base_spec.epochs),
        "tuned_pc_budget2x_training": _training_dict(budget_training, epochs=base_spec.epochs),
        "mlp_training": {
            "epochs": base_spec.epochs,
            "eta_w": base_spec.mlp_training.eta_w,
            "eta_b": base_spec.mlp_training.eta_b,
        },
        "variant_groups": dict(PHASE2D_VARIANT_GROUPS),
        "budget_diagnostic_variant": PHASE2D_BUDGET_DIAGNOSTIC_VARIANT,
        "output_layout": output_layout,
    }


def _build_diagnostic_summary(
    *,
    base_spec: ToyBenchmarkSpec,
    run_id: str,
    seed_values: Sequence[int],
    seed_records: list[dict[str, Any]],
    epoch_summary_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    default_final_values = _variant_value_list(seed_records, "default_pc_primary_metric_value")
    tuned_final_values = _variant_value_list(seed_records, "tuned_pc_primary_metric_value")
    mlp_final_values = _variant_value_list(seed_records, "mlp_primary_metric_value")
    budget_final_values = _variant_value_list(
        seed_records,
        "tuned_pc_budget2x_primary_metric_value",
    )

    budget_vs_tuned_deltas = _variant_value_list(
        seed_records,
        "primary_metric_delta_budget2x_minus_tuned_pc",
    )
    default_best_epochs = _variant_value_list(seed_records, "default_pc_best_epoch")
    tuned_best_epochs = _variant_value_list(seed_records, "tuned_pc_best_epoch")
    mlp_best_epochs = _variant_value_list(seed_records, "mlp_best_epoch")
    default_final_minus_best = _variant_value_list(
        seed_records,
        "default_pc_final_minus_best_metric",
    )
    tuned_final_minus_best = _variant_value_list(
        seed_records,
        "tuned_pc_final_minus_best_metric",
    )
    mlp_final_minus_best = _variant_value_list(seed_records, "mlp_final_minus_best_metric")

    tuned_better_than_default, default_better_than_tuned, tuned_default_ties = _count_pairwise_outcomes(
        seed_records,
        metric_name=base_spec.primary_metric_name,
        left_value_field="tuned_pc_primary_metric_value",
        right_value_field="default_pc_primary_metric_value",
        left_status_field="tuned_pc_status",
        right_status_field="default_pc_status",
    )
    tuned_better_than_mlp, mlp_better_than_tuned, tuned_mlp_ties = _count_pairwise_outcomes(
        seed_records,
        metric_name=base_spec.primary_metric_name,
        left_value_field="tuned_pc_primary_metric_value",
        right_value_field="mlp_primary_metric_value",
        left_status_field="tuned_pc_status",
        right_status_field="mlp_status",
    )
    budget_beats_tuned, tuned_beats_budget, budget_tuned_ties = _count_pairwise_outcomes(
        seed_records,
        metric_name=base_spec.primary_metric_name,
        left_value_field="tuned_pc_budget2x_primary_metric_value",
        right_value_field="tuned_pc_primary_metric_value",
        left_status_field="tuned_pc_budget2x_status",
        right_status_field="tuned_pc_status",
    )

    default_pre_corr, default_pre_count = _correlation_from_epoch_mean_curve(
        epoch_summary_rows,
        variant="default_pc",
        energy_field="pre_update_energy_mean",
    )
    default_post_corr, default_post_count = _correlation_from_epoch_mean_curve(
        epoch_summary_rows,
        variant="default_pc",
        energy_field="post_update_energy_mean",
    )
    tuned_pre_corr, tuned_pre_count = _correlation_from_epoch_mean_curve(
        epoch_summary_rows,
        variant="tuned_pc",
        energy_field="pre_update_energy_mean",
    )
    tuned_post_corr, tuned_post_count = _correlation_from_epoch_mean_curve(
        epoch_summary_rows,
        variant="tuned_pc",
        energy_field="post_update_energy_mean",
    )

    tuned_mean = _mean_or_none(tuned_final_values)
    budget_mean = _mean_or_none(budget_final_values)
    tuned_training = get_phase2c_tuned_pc_training(base_spec.benchmark_name)
    budget_training = _budget_diagnostic_training(tuned_training)

    return {
        "experiment_name": f"pc_diagnostics_{base_spec.benchmark_name}",
        "run_id": run_id,
        "phase": "Phase 2d",
        "benchmark_name": base_spec.benchmark_name,
        "task_name": base_spec.task_name,
        "primary_metric_name": base_spec.primary_metric_name,
        "primary_metric_higher_is_better": base_spec.primary_metric_higher_is_better,
        "seed_values": [int(seed) for seed in seed_values],
        "planned_seed_count": len(seed_values),
        "variant_groups": dict(PHASE2D_VARIANT_GROUPS),
        "budget_diagnostic_variant": PHASE2D_BUDGET_DIAGNOSTIC_VARIANT,
        "default_pc_success_count": sum(1 for row in seed_records if row["default_pc_status"] == "ok"),
        "tuned_pc_success_count": sum(1 for row in seed_records if row["tuned_pc_status"] == "ok"),
        "mlp_success_count": sum(1 for row in seed_records if row["mlp_status"] == "ok"),
        "budget2x_success_count": sum(
            1 for row in seed_records if row["tuned_pc_budget2x_status"] == "ok"
        ),
        "paired_tuned_vs_default_count": len(
            _paired_rows(seed_records, "tuned_pc_status", "default_pc_status")
        ),
        "paired_tuned_vs_mlp_count": len(_paired_rows(seed_records, "tuned_pc_status", "mlp_status")),
        "paired_budget2x_vs_tuned_count": len(
            _paired_rows(seed_records, "tuned_pc_budget2x_status", "tuned_pc_status")
        ),
        "default_pc_final_metric_mean": _mean_or_none(default_final_values),
        "default_pc_final_metric_std": _std_or_none(default_final_values),
        "tuned_pc_final_metric_mean": tuned_mean,
        "tuned_pc_final_metric_std": _std_or_none(tuned_final_values),
        "mlp_final_metric_mean": _mean_or_none(mlp_final_values),
        "mlp_final_metric_std": _std_or_none(mlp_final_values),
        "tuned_pc_better_than_default_pc_seed_count": tuned_better_than_default,
        "default_pc_better_than_tuned_pc_seed_count": default_better_than_tuned,
        "tuned_pc_vs_default_pc_tie_seed_count": tuned_default_ties,
        "mlp_better_than_tuned_pc_seed_count": mlp_better_than_tuned,
        "tuned_pc_better_than_mlp_seed_count": tuned_better_than_mlp,
        "tuned_pc_vs_mlp_tie_seed_count": tuned_mlp_ties,
        "default_pc_best_epoch_mean": _mean_or_none(default_best_epochs),
        "default_pc_best_epoch_std": _std_or_none(default_best_epochs),
        "tuned_pc_best_epoch_mean": _mean_or_none(tuned_best_epochs),
        "tuned_pc_best_epoch_std": _std_or_none(tuned_best_epochs),
        "mlp_best_epoch_mean": _mean_or_none(mlp_best_epochs),
        "mlp_best_epoch_std": _std_or_none(mlp_best_epochs),
        "default_pc_final_minus_best_metric_mean": _mean_or_none(default_final_minus_best),
        "default_pc_final_minus_best_metric_std": _std_or_none(default_final_minus_best),
        "tuned_pc_final_minus_best_metric_mean": _mean_or_none(tuned_final_minus_best),
        "tuned_pc_final_minus_best_metric_std": _std_or_none(tuned_final_minus_best),
        "mlp_final_minus_best_metric_mean": _mean_or_none(mlp_final_minus_best),
        "mlp_final_minus_best_metric_std": _std_or_none(mlp_final_minus_best),
        "budget_check_train_steps_default": int(tuned_training.train_steps),
        "budget_check_train_steps_double": int(budget_training.train_steps),
        "budget2x_final_metric_mean": budget_mean,
        "budget2x_final_metric_std": _std_or_none(budget_final_values),
        "budget2x_delta_vs_tuned_mean": _mean_or_none(budget_vs_tuned_deltas),
        "budget2x_delta_vs_tuned_std": _std_or_none(budget_vs_tuned_deltas),
        "budget2x_beats_tuned_seed_count": budget_beats_tuned,
        "tuned_pc_beats_budget2x_seed_count": tuned_beats_budget,
        "budget2x_vs_tuned_tie_seed_count": budget_tuned_ties,
        "budget2x_mean_beats_tuned": (
            None
            if budget_mean is None or tuned_mean is None
            else _metric_value_is_better(base_spec.primary_metric_name, budget_mean, tuned_mean)
        ),
        "budget_diagnostic_note": (
            "tuned_pc_budget2x is only a budget diagnostic branch, not a new main comparison model."
        ),
        "default_pc_pre_update_energy_metric_correlation": default_pre_corr,
        "default_pc_pre_update_energy_metric_correlation_sample_count": default_pre_count,
        "default_pc_post_update_energy_metric_correlation": default_post_corr,
        "default_pc_post_update_energy_metric_correlation_sample_count": default_post_count,
        "tuned_pc_pre_update_energy_metric_correlation": tuned_pre_corr,
        "tuned_pc_pre_update_energy_metric_correlation_sample_count": tuned_pre_count,
        "tuned_pc_post_update_energy_metric_correlation": tuned_post_corr,
        "tuned_pc_post_update_energy_metric_correlation_sample_count": tuned_post_count,
        "tie_rtol": TIE_RTOL,
        "tie_atol": TIE_ATOL,
        "notes": {
            "seed_semantics": "data_seed stays fixed while run_seed and model_init_seed vary together.",
            "interpretation": "This phase mainly measures initialization stability rather than dataset sampling variability.",
            "final_minus_best_metric": (
                "For the current regression tasks with MSE, final_minus_best_metric = "
                "final_metric - best_metric, so values should be non-negative and closer to zero "
                "mean less late-epoch degradation from the best epoch."
            ),
            "correlation_definition": (
                "Pearson correlations are computed on the aggregated epoch-level mean curves for "
                "each benchmark and PC variant, not on pooled seed x epoch rows."
            ),
        },
    }


def _plot_diagnostics_summary(
    run_dir: Path,
    base_spec: ToyBenchmarkSpec,
    epoch_summary_rows: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "plot_summary=True requires matplotlib to be installed in the current environment."
        ) from exc

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    variant_to_rows: dict[str, list[dict[str, Any]]] = {variant: [] for variant in PHASE2D_VARIANT_ORDER}
    for row in epoch_summary_rows:
        variant_to_rows[str(row["variant"])].append(row)

    figure = plt.figure(figsize=(10, 4))
    axis = figure.add_subplot(1, 1, 1)
    for variant in PHASE2D_VARIANT_ORDER:
        rows = variant_to_rows.get(variant, [])
        if not rows:
            continue
        axis.plot(
            [int(row["epoch"]) for row in rows],
            [float(row["primary_metric_mean"]) for row in rows],
            marker="o",
            label=variant,
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel(base_spec.primary_metric_name)
    axis.set_title(f"{base_spec.benchmark_name} primary metric")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots_dir / "primary_metric_by_epoch_mean.png")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    pc_variants = ("default_pc", "tuned_pc", PHASE2D_BUDGET_DIAGNOSTIC_VARIANT)
    for variant in pc_variants:
        rows = variant_to_rows.get(variant, [])
        if not rows:
            continue
        epochs = [int(row["epoch"]) for row in rows]
        pre_values = [
            np.nan if row["pre_update_energy_mean"] is None else float(row["pre_update_energy_mean"])
            for row in rows
        ]
        post_values = [
            np.nan if row["post_update_energy_mean"] is None else float(row["post_update_energy_mean"])
            for row in rows
        ]
        axes[0].plot(epochs, pre_values, marker="o", label=variant)
        axes[1].plot(epochs, post_values, marker="o", label=variant)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Pre-update energy")
    axes[0].set_title("PC pre-update energy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Post-update energy")
    axes[1].set_title("PC post-update energy")
    axes[0].legend()
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(plots_dir / "pc_energy_by_epoch_mean.png")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for variant in ("default_pc", "tuned_pc"):
        rows = variant_to_rows.get(variant, [])
        if not rows:
            continue
        pre_x = [
            float(row["pre_update_energy_mean"])
            for row in rows
            if row["pre_update_energy_mean"] is not None and row["primary_metric_mean"] is not None
        ]
        pre_y = [
            float(row["primary_metric_mean"])
            for row in rows
            if row["pre_update_energy_mean"] is not None and row["primary_metric_mean"] is not None
        ]
        post_x = [
            float(row["post_update_energy_mean"])
            for row in rows
            if row["post_update_energy_mean"] is not None and row["primary_metric_mean"] is not None
        ]
        post_y = [
            float(row["primary_metric_mean"])
            for row in rows
            if row["post_update_energy_mean"] is not None and row["primary_metric_mean"] is not None
        ]
        axes[0].scatter(pre_x, pre_y, label=variant)
        axes[1].scatter(post_x, post_y, label=variant)
    axes[0].set_xlabel("Pre-update energy mean")
    axes[0].set_ylabel(base_spec.primary_metric_name)
    axes[0].set_title("Pre-update energy vs metric")
    axes[1].set_xlabel("Post-update energy mean")
    axes[1].set_ylabel(base_spec.primary_metric_name)
    axes[1].set_title("Post-update energy vs metric")
    axes[0].legend()
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(plots_dir / "energy_vs_metric_scatter.png")
    plt.close(figure)


def run_pc_diagnostics_study(
    benchmark_name: str,
    *,
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    output_layout: OutputLayout = "single_dir",
    plot_energy: bool = False,
    plot_summary: bool = False,
    seed_values: Sequence[int] | None = None,
) -> PCDiagnosticsRunResult:
    """Run the fixed Phase 2d diagnostic study for one regression benchmark."""
    if benchmark_name not in PHASE2D_BENCHMARK_NAMES:
        raise ValueError(
            f"Phase 2d currently supports only {PHASE2D_BENCHMARK_NAMES}, got '{benchmark_name}'."
        )

    base_spec = get_benchmark_spec(benchmark_name)
    resolved_run_id = _resolve_run_id(run_id)
    resolved_seed_values = (
        list(seed_values)
        if seed_values is not None
        else list(default_seed_values_for_benchmark(benchmark_name))
    )
    run_dir = _prepare_run_dir(
        _resolve_diagnostics_root(output_root, benchmark_name, resolved_run_id, output_layout)
    )
    study_config = _build_study_config(
        base_spec,
        run_id=resolved_run_id,
        output_layout=output_layout,
        seed_values=resolved_seed_values,
    )
    _write_json(run_dir / "study_config.json", study_config)

    x, y = base_spec.make_data()
    tuned_training = get_phase2c_tuned_pc_training(benchmark_name)
    budget_training = _budget_diagnostic_training(tuned_training)

    seed_records: list[dict[str, Any]] = []
    epoch_records: list[dict[str, Any]] = []

    for seed_index, seed in enumerate(resolved_seed_values):
        seed_dir = run_dir / "seeds" / f"seed_{int(seed):04d}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        default_spec = _seeded_spec(base_spec, seed=int(seed))
        tuned_spec = _seeded_spec(base_spec, seed=int(seed), pc_training=tuned_training)
        budget_spec = _seeded_spec(base_spec, seed=int(seed), pc_training=budget_training)
        mlp_spec = _seeded_spec(base_spec, seed=int(seed))

        row: dict[str, Any] = {
            "seed_index": seed_index,
            "run_seed": int(seed),
            "data_seed": base_spec.data_seed,
            "model_init_seed": int(seed),
            "default_pc_status": "failed",
            "default_pc_failure_reason": "",
            "default_pc_primary_metric_value": None,
            "default_pc_best_epoch": None,
            "default_pc_final_minus_best_metric": None,
            "default_pc_final_pre_update_energy": None,
            "default_pc_summary_path": "",
            "tuned_pc_status": "failed",
            "tuned_pc_failure_reason": "",
            "tuned_pc_primary_metric_value": None,
            "tuned_pc_best_epoch": None,
            "tuned_pc_final_minus_best_metric": None,
            "tuned_pc_final_pre_update_energy": None,
            "tuned_pc_summary_path": "",
            "mlp_status": "failed",
            "mlp_failure_reason": "",
            "mlp_primary_metric_value": None,
            "mlp_best_epoch": None,
            "mlp_final_minus_best_metric": None,
            "mlp_final_loss": None,
            "mlp_summary_path": "",
            "tuned_pc_budget2x_status": "failed",
            "tuned_pc_budget2x_failure_reason": "",
            "tuned_pc_budget2x_primary_metric_value": None,
            "tuned_pc_budget2x_best_epoch": None,
            "tuned_pc_budget2x_final_minus_best_metric": None,
            "tuned_pc_budget2x_final_pre_update_energy": None,
            "tuned_pc_budget2x_summary_path": "",
            "primary_metric_delta_tuned_pc_minus_default_pc": None,
            "primary_metric_delta_mlp_minus_tuned_pc": None,
            "primary_metric_delta_budget2x_minus_tuned_pc": None,
            "tuned_pc_beats_default_pc": None,
            "tuned_pc_beats_mlp": None,
            "budget2x_beats_tuned": None,
        }

        try:
            default_result = run_pc_benchmark(
                default_spec,
                output_root=seed_dir,
                run_id=resolved_run_id,
                plot_energy=plot_energy,
                output_layout="single_dir",
                experiment_name="default_pc",
                x=x,
                y=y,
            )
            best_epoch, best_metric, final_metric = _best_epoch_and_metric(
                default_result.epoch_metrics,
                base_spec.primary_metric_name,
                higher_is_better=base_spec.primary_metric_higher_is_better,
            )
            summary_path = _summary_path_for_seed_variant(seed, "default_pc")
            row["default_pc_status"] = "ok"
            row["default_pc_primary_metric_value"] = float(default_result.summary["primary_metric_value"])
            row["default_pc_best_epoch"] = best_epoch
            row["default_pc_final_minus_best_metric"] = final_metric - best_metric
            row["default_pc_final_pre_update_energy"] = float(
                default_result.summary["final_pre_update_energy"]
            )
            row["default_pc_summary_path"] = summary_path
            _append_pc_epoch_records(
                epoch_records,
                seed_index=seed_index,
                seed=seed,
                variant="default_pc",
                summary_path=summary_path,
                result_rows=default_result.epoch_metrics,
                primary_metric_name=base_spec.primary_metric_name,
            )
        except Exception as exc:
            _clean_failed_variant_dir(seed_dir / "default_pc")
            row["default_pc_failure_reason"] = _failure_reason(exc)

        try:
            tuned_result = run_pc_benchmark(
                tuned_spec,
                output_root=seed_dir,
                run_id=resolved_run_id,
                plot_energy=plot_energy,
                output_layout="single_dir",
                experiment_name="tuned_pc",
                x=x,
                y=y,
            )
            best_epoch, best_metric, final_metric = _best_epoch_and_metric(
                tuned_result.epoch_metrics,
                base_spec.primary_metric_name,
                higher_is_better=base_spec.primary_metric_higher_is_better,
            )
            summary_path = _summary_path_for_seed_variant(seed, "tuned_pc")
            row["tuned_pc_status"] = "ok"
            row["tuned_pc_primary_metric_value"] = float(tuned_result.summary["primary_metric_value"])
            row["tuned_pc_best_epoch"] = best_epoch
            row["tuned_pc_final_minus_best_metric"] = final_metric - best_metric
            row["tuned_pc_final_pre_update_energy"] = float(
                tuned_result.summary["final_pre_update_energy"]
            )
            row["tuned_pc_summary_path"] = summary_path
            _append_pc_epoch_records(
                epoch_records,
                seed_index=seed_index,
                seed=seed,
                variant="tuned_pc",
                summary_path=summary_path,
                result_rows=tuned_result.epoch_metrics,
                primary_metric_name=base_spec.primary_metric_name,
            )
        except Exception as exc:
            _clean_failed_variant_dir(seed_dir / "tuned_pc")
            row["tuned_pc_failure_reason"] = _failure_reason(exc)

        try:
            budget_result = run_pc_benchmark(
                budget_spec,
                output_root=seed_dir,
                run_id=resolved_run_id,
                plot_energy=plot_energy,
                output_layout="single_dir",
                experiment_name=PHASE2D_BUDGET_DIAGNOSTIC_VARIANT,
                x=x,
                y=y,
            )
            best_epoch, best_metric, final_metric = _best_epoch_and_metric(
                budget_result.epoch_metrics,
                base_spec.primary_metric_name,
                higher_is_better=base_spec.primary_metric_higher_is_better,
            )
            summary_path = _summary_path_for_seed_variant(seed, PHASE2D_BUDGET_DIAGNOSTIC_VARIANT)
            row["tuned_pc_budget2x_status"] = "ok"
            row["tuned_pc_budget2x_primary_metric_value"] = float(
                budget_result.summary["primary_metric_value"]
            )
            row["tuned_pc_budget2x_best_epoch"] = best_epoch
            row["tuned_pc_budget2x_final_minus_best_metric"] = final_metric - best_metric
            row["tuned_pc_budget2x_final_pre_update_energy"] = float(
                budget_result.summary["final_pre_update_energy"]
            )
            row["tuned_pc_budget2x_summary_path"] = summary_path
            _append_pc_epoch_records(
                epoch_records,
                seed_index=seed_index,
                seed=seed,
                variant=PHASE2D_BUDGET_DIAGNOSTIC_VARIANT,
                summary_path=summary_path,
                result_rows=budget_result.epoch_metrics,
                primary_metric_name=base_spec.primary_metric_name,
            )
        except Exception as exc:
            _clean_failed_variant_dir(seed_dir / PHASE2D_BUDGET_DIAGNOSTIC_VARIANT)
            row["tuned_pc_budget2x_failure_reason"] = _failure_reason(exc)

        try:
            _, mlp_epoch_metrics, mlp_summary = _run_mlp_variant(
                mlp_spec,
                variant_dir=seed_dir / "mlp",
                run_id=resolved_run_id,
                output_layout="single_dir",
                x=x,
                y=y,
            )
            best_epoch, best_metric, final_metric = _best_epoch_and_metric(
                mlp_epoch_metrics,
                base_spec.primary_metric_name,
                higher_is_better=base_spec.primary_metric_higher_is_better,
            )
            summary_path = _summary_path_for_seed_variant(seed, "mlp")
            row["mlp_status"] = "ok"
            row["mlp_primary_metric_value"] = float(mlp_summary["primary_metric_value"])
            row["mlp_best_epoch"] = best_epoch
            row["mlp_final_minus_best_metric"] = final_metric - best_metric
            row["mlp_final_loss"] = float(mlp_summary["final_loss"])
            row["mlp_summary_path"] = summary_path
            _append_mlp_epoch_records(
                epoch_records,
                seed_index=seed_index,
                seed=seed,
                summary_path=summary_path,
                result_rows=mlp_epoch_metrics,
                primary_metric_name=base_spec.primary_metric_name,
            )
        except Exception as exc:
            _clean_failed_variant_dir(seed_dir / "mlp")
            row["mlp_failure_reason"] = _failure_reason(exc)

        if row["tuned_pc_status"] == "ok" and row["default_pc_status"] == "ok":
            tuned_value = float(row["tuned_pc_primary_metric_value"])
            default_value = float(row["default_pc_primary_metric_value"])
            row["primary_metric_delta_tuned_pc_minus_default_pc"] = tuned_value - default_value
            row["tuned_pc_beats_default_pc"] = _metric_value_is_better(
                base_spec.primary_metric_name,
                tuned_value,
                default_value,
            )

        if row["tuned_pc_status"] == "ok" and row["mlp_status"] == "ok":
            tuned_value = float(row["tuned_pc_primary_metric_value"])
            mlp_value = float(row["mlp_primary_metric_value"])
            row["primary_metric_delta_mlp_minus_tuned_pc"] = mlp_value - tuned_value
            row["tuned_pc_beats_mlp"] = _metric_value_is_better(
                base_spec.primary_metric_name,
                tuned_value,
                mlp_value,
            )

        if row["tuned_pc_budget2x_status"] == "ok" and row["tuned_pc_status"] == "ok":
            budget_value = float(row["tuned_pc_budget2x_primary_metric_value"])
            tuned_value = float(row["tuned_pc_primary_metric_value"])
            row["primary_metric_delta_budget2x_minus_tuned_pc"] = budget_value - tuned_value
            row["budget2x_beats_tuned"] = _metric_value_is_better(
                base_spec.primary_metric_name,
                budget_value,
                tuned_value,
            )

        seed_records.append(row)

    epoch_summary_rows = _build_epoch_summary_rows(epoch_records)
    diagnostic_summary = _build_diagnostic_summary(
        base_spec=base_spec,
        run_id=resolved_run_id,
        seed_values=resolved_seed_values,
        seed_records=seed_records,
        epoch_summary_rows=epoch_summary_rows,
    )

    _write_seed_records(run_dir / "seed_records.csv", seed_records)
    _write_epoch_records(run_dir / "epoch_records.csv", epoch_records)
    _write_epoch_summary(run_dir / "epoch_summary.csv", epoch_summary_rows)
    _write_json(run_dir / "diagnostic_summary.json", diagnostic_summary)

    if plot_summary:
        _plot_diagnostics_summary(run_dir, base_spec, epoch_summary_rows)

    return PCDiagnosticsRunResult(
        run_dir=run_dir,
        study_config=study_config,
        seed_records=seed_records,
        epoch_records=epoch_records,
        epoch_summary=epoch_summary_rows,
        diagnostic_summary=diagnostic_summary,
    )
