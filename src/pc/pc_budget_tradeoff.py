from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .benchmark_specs import (
    MLPTrainingSpec,
    PCTrainingSpec,
    ToyBenchmarkSpec,
    get_benchmark_spec,
    run_pc_benchmark,
)
from .experiment import OutputLayout
from .metrics import metric_higher_is_better
from .pc_multiseed import (
    MLPSelection,
    PHASE2_MLP_SOURCE_DEFAULT,
    PHASE2_MLP_SOURCE_PHASE2G,
    PHASE2_MLP_SOURCE_PHASE2G1,
    PHASE2_TUNED_SOURCE_LEGACY,
    PHASE2_TUNED_SOURCE_PHASE2F,
    PHASE2_TUNED_SOURCE_PHASE2G,
    PHASE2_TUNED_SOURCE_PHASE2G1,
    TunedPCSelection,
    default_seed_values_for_benchmark,
    resolve_mlp_selection,
    resolve_tuned_pc_selection,
)
from .utils import set_seed

PHASE2E_BENCHMARK_NAMES = (
    "toy_regression",
    "toy_sine_regression",
)
PHASE2E_BUDGET_MULTIPLIERS = (1, 2, 4)
PHASE2E_VARIANT_GROUPS = {
    "tuned_pc_1x": "tuned_pc_budget",
    "tuned_pc_2x": "tuned_pc_budget",
    "tuned_pc_4x": "tuned_pc_budget",
    "mlp": "mlp_reference",
}
PHASE2E_VARIANT_ORDER = (
    "tuned_pc_1x",
    "tuned_pc_2x",
    "tuned_pc_4x",
    "mlp",
)
PHASE2E_REFERENCE_VARIANT = "mlp"
TIE_RTOL = 1.0e-12
TIE_ATOL = 1.0e-12


@dataclass
class PCBudgetTradeoffRunResult:
    """Materialized outputs of one Phase 2e budget tradeoff study."""

    run_dir: Path
    study_config: dict[str, Any]
    seed_budget_records: list[dict[str, Any]]
    budget_summary: list[dict[str, Any]]
    aggregate_summary: dict[str, Any]


def _resolve_run_id(run_id: str | None) -> str:
    if run_id is not None:
        return run_id
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _tradeoff_experiment_name(
    benchmark_name: str,
    *,
    tuned_source: str,
    mlp_source: str,
) -> str:
    if (
        tuned_source == PHASE2_TUNED_SOURCE_LEGACY
        and mlp_source == PHASE2_MLP_SOURCE_DEFAULT
    ):
        return f"pc_budget_tradeoff_{benchmark_name}"
    if (
        tuned_source == PHASE2_TUNED_SOURCE_PHASE2F
        and mlp_source == PHASE2_MLP_SOURCE_DEFAULT
    ):
        return f"pc_budget_tradeoff_phase2f_{benchmark_name}"
    if (
        tuned_source == PHASE2_TUNED_SOURCE_PHASE2G
        and mlp_source == PHASE2_MLP_SOURCE_PHASE2G
    ):
        return f"pc_budget_tradeoff_phase2g_{benchmark_name}"
    if (
        tuned_source == PHASE2_TUNED_SOURCE_PHASE2G1
        and mlp_source == PHASE2_MLP_SOURCE_PHASE2G1
    ):
        return f"pc_budget_tradeoff_phase2g1_{benchmark_name}"
    return f"pc_budget_tradeoff_custom_{benchmark_name}"


def _tradeoff_phase_label(*, tuned_source: str, mlp_source: str) -> str:
    if (
        tuned_source == PHASE2_TUNED_SOURCE_LEGACY
        and mlp_source == PHASE2_MLP_SOURCE_DEFAULT
    ):
        return "Phase 2e"
    if (
        tuned_source == PHASE2_TUNED_SOURCE_PHASE2F
        and mlp_source == PHASE2_MLP_SOURCE_DEFAULT
    ):
        return "Phase 2e-refresh"
    if (
        tuned_source == PHASE2_TUNED_SOURCE_PHASE2G
        and mlp_source == PHASE2_MLP_SOURCE_PHASE2G
    ):
        return "Phase 2g-budget-refresh"
    if (
        tuned_source == PHASE2_TUNED_SOURCE_PHASE2G1
        and mlp_source == PHASE2_MLP_SOURCE_PHASE2G1
    ):
        return "Phase 2g.1-budget-refresh"
    return "Phase 2e-refresh-custom"


def _resolve_tradeoff_root(
    output_root: str | Path,
    benchmark_name: str,
    run_id: str,
    output_layout: OutputLayout,
    tuned_source: str,
    mlp_source: str,
) -> Path:
    experiment_name = _tradeoff_experiment_name(
        benchmark_name,
        tuned_source=tuned_source,
        mlp_source=mlp_source,
    )
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


def _write_seed_budget_records(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "seed_index",
        "run_seed",
        "data_seed",
        "model_init_seed",
        "variant",
        "variant_group",
        "is_reference_variant",
        "budget_multiplier",
        "train_steps",
        "eval_steps",
        "status",
        "failure_reason",
        "primary_metric_name",
        "primary_metric_value",
        "final_pre_update_energy",
        "final_post_update_energy",
        "final_loss",
        "summary_path",
        "primary_metric_delta_vs_tuned_pc_1x",
        "primary_metric_delta_tuned_pc_minus_mlp",
        "variant_beats_tuned_pc_1x",
        "variant_beats_mlp",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_budget_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "variant",
        "variant_group",
        "budget_multiplier",
        "train_steps",
        "seed_count",
        "primary_metric_mean",
        "primary_metric_std",
        "final_pre_update_energy_mean",
        "final_pre_update_energy_std",
        "final_post_update_energy_mean",
        "final_post_update_energy_std",
        "final_loss_mean",
        "final_loss_std",
        "primary_metric_delta_vs_tuned_pc_1x_mean",
        "primary_metric_delta_vs_tuned_pc_1x_std",
        "primary_metric_delta_tuned_pc_minus_mlp_mean",
        "primary_metric_delta_tuned_pc_minus_mlp_std",
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


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _std_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=0))


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


def _headline_test_comparison(
    *,
    metric_name: str,
    pc_value: float | None,
    mlp_value: float | None,
    comparison_target: str,
) -> dict[str, Any]:
    if pc_value is None or mlp_value is None:
        return {
            "headline_test_comparison_target": comparison_target,
            "headline_test_comparison_split": "test",
            "headline_test_winner": None,
            "headline_test_winner_reason": None,
            "headline_test_pc_metric_mean": pc_value,
            "headline_test_mlp_metric_mean": mlp_value,
            "headline_test_metric_difference_mlp_minus_pc": None,
            "headline_test_pc_beats_mlp": None,
        }

    comparison = _compare_metric_values(metric_name, pc_value, mlp_value)
    if comparison == "left":
        winner = "pc"
    elif comparison == "right":
        winner = "mlp"
    else:
        winner = "tie"

    if comparison == "tie":
        reason = "metrics_equal_within_tolerance"
    elif metric_higher_is_better(metric_name):
        reason = f"higher_is_better: {winner}_metric_mean is larger on held-out test"
    else:
        reason = f"lower_is_better: {winner}_metric_mean is smaller on held-out test"

    return {
        "headline_test_comparison_target": comparison_target,
        "headline_test_comparison_split": "test",
        "headline_test_winner": winner,
        "headline_test_winner_reason": reason,
        "headline_test_pc_metric_mean": pc_value,
        "headline_test_mlp_metric_mean": mlp_value,
        "headline_test_metric_difference_mlp_minus_pc": mlp_value - pc_value,
        "headline_test_pc_beats_mlp": winner == "pc",
    }


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


def _resolve_variant_group(variant: str) -> str:
    try:
        return PHASE2E_VARIANT_GROUPS[variant]
    except KeyError as exc:
        raise ValueError(f"Unsupported Phase 2e variant '{variant}'.") from exc


def _budget_multiplier_label(multiplier: int | str) -> str:
    if isinstance(multiplier, str):
        return multiplier
    return f"{int(multiplier)}x"


def _variant_name_for_multiplier(multiplier: int) -> str:
    return f"tuned_pc_{int(multiplier)}x"


def _seeded_spec(
    base_spec: ToyBenchmarkSpec,
    *,
    seed: int,
    pc_training: PCTrainingSpec | None = None,
    mlp_training: MLPTrainingSpec | None = None,
    epochs: int | None = None,
) -> ToyBenchmarkSpec:
    return replace(
        base_spec,
        run_seed=seed,
        model_init_seed=seed,
        data_seed=base_spec.data_seed,
        pc_training=base_spec.pc_training if pc_training is None else pc_training,
        mlp_training=base_spec.mlp_training if mlp_training is None else mlp_training,
        epochs=base_spec.epochs if epochs is None else epochs,
    )


def _tuned_budget_training(
    base_training: PCTrainingSpec,
    *,
    multiplier: int,
) -> PCTrainingSpec:
    return PCTrainingSpec(
        eta_x=base_training.eta_x,
        eta_w=base_training.eta_w,
        eta_b=base_training.eta_b,
        train_steps=int(base_training.train_steps * multiplier),
        eval_steps=int(base_training.eval_steps * multiplier),
        state_init=base_training.state_init,
    )


def _budget_levels(base_training: PCTrainingSpec) -> dict[str, dict[str, Any]]:
    levels: dict[str, dict[str, Any]] = {}
    for multiplier in PHASE2E_BUDGET_MULTIPLIERS:
        label = _budget_multiplier_label(multiplier)
        training = _tuned_budget_training(base_training, multiplier=multiplier)
        levels[label] = {
            "budget_multiplier": label,
            "train_steps": int(training.train_steps),
            "eval_steps": int(training.eval_steps),
        }
    return levels


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


def _run_mlp_variant(
    spec: ToyBenchmarkSpec,
    *,
    variant_dir: Path,
    run_id: str,
    output_layout: OutputLayout,
    split,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    variant_dir.mkdir(parents=True, exist_ok=True)
    set_seed(spec.run_seed)
    model = spec.build_mlp_model()
    config_payload = _mlp_config_payload(spec, run_id, variant_dir.parent, output_layout, split)
    _write_json(variant_dir / "config.json", config_payload)

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
        "phase": "Phase 2e",
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


def _summary_path_for_seed_variant(seed: int, variant: str) -> str:
    return (Path("seeds") / f"seed_{int(seed):04d}" / variant / "summary.json").as_posix()


def _variant_records(
    seed_budget_records: list[dict[str, Any]],
    *,
    variant: str,
) -> list[dict[str, Any]]:
    return [row for row in seed_budget_records if row["variant"] == variant and row["status"] == "ok"]


def _metric_improvement_amount(metric_name: str, old_value: float, new_value: float) -> float:
    if metric_higher_is_better(metric_name):
        return new_value - old_value
    return old_value - new_value


def _combined_config_source(*, tuned_source: str, mlp_source: str) -> str:
    if tuned_source == mlp_source:
        return tuned_source
    return "mixed_config_sources"


def _build_study_config(
    base_spec: ToyBenchmarkSpec,
    *,
    run_id: str,
    output_layout: OutputLayout,
    seed_values: Sequence[int],
    tuned_selection: TunedPCSelection,
    mlp_selection: MLPSelection,
    split,
) -> dict[str, Any]:
    tuned_training = tuned_selection.pc_training
    experiment_name = _tradeoff_experiment_name(
        base_spec.benchmark_name,
        tuned_source=tuned_selection.source,
        mlp_source=mlp_selection.source,
    )
    return {
        "experiment_name": experiment_name,
        "run_id": run_id,
        "phase": _tradeoff_phase_label(
            tuned_source=tuned_selection.source,
            mlp_source=mlp_selection.source,
        ),
        "benchmark_name": base_spec.benchmark_name,
        "task_name": base_spec.task_name,
        "config_source": _combined_config_source(
            tuned_source=tuned_selection.source,
            mlp_source=mlp_selection.source,
        ),
        "selected_by_metric_source": "val_metric",
        "final_report_metric_source": "test_metric",
        "seed_values": [int(seed) for seed in seed_values],
        "seed_semantics": {
            "data_seed": "fixed",
            "run_seed": "varies_with_seed",
            "model_init_seed": "varies_with_seed",
            "notes": "This phase mainly measures initialization stability rather than dataset sampling variability.",
        },
        "data_seed_fixed": base_spec.data_seed,
        "data": base_spec.data_config(split),
        "tuned_pc_source": tuned_selection.source,
        "tuned_pc_preset_name": tuned_selection.name,
        "tuned_pc_selection_config_id": tuned_selection.selection_config_id,
        "tuned_pc_selection_run_id": tuned_selection.selection_run_id,
        "tuned_pc_selection_val_metric": tuned_selection.selection_val_metric,
        "tuned_pc_selection_test_metric": tuned_selection.selection_test_metric,
        "tuned_pc_selection_eval_metric": tuned_selection.selection_eval_metric,
        "tuned_pc_selection_artifact_path": tuned_selection.selection_artifact_path,
        "tuned_pc_base_training": _training_dict(tuned_training, epochs=tuned_selection.epochs),
        "budget_levels": _budget_levels(tuned_training),
        "mlp_source": mlp_selection.source,
        "mlp_preset_name": mlp_selection.name,
        "mlp_selection_config_id": mlp_selection.selection_config_id,
        "mlp_selection_run_id": mlp_selection.selection_run_id,
        "mlp_selection_val_metric": mlp_selection.selection_val_metric,
        "mlp_selection_test_metric": mlp_selection.selection_test_metric,
        "mlp_selection_artifact_path": mlp_selection.selection_artifact_path,
        "mlp_training": {
            "epochs": mlp_selection.epochs,
            "eta_w": mlp_selection.mlp_training.eta_w,
            "eta_b": mlp_selection.mlp_training.eta_b,
        },
        "variant_groups": dict(PHASE2E_VARIANT_GROUPS),
        "budget_definition": "budget = inference step count, not wall-clock or FLOP matching",
        "output_layout": output_layout,
    }


def _build_budget_summary_rows(
    *,
    seed_budget_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for variant in PHASE2E_VARIANT_ORDER:
        rows = _variant_records(seed_budget_records, variant=variant)
        delta_vs_base_values = [
            float(row["primary_metric_delta_vs_tuned_pc_1x"])
            for row in rows
            if row["primary_metric_delta_vs_tuned_pc_1x"] is not None
        ]
        delta_vs_mlp_values = [
            float(row["primary_metric_delta_tuned_pc_minus_mlp"])
            for row in rows
            if row["primary_metric_delta_tuned_pc_minus_mlp"] is not None
        ]
        budget_multiplier = "reference" if variant == PHASE2E_REFERENCE_VARIANT else variant.removeprefix("tuned_pc_")
        train_steps: int | None
        if variant == PHASE2E_REFERENCE_VARIANT:
            train_steps = None
        else:
            train_steps = int(rows[0]["train_steps"]) if rows else None

        summary_rows.append(
            {
                "variant": variant,
                "variant_group": _resolve_variant_group(variant),
                "budget_multiplier": budget_multiplier,
                "train_steps": train_steps,
                "seed_count": len(rows),
                "primary_metric_mean": _mean_or_none(
                    [float(row["primary_metric_value"]) for row in rows]
                ),
                "primary_metric_std": _std_or_none(
                    [float(row["primary_metric_value"]) for row in rows]
                ),
                "final_pre_update_energy_mean": _mean_or_none(
                    [float(row["final_pre_update_energy"]) for row in rows if row["final_pre_update_energy"] is not None]
                ),
                "final_pre_update_energy_std": _std_or_none(
                    [float(row["final_pre_update_energy"]) for row in rows if row["final_pre_update_energy"] is not None]
                ),
                "final_post_update_energy_mean": _mean_or_none(
                    [float(row["final_post_update_energy"]) for row in rows if row["final_post_update_energy"] is not None]
                ),
                "final_post_update_energy_std": _std_or_none(
                    [float(row["final_post_update_energy"]) for row in rows if row["final_post_update_energy"] is not None]
                ),
                "final_loss_mean": _mean_or_none(
                    [float(row["final_loss"]) for row in rows if row["final_loss"] is not None]
                ),
                "final_loss_std": _std_or_none(
                    [float(row["final_loss"]) for row in rows if row["final_loss"] is not None]
                ),
                "primary_metric_delta_vs_tuned_pc_1x_mean": _mean_or_none(delta_vs_base_values),
                "primary_metric_delta_vs_tuned_pc_1x_std": _std_or_none(delta_vs_base_values),
                "primary_metric_delta_tuned_pc_minus_mlp_mean": _mean_or_none(delta_vs_mlp_values),
                "primary_metric_delta_tuned_pc_minus_mlp_std": _std_or_none(delta_vs_mlp_values),
            }
        )
    return summary_rows


def _summary_row_for_variant(
    budget_summary_rows: list[dict[str, Any]],
    variant: str,
) -> dict[str, Any]:
    for row in budget_summary_rows:
        if row["variant"] == variant:
            return row
    raise ValueError(f"Missing budget summary row for variant '{variant}'.")


def _count_pairwise_seed_wins(
    seed_budget_records: list[dict[str, Any]],
    *,
    left_variant: str,
    right_variant: str,
    metric_name: str,
) -> int:
    left_rows = {
        int(row["run_seed"]): row
        for row in seed_budget_records
        if row["variant"] == left_variant and row["status"] == "ok"
    }
    right_rows = {
        int(row["run_seed"]): row
        for row in seed_budget_records
        if row["variant"] == right_variant and row["status"] == "ok"
    }
    shared_seeds = sorted(set(left_rows) & set(right_rows))
    wins = 0
    for seed in shared_seeds:
        if _compare_metric_values(
            metric_name,
            float(left_rows[seed]["primary_metric_value"]),
            float(right_rows[seed]["primary_metric_value"]),
        ) == "left":
            wins += 1
    return wins


def _build_aggregate_summary(
    *,
    base_spec: ToyBenchmarkSpec,
    run_id: str,
    seed_values: Sequence[int],
    seed_budget_records: list[dict[str, Any]],
    budget_summary_rows: list[dict[str, Any]],
    tuned_selection: TunedPCSelection,
    mlp_selection: MLPSelection,
) -> dict[str, Any]:
    tuned_1x = _summary_row_for_variant(budget_summary_rows, "tuned_pc_1x")
    tuned_2x = _summary_row_for_variant(budget_summary_rows, "tuned_pc_2x")
    tuned_4x = _summary_row_for_variant(budget_summary_rows, "tuned_pc_4x")
    mlp = _summary_row_for_variant(budget_summary_rows, "mlp")

    tuned_rows = [tuned_1x, tuned_2x, tuned_4x]
    valid_tuned_rows = [row for row in tuned_rows if row["primary_metric_mean"] is not None]
    best_budget_row = None
    if valid_tuned_rows:
        best_budget_row = (
            max(valid_tuned_rows, key=lambda row: float(row["primary_metric_mean"]))
            if base_spec.primary_metric_higher_is_better
            else min(valid_tuned_rows, key=lambda row: float(row["primary_metric_mean"]))
        )

    gap_1x = tuned_1x["primary_metric_delta_tuned_pc_minus_mlp_mean"]
    gap_2x = tuned_2x["primary_metric_delta_tuned_pc_minus_mlp_mean"]
    gap_4x = tuned_4x["primary_metric_delta_tuned_pc_minus_mlp_mean"]

    mean_1x = tuned_1x["primary_metric_mean"]
    mean_2x = tuned_2x["primary_metric_mean"]
    mean_4x = tuned_4x["primary_metric_mean"]

    improve_12 = None
    improve_24 = None
    if mean_1x is not None and mean_2x is not None:
        improve_12 = _metric_improvement_amount(
            base_spec.primary_metric_name,
            float(mean_1x),
            float(mean_2x),
        )
    if mean_2x is not None and mean_4x is not None:
        improve_24 = _metric_improvement_amount(
            base_spec.primary_metric_name,
            float(mean_2x),
            float(mean_4x),
        )

    evidence_of_diminishing_returns = None
    if improve_12 is not None and improve_24 is not None and mean_2x is not None and mean_4x is not None:
        budget4x_beats_2x = _metric_value_is_better(
            base_spec.primary_metric_name,
            float(mean_4x),
            float(mean_2x),
        )
        evidence_of_diminishing_returns = improve_12 > 0.0 and (
            (not budget4x_beats_2x) or (improve_12 > improve_24)
        )

    final_pre_update_energy_mean_by_budget = {
        "1x": tuned_1x["final_pre_update_energy_mean"],
        "2x": tuned_2x["final_pre_update_energy_mean"],
        "4x": tuned_4x["final_pre_update_energy_mean"],
    }
    final_post_update_energy_mean_by_budget = {
        "1x": tuned_1x["final_post_update_energy_mean"],
        "2x": tuned_2x["final_post_update_energy_mean"],
        "4x": tuned_4x["final_post_update_energy_mean"],
    }
    headline_test = _headline_test_comparison(
        metric_name=base_spec.primary_metric_name,
        pc_value=tuned_1x["primary_metric_mean"],
        mlp_value=mlp["primary_metric_mean"],
        comparison_target="selected_pc_1x_vs_selected_mlp",
    )

    return {
        "experiment_name": _tradeoff_experiment_name(
            base_spec.benchmark_name,
            tuned_source=tuned_selection.source,
            mlp_source=mlp_selection.source,
        ),
        "run_id": run_id,
        "phase": _tradeoff_phase_label(
            tuned_source=tuned_selection.source,
            mlp_source=mlp_selection.source,
        ),
        "benchmark_name": base_spec.benchmark_name,
        "task_name": base_spec.task_name,
        "primary_metric_name": base_spec.primary_metric_name,
        "primary_metric_higher_is_better": base_spec.primary_metric_higher_is_better,
        "config_source": _combined_config_source(
            tuned_source=tuned_selection.source,
            mlp_source=mlp_selection.source,
        ),
        "selected_by_metric_source": "val_metric",
        "final_report_metric_source": "test_metric",
        "seed_values": [int(seed) for seed in seed_values],
        "variant_groups": dict(PHASE2E_VARIANT_GROUPS),
        "budget_levels": _budget_levels(tuned_selection.pc_training),
        "tuned_pc_source": tuned_selection.source,
        "tuned_pc_preset_name": tuned_selection.name,
        "selected_pc_base_config": _training_dict(
            tuned_selection.pc_training,
            epochs=tuned_selection.epochs,
        ),
        "tuned_pc_selection_config_id": tuned_selection.selection_config_id,
        "tuned_pc_selection_run_id": tuned_selection.selection_run_id,
        "tuned_pc_selection_val_metric": tuned_selection.selection_val_metric,
        "tuned_pc_selection_test_metric": tuned_selection.selection_test_metric,
        "tuned_pc_selection_eval_metric": tuned_selection.selection_eval_metric,
        "tuned_pc_selection_artifact_path": tuned_selection.selection_artifact_path,
        "mlp_source": mlp_selection.source,
        "mlp_preset_name": mlp_selection.name,
        "selected_mlp_config": {
            "epochs": mlp_selection.epochs,
            "eta_w": mlp_selection.mlp_training.eta_w,
            "eta_b": mlp_selection.mlp_training.eta_b,
        },
        "mlp_selection_config_id": mlp_selection.selection_config_id,
        "mlp_selection_run_id": mlp_selection.selection_run_id,
        "mlp_selection_val_metric": mlp_selection.selection_val_metric,
        "mlp_selection_test_metric": mlp_selection.selection_test_metric,
        "mlp_selection_artifact_path": mlp_selection.selection_artifact_path,
        "budget_definition": "budget = inference step count, not wall-clock or FLOP matching",
        "selection_split": "validation",
        "final_report_split": "test",
        "planned_seed_count": len(seed_values),
        "mlp_success_count": int(mlp["seed_count"]),
        "tuned_pc_1x_success_count": int(tuned_1x["seed_count"]),
        "tuned_pc_2x_success_count": int(tuned_2x["seed_count"]),
        "tuned_pc_4x_success_count": int(tuned_4x["seed_count"]),
        "tuned_pc_1x_primary_metric_mean": tuned_1x["primary_metric_mean"],
        "tuned_pc_1x_primary_metric_std": tuned_1x["primary_metric_std"],
        "tuned_pc_2x_primary_metric_mean": tuned_2x["primary_metric_mean"],
        "tuned_pc_2x_primary_metric_std": tuned_2x["primary_metric_std"],
        "tuned_pc_4x_primary_metric_mean": tuned_4x["primary_metric_mean"],
        "tuned_pc_4x_primary_metric_std": tuned_4x["primary_metric_std"],
        "mlp_primary_metric_mean": mlp["primary_metric_mean"],
        "mlp_primary_metric_std": mlp["primary_metric_std"],
        "best_budget_variant": None if best_budget_row is None else best_budget_row["variant"],
        "best_budget_multiplier": None if best_budget_row is None else best_budget_row["budget_multiplier"],
        "best_budget_train_steps": None if best_budget_row is None else best_budget_row["train_steps"],
        "best_budget_primary_metric_mean": None if best_budget_row is None else best_budget_row["primary_metric_mean"],
        "best_budget_beats_tuned_pc_1x": (
            None
            if best_budget_row is None or best_budget_row["primary_metric_mean"] is None or tuned_1x["primary_metric_mean"] is None
            else _metric_value_is_better(
                base_spec.primary_metric_name,
                float(best_budget_row["primary_metric_mean"]),
                float(tuned_1x["primary_metric_mean"]),
            )
        ),
        "best_budget_beats_mlp": (
            None
            if best_budget_row is None or best_budget_row["primary_metric_mean"] is None or mlp["primary_metric_mean"] is None
            else _metric_value_is_better(
                base_spec.primary_metric_name,
                float(best_budget_row["primary_metric_mean"]),
                float(mlp["primary_metric_mean"]),
            )
        ),
        "budget2x_beats_1x_seed_count": _count_pairwise_seed_wins(
            seed_budget_records,
            left_variant="tuned_pc_2x",
            right_variant="tuned_pc_1x",
            metric_name=base_spec.primary_metric_name,
        ),
        "budget4x_beats_1x_seed_count": _count_pairwise_seed_wins(
            seed_budget_records,
            left_variant="tuned_pc_4x",
            right_variant="tuned_pc_1x",
            metric_name=base_spec.primary_metric_name,
        ),
        "budget4x_beats_2x_seed_count": _count_pairwise_seed_wins(
            seed_budget_records,
            left_variant="tuned_pc_4x",
            right_variant="tuned_pc_2x",
            metric_name=base_spec.primary_metric_name,
        ),
        "gap_to_mlp_at_1x_mean": gap_1x,
        "gap_to_mlp_at_2x_mean": gap_2x,
        "gap_to_mlp_at_4x_mean": gap_4x,
        "gap_to_mlp_shrinks_from_1x_to_2x": (
            None if gap_1x is None or gap_2x is None else abs(float(gap_2x)) < abs(float(gap_1x))
        ),
        "gap_to_mlp_shrinks_from_2x_to_4x": (
            None if gap_2x is None or gap_4x is None else abs(float(gap_4x)) < abs(float(gap_2x))
        ),
        "final_pre_update_energy_mean_by_budget": final_pre_update_energy_mean_by_budget,
        "final_post_update_energy_mean_by_budget": final_post_update_energy_mean_by_budget,
        "evidence_of_diminishing_returns": evidence_of_diminishing_returns,
        **headline_test,
        "tie_rtol": TIE_RTOL,
        "tie_atol": TIE_ATOL,
        "notes": {
            "budget_definition": "This is not a wall-clock- or FLOP-matched efficiency comparison. It studies tuned PC inference budget vs performance with MLP as a fixed reference.",
            "primary_metric_delta_tuned_pc_minus_mlp": "tuned_pc_budget_metric - mlp_metric",
            "gap_to_mlp_for_mse": "For the current regression tasks with MSE, a positive gap means tuned PC is still worse than MLP.",
            "refresh": (
                "This run uses the legacy Phase 2c tuned preset and the benchmark-default MLP config."
                if (
                    tuned_selection.source == PHASE2_TUNED_SOURCE_LEGACY
                    and mlp_selection.source == PHASE2_MLP_SOURCE_DEFAULT
                )
                else (
                    "This run uses the best eval-ranked Phase 2f joint-search PC config and the benchmark-default MLP config."
                    if (
                        tuned_selection.source == PHASE2_TUNED_SOURCE_PHASE2F
                        and mlp_selection.source == PHASE2_MLP_SOURCE_DEFAULT
                    )
                    else (
                        "This run uses the Phase 2g matched-search selected PC and MLP configs, with final comparisons reported on held-out test metrics."
                        if (
                            tuned_selection.source == PHASE2_TUNED_SOURCE_PHASE2G
                            and mlp_selection.source == PHASE2_MLP_SOURCE_PHASE2G
                        )
                        else (
                            "This run uses the Phase 2g.1 boundary-check refined PC and MLP configs, with final comparisons reported on held-out test metrics."
                            if (
                                tuned_selection.source == PHASE2_TUNED_SOURCE_PHASE2G1
                                and mlp_selection.source == PHASE2_MLP_SOURCE_PHASE2G1
                            )
                            else "This run uses a custom combination of selected PC/MLP configs."
                        )
                    )
                )
            ),
        },
    }


def _plot_tradeoff_summary(
    run_dir: Path,
    base_spec: ToyBenchmarkSpec,
    budget_summary_rows: list[dict[str, Any]],
    aggregate_summary: dict[str, Any],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "plot_summary=True requires matplotlib to be installed in the current environment."
        ) from exc

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    tuned_rows = [
        _summary_row_for_variant(budget_summary_rows, "tuned_pc_1x"),
        _summary_row_for_variant(budget_summary_rows, "tuned_pc_2x"),
        _summary_row_for_variant(budget_summary_rows, "tuned_pc_4x"),
    ]
    labels = [str(row["budget_multiplier"]) for row in tuned_rows]
    means = [float(row["primary_metric_mean"]) for row in tuned_rows]
    stds = [
        0.0 if row["primary_metric_std"] is None else float(row["primary_metric_std"])
        for row in tuned_rows
    ]
    mlp_mean = aggregate_summary["mlp_primary_metric_mean"]

    figure = plt.figure(figsize=(8, 4))
    axis = figure.add_subplot(1, 1, 1)
    axis.errorbar(labels, means, yerr=stds, marker="o", capsize=4, label="tuned_pc")
    if mlp_mean is not None:
        axis.axhline(float(mlp_mean), color="tab:red", linestyle="--", label="mlp")
    axis.set_xlabel("Budget multiplier")
    axis.set_ylabel(base_spec.primary_metric_name)
    axis.set_title(f"{base_spec.benchmark_name} tuned PC budget curve")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots_dir / "tuned_pc_budget_curve.png")
    plt.close(figure)

    figure = plt.figure(figsize=(8, 4))
    axis = figure.add_subplot(1, 1, 1)
    pre_values = [
        np.nan if row["final_pre_update_energy_mean"] is None else float(row["final_pre_update_energy_mean"])
        for row in tuned_rows
    ]
    post_values = [
        np.nan if row["final_post_update_energy_mean"] is None else float(row["final_post_update_energy_mean"])
        for row in tuned_rows
    ]
    axis.plot(labels, pre_values, marker="o", label="pre_update_energy")
    axis.plot(labels, post_values, marker="o", label="post_update_energy")
    axis.set_xlabel("Budget multiplier")
    axis.set_ylabel("Energy")
    axis.set_title(f"{base_spec.benchmark_name} final PC energy by budget")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots_dir / "pc_energy_by_budget.png")
    plt.close(figure)


def run_pc_budget_tradeoff_study(
    benchmark_name: str,
    *,
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    output_layout: OutputLayout = "single_dir",
    plot_energy: bool = False,
    plot_summary: bool = False,
    seed_values: Sequence[int] | None = None,
    tuned_source: str = PHASE2_TUNED_SOURCE_PHASE2G1,
    mlp_source: str | None = None,
    joint_search_output_root: str | Path = "outputs",
) -> PCBudgetTradeoffRunResult:
    """Run the fixed Phase 2e tuned-PC budget tradeoff study for one regression benchmark."""
    if benchmark_name not in PHASE2E_BENCHMARK_NAMES:
        raise ValueError(
            f"Phase 2e currently supports only {PHASE2E_BENCHMARK_NAMES}, got '{benchmark_name}'."
        )

    base_spec = get_benchmark_spec(benchmark_name)
    split = base_spec.make_dataset_split()
    resolved_mlp_source = (
        PHASE2_MLP_SOURCE_PHASE2G1
        if mlp_source is None and tuned_source == PHASE2_TUNED_SOURCE_PHASE2G1
        else PHASE2_MLP_SOURCE_PHASE2G
        if mlp_source is None and tuned_source == PHASE2_TUNED_SOURCE_PHASE2G
        else PHASE2_MLP_SOURCE_DEFAULT
        if mlp_source is None
        else mlp_source
    )
    tuned_selection = resolve_tuned_pc_selection(
        benchmark_name,
        tuned_source=tuned_source,
        joint_search_output_root=joint_search_output_root,
    )
    mlp_selection = resolve_mlp_selection(
        benchmark_name,
        mlp_source=resolved_mlp_source,
        joint_search_output_root=joint_search_output_root,
    )
    resolved_run_id = _resolve_run_id(run_id)
    resolved_seed_values = (
        list(seed_values)
        if seed_values is not None
        else list(default_seed_values_for_benchmark(benchmark_name))
    )
    run_dir = _prepare_run_dir(
        _resolve_tradeoff_root(
            output_root,
            benchmark_name,
            resolved_run_id,
            output_layout,
            tuned_selection.source,
            mlp_selection.source,
        )
    )
    study_config = _build_study_config(
        base_spec,
        run_id=resolved_run_id,
        output_layout=output_layout,
        seed_values=resolved_seed_values,
        tuned_selection=tuned_selection,
        mlp_selection=mlp_selection,
        split=split,
    )
    _write_json(run_dir / "study_config.json", study_config)

    tuned_base_training = tuned_selection.pc_training
    tuned_budget_trainings = {
        multiplier: _tuned_budget_training(tuned_base_training, multiplier=multiplier)
        for multiplier in PHASE2E_BUDGET_MULTIPLIERS
    }

    seed_budget_records: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(resolved_seed_values):
        seed_dir = run_dir / "seeds" / f"seed_{int(seed):04d}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        variant_rows_by_name: dict[str, dict[str, Any]] = {}

        for multiplier in PHASE2E_BUDGET_MULTIPLIERS:
            variant = _variant_name_for_multiplier(multiplier)
            training = tuned_budget_trainings[multiplier]
            spec = _seeded_spec(
                base_spec,
                seed=int(seed),
                pc_training=training,
                epochs=tuned_selection.epochs,
            )
            row: dict[str, Any] = {
                "seed_index": seed_index,
                "run_seed": int(seed),
                "data_seed": base_spec.data_seed,
                "model_init_seed": int(seed),
                "variant": variant,
                "variant_group": _resolve_variant_group(variant),
                "is_reference_variant": False,
                "budget_multiplier": _budget_multiplier_label(multiplier),
                "train_steps": int(training.train_steps),
                "eval_steps": int(training.eval_steps),
                "status": "failed",
                "failure_reason": "",
                "primary_metric_name": base_spec.primary_metric_name,
                "primary_metric_value": None,
                "final_pre_update_energy": None,
                "final_post_update_energy": None,
                "final_loss": None,
                "summary_path": "",
                "primary_metric_delta_vs_tuned_pc_1x": None,
                "primary_metric_delta_tuned_pc_minus_mlp": None,
                "variant_beats_tuned_pc_1x": None,
                "variant_beats_mlp": None,
            }
            try:
                result = run_pc_benchmark(
                    spec,
                    output_root=seed_dir,
                    run_id=resolved_run_id,
                    plot_energy=plot_energy,
                    output_layout="single_dir",
                    experiment_name=variant,
                    split=split,
                )
                row["status"] = "ok"
                row["primary_metric_value"] = float(result.summary["test_metric"])
                row["final_pre_update_energy"] = float(result.summary["final_pre_update_energy"])
                row["final_post_update_energy"] = float(result.summary["final_post_update_energy"])
                row["summary_path"] = _summary_path_for_seed_variant(seed, variant)
            except Exception as exc:
                _clean_failed_variant_dir(seed_dir / variant)
                row["failure_reason"] = _failure_reason(exc)
            variant_rows_by_name[variant] = row
            seed_budget_records.append(row)

        mlp_variant = PHASE2E_REFERENCE_VARIANT
        mlp_row: dict[str, Any] = {
            "seed_index": seed_index,
            "run_seed": int(seed),
            "data_seed": base_spec.data_seed,
            "model_init_seed": int(seed),
            "variant": mlp_variant,
            "variant_group": _resolve_variant_group(mlp_variant),
            "is_reference_variant": True,
            "budget_multiplier": "reference",
            "train_steps": None,
            "eval_steps": None,
            "status": "failed",
            "failure_reason": "",
            "primary_metric_name": base_spec.primary_metric_name,
            "primary_metric_value": None,
            "final_pre_update_energy": None,
            "final_post_update_energy": None,
            "final_loss": None,
            "summary_path": "",
            "primary_metric_delta_vs_tuned_pc_1x": None,
            "primary_metric_delta_tuned_pc_minus_mlp": None,
            "variant_beats_tuned_pc_1x": None,
            "variant_beats_mlp": None,
        }
        mlp_spec = _seeded_spec(
            base_spec,
            seed=int(seed),
            mlp_training=mlp_selection.mlp_training,
            epochs=mlp_selection.epochs,
        )
        try:
            _, _, mlp_summary = _run_mlp_variant(
                mlp_spec,
                variant_dir=seed_dir / mlp_variant,
                run_id=resolved_run_id,
                output_layout="single_dir",
                split=split,
            )
            mlp_row["status"] = "ok"
            mlp_row["primary_metric_value"] = float(mlp_summary["test_metric"])
            mlp_row["final_loss"] = float(mlp_summary["final_loss"])
            mlp_row["summary_path"] = _summary_path_for_seed_variant(seed, mlp_variant)
        except Exception as exc:
            _clean_failed_variant_dir(seed_dir / mlp_variant)
            mlp_row["failure_reason"] = _failure_reason(exc)
        variant_rows_by_name[mlp_variant] = mlp_row
        seed_budget_records.append(mlp_row)

        base_row = variant_rows_by_name["tuned_pc_1x"]
        base_value = (
            None if base_row["status"] != "ok" else float(base_row["primary_metric_value"])
        )
        mlp_value = (
            None if mlp_row["status"] != "ok" else float(mlp_row["primary_metric_value"])
        )

        for variant in PHASE2E_VARIANT_ORDER:
            row = variant_rows_by_name[variant]
            if row["status"] == "ok" and base_value is not None:
                row["primary_metric_delta_vs_tuned_pc_1x"] = float(row["primary_metric_value"]) - base_value
                row["variant_beats_tuned_pc_1x"] = _metric_value_is_better(
                    base_spec.primary_metric_name,
                    float(row["primary_metric_value"]),
                    base_value,
                )
            if row["status"] == "ok" and mlp_value is not None:
                row["variant_beats_mlp"] = _metric_value_is_better(
                    base_spec.primary_metric_name,
                    float(row["primary_metric_value"]),
                    mlp_value,
                )
            if (
                variant != PHASE2E_REFERENCE_VARIANT
                and row["status"] == "ok"
                and mlp_value is not None
            ):
                row["primary_metric_delta_tuned_pc_minus_mlp"] = (
                    float(row["primary_metric_value"]) - mlp_value
                )

    budget_summary_rows = _build_budget_summary_rows(seed_budget_records=seed_budget_records)
    aggregate_summary = _build_aggregate_summary(
        base_spec=base_spec,
        run_id=resolved_run_id,
        seed_values=resolved_seed_values,
        seed_budget_records=seed_budget_records,
        budget_summary_rows=budget_summary_rows,
        tuned_selection=tuned_selection,
        mlp_selection=mlp_selection,
    )

    _write_seed_budget_records(run_dir / "seed_budget_records.csv", seed_budget_records)
    _write_budget_summary(run_dir / "budget_summary.csv", budget_summary_rows)
    _write_json(run_dir / "aggregate_summary.json", aggregate_summary)

    if plot_summary:
        _plot_tradeoff_summary(run_dir, base_spec, budget_summary_rows, aggregate_summary)

    return PCBudgetTradeoffRunResult(
        run_dir=run_dir,
        study_config=study_config,
        seed_budget_records=seed_budget_records,
        budget_summary=budget_summary_rows,
        aggregate_summary=aggregate_summary,
    )
