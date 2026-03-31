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


def _resolve_tradeoff_root(
    output_root: str | Path,
    benchmark_name: str,
    run_id: str,
    output_layout: OutputLayout,
) -> Path:
    experiment_name = f"pc_budget_tradeoff_{benchmark_name}"
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
) -> ToyBenchmarkSpec:
    return replace(
        base_spec,
        run_seed=seed,
        model_init_seed=seed,
        data_seed=base_spec.data_seed,
        pc_training=base_spec.pc_training if pc_training is None else pc_training,
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


def _build_study_config(
    base_spec: ToyBenchmarkSpec,
    *,
    run_id: str,
    output_layout: OutputLayout,
    seed_values: Sequence[int],
) -> dict[str, Any]:
    tuned_training = get_phase2c_tuned_pc_training(base_spec.benchmark_name)
    return {
        "experiment_name": f"pc_budget_tradeoff_{base_spec.benchmark_name}",
        "run_id": run_id,
        "phase": "Phase 2e",
        "benchmark_name": base_spec.benchmark_name,
        "task_name": base_spec.task_name,
        "seed_values": [int(seed) for seed in seed_values],
        "seed_semantics": {
            "data_seed": "fixed",
            "run_seed": "varies_with_seed",
            "model_init_seed": "varies_with_seed",
            "notes": "This phase mainly measures initialization stability rather than dataset sampling variability.",
        },
        "data_seed_fixed": base_spec.data_seed,
        "tuned_pc_base_training": _training_dict(tuned_training, epochs=base_spec.epochs),
        "budget_levels": _budget_levels(tuned_training),
        "mlp_training": {
            "epochs": base_spec.epochs,
            "eta_w": base_spec.mlp_training.eta_w,
            "eta_b": base_spec.mlp_training.eta_b,
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

    return {
        "experiment_name": f"pc_budget_tradeoff_{base_spec.benchmark_name}",
        "run_id": run_id,
        "phase": "Phase 2e",
        "benchmark_name": base_spec.benchmark_name,
        "task_name": base_spec.task_name,
        "primary_metric_name": base_spec.primary_metric_name,
        "primary_metric_higher_is_better": base_spec.primary_metric_higher_is_better,
        "seed_values": [int(seed) for seed in seed_values],
        "variant_groups": dict(PHASE2E_VARIANT_GROUPS),
        "budget_levels": _budget_levels(get_phase2c_tuned_pc_training(base_spec.benchmark_name)),
        "budget_definition": "budget = inference step count, not wall-clock or FLOP matching",
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
        "tie_rtol": TIE_RTOL,
        "tie_atol": TIE_ATOL,
        "notes": {
            "budget_definition": "This is not a wall-clock- or FLOP-matched efficiency comparison. It studies tuned PC inference budget vs performance with MLP as a fixed reference.",
            "primary_metric_delta_tuned_pc_minus_mlp": "tuned_pc_budget_metric - mlp_metric",
            "gap_to_mlp_for_mse": "For the current regression tasks with MSE, a positive gap means tuned PC is still worse than MLP.",
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
) -> PCBudgetTradeoffRunResult:
    """Run the fixed Phase 2e tuned-PC budget tradeoff study for one regression benchmark."""
    if benchmark_name not in PHASE2E_BENCHMARK_NAMES:
        raise ValueError(
            f"Phase 2e currently supports only {PHASE2E_BENCHMARK_NAMES}, got '{benchmark_name}'."
        )

    base_spec = get_benchmark_spec(benchmark_name)
    resolved_run_id = _resolve_run_id(run_id)
    resolved_seed_values = (
        list(seed_values)
        if seed_values is not None
        else list(default_seed_values_for_benchmark(benchmark_name))
    )
    run_dir = _prepare_run_dir(
        _resolve_tradeoff_root(output_root, benchmark_name, resolved_run_id, output_layout)
    )
    study_config = _build_study_config(
        base_spec,
        run_id=resolved_run_id,
        output_layout=output_layout,
        seed_values=resolved_seed_values,
    )
    _write_json(run_dir / "study_config.json", study_config)

    x, y = base_spec.make_data()
    tuned_base_training = get_phase2c_tuned_pc_training(benchmark_name)
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
            spec = _seeded_spec(base_spec, seed=int(seed), pc_training=training)
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
                    x=x,
                    y=y,
                )
                row["status"] = "ok"
                row["primary_metric_value"] = float(result.summary["primary_metric_value"])
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
        mlp_spec = _seeded_spec(base_spec, seed=int(seed))
        try:
            _, _, mlp_summary = _run_mlp_variant(
                mlp_spec,
                variant_dir=seed_dir / mlp_variant,
                run_id=resolved_run_id,
                output_layout="single_dir",
                x=x,
                y=y,
            )
            mlp_row["status"] = "ok"
            mlp_row["primary_metric_value"] = float(mlp_summary["primary_metric_value"])
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
