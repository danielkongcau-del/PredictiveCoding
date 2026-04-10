from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from .fmpc_tf2 import (
    FMPCTF2Config,
    build_tf2_corrective_transport_default_config,
    build_tf2_corrective_transport_terminal_angleclip_default_config,
    run_fmpc_tf2_experiment,
)

MainlineAdoptionCellKey = Literal[
    "tf2_corrective_transport_default",
    "tf2_corrective_transport_terminal_angleclip_default",
    "residualized_local_field_poly_rt2__no_terminal_stabilizer",
    "residualized_local_field_poly_rt2__terminal_hard_replace_upper_bound",
]


@dataclass(frozen=True)
class _CellSpec:
    cell_key: MainlineAdoptionCellKey
    candidate_key: str


@dataclass
class FMPCTF2MainlineAdoptionSuiteConfig:
    """Minimal mainline confirmation for the integrated TF2 terminal local-field preset."""

    experiment_name: str = "fmpc_tf2_mainline_adoption_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: Literal["single_dir", "run_id_subdir"] = "single_dir"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    epochs: int = 60
    batch_size: int = 128
    eval_steps: int = 15
    layer_dims: tuple[int, ...] = (64, 64, 10)
    confirmation_test_gain_vs_old_preset: float = 0.005
    required_gate_seed_majority_rate: float = 0.5
    selected_epoch_passes_gate_rate_floor: float = 0.5
    modest_runtime_overhead_seconds: float = 2.0

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"


@dataclass
class FMPCTF2MainlineAdoptionSuiteRunResult:
    run_dir: Path
    config: dict[str, Any]
    run_rows: list[dict[str, Any]]
    summary: dict[str, Any]


def _resolve_run_dir(
    output_root: str | Path,
    experiment_name: str,
    run_id: str,
    output_layout: Literal["single_dir", "run_id_subdir"],
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("rows must contain at least one entry.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _relative_posix(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _cell_specs() -> list[_CellSpec]:
    return [
        _CellSpec(
            cell_key="tf2_corrective_transport_default",
            candidate_key="baseline_plain_raw",
        ),
        _CellSpec(
            cell_key="tf2_corrective_transport_terminal_angleclip_default",
            candidate_key="residualized_local_field_poly_rt2__terminal_angleclip_package",
        ),
        _CellSpec(
            cell_key="residualized_local_field_poly_rt2__no_terminal_stabilizer",
            candidate_key="residualized_local_field_poly_rt2__no_terminal_stabilizer",
        ),
        _CellSpec(
            cell_key="residualized_local_field_poly_rt2__terminal_hard_replace_upper_bound",
            candidate_key="residualized_local_field_poly_rt2__terminal_hard_replace_upper_bound",
        ),
    ]


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Mean requires at least one value.")
    return float(sum(values) / float(len(values)))


def _std(values: list[float]) -> float:
    if not values:
        raise ValueError("Std requires at least one value.")
    mean_value = _mean(values)
    variance = sum((float(value) - mean_value) ** 2 for value in values) / float(len(values))
    return float(variance**0.5)


def _as_rate(values: list[bool]) -> float:
    if not values:
        raise ValueError("Rate requires at least one value.")
    return float(sum(1.0 for value in values if bool(value)) / float(len(values)))


def _suite_config_payload(config: FMPCTF2MainlineAdoptionSuiteConfig) -> dict[str, Any]:
    return {
        "phase": "Phase TF2",
        "stage": "mainline_terminal_localfield_adoption_confirmation",
        "adoption_oriented": True,
        "comparison_cells": [spec.cell_key for spec in _cell_specs()],
        "seeds": [int(seed) for seed in config.seeds],
        "fixed": {
            "root_preset_family": "tf2_corrective_transport_default",
            "feature_aware_tangents": False,
            "micro_steps": 4,
            "incremental_weight_updates": False,
            "supervision_policy": "local_only",
            "theta_update_cadence": "terminal_only",
            "theta_update_budget": "matched",
            "identity_loss_weight": 0.2,
            "warmup_epochs": 5,
            "hybrid_ramp_epochs": 10,
            "bootstrap_integrator": "rk2",
            "bootstrap_substeps": 4,
            "terminal_source_family": "teacher_free_local_field",
            "selector_policy": "gate_constrained_accuracy_then_val_accuracy",
        },
        "new_preset_name": "tf2_corrective_transport_terminal_angleclip_default",
        "historical_reference_preset": "tf2_corrective_transport_default",
        "confirmation_thresholds": {
            "mean_test_accuracy_gain_vs_old_preset": float(config.confirmation_test_gain_vs_old_preset),
            "required_gate_seed_majority_rate": float(config.required_gate_seed_majority_rate),
            "selected_epoch_passes_gate_rate_floor": float(config.selected_epoch_passes_gate_rate_floor),
            "modest_runtime_overhead_seconds": float(config.modest_runtime_overhead_seconds),
        },
    }


def _build_run_config(
    spec: _CellSpec,
    *,
    seed: int,
    suite_config: FMPCTF2MainlineAdoptionSuiteConfig,
    base_run_dir: Path,
) -> FMPCTF2Config:
    common_overrides = {
        "run_seed": int(seed),
        "data_seed": int(seed),
        "model_init_seed": int(seed),
        "psi_init_seed": int(seed),
        "batch_order_seed": int(seed),
        "epochs": int(suite_config.epochs),
        "batch_size": int(suite_config.batch_size),
        "eval_steps": int(suite_config.eval_steps),
        "layer_dims": suite_config.layer_dims,
        "output_root": base_run_dir / "runs",
        "experiment_name": spec.cell_key,
        "output_layout": "run_id_subdir",
        "run_id": f"seed_{int(seed)}",
    }
    if spec.cell_key == "tf2_corrective_transport_default":
        return build_tf2_corrective_transport_default_config(**common_overrides)
    if spec.cell_key == "tf2_corrective_transport_terminal_angleclip_default":
        return build_tf2_corrective_transport_terminal_angleclip_default_config(**common_overrides)
    if spec.cell_key == "residualized_local_field_poly_rt2__no_terminal_stabilizer":
        return build_tf2_corrective_transport_terminal_angleclip_default_config(
            **common_overrides,
            preset_name=None,
            terminal_local_field_direction_intervention="none",
        )
    if spec.cell_key == "residualized_local_field_poly_rt2__terminal_hard_replace_upper_bound":
        return build_tf2_corrective_transport_terminal_angleclip_default_config(
            **common_overrides,
            preset_name=None,
            terminal_local_field_direction_intervention="local_field_direction_hard_replace_keep_live_norm",
        )
    raise ValueError(f"Unsupported cell_key '{spec.cell_key}'.")


def _success_run_row(
    *,
    spec: _CellSpec,
    seed: int,
    result: Any,
    base_run_dir: Path,
) -> dict[str, Any]:
    timing = dict(result.summary.get("timing", {}))
    return {
        "cell_key": spec.cell_key,
        "candidate_key": spec.candidate_key,
        "preset_name": result.summary["preset_name"],
        "psi_family": result.summary["psi_family"],
        "time_encoding_variant": result.summary["time_encoding_variant"],
        "terminal_local_field_direction_intervention": result.summary[
            "terminal_local_field_direction_intervention"
        ],
        "terminal_local_field_angle_clip_degrees": result.summary["terminal_local_field_angle_clip_degrees"],
        "seed": int(seed),
        "val_accuracy": float(result.summary["val_accuracy"]),
        "test_accuracy": float(result.summary["test_accuracy"]),
        "gate_passing_epoch_count": int(result.summary["gate_passing_epoch_count"]),
        "selected_epoch": int(result.summary["best_epoch"]),
        "selected_epoch_passes_gate": bool(result.summary["selected_epoch_passes_gate"]),
        "selector_fallback_used": bool(result.summary["selector_fallback_used"]),
        "val_transported_final_energy": float(result.summary["val_transported_final_energy"]),
        "total_wall_time_seconds": float(timing.get("train_wall_time_seconds", 0.0))
        + float(timing.get("final_evaluation_wall_time_seconds", 0.0)),
        "run_status": "success",
        "run_summary_path": _relative_posix(base_run_dir, result.run_dir / "summary.json"),
    }


def _aggregate_run_group(run_rows: list[dict[str, Any]], *, cell_key: MainlineAdoptionCellKey) -> dict[str, Any]:
    relevant = [row for row in run_rows if row["cell_key"] == cell_key and row["run_status"] == "success"]
    if not relevant:
        return {
            "mean_val_accuracy": None,
            "std_val_accuracy": None,
            "mean_test_accuracy": None,
            "std_test_accuracy": None,
            "mean_gate_passing_epoch_count": None,
            "seed_gate_positive_rate": None,
            "selected_epoch_passes_gate_rate": None,
            "selector_fallback_used_rate": None,
            "mean_val_transported_final_energy": None,
            "mean_total_wall_time_seconds": None,
            "std_total_wall_time_seconds": None,
        }
    val_values = [float(row["val_accuracy"]) for row in relevant]
    test_values = [float(row["test_accuracy"]) for row in relevant]
    gate_values = [float(row["gate_passing_epoch_count"]) for row in relevant]
    energy_values = [float(row["val_transported_final_energy"]) for row in relevant]
    wall_values = [float(row["total_wall_time_seconds"]) for row in relevant]
    selected_epoch_passes_gate = [bool(row["selected_epoch_passes_gate"]) for row in relevant]
    selector_fallback_used = [bool(row["selector_fallback_used"]) for row in relevant]
    seed_gate_positive = [float(row["gate_passing_epoch_count"]) > 0.0 for row in relevant]
    return {
        "mean_val_accuracy": _mean(val_values),
        "std_val_accuracy": _std(val_values),
        "mean_test_accuracy": _mean(test_values),
        "std_test_accuracy": _std(test_values),
        "mean_gate_passing_epoch_count": _mean(gate_values),
        "seed_gate_positive_rate": _as_rate(seed_gate_positive),
        "selected_epoch_passes_gate_rate": _as_rate(selected_epoch_passes_gate),
        "selector_fallback_used_rate": _as_rate(selector_fallback_used),
        "mean_val_transported_final_energy": _mean(energy_values),
        "mean_total_wall_time_seconds": _mean(wall_values),
        "std_total_wall_time_seconds": _std(wall_values),
    }


def _pairwise_vs_reference(
    run_rows: list[dict[str, Any]],
    *,
    candidate_cell_key: MainlineAdoptionCellKey,
    reference_cell_key: MainlineAdoptionCellKey,
) -> dict[str, Any]:
    candidate_by_seed = {
        int(row["seed"]): row
        for row in run_rows
        if row["cell_key"] == candidate_cell_key and row["run_status"] == "success"
    }
    reference_by_seed = {
        int(row["seed"]): row
        for row in run_rows
        if row["cell_key"] == reference_cell_key and row["run_status"] == "success"
    }
    shared_seeds = sorted(set(candidate_by_seed).intersection(reference_by_seed))
    if not shared_seeds:
        return {
            "mean_val_accuracy_delta": None,
            "mean_test_accuracy_delta": None,
            "mean_gate_passing_epoch_count_delta": None,
            "seed_gate_positive_rate_delta": None,
            "selected_epoch_passes_gate_rate_delta": None,
            "selector_fallback_used_rate_delta": None,
            "mean_val_transported_final_energy_delta": None,
            "mean_runtime_delta": None,
        }
    return {
        "mean_val_accuracy_delta": _mean(
            [
                float(candidate_by_seed[seed]["val_accuracy"]) - float(reference_by_seed[seed]["val_accuracy"])
                for seed in shared_seeds
            ]
        ),
        "mean_test_accuracy_delta": _mean(
            [
                float(candidate_by_seed[seed]["test_accuracy"]) - float(reference_by_seed[seed]["test_accuracy"])
                for seed in shared_seeds
            ]
        ),
        "mean_gate_passing_epoch_count_delta": _mean(
            [
                float(candidate_by_seed[seed]["gate_passing_epoch_count"])
                - float(reference_by_seed[seed]["gate_passing_epoch_count"])
                for seed in shared_seeds
            ]
        ),
        "seed_gate_positive_rate_delta": _mean(
            [
                float(float(candidate_by_seed[seed]["gate_passing_epoch_count"]) > 0.0)
                - float(float(reference_by_seed[seed]["gate_passing_epoch_count"]) > 0.0)
                for seed in shared_seeds
            ]
        ),
        "selected_epoch_passes_gate_rate_delta": _mean(
            [
                float(bool(candidate_by_seed[seed]["selected_epoch_passes_gate"]))
                - float(bool(reference_by_seed[seed]["selected_epoch_passes_gate"]))
                for seed in shared_seeds
            ]
        ),
        "selector_fallback_used_rate_delta": _mean(
            [
                float(bool(candidate_by_seed[seed]["selector_fallback_used"]))
                - float(bool(reference_by_seed[seed]["selector_fallback_used"]))
                for seed in shared_seeds
            ]
        ),
        "mean_val_transported_final_energy_delta": _mean(
            [
                float(candidate_by_seed[seed]["val_transported_final_energy"])
                - float(reference_by_seed[seed]["val_transported_final_energy"])
                for seed in shared_seeds
            ]
        ),
        "mean_runtime_delta": _mean(
            [
                float(candidate_by_seed[seed]["total_wall_time_seconds"])
                - float(reference_by_seed[seed]["total_wall_time_seconds"])
                for seed in shared_seeds
            ]
        ),
    }


def run_fmpc_tf2_mainline_adoption_suite(
    config: FMPCTF2MainlineAdoptionSuiteConfig,
) -> FMPCTF2MainlineAdoptionSuiteRunResult:
    run_dir = _prepare_run_dir(
        _resolve_run_dir(
            config.output_root,
            config.experiment_name,
            config.resolved_run_id(),
            config.output_layout,
        )
    )
    _write_json(run_dir / "config.json", _suite_config_payload(config))

    run_rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        for spec in _cell_specs():
            run_config = _build_run_config(
                spec,
                seed=int(seed),
                suite_config=config,
                base_run_dir=run_dir,
            )
            result = run_fmpc_tf2_experiment(run_config)
            run_rows.append(
                _success_run_row(
                    spec=spec,
                    seed=int(seed),
                    result=result,
                    base_run_dir=run_dir,
                )
            )
    _write_csv(run_dir / "end_to_end_runs.csv", run_rows)

    old_summary = _aggregate_run_group(run_rows, cell_key="tf2_corrective_transport_default")
    new_summary = _aggregate_run_group(
        run_rows,
        cell_key="tf2_corrective_transport_terminal_angleclip_default",
    )
    unstabilized_summary = _aggregate_run_group(
        run_rows,
        cell_key="residualized_local_field_poly_rt2__no_terminal_stabilizer",
    )
    hard_replace_summary = _aggregate_run_group(
        run_rows,
        cell_key="residualized_local_field_poly_rt2__terminal_hard_replace_upper_bound",
    )
    pairwise_new_vs_old = _pairwise_vs_reference(
        run_rows,
        candidate_cell_key="tf2_corrective_transport_terminal_angleclip_default",
        reference_cell_key="tf2_corrective_transport_default",
    )
    pairwise_new_vs_unstabilized = _pairwise_vs_reference(
        run_rows,
        candidate_cell_key="tf2_corrective_transport_terminal_angleclip_default",
        reference_cell_key="residualized_local_field_poly_rt2__no_terminal_stabilizer",
    )
    pairwise_hard_replace_vs_new = _pairwise_vs_reference(
        run_rows,
        candidate_cell_key="residualized_local_field_poly_rt2__terminal_hard_replace_upper_bound",
        reference_cell_key="tf2_corrective_transport_terminal_angleclip_default",
    )

    new_beats_old = bool(
        pairwise_new_vs_old["mean_test_accuracy_delta"] is not None
        and float(pairwise_new_vs_old["mean_test_accuracy_delta"]) >= float(config.confirmation_test_gain_vs_old_preset)
        and float(pairwise_new_vs_old["mean_val_accuracy_delta"]) >= 0.0
    )
    new_gate_robust = bool(
        new_summary["seed_gate_positive_rate"] is not None
        and float(new_summary["seed_gate_positive_rate"]) > float(config.required_gate_seed_majority_rate)
        and float(new_summary["selected_epoch_passes_gate_rate"])
        >= float(config.selected_epoch_passes_gate_rate_floor)
    )
    new_runtime_modest = bool(
        pairwise_new_vs_old["mean_runtime_delta"] is not None
        and float(pairwise_new_vs_old["mean_runtime_delta"]) <= float(config.modest_runtime_overhead_seconds)
    )
    hard_replace_is_upper_bound = bool(
        hard_replace_summary["mean_test_accuracy"] is not None
        and new_summary["mean_test_accuracy"] is not None
        and float(hard_replace_summary["mean_test_accuracy"]) >= float(new_summary["mean_test_accuracy"])
        and (
            float(hard_replace_summary["seed_gate_positive_rate"]) < float(new_summary["seed_gate_positive_rate"])
            or float(hard_replace_summary["selected_epoch_passes_gate_rate"])
            < float(new_summary["selected_epoch_passes_gate_rate"])
            or float(hard_replace_summary["selector_fallback_used_rate"])
            > float(new_summary["selector_fallback_used_rate"])
        )
    )
    should_adopt = bool(new_beats_old and new_gate_robust and new_runtime_modest and hard_replace_is_upper_bound)

    if should_adopt:
        dominant_interpretation = (
            "the integrated angle-clip package preserves the package-level confirmation result and "
            "should replace the historical plain corrective reference as the next TF2 experimental default"
        )
        recommended_preset = "tf2_corrective_transport_terminal_angleclip_default"
    else:
        dominant_interpretation = (
            "the integrated angle-clip package does not yet clear the adoption thresholds cleanly enough "
            "to replace the historical plain corrective reference"
        )
        recommended_preset = "tf2_corrective_transport_default"

    summary = {
        "phase": "Phase TF2",
        "stage": "mainline_terminal_localfield_adoption_confirmation",
        "adoption_oriented": True,
        "historical_reference_preset": "tf2_corrective_transport_default",
        "candidate_preset": "tf2_corrective_transport_terminal_angleclip_default",
        "comparison_cells": [spec.cell_key for spec in _cell_specs()],
        "end_to_end_summary": {
            "tf2_corrective_transport_default": old_summary,
            "tf2_corrective_transport_terminal_angleclip_default": new_summary,
            "residualized_local_field_poly_rt2__no_terminal_stabilizer": unstabilized_summary,
            "residualized_local_field_poly_rt2__terminal_hard_replace_upper_bound": hard_replace_summary,
        },
        "pairwise_delta_new_preset_vs_old_preset": pairwise_new_vs_old,
        "pairwise_delta_new_preset_vs_unstabilized_challenger": pairwise_new_vs_unstabilized,
        "pairwise_delta_hard_replace_vs_new_preset": pairwise_hard_replace_vs_new,
        "confirmation_thresholds": {
            "mean_test_accuracy_gain_vs_old_preset": float(config.confirmation_test_gain_vs_old_preset),
            "required_gate_seed_majority_rate": float(config.required_gate_seed_majority_rate),
            "selected_epoch_passes_gate_rate_floor": float(config.selected_epoch_passes_gate_rate_floor),
            "modest_runtime_overhead_seconds": float(config.modest_runtime_overhead_seconds),
        },
        "does_new_preset_beat_old_corrective_default": bool(new_beats_old),
        "is_new_preset_gain_robust_across_seeds_and_gate_constrained_selection": bool(new_gate_robust),
        "is_new_preset_runtime_overhead_modest": bool(new_runtime_modest),
        "is_hard_replace_still_only_upper_bound_control": bool(hard_replace_is_upper_bound),
        "should_promote_new_preset_as_next_tf2_experimental_default": bool(should_adopt),
        "recommended_tf2_experimental_default_preset": recommended_preset,
        "tradeoff_note": (
            "Hard replace remains a diagnostic upper-bound if it keeps a higher mean test score but loses "
            "gate robustness or selector stability relative to the angle-clip package."
        ),
        "dominant_interpretation": dominant_interpretation,
        "next_single_move": (
            "promote the integrated angle-clip package into the named TF2 experimental default path"
            if should_adopt
            else "keep the historical corrective default and inspect where the integrated package lost robustness"
        ),
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF2MainlineAdoptionSuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config),
        run_rows=run_rows,
        summary=summary,
    )
