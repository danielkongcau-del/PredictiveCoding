from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .fmpc_tf2 import TF2PresetName, build_tf2_preset_config, run_fmpc_tf2_experiment
from .real_pc import RealPCConfig, run_digits_pc_experiment


@dataclass
class FMPCTF2ExternalComparisonSuiteConfig:
    """Run a narrow TF2 external comparison against the canonical slow-PC digits baseline."""

    experiment_name: str = "fmpc_tf2_external_comparison_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    tf2_preset_names: tuple[TF2PresetName, ...] = (
        "tf2_corrective_transport_terminal_angleclip_default",
        "tf2_corrective_transport_default",
        "tf2_canonical",
    )
    slow_pc_method_name: str = "canonical_slow_pc_digits_baseline"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    tf2_epochs: int = 60
    tf2_batch_size: int = 128
    tf2_eval_steps: int = 15
    tf2_layer_dims: tuple[int, ...] = (64, 64, 10)
    slow_pc_epochs: int = 60
    slow_pc_batch_size: int = 64
    slow_pc_train_steps: int = 30
    slow_pc_eval_steps: int = 30
    slow_pc_layer_dims: tuple[int, ...] = (64, 64, 10)
    material_gap_narrowing_threshold: float = 0.005
    broad_stage_gap_threshold: float = 0.05

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"


@dataclass
class FMPCTF2ExternalComparisonSuiteRunResult:
    run_dir: Path
    config: dict[str, Any]
    aggregate_rows: list[dict[str, Any]]
    summary: dict[str, Any]


def _resolve_run_dir(output_root: str | Path, experiment_name: str, run_id: str, output_layout: str) -> Path:
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
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _relative_posix(base_dir: Path, target: Path) -> str:
    return target.relative_to(base_dir).as_posix()


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


def _rate(values: list[bool]) -> float:
    if not values:
        raise ValueError("Rate requires at least one value.")
    return float(sum(1.0 for value in values if bool(value)) / float(len(values)))


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _local_baseline_audit() -> dict[str, Any]:
    digits_pc_summary = Path("outputs/digits_pc/summary.json")
    digits_baselines_summary = Path("outputs/digits_baselines/summary.json")
    return {
        "digits_pc_summary_exists": bool(digits_pc_summary.exists()),
        "digits_baselines_summary_exists": bool(digits_baselines_summary.exists()),
        "reused_existing_canonical_slow_pc_artifacts": False,
        "reuse_reason": (
            "Local canonical slow-PC multiseed artifacts were not available, so the suite reran the "
            "canonical slow-PC digits baseline per seed inside the comparison run."
        ),
    }


def _suite_config_payload(config: FMPCTF2ExternalComparisonSuiteConfig) -> dict[str, Any]:
    return {
        "phase": "Phase TF2",
        "stage": "external_comparison_gap_closure",
        "tf2_presets": [str(name) for name in config.tf2_preset_names],
        "slow_pc_reference_method": config.slow_pc_method_name,
        "seeds": [int(seed) for seed in config.seeds],
        "tf2_fixed": {
            "feature_aware_tangents": False,
            "selector_policy": "gate_constrained_accuracy_then_val_accuracy",
            "validation_only_selection": True,
            "test_report_only": True,
        },
        "slow_pc_fixed": {
            "selection_metric_source": "val_metric",
            "report_metric_source": "test_metric",
        },
        "baseline_artifact_audit": _local_baseline_audit(),
        "thresholds": {
            "material_gap_narrowing_threshold": float(config.material_gap_narrowing_threshold),
            "broad_stage_gap_threshold": float(config.broad_stage_gap_threshold),
        },
    }


def _tf2_run_id(preset_name: TF2PresetName, seed: int) -> str:
    return f"seed_{seed}"


def _slow_pc_run_id(seed: int) -> str:
    return f"seed_{seed}"


def _tf2_row(
    run_index: int,
    run_dir: Path,
    preset_name: TF2PresetName,
    seed: int,
    result: Any,
) -> dict[str, Any]:
    summary = result.summary
    timing = dict(summary.get("timing", {}))
    return {
        "run_index": int(run_index),
        "method_name": str(preset_name),
        "family_or_preset_name": str(preset_name),
        "seed": int(seed),
        "run_id": str(result.run_dir.name),
        "run_summary_path": _relative_posix(run_dir, result.run_dir / "summary.json"),
        "checkpoint_selector": str(summary["checkpoint_selector"]),
        "val_accuracy": float(summary["val_accuracy"]),
        "test_accuracy": float(summary["test_accuracy"]),
        "gate_passing_epoch_count": int(summary["gate_passing_epoch_count"]),
        "selected_epoch": int(summary["best_epoch"]),
        "selected_epoch_passes_gate": bool(summary["selected_epoch_passes_gate"]),
        "selector_fallback_used": bool(summary["selector_fallback_used"]),
        "val_transported_final_energy": float(summary["val_transported_final_energy"]),
        "runtime_proxy_seconds": float(timing.get("train_wall_time_seconds", 0.0))
        + float(timing.get("final_evaluation_wall_time_seconds", 0.0)),
        "uses_transport_only_training": True,
        "uses_slow_iterative_pc_inference": False,
    }


def _slow_pc_row(
    run_index: int,
    run_dir: Path,
    seed: int,
    result: Any,
    method_name: str,
) -> dict[str, Any]:
    summary = result.summary
    timing = dict(summary.get("timing", {}))
    return {
        "run_index": int(run_index),
        "method_name": str(method_name),
        "family_or_preset_name": str(method_name),
        "seed": int(seed),
        "run_id": str(result.run_dir.name),
        "run_summary_path": _relative_posix(run_dir, result.run_dir / "summary.json"),
        "checkpoint_selector": "val_accuracy_only",
        "val_accuracy": float(summary["val_metric"]),
        "test_accuracy": float(summary["test_metric"]),
        "gate_passing_epoch_count": None,
        "selected_epoch": int(summary["best_epoch"]),
        "selected_epoch_passes_gate": None,
        "selector_fallback_used": None,
        "val_transported_final_energy": None,
        "runtime_proxy_seconds": float(timing.get("train_wall_time_seconds", 0.0))
        + float(timing.get("final_evaluation_wall_time_seconds", 0.0)),
        "uses_transport_only_training": False,
        "uses_slow_iterative_pc_inference": True,
    }


def _method_rows(rows: list[dict[str, Any]], method_name: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["method_name"]) == method_name]


def _method_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Method summary requires at least one row.")
    val_accuracies = [float(row["val_accuracy"]) for row in rows]
    test_accuracies = [float(row["test_accuracy"]) for row in rows]
    selected_epochs = [float(row["selected_epoch"]) for row in rows]
    runtime_values = [float(row["runtime_proxy_seconds"]) for row in rows]
    gate_values = [
        float(value)
        for value in (_float_or_none(row["gate_passing_epoch_count"]) for row in rows)
        if value is not None
    ]
    selected_epoch_passes_gate_values = [
        bool(value)
        for value in (_bool_or_none(row["selected_epoch_passes_gate"]) for row in rows)
        if value is not None
    ]
    selector_fallback_values = [
        bool(value)
        for value in (_bool_or_none(row["selector_fallback_used"]) for row in rows)
        if value is not None
    ]
    energy_values = [
        float(value)
        for value in (_float_or_none(row["val_transported_final_energy"]) for row in rows)
        if value is not None
    ]
    seed_gate_positive_values = [value > 0.0 for value in gate_values]
    return {
        "num_runs": int(len(rows)),
        "mean_val_accuracy": _mean(val_accuracies),
        "std_val_accuracy": _std(val_accuracies),
        "mean_test_accuracy": _mean(test_accuracies),
        "std_test_accuracy": _std(test_accuracies),
        "mean_gate_passing_epoch_count": None if not gate_values else _mean(gate_values),
        "mean_selected_epoch": _mean(selected_epochs),
        "seed_gate_positive_rate": None if not gate_values else _rate(seed_gate_positive_values),
        "selected_epoch_passes_gate_rate": (
            None if not selected_epoch_passes_gate_values else _rate(selected_epoch_passes_gate_values)
        ),
        "selector_fallback_used_rate": None if not selector_fallback_values else _rate(selector_fallback_values),
        "mean_val_transported_final_energy": None if not energy_values else _mean(energy_values),
        "mean_runtime_proxy_seconds": _mean(runtime_values),
        "std_runtime_proxy_seconds": _std(runtime_values),
    }


def _pairwise_difference(
    rows: list[dict[str, Any]],
    *,
    candidate_method: str,
    reference_method: str,
) -> dict[str, Any]:
    candidate_by_seed = {int(row["seed"]): row for row in _method_rows(rows, candidate_method)}
    reference_by_seed = {int(row["seed"]): row for row in _method_rows(rows, reference_method)}
    shared_seeds = sorted(set(candidate_by_seed).intersection(reference_by_seed))
    if not shared_seeds:
        raise ValueError("Pairwise comparison requires at least one shared seed.")

    def _metric_deltas(field: str) -> list[float]:
        deltas: list[float] = []
        for seed in shared_seeds:
            candidate_value = _float_or_none(candidate_by_seed[seed][field])
            reference_value = _float_or_none(reference_by_seed[seed][field])
            if candidate_value is None or reference_value is None:
                continue
            deltas.append(float(candidate_value - reference_value))
        return deltas

    selected_epoch_passes_gate_candidate = [
        value
        for value in (_bool_or_none(candidate_by_seed[seed]["selected_epoch_passes_gate"]) for seed in shared_seeds)
        if value is not None
    ]
    selected_epoch_passes_gate_reference = [
        value
        for value in (_bool_or_none(reference_by_seed[seed]["selected_epoch_passes_gate"]) for seed in shared_seeds)
        if value is not None
    ]
    selector_fallback_candidate = [
        value
        for value in (_bool_or_none(candidate_by_seed[seed]["selector_fallback_used"]) for seed in shared_seeds)
        if value is not None
    ]
    selector_fallback_reference = [
        value
        for value in (_bool_or_none(reference_by_seed[seed]["selector_fallback_used"]) for seed in shared_seeds)
        if value is not None
    ]

    gate_candidate = _metric_deltas("gate_passing_epoch_count")
    candidate_gate_values = [
        float(value)
        for value in (
            _float_or_none(candidate_by_seed[seed]["gate_passing_epoch_count"]) for seed in shared_seeds
        )
        if value is not None
    ]
    reference_gate_values = [
        float(value)
        for value in (
            _float_or_none(reference_by_seed[seed]["gate_passing_epoch_count"]) for seed in shared_seeds
        )
        if value is not None
    ]
    seed_gate_positive_delta = None
    if candidate_gate_values and reference_gate_values:
        seed_gate_positive_delta = _rate([value > 0.0 for value in candidate_gate_values]) - _rate(
            [value > 0.0 for value in reference_gate_values]
        )

    return {
        "shared_seeds": [int(seed) for seed in shared_seeds],
        "mean_val_accuracy_delta": _mean(_metric_deltas("val_accuracy")),
        "mean_test_accuracy_delta": _mean(_metric_deltas("test_accuracy")),
        "mean_gate_passing_epoch_count_delta": None if not gate_candidate else _mean(gate_candidate),
        "mean_selected_epoch_delta": _mean(_metric_deltas("selected_epoch")),
        "selected_epoch_passes_gate_rate_delta": (
            None
            if not selected_epoch_passes_gate_candidate or not selected_epoch_passes_gate_reference
            else _rate(selected_epoch_passes_gate_candidate) - _rate(selected_epoch_passes_gate_reference)
        ),
        "selector_fallback_used_rate_delta": (
            None
            if not selector_fallback_candidate or not selector_fallback_reference
            else _rate(selector_fallback_candidate) - _rate(selector_fallback_reference)
        ),
        "seed_gate_positive_rate_delta": seed_gate_positive_delta,
        "mean_val_transported_final_energy_delta": (
            None
            if not _metric_deltas("val_transported_final_energy")
            else _mean(_metric_deltas("val_transported_final_energy"))
        ),
        "mean_runtime_proxy_seconds_delta": _mean(_metric_deltas("runtime_proxy_seconds")),
    }


def _best_pre_adoption_tf2_method(by_method: dict[str, dict[str, Any]]) -> str:
    candidates = [
        "tf2_corrective_transport_default",
        "tf2_canonical",
    ]
    return max(
        candidates,
        key=lambda method_name: (
            float(by_method[method_name]["mean_test_accuracy"]),
            float(by_method[method_name]["mean_val_accuracy"]),
        ),
    )


def _recommended_next_stage(
    adopted_summary: dict[str, Any],
    slow_pc_summary: dict[str, Any],
    config: FMPCTF2ExternalComparisonSuiteConfig,
) -> str:
    slow_gap = float(slow_pc_summary["mean_test_accuracy"]) - float(adopted_summary["mean_test_accuracy"])
    if slow_gap <= float(config.broad_stage_gap_threshold):
        return "prepare later broader stage"
    return "continue TF2 bridge inside the adopted package"


def run_fmpc_tf2_external_comparison_suite(
    config: FMPCTF2ExternalComparisonSuiteConfig,
) -> FMPCTF2ExternalComparisonSuiteRunResult:
    """Run a narrow external comparison centered on the adopted TF2 default."""

    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, config.resolved_run_id(), config.output_layout)
    )
    _write_json(run_dir / "config.json", _suite_config_payload(config))

    rows: list[dict[str, Any]] = []
    runs_root = run_dir / "runs"
    run_index = 0

    for preset_name in config.tf2_preset_names:
        for seed in config.seeds:
            run_index += 1
            tf2_config = build_tf2_preset_config(
                preset_name,
                output_root=runs_root,
                experiment_name=str(preset_name),
                output_layout="run_id_subdir",
                run_id=_tf2_run_id(preset_name, seed),
                run_seed=seed,
                data_seed=seed,
                model_init_seed=seed,
                psi_init_seed=seed,
                batch_order_seed=seed,
                epochs=int(config.tf2_epochs),
                batch_size=int(config.tf2_batch_size),
                eval_steps=int(config.tf2_eval_steps),
                layer_dims=config.tf2_layer_dims,
            )
            tf2_result = run_fmpc_tf2_experiment(tf2_config)
            rows.append(_tf2_row(run_index, run_dir, preset_name, seed, tf2_result))

    for seed in config.seeds:
        run_index += 1
        slow_pc_config = RealPCConfig(
            output_root=runs_root,
            experiment_name="digits_pc",
            output_layout="run_id_subdir",
            run_id=_slow_pc_run_id(seed),
            plot_curves=False,
            run_seed=seed,
            data_seed=seed,
            model_init_seed=seed,
            batch_order_seed=seed,
            epochs=int(config.slow_pc_epochs),
            batch_size=int(config.slow_pc_batch_size),
            train_steps=int(config.slow_pc_train_steps),
            eval_steps=int(config.slow_pc_eval_steps),
            layer_dims=config.slow_pc_layer_dims,
        )
        slow_pc_result = run_digits_pc_experiment(slow_pc_config)
        rows.append(_slow_pc_row(run_index, run_dir, seed, slow_pc_result, config.slow_pc_method_name))

    csv_rows = [
        {
            **row,
            "selected_epoch_passes_gate": "" if row["selected_epoch_passes_gate"] is None else str(bool(row["selected_epoch_passes_gate"])),
            "selector_fallback_used": "" if row["selector_fallback_used"] is None else str(bool(row["selector_fallback_used"])),
            "gate_passing_epoch_count": "" if row["gate_passing_epoch_count"] is None else int(row["gate_passing_epoch_count"]),
            "val_transported_final_energy": "" if row["val_transported_final_energy"] is None else float(row["val_transported_final_energy"]),
            "uses_transport_only_training": str(bool(row["uses_transport_only_training"])),
            "uses_slow_iterative_pc_inference": str(bool(row["uses_slow_iterative_pc_inference"])),
        }
        for row in rows
    ]
    _write_csv(run_dir / "aggregate_runs.csv", csv_rows)

    method_names = [*config.tf2_preset_names, config.slow_pc_method_name]
    by_method = {method_name: _method_summary(_method_rows(rows, method_name)) for method_name in method_names}
    adopted_method = "tf2_corrective_transport_terminal_angleclip_default"
    historical_method = "tf2_corrective_transport_default"
    canonical_method = "tf2_canonical"
    slow_method = config.slow_pc_method_name

    adopted_summary = by_method[adopted_method]
    historical_summary = by_method[historical_method]
    canonical_summary = by_method[canonical_method]
    slow_summary = by_method[slow_method]
    best_pre_adoption_method = _best_pre_adoption_tf2_method(by_method)
    best_pre_adoption_summary = by_method[best_pre_adoption_method]

    adopted_vs_historical = _pairwise_difference(rows, candidate_method=adopted_method, reference_method=historical_method)
    adopted_vs_canonical = _pairwise_difference(rows, candidate_method=adopted_method, reference_method=canonical_method)
    adopted_vs_slow = _pairwise_difference(rows, candidate_method=adopted_method, reference_method=slow_method)
    adopted_vs_best_pre_adoption = _pairwise_difference(
        rows,
        candidate_method=adopted_method,
        reference_method=best_pre_adoption_method,
    )

    adopted_test_gap_to_slow = float(adopted_summary["mean_test_accuracy"]) - float(slow_summary["mean_test_accuracy"])
    historical_test_gap_to_slow = float(historical_summary["mean_test_accuracy"]) - float(slow_summary["mean_test_accuracy"])
    canonical_test_gap_to_slow = float(canonical_summary["mean_test_accuracy"]) - float(slow_summary["mean_test_accuracy"])
    best_pre_adoption_test_gap_to_slow = float(best_pre_adoption_summary["mean_test_accuracy"]) - float(
        slow_summary["mean_test_accuracy"]
    )
    adopted_gap_narrowing_vs_best_pre_adoption = float(adopted_test_gap_to_slow - best_pre_adoption_test_gap_to_slow)
    materially_narrows_gap = bool(
        adopted_gap_narrowing_vs_best_pre_adoption >= float(config.material_gap_narrowing_threshold)
    )

    historical_reference_status = (
        "now_mainly_historical"
        if float(adopted_vs_historical["mean_test_accuracy_delta"]) >= float(config.material_gap_narrowing_threshold)
        else "still_informative_working_reference"
    )
    tf2_canonical_status = (
        "clearly_subordinate_in_current_phase"
        if float(adopted_vs_canonical["mean_test_accuracy_delta"]) >= float(config.material_gap_narrowing_threshold)
        else "still_active_hypothesis_candidate"
    )
    recommended_next_stage = _recommended_next_stage(adopted_summary, slow_summary, config)

    summary = {
        "phase": "Phase TF2",
        "stage": "external_comparison_gap_closure",
        "num_runs": int(len(rows)),
        "baseline_artifact_audit": _local_baseline_audit(),
        "mean_std_val_accuracy_by_method": {
            method_name: {
                "mean": float(method_summary["mean_val_accuracy"]),
                "std": float(method_summary["std_val_accuracy"]),
            }
            for method_name, method_summary in by_method.items()
        },
        "mean_std_test_accuracy_by_method": {
            method_name: {
                "mean": float(method_summary["mean_test_accuracy"]),
                "std": float(method_summary["std_test_accuracy"]),
            }
            for method_name, method_summary in by_method.items()
        },
        "mean_gate_passing_epoch_count_by_tf2_method": {
            method_name: by_method[method_name]["mean_gate_passing_epoch_count"]
            for method_name in config.tf2_preset_names
        },
        "mean_selected_epoch_by_method": {
            method_name: float(method_summary["mean_selected_epoch"])
            for method_name, method_summary in by_method.items()
        },
        "selected_epoch_passes_gate_rate_by_tf2_method": {
            method_name: by_method[method_name]["selected_epoch_passes_gate_rate"]
            for method_name in config.tf2_preset_names
        },
        "selector_fallback_used_rate_by_tf2_method": {
            method_name: by_method[method_name]["selector_fallback_used_rate"]
            for method_name in config.tf2_preset_names
        },
        "seed_gate_positive_rate_by_tf2_method": {
            method_name: by_method[method_name]["seed_gate_positive_rate"]
            for method_name in config.tf2_preset_names
        },
        "mean_val_transported_final_energy_by_tf2_method": {
            method_name: by_method[method_name]["mean_val_transported_final_energy"]
            for method_name in config.tf2_preset_names
        },
        "mean_runtime_proxy_seconds_by_method": {
            method_name: float(method_summary["mean_runtime_proxy_seconds"])
            for method_name, method_summary in by_method.items()
        },
        "by_method": by_method,
        "pairwise_adopted_default_vs_historical_corrective_reference": adopted_vs_historical,
        "pairwise_adopted_default_vs_tf2_canonical": adopted_vs_canonical,
        "pairwise_adopted_default_vs_canonical_slow_pc_digits_baseline": adopted_vs_slow,
        "best_pre_adoption_tf2_comparator": best_pre_adoption_method,
        "adopted_default_gap_to_slow_pc_baseline": {
            "mean_test_accuracy_gap": float(adopted_test_gap_to_slow),
            "mean_val_accuracy_gap": float(adopted_summary["mean_val_accuracy"]) - float(slow_summary["mean_val_accuracy"]),
        },
        "historical_corrective_reference_gap_to_slow_pc_baseline": {
            "mean_test_accuracy_gap": float(historical_test_gap_to_slow),
            "mean_val_accuracy_gap": float(historical_summary["mean_val_accuracy"]) - float(slow_summary["mean_val_accuracy"]),
        },
        "tf2_canonical_gap_to_slow_pc_baseline": {
            "mean_test_accuracy_gap": float(canonical_test_gap_to_slow),
            "mean_val_accuracy_gap": float(canonical_summary["mean_val_accuracy"]) - float(slow_summary["mean_val_accuracy"]),
        },
        "adopted_default_gap_narrowing_vs_best_pre_adoption_tf2": float(adopted_gap_narrowing_vs_best_pre_adoption),
        "adopted_default_materially_narrows_slow_pc_gap": materially_narrows_gap,
        "historical_corrective_reference_status": historical_reference_status,
        "tf2_canonical_status": tf2_canonical_status,
        "recommended_next_stage": recommended_next_stage,
        "aggregate_csv_path": "aggregate_runs.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)

    return FMPCTF2ExternalComparisonSuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config),
        aggregate_rows=rows,
        summary=summary,
    )
