from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..datasets import load_digits_split
from ..energy import compute_cache
from ..tf1.fmpc_tf1_flow import (
    build_tf1_context,
    hidden_energy_from_state,
    hidden_states_from_state,
    rollout_hidden_transport,
)
from .fmpc_tf2 import (
    FMPCTF2Config,
    TF2PresetName,
    _learned_velocity_fn,
    build_tf2_preset_config,
    run_fmpc_tf2_experiment,
)
from ..inference import run_teacher_inference_export
from ..metrics import hidden_state_rms_gap, regression_mse
from ..real_pc import RealPCConfig, run_digits_pc_experiment


@dataclass
class FMPCTF2GapDecompositionSuiteConfig:
    """Run a narrow adopted-package vs slow-PC gap-decomposition pass."""

    experiment_name: str = "fmpc_tf2_gap_decomposition_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    adopted_preset_name: TF2PresetName = "tf2_corrective_transport_terminal_angleclip_default"
    include_historical_corrective_reference: bool = True
    historical_preset_name: TF2PresetName = "tf2_corrective_transport_default"
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
    selector_effect_rate_threshold: float = 0.25
    diagnosis_ratio_threshold: float = 1.25

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"


@dataclass
class FMPCTF2GapDecompositionSuiteRunResult:
    run_dir: Path
    config: dict[str, Any]
    aggregate_rows: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass(frozen=True)
class _SupervisedAlignmentMetrics:
    transport_output_mse: float
    internal_slow_pc_output_mse: float
    transport_supervised_final_energy: float
    internal_slow_pc_supervised_final_energy: float
    endpoint_hidden_state_rms_gap_to_internal_slow_pc: float
    endpoint_output_state_rms_gap_to_internal_slow_pc: float
    hidden_state_rms_gap_by_knot: list[float]
    output_state_rms_gap_by_knot: list[float]
    hidden_state_rms_gap_increase_from_k0_by_knot: list[float]
    output_state_rms_gap_increase_from_k0_by_knot: list[float]
    peak_hidden_gap_knot_index: int
    peak_output_gap_knot_index: int
    rollout_knot_times: list[float]


@dataclass(frozen=True)
class _SlowPCSupervisedMetrics:
    output_mse: float
    supervised_final_energy: float


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


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _bool_or_none(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    return bool(value)


def _mean_list(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("vectors must contain at least one element.")
    width = len(vectors[0])
    if any(len(vector) != width for vector in vectors):
        raise ValueError("All vectors must share the same width.")
    array = np.asarray(vectors, dtype=np.float64)
    return [float(value) for value in np.mean(array, axis=0)]


def _peak_index(values: list[float]) -> int:
    if not values:
        raise ValueError("values must contain at least one element.")
    return int(np.argmax(np.asarray(values, dtype=np.float64)))


def _local_baseline_audit() -> dict[str, Any]:
    external_summary = Path("outputs/tf2/fmpc_tf2_external_comparison_suite/aggregate_summary.json")
    digits_pc_summary = Path("outputs/digits_pc/summary.json")
    return {
        "external_comparison_summary_exists": bool(external_summary.exists()),
        "digits_pc_summary_exists": bool(digits_pc_summary.exists()),
        "reused_existing_canonical_slow_pc_artifacts": False,
        "reuse_reason": (
            "Existing local summaries did not contain the per-seed supervised slow-PC state/output "
            "artifacts needed for gap decomposition, so the suite reran the minimum required runs."
        ),
    }


def _suite_config_payload(config: FMPCTF2GapDecompositionSuiteConfig) -> dict[str, Any]:
    return {
        "phase": "Phase TF2",
        "stage": "adopted_package_vs_slow_pc_gap_decomposition",
        "adopted_preset_name": config.adopted_preset_name,
        "include_historical_corrective_reference": bool(config.include_historical_corrective_reference),
        "historical_preset_name": config.historical_preset_name,
        "slow_pc_method_name": config.slow_pc_method_name,
        "seeds": [int(seed) for seed in config.seeds],
        "tf2_fixed": {
            "feature_aware_tangents": False,
            "selector_policy": "gate_constrained_accuracy_then_val_accuracy",
            "validation_only_selection": True,
            "test_report_only": True,
        },
        "diagnostics": {
            "official_external_metrics": [
                "val_accuracy",
                "test_accuracy",
                "selected_epoch",
                "gate_passing_epoch_count",
                "selector_fallback_used_rate",
                "runtime_proxy_seconds",
            ],
            "internal_supervised_gap_metrics": [
                "transport_output_mse",
                "internal_slow_pc_output_mse",
                "transport_supervised_final_energy",
                "internal_slow_pc_supervised_final_energy",
                "endpoint_hidden_state_rms_gap_to_internal_slow_pc",
                "endpoint_output_state_rms_gap_to_internal_slow_pc",
            ],
            "adopted_package_validation_knot_breakdown": True,
        },
        "baseline_artifact_audit": _local_baseline_audit(),
        "thresholds": {
            "selector_effect_rate_threshold": float(config.selector_effect_rate_threshold),
            "diagnosis_ratio_threshold": float(config.diagnosis_ratio_threshold),
        },
    }


def _tf2_run_id(preset_name: TF2PresetName, seed: int) -> str:
    short_name_map = {
        "tf2_corrective_transport_terminal_angleclip_default": "adopted",
        "tf2_corrective_transport_default": "historical",
        "tf2_canonical": "canonical",
    }
    return f"{short_name_map[str(preset_name)]}_s{seed}"


def _slow_pc_run_id(seed: int) -> str:
    return f"slowpc_s{seed}"


def _load_digits_split_for_tf2_config(config: FMPCTF2Config) -> Any:
    return load_digits_split(
        train_fraction=float(config.train_fraction),
        val_fraction=float(config.val_fraction),
        test_fraction=float(config.test_fraction),
        split_seed=int(config.data_seed),
    )


def _predicted_outputs_from_hidden_state(context: Any, z: np.ndarray) -> np.ndarray:
    states = hidden_states_from_state(context, z)
    cache = compute_cache(states, context.layers)
    prediction = cache.predictions[-1]
    if prediction is None:
        raise ValueError("Final-layer prediction is missing from the cache.")
    return np.asarray(prediction, dtype=np.float64)


def _teacher_supervised_outputs(model: Any, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    teacher_export = run_teacher_inference_export(
        model.layers,
        x,
        y=y,
        init=model.state_init,
        mode="train",
        eta_x=model.eta_x,
        steps=int(model.eval_steps),
        backend=str(model.inference_backend),
        record_trace=False,
        record_trajectory=False,
    )
    cache = compute_cache(teacher_export.final_states, model.layers)
    predictions = cache.predictions[-1]
    if predictions is None:
        raise ValueError("Final-layer prediction is missing from the teacher cache.")
    context = build_tf1_context(model, x, y)
    return (
        np.asarray(teacher_export.z_star, dtype=np.float64),
        np.asarray(predictions, dtype=np.float64),
        float(hidden_energy_from_state(context, teacher_export.z_star)),
    )


def _evaluate_tf2_supervised_alignment(
    model: Any,
    psi_network: Any,
    config: FMPCTF2Config,
    x: np.ndarray,
    y: np.ndarray,
) -> _SupervisedAlignmentMetrics:
    context = build_tf1_context(model, x, y)
    velocity_fn = _learned_velocity_fn(context, psi_network, config)
    rollout = rollout_hidden_transport(
        context,
        context.z0,
        transport_steps=int(config.micro_steps),
        mode="learned",
        velocity_fn=velocity_fn,
    )

    teacher_z_star, teacher_predictions, teacher_energy = _teacher_supervised_outputs(model, x, y)
    transport_predictions = _predicted_outputs_from_hidden_state(context, rollout.z_knots[-1])

    hidden_gaps = [hidden_state_rms_gap(z_knot, teacher_z_star) for z_knot in rollout.z_knots]
    output_gaps = [
        hidden_state_rms_gap(_predicted_outputs_from_hidden_state(context, z_knot), teacher_predictions)
        for z_knot in rollout.z_knots
    ]
    hidden_gap_k0 = float(hidden_gaps[0])
    output_gap_k0 = float(output_gaps[0])
    hidden_gap_increase = [float(value - hidden_gap_k0) for value in hidden_gaps]
    output_gap_increase = [float(value - output_gap_k0) for value in output_gaps]

    return _SupervisedAlignmentMetrics(
        transport_output_mse=regression_mse(transport_predictions, y),
        internal_slow_pc_output_mse=regression_mse(teacher_predictions, y),
        transport_supervised_final_energy=float(rollout.final_energy),
        internal_slow_pc_supervised_final_energy=float(teacher_energy),
        endpoint_hidden_state_rms_gap_to_internal_slow_pc=float(hidden_gaps[-1]),
        endpoint_output_state_rms_gap_to_internal_slow_pc=float(output_gaps[-1]),
        hidden_state_rms_gap_by_knot=[float(value) for value in hidden_gaps],
        output_state_rms_gap_by_knot=[float(value) for value in output_gaps],
        hidden_state_rms_gap_increase_from_k0_by_knot=hidden_gap_increase,
        output_state_rms_gap_increase_from_k0_by_knot=output_gap_increase,
        peak_hidden_gap_knot_index=_peak_index(hidden_gaps),
        peak_output_gap_knot_index=_peak_index(output_gaps),
        rollout_knot_times=[float(value) for value in rollout.knot_times],
    )


def _evaluate_slow_pc_supervised_metrics(model: Any, x: np.ndarray, y: np.ndarray) -> _SlowPCSupervisedMetrics:
    _, teacher_predictions, teacher_energy = _teacher_supervised_outputs(model, x, y)
    return _SlowPCSupervisedMetrics(
        output_mse=regression_mse(teacher_predictions, y),
        supervised_final_energy=float(teacher_energy),
    )


def _tf2_row(
    run_index: int,
    run_dir: Path,
    preset_name: TF2PresetName,
    seed: int,
    result: Any,
    val_alignment: _SupervisedAlignmentMetrics,
    test_alignment: _SupervisedAlignmentMetrics,
) -> dict[str, Any]:
    summary = result.summary
    timing = dict(summary.get("timing", {}))
    return {
        "run_index": int(run_index),
        "method_name": str(preset_name),
        "seed": int(seed),
        "run_id": str(result.run_dir.name),
        "run_summary_path": _relative_posix(run_dir, result.run_dir / "summary.json"),
        "is_tf2_method": True,
        "val_accuracy": float(summary["val_accuracy"]),
        "test_accuracy": float(summary["test_accuracy"]),
        "selected_epoch": int(summary["best_epoch"]),
        "gate_passing_epoch_count": int(summary["gate_passing_epoch_count"]),
        "selected_epoch_passes_gate": bool(summary["selected_epoch_passes_gate"]),
        "selector_fallback_used": bool(summary["selector_fallback_used"]),
        "val_report_output_mse": float(summary["val_loss"]),
        "test_report_output_mse": float(summary["test_loss"]),
        "val_transported_final_energy": float(summary["val_transported_final_energy"]),
        "test_transported_final_energy": float(summary["test_transported_final_energy"]),
        "val_supervised_output_mse": float(val_alignment.transport_output_mse),
        "test_supervised_output_mse": float(test_alignment.transport_output_mse),
        "val_internal_slow_pc_output_mse": float(val_alignment.internal_slow_pc_output_mse),
        "test_internal_slow_pc_output_mse": float(test_alignment.internal_slow_pc_output_mse),
        "val_internal_slow_pc_supervised_final_energy": float(val_alignment.internal_slow_pc_supervised_final_energy),
        "test_internal_slow_pc_supervised_final_energy": float(test_alignment.internal_slow_pc_supervised_final_energy),
        "val_endpoint_hidden_state_rms_gap_to_internal_slow_pc": float(
            val_alignment.endpoint_hidden_state_rms_gap_to_internal_slow_pc
        ),
        "test_endpoint_hidden_state_rms_gap_to_internal_slow_pc": float(
            test_alignment.endpoint_hidden_state_rms_gap_to_internal_slow_pc
        ),
        "val_endpoint_output_state_rms_gap_to_internal_slow_pc": float(
            val_alignment.endpoint_output_state_rms_gap_to_internal_slow_pc
        ),
        "test_endpoint_output_state_rms_gap_to_internal_slow_pc": float(
            test_alignment.endpoint_output_state_rms_gap_to_internal_slow_pc
        ),
        "val_transport_minus_internal_slow_pc_output_mse": float(
            val_alignment.transport_output_mse - val_alignment.internal_slow_pc_output_mse
        ),
        "test_transport_minus_internal_slow_pc_output_mse": float(
            test_alignment.transport_output_mse - test_alignment.internal_slow_pc_output_mse
        ),
        "val_transport_minus_internal_slow_pc_supervised_final_energy": float(
            val_alignment.transport_supervised_final_energy - val_alignment.internal_slow_pc_supervised_final_energy
        ),
        "test_transport_minus_internal_slow_pc_supervised_final_energy": float(
            test_alignment.transport_supervised_final_energy - test_alignment.internal_slow_pc_supervised_final_energy
        ),
        "runtime_proxy_seconds": float(timing.get("train_wall_time_seconds", 0.0))
        + float(timing.get("final_evaluation_wall_time_seconds", 0.0)),
    }


def _slow_pc_row(
    run_index: int,
    run_dir: Path,
    method_name: str,
    seed: int,
    result: Any,
    val_supervised: _SlowPCSupervisedMetrics,
    test_supervised: _SlowPCSupervisedMetrics,
) -> dict[str, Any]:
    summary = result.summary
    timing = dict(summary.get("timing", {}))
    return {
        "run_index": int(run_index),
        "method_name": str(method_name),
        "seed": int(seed),
        "run_id": str(result.run_dir.name),
        "run_summary_path": _relative_posix(run_dir, result.run_dir / "summary.json"),
        "is_tf2_method": False,
        "val_accuracy": float(summary["val_metric"]),
        "test_accuracy": float(summary["test_metric"]),
        "selected_epoch": int(summary["best_epoch"]),
        "gate_passing_epoch_count": "",
        "selected_epoch_passes_gate": "",
        "selector_fallback_used": "",
        "val_report_output_mse": float(summary["val_loss"]),
        "test_report_output_mse": float(summary["test_loss"]),
        "val_transported_final_energy": "",
        "test_transported_final_energy": "",
        "val_supervised_output_mse": float(val_supervised.output_mse),
        "test_supervised_output_mse": float(test_supervised.output_mse),
        "val_internal_slow_pc_output_mse": float(val_supervised.output_mse),
        "test_internal_slow_pc_output_mse": float(test_supervised.output_mse),
        "val_internal_slow_pc_supervised_final_energy": float(val_supervised.supervised_final_energy),
        "test_internal_slow_pc_supervised_final_energy": float(test_supervised.supervised_final_energy),
        "val_endpoint_hidden_state_rms_gap_to_internal_slow_pc": 0.0,
        "test_endpoint_hidden_state_rms_gap_to_internal_slow_pc": 0.0,
        "val_endpoint_output_state_rms_gap_to_internal_slow_pc": 0.0,
        "test_endpoint_output_state_rms_gap_to_internal_slow_pc": 0.0,
        "val_transport_minus_internal_slow_pc_output_mse": "",
        "test_transport_minus_internal_slow_pc_output_mse": "",
        "val_transport_minus_internal_slow_pc_supervised_final_energy": "",
        "test_transport_minus_internal_slow_pc_supervised_final_energy": "",
        "runtime_proxy_seconds": float(timing.get("train_wall_time_seconds", 0.0))
        + float(timing.get("final_evaluation_wall_time_seconds", 0.0)),
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
    report_val_output_mse = [float(row["val_report_output_mse"]) for row in rows]
    report_test_output_mse = [float(row["test_report_output_mse"]) for row in rows]
    supervised_val_output_mse = [float(row["val_supervised_output_mse"]) for row in rows]
    supervised_test_output_mse = [float(row["test_supervised_output_mse"]) for row in rows]
    internal_val_output_mse = [float(row["val_internal_slow_pc_output_mse"]) for row in rows]
    internal_test_output_mse = [float(row["test_internal_slow_pc_output_mse"]) for row in rows]
    internal_val_energy = [float(row["val_internal_slow_pc_supervised_final_energy"]) for row in rows]
    internal_test_energy = [float(row["test_internal_slow_pc_supervised_final_energy"]) for row in rows]
    endpoint_hidden_gap_val = [
        float(row["val_endpoint_hidden_state_rms_gap_to_internal_slow_pc"]) for row in rows
    ]
    endpoint_hidden_gap_test = [
        float(row["test_endpoint_hidden_state_rms_gap_to_internal_slow_pc"]) for row in rows
    ]
    endpoint_output_gap_val = [
        float(row["val_endpoint_output_state_rms_gap_to_internal_slow_pc"]) for row in rows
    ]
    endpoint_output_gap_test = [
        float(row["test_endpoint_output_state_rms_gap_to_internal_slow_pc"]) for row in rows
    ]
    gate_values = [
        float(value)
        for value in (_float_or_none(row["gate_passing_epoch_count"]) for row in rows)
        if value is not None
    ]
    gate_pass_values = [
        bool(value)
        for value in (_bool_or_none(row["selected_epoch_passes_gate"]) for row in rows)
        if value is not None
    ]
    selector_fallback_values = [
        bool(value)
        for value in (_bool_or_none(row["selector_fallback_used"]) for row in rows)
        if value is not None
    ]
    transported_val_energy = [
        float(value)
        for value in (_float_or_none(row["val_transported_final_energy"]) for row in rows)
        if value is not None
    ]
    transported_test_energy = [
        float(value)
        for value in (_float_or_none(row["test_transported_final_energy"]) for row in rows)
        if value is not None
    ]
    transport_output_gap_val = [
        float(value)
        for value in (_float_or_none(row["val_transport_minus_internal_slow_pc_output_mse"]) for row in rows)
        if value is not None
    ]
    transport_output_gap_test = [
        float(value)
        for value in (_float_or_none(row["test_transport_minus_internal_slow_pc_output_mse"]) for row in rows)
        if value is not None
    ]
    transport_energy_gap_val = [
        float(value)
        for value in (
            _float_or_none(row["val_transport_minus_internal_slow_pc_supervised_final_energy"]) for row in rows
        )
        if value is not None
    ]
    transport_energy_gap_test = [
        float(value)
        for value in (
            _float_or_none(row["test_transport_minus_internal_slow_pc_supervised_final_energy"]) for row in rows
        )
        if value is not None
    ]

    return {
        "num_runs": len(rows),
        "mean_val_accuracy": _mean(val_accuracies),
        "std_val_accuracy": _std(val_accuracies),
        "mean_test_accuracy": _mean(test_accuracies),
        "std_test_accuracy": _std(test_accuracies),
        "mean_selected_epoch": _mean(selected_epochs),
        "mean_gate_passing_epoch_count": None if not gate_values else _mean(gate_values),
        "selected_epoch_passes_gate_rate": None if not gate_pass_values else _rate(gate_pass_values),
        "selector_fallback_used_rate": None if not selector_fallback_values else _rate(selector_fallback_values),
        "seed_gate_positive_rate": None if not gate_values else _rate([value > 0.0 for value in gate_values]),
        "mean_val_report_output_mse": _mean(report_val_output_mse),
        "mean_test_report_output_mse": _mean(report_test_output_mse),
        "mean_val_supervised_output_mse": _mean(supervised_val_output_mse),
        "mean_test_supervised_output_mse": _mean(supervised_test_output_mse),
        "mean_val_internal_slow_pc_output_mse": _mean(internal_val_output_mse),
        "mean_test_internal_slow_pc_output_mse": _mean(internal_test_output_mse),
        "mean_val_internal_slow_pc_supervised_final_energy": _mean(internal_val_energy),
        "mean_test_internal_slow_pc_supervised_final_energy": _mean(internal_test_energy),
        "mean_val_endpoint_hidden_state_rms_gap_to_internal_slow_pc": _mean(endpoint_hidden_gap_val),
        "mean_test_endpoint_hidden_state_rms_gap_to_internal_slow_pc": _mean(endpoint_hidden_gap_test),
        "mean_val_endpoint_output_state_rms_gap_to_internal_slow_pc": _mean(endpoint_output_gap_val),
        "mean_test_endpoint_output_state_rms_gap_to_internal_slow_pc": _mean(endpoint_output_gap_test),
        "mean_val_transported_final_energy": None if not transported_val_energy else _mean(transported_val_energy),
        "mean_test_transported_final_energy": None if not transported_test_energy else _mean(transported_test_energy),
        "mean_val_transport_minus_internal_slow_pc_output_mse": (
            None if not transport_output_gap_val else _mean(transport_output_gap_val)
        ),
        "mean_test_transport_minus_internal_slow_pc_output_mse": (
            None if not transport_output_gap_test else _mean(transport_output_gap_test)
        ),
        "mean_val_transport_minus_internal_slow_pc_supervised_final_energy": (
            None if not transport_energy_gap_val else _mean(transport_energy_gap_val)
        ),
        "mean_test_transport_minus_internal_slow_pc_supervised_final_energy": (
            None if not transport_energy_gap_test else _mean(transport_energy_gap_test)
        ),
        "mean_runtime_proxy_seconds": _mean(runtime_values),
        "std_runtime_proxy_seconds": _std(runtime_values),
    }


def _pairwise_delta(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]) -> dict[str, Any]:
    left_by_seed = {int(row["seed"]): row for row in left_rows}
    right_by_seed = {int(row["seed"]): row for row in right_rows}
    shared_seeds = sorted(set(left_by_seed) & set(right_by_seed))
    if not shared_seeds:
        raise ValueError("Pairwise comparison requires at least one shared seed.")

    def metric_deltas(metric_name: str) -> list[float]:
        values: list[float] = []
        for seed in shared_seeds:
            left_value = _float_or_none(left_by_seed[seed][metric_name])
            right_value = _float_or_none(right_by_seed[seed][metric_name])
            if left_value is None or right_value is None:
                continue
            values.append(float(left_value - right_value))
        return values

    def rate_delta(metric_name: str) -> float | None:
        left_values = [
            bool(value)
            for value in (_bool_or_none(left_by_seed[seed][metric_name]) for seed in shared_seeds)
            if value is not None
        ]
        right_values = [
            bool(value)
            for value in (_bool_or_none(right_by_seed[seed][metric_name]) for seed in shared_seeds)
            if value is not None
        ]
        if not left_values or not right_values:
            return None
        return float(_rate(left_values) - _rate(right_values))

    def mean_or_none(values: list[float]) -> float | None:
        return None if not values else _mean(values)

    return {
        "shared_seeds": [int(seed) for seed in shared_seeds],
        "mean_val_accuracy_delta": mean_or_none(metric_deltas("val_accuracy")),
        "mean_test_accuracy_delta": mean_or_none(metric_deltas("test_accuracy")),
        "mean_selected_epoch_delta": mean_or_none(metric_deltas("selected_epoch")),
        "mean_gate_passing_epoch_count_delta": mean_or_none(metric_deltas("gate_passing_epoch_count")),
        "selected_epoch_passes_gate_rate_delta": rate_delta("selected_epoch_passes_gate"),
        "selector_fallback_used_rate_delta": rate_delta("selector_fallback_used"),
        "mean_val_report_output_mse_delta": mean_or_none(metric_deltas("val_report_output_mse")),
        "mean_test_report_output_mse_delta": mean_or_none(metric_deltas("test_report_output_mse")),
        "mean_val_supervised_output_mse_delta": mean_or_none(metric_deltas("val_supervised_output_mse")),
        "mean_test_supervised_output_mse_delta": mean_or_none(metric_deltas("test_supervised_output_mse")),
        "mean_val_internal_slow_pc_supervised_final_energy_delta": mean_or_none(
            metric_deltas("val_internal_slow_pc_supervised_final_energy")
        ),
        "mean_test_internal_slow_pc_supervised_final_energy_delta": mean_or_none(
            metric_deltas("test_internal_slow_pc_supervised_final_energy")
        ),
        "mean_val_endpoint_hidden_state_rms_gap_delta": mean_or_none(
            metric_deltas("val_endpoint_hidden_state_rms_gap_to_internal_slow_pc")
        ),
        "mean_test_endpoint_hidden_state_rms_gap_delta": mean_or_none(
            metric_deltas("test_endpoint_hidden_state_rms_gap_to_internal_slow_pc")
        ),
        "mean_val_endpoint_output_state_rms_gap_delta": mean_or_none(
            metric_deltas("val_endpoint_output_state_rms_gap_to_internal_slow_pc")
        ),
        "mean_test_endpoint_output_state_rms_gap_delta": mean_or_none(
            metric_deltas("test_endpoint_output_state_rms_gap_to_internal_slow_pc")
        ),
        "mean_val_transport_minus_internal_slow_pc_output_mse_delta": mean_or_none(
            metric_deltas("val_transport_minus_internal_slow_pc_output_mse")
        ),
        "mean_test_transport_minus_internal_slow_pc_output_mse_delta": mean_or_none(
            metric_deltas("test_transport_minus_internal_slow_pc_output_mse")
        ),
        "mean_val_transported_final_energy_delta": mean_or_none(metric_deltas("val_transported_final_energy")),
        "mean_runtime_proxy_seconds_delta": mean_or_none(metric_deltas("runtime_proxy_seconds")),
    }


def _knot_region_from_peak_index(peak_index: int, micro_steps: int) -> str:
    if peak_index <= 1:
        return "early_rollout"
    if peak_index >= micro_steps:
        return "terminal_rollout"
    if peak_index >= micro_steps - 1:
        return "late_rollout"
    return "mid_rollout"


def _diagnose_gap(
    config: FMPCTF2GapDecompositionSuiteConfig,
    adopted_summary: dict[str, Any],
    slow_pc_summary: dict[str, Any],
    knot_breakdown: dict[str, Any],
) -> tuple[str, str]:
    selector_fallback_rate = adopted_summary["selector_fallback_used_rate"]
    if selector_fallback_rate is not None and float(selector_fallback_rate) > float(config.selector_effect_rate_threshold):
        return (
            "selector_checkpoint_effects",
            "run one narrow selector-robustness confirmation inside the adopted package without changing selector rules",
        )

    adopted_transport_penalty = float(adopted_summary["mean_val_transport_minus_internal_slow_pc_output_mse"] or 0.0)
    adopted_model_gap = float(
        adopted_summary["mean_val_internal_slow_pc_output_mse"]
        - slow_pc_summary["mean_val_internal_slow_pc_output_mse"]
    )
    peak_region = str(knot_breakdown["dominant_deviation_region"])

    if adopted_transport_penalty > float(config.diagnosis_ratio_threshold) * max(1e-12, adopted_model_gap):
        if peak_region in {"late_rollout", "terminal_rollout"}:
            return (
                "hidden_state_transport_quality",
                "run one narrow adopted-package preterminal local-field direction trust-region pass on the penultimate micro-step",
            )
        return (
            "hidden_state_transport_quality",
            "run one narrow adopted-package rollout transport-quality pass focused on the largest validation-knot deviation",
        )

    if adopted_model_gap > float(config.diagnosis_ratio_threshold) * max(1e-12, adopted_transport_penalty):
        return (
            "output_readout_mismatch",
            "run one narrow adopted-package readout-alignment confirmation without changing the TF2 transport family",
        )

    return (
        "mixed_picture_hidden_transport_plus_readout_gap",
        "run one narrow adopted-package preterminal local-field direction trust-region pass on the penultimate micro-step",
    )


def run_fmpc_tf2_gap_decomposition_suite(
    config: FMPCTF2GapDecompositionSuiteConfig,
) -> FMPCTF2GapDecompositionSuiteRunResult:
    run_dir = _prepare_run_dir(
        _resolve_run_dir(
            config.output_root,
            config.experiment_name,
            config.resolved_run_id(),
            config.output_layout,
        )
    )
    _write_json(run_dir / "config.json", _suite_config_payload(config))

    aggregate_rows: list[dict[str, Any]] = []
    adopted_val_knot_hidden_gaps: list[list[float]] = []
    adopted_val_knot_output_gaps: list[list[float]] = []
    adopted_val_knot_hidden_gap_increase: list[list[float]] = []
    adopted_val_knot_output_gap_increase: list[list[float]] = []
    adopted_val_peak_hidden_indices: list[float] = []
    adopted_val_peak_output_indices: list[float] = []
    adopted_rollout_knot_times: list[list[float]] = []

    tf2_root = run_dir / "tf2_runs"
    slow_pc_root = run_dir / "slow_pc_runs"
    tf2_root.mkdir(parents=True, exist_ok=True)
    slow_pc_root.mkdir(parents=True, exist_ok=True)

    tf2_preset_names: list[TF2PresetName] = [config.adopted_preset_name]
    if config.include_historical_corrective_reference:
        tf2_preset_names.append(config.historical_preset_name)

    run_index = 0
    for preset_name in tf2_preset_names:
        for seed in config.seeds:
            tf2_config = build_tf2_preset_config(
                preset_name,
                experiment_name="tf2",
                output_root=tf2_root,
                output_layout="run_id_subdir",
                run_id=_tf2_run_id(preset_name, int(seed)),
                run_seed=int(seed),
                data_seed=int(seed),
                model_init_seed=int(seed),
                psi_init_seed=int(seed),
                batch_order_seed=int(seed),
                epochs=int(config.tf2_epochs),
                batch_size=int(config.tf2_batch_size),
                eval_steps=int(config.tf2_eval_steps),
                layer_dims=tuple(config.tf2_layer_dims),
            )
            result = run_fmpc_tf2_experiment(tf2_config)
            if result.model is None or result.psi_network is None:
                raise ValueError("TF2 suite requires runtime model and psi network objects.")
            split = _load_digits_split_for_tf2_config(tf2_config)
            val_alignment = _evaluate_tf2_supervised_alignment(
                result.model,
                result.psi_network,
                tf2_config,
                split.x_val,
                split.y_val,
            )
            test_alignment = _evaluate_tf2_supervised_alignment(
                result.model,
                result.psi_network,
                tf2_config,
                split.x_test,
                split.y_test,
            )
            aggregate_rows.append(
                _tf2_row(
                    run_index=run_index,
                    run_dir=run_dir,
                    preset_name=preset_name,
                    seed=int(seed),
                    result=result,
                    val_alignment=val_alignment,
                    test_alignment=test_alignment,
                )
            )
            if preset_name == config.adopted_preset_name:
                adopted_val_knot_hidden_gaps.append(val_alignment.hidden_state_rms_gap_by_knot)
                adopted_val_knot_output_gaps.append(val_alignment.output_state_rms_gap_by_knot)
                adopted_val_knot_hidden_gap_increase.append(
                    val_alignment.hidden_state_rms_gap_increase_from_k0_by_knot
                )
                adopted_val_knot_output_gap_increase.append(
                    val_alignment.output_state_rms_gap_increase_from_k0_by_knot
                )
                adopted_val_peak_hidden_indices.append(float(val_alignment.peak_hidden_gap_knot_index))
                adopted_val_peak_output_indices.append(float(val_alignment.peak_output_gap_knot_index))
                adopted_rollout_knot_times.append(val_alignment.rollout_knot_times)
            run_index += 1

    for seed in config.seeds:
        slow_pc_config = RealPCConfig(
            experiment_name="pc",
            output_root=slow_pc_root,
            output_layout="run_id_subdir",
            run_id=_slow_pc_run_id(int(seed)),
            run_seed=int(seed),
            data_seed=int(seed),
            model_init_seed=int(seed),
            batch_order_seed=int(seed),
            epochs=int(config.slow_pc_epochs),
            batch_size=int(config.slow_pc_batch_size),
            train_steps=int(config.slow_pc_train_steps),
            eval_steps=int(config.slow_pc_eval_steps),
            layer_dims=tuple(config.slow_pc_layer_dims),
        )
        result = run_digits_pc_experiment(slow_pc_config, return_runtime_objects=True)
        if result.model is None or result.split is None:
            raise ValueError("Slow-PC suite requires runtime model and split objects.")
        val_supervised = _evaluate_slow_pc_supervised_metrics(result.model, result.split.x_val, result.split.y_val)
        test_supervised = _evaluate_slow_pc_supervised_metrics(result.model, result.split.x_test, result.split.y_test)
        aggregate_rows.append(
            _slow_pc_row(
                run_index=run_index,
                run_dir=run_dir,
                method_name=config.slow_pc_method_name,
                seed=int(seed),
                result=result,
                val_supervised=val_supervised,
                test_supervised=test_supervised,
            )
        )
        run_index += 1

    csv_rows = [
        {
            **row,
            "gate_passing_epoch_count": (
                "" if row["gate_passing_epoch_count"] in (None, "") else int(row["gate_passing_epoch_count"])
            ),
            "selected_epoch_passes_gate": (
                "" if row["selected_epoch_passes_gate"] in (None, "") else bool(row["selected_epoch_passes_gate"])
            ),
            "selector_fallback_used": (
                "" if row["selector_fallback_used"] in (None, "") else bool(row["selector_fallback_used"])
            ),
        }
        for row in aggregate_rows
    ]
    _write_csv(run_dir / "aggregate_runs.csv", csv_rows)

    by_method = {
        method_name: _method_summary(_method_rows(aggregate_rows, method_name))
        for method_name in {str(row["method_name"]) for row in aggregate_rows}
    }
    adopted_rows = _method_rows(aggregate_rows, config.adopted_preset_name)
    slow_pc_rows = _method_rows(aggregate_rows, config.slow_pc_method_name)
    historical_rows = (
        _method_rows(aggregate_rows, config.historical_preset_name)
        if config.include_historical_corrective_reference
        else []
    )
    adopted_summary = by_method[config.adopted_preset_name]
    slow_pc_summary = by_method[config.slow_pc_method_name]

    knot_breakdown = {
        "rollout_knot_indices": list(range(len(adopted_val_knot_hidden_gaps[0]))),
        "rollout_knot_times": _mean_list(adopted_rollout_knot_times),
        "mean_hidden_state_rms_gap_to_internal_slow_pc_by_knot": _mean_list(adopted_val_knot_hidden_gaps),
        "mean_output_state_rms_gap_to_internal_slow_pc_by_knot": _mean_list(adopted_val_knot_output_gaps),
        "mean_hidden_state_rms_gap_increase_from_k0_by_knot": _mean_list(adopted_val_knot_hidden_gap_increase),
        "mean_output_state_rms_gap_increase_from_k0_by_knot": _mean_list(adopted_val_knot_output_gap_increase),
        "mean_peak_hidden_gap_knot_index": _mean(adopted_val_peak_hidden_indices),
        "mean_peak_output_gap_knot_index": _mean(adopted_val_peak_output_indices),
    }
    knot_breakdown["peak_hidden_gap_knot_index"] = _peak_index(
        knot_breakdown["mean_hidden_state_rms_gap_to_internal_slow_pc_by_knot"]
    )
    knot_breakdown["peak_output_gap_knot_index"] = _peak_index(
        knot_breakdown["mean_output_state_rms_gap_to_internal_slow_pc_by_knot"]
    )
    knot_breakdown["dominant_deviation_region"] = _knot_region_from_peak_index(
        int(knot_breakdown["peak_hidden_gap_knot_index"]),
        micro_steps=int(len(knot_breakdown["rollout_knot_indices"]) - 1),
    )

    diagnosis, recommended_next_move = _diagnose_gap(
        config,
        adopted_summary,
        slow_pc_summary,
        knot_breakdown,
    )

    summary: dict[str, Any] = {
        "phase": "Phase TF2",
        "stage": "adopted_package_vs_slow_pc_gap_decomposition",
        "num_runs": len(aggregate_rows),
        "baseline_artifact_audit": _local_baseline_audit(),
        "mean_std_val_accuracy_by_method": {
            method_name: {
                "mean": by_method[method_name]["mean_val_accuracy"],
                "std": by_method[method_name]["std_val_accuracy"],
            }
            for method_name in by_method
        },
        "mean_std_test_accuracy_by_method": {
            method_name: {
                "mean": by_method[method_name]["mean_test_accuracy"],
                "std": by_method[method_name]["std_test_accuracy"],
            }
            for method_name in by_method
        },
        "mean_selected_epoch_by_method": {
            method_name: by_method[method_name]["mean_selected_epoch"] for method_name in by_method
        },
        "mean_gate_passing_epoch_count_by_tf2_method": {
            method_name: by_method[method_name]["mean_gate_passing_epoch_count"]
            for method_name in by_method
            if by_method[method_name]["mean_gate_passing_epoch_count"] is not None
        },
        "selected_epoch_passes_gate_rate_by_tf2_method": {
            method_name: by_method[method_name]["selected_epoch_passes_gate_rate"]
            for method_name in by_method
            if by_method[method_name]["selected_epoch_passes_gate_rate"] is not None
        },
        "selector_fallback_used_rate_by_tf2_method": {
            method_name: by_method[method_name]["selector_fallback_used_rate"]
            for method_name in by_method
            if by_method[method_name]["selector_fallback_used_rate"] is not None
        },
        "seed_gate_positive_rate_by_tf2_method": {
            method_name: by_method[method_name]["seed_gate_positive_rate"]
            for method_name in by_method
            if by_method[method_name]["seed_gate_positive_rate"] is not None
        },
        "mean_runtime_proxy_seconds_by_method": {
            method_name: by_method[method_name]["mean_runtime_proxy_seconds"] for method_name in by_method
        },
        "mean_report_output_mse_by_method": {
            method_name: {
                "mean_val": by_method[method_name]["mean_val_report_output_mse"],
                "mean_test": by_method[method_name]["mean_test_report_output_mse"],
            }
            for method_name in by_method
        },
        "mean_supervised_output_mse_by_method": {
            method_name: {
                "mean_val": by_method[method_name]["mean_val_supervised_output_mse"],
                "mean_test": by_method[method_name]["mean_test_supervised_output_mse"],
            }
            for method_name in by_method
        },
        "mean_supervised_final_energy_by_method": {
            method_name: {
                "mean_val": by_method[method_name]["mean_val_internal_slow_pc_supervised_final_energy"],
                "mean_test": by_method[method_name]["mean_test_internal_slow_pc_supervised_final_energy"],
            }
            for method_name in by_method
        },
        "mean_val_transported_final_energy_by_tf2_method": {
            method_name: by_method[method_name]["mean_val_transported_final_energy"]
            for method_name in by_method
            if by_method[method_name]["mean_val_transported_final_energy"] is not None
        },
        "mean_endpoint_hidden_state_rms_gap_to_internal_slow_pc_by_method": {
            method_name: {
                "mean_val": by_method[method_name]["mean_val_endpoint_hidden_state_rms_gap_to_internal_slow_pc"],
                "mean_test": by_method[method_name]["mean_test_endpoint_hidden_state_rms_gap_to_internal_slow_pc"],
            }
            for method_name in by_method
        },
        "mean_endpoint_output_state_rms_gap_to_internal_slow_pc_by_method": {
            method_name: {
                "mean_val": by_method[method_name]["mean_val_endpoint_output_state_rms_gap_to_internal_slow_pc"],
                "mean_test": by_method[method_name]["mean_test_endpoint_output_state_rms_gap_to_internal_slow_pc"],
            }
            for method_name in by_method
        },
        "mean_transport_minus_internal_slow_pc_output_mse_by_tf2_method": {
            method_name: {
                "mean_val": by_method[method_name]["mean_val_transport_minus_internal_slow_pc_output_mse"],
                "mean_test": by_method[method_name]["mean_test_transport_minus_internal_slow_pc_output_mse"],
            }
            for method_name in by_method
            if by_method[method_name]["mean_val_transport_minus_internal_slow_pc_output_mse"] is not None
        },
        "mean_transport_minus_internal_slow_pc_supervised_final_energy_by_tf2_method": {
            method_name: {
                "mean_val": by_method[method_name]["mean_val_transport_minus_internal_slow_pc_supervised_final_energy"],
                "mean_test": by_method[method_name]["mean_test_transport_minus_internal_slow_pc_supervised_final_energy"],
            }
            for method_name in by_method
            if by_method[method_name]["mean_val_transport_minus_internal_slow_pc_supervised_final_energy"] is not None
        },
        "by_method": by_method,
        "pairwise_adopted_default_vs_historical_corrective_reference": (
            None if not historical_rows else _pairwise_delta(adopted_rows, historical_rows)
        ),
        "pairwise_adopted_default_vs_canonical_slow_pc_digits_baseline": _pairwise_delta(adopted_rows, slow_pc_rows),
        "adopted_package_validation_knot_breakdown": knot_breakdown,
        "adopted_package_gap_profile": {
            "external_val_accuracy_gap_to_slow_pc_baseline": float(
                adopted_summary["mean_val_accuracy"] - slow_pc_summary["mean_val_accuracy"]
            ),
            "external_test_accuracy_gap_to_slow_pc_baseline": float(
                adopted_summary["mean_test_accuracy"] - slow_pc_summary["mean_test_accuracy"]
            ),
            "val_transport_minus_internal_slow_pc_output_mse": float(
                adopted_summary["mean_val_transport_minus_internal_slow_pc_output_mse"] or 0.0
            ),
            "test_transport_minus_internal_slow_pc_output_mse": float(
                adopted_summary["mean_test_transport_minus_internal_slow_pc_output_mse"] or 0.0
            ),
            "val_internal_slow_pc_output_mse_gap_to_canonical_slow_pc_baseline": float(
                adopted_summary["mean_val_internal_slow_pc_output_mse"]
                - slow_pc_summary["mean_val_internal_slow_pc_output_mse"]
            ),
            "test_internal_slow_pc_output_mse_gap_to_canonical_slow_pc_baseline": float(
                adopted_summary["mean_test_internal_slow_pc_output_mse"]
                - slow_pc_summary["mean_test_internal_slow_pc_output_mse"]
            ),
            "val_transport_minus_internal_slow_pc_supervised_final_energy": float(
                adopted_summary["mean_val_transport_minus_internal_slow_pc_supervised_final_energy"] or 0.0
            ),
            "test_transport_minus_internal_slow_pc_supervised_final_energy": float(
                adopted_summary["mean_test_transport_minus_internal_slow_pc_supervised_final_energy"] or 0.0
            ),
        },
        "remaining_gap_primary_diagnosis": diagnosis,
        "recommended_next_narrow_tf2_move": recommended_next_move,
        "aggregate_csv_path": "aggregate_runs.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF2GapDecompositionSuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config),
        aggregate_rows=aggregate_rows,
        summary=summary,
    )
