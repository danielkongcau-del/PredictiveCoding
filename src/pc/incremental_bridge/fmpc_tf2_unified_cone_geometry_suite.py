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
from ..transport_core_v1.fmpc_tf1_flow import build_tf1_context
from .fmpc_tf2 import (
    TF2TerminalLocalFieldDirectionIntervention,
    _action_from_step,
    _apply_terminal_local_field_direction_intervention,
    _extract_detached_local_flow_anchor,
    _final_hidden_block_slice,
    _plan_tf2_micro_step,
    _project_onto_orthogonal_complement,
    _project_onto_rowspace,
    _rowspace_basis_from_output_weight,
    build_tf2_corrective_transport_terminal_angleclip_default_config,
    run_fmpc_tf2_experiment,
)
from .fmpc_tf2_endpoint_basis_suite import (
    _delta_geometry,
    _interface_geometry,
    _rowspace_basis,
)
from .fmpc_tf2_readout_refit_suite import _build_feature_bundle, _load_reference_context


@dataclass(frozen=True)
class _CaseSpec:
    case_name: str
    description: str
    terminal_intervention: TF2TerminalLocalFieldDirectionIntervention
    rowspace_angle_clip_degrees: float
    orthogonal_angle_clip_degrees: float


@dataclass
class FMPCTF2UnifiedConeGeometrySuiteConfig:
    """Run a narrow adopted-package unified-cone vs split-subspace geometry diagnostic."""

    experiment_name: str = "fmpc_tf2_unified_cone_geometry_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 60
    batch_size: int = 128
    eval_steps: int = 15
    layer_dims: tuple[int, ...] = (64, 64, 10)
    reference_summary_path: str | Path = "outputs/incremental_bridge/fmpc_tf2_external_comparison_suite/aggregate_summary.json"

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"

    def case_specs(self) -> tuple[_CaseSpec, ...]:
        return (
            _CaseSpec(
                case_name="adopted_control",
                description="Current adopted full-vector terminal angle clip.",
                terminal_intervention="local_field_direction_angle_clip_keep_live_norm",
                rowspace_angle_clip_degrees=30.0,
                orthogonal_angle_clip_degrees=30.0,
            ),
            _CaseSpec(
                case_name="rowspace_only_angle_clip",
                description="Clip only the terminal row-space component; keep the orthogonal component unchanged.",
                terminal_intervention="local_field_direction_angle_clip_keep_live_norm_rowspace_only",
                rowspace_angle_clip_degrees=30.0,
                orthogonal_angle_clip_degrees=30.0,
            ),
            _CaseSpec(
                case_name="orthogonal_only_angle_clip",
                description="Clip only the terminal orthogonal component; keep the row-space component unchanged.",
                terminal_intervention="local_field_direction_angle_clip_keep_live_norm_orthogonal_only",
                rowspace_angle_clip_degrees=30.0,
                orthogonal_angle_clip_degrees=30.0,
            ),
            _CaseSpec(
                case_name="split_threshold_row_strict_20_45",
                description="Keep both subspaces active with stricter row-space clip than orthogonal clip.",
                terminal_intervention="local_field_direction_angle_clip_keep_live_norm_split_threshold",
                rowspace_angle_clip_degrees=20.0,
                orthogonal_angle_clip_degrees=45.0,
            ),
            _CaseSpec(
                case_name="split_threshold_orthogonal_strict_45_20",
                description="Keep both subspaces active with stricter orthogonal clip than row-space clip.",
                terminal_intervention="local_field_direction_angle_clip_keep_live_norm_split_threshold",
                rowspace_angle_clip_degrees=45.0,
                orthogonal_angle_clip_degrees=20.0,
            ),
            _CaseSpec(
                case_name="split_threshold_balanced_30_30",
                description="Balanced split-threshold sanity check with equal row-space and orthogonal clip strengths.",
                terminal_intervention="local_field_direction_angle_clip_keep_live_norm_split_threshold",
                rowspace_angle_clip_degrees=30.0,
                orthogonal_angle_clip_degrees=30.0,
            ),
        )


@dataclass
class FMPCTF2UnifiedConeGeometrySuiteRunResult:
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


def _vector_norms(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(x, dtype=np.float64), axis=1)


def _rowwise_angles_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        raise ValueError("left and right must share the same shape.")
    if left_array.ndim != 2:
        raise ValueError("Angle computation expects shape (batch, features).")
    left_norms = _vector_norms(left_array)
    right_norms = _vector_norms(right_array)
    valid = (left_norms > 1e-12) & (right_norms > 1e-12)
    angles = np.zeros(left_array.shape[0], dtype=np.float64)
    if np.any(valid):
        numerators = np.sum(left_array[valid] * right_array[valid], axis=1)
        denominators = np.maximum(left_norms[valid] * right_norms[valid], 1e-12)
        cosines = np.clip(numerators / denominators, -1.0, 1.0)
        angles[valid] = np.rad2deg(np.arccos(cosines))
    return angles


def _mean_norm(vectors: np.ndarray) -> float:
    return float(np.mean(_vector_norms(vectors)))


def _mean_ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    numerator_norms = _vector_norms(numerator)
    denominator_norms = _vector_norms(denominator)
    return float(np.mean(numerator_norms / np.maximum(denominator_norms, 1e-12)))


def _mean_absolute_ratio_delta(
    numerator_left: np.ndarray,
    denominator_left: np.ndarray,
    numerator_right: np.ndarray,
    denominator_right: np.ndarray,
) -> float:
    left_ratio = _vector_norms(numerator_left) / np.maximum(_vector_norms(denominator_left), 1e-12)
    right_ratio = _vector_norms(numerator_right) / np.maximum(_vector_norms(denominator_right), 1e-12)
    return float(np.mean(np.abs(right_ratio - left_ratio)))


def _activation_rate(before: np.ndarray, after: np.ndarray, *, tolerance: float = 1e-8) -> float:
    before_array = np.asarray(before, dtype=np.float64)
    after_array = np.asarray(after, dtype=np.float64)
    return float(np.mean(_vector_norms(after_array - before_array) > float(tolerance)))


def _runtime_proxy_seconds(summary: dict[str, Any]) -> float:
    timing = dict(summary.get("timing", {}))
    return float(timing.get("train_wall_time_seconds", 0.0)) + float(
        timing.get("final_evaluation_wall_time_seconds", 0.0)
    )


def _report_only_reference_delta(case_summary: dict[str, Any], reference_summary: dict[str, Any]) -> dict[str, float]:
    payload = {
        "mean_val_accuracy_delta": float(case_summary["mean_val_accuracy"] - reference_summary["mean_val_accuracy"]),
        "mean_test_accuracy_delta": float(case_summary["mean_test_accuracy"] - reference_summary["mean_test_accuracy"]),
        "mean_runtime_proxy_seconds_delta": float(
            case_summary["mean_runtime_proxy_seconds"] - reference_summary["mean_runtime_proxy_seconds"]
        ),
    }
    if "mean_val_report_output_mse" in reference_summary:
        payload["mean_val_report_output_mse_delta"] = float(
            case_summary["mean_val_report_output_mse"] - reference_summary["mean_val_report_output_mse"]
        )
    if "mean_test_report_output_mse" in reference_summary:
        payload["mean_test_report_output_mse_delta"] = float(
            case_summary["mean_test_report_output_mse"] - reference_summary["mean_test_report_output_mse"]
        )
    return payload


def _suite_config_payload(config: FMPCTF2UnifiedConeGeometrySuiteConfig) -> dict[str, Any]:
    return {
        "phase": "Phase Incremental Bridge",
        "stage": "adopted_package_unified_cone_vs_split_subspace_geometry",
        "base_preset_name": "tf2_corrective_transport_terminal_angleclip_default",
        "transport_family_fixed": True,
        "candidate_specs": [
            {
                "case_name": spec.case_name,
                "description": spec.description,
                "terminal_intervention": spec.terminal_intervention,
                "rowspace_angle_clip_degrees": float(spec.rowspace_angle_clip_degrees),
                "orthogonal_angle_clip_degrees": float(spec.orthogonal_angle_clip_degrees),
            }
            for spec in config.case_specs()
        ],
        "seeds": [int(seed) for seed in config.seeds],
        "diagnostics": {
            "terminal_action_geometry": True,
            "rowspace_orthogonal_coupling_metrics": True,
            "validation_only_terminal_geometry_selection_view": True,
        },
    }


def _case_run_id(case_name: str, seed: int) -> str:
    short_name = {
        "adopted_control": "control",
        "rowspace_only_angle_clip": "rowclip",
        "orthogonal_only_angle_clip": "orthclip",
        "split_threshold_row_strict_20_45": "split_row",
        "split_threshold_orthogonal_strict_45_20": "split_orth",
        "split_threshold_balanced_30_30": "split_bal",
    }[case_name]
    return f"{short_name}_s{seed}"


def _terminal_action_geometry(
    model: Any,
    psi_network: Any,
    config: Any,
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    context = build_tf1_context(model, x, y)
    output_weight = np.asarray(model.layers[-1].weight, dtype=np.float64)
    final_hidden_slice = _final_hidden_block_slice(context)
    basis = _rowspace_basis_from_output_weight(output_weight)

    knots = np.linspace(0.0, 1.0, int(config.micro_steps) + 1, dtype=np.float64)
    z_on = np.asarray(context.z0, dtype=np.float64).copy()
    z_lf = np.asarray(context.z0, dtype=np.float64).copy()

    for step_index in range(int(config.micro_steps)):
        t_k = float(knots[step_index])
        r_k = 1.0 - t_k
        dt = float(knots[step_index + 1] - knots[step_index])
        plan = _plan_tf2_micro_step(
            context,
            psi_network,
            config,
            z_on,
            z_lf,
            t_k=t_k,
            dt=dt,
            r_k=r_k,
        )
        if step_index != int(config.micro_steps) - 1:
            z_on = np.asarray(plan.z_on_next, dtype=np.float64)
            z_lf = np.asarray(plan.z_lf_next, dtype=np.float64)
            continue

        raw_action = _action_from_step(z_on, plan.z_on_next, dt)
        local_field_action = _extract_detached_local_flow_anchor(plan.psi_inputs, config)
        stabilized_action, _ = _apply_terminal_local_field_direction_intervention(
            z_on,
            dt,
            plan,
            config,
            context=context,
            output_weight=output_weight,
        )

        raw_final = np.asarray(raw_action[:, final_hidden_slice], dtype=np.float64)
        anchor_final = np.asarray(local_field_action[:, final_hidden_slice], dtype=np.float64)
        stabilized_final = np.asarray(stabilized_action[:, final_hidden_slice], dtype=np.float64)

        raw_row = _project_onto_rowspace(raw_final, basis)
        raw_orth = _project_onto_orthogonal_complement(raw_final, basis)
        anchor_row = _project_onto_rowspace(anchor_final, basis)
        anchor_orth = _project_onto_orthogonal_complement(anchor_final, basis)
        stabilized_row = _project_onto_rowspace(stabilized_final, basis)
        stabilized_orth = _project_onto_orthogonal_complement(stabilized_final, basis)

        full_raw_vs_anchor = _rowwise_angles_deg(raw_action, local_field_action)
        full_stabilized_vs_anchor = _rowwise_angles_deg(stabilized_action, local_field_action)
        full_raw_vs_stabilized = _rowwise_angles_deg(raw_action, stabilized_action)
        row_raw_vs_anchor = _rowwise_angles_deg(raw_row, anchor_row)
        row_stabilized_vs_anchor = _rowwise_angles_deg(stabilized_row, anchor_row)
        row_raw_vs_stabilized = _rowwise_angles_deg(raw_row, stabilized_row)
        orth_raw_vs_anchor = _rowwise_angles_deg(raw_orth, anchor_orth)
        orth_stabilized_vs_anchor = _rowwise_angles_deg(stabilized_orth, anchor_orth)
        orth_raw_vs_stabilized = _rowwise_angles_deg(raw_orth, stabilized_orth)

        return {
            "full_space_angle_raw_vs_lf_deg": float(np.mean(full_raw_vs_anchor)),
            "full_space_angle_stabilized_vs_lf_deg": float(np.mean(full_stabilized_vs_anchor)),
            "full_space_angle_improvement_deg": float(np.mean(full_raw_vs_anchor - full_stabilized_vs_anchor)),
            "full_space_correction_angle_deg": float(np.mean(full_raw_vs_stabilized)),
            "rowspace_angle_raw_vs_lf_deg": float(np.mean(row_raw_vs_anchor)),
            "rowspace_angle_stabilized_vs_lf_deg": float(np.mean(row_stabilized_vs_anchor)),
            "rowspace_angle_improvement_deg": float(np.mean(row_raw_vs_anchor - row_stabilized_vs_anchor)),
            "rowspace_correction_angle_deg": float(np.mean(row_raw_vs_stabilized)),
            "orthogonal_angle_raw_vs_lf_deg": float(np.mean(orth_raw_vs_anchor)),
            "orthogonal_angle_stabilized_vs_lf_deg": float(np.mean(orth_stabilized_vs_anchor)),
            "orthogonal_angle_improvement_deg": float(np.mean(orth_raw_vs_anchor - orth_stabilized_vs_anchor)),
            "orthogonal_correction_angle_deg": float(np.mean(orth_raw_vs_stabilized)),
            "subspace_correction_mismatch_deg": float(np.mean(np.abs(row_raw_vs_stabilized - orth_raw_vs_stabilized))),
            "full_cone_clip_activation_rate": _activation_rate(raw_action, stabilized_action),
            "rowspace_clip_activation_rate": _activation_rate(raw_row, stabilized_row),
            "orthogonal_clip_activation_rate": _activation_rate(raw_orth, stabilized_orth),
            "full_space_angle_above_30deg_rate_after_stabilization": float(
                np.mean(full_stabilized_vs_anchor > 30.0 + 1e-8)
            ),
            "raw_action_norm": _mean_norm(raw_action),
            "local_field_action_norm": _mean_norm(local_field_action),
            "stabilized_action_norm": _mean_norm(stabilized_action),
            "raw_rowspace_norm": _mean_norm(raw_row),
            "raw_orthogonal_norm": _mean_norm(raw_orth),
            "stabilized_rowspace_norm": _mean_norm(stabilized_row),
            "stabilized_orthogonal_norm": _mean_norm(stabilized_orth),
            "row_orth_norm_ratio_raw": _mean_ratio(raw_row, raw_orth),
            "row_orth_norm_ratio_stabilized": _mean_ratio(stabilized_row, stabilized_orth),
            "row_orth_norm_ratio_abs_delta": _mean_absolute_ratio_delta(
                raw_row,
                raw_orth,
                stabilized_row,
                stabilized_orth,
            ),
        }
    raise ValueError("Failed to reach the terminal micro-step.")


def _pairwise_delta(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]) -> dict[str, Any]:
    left_by_seed = {int(row["seed"]): row for row in left_rows}
    right_by_seed = {int(row["seed"]): row for row in right_rows}
    shared_seeds = sorted(set(left_by_seed) & set(right_by_seed))
    if not shared_seeds:
        raise ValueError("Pairwise comparison requires at least one shared seed.")

    def mean_delta(metric_name: str) -> float:
        return _mean(
            [float(left_by_seed[seed][metric_name]) - float(right_by_seed[seed][metric_name]) for seed in shared_seeds]
        )

    return {
        "shared_seeds": [int(seed) for seed in shared_seeds],
        "mean_val_accuracy_delta": mean_delta("val_accuracy"),
        "mean_test_accuracy_delta": mean_delta("test_accuracy"),
        "mean_gate_passing_epoch_count_delta": mean_delta("gate_passing_epoch_count"),
        "mean_selected_epoch_passes_gate_rate_delta": mean_delta("selected_epoch_passes_gate"),
        "mean_selector_fallback_used_rate_delta": mean_delta("selector_fallback_used"),
        "mean_val_transported_final_energy_delta": mean_delta("val_transported_final_energy"),
        "mean_val_report_output_mse_delta": mean_delta("val_report_output_mse"),
        "mean_val_supervised_transport_output_mse_delta": mean_delta("val_supervised_transport_output_mse"),
        "mean_val_delta_h_rms_total_delta": mean_delta("val_delta_h_rms_total"),
        "mean_val_delta_h_rms_rowspace_delta": mean_delta("val_delta_h_rms_rowspace"),
        "mean_val_delta_h_rms_orthogonal_delta": mean_delta("val_delta_h_rms_orthogonal"),
        "mean_val_delta_h_rowspace_fraction_delta": mean_delta("val_delta_h_rowspace_fraction"),
        "mean_val_full_space_angle_stabilized_vs_lf_deg_delta": mean_delta("val_full_space_angle_stabilized_vs_lf_deg"),
        "mean_val_full_space_angle_above_30deg_rate_after_stabilization_delta": mean_delta(
            "val_full_space_angle_above_30deg_rate_after_stabilization"
        ),
        "mean_val_subspace_correction_mismatch_deg_delta": mean_delta("val_subspace_correction_mismatch_deg"),
        "mean_val_row_orth_norm_ratio_abs_delta_delta": mean_delta("val_row_orth_norm_ratio_abs_delta"),
        "mean_runtime_proxy_seconds_delta": mean_delta("runtime_proxy_seconds"),
    }


def _case_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Case summary requires at least one row.")

    def mean_metric(metric_name: str) -> float:
        return _mean([float(row[metric_name]) for row in rows])

    def std_metric(metric_name: str) -> float:
        return _std([float(row[metric_name]) for row in rows])

    metrics = {
        "num_runs": len(rows),
        "mean_val_accuracy": mean_metric("val_accuracy"),
        "std_val_accuracy": std_metric("val_accuracy"),
        "mean_test_accuracy": mean_metric("test_accuracy"),
        "std_test_accuracy": std_metric("test_accuracy"),
        "mean_gate_passing_epoch_count": mean_metric("gate_passing_epoch_count"),
        "selected_epoch_passes_gate_rate": mean_metric("selected_epoch_passes_gate"),
        "selector_fallback_used_rate": mean_metric("selector_fallback_used"),
        "mean_val_transported_final_energy": mean_metric("val_transported_final_energy"),
        "mean_val_report_output_mse": mean_metric("val_report_output_mse"),
        "std_val_report_output_mse": std_metric("val_report_output_mse"),
        "mean_test_report_output_mse": mean_metric("test_report_output_mse"),
        "std_test_report_output_mse": std_metric("test_report_output_mse"),
        "mean_val_supervised_transport_output_mse": mean_metric("val_supervised_transport_output_mse"),
        "std_val_supervised_transport_output_mse": std_metric("val_supervised_transport_output_mse"),
        "mean_test_supervised_transport_output_mse": mean_metric("test_supervised_transport_output_mse"),
        "std_test_supervised_transport_output_mse": std_metric("test_supervised_transport_output_mse"),
        "mean_val_delta_h_rms_total": mean_metric("val_delta_h_rms_total"),
        "mean_val_delta_h_rms_rowspace": mean_metric("val_delta_h_rms_rowspace"),
        "mean_val_delta_h_rms_orthogonal": mean_metric("val_delta_h_rms_orthogonal"),
        "mean_val_delta_h_rowspace_fraction": mean_metric("val_delta_h_rowspace_fraction"),
        "mean_test_delta_h_rms_total": mean_metric("test_delta_h_rms_total"),
        "mean_test_delta_h_rms_rowspace": mean_metric("test_delta_h_rms_rowspace"),
        "mean_test_delta_h_rms_orthogonal": mean_metric("test_delta_h_rms_orthogonal"),
        "mean_test_delta_h_rowspace_fraction": mean_metric("test_delta_h_rowspace_fraction"),
        "mean_selected_epoch": mean_metric("selected_epoch"),
        "mean_runtime_proxy_seconds": mean_metric("runtime_proxy_seconds"),
    }
    geometry_metrics = (
        "val_full_space_angle_raw_vs_lf_deg",
        "val_full_space_angle_stabilized_vs_lf_deg",
        "val_full_space_angle_improvement_deg",
        "val_full_space_correction_angle_deg",
        "val_rowspace_angle_raw_vs_lf_deg",
        "val_rowspace_angle_stabilized_vs_lf_deg",
        "val_rowspace_angle_improvement_deg",
        "val_rowspace_correction_angle_deg",
        "val_orthogonal_angle_raw_vs_lf_deg",
        "val_orthogonal_angle_stabilized_vs_lf_deg",
        "val_orthogonal_angle_improvement_deg",
        "val_orthogonal_correction_angle_deg",
        "val_subspace_correction_mismatch_deg",
        "val_full_cone_clip_activation_rate",
        "val_rowspace_clip_activation_rate",
        "val_orthogonal_clip_activation_rate",
        "val_full_space_angle_above_30deg_rate_after_stabilization",
        "val_raw_action_norm",
        "val_local_field_action_norm",
        "val_stabilized_action_norm",
        "val_raw_rowspace_norm",
        "val_raw_orthogonal_norm",
        "val_stabilized_rowspace_norm",
        "val_stabilized_orthogonal_norm",
        "val_row_orth_norm_ratio_raw",
        "val_row_orth_norm_ratio_stabilized",
        "val_row_orth_norm_ratio_abs_delta",
    )
    metrics.update({f"mean_{metric_name}": mean_metric(metric_name) for metric_name in geometry_metrics})
    return metrics


def _diagnose(summary_by_candidate: dict[str, dict[str, Any]]) -> tuple[str, dict[str, float], str]:
    control = summary_by_candidate["adopted_control"]
    split_names = [
        "rowspace_only_angle_clip",
        "orthogonal_only_angle_clip",
        "split_threshold_row_strict_20_45",
        "split_threshold_orthogonal_strict_45_20",
        "split_threshold_balanced_30_30",
    ]
    split_candidates = [summary_by_candidate[name] for name in split_names]

    best_split_val = max(float(candidate["mean_val_accuracy"]) for candidate in split_candidates)
    best_split_full_angle = min(
        float(candidate["mean_val_full_space_angle_stabilized_vs_lf_deg"]) for candidate in split_candidates
    )
    best_split_full_violation_rate = min(
        float(candidate["mean_val_full_space_angle_above_30deg_rate_after_stabilization"])
        for candidate in split_candidates
    )
    best_split_subspace_mismatch = min(
        float(candidate["mean_val_subspace_correction_mismatch_deg"]) for candidate in split_candidates
    )
    best_split_ratio_delta = min(float(candidate["mean_val_row_orth_norm_ratio_abs_delta"]) for candidate in split_candidates)

    evidence = {
        "control_mean_val_accuracy": float(control["mean_val_accuracy"]),
        "best_split_mean_val_accuracy": float(best_split_val),
        "control_mean_val_full_space_angle_stabilized_vs_lf_deg": float(
            control["mean_val_full_space_angle_stabilized_vs_lf_deg"]
        ),
        "best_split_mean_val_full_space_angle_stabilized_vs_lf_deg": float(best_split_full_angle),
        "control_mean_val_full_space_angle_above_30deg_rate_after_stabilization": float(
            control["mean_val_full_space_angle_above_30deg_rate_after_stabilization"]
        ),
        "best_split_mean_val_full_space_angle_above_30deg_rate_after_stabilization": float(best_split_full_violation_rate),
        "control_mean_val_subspace_correction_mismatch_deg": float(
            control["mean_val_subspace_correction_mismatch_deg"]
        ),
        "best_split_mean_val_subspace_correction_mismatch_deg": float(best_split_subspace_mismatch),
        "control_mean_val_row_orth_norm_ratio_abs_delta": float(control["mean_val_row_orth_norm_ratio_abs_delta"]),
        "best_split_mean_val_row_orth_norm_ratio_abs_delta": float(best_split_ratio_delta),
    }

    diagnosis = (
        "unified_cone_preserves_a_shared_full_space_angular_constraint_that_split_subspace_cones_do_not_factorize"
    )
    if (
        float(control["mean_val_full_space_angle_stabilized_vs_lf_deg"]) <= float(best_split_full_angle)
        and float(control["mean_val_full_space_angle_above_30deg_rate_after_stabilization"])
        <= float(best_split_full_violation_rate)
    ):
        if float(control["mean_val_row_orth_norm_ratio_abs_delta"]) > float(best_split_ratio_delta):
            diagnosis = (
                "unified_cone_gain_is_best_explained_by_shared_full_space_geometry_not_by_literal_subspace_ratio_preservation"
            )
        elif float(control["mean_val_subspace_correction_mismatch_deg"]) <= float(best_split_subspace_mismatch):
            diagnosis = (
                "unified_cone_preserves_shared_full_space_geometry_and_better_matched_cross_subspace_corrections"
            )
    else:
        diagnosis = "mixed_picture_with_weak_shared_full_space_geometry_signal"

    recommended_next_move = (
        "run one narrow geometry-preserving unified-cone-shape diagnostic inside the adopted full-vector family, "
        "rather than another split-subspace threshold sweep"
    )
    return diagnosis, evidence, recommended_next_move


def run_fmpc_tf2_unified_cone_geometry_suite(
    config: FMPCTF2UnifiedConeGeometrySuiteConfig,
) -> FMPCTF2UnifiedConeGeometrySuiteRunResult:
    run_dir = _prepare_run_dir(
        _resolve_run_dir(
            config.output_root,
            config.experiment_name,
            config.resolved_run_id(),
            config.output_layout,
        )
    )
    _write_json(run_dir / "config.json", _suite_config_payload(config))

    tf2_root = run_dir / "tf2_runs"
    tf2_root.mkdir(parents=True, exist_ok=True)
    aggregate_rows: list[dict[str, Any]] = []

    for case in config.case_specs():
        for seed in config.seeds:
            tf2_config = build_tf2_corrective_transport_terminal_angleclip_default_config(
                experiment_name="tf2",
                output_root=tf2_root,
                output_layout="run_id_subdir",
                run_id=_case_run_id(case.case_name, int(seed)),
                run_seed=int(seed),
                data_seed=int(seed),
                model_init_seed=int(seed),
                psi_init_seed=int(seed),
                batch_order_seed=int(seed),
                epochs=int(config.epochs),
                batch_size=int(config.batch_size),
                eval_steps=int(config.eval_steps),
                layer_dims=tuple(config.layer_dims),
                terminal_local_field_direction_intervention=case.terminal_intervention,
                terminal_local_field_rowspace_angle_clip_degrees=float(case.rowspace_angle_clip_degrees),
                terminal_local_field_orthogonal_angle_clip_degrees=float(case.orthogonal_angle_clip_degrees),
            )
            result = run_fmpc_tf2_experiment(tf2_config)
            if result.model is None or result.psi_network is None:
                raise ValueError("Unified-cone geometry suite requires runtime model and psi network objects.")

            split = load_digits_split(
                split_seed=int(tf2_config.data_seed),
                train_fraction=float(tf2_config.train_fraction),
                val_fraction=float(tf2_config.val_fraction),
                test_fraction=float(tf2_config.test_fraction),
            )
            val_bundle = _build_feature_bundle(result.model, result.psi_network, tf2_config, split.x_val, split.y_val)
            test_bundle = _build_feature_bundle(result.model, result.psi_network, tf2_config, split.x_test, split.y_test)
            weight = np.asarray(result.model.layers[-1].weight, dtype=np.float64)
            bias = np.asarray(result.model.layers[-1].bias, dtype=np.float64)
            basis = _rowspace_basis(weight)
            transported_val = _interface_geometry(val_bundle.transported_penultimate, split.y_val, weight, bias)
            transported_test = _interface_geometry(test_bundle.transported_penultimate, split.y_test, weight, bias)
            delta_val = _delta_geometry(val_bundle.transported_penultimate, val_bundle.slow_pc_penultimate, split.y_val, basis)
            delta_test = _delta_geometry(
                test_bundle.transported_penultimate,
                test_bundle.slow_pc_penultimate,
                split.y_test,
                basis,
            )
            geometry_val = _terminal_action_geometry(result.model, result.psi_network, tf2_config, split.x_val, split.y_val)
            summary = result.summary
            aggregate_rows.append(
                {
                    "case_name": case.case_name,
                    "seed": int(seed),
                    "run_id": str(result.run_dir.name),
                    "run_summary_path": _relative_posix(run_dir, result.run_dir / "summary.json"),
                    "terminal_local_field_direction_intervention": case.terminal_intervention,
                    "terminal_local_field_rowspace_angle_clip_degrees": float(case.rowspace_angle_clip_degrees),
                    "terminal_local_field_orthogonal_angle_clip_degrees": float(case.orthogonal_angle_clip_degrees),
                    "selected_epoch": int(summary["best_epoch"]),
                    "val_accuracy": float(summary["val_accuracy"]),
                    "test_accuracy": float(summary["test_accuracy"]),
                    "gate_passing_epoch_count": int(summary["gate_passing_epoch_count"]),
                    "selected_epoch_passes_gate": 1.0 if bool(summary["selected_epoch_passes_gate"]) else 0.0,
                    "selector_fallback_used": 1.0 if bool(summary["selector_fallback_used"]) else 0.0,
                    "val_transported_final_energy": float(summary["val_transported_final_energy"]),
                    "val_report_output_mse": float(summary["val_loss"]),
                    "test_report_output_mse": float(summary["test_loss"]),
                    "val_supervised_transport_output_mse": float(transported_val["frozen_head_output_mse"]),
                    "test_supervised_transport_output_mse": float(transported_test["frozen_head_output_mse"]),
                    "val_delta_h_rms_total": float(delta_val["delta_h_rms_total"]),
                    "val_delta_h_rms_rowspace": float(delta_val["delta_h_rms_rowspace"]),
                    "val_delta_h_rms_orthogonal": float(delta_val["delta_h_rms_orthogonal"]),
                    "val_delta_h_rowspace_fraction": float(delta_val["delta_h_rowspace_fraction"]),
                    "test_delta_h_rms_total": float(delta_test["delta_h_rms_total"]),
                    "test_delta_h_rms_rowspace": float(delta_test["delta_h_rms_rowspace"]),
                    "test_delta_h_rms_orthogonal": float(delta_test["delta_h_rms_orthogonal"]),
                    "test_delta_h_rowspace_fraction": float(delta_test["delta_h_rowspace_fraction"]),
                    **{f"val_{key}": float(value) for key, value in geometry_val.items()},
                    "runtime_proxy_seconds": _runtime_proxy_seconds(summary),
                }
            )

    _write_csv(run_dir / "aggregate_runs.csv", aggregate_rows)

    by_candidate = {
        case.case_name: _case_summary([row for row in aggregate_rows if str(row["case_name"]) == case.case_name])
        for case in config.case_specs()
    }
    control_rows = [row for row in aggregate_rows if str(row["case_name"]) == "adopted_control"]
    pairwise_vs_control = {
        case.case_name: _pairwise_delta([row for row in aggregate_rows if str(row["case_name"]) == case.case_name], control_rows)
        for case in config.case_specs()
        if case.case_name != "adopted_control"
    }

    diagnosis, evidence, recommended_next_move = _diagnose(by_candidate)

    reference_context = _load_reference_context(config.reference_summary_path)
    report_only_reference: dict[str, Any] | None = None
    if reference_context is not None:
        by_method = dict(reference_context.get("by_method", {}))
        slow_pc_reference = by_method.get("canonical_slow_pc_digits_baseline")
        historical_reference = by_method.get("tf2_corrective_transport_default")
        report_only_reference = {
            "canonical_slow_pc_digits_baseline": slow_pc_reference,
            "historical_corrective_reference": historical_reference,
            "candidate_vs_canonical_slow_pc_digits_baseline": (
                None
                if slow_pc_reference is None
                else {
                    case_name: _report_only_reference_delta(case_summary, slow_pc_reference)
                    for case_name, case_summary in by_candidate.items()
                }
            ),
            "candidate_vs_historical_corrective_reference": (
                None
                if historical_reference is None
                else {
                    case_name: _report_only_reference_delta(case_summary, historical_reference)
                    for case_name, case_summary in by_candidate.items()
                }
            ),
        }

    summary = {
        "phase": "Phase Incremental Bridge",
        "stage": "adopted_package_unified_cone_vs_split_subspace_geometry",
        "num_runs": len(aggregate_rows),
        "by_candidate": by_candidate,
        "pairwise_vs_control": pairwise_vs_control,
        "report_only_external_reference": report_only_reference,
        "diagnosis": diagnosis,
        "diagnosis_evidence": evidence,
        "recommended_next_move": recommended_next_move,
        "should_promote_split_subspace_replacement": False,
        "promoted_candidate_name": None,
        "decision": "keep_current_adopted_default_unchanged",
        "aggregate_csv_path": "aggregate_runs.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF2UnifiedConeGeometrySuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config),
        aggregate_rows=aggregate_rows,
        summary=summary,
    )
