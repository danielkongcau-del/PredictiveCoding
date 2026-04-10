from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..datasets import load_digits_split
from .fmpc_tf2 import (
    TF2TerminalLocalFieldDirectionIntervention,
    build_tf2_corrective_transport_terminal_angleclip_default_config,
    run_fmpc_tf2_experiment,
)
from .fmpc_tf2_endpoint_basis_suite import _delta_geometry, _interface_geometry, _rowspace_basis
from .fmpc_tf2_readout_refit_suite import _build_feature_bundle, _load_reference_context
from .fmpc_tf2_unified_cone_shape_suite import (
    _case_summary,
    _pairwise_delta,
    _prepare_run_dir,
    _relative_posix,
    _report_only_reference_delta,
    _resolve_run_dir,
    _runtime_proxy_seconds,
    _terminal_action_geometry,
    _write_csv,
    _write_json,
)


@dataclass(frozen=True)
class _CaseSpec:
    case_name: str
    description: str
    terminal_intervention: TF2TerminalLocalFieldDirectionIntervention
    full_angle_clip_degrees: float


@dataclass
class FMPCTF2SmoothUnifiedConeSuiteConfig:
    """Run a very narrow adopted-package smooth unified-cone diagnostic."""

    experiment_name: str = "fmpc_tf2_smooth_unified_cone_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 60
    batch_size: int = 128
    eval_steps: int = 15
    layer_dims: tuple[int, ...] = (64, 64, 10)
    reference_summary_path: str | Path = "outputs/tf2/fmpc_tf2_unified_cone_shape_suite/aggregate_summary.json"
    material_test_gain_threshold: float = 0.005
    min_gate_count_recovery_vs_hard20: float = 2.0
    min_selected_gate_rate_recovery_vs_hard20: float = 0.20
    max_selector_fallback_rate: float = 0.0

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"

    def case_specs(self) -> tuple[_CaseSpec, ...]:
        return (
            _CaseSpec(
                case_name="adopted_control",
                description="Current adopted hard 30 degree full-vector terminal angle clip.",
                terminal_intervention="local_field_direction_angle_clip_keep_live_norm",
                full_angle_clip_degrees=30.0,
            ),
            _CaseSpec(
                case_name="smooth_unified_cone_projection_30",
                description="Smooth full-space unified cone projection anchored to the same local-field direction.",
                terminal_intervention="local_field_direction_smooth_unified_cone_projection_keep_live_norm",
                full_angle_clip_degrees=30.0,
            ),
        )


@dataclass
class FMPCTF2SmoothUnifiedConeSuiteRunResult:
    run_dir: Path
    config: dict[str, Any]
    aggregate_rows: list[dict[str, Any]]
    summary: dict[str, Any]


def _suite_config_payload(config: FMPCTF2SmoothUnifiedConeSuiteConfig) -> dict[str, Any]:
    return {
        "phase": "Phase TF2",
        "stage": "adopted_package_smooth_unified_cone_diagnostic",
        "base_preset_name": "tf2_corrective_transport_terminal_angleclip_default",
        "transport_family_fixed": True,
        "candidate_specs": [
            {
                "case_name": spec.case_name,
                "description": spec.description,
                "terminal_intervention": spec.terminal_intervention,
                "full_angle_clip_degrees": float(spec.full_angle_clip_degrees),
            }
            for spec in config.case_specs()
        ],
        "seeds": [int(seed) for seed in config.seeds],
        "diagnostics": {
            "terminal_action_geometry": True,
            "smooth_unified_full_space_cone_comparison": True,
            "validation_only_terminal_geometry_selection_view": True,
            "previous_hard_20_reference_used": True,
        },
        "thresholds": {
            "material_test_gain_threshold": float(config.material_test_gain_threshold),
            "min_gate_count_recovery_vs_hard20": float(config.min_gate_count_recovery_vs_hard20),
            "min_selected_gate_rate_recovery_vs_hard20": float(config.min_selected_gate_rate_recovery_vs_hard20),
            "max_selector_fallback_rate": float(config.max_selector_fallback_rate),
        },
    }


def _case_run_id(case_name: str, seed: int) -> str:
    short_name = {
        "adopted_control": "control30",
        "smooth_unified_cone_projection_30": "smooth30",
    }[case_name]
    return f"{short_name}_s{seed}"


def _load_previous_hard_interior_reference(reference_summary_path: str | Path) -> dict[str, Any] | None:
    reference_context = _load_reference_context(reference_summary_path)
    if reference_context is None:
        return None
    return dict(reference_context.get("by_candidate", {})).get("unified_cone_interior_margin_20")


def _report_only_hard20_delta(case_summary: dict[str, Any], reference_summary: dict[str, Any]) -> dict[str, float]:
    return {
        "mean_val_accuracy_delta": float(case_summary["mean_val_accuracy"] - reference_summary["mean_val_accuracy"]),
        "mean_test_accuracy_delta": float(case_summary["mean_test_accuracy"] - reference_summary["mean_test_accuracy"]),
        "mean_gate_passing_epoch_count_delta": float(
            case_summary["mean_gate_passing_epoch_count"] - reference_summary["mean_gate_passing_epoch_count"]
        ),
        "selected_epoch_passes_gate_rate_delta": float(
            case_summary["selected_epoch_passes_gate_rate"] - reference_summary["selected_epoch_passes_gate_rate"]
        ),
        "selector_fallback_used_rate_delta": float(
            case_summary["selector_fallback_used_rate"] - reference_summary["selector_fallback_used_rate"]
        ),
        "mean_val_report_output_mse_delta": float(
            case_summary["mean_val_report_output_mse"] - reference_summary["mean_val_report_output_mse"]
        ),
        "mean_val_supervised_transport_output_mse_delta": float(
            case_summary["mean_val_supervised_transport_output_mse"]
            - reference_summary["mean_val_supervised_transport_output_mse"]
        ),
        "mean_val_delta_h_rms_rowspace_delta": float(
            case_summary["mean_val_delta_h_rms_rowspace"] - reference_summary["mean_val_delta_h_rms_rowspace"]
        ),
        "mean_val_delta_h_rowspace_fraction_delta": float(
            case_summary["mean_val_delta_h_rowspace_fraction"] - reference_summary["mean_val_delta_h_rowspace_fraction"]
        ),
        "mean_val_full_space_angle_stabilized_vs_lf_deg_delta": float(
            case_summary["mean_val_full_space_angle_stabilized_vs_lf_deg"]
            - reference_summary["mean_val_full_space_angle_stabilized_vs_lf_deg"]
        ),
        "mean_val_full_space_angle_above_30deg_rate_after_stabilization_delta": float(
            case_summary["mean_val_full_space_angle_above_30deg_rate_after_stabilization"]
            - reference_summary["mean_val_full_space_angle_above_30deg_rate_after_stabilization"]
        ),
    }


def _diagnose(
    summary_by_candidate: dict[str, dict[str, Any]],
    smooth_vs_control: dict[str, Any],
    previous_hard_interior_reference: dict[str, Any] | None,
) -> tuple[str, dict[str, float], str]:
    control = summary_by_candidate["adopted_control"]
    smooth = summary_by_candidate["smooth_unified_cone_projection_30"]
    evidence = {
        "control_mean_val_accuracy": float(control["mean_val_accuracy"]),
        "smooth_mean_val_accuracy": float(smooth["mean_val_accuracy"]),
        "control_mean_test_accuracy": float(control["mean_test_accuracy"]),
        "smooth_mean_test_accuracy": float(smooth["mean_test_accuracy"]),
        "control_mean_gate_passing_epoch_count": float(control["mean_gate_passing_epoch_count"]),
        "smooth_mean_gate_passing_epoch_count": float(smooth["mean_gate_passing_epoch_count"]),
        "control_selected_epoch_passes_gate_rate": float(control["selected_epoch_passes_gate_rate"]),
        "smooth_selected_epoch_passes_gate_rate": float(smooth["selected_epoch_passes_gate_rate"]),
        "control_selector_fallback_used_rate": float(control["selector_fallback_used_rate"]),
        "smooth_selector_fallback_used_rate": float(smooth["selector_fallback_used_rate"]),
        "control_mean_val_delta_h_rms_rowspace": float(control["mean_val_delta_h_rms_rowspace"]),
        "smooth_mean_val_delta_h_rms_rowspace": float(smooth["mean_val_delta_h_rms_rowspace"]),
        "control_mean_val_delta_h_rowspace_fraction": float(control["mean_val_delta_h_rowspace_fraction"]),
        "smooth_mean_val_delta_h_rowspace_fraction": float(smooth["mean_val_delta_h_rowspace_fraction"]),
        "control_mean_val_full_space_angle_stabilized_vs_lf_deg": float(
            control["mean_val_full_space_angle_stabilized_vs_lf_deg"]
        ),
        "smooth_mean_val_full_space_angle_stabilized_vs_lf_deg": float(
            smooth["mean_val_full_space_angle_stabilized_vs_lf_deg"]
        ),
        "control_mean_val_full_space_angle_above_30deg_rate_after_stabilization": float(
            control["mean_val_full_space_angle_above_30deg_rate_after_stabilization"]
        ),
        "smooth_mean_val_full_space_angle_above_30deg_rate_after_stabilization": float(
            smooth["mean_val_full_space_angle_above_30deg_rate_after_stabilization"]
        ),
    }
    if previous_hard_interior_reference is not None:
        evidence.update(
            {
                "previous_hard20_mean_val_accuracy": float(previous_hard_interior_reference["mean_val_accuracy"]),
                "previous_hard20_mean_test_accuracy": float(previous_hard_interior_reference["mean_test_accuracy"]),
                "previous_hard20_mean_gate_passing_epoch_count": float(
                    previous_hard_interior_reference["mean_gate_passing_epoch_count"]
                ),
                "previous_hard20_selected_epoch_passes_gate_rate": float(
                    previous_hard_interior_reference["selected_epoch_passes_gate_rate"]
                ),
                "previous_hard20_selector_fallback_used_rate": float(
                    previous_hard_interior_reference["selector_fallback_used_rate"]
                ),
            }
        )

    diagnosis = "current_hard_30_degree_unified_cone_remains_local_winner_inside_this_family"
    if (
        float(smooth_vs_control["mean_val_accuracy_delta"]) > 0.0
        and float(smooth_vs_control["mean_test_accuracy_delta"]) > 0.0
        and float(smooth_vs_control["mean_val_delta_h_rms_rowspace_delta"]) <= 0.0
        and float(smooth_vs_control["mean_val_delta_h_rowspace_fraction_delta"]) <= 0.0
    ):
        diagnosis = "smooth_unified_cone_improves_accuracy_and_rowspace_metrics_but_not_enough_to_replace_hard_30"
    recommended_next_move = (
        "if TF2 work continues inside the adopted full-vector cone family, either close this family as locally "
        "saturated or run at most one confirmation-level check before opening a different package-internal question"
    )
    return diagnosis, evidence, recommended_next_move


def _qualifies_for_adoption(
    smooth_summary: dict[str, Any],
    control_summary: dict[str, Any],
    smooth_vs_control: dict[str, Any],
    previous_hard_interior_reference: dict[str, Any] | None,
    config: FMPCTF2SmoothUnifiedConeSuiteConfig,
) -> bool:
    if previous_hard_interior_reference is None:
        return False
    gate_count_recovery = float(smooth_summary["mean_gate_passing_epoch_count"]) - float(
        previous_hard_interior_reference["mean_gate_passing_epoch_count"]
    )
    gate_rate_recovery = float(smooth_summary["selected_epoch_passes_gate_rate"]) - float(
        previous_hard_interior_reference["selected_epoch_passes_gate_rate"]
    )
    return (
        float(smooth_vs_control["mean_test_accuracy_delta"]) >= 0.0
        and float(smooth_vs_control["mean_val_accuracy_delta"]) >= 0.0
        and float(smooth_vs_control["mean_val_delta_h_rms_rowspace_delta"]) <= 0.0
        and float(smooth_vs_control["mean_val_delta_h_rowspace_fraction_delta"]) <= 0.0
        and float(smooth_summary["selector_fallback_used_rate"]) <= float(config.max_selector_fallback_rate)
        and float(smooth_summary["mean_val_full_space_angle_above_30deg_rate_after_stabilization"]) <= 0.0
        and gate_count_recovery >= float(config.min_gate_count_recovery_vs_hard20)
        and gate_rate_recovery >= float(config.min_selected_gate_rate_recovery_vs_hard20)
        and float(smooth_summary["mean_test_accuracy"])
        >= float(previous_hard_interior_reference["mean_test_accuracy"]) - float(config.material_test_gain_threshold)
        and float(smooth_summary["mean_test_accuracy"]) >= float(control_summary["mean_test_accuracy"])
    )


def run_fmpc_tf2_smooth_unified_cone_suite(
    config: FMPCTF2SmoothUnifiedConeSuiteConfig,
) -> FMPCTF2SmoothUnifiedConeSuiteRunResult:
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
                terminal_local_field_angle_clip_degrees=float(case.full_angle_clip_degrees),
            )
            result = run_fmpc_tf2_experiment(tf2_config)
            if result.model is None or result.psi_network is None:
                raise ValueError("Smooth unified-cone suite requires runtime model and psi network objects.")

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
                    "terminal_local_field_angle_clip_degrees": float(case.full_angle_clip_degrees),
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
    smooth_rows = [row for row in aggregate_rows if str(row["case_name"]) == "smooth_unified_cone_projection_30"]
    pairwise_vs_control = {
        "smooth_unified_cone_projection_30": _pairwise_delta(smooth_rows, control_rows),
    }

    previous_hard_interior_reference = _load_previous_hard_interior_reference(config.reference_summary_path)
    diagnosis, evidence, recommended_next_move = _diagnose(
        by_candidate,
        pairwise_vs_control["smooth_unified_cone_projection_30"],
        previous_hard_interior_reference,
    )
    promoted_candidate_name = (
        "smooth_unified_cone_projection_30"
        if _qualifies_for_adoption(
            by_candidate["smooth_unified_cone_projection_30"],
            by_candidate["adopted_control"],
            pairwise_vs_control["smooth_unified_cone_projection_30"],
            previous_hard_interior_reference,
            config,
        )
        else None
    )

    report_only_previous_hard_interior_reference = None
    smooth_vs_previous_hard_interior_reference = None
    if previous_hard_interior_reference is not None:
        report_only_previous_hard_interior_reference = previous_hard_interior_reference
        smooth_vs_previous_hard_interior_reference = _report_only_hard20_delta(
            by_candidate["smooth_unified_cone_projection_30"],
            previous_hard_interior_reference,
        )

    summary = {
        "phase": "Phase TF2",
        "stage": "adopted_package_smooth_unified_cone_diagnostic",
        "num_runs": len(aggregate_rows),
        "by_candidate": by_candidate,
        "pairwise_vs_control": pairwise_vs_control,
        "previous_hard_interior_reference": report_only_previous_hard_interior_reference,
        "smooth_candidate_vs_previous_hard_interior_reference": smooth_vs_previous_hard_interior_reference,
        "diagnosis": diagnosis,
        "diagnosis_evidence": evidence,
        "recommended_next_move": recommended_next_move,
        "should_promote_smooth_unified_cone": promoted_candidate_name is not None,
        "promoted_candidate_name": promoted_candidate_name,
        "decision": (
            "promote_smooth_unified_cone_projection"
            if promoted_candidate_name is not None
            else "keep_current_adopted_default_unchanged"
        ),
        "aggregate_csv_path": "aggregate_runs.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF2SmoothUnifiedConeSuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config),
        aggregate_rows=aggregate_rows,
        summary=summary,
    )
