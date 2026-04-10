from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np

from . import fmpc_tf2 as fmpc_tf2_module
from ..datasets import load_digits_split
from ..stage_03_transport_core_v1.fmpc_tf1 import _select_tf1_checkpoint_epoch, build_tf1_epoch_selection_diagnostics
from .fmpc_tf2 import FMPCTF2Config, FMPCTF2EpochMetrics, FMPCTF2EpochSnapshot
from .fmpc_tf2_basis_drift_localization_suite import _slow_pc_penultimate_by_knot
from .fmpc_tf2_endpoint_basis_suite import (
    _delta_geometry,
    _mean,
    _relative_posix,
    _rowspace_basis,
    _runtime_proxy_seconds,
    _std,
    _transport_penultimate_by_knot,
)
from .fmpc_tf2_partial_open_loop_handoff_suite import _ReplayPlanSlot, _build_cached_plan
from ..metrics import majority_class_baseline_accuracy
from ..minibatch import iter_minibatches
from ..utils import set_seed

_DirectionSourceMode = Literal["detached_local_field", "onpolicy_live_local_field"]
_NormHandlingMode = Literal["keep_live_norm", "use_anchor_norm"]
_HandoffMode = Literal["live_onpolicy", "cached_onpolicy_handoff"]
_Diagnosis = Literal[
    "preterminal_direction_source_is_primary_blocker",
    "preterminal_handoff_state_is_primary_blocker",
    "preterminal_norm_handling_is_primary_blocker",
    "mixed_preterminal_formulation_blocker",
]


@dataclass(frozen=True)
class _CaseSpec:
    case_name: str
    description: str
    intervention_step_offsets: tuple[int, ...]
    direction_source_mode: _DirectionSourceMode = "detached_local_field"
    norm_handling_mode: _NormHandlingMode = "keep_live_norm"
    handoff_mode: _HandoffMode = "live_onpolicy"


@dataclass
class FMPCTF2PreterminalSourceLocalizationSuiteConfig:
    """Run a narrow source-localization diagnostic on the adopted TF2 package."""

    experiment_name: str = "fmpc_tf2_preterminal_source_localization_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 60
    batch_size: int = 128
    eval_steps: int = 15
    layer_dims: tuple[int, ...] = (64, 64, 10)
    primary_seed_gate_positive_rate_threshold: float = 0.6
    primary_selected_gate_rate_threshold: float = 0.6
    primary_gate_count_margin: float = 2.0
    primary_rate_margin: float = 0.2

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"

    def case_specs(self) -> tuple[_CaseSpec, ...]:
        return (
            _CaseSpec(
                case_name="adopted_control_terminal_only",
                description="Current adopted package: full-vector hard 30 degree cone on the terminal micro-step only.",
                intervention_step_offsets=(-1,),
            ),
            _CaseSpec(
                case_name="failed_penultimate_plus_terminal_live",
                description="Diagnostic anchor: same adopted full-vector hard 30 degree cone on the penultimate and terminal micro-steps.",
                intervention_step_offsets=(-2, -1),
            ),
            _CaseSpec(
                case_name="preterminal_onpolicy_local_field_source_only",
                description="Penultimate plus terminal control, but swap only the preterminal direction source from detached local-field-only anchor to on-policy live local field.",
                intervention_step_offsets=(-2, -1),
                direction_source_mode="onpolicy_live_local_field",
            ),
            _CaseSpec(
                case_name="preterminal_anchor_norm_only",
                description="Penultimate plus terminal control, but swap only the preterminal norm handling from keep-live-norm to anchor-norm.",
                intervention_step_offsets=(-2, -1),
                norm_handling_mode="use_anchor_norm",
            ),
            _CaseSpec(
                case_name="preterminal_cached_onpolicy_handoff_only",
                description="Penultimate plus terminal control, but swap only the preterminal on-policy handoff state to the cached batch-start on-policy successor.",
                intervention_step_offsets=(-2, -1),
                handoff_mode="cached_onpolicy_handoff",
            ),
        )


@dataclass
class FMPCTF2PreterminalSourceLocalizationSuiteRunResult:
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


def _suite_config_payload(config: FMPCTF2PreterminalSourceLocalizationSuiteConfig) -> dict[str, Any]:
    return {
        "phase": "FMPC Stage 04 Incremental Bridge",
        "stage": "adopted_package_preterminal_update_source_localization",
        "base_preset_name": "tf2_corrective_transport_terminal_angleclip_default",
        "transport_family_fixed": True,
        "candidate_specs": [
            {
                "case_name": spec.case_name,
                "description": spec.description,
                "intervention_step_offsets": [int(value) for value in spec.intervention_step_offsets],
                "direction_source_mode": spec.direction_source_mode,
                "norm_handling_mode": spec.norm_handling_mode,
                "handoff_mode": spec.handoff_mode,
            }
            for spec in config.case_specs()
        ],
        "seeds": [int(seed) for seed in config.seeds],
        "selector_contract_fixed": True,
        "diagnostic_only": True,
        "thresholds": {
            "primary_seed_gate_positive_rate_threshold": float(config.primary_seed_gate_positive_rate_threshold),
            "primary_selected_gate_rate_threshold": float(config.primary_selected_gate_rate_threshold),
            "primary_gate_count_margin": float(config.primary_gate_count_margin),
            "primary_rate_margin": float(config.primary_rate_margin),
        },
    }


def _run_id(case_name: str, seed: int) -> str:
    return f"{case_name}_s{seed}"


def _preterminal_step_index(config: FMPCTF2Config) -> int:
    return int(config.micro_steps) - 2


def _step_indices(config: FMPCTF2Config) -> tuple[int, ...]:
    return fmpc_tf2_module._resolved_terminal_local_field_intervention_step_indices(config)


def _apply_same_geometry_full_vector_clip(
    context: Any,
    plan: Any,
    config: FMPCTF2Config,
    z_on_k: np.ndarray,
    dt: float,
    *,
    direction_source_mode: _DirectionSourceMode,
    norm_handling_mode: _NormHandlingMode,
) -> tuple[np.ndarray, np.ndarray]:
    raw_action = fmpc_tf2_module._action_from_step(z_on_k, plan.z_on_next, dt)
    if direction_source_mode == "detached_local_field":
        anchor_action = fmpc_tf2_module._extract_detached_local_flow_anchor(plan.psi_inputs, config)
    elif direction_source_mode == "onpolicy_live_local_field":
        anchor_action = fmpc_tf2_module.hidden_local_flow(context, z_on_k)
    else:
        raise ValueError(f"Unsupported direction_source_mode '{direction_source_mode}'.")

    if norm_handling_mode == "keep_live_norm":
        target_norm = fmpc_tf2_module._vector_norms(raw_action)[:, None]
    elif norm_handling_mode == "use_anchor_norm":
        target_norm = fmpc_tf2_module._vector_norms(anchor_action)[:, None]
    else:
        raise ValueError(f"Unsupported norm_handling_mode '{norm_handling_mode}'.")

    stabilized_action = fmpc_tf2_module._clip_direction_to_anchor_cone(
        fmpc_tf2_module._safe_direction(raw_action),
        fmpc_tf2_module._safe_direction(anchor_action),
        max_angle_degrees=float(config.terminal_local_field_angle_clip_degrees),
    ) * target_norm
    stabilized_next = np.asarray(z_on_k, dtype=np.float64) + float(dt) * stabilized_action
    fmpc_tf2_module.ensure_finite_array(stabilized_action, "tf2_preterminal_source_localization_stabilized_action")
    fmpc_tf2_module.ensure_finite_array(stabilized_next, "tf2_preterminal_source_localization_stabilized_next")
    return stabilized_action, stabilized_next


def _run_customized_micro_step(
    model: Any,
    psi_network: Any,
    config: FMPCTF2Config,
    context: Any,
    z_on_k: np.ndarray,
    z_lf_k: np.ndarray,
    *,
    t_k: float,
    dt: float,
    r_k: float,
    lambda_id: float,
    apply_theta_update: bool,
    theta_eta_w: float,
    theta_eta_b: float,
    is_terminal_step: bool,
    apply_direction_intervention: bool,
    onpolicy_mix_ratio: float | None,
    case_spec: _CaseSpec,
    cached_slot: _ReplayPlanSlot | None,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    plan = fmpc_tf2_module._plan_tf2_micro_step(
        context,
        psi_network,
        config,
        z_on_k,
        z_lf_k,
        t_k=t_k,
        dt=dt,
        r_k=r_k,
        onpolicy_mix_ratio=onpolicy_mix_ratio,
    )
    z_on_next = plan.z_on_next.copy()
    z_lf_next = plan.z_lf_next.copy()
    effective_prediction = fmpc_tf2_module._psi_predict(psi_network, plan.psi_inputs, config)

    preterminal_index = _preterminal_step_index(config)
    step_index = int(round(t_k / dt)) if dt > 0.0 else 0
    is_custom_preterminal_step = (
        step_index == preterminal_index
        and apply_direction_intervention
        and (
            case_spec.direction_source_mode != "detached_local_field"
            or case_spec.norm_handling_mode != "keep_live_norm"
            or case_spec.handoff_mode != "live_onpolicy"
        )
    )

    if apply_direction_intervention and config.terminal_local_field_direction_intervention != "none":
        if is_custom_preterminal_step:
            effective_prediction, z_on_next = _apply_same_geometry_full_vector_clip(
                context,
                plan,
                config,
                z_on_k,
                dt,
                direction_source_mode=case_spec.direction_source_mode,
                norm_handling_mode=case_spec.norm_handling_mode,
            )
        else:
            effective_prediction, z_on_next = fmpc_tf2_module._apply_terminal_local_field_direction_intervention(
                z_on_k,
                dt,
                plan,
                config,
                context=context,
                output_weight=model.layers[-1].weight,
            )

    if is_custom_preterminal_step and case_spec.handoff_mode == "cached_onpolicy_handoff":
        if cached_slot is None:
            raise ValueError("cached_slot is required for cached_onpolicy_handoff.")
        z_on_next = np.asarray(cached_slot.z_on_next, dtype=np.float64).copy()
        z_lf_next = plan.z_lf_next.copy()

    if apply_theta_update:
        output_alignment_scale = fmpc_tf2_module._output_alignment_scale_for_step(
            config,
            is_terminal_step=is_terminal_step,
        )
        theta_energy = fmpc_tf2_module._theta_update_from_transported_state(
            model,
            context,
            z_on_next,
            eta_w=theta_eta_w,
            eta_b=theta_eta_b,
            output_alignment_scale=output_alignment_scale,
        )
    else:
        theta_energy = fmpc_tf2_module.hidden_energy_from_state(context, z_on_next)

    boot_loss = float(np.mean((effective_prediction - plan.boot_targets) ** 2))
    identity_loss = float(np.mean((effective_prediction - plan.identity_targets) ** 2))
    if lambda_id > 0.0:
        combined_target = (plan.boot_targets + (lambda_id * plan.identity_targets)) / (1.0 + lambda_id)
        loss_scale = 1.0 + lambda_id
    else:
        combined_target = plan.boot_targets
        loss_scale = 1.0
    total_loss = boot_loss + (lambda_id * identity_loss)
    fmpc_tf2_module._weighted_mse_step(
        psi_network,
        plan.psi_inputs,
        combined_target,
        loss_scale=loss_scale,
    )
    return z_on_next, z_lf_next, total_loss, boot_loss, identity_loss, float(theta_energy)


def _train_one_batch_source_localization(
    model: Any,
    psi_network: Any,
    config: FMPCTF2Config,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    *,
    lambda_id: float,
    epoch_index: int,
    case_spec: _CaseSpec,
) -> tuple[float, float, float, float]:
    context = fmpc_tf2_module.build_tf1_context(model, x_batch, y_batch)
    knots = np.linspace(0.0, 1.0, int(config.micro_steps) + 1, dtype=np.float64)
    z_on = context.z0.copy()
    z_lf = context.z0.copy()
    active_cadence = fmpc_tf2_module._active_theta_update_cadence(config, epoch_index)
    intervention_step_indices = _step_indices(config)
    micro_eta_w, micro_eta_b = fmpc_tf2_module._theta_micro_learning_rates(config, active_cadence)
    active_mix_ratio = fmpc_tf2_module._active_onpolicy_mix_ratio(config, epoch_index)
    cached_slots = (
        _build_cached_plan(context, psi_network, config, lambda_id=lambda_id)
        if case_spec.handoff_mode == "cached_onpolicy_handoff"
        else None
    )
    total_losses: list[float] = []
    boot_losses: list[float] = []
    identity_losses: list[float] = []

    for step_index in range(int(config.micro_steps)):
        t_k = float(knots[step_index])
        r_k = 1.0 - t_k
        dt = float(knots[step_index + 1] - knots[step_index])
        is_terminal_step = bool(step_index == (int(config.micro_steps) - 1))
        is_custom_case = case_spec.case_name != "adopted_control_terminal_only"
        if not is_custom_case:
            z_on, z_lf, total_loss, boot_loss, identity_loss, _, _ = fmpc_tf2_module._run_tf2_micro_step(
                model,
                psi_network,
                config,
                context,
                z_on,
                z_lf,
                t_k=t_k,
                dt=dt,
                r_k=r_k,
                lambda_id=lambda_id,
                apply_theta_update=fmpc_tf2_module._theta_update_due_for_step(active_cadence, step_index),
                theta_eta_w=micro_eta_w,
                theta_eta_b=micro_eta_b,
                is_terminal_step=is_terminal_step,
                apply_direction_intervention=bool(step_index in intervention_step_indices),
                onpolicy_mix_ratio=active_mix_ratio,
            )
        else:
            z_on, z_lf, total_loss, boot_loss, identity_loss, _ = _run_customized_micro_step(
                model,
                psi_network,
                config,
                context,
                z_on,
                z_lf,
                t_k=t_k,
                dt=dt,
                r_k=r_k,
                lambda_id=lambda_id,
                apply_theta_update=fmpc_tf2_module._theta_update_due_for_step(active_cadence, step_index),
                theta_eta_w=micro_eta_w,
                theta_eta_b=micro_eta_b,
                is_terminal_step=is_terminal_step,
                apply_direction_intervention=bool(step_index in intervention_step_indices),
                onpolicy_mix_ratio=active_mix_ratio,
                case_spec=case_spec,
                cached_slot=None if cached_slots is None else cached_slots[step_index],
            )
        total_losses.append(total_loss)
        boot_losses.append(boot_loss)
        identity_losses.append(identity_loss)

    transported_final_energy = fmpc_tf2_module.hidden_energy_from_state(context, z_on)
    if active_cadence == "terminal_only":
        fmpc_tf2_module._theta_update_from_transported_state(
            model,
            context,
            z_on,
            eta_w=float(config.eta_w),
            eta_b=fmpc_tf2_module._resolved_eta_b(config),
        )

    return (
        float(np.mean(total_losses)),
        float(np.mean(boot_losses)),
        float(np.mean(identity_losses)),
        float(transported_final_energy),
    )


def _terminal_rowspace_metrics(
    model: Any,
    psi_network: Any,
    config: FMPCTF2Config,
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    context = fmpc_tf2_module.build_tf1_context(model, x, y)
    transported_knots, knot_times = _transport_penultimate_by_knot(model, psi_network, config, x, y)
    slow_pc_knots, _ = _slow_pc_penultimate_by_knot(model, context, knot_times)
    basis = _rowspace_basis(np.asarray(model.layers[-1].weight, dtype=np.float64))
    delta = _delta_geometry(transported_knots[-1], slow_pc_knots[-1], y, basis)
    return {
        "terminal_rowspace_rms": float(delta["delta_h_rms_rowspace"]),
        "terminal_rowspace_fraction": float(delta["delta_h_rowspace_fraction"]),
    }


def _success_row(
    run_dir: Path,
    case_spec: _CaseSpec,
    seed: int,
    summary: dict[str, Any],
    *,
    val_terminal_rowspace_rms: float,
    val_terminal_rowspace_fraction: float,
) -> dict[str, Any]:
    return {
        "case_name": case_spec.case_name,
        "seed": int(seed),
        "run_id": _run_id(case_spec.case_name, int(seed)),
        "run_summary_path": _relative_posix(
            run_dir,
            run_dir / "runs" / case_spec.case_name / f"seed_{int(seed)}" / "summary.json",
        ),
        "intervention_step_offsets": ",".join(str(value) for value in case_spec.intervention_step_offsets),
        "direction_source_mode": case_spec.direction_source_mode,
        "norm_handling_mode": case_spec.norm_handling_mode,
        "handoff_mode": case_spec.handoff_mode,
        "run_status": "success",
        "failure_kind": "",
        "nan_incidence_flag": 0.0,
        "selected_epoch": int(summary["best_epoch"]),
        "val_accuracy": float(summary["val_accuracy"]),
        "test_accuracy": float(summary["test_accuracy"]),
        "gate_passing_epoch_count": int(summary["gate_passing_epoch_count"]),
        "seed_gate_positive": 1.0 if int(summary["gate_passing_epoch_count"]) > 0 else 0.0,
        "selected_epoch_passes_gate": 1.0 if bool(summary["selected_epoch_passes_gate"]) else 0.0,
        "selector_fallback_used": 1.0 if bool(summary["selector_fallback_used"]) else 0.0,
        "val_transported_final_energy": float(summary["val_transported_final_energy"]),
        "val_report_output_mse": float(summary["val_loss"]),
        "val_terminal_rowspace_rms": float(val_terminal_rowspace_rms),
        "val_terminal_rowspace_fraction": float(val_terminal_rowspace_fraction),
        "runtime_proxy_seconds": _runtime_proxy_seconds(summary),
    }


def _failure_row(
    case_spec: _CaseSpec,
    seed: int,
    error: Exception,
) -> dict[str, Any]:
    text = str(error)
    lower = text.lower()
    failure_kind = "nan_or_inf" if ("nan" in lower or "inf" in lower) else type(error).__name__
    return {
        "case_name": case_spec.case_name,
        "seed": int(seed),
        "run_id": _run_id(case_spec.case_name, int(seed)),
        "run_summary_path": "",
        "intervention_step_offsets": ",".join(str(value) for value in case_spec.intervention_step_offsets),
        "direction_source_mode": case_spec.direction_source_mode,
        "norm_handling_mode": case_spec.norm_handling_mode,
        "handoff_mode": case_spec.handoff_mode,
        "run_status": "failure",
        "failure_kind": failure_kind,
        "nan_incidence_flag": 1.0 if failure_kind == "nan_or_inf" else 0.0,
        "selected_epoch": -1,
        "val_accuracy": 0.0,
        "test_accuracy": 0.0,
        "gate_passing_epoch_count": 0,
        "seed_gate_positive": 0.0,
        "selected_epoch_passes_gate": 0.0,
        "selector_fallback_used": 1.0,
        "val_transported_final_energy": float("inf"),
        "val_report_output_mse": float("inf"),
        "val_terminal_rowspace_rms": float("inf"),
        "val_terminal_rowspace_fraction": float("inf"),
        "runtime_proxy_seconds": 0.0,
    }


def _case_rows(rows: list[dict[str, Any]], case_name: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["case_name"]) == case_name and str(row["run_status"]) == "success"]


def _case_summary(rows: list[dict[str, Any]], all_case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "num_runs": 0,
            "mean_val_accuracy": None,
            "std_val_accuracy": None,
            "mean_test_accuracy": None,
            "std_test_accuracy": None,
            "mean_gate_passing_epoch_count": None,
            "seed_gate_positive_rate": 0.0,
            "selected_epoch_passes_gate_rate": 0.0,
            "selector_fallback_used_rate": 1.0,
            "mean_val_transported_final_energy": None,
            "mean_val_report_output_mse": None,
            "mean_val_terminal_rowspace_rms": None,
            "mean_val_terminal_rowspace_fraction": None,
            "mean_runtime_proxy_seconds": None,
            "failure_incidence_rate": 1.0,
            "nan_incidence_rate": _mean([float(row["nan_incidence_flag"]) for row in all_case_rows]),
        }
    return {
        "num_runs": len(rows),
        "mean_val_accuracy": _mean([float(row["val_accuracy"]) for row in rows]),
        "std_val_accuracy": _std([float(row["val_accuracy"]) for row in rows]),
        "mean_test_accuracy": _mean([float(row["test_accuracy"]) for row in rows]),
        "std_test_accuracy": _std([float(row["test_accuracy"]) for row in rows]),
        "mean_gate_passing_epoch_count": _mean([float(row["gate_passing_epoch_count"]) for row in rows]),
        "seed_gate_positive_rate": _mean([float(row["seed_gate_positive"]) for row in rows]),
        "selected_epoch_passes_gate_rate": _mean([float(row["selected_epoch_passes_gate"]) for row in rows]),
        "selector_fallback_used_rate": _mean([float(row["selector_fallback_used"]) for row in rows]),
        "mean_val_transported_final_energy": _mean([float(row["val_transported_final_energy"]) for row in rows]),
        "mean_val_report_output_mse": _mean([float(row["val_report_output_mse"]) for row in rows]),
        "mean_val_terminal_rowspace_rms": _mean([float(row["val_terminal_rowspace_rms"]) for row in rows]),
        "mean_val_terminal_rowspace_fraction": _mean([float(row["val_terminal_rowspace_fraction"]) for row in rows]),
        "mean_runtime_proxy_seconds": _mean([float(row["runtime_proxy_seconds"]) for row in rows]),
        "failure_incidence_rate": 1.0 - (float(len(rows)) / float(len(all_case_rows))),
        "nan_incidence_rate": _mean([float(row["nan_incidence_flag"]) for row in all_case_rows]),
    }


def _pairwise_delta(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]) -> dict[str, Any]:
    left_by_seed = {int(row["seed"]): row for row in left_rows}
    right_by_seed = {int(row["seed"]): row for row in right_rows}
    shared_seeds = sorted(set(left_by_seed) & set(right_by_seed))
    if not shared_seeds:
        return {"shared_seeds": []}

    def mean_delta(metric_name: str) -> float:
        return _mean(
            [float(left_by_seed[seed][metric_name]) - float(right_by_seed[seed][metric_name]) for seed in shared_seeds]
        )

    return {
        "shared_seeds": [int(seed) for seed in shared_seeds],
        "mean_val_accuracy_delta": mean_delta("val_accuracy"),
        "mean_test_accuracy_delta": mean_delta("test_accuracy"),
        "mean_gate_passing_epoch_count_delta": mean_delta("gate_passing_epoch_count"),
        "mean_seed_gate_positive_rate_delta": mean_delta("seed_gate_positive"),
        "mean_selected_epoch_passes_gate_rate_delta": mean_delta("selected_epoch_passes_gate"),
        "mean_selector_fallback_used_rate_delta": mean_delta("selector_fallback_used"),
        "mean_val_transported_final_energy_delta": mean_delta("val_transported_final_energy"),
        "mean_val_report_output_mse_delta": mean_delta("val_report_output_mse"),
        "mean_val_terminal_rowspace_rms_delta": mean_delta("val_terminal_rowspace_rms"),
        "mean_val_terminal_rowspace_fraction_delta": mean_delta("val_terminal_rowspace_fraction"),
        "mean_runtime_proxy_seconds_delta": mean_delta("runtime_proxy_seconds"),
    }


def _candidate_factor(case_name: str) -> str:
    if case_name == "preterminal_onpolicy_local_field_source_only":
        return "direction_source"
    if case_name == "preterminal_cached_onpolicy_handoff_only":
        return "handoff_state"
    if case_name == "preterminal_anchor_norm_only":
        return "norm_handling"
    raise ValueError(f"Unsupported candidate case_name '{case_name}'.")


def _diagnose_and_recommend(
    config: FMPCTF2PreterminalSourceLocalizationSuiteConfig,
    by_case: dict[str, dict[str, Any]],
) -> tuple[_Diagnosis, dict[str, Any], str]:
    failed_anchor = by_case["failed_penultimate_plus_terminal_live"]
    candidate_names = [
        "preterminal_onpolicy_local_field_source_only",
        "preterminal_anchor_norm_only",
        "preterminal_cached_onpolicy_handoff_only",
    ]

    ranked = sorted(
        candidate_names,
        key=lambda name: (
            float(by_case[name]["seed_gate_positive_rate"]),
            float(by_case[name]["selected_epoch_passes_gate_rate"]),
            float(by_case[name]["mean_gate_passing_epoch_count"]),
            -float(by_case[name]["selector_fallback_used_rate"]),
            -float(by_case[name]["mean_val_transported_final_energy"]),
            float(by_case[name]["mean_val_accuracy"]),
        ),
        reverse=True,
    )
    best_name = ranked[0]
    second_name = ranked[1]
    best = by_case[best_name]
    second = by_case[second_name]
    evidence = {
        "failed_anchor_reference": failed_anchor,
        "best_case": best_name,
        "best_case_factor": _candidate_factor(best_name),
        "ranked_candidates": ranked,
    }
    if (
        float(best["seed_gate_positive_rate"]) >= float(config.primary_seed_gate_positive_rate_threshold)
        and float(best["selected_epoch_passes_gate_rate"]) >= float(config.primary_selected_gate_rate_threshold)
        and (
            float(best["mean_gate_passing_epoch_count"]) - float(second["mean_gate_passing_epoch_count"])
            >= float(config.primary_gate_count_margin)
            or float(best["selected_epoch_passes_gate_rate"]) - float(second["selected_epoch_passes_gate_rate"])
            >= float(config.primary_rate_margin)
        )
    ):
        factor = _candidate_factor(best_name)
        if factor == "direction_source":
            return (
                "preterminal_direction_source_is_primary_blocker",
                evidence,
                "run one narrow adopted-package confirmation on the smallest preterminal live-local-field-source intervention that preserves the current selector/gate contract",
            )
        if factor == "handoff_state":
            return (
                "preterminal_handoff_state_is_primary_blocker",
                evidence,
                "run one narrow adopted-package confirmation on the smallest preterminal on-policy handoff reformulation that preserves the current selector/gate contract",
            )
        return (
            "preterminal_norm_handling_is_primary_blocker",
            evidence,
            "run one narrow adopted-package confirmation on the smallest preterminal norm-handling intervention that preserves the current selector/gate contract",
        )
    return (
        "mixed_preterminal_formulation_blocker",
        evidence,
        "run one narrow preterminal update formulation diagnostic on direction-source and handoff coupling rather than another cone-family sweep",
    )


def _make_run_dir(run_dir: Path, case_name: str, seed: int) -> Path:
    target = run_dir / "runs" / case_name / f"seed_{int(seed)}"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _write_run_artifacts(
    run_dir: Path,
    config: FMPCTF2Config,
    case_spec: _CaseSpec,
    epoch_rows: list[dict[str, Any]],
    selection_diagnostics: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    config_payload = fmpc_tf2_module._config_payload(config)
    config_payload["suite_case_name"] = case_spec.case_name
    config_payload["direction_source_mode"] = case_spec.direction_source_mode
    config_payload["norm_handling_mode"] = case_spec.norm_handling_mode
    config_payload["handoff_mode"] = case_spec.handoff_mode
    config_payload["diagnostic_only"] = True
    fmpc_tf2_module._write_json(run_dir / "config.json", config_payload)
    fmpc_tf2_module._write_epoch_metrics(run_dir / "epoch_metrics.csv", epoch_rows)
    fmpc_tf2_module._write_json(run_dir / "selection_diagnostics.json", selection_diagnostics)
    fmpc_tf2_module._write_json(run_dir / "summary.json", summary)


def _run_one_case_seed(
    suite_run_dir: Path,
    suite_config: FMPCTF2PreterminalSourceLocalizationSuiteConfig,
    case_spec: _CaseSpec,
    seed: int,
) -> dict[str, Any]:
    set_seed(int(seed))
    config = fmpc_tf2_module.build_tf2_corrective_transport_terminal_angleclip_default_config(
        experiment_name="tf2",
        output_root=suite_run_dir / "tf2_runs",
        output_layout="run_id_subdir",
        run_id=_run_id(case_spec.case_name, int(seed)),
        run_seed=int(seed),
        data_seed=int(seed),
        model_init_seed=int(seed),
        psi_init_seed=int(seed),
        batch_order_seed=int(seed),
        epochs=int(suite_config.epochs),
        batch_size=int(suite_config.batch_size),
        eval_steps=int(suite_config.eval_steps),
        layer_dims=tuple(suite_config.layer_dims),
        terminal_local_field_intervention_step_offsets=tuple(case_spec.intervention_step_offsets),
    )
    split = load_digits_split(
        split_seed=int(config.data_seed),
        train_fraction=float(config.train_fraction),
        val_fraction=float(config.val_fraction),
        test_fraction=float(config.test_fraction),
    )
    model = fmpc_tf2_module._make_pc_model(config)
    psi_network = fmpc_tf2_module._make_psi_network(config)

    epoch_rows: list[dict[str, Any]] = []
    epoch_snapshots: list[FMPCTF2EpochSnapshot] = []
    train_start = perf_counter()
    for epoch_index in range(int(config.epochs)):
        lambda_id = fmpc_tf2_module._lambda_id_for_epoch(config, epoch_index)
        stage = fmpc_tf2_module._stage_for_epoch(config, epoch_index)
        batch_losses: list[float] = []
        batch_boot_losses: list[float] = []
        batch_identity_losses: list[float] = []
        batch_transport_energies: list[float] = []
        batch_seed = int(config.batch_order_seed) + int(epoch_index)
        for x_batch, y_batch in iter_minibatches(
            split.x_train,
            split.y_train,
            int(config.batch_size),
            shuffle=bool(config.shuffle_batches),
            seed=batch_seed,
        ):
            train_loss, boot_loss, identity_loss, transported_energy = _train_one_batch_source_localization(
                model,
                psi_network,
                config,
                x_batch,
                y_batch,
                lambda_id=lambda_id,
                epoch_index=epoch_index,
                case_spec=case_spec,
            )
            batch_losses.append(train_loss)
            batch_boot_losses.append(boot_loss)
            batch_identity_losses.append(identity_loss)
            batch_transport_energies.append(transported_energy)

        val_transport = fmpc_tf2_module._evaluate_transport_split(
            model,
            psi_network,
            config,
            split.x_val,
            split.y_val,
        )
        _, val_accuracy = fmpc_tf2_module._evaluate_slow_pc_accuracy(model, split.x_val, split.y_val)
        val_baseline_accuracy = majority_class_baseline_accuracy(split.y_val)
        val_energy_delta_vs_identity = (
            val_transport.transported_final_energy - val_transport.identity_final_energy
        )
        val_energy_delta_vs_local_field_only = (
            val_transport.transported_final_energy - val_transport.local_field_only_final_energy
        )
        epoch_rows.append(
            asdict(
                FMPCTF2EpochMetrics(
                    epoch=epoch_index + 1,
                    lambda_id=float(lambda_id),
                    stage=stage,
                    train_loss=float(np.mean(batch_losses)),
                    train_boot_loss=float(np.mean(batch_boot_losses)),
                    train_identity_loss=float(np.mean(batch_identity_losses)),
                    train_transported_final_energy=float(np.mean(batch_transport_energies)),
                    val_transported_final_energy=val_transport.transported_final_energy,
                    val_identity_final_energy=val_transport.identity_final_energy,
                    val_local_field_only_final_energy=val_transport.local_field_only_final_energy,
                    val_energy_delta_vs_identity=float(val_energy_delta_vs_identity),
                    val_energy_delta_vs_local_field_only=float(val_energy_delta_vs_local_field_only),
                    val_accuracy=val_accuracy,
                    val_baseline_accuracy=val_baseline_accuracy,
                )
            )
        )
        epoch_snapshots.append(
            FMPCTF2EpochSnapshot(
                epoch=epoch_index + 1,
                model_snapshot=fmpc_tf2_module._snapshot_pc_parameters(model),
                psi_snapshot=fmpc_tf2_module._snapshot_mlp_parameters(psi_network),
            )
        )
    train_wall_time_seconds = float(perf_counter() - train_start)

    selection_diagnostics = build_tf1_epoch_selection_diagnostics(epoch_rows)
    checkpoint_selection = _select_tf1_checkpoint_epoch(
        epoch_rows,
        config.checkpoint_selector,
        selection_diagnostics=selection_diagnostics,
    )
    selected_epoch = int(checkpoint_selection["selected_epoch"])
    selected_snapshot = next(snapshot for snapshot in epoch_snapshots if int(snapshot.epoch) == int(selected_epoch))
    fmpc_tf2_module._restore_pc_parameters(model, selected_snapshot.model_snapshot)
    fmpc_tf2_module._restore_mlp_parameters(psi_network, selected_snapshot.psi_snapshot)

    evaluation_start = perf_counter()
    val_transport = fmpc_tf2_module._evaluate_transport_split(model, psi_network, config, split.x_val, split.y_val)
    test_transport = fmpc_tf2_module._evaluate_transport_split(model, psi_network, config, split.x_test, split.y_test)
    val_loss, val_accuracy = fmpc_tf2_module._evaluate_slow_pc_accuracy(model, split.x_val, split.y_val)
    test_loss, test_accuracy = fmpc_tf2_module._evaluate_slow_pc_accuracy(model, split.x_test, split.y_test)
    evaluation_wall_time_seconds = float(perf_counter() - evaluation_start)
    val_baseline_accuracy = majority_class_baseline_accuracy(split.y_val)
    val_energy_delta_vs_identity = val_transport.transported_final_energy - val_transport.identity_final_energy
    val_energy_delta_vs_local_field_only = (
        val_transport.transported_final_energy - val_transport.local_field_only_final_energy
    )

    terminal_metrics = _terminal_rowspace_metrics(model, psi_network, config, split.x_val, split.y_val)
    summary = {
        "phase": "FMPC Stage 04 Incremental Bridge",
        "stage": "ifmpc_bridge_stage",
        "preset_name": config.preset_name,
        "suite_case_name": case_spec.case_name,
        "direction_source_mode": case_spec.direction_source_mode,
        "norm_handling_mode": case_spec.norm_handling_mode,
        "handoff_mode": case_spec.handoff_mode,
        "teacher_free": True,
        "uses_teacher_artifacts": False,
        "jpc_runtime_dependency": False,
        "family_lineage": config.family_lineage,
        "feature_aware_tangents": bool(config.feature_aware_tangents),
        "psi_family": config.psi_family,
        "time_encoding_variant": config.time_encoding_variant,
        "terminal_local_field_direction_intervention": config.terminal_local_field_direction_intervention,
        "terminal_local_field_angle_clip_degrees": float(config.terminal_local_field_angle_clip_degrees),
        "terminal_local_field_intervention_step_offsets": [
            int(value) for value in config.terminal_local_field_intervention_step_offsets
        ],
        "checkpoint_selector": config.checkpoint_selector,
        "selection_metric_source": "val_metric",
        "report_metric_source": "test_metric",
        "best_epoch": int(selected_epoch),
        "selected_epoch_passes_gate": bool(checkpoint_selection["selected_epoch_passes_gate"]),
        "gate_passing_epoch_count": int(checkpoint_selection["gate_passing_epoch_count"]),
        "selector_fallback_used": bool(checkpoint_selection["selector_fallback_used"]),
        "selected_epoch_selection_reason": str(checkpoint_selection["selected_epoch_selection_reason"]),
        "val_transported_final_energy": float(val_transport.transported_final_energy),
        "test_transported_final_energy": float(test_transport.transported_final_energy),
        "val_energy_delta_vs_identity": float(val_energy_delta_vs_identity),
        "val_energy_delta_vs_local_field_only": float(val_energy_delta_vs_local_field_only),
        "val_accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy),
        "val_loss": float(val_loss),
        "test_loss": float(test_loss),
        "val_baseline_accuracy": float(val_baseline_accuracy),
        "val_terminal_rowspace_rms": float(terminal_metrics["terminal_rowspace_rms"]),
        "val_terminal_rowspace_fraction": float(terminal_metrics["terminal_rowspace_fraction"]),
        "timing": {
            "train_wall_time_seconds": train_wall_time_seconds,
            "final_evaluation_wall_time_seconds": evaluation_wall_time_seconds,
        },
        "validation_gate": {
            "passes_identity_comparison": bool(
                val_transport.transported_final_energy < val_transport.identity_final_energy
            ),
            "passes_local_field_only_comparison": bool(
                val_transport.transported_final_energy <= val_transport.local_field_only_final_energy
            ),
            "passes_majority_baseline_accuracy": bool(val_accuracy > val_baseline_accuracy),
        },
    }
    per_run_dir = _make_run_dir(suite_run_dir, case_spec.case_name, int(seed))
    _write_run_artifacts(per_run_dir, config, case_spec, epoch_rows, selection_diagnostics, summary)
    return _success_row(
        suite_run_dir,
        case_spec,
        int(seed),
        summary,
        val_terminal_rowspace_rms=terminal_metrics["terminal_rowspace_rms"],
        val_terminal_rowspace_fraction=terminal_metrics["terminal_rowspace_fraction"],
    )


def run_fmpc_tf2_preterminal_source_localization_suite(
    config: FMPCTF2PreterminalSourceLocalizationSuiteConfig,
) -> FMPCTF2PreterminalSourceLocalizationSuiteRunResult:
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
    for case_spec in config.case_specs():
        for seed in config.seeds:
            try:
                aggregate_rows.append(_run_one_case_seed(run_dir, config, case_spec, int(seed)))
            except Exception as error:  # pragma: no cover - recorded for diagnostic robustness
                aggregate_rows.append(_failure_row(case_spec, int(seed), error))

    _write_csv(run_dir / "aggregate_runs.csv", aggregate_rows)

    by_case: dict[str, dict[str, Any]] = {}
    pairwise_vs_failed_anchor: dict[str, dict[str, Any]] = {}
    pairwise_vs_control: dict[str, dict[str, Any]] = {}
    failed_anchor_rows = _case_rows(aggregate_rows, "failed_penultimate_plus_terminal_live")
    control_rows = _case_rows(aggregate_rows, "adopted_control_terminal_only")
    for case_spec in config.case_specs():
        case_rows_all = [row for row in aggregate_rows if str(row["case_name"]) == case_spec.case_name]
        case_rows = _case_rows(aggregate_rows, case_spec.case_name)
        by_case[case_spec.case_name] = _case_summary(case_rows, case_rows_all)
        if case_spec.case_name != "failed_penultimate_plus_terminal_live":
            pairwise_vs_failed_anchor[case_spec.case_name] = _pairwise_delta(case_rows, failed_anchor_rows)
        if case_spec.case_name != "adopted_control_terminal_only":
            pairwise_vs_control[case_spec.case_name] = _pairwise_delta(case_rows, control_rows)

    diagnosis, diagnosis_evidence, recommended_next_move = _diagnose_and_recommend(config, by_case)
    summary = {
        "phase": "FMPC Stage 04 Incremental Bridge",
        "stage": "adopted_package_preterminal_update_source_localization",
        "num_runs": len(aggregate_rows),
        "by_case": by_case,
        "pairwise_vs_failed_anchor": pairwise_vs_failed_anchor,
        "pairwise_vs_control": pairwise_vs_control,
        "diagnosis": diagnosis,
        "diagnosis_evidence": diagnosis_evidence,
        "recommended_next_narrow_tf2_move": recommended_next_move,
        "aggregate_runs_csv_path": "aggregate_runs.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF2PreterminalSourceLocalizationSuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config),
        aggregate_rows=aggregate_rows,
        summary=summary,
    )
