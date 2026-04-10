from __future__ import annotations

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
from .fmpc_tf2_partial_open_loop_handoff_suite import _ReplayPlanSlot, _build_cached_plan
from .fmpc_tf2_preterminal_source_localization_suite import (
    _apply_same_geometry_full_vector_clip,
    _case_rows,
    _case_summary,
    _failure_row,
    _pairwise_delta,
    _prepare_run_dir,
    _preterminal_step_index,
    _relative_posix,
    _resolve_run_dir,
    _runtime_proxy_seconds,
    _step_indices,
    _terminal_rowspace_metrics,
    _write_csv,
    _write_json,
    _write_run_artifacts,
)
from .fmpc_tf2_successor_value_confirmation_suite import _blend_successor_value
from ..metrics import majority_class_baseline_accuracy
from ..minibatch import iter_minibatches
from ..utils import ensure_finite_array, set_seed


_SuccessorValueMode = Literal[
    "live_successor_value",
    "blend_successor_value_alpha_025",
    "live_carry_cached_increment",
    "cached_carry_live_increment",
]
_Diagnosis = Literal[
    "live_successor_increment_is_primary_blocker",
    "live_predecessor_carry_is_primary_blocker",
    "carry_increment_interaction_is_primary_blocker",
]


@dataclass(frozen=True)
class _CaseSpec:
    case_name: str
    description: str
    intervention_step_offsets: tuple[int, ...]
    successor_value_mode: _SuccessorValueMode = "live_successor_value"


@dataclass
class FMPCTF2SuccessorValueSourceSuiteConfig:
    """Run a very narrow source-localization diagnostic inside live successor-value formulation."""

    experiment_name: str = "fmpc_tf2_successor_value_source_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 60
    batch_size: int = 128
    eval_steps: int = 15
    layer_dims: tuple[int, ...] = (64, 64, 10)
    full_gate_seed_gate_positive_rate: float = 1.0
    full_gate_selected_epoch_passes_gate_rate: float = 1.0
    full_gate_selector_fallback_used_rate: float = 0.0

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
                description="Higher-gain non-adopted reference: same adopted full-vector hard 30 degree cone on the penultimate and terminal micro-steps with the full live successor value.",
                intervention_step_offsets=(-2, -1),
                successor_value_mode="live_successor_value",
            ),
            _CaseSpec(
                case_name="safe_blended_successor_value_alpha_025",
                description="Current best narrow safe reference: penultimate successor value blended 25% live / 75% cached.",
                intervention_step_offsets=(-2, -1),
                successor_value_mode="blend_successor_value_alpha_025",
            ),
            _CaseSpec(
                case_name="live_carry_cached_increment",
                description="Keep the live predecessor carry state z_on_k and replace only the successor increment with its cached analogue.",
                intervention_step_offsets=(-2, -1),
                successor_value_mode="live_carry_cached_increment",
            ),
            _CaseSpec(
                case_name="cached_carry_live_increment",
                description="Replace only the predecessor carry state with the cached analogue while keeping the successor increment live/on-policy.",
                intervention_step_offsets=(-2, -1),
                successor_value_mode="cached_carry_live_increment",
            ),
        )


@dataclass
class FMPCTF2SuccessorValueSourceSuiteRunResult:
    run_dir: Path
    config: dict[str, Any]
    aggregate_rows: list[dict[str, Any]]
    summary: dict[str, Any]


def _successor_value_component_table() -> list[dict[str, Any]]:
    return [
        {
            "component_name": "z_on_k_live",
            "where_produced": "batch rollout loop carry state before calling _plan_tf2_micro_step",
            "source_type": "live_on_policy",
            "used_for": "predecessor carry state in z_on_next = z_on_k + Δ_on",
        },
        {
            "component_name": "z_on_next_live",
            "where_produced": "_plan_tf2_micro_step: z_on_k + dt * on_velocity, optionally stabilized by the existing full-vector angle clip",
            "source_type": "live_on_policy",
            "used_for": "preterminal successor value handed to the next micro-step",
        },
        {
            "component_name": "Δ_on_live",
            "where_produced": "derived inside this suite as z_on_next_live - z_on_k_live",
            "source_type": "live_on_policy",
            "used_for": "successor increment component of the live preterminal successor-value formulation",
        },
        {
            "component_name": "z_on_k_cached",
            "where_produced": "_build_cached_plan(...).z_on_k",
            "source_type": "cached_batch_start_reference",
            "used_for": "cached predecessor carry analogue",
        },
        {
            "component_name": "z_on_next_cached",
            "where_produced": "_build_cached_plan(...).z_on_next",
            "source_type": "cached_batch_start_reference",
            "used_for": "cached successor value analogue",
        },
        {
            "component_name": "Δ_on_cached",
            "where_produced": "derived inside this suite as z_on_next_cached - z_on_k_cached",
            "source_type": "cached_batch_start_reference",
            "used_for": "cached successor increment analogue",
        },
    ]


def _suite_config_payload(config: FMPCTF2SuccessorValueSourceSuiteConfig) -> dict[str, Any]:
    return {
        "phase": "FMPC Stage 04 Incremental Bridge",
        "stage": "adopted_package_successor_value_source_localization",
        "base_preset_name": "tf2_corrective_transport_terminal_angleclip_default",
        "transport_family_fixed": True,
        "candidate_specs": [
            {
                "case_name": spec.case_name,
                "description": spec.description,
                "intervention_step_offsets": [int(value) for value in spec.intervention_step_offsets],
                "successor_value_mode": spec.successor_value_mode,
            }
            for spec in config.case_specs()
        ],
        "successor_value_component_table": _successor_value_component_table(),
        "seeds": [int(seed) for seed in config.seeds],
        "selector_contract_fixed": True,
        "diagnostic_only": True,
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
        "run_id": f"{case_spec.case_name}_s{int(seed)}",
        "run_summary_path": _relative_posix(
            run_dir,
            run_dir / "runs" / case_spec.case_name / f"seed_{int(seed)}" / "summary.json",
        ),
        "intervention_step_offsets": ",".join(str(value) for value in case_spec.intervention_step_offsets),
        "successor_value_mode": case_spec.successor_value_mode,
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


def _recovery_fraction_higher_is_better(control: float, anchor: float, candidate: float) -> float:
    denominator = float(anchor) - float(control)
    if abs(denominator) <= 1e-12:
        return 0.0
    return (float(candidate) - float(control)) / denominator


def _recovery_fraction_lower_is_better(control: float, anchor: float, candidate: float) -> float:
    denominator = float(control) - float(anchor)
    if abs(denominator) <= 1e-12:
        return 0.0
    return (float(control) - float(candidate)) / denominator


def _gate_recovery_fraction(control: float, failed_anchor: float, candidate: float) -> float:
    denominator = float(control) - float(failed_anchor)
    if abs(denominator) <= 1e-12:
        return 0.0
    return (float(candidate) - float(failed_anchor)) / denominator


def _case_recovery_metrics(
    control: dict[str, Any],
    failed_anchor: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "accuracy_gain_retained_fraction_from_control_to_failed_anchor": _recovery_fraction_higher_is_better(
            float(control["mean_val_accuracy"]),
            float(failed_anchor["mean_val_accuracy"]),
            float(candidate["mean_val_accuracy"]),
        ),
        "gate_robustness_recovery_fraction_from_failed_anchor_to_control": _gate_recovery_fraction(
            float(control["mean_gate_passing_epoch_count"]),
            float(failed_anchor["mean_gate_passing_epoch_count"]),
            float(candidate["mean_gate_passing_epoch_count"]),
        ),
        "terminal_rowspace_rms_gain_retained_fraction_from_control_to_failed_anchor": _recovery_fraction_lower_is_better(
            float(control["mean_val_terminal_rowspace_rms"]),
            float(failed_anchor["mean_val_terminal_rowspace_rms"]),
            float(candidate["mean_val_terminal_rowspace_rms"]),
        ),
    }


def _gate_contract_intact(
    config: FMPCTF2SuccessorValueSourceSuiteConfig,
    case_summary: dict[str, Any],
) -> bool:
    return (
        float(case_summary["seed_gate_positive_rate"]) >= float(config.full_gate_seed_gate_positive_rate)
        and float(case_summary["selected_epoch_passes_gate_rate"])
        >= float(config.full_gate_selected_epoch_passes_gate_rate)
        and float(case_summary["selector_fallback_used_rate"]) <= float(config.full_gate_selector_fallback_used_rate)
    )


def _reformulate_successor_value(
    mode: _SuccessorValueMode,
    *,
    live_z_on_k: np.ndarray,
    live_z_on_next: np.ndarray,
    cached_slot: _ReplayPlanSlot,
) -> np.ndarray:
    live_carry = np.asarray(live_z_on_k, dtype=np.float64)
    live_successor = np.asarray(live_z_on_next, dtype=np.float64)
    live_increment = live_successor - live_carry
    cached_carry = np.asarray(cached_slot.z_on_k, dtype=np.float64)
    cached_successor = np.asarray(cached_slot.z_on_next, dtype=np.float64)
    cached_increment = cached_successor - cached_carry
    if mode == "live_successor_value":
        result = live_successor
    elif mode == "blend_successor_value_alpha_025":
        result = _blend_successor_value(live_successor, cached_successor, 0.25)
    elif mode == "live_carry_cached_increment":
        result = live_carry + cached_increment
    elif mode == "cached_carry_live_increment":
        result = cached_carry + live_increment
    else:  # pragma: no cover
        raise ValueError(f"Unsupported successor_value_mode '{mode}'.")
    ensure_finite_array(result, f"tf2_{mode}_z_on_next")
    return np.asarray(result, dtype=np.float64, copy=False)


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
        and case_spec.successor_value_mode != "live_successor_value"
    )

    if apply_direction_intervention and config.terminal_local_field_direction_intervention != "none":
        if is_custom_preterminal_step:
            effective_prediction, z_on_next = _apply_same_geometry_full_vector_clip(
                context,
                plan,
                config,
                z_on_k,
                dt,
                direction_source_mode="detached_local_field",
                norm_handling_mode="keep_live_norm",
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

    if is_custom_preterminal_step:
        if cached_slot is None:
            raise ValueError("cached_slot is required for successor-value source localization.")
        z_on_next = _reformulate_successor_value(
            case_spec.successor_value_mode,
            live_z_on_k=z_on_k,
            live_z_on_next=z_on_next,
            cached_slot=cached_slot,
        )

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


def _train_one_batch_successor_value_source(
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
        if case_spec.case_name != "adopted_control_terminal_only"
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
        if case_spec.case_name == "adopted_control_terminal_only":
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


def _make_run_dir(run_dir: Path, case_name: str, seed: int) -> Path:
    target = run_dir / "runs" / case_name / f"seed_{int(seed)}"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _run_one_case_seed(
    suite_run_dir: Path,
    suite_config: FMPCTF2SuccessorValueSourceSuiteConfig,
    case_spec: _CaseSpec,
    seed: int,
) -> dict[str, Any]:
    set_seed(int(seed))
    config = fmpc_tf2_module.build_tf2_corrective_transport_terminal_angleclip_default_config(
        experiment_name="tf2",
        output_root=suite_run_dir / "tf2_runs",
        output_layout="run_id_subdir",
        run_id=f"{case_spec.case_name}_s{int(seed)}",
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
            train_loss, boot_loss, identity_loss, transported_energy = _train_one_batch_successor_value_source(
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
        val_energy_delta_vs_identity = val_transport.transported_final_energy - val_transport.identity_final_energy
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
        "successor_value_mode": case_spec.successor_value_mode,
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
    _write_run_artifacts(
        per_run_dir,
        config,
        type(
            "_CompatCaseSpec",
            (),
            {
                "case_name": case_spec.case_name,
                "direction_source_mode": "detached_local_field",
                "norm_handling_mode": "keep_live_norm",
                "handoff_mode": case_spec.successor_value_mode,
            },
        )(),
        epoch_rows,
        selection_diagnostics,
        summary,
    )
    return _success_row(
        suite_run_dir,
        case_spec,
        int(seed),
        summary,
        val_terminal_rowspace_rms=terminal_metrics["terminal_rowspace_rms"],
        val_terminal_rowspace_fraction=terminal_metrics["terminal_rowspace_fraction"],
    )


def _diagnose_and_recommend(
    config: FMPCTF2SuccessorValueSourceSuiteConfig,
    by_case: dict[str, dict[str, Any]],
    recovery_by_case: dict[str, dict[str, Any]],
) -> tuple[_Diagnosis, dict[str, Any], str]:
    control = by_case["adopted_control_terminal_only"]
    failed_anchor = by_case["failed_penultimate_plus_terminal_live"]
    safe_anchor = by_case["safe_blended_successor_value_alpha_025"]
    increment_candidate = by_case["live_carry_cached_increment"]
    carry_candidate = by_case["cached_carry_live_increment"]
    increment_gate = _gate_contract_intact(config, increment_candidate)
    carry_gate = _gate_contract_intact(config, carry_candidate)
    evidence = {
        "control_case": control,
        "failed_anchor_case": failed_anchor,
        "safe_anchor_case": safe_anchor,
        "increment_candidate_case": increment_candidate,
        "carry_candidate_case": carry_candidate,
        "increment_candidate_recovery": recovery_by_case["live_carry_cached_increment"],
        "carry_candidate_recovery": recovery_by_case["cached_carry_live_increment"],
    }
    if increment_gate and not carry_gate:
        return (
            "live_successor_increment_is_primary_blocker",
            evidence,
            "run one confirmation-level reformulation on the preterminal successor increment only, without reopening any cone-family or broad blend sweep",
        )
    if carry_gate and not increment_gate:
        return (
            "live_predecessor_carry_is_primary_blocker",
            evidence,
            "run one confirmation-level reformulation on the preterminal predecessor carry state only, without reopening any cone-family or broad blend sweep",
        )
    return (
        "carry_increment_interaction_is_primary_blocker",
        evidence,
        "run one minimal interaction-preserving reformulation diagnostic next, keeping the work on the same preterminal successor-value component only",
    )


def run_fmpc_tf2_successor_value_source_suite(
    config: FMPCTF2SuccessorValueSourceSuiteConfig,
) -> FMPCTF2SuccessorValueSourceSuiteRunResult:
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
            except Exception as error:  # pragma: no cover
                aggregate_rows.append(
                    _failure_row(
                        type(
                            "_CompatFailureCaseSpec",
                            (),
                            {
                                "case_name": case_spec.case_name,
                                "intervention_step_offsets": case_spec.intervention_step_offsets,
                                "direction_source_mode": "detached_local_field",
                                "norm_handling_mode": "keep_live_norm",
                                "handoff_mode": case_spec.successor_value_mode,
                            },
                        )(),
                        int(seed),
                        error,
                    )
                )

    _write_csv(run_dir / "aggregate_runs.csv", aggregate_rows)

    by_case: dict[str, dict[str, Any]] = {}
    pairwise_vs_control: dict[str, dict[str, Any]] = {}
    pairwise_vs_failed_anchor: dict[str, dict[str, Any]] = {}
    recovery_fractions_vs_failed_anchor: dict[str, dict[str, Any]] = {}
    control_rows = _case_rows(aggregate_rows, "adopted_control_terminal_only")
    failed_anchor_rows = _case_rows(aggregate_rows, "failed_penultimate_plus_terminal_live")
    for case_spec in config.case_specs():
        case_rows_all = [row for row in aggregate_rows if str(row["case_name"]) == case_spec.case_name]
        case_rows = _case_rows(aggregate_rows, case_spec.case_name)
        by_case[case_spec.case_name] = _case_summary(case_rows, case_rows_all)
        if case_spec.case_name != "adopted_control_terminal_only":
            pairwise_vs_control[case_spec.case_name] = _pairwise_delta(case_rows, control_rows)
        if case_spec.case_name != "failed_penultimate_plus_terminal_live":
            pairwise_vs_failed_anchor[case_spec.case_name] = _pairwise_delta(case_rows, failed_anchor_rows)

    control_summary = by_case["adopted_control_terminal_only"]
    failed_anchor_summary = by_case["failed_penultimate_plus_terminal_live"]
    for case_name in (
        "safe_blended_successor_value_alpha_025",
        "live_carry_cached_increment",
        "cached_carry_live_increment",
    ):
        recovery_fractions_vs_failed_anchor[case_name] = _case_recovery_metrics(
            control_summary,
            failed_anchor_summary,
            by_case[case_name],
        )

    diagnosis, diagnosis_evidence, recommended_next_move = _diagnose_and_recommend(
        config,
        by_case,
        recovery_fractions_vs_failed_anchor,
    )
    summary = {
        "phase": "FMPC Stage 04 Incremental Bridge",
        "stage": "adopted_package_successor_value_source_localization",
        "num_runs": len(aggregate_rows),
        "successor_value_component_table": _successor_value_component_table(),
        "by_case": by_case,
        "pairwise_vs_control": pairwise_vs_control,
        "pairwise_vs_failed_anchor": pairwise_vs_failed_anchor,
        "recovery_fractions_vs_failed_anchor": recovery_fractions_vs_failed_anchor,
        "diagnosis": diagnosis,
        "diagnosis_evidence": diagnosis_evidence,
        "recommended_next_narrow_tf2_move": recommended_next_move,
        "aggregate_runs_csv_path": "aggregate_runs.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF2SuccessorValueSourceSuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config),
        aggregate_rows=aggregate_rows,
        summary=summary,
    )
