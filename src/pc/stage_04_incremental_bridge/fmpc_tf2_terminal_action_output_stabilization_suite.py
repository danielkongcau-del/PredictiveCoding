from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np

from . import fmpc_tf2 as fmpc_tf2_module
from ..datasets import load_digits_split
from ..stage_03_transport_core_v1.fmpc_tf1 import _select_tf1_checkpoint_epoch, build_tf1_epoch_selection_diagnostics
from .fmpc_tf2 import FMPCTF2Config, FMPCTF2EpochMetrics
from .fmpc_tf2_partial_open_loop_handoff_suite import (
    FMPCTF2PartialHandoffCandidate,
    _ReplayPlanSlot,
    _aggregate_run_group,
    _build_cached_plan,
    _build_candidate_config,
    _candidate_registry,
    _candidate_replay_inputs,
    _pairwise_vs_reference,
    _relative_posix,
)
from ..metrics import majority_class_baseline_accuracy

TerminalActionMode = Literal[
    "closed_loop_live_plan",
    "open_loop_baseline_plan_replay",
    "terminal_live_action_raw",
    "terminal_replay_action_vector_only",
    "terminal_replay_action_direction_only",
    "terminal_replay_action_norm_only",
    "terminal_replay_post_action_state_only",
    "k4_open_loop_baseline_plan_replay",
]
ActionInterventionMode = Literal[
    "closed_loop_live_plan",
    "open_loop_baseline_plan_replay",
    "live_action_raw",
    "replay_action_vector_only",
    "replay_action_direction_only",
    "replay_action_norm_only",
    "replay_post_action_state_only",
    "full_replay",
]


@dataclass(frozen=True)
class _CellSpec:
    candidate_key: str
    mode: TerminalActionMode
    replay_prefix_steps: int
    action_intervention_mode: ActionInterventionMode


@dataclass
class FMPCTF2TerminalActionOutputStabilizationSuiteConfig:
    """Diagnostic-only terminal-step action/output stabilization suite."""

    experiment_name: str = "fmpc_tf2_terminal_action_output_stabilization_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: Literal["single_dir", "run_id_subdir"] = "single_dir"
    seeds: tuple[int, ...] = (0, 1, 2)
    epochs: int = 60
    batch_size: int = 128
    eval_steps: int = 15
    layer_dims: tuple[int, ...] = (64, 64, 10)
    material_test_gain: float = 0.005
    baseline_similarity_tolerance: float = 0.003

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"


@dataclass
class FMPCTF2TerminalActionOutputStabilizationSuiteRunResult:
    run_dir: Path
    config: dict[str, Any]
    run_rows: list[dict[str, Any]]
    terminal_drift_rows: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass
class _RunArtifacts:
    mode: TerminalActionMode
    candidate: FMPCTF2PartialHandoffCandidate
    seed: int
    config: FMPCTF2Config
    run_dir: Path
    epoch_rows: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass
class _CellRuntime:
    spec: _CellSpec
    candidate: FMPCTF2PartialHandoffCandidate
    config: FMPCTF2Config
    run_dir: Path
    model: Any
    psi_network: Any
    epoch_rows: list[dict[str, Any]] = field(default_factory=list)
    epoch_snapshots: list[Any] = field(default_factory=list)
    train_wall_time_seconds: float = 0.0


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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Mean requires at least one value.")
    return float(sum(values) / float(len(values)))


def _baseline_specs() -> list[_CellSpec]:
    return [
        _CellSpec(
            candidate_key="baseline_plain_raw",
            mode="closed_loop_live_plan",
            replay_prefix_steps=0,
            action_intervention_mode="closed_loop_live_plan",
        ),
        _CellSpec(
            candidate_key="baseline_plain_raw",
            mode="open_loop_baseline_plan_replay",
            replay_prefix_steps=4,
            action_intervention_mode="open_loop_baseline_plan_replay",
        ),
    ]


def _challenger_specs() -> list[_CellSpec]:
    return [
        _CellSpec(
            candidate_key="residualized_local_field_poly_rt2",
            mode="terminal_live_action_raw",
            replay_prefix_steps=3,
            action_intervention_mode="live_action_raw",
        ),
        _CellSpec(
            candidate_key="residualized_local_field_poly_rt2",
            mode="terminal_replay_action_vector_only",
            replay_prefix_steps=3,
            action_intervention_mode="replay_action_vector_only",
        ),
        _CellSpec(
            candidate_key="residualized_local_field_poly_rt2",
            mode="terminal_replay_action_direction_only",
            replay_prefix_steps=3,
            action_intervention_mode="replay_action_direction_only",
        ),
        _CellSpec(
            candidate_key="residualized_local_field_poly_rt2",
            mode="terminal_replay_action_norm_only",
            replay_prefix_steps=3,
            action_intervention_mode="replay_action_norm_only",
        ),
        _CellSpec(
            candidate_key="residualized_local_field_poly_rt2",
            mode="terminal_replay_post_action_state_only",
            replay_prefix_steps=3,
            action_intervention_mode="replay_post_action_state_only",
        ),
        _CellSpec(
            candidate_key="residualized_local_field_poly_rt2",
            mode="k4_open_loop_baseline_plan_replay",
            replay_prefix_steps=4,
            action_intervention_mode="full_replay",
        ),
    ]


def _cell_specs() -> list[_CellSpec]:
    return [*_baseline_specs(), *_challenger_specs()]


def _suite_config_payload(
    config: FMPCTF2TerminalActionOutputStabilizationSuiteConfig,
    candidates: dict[str, FMPCTF2PartialHandoffCandidate],
) -> dict[str, Any]:
    return {
        "phase": "FMPC Stage 04 Incremental Bridge",
        "stage": "terminal_action_output_stabilization_suite",
        "seeds": [int(seed) for seed in config.seeds],
        "candidates": [
            {
                "key": candidate.key,
                "psi_family": candidate.psi_family,
                "time_encoding_variant": candidate.time_encoding_variant,
                "notes": candidate.notes,
            }
            for candidate in candidates.values()
        ],
        "baseline_controls": [spec.mode for spec in _baseline_specs()],
        "challenger_modes": [spec.mode for spec in _challenger_specs()],
        "fixed": {
            "preset_name": "tf2_corrective_transport_default",
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
        "diagnostic_only": True,
        "material_test_gain": float(config.material_test_gain),
        "baseline_similarity_tolerance": float(config.baseline_similarity_tolerance),
        "terminal_definition": (
            "All challenger modes replay cached baseline steps 0-2 and keep the terminal supervision "
            "bundle fixed to the cached baseline step-3 bundle while varying only the terminal action "
            "or post-action next-state intervention."
        ),
    }


def _relative_target_delta(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_array = np.asarray(reference, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    diff_norm = np.linalg.norm(reference_array - candidate_array, axis=1)
    reference_norm = np.linalg.norm(reference_array, axis=1)
    scale = np.maximum(reference_norm, 1e-12)
    return float(np.mean(diff_norm / scale))


def _mean_cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_array = np.asarray(lhs, dtype=np.float64)
    rhs_array = np.asarray(rhs, dtype=np.float64)
    numerator = np.sum(lhs_array * rhs_array, axis=1)
    lhs_norm = np.linalg.norm(lhs_array, axis=1)
    rhs_norm = np.linalg.norm(rhs_array, axis=1)
    denominator = lhs_norm * rhs_norm
    cosine = np.ones_like(numerator, dtype=np.float64)
    valid = denominator > 1e-12
    cosine[valid] = numerator[valid] / denominator[valid]
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.mean(cosine))


def _mean_state_displacement(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_array = np.asarray(reference, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    return float(np.mean(np.linalg.norm(reference_array - candidate_array, axis=1)))


def _make_cell(
    base_run_dir: Path,
    candidate: FMPCTF2PartialHandoffCandidate,
    spec: _CellSpec,
    *,
    seed: int,
    suite_config: FMPCTF2TerminalActionOutputStabilizationSuiteConfig,
) -> _CellRuntime:
    config = _build_candidate_config(candidate, seed=seed, suite_config=suite_config)
    run_dir = base_run_dir / "runs" / spec.mode / candidate.key / f"seed_{int(seed)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return _CellRuntime(
        spec=spec,
        candidate=candidate,
        config=config,
        run_dir=run_dir,
        model=fmpc_tf2_module._make_pc_model(config),
        psi_network=fmpc_tf2_module._make_psi_network(config),
    )


def _action_from_step(z_k: np.ndarray, z_next: np.ndarray, dt: float) -> np.ndarray:
    return (np.asarray(z_next, dtype=np.float64) - np.asarray(z_k, dtype=np.float64)) / float(dt)


def _vector_norms(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(x, dtype=np.float64), axis=1)


def _safe_direction(x: np.ndarray) -> np.ndarray:
    array = np.asarray(x, dtype=np.float64)
    norms = _vector_norms(array)[:, None]
    return np.divide(array, np.maximum(norms, 1e-12), out=np.zeros_like(array), where=norms > 1e-12)


def _action_metrics(
    *,
    baseline_action: np.ndarray,
    mode_action: np.ndarray,
    live_action: np.ndarray,
    baseline_slot: _ReplayPlanSlot,
    boot_targets: np.ndarray,
    identity_targets: np.ndarray,
    lambda_id: float,
    mode_z_on_next: np.ndarray,
    live_z_on_next: np.ndarray,
) -> dict[str, float]:
    baseline_norm = _vector_norms(baseline_action)
    mode_norm = _vector_norms(mode_action)
    norm_ratio = np.divide(mode_norm, np.maximum(baseline_norm, 1e-12))
    return {
        "relative_terminal_action_vector_delta_vs_replay": _relative_target_delta(baseline_action, mode_action),
        "terminal_action_direction_cosine_similarity_vs_replay": _mean_cosine_similarity(
            _safe_direction(baseline_action),
            _safe_direction(mode_action),
        ),
        "terminal_action_norm_ratio_vs_replay": float(np.mean(norm_ratio)),
        "terminal_action_norm_delta_vs_replay": float(np.mean(np.abs(mode_norm - baseline_norm))),
        "relative_terminal_bootstrap_target_delta": _relative_target_delta(
            baseline_slot.boot_targets,
            boot_targets,
        ),
        "relative_terminal_identity_target_delta": _relative_target_delta(
            baseline_slot.identity_targets,
            identity_targets,
        ),
        "terminal_lambda_id_delta": abs(float(lambda_id) - float(baseline_slot.lambda_id)),
        "terminal_state_slot_displacement_after_action": _mean_state_displacement(
            live_z_on_next,
            mode_z_on_next,
        ),
        "terminal_next_state_displacement_vs_baseline_replay": _mean_state_displacement(
            baseline_slot.z_on_next,
            mode_z_on_next,
        ),
    }


def _write_run_artifacts(
    run_dir: Path,
    config: FMPCTF2Config,
    *,
    spec: _CellSpec,
    epoch_rows: list[dict[str, Any]],
    selection_diagnostics: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    config_payload = fmpc_tf2_module._config_payload(config)
    config_payload["handoff_mode"] = spec.mode
    config_payload["replay_prefix_steps"] = int(spec.replay_prefix_steps)
    config_payload["action_intervention_mode"] = spec.action_intervention_mode
    config_payload["diagnostic_only"] = True
    if "baseline_plan_replay" in spec.mode:
        config_payload["replay_reference_candidate"] = "baseline_plain_raw"
    fmpc_tf2_module._write_json(run_dir / "config.json", config_payload)
    fmpc_tf2_module._write_epoch_metrics(run_dir / "epoch_metrics.csv", epoch_rows)
    # Keep the artifact filename short enough to stay under Windows path-length limits
    # for deep diagnostic run directories.
    fmpc_tf2_module._write_json(run_dir / "selection.json", selection_diagnostics)
    fmpc_tf2_module._write_json(run_dir / "summary.json", summary)


def _train_one_batch_terminal_action_stabilization(
    model: Any,
    psi_network: Any,
    config: FMPCTF2Config,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    *,
    baseline_slots: list[_ReplayPlanSlot],
    spec: _CellSpec,
) -> tuple[float, float, float, float, dict[str, float]]:
    if config.supervision_policy != "local_only":
        raise ValueError("Terminal action stabilization diagnostic is defined only for local_only supervision.")
    if fmpc_tf2_module._resolved_theta_update_cadence(config) != "terminal_only":
        raise ValueError("Terminal action stabilization diagnostic expects terminal_only theta cadence.")
    if len(baseline_slots) != int(config.micro_steps):
        raise ValueError("baseline_slots must align with config.micro_steps.")

    context = fmpc_tf2_module.build_tf1_context(model, x_batch, y_batch)
    active_mix_ratio = fmpc_tf2_module._active_onpolicy_mix_ratio(config, None)
    z_on = context.z0.copy()
    z_lf = context.z0.copy()
    total_losses: list[float] = []
    boot_losses: list[float] = []
    identity_losses: list[float] = []
    terminal_metrics: dict[str, float] | None = None

    for step_index, baseline_slot in enumerate(baseline_slots):
        if spec.action_intervention_mode == "closed_loop_live_plan":
            live_plan = fmpc_tf2_module._plan_tf2_micro_step(
                context,
                psi_network,
                config,
                z_on,
                z_lf,
                t_k=baseline_slot.t_k,
                dt=baseline_slot.dt,
                r_k=baseline_slot.r_k,
                onpolicy_mix_ratio=active_mix_ratio,
            )
            psi_inputs = live_plan.psi_inputs
            boot_targets = live_plan.boot_targets
            identity_targets = live_plan.identity_targets
            lambda_id = float(baseline_slot.lambda_id)
            effective_prediction = fmpc_tf2_module._psi_predict(psi_network, psi_inputs, config)
            z_on_next = live_plan.z_on_next.copy()
            z_lf_next = live_plan.z_lf_next.copy()
        elif step_index < int(spec.replay_prefix_steps) or spec.action_intervention_mode in {
            "open_loop_baseline_plan_replay",
            "full_replay",
        }:
            psi_inputs = _candidate_replay_inputs(config, baseline_slot)
            boot_targets = baseline_slot.boot_targets
            identity_targets = baseline_slot.identity_targets
            lambda_id = float(baseline_slot.lambda_id)
            effective_prediction = fmpc_tf2_module._psi_predict(psi_network, psi_inputs, config)
            z_on_next = baseline_slot.z_on_next.copy()
            z_lf_next = baseline_slot.z_lf_next.copy()
        else:
            live_plan = fmpc_tf2_module._plan_tf2_micro_step(
                context,
                psi_network,
                config,
                z_on,
                z_lf,
                t_k=baseline_slot.t_k,
                dt=baseline_slot.dt,
                r_k=baseline_slot.r_k,
                onpolicy_mix_ratio=active_mix_ratio,
            )
            psi_inputs = live_plan.psi_inputs
            boot_targets = baseline_slot.boot_targets
            identity_targets = baseline_slot.identity_targets
            lambda_id = float(baseline_slot.lambda_id)

            baseline_action = _action_from_step(z_on, baseline_slot.z_on_next, baseline_slot.dt)
            live_action = _action_from_step(z_on, live_plan.z_on_next, baseline_slot.dt)
            baseline_direction = _safe_direction(baseline_action)
            live_direction = _safe_direction(live_action)
            baseline_norm = _vector_norms(baseline_action)[:, None]
            live_norm = _vector_norms(live_action)[:, None]

            if spec.action_intervention_mode == "live_action_raw":
                mode_action = live_action
                effective_prediction = live_action
                z_on_next = live_plan.z_on_next.copy()
            elif spec.action_intervention_mode == "replay_action_vector_only":
                mode_action = baseline_action
                effective_prediction = baseline_action
                z_on_next = np.asarray(z_on, dtype=np.float64) + float(baseline_slot.dt) * baseline_action
            elif spec.action_intervention_mode == "replay_action_direction_only":
                mode_action = baseline_direction * live_norm
                effective_prediction = mode_action
                z_on_next = np.asarray(z_on, dtype=np.float64) + float(baseline_slot.dt) * mode_action
            elif spec.action_intervention_mode == "replay_action_norm_only":
                mode_action = live_direction * baseline_norm
                effective_prediction = mode_action
                z_on_next = np.asarray(z_on, dtype=np.float64) + float(baseline_slot.dt) * mode_action
            elif spec.action_intervention_mode == "replay_post_action_state_only":
                mode_action = live_action
                effective_prediction = live_action
                z_on_next = baseline_slot.z_on_next.copy()
            else:
                raise ValueError(f"Unsupported action_intervention_mode '{spec.action_intervention_mode}'.")

            z_lf_next = live_plan.z_lf_next.copy()
            terminal_metrics = _action_metrics(
                baseline_action=baseline_action,
                mode_action=mode_action,
                live_action=live_action,
                baseline_slot=baseline_slot,
                boot_targets=boot_targets,
                identity_targets=identity_targets,
                lambda_id=lambda_id,
                mode_z_on_next=z_on_next,
                live_z_on_next=live_plan.z_on_next,
            )

        boot_loss = float(np.mean((effective_prediction - boot_targets) ** 2))
        identity_loss = float(np.mean((effective_prediction - identity_targets) ** 2))
        if lambda_id > 0.0:
            combined_target = (boot_targets + (lambda_id * identity_targets)) / (1.0 + lambda_id)
            loss_scale = 1.0 + lambda_id
        else:
            combined_target = boot_targets
            loss_scale = 1.0
        total_loss = boot_loss + (lambda_id * identity_loss)
        fmpc_tf2_module._weighted_mse_step(
            psi_network,
            psi_inputs,
            combined_target,
            loss_scale=loss_scale,
        )
        total_losses.append(total_loss)
        boot_losses.append(boot_loss)
        identity_losses.append(identity_loss)
        z_on = z_on_next
        z_lf = z_lf_next

    if terminal_metrics is None:
        terminal_metrics = {
            "relative_terminal_action_vector_delta_vs_replay": 0.0,
            "terminal_action_direction_cosine_similarity_vs_replay": 1.0,
            "terminal_action_norm_ratio_vs_replay": 1.0,
            "terminal_action_norm_delta_vs_replay": 0.0,
            "relative_terminal_bootstrap_target_delta": 0.0,
            "relative_terminal_identity_target_delta": 0.0,
            "terminal_lambda_id_delta": 0.0,
            "terminal_state_slot_displacement_after_action": 0.0,
            "terminal_next_state_displacement_vs_baseline_replay": 0.0,
        }

    transported_final_energy = fmpc_tf2_module.hidden_energy_from_state(context, z_on)
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
        terminal_metrics,
    )


def _finalize_cell(
    base_run_dir: Path,
    cell: _CellRuntime,
    split: Any,
    *,
    seed: int,
) -> _RunArtifacts:
    selection_diagnostics = build_tf1_epoch_selection_diagnostics(cell.epoch_rows)
    checkpoint_selection = _select_tf1_checkpoint_epoch(
        cell.epoch_rows,
        cell.config.checkpoint_selector,
        selection_diagnostics=selection_diagnostics,
    )
    selected_epoch = int(checkpoint_selection["selected_epoch"])
    selected_snapshot = next(
        snapshot for snapshot in cell.epoch_snapshots if int(snapshot.epoch) == int(selected_epoch)
    )
    fmpc_tf2_module._restore_pc_parameters(cell.model, selected_snapshot.model_snapshot)
    fmpc_tf2_module._restore_mlp_parameters(cell.psi_network, selected_snapshot.psi_snapshot)

    evaluation_start = perf_counter()
    val_transport = fmpc_tf2_module._evaluate_transport_split(
        cell.model,
        cell.psi_network,
        cell.config,
        split.x_val,
        split.y_val,
    )
    test_transport = fmpc_tf2_module._evaluate_transport_split(
        cell.model,
        cell.psi_network,
        cell.config,
        split.x_test,
        split.y_test,
    )
    val_loss, val_accuracy = fmpc_tf2_module._evaluate_slow_pc_accuracy(cell.model, split.x_val, split.y_val)
    test_loss, test_accuracy = fmpc_tf2_module._evaluate_slow_pc_accuracy(cell.model, split.x_test, split.y_test)
    evaluation_wall_time_seconds = float(perf_counter() - evaluation_start)

    val_baseline_accuracy = majority_class_baseline_accuracy(split.y_val)
    test_baseline_accuracy = majority_class_baseline_accuracy(split.y_test)
    resolved_theta_update_cadence = fmpc_tf2_module._resolved_theta_update_cadence(cell.config)
    theta_micro_lr, theta_micro_bias_lr = fmpc_tf2_module._theta_micro_learning_rates(
        cell.config,
        resolved_theta_update_cadence,
    )
    summary = {
        "phase": "FMPC Stage 04 Incremental Bridge",
        "stage": "terminal_action_output_stabilization_diagnostic",
        "diagnostic_only": True,
        "preset_name": cell.config.preset_name,
        "candidate_key": cell.candidate.key,
        "handoff_mode": cell.spec.mode,
        "replay_prefix_steps": int(cell.spec.replay_prefix_steps),
        "action_intervention_mode": cell.spec.action_intervention_mode,
        "teacher_free": True,
        "uses_teacher_artifacts": False,
        "feature_aware_tangents": bool(cell.config.feature_aware_tangents),
        "identity_tangent_mode": fmpc_tf2_module._identity_tangent_mode(cell.config),
        "micro_steps": int(cell.config.micro_steps),
        "supervision_policy": cell.config.supervision_policy,
        "theta_update_cadence": resolved_theta_update_cadence,
        "theta_update_budget": cell.config.theta_update_budget,
        "theta_micro_lr": float(theta_micro_lr),
        "theta_micro_bias_lr": float(theta_micro_bias_lr),
        "bootstrap_integrator": cell.config.bootstrap_integrator,
        "bootstrap_substeps": int(cell.config.bootstrap_substeps),
        "identity_loss_weight": float(cell.config.identity_loss_weight),
        "warmup_epochs": int(cell.config.warmup_epochs),
        "hybrid_ramp_epochs": int(cell.config.hybrid_ramp_epochs),
        "val_accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy),
        "val_loss": float(val_loss),
        "test_loss": float(test_loss),
        "gate_passing_epoch_count": int(checkpoint_selection["gate_passing_epoch_count"]),
        "selected_epoch": int(selected_epoch),
        "selected_epoch_passes_gate": bool(checkpoint_selection["selected_epoch_passes_gate"]),
        "selector_fallback_used": bool(checkpoint_selection["selector_fallback_used"]),
        "val_transported_final_energy": float(val_transport.transported_final_energy),
        "test_transported_final_energy": float(test_transport.transported_final_energy),
        "val_energy_delta_vs_identity": float(
            val_transport.transported_final_energy - val_transport.identity_final_energy
        ),
        "test_energy_delta_vs_identity": float(
            test_transport.transported_final_energy - test_transport.identity_final_energy
        ),
        "val_energy_delta_vs_local_field_only": float(
            val_transport.transported_final_energy - val_transport.local_field_only_final_energy
        ),
        "test_energy_delta_vs_local_field_only": float(
            test_transport.transported_final_energy - test_transport.local_field_only_final_energy
        ),
        "val_baseline_accuracy": float(val_baseline_accuracy),
        "test_baseline_accuracy": float(test_baseline_accuracy),
        "timing": {
            "train_wall_time_seconds": float(cell.train_wall_time_seconds),
            "final_evaluation_wall_time_seconds": evaluation_wall_time_seconds,
        },
    }
    _write_run_artifacts(
        cell.run_dir,
        cell.config,
        spec=cell.spec,
        epoch_rows=cell.epoch_rows,
        selection_diagnostics=selection_diagnostics,
        summary=summary,
    )
    return _RunArtifacts(
        mode=cell.spec.mode,
        candidate=cell.candidate,
        seed=int(seed),
        config=cell.config,
        run_dir=cell.run_dir,
        epoch_rows=cell.epoch_rows,
        summary=summary,
    )


def _success_run_row(*, artifact: _RunArtifacts, base_run_dir: Path) -> dict[str, Any]:
    timing = dict(artifact.summary.get("timing", {}))
    return {
        "handoff_mode": artifact.mode,
        "candidate_key": artifact.candidate.key,
        "psi_family": artifact.candidate.psi_family,
        "time_encoding_variant": artifact.candidate.time_encoding_variant,
        "replay_prefix_steps": int(artifact.summary["replay_prefix_steps"]),
        "action_intervention_mode": artifact.summary["action_intervention_mode"],
        "seed": int(artifact.seed),
        "val_accuracy": float(artifact.summary["val_accuracy"]),
        "test_accuracy": float(artifact.summary["test_accuracy"]),
        "gate_passing_epoch_count": int(artifact.summary["gate_passing_epoch_count"]),
        "val_transported_final_energy": float(artifact.summary["val_transported_final_energy"]),
        "selected_epoch": int(artifact.summary["selected_epoch"]),
        "selected_epoch_passes_gate": bool(artifact.summary["selected_epoch_passes_gate"]),
        "selector_fallback_used": bool(artifact.summary["selector_fallback_used"]),
        "total_wall_time_seconds": float(
            timing.get("train_wall_time_seconds", 0.0)
            + timing.get("final_evaluation_wall_time_seconds", 0.0)
        ),
        "run_status": "success",
        "run_summary_path": _relative_posix(base_run_dir, artifact.run_dir / "summary.json"),
    }


def _aggregate_terminal_drift(terminal_rows: list[dict[str, Any]], *, mode: TerminalActionMode) -> dict[str, float | None]:
    relevant = [row for row in terminal_rows if row["handoff_mode"] == mode]
    if not relevant:
        return {
            "mean_relative_terminal_action_vector_delta_vs_replay": None,
            "mean_terminal_action_direction_cosine_similarity_vs_replay": None,
            "mean_terminal_action_norm_ratio_vs_replay": None,
            "mean_terminal_action_norm_delta_vs_replay": None,
            "mean_relative_terminal_bootstrap_target_delta": None,
            "mean_relative_terminal_identity_target_delta": None,
            "mean_terminal_lambda_id_delta": None,
            "mean_terminal_state_slot_displacement_after_action": None,
            "mean_terminal_next_state_displacement_vs_baseline_replay": None,
        }
    return {
        "mean_relative_terminal_action_vector_delta_vs_replay": _mean(
            [float(row["relative_terminal_action_vector_delta_vs_replay"]) for row in relevant]
        ),
        "mean_terminal_action_direction_cosine_similarity_vs_replay": _mean(
            [float(row["terminal_action_direction_cosine_similarity_vs_replay"]) for row in relevant]
        ),
        "mean_terminal_action_norm_ratio_vs_replay": _mean(
            [float(row["terminal_action_norm_ratio_vs_replay"]) for row in relevant]
        ),
        "mean_terminal_action_norm_delta_vs_replay": _mean(
            [float(row["terminal_action_norm_delta_vs_replay"]) for row in relevant]
        ),
        "mean_relative_terminal_bootstrap_target_delta": _mean(
            [float(row["relative_terminal_bootstrap_target_delta"]) for row in relevant]
        ),
        "mean_relative_terminal_identity_target_delta": _mean(
            [float(row["relative_terminal_identity_target_delta"]) for row in relevant]
        ),
        "mean_terminal_lambda_id_delta": _mean([float(row["terminal_lambda_id_delta"]) for row in relevant]),
        "mean_terminal_state_slot_displacement_after_action": _mean(
            [float(row["terminal_state_slot_displacement_after_action"]) for row in relevant]
        ),
        "mean_terminal_next_state_displacement_vs_baseline_replay": _mean(
            [float(row["terminal_next_state_displacement_vs_baseline_replay"]) for row in relevant]
        ),
    }


def _run_one_seed(
    base_run_dir: Path,
    suite_config: FMPCTF2TerminalActionOutputStabilizationSuiteConfig,
    *,
    candidates: dict[str, FMPCTF2PartialHandoffCandidate],
    seed: int,
) -> tuple[list[_RunArtifacts], list[dict[str, Any]]]:
    cells: dict[tuple[str, TerminalActionMode], _CellRuntime] = {}
    for spec in _cell_specs():
        cells[(spec.candidate_key, spec.mode)] = _make_cell(
            base_run_dir,
            candidates[spec.candidate_key],
            spec,
            seed=seed,
            suite_config=suite_config,
        )

    baseline_control = cells[("baseline_plain_raw", "closed_loop_live_plan")]
    split = load_digits_split(
        split_seed=baseline_control.config.data_seed,
        train_fraction=baseline_control.config.train_fraction,
        val_fraction=baseline_control.config.val_fraction,
        test_fraction=baseline_control.config.test_fraction,
    )
    terminal_drift_rows: list[dict[str, Any]] = []

    for epoch_index in range(int(baseline_control.config.epochs)):
        lambda_id = fmpc_tf2_module._lambda_id_for_epoch(baseline_control.config, epoch_index)
        stage = fmpc_tf2_module._stage_for_epoch(baseline_control.config, epoch_index)
        epoch_metric_lists: dict[tuple[str, TerminalActionMode], dict[str, list[float]]] = {
            cell_key: {
                "train_loss": [],
                "train_boot_loss": [],
                "train_identity_loss": [],
                "train_transported_final_energy": [],
            }
            for cell_key in cells
        }
        epoch_terminal_metric_lists: dict[TerminalActionMode, dict[str, list[float]]] = {
            spec.mode: {
                "relative_terminal_action_vector_delta_vs_replay": [],
                "terminal_action_direction_cosine_similarity_vs_replay": [],
                "terminal_action_norm_ratio_vs_replay": [],
                "terminal_action_norm_delta_vs_replay": [],
                "relative_terminal_bootstrap_target_delta": [],
                "relative_terminal_identity_target_delta": [],
                "terminal_lambda_id_delta": [],
                "terminal_state_slot_displacement_after_action": [],
                "terminal_next_state_displacement_vs_baseline_replay": [],
            }
            for spec in _challenger_specs()
        }
        batch_seed = baseline_control.config.batch_order_seed + epoch_index
        for x_batch, y_batch in fmpc_tf2_module.iter_minibatches(
            split.x_train,
            split.y_train,
            baseline_control.config.batch_size,
            shuffle=baseline_control.config.shuffle_batches,
            seed=batch_seed,
        ):
            baseline_context = fmpc_tf2_module.build_tf1_context(baseline_control.model, x_batch, y_batch)
            baseline_slots = _build_cached_plan(
                baseline_context,
                baseline_control.psi_network,
                baseline_control.config,
                lambda_id=float(lambda_id),
            )

            for spec in _cell_specs():
                cell = cells[(spec.candidate_key, spec.mode)]
                batch_start = perf_counter()
                train_loss, boot_loss, identity_loss, transported_energy, terminal_metrics = (
                    _train_one_batch_terminal_action_stabilization(
                        cell.model,
                        cell.psi_network,
                        cell.config,
                        x_batch,
                        y_batch,
                        baseline_slots=baseline_slots,
                        spec=spec,
                    )
                )
                cell.train_wall_time_seconds += float(perf_counter() - batch_start)
                metric_lists = epoch_metric_lists[(spec.candidate_key, spec.mode)]
                metric_lists["train_loss"].append(train_loss)
                metric_lists["train_boot_loss"].append(boot_loss)
                metric_lists["train_identity_loss"].append(identity_loss)
                metric_lists["train_transported_final_energy"].append(transported_energy)
                if spec.candidate_key != "residualized_local_field_poly_rt2":
                    continue
                for metric_name, metric_value in terminal_metrics.items():
                    epoch_terminal_metric_lists[spec.mode][metric_name].append(float(metric_value))

        for spec in _cell_specs():
            cell = cells[(spec.candidate_key, spec.mode)]
            val_transport = fmpc_tf2_module._evaluate_transport_split(
                cell.model,
                cell.psi_network,
                cell.config,
                split.x_val,
                split.y_val,
            )
            _, val_accuracy = fmpc_tf2_module._evaluate_slow_pc_accuracy(cell.model, split.x_val, split.y_val)
            val_baseline_accuracy = majority_class_baseline_accuracy(split.y_val)
            val_energy_delta_vs_identity = val_transport.transported_final_energy - val_transport.identity_final_energy
            val_energy_delta_vs_local_field_only = (
                val_transport.transported_final_energy - val_transport.local_field_only_final_energy
            )
            metric_lists = epoch_metric_lists[(spec.candidate_key, spec.mode)]
            cell.epoch_rows.append(
                asdict(
                    FMPCTF2EpochMetrics(
                        epoch=epoch_index + 1,
                        lambda_id=float(lambda_id),
                        stage=stage,
                        train_loss=float(np.mean(metric_lists["train_loss"])),
                        train_boot_loss=float(np.mean(metric_lists["train_boot_loss"])),
                        train_identity_loss=float(np.mean(metric_lists["train_identity_loss"])),
                        train_transported_final_energy=float(
                            np.mean(metric_lists["train_transported_final_energy"])
                        ),
                        val_transported_final_energy=float(val_transport.transported_final_energy),
                        val_identity_final_energy=float(val_transport.identity_final_energy),
                        val_local_field_only_final_energy=float(val_transport.local_field_only_final_energy),
                        val_energy_delta_vs_identity=float(val_energy_delta_vs_identity),
                        val_energy_delta_vs_local_field_only=float(val_energy_delta_vs_local_field_only),
                        val_accuracy=float(val_accuracy),
                        val_baseline_accuracy=float(val_baseline_accuracy),
                    )
                )
            )
            cell.epoch_snapshots.append(
                fmpc_tf2_module.FMPCTF2EpochSnapshot(
                    epoch=epoch_index + 1,
                    model_snapshot=fmpc_tf2_module._snapshot_pc_parameters(cell.model),
                    psi_snapshot=fmpc_tf2_module._snapshot_mlp_parameters(cell.psi_network),
                )
            )

        for spec in _challenger_specs():
            aggregated = {
                metric_name: _mean(metric_values)
                for metric_name, metric_values in epoch_terminal_metric_lists[spec.mode].items()
            }
            terminal_drift_rows.append(
                {
                    "seed": int(seed),
                    "epoch": int(epoch_index + 1),
                    "candidate_key": spec.candidate_key,
                    "handoff_mode": spec.mode,
                    "replay_prefix_steps": int(spec.replay_prefix_steps),
                    "action_intervention_mode": spec.action_intervention_mode,
                    **aggregated,
                }
            )

    artifacts: list[_RunArtifacts] = []
    for spec in _cell_specs():
        cell = cells[(spec.candidate_key, spec.mode)]
        artifacts.append(_finalize_cell(base_run_dir, cell, split, seed=seed))
    return artifacts, terminal_drift_rows


def run_fmpc_tf2_terminal_action_output_stabilization_suite(
    config: FMPCTF2TerminalActionOutputStabilizationSuiteConfig,
) -> FMPCTF2TerminalActionOutputStabilizationSuiteRunResult:
    candidates = _candidate_registry()
    run_dir = _prepare_run_dir(
        _resolve_run_dir(
            config.output_root,
            config.experiment_name,
            config.resolved_run_id(),
            config.output_layout,
        )
    )
    _write_json(run_dir / "config.json", _suite_config_payload(config, candidates))

    run_rows: list[dict[str, Any]] = []
    terminal_drift_rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        artifacts, seed_terminal_rows = _run_one_seed(
            run_dir,
            config,
            candidates=candidates,
            seed=int(seed),
        )
        run_rows.extend([_success_run_row(artifact=artifact, base_run_dir=run_dir) for artifact in artifacts])
        terminal_drift_rows.extend(seed_terminal_rows)

    _write_csv(run_dir / "end_to_end_runs.csv", run_rows)
    _write_csv(run_dir / "terminal_action_drift_epoch_metrics.csv", terminal_drift_rows)

    baseline_live = _aggregate_run_group(
        run_rows,
        mode="closed_loop_live_plan",
        candidate_key="baseline_plain_raw",
    )
    baseline_open = _aggregate_run_group(
        run_rows,
        mode="open_loop_baseline_plan_replay",
        candidate_key="baseline_plain_raw",
    )
    challenger_modes: list[TerminalActionMode] = [
        "terminal_live_action_raw",
        "terminal_replay_action_vector_only",
        "terminal_replay_action_direction_only",
        "terminal_replay_action_norm_only",
        "terminal_replay_post_action_state_only",
        "k4_open_loop_baseline_plan_replay",
    ]
    challenger_summary = {
        mode: _aggregate_run_group(
            run_rows,
            mode=mode,
            candidate_key="residualized_local_field_poly_rt2",
        )
        for mode in challenger_modes
    }
    terminal_action_drift_summary = {
        mode: _aggregate_terminal_drift(terminal_drift_rows, mode=mode)
        for mode in challenger_modes
    }

    pairwise_vs_baseline_live = {
        "baseline_plain_raw__closed_loop_live_plan": _pairwise_vs_reference(
            run_rows,
            candidate_key="baseline_plain_raw",
            mode="closed_loop_live_plan",
            reference_candidate_key="baseline_plain_raw",
            reference_mode="closed_loop_live_plan",
        ),
        "baseline_plain_raw__open_loop_baseline_plan_replay": _pairwise_vs_reference(
            run_rows,
            candidate_key="baseline_plain_raw",
            mode="open_loop_baseline_plan_replay",
            reference_candidate_key="baseline_plain_raw",
            reference_mode="closed_loop_live_plan",
        ),
        **{
            f"residualized_local_field_poly_rt2__{mode}": _pairwise_vs_reference(
                run_rows,
                candidate_key="residualized_local_field_poly_rt2",
                mode=mode,
                reference_candidate_key="baseline_plain_raw",
                reference_mode="closed_loop_live_plan",
            )
            for mode in challenger_modes
        },
    }
    pairwise_vs_terminal_live_raw = {
        mode: _pairwise_vs_reference(
            run_rows,
            candidate_key="residualized_local_field_poly_rt2",
            mode=mode,
            reference_candidate_key="residualized_local_field_poly_rt2",
            reference_mode="terminal_live_action_raw",
        )
        for mode in challenger_modes
    }
    pairwise_vs_k4 = {
        mode: _pairwise_vs_reference(
            run_rows,
            candidate_key="residualized_local_field_poly_rt2",
            mode=mode,
            reference_candidate_key="residualized_local_field_poly_rt2",
            reference_mode="k4_open_loop_baseline_plan_replay",
        )
        for mode in challenger_modes
    }

    material = float(config.material_test_gain)
    tolerance = float(config.baseline_similarity_tolerance)

    def _rescues(mode: TerminalActionMode) -> bool:
        same_family = pairwise_vs_terminal_live_raw[mode]
        vs_k4 = pairwise_vs_k4[mode]
        return (
            float(same_family["mean_test_accuracy_delta"]) >= material
            and abs(float(vs_k4["mean_test_accuracy_delta"])) <= tolerance
        )

    vector_rescue = _rescues("terminal_replay_action_vector_only")
    direction_rescue = _rescues("terminal_replay_action_direction_only")
    norm_rescue = _rescues("terminal_replay_action_norm_only")
    post_state_rescue = _rescues("terminal_replay_post_action_state_only")

    if vector_rescue:
        dominant_interpretation = "terminal_raw_action_vector_is_dominant"
        next_single_narrow_move = (
            "run one narrow terminal trust-region diagnostic that constrains the challenger's final action "
            "toward the replayed baseline direction-and-scale neighborhood without reopening broader TF2 search"
        )
    elif direction_rescue and not norm_rescue:
        dominant_interpretation = "terminal_action_direction_error_is_dominant"
        next_single_narrow_move = (
            "run one narrow terminal direction-stabilization diagnostic that anchors only the final-step "
            "action direction while leaving action scale and target construction unchanged"
        )
    elif norm_rescue and not direction_rescue:
        dominant_interpretation = "terminal_action_magnitude_instability_is_dominant"
        next_single_narrow_move = (
            "run one narrow terminal norm-stabilization diagnostic that clips or anchors only the final-step "
            "action norm while keeping direction and supervision semantics unchanged"
        )
    elif post_state_rescue and not vector_rescue:
        dominant_interpretation = "broader_post_action_terminal_transition_mismatch_is_dominant"
        next_single_narrow_move = (
            "run one narrow terminal state-transition stabilization diagnostic that preserves the challenger's "
            "action prediction but regularizes the terminal next-state handoff"
        )
    else:
        dominant_interpretation = "broader_terminal_live_action_transition_coupling_remains"
        next_single_narrow_move = (
            "run one narrow terminal trust-region or action-stabilization diagnostic instead of further "
            "target decomposition, because neither action-direction nor action-norm replay rescues the run"
        )

    summary = {
        "phase": "FMPC Stage 04 Incremental Bridge",
        "stage": "terminal_action_output_stabilization_diagnostic",
        "diagnostic_only": True,
        "end_to_end_summary": {
            "baseline_plain_raw": {
                "closed_loop_live_plan": baseline_live,
                "open_loop_baseline_plan_replay": baseline_open,
            },
            "residualized_local_field_poly_rt2": challenger_summary,
        },
        "terminal_action_drift_summary": terminal_action_drift_summary,
        "pairwise_delta_vs_baseline_closed_loop_run": pairwise_vs_baseline_live,
        "pairwise_delta_vs_challenger_terminal_live_action_raw": pairwise_vs_terminal_live_raw,
        "pairwise_delta_vs_challenger_k4_open_loop_replay_run": pairwise_vs_k4,
        "is_terminal_action_vector_dominant": bool(vector_rescue),
        "is_terminal_action_direction_dominant": bool(direction_rescue and not norm_rescue),
        "is_terminal_action_norm_dominant": bool(norm_rescue and not direction_rescue),
        "is_terminal_post_action_transition_dominant": bool(post_state_rescue and not vector_rescue),
        "dominant_interpretation": dominant_interpretation,
        "next_single_narrow_move": next_single_narrow_move,
        "artifacts": {
            "end_to_end_runs_csv": "end_to_end_runs.csv",
            "terminal_action_drift_epoch_metrics_csv": "terminal_action_drift_epoch_metrics.csv",
        },
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF2TerminalActionOutputStabilizationSuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config, candidates),
        run_rows=run_rows,
        terminal_drift_rows=terminal_drift_rows,
        summary=summary,
    )
