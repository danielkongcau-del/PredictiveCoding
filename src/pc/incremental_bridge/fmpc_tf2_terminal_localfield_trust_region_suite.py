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
from ..transport_core_v1.fmpc_tf1 import _select_tf1_checkpoint_epoch, build_tf1_epoch_selection_diagnostics
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

TerminalLocalFieldTrustRegionMode = Literal[
    "closed_loop_live_plan",
    "open_loop_baseline_plan_replay",
    "terminal_live_raw_closed_loop",
    "terminal_local_field_direction_hard_replace_keep_live_norm",
    "terminal_local_field_direction_strong_mix_keep_live_norm",
    "terminal_local_field_direction_angle_clip_keep_live_norm",
    "terminal_replay_action_vector_only",
]
LocalFieldTrustRegionInterventionMode = Literal[
    "closed_loop_live_plan",
    "open_loop_baseline_plan_replay",
    "live_raw_closed_loop",
    "local_field_direction_hard_replace_keep_live_norm",
    "local_field_direction_strong_mix_keep_live_norm",
    "local_field_direction_angle_clip_keep_live_norm",
    "replay_action_vector_only",
]


@dataclass(frozen=True)
class _CellSpec:
    candidate_key: str
    mode: TerminalLocalFieldTrustRegionMode
    replay_prefix_steps: int
    direction_anchor_mode: LocalFieldTrustRegionInterventionMode


@dataclass
class FMPCTF2TerminalLocalFieldTrustRegionSuiteConfig:
    """Diagnostic-only terminal-step local-field trust-region suite."""

    experiment_name: str = "fmpc_tf2_terminal_localfield_trust_region_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: Literal["single_dir", "run_id_subdir"] = "single_dir"
    seeds: tuple[int, ...] = (0, 1, 2)
    epochs: int = 60
    batch_size: int = 128
    eval_steps: int = 15
    layer_dims: tuple[int, ...] = (64, 64, 10)
    material_test_gain: float = 0.005
    strong_mix_weight_toward_local_field: float = 0.75
    angle_clip_degrees: float = 30.0

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"


@dataclass
class FMPCTF2TerminalLocalFieldTrustRegionSuiteRunResult:
    run_dir: Path
    config: dict[str, Any]
    run_rows: list[dict[str, Any]]
    trust_region_drift_rows: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass
class _RunArtifacts:
    mode: TerminalLocalFieldTrustRegionMode
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
            direction_anchor_mode="closed_loop_live_plan",
        ),
        _CellSpec(
            candidate_key="baseline_plain_raw",
            mode="open_loop_baseline_plan_replay",
            replay_prefix_steps=4,
            direction_anchor_mode="open_loop_baseline_plan_replay",
        ),
    ]


def _challenger_specs() -> list[_CellSpec]:
    return [
        _CellSpec(
            candidate_key="residualized_local_field_poly_rt2",
            mode="terminal_live_raw_closed_loop",
            replay_prefix_steps=0,
            direction_anchor_mode="live_raw_closed_loop",
        ),
        _CellSpec(
            candidate_key="residualized_local_field_poly_rt2",
            mode="terminal_local_field_direction_hard_replace_keep_live_norm",
            replay_prefix_steps=0,
            direction_anchor_mode="local_field_direction_hard_replace_keep_live_norm",
        ),
        _CellSpec(
            candidate_key="residualized_local_field_poly_rt2",
            mode="terminal_local_field_direction_strong_mix_keep_live_norm",
            replay_prefix_steps=0,
            direction_anchor_mode="local_field_direction_strong_mix_keep_live_norm",
        ),
        _CellSpec(
            candidate_key="residualized_local_field_poly_rt2",
            mode="terminal_local_field_direction_angle_clip_keep_live_norm",
            replay_prefix_steps=0,
            direction_anchor_mode="local_field_direction_angle_clip_keep_live_norm",
        ),
        _CellSpec(
            candidate_key="residualized_local_field_poly_rt2",
            mode="terminal_replay_action_vector_only",
            replay_prefix_steps=0,
            direction_anchor_mode="replay_action_vector_only",
        ),
    ]


def _cell_specs() -> list[_CellSpec]:
    return [*_baseline_specs(), *_challenger_specs()]


def _suite_config_payload(
    config: FMPCTF2TerminalLocalFieldTrustRegionSuiteConfig,
    candidates: dict[str, FMPCTF2PartialHandoffCandidate],
) -> dict[str, Any]:
    return {
        "phase": "Phase Incremental Bridge",
        "stage": "terminal_localfield_trust_region_suite",
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
        "strong_mix_weight_toward_local_field": float(config.strong_mix_weight_toward_local_field),
        "angle_clip_degrees": float(config.angle_clip_degrees),
        "terminal_definition": (
            "The challenger runs its true harmful closed-loop live plan through steps 0-2 and keeps "
            "the terminal supervision bundle live at step 3; only the final-step action direction is "
            "stabilized toward the teacher-free local-field direction in diagnostic trust-region modes."
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
    suite_config: FMPCTF2TerminalLocalFieldTrustRegionSuiteConfig,
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


def _normalized_halfway_direction(lhs_direction: np.ndarray, rhs_direction: np.ndarray) -> np.ndarray:
    return _safe_direction(np.asarray(lhs_direction, dtype=np.float64) + np.asarray(rhs_direction, dtype=np.float64))


def _normalized_weighted_direction(
    raw_direction: np.ndarray,
    anchor_direction: np.ndarray,
    *,
    anchor_weight: float,
) -> np.ndarray:
    raw_weight = 1.0 - float(anchor_weight)
    return _safe_direction(
        (raw_weight * np.asarray(raw_direction, dtype=np.float64))
        + (float(anchor_weight) * np.asarray(anchor_direction, dtype=np.float64))
    )


def _clip_direction_to_anchor_cone(
    raw_direction: np.ndarray,
    anchor_direction: np.ndarray,
    *,
    max_angle_degrees: float,
) -> np.ndarray:
    raw = np.asarray(raw_direction, dtype=np.float64)
    anchor = np.asarray(anchor_direction, dtype=np.float64)
    if raw.shape != anchor.shape:
        raise ValueError("raw_direction and anchor_direction must share the same shape.")
    clipped = np.zeros_like(raw)
    cos_threshold = float(np.cos(np.deg2rad(float(max_angle_degrees))))
    sin_threshold = float(np.sin(np.deg2rad(float(max_angle_degrees))))
    for row_index in range(int(raw.shape[0])):
        raw_row = raw[row_index]
        anchor_row = anchor[row_index]
        raw_norm = float(np.linalg.norm(raw_row))
        anchor_norm = float(np.linalg.norm(anchor_row))
        if raw_norm <= 1e-12 or anchor_norm <= 1e-12:
            clipped[row_index] = raw_row
            continue
        raw_unit = raw_row / raw_norm
        anchor_unit = anchor_row / anchor_norm
        cosine = float(np.clip(np.dot(raw_unit, anchor_unit), -1.0, 1.0))
        if cosine >= cos_threshold:
            clipped[row_index] = raw_unit
            continue
        tangent = raw_unit - (cosine * anchor_unit)
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1e-12:
            clipped[row_index] = anchor_unit
            continue
        tangent_unit = tangent / tangent_norm
        clipped[row_index] = (cos_threshold * anchor_unit) + (sin_threshold * tangent_unit)
    return _safe_direction(clipped)


def _trust_region_metrics(
    *,
    baseline_action: np.ndarray,
    live_action: np.ndarray,
    mode_action: np.ndarray,
    local_field_action: np.ndarray,
    baseline_z_on_next: np.ndarray,
    mode_z_on_next: np.ndarray,
    live_z_on_next: np.ndarray,
) -> dict[str, float]:
    live_norm = _vector_norms(live_action)
    mode_norm = _vector_norms(mode_action)
    norm_ratio = np.divide(mode_norm, np.maximum(live_norm, 1e-12))
    return {
        "relative_terminal_action_vector_delta_vs_baseline_replay": _relative_target_delta(
            baseline_action,
            mode_action,
        ),
        "terminal_action_direction_cosine_similarity_vs_baseline_replay": _mean_cosine_similarity(
            _safe_direction(baseline_action),
            _safe_direction(mode_action),
        ),
        "terminal_action_direction_cosine_similarity_vs_terminal_local_field": _mean_cosine_similarity(
            _safe_direction(local_field_action),
            _safe_direction(mode_action),
        ),
        "terminal_action_norm_ratio_vs_live_raw": float(np.mean(norm_ratio)),
        "terminal_action_norm_delta_vs_live_raw": float(np.mean(np.abs(mode_norm - live_norm))),
        "terminal_state_slot_displacement": _mean_state_displacement(
            live_z_on_next,
            mode_z_on_next,
        ),
        "terminal_next_state_displacement_vs_baseline_replay": _mean_state_displacement(
            baseline_z_on_next,
            mode_z_on_next,
        ),
        "terminal_next_state_displacement_vs_live_raw": _mean_state_displacement(
            live_z_on_next,
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
    config_payload["direction_anchor_mode"] = spec.direction_anchor_mode
    config_payload["diagnostic_only"] = True
    if "baseline_plan_replay" in spec.mode:
        config_payload["replay_reference_candidate"] = "baseline_plain_raw"
    fmpc_tf2_module._write_json(run_dir / "config.json", config_payload)
    fmpc_tf2_module._write_epoch_metrics(run_dir / "epoch_metrics.csv", epoch_rows)
    fmpc_tf2_module._write_json(run_dir / "selection_diagnostics.json", selection_diagnostics)
    fmpc_tf2_module._write_json(run_dir / "summary.json", summary)


def _train_one_batch_terminal_localfield_trust_region(
    model: Any,
    psi_network: Any,
    config: FMPCTF2Config,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    *,
    baseline_slots: list[_ReplayPlanSlot],
    spec: _CellSpec,
    strong_mix_weight_toward_local_field: float,
    angle_clip_degrees: float,
) -> tuple[float, float, float, float, dict[str, float]]:
    if config.supervision_policy != "local_only":
        raise ValueError("Terminal local-field trust-region diagnostic is defined only for local_only supervision.")
    if fmpc_tf2_module._resolved_theta_update_cadence(config) != "terminal_only":
        raise ValueError("Terminal local-field trust-region diagnostic expects terminal_only theta cadence.")
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
        if spec.direction_anchor_mode == "closed_loop_live_plan":
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
        elif spec.direction_anchor_mode == "open_loop_baseline_plan_replay":
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
            boot_targets = live_plan.boot_targets
            identity_targets = live_plan.identity_targets
            lambda_id = float(baseline_slot.lambda_id)

            if step_index < int(config.micro_steps) - 1:
                effective_prediction = fmpc_tf2_module._psi_predict(psi_network, psi_inputs, config)
                z_on_next = live_plan.z_on_next.copy()
                z_lf_next = live_plan.z_lf_next.copy()
            else:
                baseline_action = _action_from_step(
                    baseline_slot.z_on_k,
                    baseline_slot.z_on_next,
                    baseline_slot.dt,
                )
                live_action = _action_from_step(z_on, live_plan.z_on_next, baseline_slot.dt)
                live_direction = _safe_direction(live_action)
                live_norm = _vector_norms(live_action)[:, None]
                local_field_action = fmpc_tf2_module._extract_detached_local_flow_anchor(
                    live_plan.psi_inputs,
                    config,
                )
                local_field_direction = _safe_direction(local_field_action)

                if spec.direction_anchor_mode == "live_raw_closed_loop":
                    mode_action = live_action
                elif spec.direction_anchor_mode == "local_field_direction_hard_replace_keep_live_norm":
                    mode_action = local_field_direction * live_norm
                elif spec.direction_anchor_mode == "local_field_direction_strong_mix_keep_live_norm":
                    mode_action = _normalized_weighted_direction(
                        live_direction,
                        local_field_direction,
                        anchor_weight=float(strong_mix_weight_toward_local_field),
                    ) * live_norm
                elif spec.direction_anchor_mode == "local_field_direction_angle_clip_keep_live_norm":
                    mode_action = _clip_direction_to_anchor_cone(
                        live_direction,
                        local_field_direction,
                        max_angle_degrees=float(angle_clip_degrees),
                    ) * live_norm
                elif spec.direction_anchor_mode == "replay_action_vector_only":
                    mode_action = baseline_action
                else:
                    raise ValueError(f"Unsupported direction_anchor_mode '{spec.direction_anchor_mode}'.")

                effective_prediction = mode_action
                z_on_next = np.asarray(z_on, dtype=np.float64) + float(baseline_slot.dt) * mode_action
                z_lf_next = live_plan.z_lf_next.copy()
                terminal_metrics = _trust_region_metrics(
                    baseline_action=baseline_action,
                    live_action=live_action,
                    mode_action=mode_action,
                    local_field_action=local_field_action,
                    baseline_z_on_next=baseline_slot.z_on_next,
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
            "relative_terminal_action_vector_delta_vs_baseline_replay": 0.0,
            "terminal_action_direction_cosine_similarity_vs_baseline_replay": 1.0,
            "terminal_action_direction_cosine_similarity_vs_terminal_local_field": 1.0,
            "terminal_action_norm_ratio_vs_live_raw": 1.0,
            "terminal_action_norm_delta_vs_live_raw": 0.0,
            "terminal_state_slot_displacement": 0.0,
            "terminal_next_state_displacement_vs_baseline_replay": 0.0,
            "terminal_next_state_displacement_vs_live_raw": 0.0,
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
        "phase": "Phase Incremental Bridge",
        "stage": "terminal_localfield_trust_region_diagnostic",
        "diagnostic_only": True,
        "preset_name": cell.config.preset_name,
        "candidate_key": cell.candidate.key,
        "handoff_mode": cell.spec.mode,
        "replay_prefix_steps": int(cell.spec.replay_prefix_steps),
        "direction_anchor_mode": cell.spec.direction_anchor_mode,
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
        "direction_anchor_mode": artifact.summary["direction_anchor_mode"],
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


def _aggregate_trust_region_drift(
    drift_rows: list[dict[str, Any]],
    *,
    mode: TerminalLocalFieldTrustRegionMode,
) -> dict[str, float | None]:
    relevant = [row for row in drift_rows if row["handoff_mode"] == mode]
    if not relevant:
        return {
            "mean_relative_terminal_action_vector_delta_vs_baseline_replay": None,
            "mean_terminal_action_direction_cosine_similarity_vs_baseline_replay": None,
            "mean_terminal_action_direction_cosine_similarity_vs_terminal_local_field": None,
            "mean_terminal_action_norm_ratio_vs_live_raw": None,
            "mean_terminal_action_norm_delta_vs_live_raw": None,
            "mean_terminal_state_slot_displacement": None,
            "mean_terminal_next_state_displacement_vs_baseline_replay": None,
            "mean_terminal_next_state_displacement_vs_live_raw": None,
        }
    return {
        "mean_relative_terminal_action_vector_delta_vs_baseline_replay": _mean(
            [float(row["relative_terminal_action_vector_delta_vs_baseline_replay"]) for row in relevant]
        ),
        "mean_terminal_action_direction_cosine_similarity_vs_baseline_replay": _mean(
            [float(row["terminal_action_direction_cosine_similarity_vs_baseline_replay"]) for row in relevant]
        ),
        "mean_terminal_action_direction_cosine_similarity_vs_terminal_local_field": _mean(
            [
                float(row["terminal_action_direction_cosine_similarity_vs_terminal_local_field"])
                for row in relevant
            ]
        ),
        "mean_terminal_action_norm_ratio_vs_live_raw": _mean(
            [float(row["terminal_action_norm_ratio_vs_live_raw"]) for row in relevant]
        ),
        "mean_terminal_action_norm_delta_vs_live_raw": _mean(
            [float(row["terminal_action_norm_delta_vs_live_raw"]) for row in relevant]
        ),
        "mean_terminal_state_slot_displacement": _mean(
            [float(row["terminal_state_slot_displacement"]) for row in relevant]
        ),
        "mean_terminal_next_state_displacement_vs_baseline_replay": _mean(
            [float(row["terminal_next_state_displacement_vs_baseline_replay"]) for row in relevant]
        ),
        "mean_terminal_next_state_displacement_vs_live_raw": _mean(
            [float(row["terminal_next_state_displacement_vs_live_raw"]) for row in relevant]
        ),
    }


def _run_one_seed(
    base_run_dir: Path,
    suite_config: FMPCTF2TerminalLocalFieldTrustRegionSuiteConfig,
    *,
    candidates: dict[str, FMPCTF2PartialHandoffCandidate],
    seed: int,
) -> tuple[list[_RunArtifacts], list[dict[str, Any]]]:
    cells: dict[tuple[str, TerminalLocalFieldTrustRegionMode], _CellRuntime] = {}
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
    trust_region_drift_rows: list[dict[str, Any]] = []

    for epoch_index in range(int(baseline_control.config.epochs)):
        lambda_id = fmpc_tf2_module._lambda_id_for_epoch(baseline_control.config, epoch_index)
        stage = fmpc_tf2_module._stage_for_epoch(baseline_control.config, epoch_index)
        epoch_metric_lists: dict[tuple[str, TerminalLocalFieldTrustRegionMode], dict[str, list[float]]] = {
            cell_key: {
                "train_loss": [],
                "train_boot_loss": [],
                "train_identity_loss": [],
                "train_transported_final_energy": [],
            }
            for cell_key in cells
        }
        epoch_direction_metric_lists: dict[TerminalLocalFieldTrustRegionMode, dict[str, list[float]]] = {
            spec.mode: {
                "relative_terminal_action_vector_delta_vs_baseline_replay": [],
                "terminal_action_direction_cosine_similarity_vs_baseline_replay": [],
                "terminal_action_direction_cosine_similarity_vs_terminal_local_field": [],
                "terminal_action_norm_ratio_vs_live_raw": [],
                "terminal_action_norm_delta_vs_live_raw": [],
                "terminal_state_slot_displacement": [],
                "terminal_next_state_displacement_vs_baseline_replay": [],
                "terminal_next_state_displacement_vs_live_raw": [],
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
                train_loss, boot_loss, identity_loss, transported_energy, direction_metrics = (
                    _train_one_batch_terminal_localfield_trust_region(
                        cell.model,
                        cell.psi_network,
                        cell.config,
                        x_batch,
                        y_batch,
                        baseline_slots=baseline_slots,
                        spec=spec,
                        strong_mix_weight_toward_local_field=suite_config.strong_mix_weight_toward_local_field,
                        angle_clip_degrees=suite_config.angle_clip_degrees,
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
                for metric_name, metric_value in direction_metrics.items():
                    epoch_direction_metric_lists[spec.mode][metric_name].append(float(metric_value))

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
                for metric_name, metric_values in epoch_direction_metric_lists[spec.mode].items()
            }
            trust_region_drift_rows.append(
                {
                    "seed": int(seed),
                    "epoch": int(epoch_index + 1),
                    "candidate_key": spec.candidate_key,
                    "handoff_mode": spec.mode,
                    "replay_prefix_steps": int(spec.replay_prefix_steps),
                    "direction_anchor_mode": spec.direction_anchor_mode,
                    **aggregated,
                }
            )

    artifacts: list[_RunArtifacts] = []
    for spec in _cell_specs():
        cell = cells[(spec.candidate_key, spec.mode)]
        artifacts.append(_finalize_cell(base_run_dir, cell, split, seed=seed))
    return artifacts, trust_region_drift_rows


def run_fmpc_tf2_terminal_localfield_trust_region_suite(
    config: FMPCTF2TerminalLocalFieldTrustRegionSuiteConfig,
) -> FMPCTF2TerminalLocalFieldTrustRegionSuiteRunResult:
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
    trust_region_drift_rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        artifacts, seed_drift_rows = _run_one_seed(
            run_dir,
            config,
            candidates=candidates,
            seed=int(seed),
        )
        run_rows.extend([_success_run_row(artifact=artifact, base_run_dir=run_dir) for artifact in artifacts])
        trust_region_drift_rows.extend(seed_drift_rows)

    _write_csv(run_dir / "end_to_end_runs.csv", run_rows)
    _write_csv(run_dir / "terminal_localfield_trust_region_drift_epoch_metrics.csv", trust_region_drift_rows)

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
    challenger_modes: list[TerminalLocalFieldTrustRegionMode] = [
        "terminal_live_raw_closed_loop",
        "terminal_local_field_direction_hard_replace_keep_live_norm",
        "terminal_local_field_direction_strong_mix_keep_live_norm",
        "terminal_local_field_direction_angle_clip_keep_live_norm",
        "terminal_replay_action_vector_only",
    ]
    challenger_summary = {
        mode: _aggregate_run_group(
            run_rows,
            mode=mode,
            candidate_key="residualized_local_field_poly_rt2",
        )
        for mode in challenger_modes
    }
    terminal_localfield_trust_region_drift_summary = {
        mode: _aggregate_trust_region_drift(trust_region_drift_rows, mode=mode)
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
            reference_mode="terminal_live_raw_closed_loop",
        )
        for mode in challenger_modes
    }

    material = float(config.material_test_gain)
    def _rescues_with_gate(mode: TerminalLocalFieldTrustRegionMode) -> bool:
        same_family = pairwise_vs_terminal_live_raw[mode]
        mode_summary = challenger_summary[mode]
        return (
            float(same_family["mean_test_accuracy_delta"]) >= material
            and float(mode_summary["mean_gate_passing_epoch_count"]) > 0.0
        )

    hard_replace_rescue = _rescues_with_gate("terminal_local_field_direction_hard_replace_keep_live_norm")
    strong_mix_rescue = _rescues_with_gate("terminal_local_field_direction_strong_mix_keep_live_norm")
    angle_clip_rescue = _rescues_with_gate("terminal_local_field_direction_angle_clip_keep_live_norm")
    any_localfield_rescue = bool(hard_replace_rescue or strong_mix_rescue or angle_clip_rescue)

    if hard_replace_rescue:
        dominant_interpretation = "strong_terminal_local_field_direction_anchor_is_viable_in_true_closed_loop"
        next_single_narrow_move = (
            "promote the result into one narrow mainline-safe terminal local-field trust-region adoption "
            "study before considering any broader late-step mechanism search"
        )
    elif strong_mix_rescue:
        dominant_interpretation = "soft_terminal_local_field_direction_trust_region_is_sufficient_in_true_closed_loop"
        next_single_narrow_move = (
            "run one narrow adoption pass around the strong local-field direction mix without reopening "
            "broader TF2 target or psi-family search"
        )
    elif angle_clip_rescue:
        dominant_interpretation = "terminal_direction_failure_is_cone_violation_like_in_true_closed_loop"
        next_single_narrow_move = (
            "run one narrow terminal cone-trust-region follow-up around the 30-degree local-field angle clip"
        )
    else:
        dominant_interpretation = "local_field_direction_is_not_yet_sufficient_as_a_live_terminal_stabilizer"
        next_single_narrow_move = (
            "target a broader late-step coupling mechanism rather than more target decomposition, because "
            "none of the terminal local-field trust-region variants created gate-passing rescue in true closed-loop"
        )

    summary = {
        "phase": "Phase Incremental Bridge",
        "stage": "terminal_localfield_trust_region_diagnostic",
        "diagnostic_only": True,
        "end_to_end_summary": {
            "baseline_plain_raw": {
                "closed_loop_live_plan": baseline_live,
                "open_loop_baseline_plan_replay": baseline_open,
            },
            "residualized_local_field_poly_rt2": challenger_summary,
        },
        "terminal_localfield_trust_region_drift_summary": terminal_localfield_trust_region_drift_summary,
        "pairwise_delta_vs_baseline_closed_loop_run": pairwise_vs_baseline_live,
        "pairwise_delta_vs_challenger_terminal_live_raw_closed_loop": pairwise_vs_terminal_live_raw,
        "is_hard_replace_local_field_direction_rescue": bool(hard_replace_rescue),
        "is_strong_mix_local_field_direction_rescue": bool(strong_mix_rescue),
        "is_angle_clip_local_field_direction_rescue": bool(angle_clip_rescue),
        "is_any_local_field_trust_region_rescue": bool(any_localfield_rescue),
        "dominant_interpretation": dominant_interpretation,
        "next_single_narrow_move": next_single_narrow_move,
        "artifacts": {
            "end_to_end_runs_csv": "end_to_end_runs.csv",
            "terminal_localfield_trust_region_drift_epoch_metrics_csv": "terminal_localfield_trust_region_drift_epoch_metrics.csv",
        },
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF2TerminalLocalFieldTrustRegionSuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config, candidates),
        run_rows=run_rows,
        trust_region_drift_rows=trust_region_drift_rows,
        summary=summary,
    )
