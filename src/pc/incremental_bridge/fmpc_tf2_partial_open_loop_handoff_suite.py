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
from .fmpc_tf2 import FMPCTF2Config, FMPCTF2EpochMetrics, build_tf2_corrective_transport_default_config
from ..metrics import majority_class_baseline_accuracy

HandoffMode = Literal[
    "closed_loop_live_plan",
    "open_loop_baseline_plan_replay",
    "handoff_k0_closed_loop_live_plan",
    "handoff_k1_baseline_prefix_then_live",
    "handoff_k2_baseline_prefix_then_live",
    "handoff_k3_baseline_prefix_then_live",
    "handoff_k4_open_loop_baseline_plan_replay",
]


@dataclass(frozen=True)
class FMPCTF2PartialHandoffCandidate:
    key: str
    psi_family: Literal["baseline_plain", "residualized_local_field"]
    time_encoding_variant: Literal["raw", "poly_rt2"]
    notes: str


@dataclass(frozen=True)
class _CachedFeatureBlock:
    g_t: np.ndarray
    e_out_t: np.ndarray
    F_t: np.ndarray


@dataclass(frozen=True)
class _ReplayPlanSlot:
    step_index: int
    t_k: float
    r_k: float
    dt: float
    z_on_k: np.ndarray
    z_lf_k: np.ndarray
    target_onehot: np.ndarray
    features: _CachedFeatureBlock
    psi_inputs: np.ndarray
    boot_targets: np.ndarray
    identity_targets: np.ndarray
    z_on_next: np.ndarray
    z_lf_next: np.ndarray
    lambda_id: float
    identity_tangent_mode: str
    source_counts: dict[str, int]


@dataclass(frozen=True)
class _CellSpec:
    candidate_key: str
    mode: HandoffMode
    replay_prefix_steps: int


@dataclass
class FMPCTF2PartialOpenLoopHandoffSuiteConfig:
    """Diagnostic-only partial open-loop handoff suite for the TF2 corrective default."""

    experiment_name: str = "fmpc_tf2_partial_open_loop_handoff_suite"
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
class FMPCTF2PartialOpenLoopHandoffSuiteRunResult:
    run_dir: Path
    config: dict[str, Any]
    run_rows: list[dict[str, Any]]
    step_drift_rows: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass
class _RunArtifacts:
    mode: HandoffMode
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
    summary: dict[str, Any] | None = None


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


def _std(values: list[float]) -> float:
    if not values:
        raise ValueError("Std requires at least one value.")
    mean_value = _mean(values)
    variance = sum((float(value) - mean_value) ** 2 for value in values) / float(len(values))
    return float(variance**0.5)


def _relative_posix(base_dir: Path, target: Path) -> str:
    return target.relative_to(base_dir).as_posix()


def _candidate_registry() -> dict[str, FMPCTF2PartialHandoffCandidate]:
    return {
        "baseline_plain_raw": FMPCTF2PartialHandoffCandidate(
            key="baseline_plain_raw",
            psi_family="baseline_plain",
            time_encoding_variant="raw",
            notes="current corrective default baseline",
        ),
        "residualized_local_field_poly_rt2": FMPCTF2PartialHandoffCandidate(
            key="residualized_local_field_poly_rt2",
            psi_family="residualized_local_field",
            time_encoding_variant="poly_rt2",
            notes="offline-better challenger from the psi-expressivity suite",
        ),
    }


def _challenger_handoff_modes() -> list[_CellSpec]:
    return [
        _CellSpec("residualized_local_field_poly_rt2", "handoff_k0_closed_loop_live_plan", 0),
        _CellSpec("residualized_local_field_poly_rt2", "handoff_k1_baseline_prefix_then_live", 1),
        _CellSpec("residualized_local_field_poly_rt2", "handoff_k2_baseline_prefix_then_live", 2),
        _CellSpec("residualized_local_field_poly_rt2", "handoff_k3_baseline_prefix_then_live", 3),
        _CellSpec("residualized_local_field_poly_rt2", "handoff_k4_open_loop_baseline_plan_replay", 4),
    ]


def _cell_specs() -> list[_CellSpec]:
    return [
        _CellSpec("baseline_plain_raw", "closed_loop_live_plan", 0),
        _CellSpec("baseline_plain_raw", "open_loop_baseline_plan_replay", 4),
        *_challenger_handoff_modes(),
    ]


def _suite_config_payload(
    config: FMPCTF2PartialOpenLoopHandoffSuiteConfig,
    candidates: dict[str, FMPCTF2PartialHandoffCandidate],
) -> dict[str, Any]:
    return {
        "phase": "Phase Incremental Bridge",
        "stage": "partial_open_loop_handoff_suite",
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
        "challenger_handoff_modes": [spec.mode for spec in _challenger_handoff_modes()],
        "baseline_controls": ["closed_loop_live_plan", "open_loop_baseline_plan_replay"],
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
        "handoff_definition": (
            "challenger handoff modes replay the cached baseline batch-start plan for the first k micro-steps, "
            "then continue from the handed-off state with the challenger's own live closed-loop rollout."
        ),
    }


def _build_candidate_config(
    candidate: FMPCTF2PartialHandoffCandidate,
    *,
    seed: int,
    suite_config: FMPCTF2PartialOpenLoopHandoffSuiteConfig,
) -> FMPCTF2Config:
    return build_tf2_corrective_transport_default_config(
        run_seed=seed,
        data_seed=seed,
        model_init_seed=seed,
        psi_init_seed=seed,
        batch_order_seed=seed,
        epochs=suite_config.epochs,
        batch_size=suite_config.batch_size,
        eval_steps=suite_config.eval_steps,
        layer_dims=suite_config.layer_dims,
        psi_family=candidate.psi_family,
        time_encoding_variant=candidate.time_encoding_variant,
    )


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


def _mean_state_slot_displacement(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_array = np.asarray(reference, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    return float(np.mean(np.linalg.norm(reference_array - candidate_array, axis=1)))


def _feature_block_from_state(context: Any, z_t: np.ndarray) -> _CachedFeatureBlock:
    features = fmpc_tf2_module.teacher_free_state_features(context, z_t)
    return _CachedFeatureBlock(
        g_t=np.asarray(features.g_t, dtype=np.float64).copy(),
        e_out_t=np.asarray(features.e_out_t, dtype=np.float64).copy(),
        F_t=np.asarray(features.F_t, dtype=np.float64).copy(),
    )


def _build_cached_plan(
    context: Any,
    psi_network: Any,
    config: FMPCTF2Config,
    *,
    lambda_id: float,
) -> list[_ReplayPlanSlot]:
    if config.supervision_policy != "local_only":
        raise ValueError("Partial handoff diagnostic is defined only for local_only supervision.")
    knots = np.linspace(0.0, 1.0, int(config.micro_steps) + 1, dtype=np.float64)
    active_mix_ratio = fmpc_tf2_module._active_onpolicy_mix_ratio(config, None)
    z_on = context.z0.copy()
    z_lf = context.z0.copy()
    slots: list[_ReplayPlanSlot] = []
    for step_index in range(int(config.micro_steps)):
        t_k = float(knots[step_index])
        r_k = 1.0 - t_k
        dt = float(knots[step_index + 1] - knots[step_index])
        feature_block = _feature_block_from_state(context, z_lf)
        plan = fmpc_tf2_module._plan_tf2_micro_step(
            context,
            psi_network,
            config,
            z_on,
            z_lf,
            t_k=t_k,
            dt=dt,
            r_k=r_k,
            onpolicy_mix_ratio=active_mix_ratio,
        )
        slots.append(
            _ReplayPlanSlot(
                step_index=int(step_index),
                t_k=t_k,
                r_k=r_k,
                dt=dt,
                z_on_k=z_on.copy(),
                z_lf_k=z_lf.copy(),
                target_onehot=np.asarray(context.targets, dtype=np.float64).copy(),
                features=feature_block,
                psi_inputs=plan.psi_inputs.copy(),
                boot_targets=plan.boot_targets.copy(),
                identity_targets=plan.identity_targets.copy(),
                z_on_next=plan.z_on_next.copy(),
                z_lf_next=plan.z_lf_next.copy(),
                lambda_id=float(lambda_id),
                identity_tangent_mode=fmpc_tf2_module._identity_tangent_mode(config),
                source_counts=dict(plan.source_counts),
            )
        )
        z_on = plan.z_on_next.copy()
        z_lf = plan.z_lf_next.copy()
    return slots


def _candidate_replay_inputs(config: FMPCTF2Config, slot: _ReplayPlanSlot) -> np.ndarray:
    return fmpc_tf2_module._build_psi_input(
        config,
        slot.z_lf_k,
        slot.target_onehot,
        t=slot.t_k,
        r=slot.r_k,
        features=slot.features,
    )


def _step_drift_from_plans(
    candidate_config: FMPCTF2Config,
    baseline_slot: _ReplayPlanSlot,
    live_plan: Any | None,
) -> dict[str, float]:
    if live_plan is None:
        return {
            "relative_bootstrap_target_delta": 0.0,
            "relative_identity_target_delta": 0.0,
            "relative_psi_input_delta": 0.0,
            "bootstrap_target_cosine_similarity": 1.0,
            "identity_target_cosine_similarity": 1.0,
            "state_slot_displacement": 0.0,
        }
    replay_inputs = _candidate_replay_inputs(candidate_config, baseline_slot)
    return {
        "relative_bootstrap_target_delta": _relative_target_delta(
            baseline_slot.boot_targets,
            live_plan.boot_targets,
        ),
        "relative_identity_target_delta": _relative_target_delta(
            baseline_slot.identity_targets,
            live_plan.identity_targets,
        ),
        "relative_psi_input_delta": _relative_target_delta(
            replay_inputs,
            live_plan.psi_inputs,
        ),
        "bootstrap_target_cosine_similarity": _mean_cosine_similarity(
            baseline_slot.boot_targets,
            live_plan.boot_targets,
        ),
        "identity_target_cosine_similarity": _mean_cosine_similarity(
            baseline_slot.identity_targets,
            live_plan.identity_targets,
        ),
        "state_slot_displacement": _mean_state_slot_displacement(
            baseline_slot.z_on_next,
            live_plan.z_on_next,
        ),
    }


def _train_one_batch_partial_handoff(
    model: Any,
    psi_network: Any,
    config: FMPCTF2Config,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    *,
    baseline_slots: list[_ReplayPlanSlot],
    replay_prefix_steps: int,
) -> tuple[float, float, float, float, list[dict[str, float]]]:
    if config.supervision_policy != "local_only":
        raise ValueError("Partial handoff diagnostic is defined only for local_only supervision.")
    if fmpc_tf2_module._resolved_theta_update_cadence(config) != "terminal_only":
        raise ValueError("Partial handoff diagnostic expects terminal_only theta cadence.")
    if len(baseline_slots) != int(config.micro_steps):
        raise ValueError("baseline_slots must align with config.micro_steps.")

    context = fmpc_tf2_module.build_tf1_context(model, x_batch, y_batch)
    active_mix_ratio = fmpc_tf2_module._active_onpolicy_mix_ratio(config, None)
    z_on = context.z0.copy()
    z_lf = context.z0.copy()
    total_losses: list[float] = []
    boot_losses: list[float] = []
    identity_losses: list[float] = []
    step_drift_metrics: list[dict[str, float]] = []

    for step_index, baseline_slot in enumerate(baseline_slots):
        if step_index < replay_prefix_steps:
            psi_inputs = _candidate_replay_inputs(config, baseline_slot)
            boot_targets = baseline_slot.boot_targets
            identity_targets = baseline_slot.identity_targets
            z_on_next = baseline_slot.z_on_next.copy()
            z_lf_next = baseline_slot.z_lf_next.copy()
            live_plan = None
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
            z_on_next = live_plan.z_on_next.copy()
            z_lf_next = live_plan.z_lf_next.copy()

        step_drift_metrics.append(_step_drift_from_plans(config, baseline_slot, live_plan))

        psi_predictions = fmpc_tf2_module._psi_predict(psi_network, psi_inputs, config)
        boot_loss = float(np.mean((psi_predictions - boot_targets) ** 2))
        identity_loss = float(np.mean((psi_predictions - identity_targets) ** 2))
        if baseline_slot.lambda_id > 0.0:
            combined_target = (
                boot_targets + (float(baseline_slot.lambda_id) * identity_targets)
            ) / (1.0 + float(baseline_slot.lambda_id))
            loss_scale = 1.0 + float(baseline_slot.lambda_id)
        else:
            combined_target = boot_targets
            loss_scale = 1.0
        total_loss = boot_loss + (float(baseline_slot.lambda_id) * identity_loss)
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
        step_drift_metrics,
    )


def _write_run_artifacts(
    run_dir: Path,
    config: FMPCTF2Config,
    *,
    mode: HandoffMode,
    epoch_rows: list[dict[str, Any]],
    selection_diagnostics: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    config_payload = fmpc_tf2_module._config_payload(config)
    config_payload["handoff_mode"] = mode
    config_payload["diagnostic_only"] = True
    if "baseline_plan_replay" in mode:
        config_payload["replay_reference_candidate"] = "baseline_plain_raw"
    fmpc_tf2_module._write_json(run_dir / "config.json", config_payload)
    fmpc_tf2_module._write_epoch_metrics(run_dir / "epoch_metrics.csv", epoch_rows)
    fmpc_tf2_module._write_json(run_dir / "selection_diagnostics.json", selection_diagnostics)
    fmpc_tf2_module._write_json(run_dir / "summary.json", summary)


def _make_cell(
    base_run_dir: Path,
    candidate: FMPCTF2PartialHandoffCandidate,
    spec: _CellSpec,
    *,
    seed: int,
    suite_config: FMPCTF2PartialOpenLoopHandoffSuiteConfig,
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


def _success_run_row(*, artifact: _RunArtifacts, base_run_dir: Path) -> dict[str, Any]:
    timing = dict(artifact.summary.get("timing", {}))
    return {
        "handoff_mode": artifact.mode,
        "candidate_key": artifact.candidate.key,
        "psi_family": artifact.candidate.psi_family,
        "time_encoding_variant": artifact.candidate.time_encoding_variant,
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


def _aggregate_run_group(
    run_rows: list[dict[str, Any]],
    *,
    mode: HandoffMode,
    candidate_key: str,
) -> dict[str, Any]:
    relevant = [
        row
        for row in run_rows
        if row["handoff_mode"] == mode
        and row["candidate_key"] == candidate_key
        and row["run_status"] == "success"
    ]
    if not relevant:
        return {
            "mean_val_accuracy": None,
            "std_val_accuracy": None,
            "mean_test_accuracy": None,
            "std_test_accuracy": None,
            "mean_gate_passing_epoch_count": None,
            "mean_val_transported_final_energy": None,
            "mean_total_wall_time_seconds": None,
            "std_total_wall_time_seconds": None,
        }
    val_values = [float(row["val_accuracy"]) for row in relevant]
    test_values = [float(row["test_accuracy"]) for row in relevant]
    gate_values = [float(row["gate_passing_epoch_count"]) for row in relevant]
    energy_values = [float(row["val_transported_final_energy"]) for row in relevant]
    wall_values = [float(row["total_wall_time_seconds"]) for row in relevant]
    return {
        "mean_val_accuracy": _mean(val_values),
        "std_val_accuracy": _std(val_values),
        "mean_test_accuracy": _mean(test_values),
        "std_test_accuracy": _std(test_values),
        "mean_gate_passing_epoch_count": _mean(gate_values),
        "mean_val_transported_final_energy": _mean(energy_values),
        "mean_total_wall_time_seconds": _mean(wall_values),
        "std_total_wall_time_seconds": _std(wall_values),
    }


def _pairwise_vs_reference(
    run_rows: list[dict[str, Any]],
    *,
    candidate_key: str,
    mode: HandoffMode,
    reference_candidate_key: str,
    reference_mode: HandoffMode,
) -> dict[str, Any]:
    candidate_by_seed = {
        int(row["seed"]): row
        for row in run_rows
        if row["candidate_key"] == candidate_key
        and row["handoff_mode"] == mode
        and row["run_status"] == "success"
    }
    reference_by_seed = {
        int(row["seed"]): row
        for row in run_rows
        if row["candidate_key"] == reference_candidate_key
        and row["handoff_mode"] == reference_mode
        and row["run_status"] == "success"
    }
    shared_seeds = sorted(set(candidate_by_seed).intersection(reference_by_seed))
    if not shared_seeds:
        return {"mean_val_accuracy_delta": None, "mean_test_accuracy_delta": None, "mean_runtime_delta": None}
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
        "mean_runtime_delta": _mean(
            [
                float(candidate_by_seed[seed]["total_wall_time_seconds"])
                - float(reference_by_seed[seed]["total_wall_time_seconds"])
                for seed in shared_seeds
            ]
        ),
    }


def _aggregate_step_drift(
    step_rows: list[dict[str, Any]],
    *,
    candidate_key: str,
    mode: HandoffMode,
    step_index: int | None = None,
) -> dict[str, float]:
    relevant = [
        row
        for row in step_rows
        if row["candidate_key"] == candidate_key
        and row["handoff_mode"] == mode
        and (step_index is None or int(row["step_index"]) == int(step_index))
    ]
    if not relevant:
        return {
            "mean_relative_bootstrap_target_delta": None,
            "mean_relative_identity_target_delta": None,
            "mean_relative_psi_input_delta": None,
            "mean_bootstrap_target_cosine_similarity": None,
            "mean_identity_target_cosine_similarity": None,
            "mean_state_slot_displacement": None,
        }
    return {
        "mean_relative_bootstrap_target_delta": _mean(
            [float(row["relative_bootstrap_target_delta"]) for row in relevant]
        ),
        "mean_relative_identity_target_delta": _mean(
            [float(row["relative_identity_target_delta"]) for row in relevant]
        ),
        "mean_relative_psi_input_delta": _mean(
            [float(row["relative_psi_input_delta"]) for row in relevant]
        ),
        "mean_bootstrap_target_cosine_similarity": _mean(
            [float(row["bootstrap_target_cosine_similarity"]) for row in relevant]
        ),
        "mean_identity_target_cosine_similarity": _mean(
            [float(row["identity_target_cosine_similarity"]) for row in relevant]
        ),
        "mean_state_slot_displacement": _mean(
            [float(row["state_slot_displacement"]) for row in relevant]
        ),
    }


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
    val_loss, val_accuracy = fmpc_tf2_module._evaluate_slow_pc_accuracy(
        cell.model,
        split.x_val,
        split.y_val,
    )
    test_loss, test_accuracy = fmpc_tf2_module._evaluate_slow_pc_accuracy(
        cell.model,
        split.x_test,
        split.y_test,
    )
    evaluation_wall_time_seconds = float(perf_counter() - evaluation_start)

    val_baseline_accuracy = majority_class_baseline_accuracy(split.y_val)
    test_baseline_accuracy = majority_class_baseline_accuracy(split.y_test)
    resolved_theta_update_cadence = fmpc_tf2_module._resolved_theta_update_cadence(cell.config)
    theta_micro_lr, theta_micro_bias_lr = fmpc_tf2_module._theta_micro_learning_rates(
        cell.config,
        resolved_theta_update_cadence,
    )
    train_wall_time_seconds = float(cell.summary["train_wall_time_seconds"]) if cell.summary else 0.0

    summary = {
        "phase": "Phase Incremental Bridge",
        "stage": "partial_open_loop_handoff_diagnostic",
        "diagnostic_only": True,
        "preset_name": cell.config.preset_name,
        "candidate_key": cell.candidate.key,
        "handoff_mode": cell.spec.mode,
        "replay_prefix_steps": int(cell.spec.replay_prefix_steps),
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
            "train_wall_time_seconds": train_wall_time_seconds,
            "final_evaluation_wall_time_seconds": evaluation_wall_time_seconds,
        },
    }
    _write_run_artifacts(
        cell.run_dir,
        cell.config,
        mode=cell.spec.mode,
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


def _run_one_seed(
    base_run_dir: Path,
    suite_config: FMPCTF2PartialOpenLoopHandoffSuiteConfig,
    *,
    candidates: dict[str, FMPCTF2PartialHandoffCandidate],
    seed: int,
) -> tuple[list[_RunArtifacts], list[dict[str, Any]]]:
    cells: dict[tuple[str, HandoffMode], _CellRuntime] = {}
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
    step_drift_rows: list[dict[str, Any]] = []
    train_start = perf_counter()

    for epoch_index in range(int(baseline_control.config.epochs)):
        lambda_id = fmpc_tf2_module._lambda_id_for_epoch(baseline_control.config, epoch_index)
        stage = fmpc_tf2_module._stage_for_epoch(baseline_control.config, epoch_index)
        epoch_metric_lists: dict[tuple[str, HandoffMode], dict[str, list[float]]] = {
            cell_key: {
                "train_loss": [],
                "train_boot_loss": [],
                "train_identity_loss": [],
                "train_transported_final_energy": [],
            }
            for cell_key in cells
        }
        epoch_step_drift_lists: dict[HandoffMode, dict[int, dict[str, list[float]]]] = {
            spec.mode: {
                step_index: {
                    "relative_bootstrap_target_delta": [],
                    "relative_identity_target_delta": [],
                    "relative_psi_input_delta": [],
                    "bootstrap_target_cosine_similarity": [],
                    "identity_target_cosine_similarity": [],
                    "state_slot_displacement": [],
                }
                for step_index in range(int(baseline_control.config.micro_steps))
            }
            for spec in _challenger_handoff_modes()
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
                train_loss, boot_loss, identity_loss, transported_energy, step_metrics = _train_one_batch_partial_handoff(
                    cell.model,
                    cell.psi_network,
                    cell.config,
                    x_batch,
                    y_batch,
                    baseline_slots=baseline_slots,
                    replay_prefix_steps=int(spec.replay_prefix_steps),
                )
                epoch_metric_lists[(spec.candidate_key, spec.mode)]["train_loss"].append(train_loss)
                epoch_metric_lists[(spec.candidate_key, spec.mode)]["train_boot_loss"].append(boot_loss)
                epoch_metric_lists[(spec.candidate_key, spec.mode)]["train_identity_loss"].append(identity_loss)
                epoch_metric_lists[(spec.candidate_key, spec.mode)]["train_transported_final_energy"].append(
                    transported_energy
                )
                if spec.candidate_key != "residualized_local_field_poly_rt2":
                    continue
                for step_index, metrics in enumerate(step_metrics):
                    for metric_name, metric_value in metrics.items():
                        epoch_step_drift_lists[spec.mode][step_index][metric_name].append(float(metric_value))

        for spec in _cell_specs():
            cell = cells[(spec.candidate_key, spec.mode)]
            val_transport = fmpc_tf2_module._evaluate_transport_split(
                cell.model,
                cell.psi_network,
                cell.config,
                split.x_val,
                split.y_val,
            )
            _, val_accuracy = fmpc_tf2_module._evaluate_slow_pc_accuracy(
                cell.model,
                split.x_val,
                split.y_val,
            )
            val_baseline_accuracy = majority_class_baseline_accuracy(split.y_val)
            val_energy_delta_vs_identity = (
                val_transport.transported_final_energy - val_transport.identity_final_energy
            )
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

        for spec in _challenger_handoff_modes():
            for step_index in range(int(baseline_control.config.micro_steps)):
                aggregated = {
                    metric_name: _mean(metric_values)
                    for metric_name, metric_values in epoch_step_drift_lists[spec.mode][step_index].items()
                }
                step_drift_rows.append(
                    {
                        "seed": int(seed),
                        "epoch": int(epoch_index + 1),
                        "candidate_key": spec.candidate_key,
                        "handoff_mode": spec.mode,
                        "replay_prefix_steps": int(spec.replay_prefix_steps),
                        "step_index": int(step_index),
                        **aggregated,
                    }
                )

    train_wall_time_seconds = float(perf_counter() - train_start)
    for cell in cells.values():
        cell.summary = {"train_wall_time_seconds": train_wall_time_seconds}

    artifacts: list[_RunArtifacts] = []
    for spec in _cell_specs():
        cell = cells[(spec.candidate_key, spec.mode)]
        artifacts.append(
            _finalize_cell(
                base_run_dir,
                cell,
                split,
                seed=seed,
            )
        )
    return artifacts, step_drift_rows


def run_fmpc_tf2_partial_open_loop_handoff_suite(
    config: FMPCTF2PartialOpenLoopHandoffSuiteConfig,
) -> FMPCTF2PartialOpenLoopHandoffSuiteRunResult:
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
    step_drift_rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        artifacts, seed_step_rows = _run_one_seed(
            run_dir,
            config,
            candidates=candidates,
            seed=int(seed),
        )
        run_rows.extend([_success_run_row(artifact=artifact, base_run_dir=run_dir) for artifact in artifacts])
        step_drift_rows.extend(seed_step_rows)

    _write_csv(run_dir / "end_to_end_runs.csv", run_rows)
    _write_csv(run_dir / "stepwise_handoff_drift_epoch_metrics.csv", step_drift_rows)

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
    challenger_k0 = _aggregate_run_group(
        run_rows,
        mode="handoff_k0_closed_loop_live_plan",
        candidate_key="residualized_local_field_poly_rt2",
    )
    challenger_k1 = _aggregate_run_group(
        run_rows,
        mode="handoff_k1_baseline_prefix_then_live",
        candidate_key="residualized_local_field_poly_rt2",
    )
    challenger_k2 = _aggregate_run_group(
        run_rows,
        mode="handoff_k2_baseline_prefix_then_live",
        candidate_key="residualized_local_field_poly_rt2",
    )
    challenger_k3 = _aggregate_run_group(
        run_rows,
        mode="handoff_k3_baseline_prefix_then_live",
        candidate_key="residualized_local_field_poly_rt2",
    )
    challenger_k4 = _aggregate_run_group(
        run_rows,
        mode="handoff_k4_open_loop_baseline_plan_replay",
        candidate_key="residualized_local_field_poly_rt2",
    )

    stepwise_drift_summary = {
        mode: {
            f"step_{step_index}": _aggregate_step_drift(
                step_drift_rows,
                candidate_key="residualized_local_field_poly_rt2",
                mode=mode,
                step_index=step_index,
            )
            for step_index in range(4)
        }
        for mode in (
            "handoff_k0_closed_loop_live_plan",
            "handoff_k1_baseline_prefix_then_live",
            "handoff_k2_baseline_prefix_then_live",
            "handoff_k3_baseline_prefix_then_live",
            "handoff_k4_open_loop_baseline_plan_replay",
        )
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
        "residualized_local_field_poly_rt2__handoff_k0_closed_loop_live_plan": _pairwise_vs_reference(
            run_rows,
            candidate_key="residualized_local_field_poly_rt2",
            mode="handoff_k0_closed_loop_live_plan",
            reference_candidate_key="baseline_plain_raw",
            reference_mode="closed_loop_live_plan",
        ),
        "residualized_local_field_poly_rt2__handoff_k1_baseline_prefix_then_live": _pairwise_vs_reference(
            run_rows,
            candidate_key="residualized_local_field_poly_rt2",
            mode="handoff_k1_baseline_prefix_then_live",
            reference_candidate_key="baseline_plain_raw",
            reference_mode="closed_loop_live_plan",
        ),
        "residualized_local_field_poly_rt2__handoff_k2_baseline_prefix_then_live": _pairwise_vs_reference(
            run_rows,
            candidate_key="residualized_local_field_poly_rt2",
            mode="handoff_k2_baseline_prefix_then_live",
            reference_candidate_key="baseline_plain_raw",
            reference_mode="closed_loop_live_plan",
        ),
        "residualized_local_field_poly_rt2__handoff_k3_baseline_prefix_then_live": _pairwise_vs_reference(
            run_rows,
            candidate_key="residualized_local_field_poly_rt2",
            mode="handoff_k3_baseline_prefix_then_live",
            reference_candidate_key="baseline_plain_raw",
            reference_mode="closed_loop_live_plan",
        ),
        "residualized_local_field_poly_rt2__handoff_k4_open_loop_baseline_plan_replay": _pairwise_vs_reference(
            run_rows,
            candidate_key="residualized_local_field_poly_rt2",
            mode="handoff_k4_open_loop_baseline_plan_replay",
            reference_candidate_key="baseline_plain_raw",
            reference_mode="closed_loop_live_plan",
        ),
    }

    pairwise_vs_challenger_k0 = {
        mode: _pairwise_vs_reference(
            run_rows,
            candidate_key="residualized_local_field_poly_rt2",
            mode=mode,
            reference_candidate_key="residualized_local_field_poly_rt2",
            reference_mode="handoff_k0_closed_loop_live_plan",
        )
        for mode in (
            "handoff_k0_closed_loop_live_plan",
            "handoff_k1_baseline_prefix_then_live",
            "handoff_k2_baseline_prefix_then_live",
            "handoff_k3_baseline_prefix_then_live",
            "handoff_k4_open_loop_baseline_plan_replay",
        )
    }
    pairwise_vs_challenger_k4 = {
        mode: _pairwise_vs_reference(
            run_rows,
            candidate_key="residualized_local_field_poly_rt2",
            mode=mode,
            reference_candidate_key="residualized_local_field_poly_rt2",
            reference_mode="handoff_k4_open_loop_baseline_plan_replay",
        )
        for mode in (
            "handoff_k0_closed_loop_live_plan",
            "handoff_k1_baseline_prefix_then_live",
            "handoff_k2_baseline_prefix_then_live",
            "handoff_k3_baseline_prefix_then_live",
            "handoff_k4_open_loop_baseline_plan_replay",
        )
    }

    rescue_mode: str | None = None
    for mode in (
        "handoff_k1_baseline_prefix_then_live",
        "handoff_k2_baseline_prefix_then_live",
        "handoff_k3_baseline_prefix_then_live",
        "handoff_k4_open_loop_baseline_plan_replay",
    ):
        same_family = pairwise_vs_challenger_k0[mode]
        vs_k4 = pairwise_vs_challenger_k4[mode]
        if (
            float(same_family["mean_test_accuracy_delta"]) >= float(config.material_test_gain)
            and abs(float(vs_k4["mean_test_accuracy_delta"])) <= float(config.baseline_similarity_tolerance)
        ):
            rescue_mode = mode
            break

    if rescue_mode == "handoff_k1_baseline_prefix_then_live":
        dominant_interpretation = "failure_triggered_by_the_first_self_induced_closed_loop_steps"
        next_single_narrow_move = (
            "run one narrow first-step-only versus first-two-step replay split to separate whether the very first "
            "challenger-controlled transition or the first short closed-loop segment causes the collapse"
        )
    elif rescue_mode == "handoff_k2_baseline_prefix_then_live":
        dominant_interpretation = "failure_emerges_when_the_challenger_controls_the_mid_to_late_trajectory"
        next_single_narrow_move = (
            "run one narrow late-trajectory handoff diagnostic that holds the first two steps fixed and "
            "separates step-3 from step-4 challenger control"
        )
    elif rescue_mode == "handoff_k3_baseline_prefix_then_live":
        dominant_interpretation = "instability_concentrated_in_the_final_live_handoff_step"
        next_single_narrow_move = (
            "run one narrow final-step-only replay diagnostic that isolates the terminal challenger-controlled "
            "micro-step without changing the rest of the rollout"
        )
    elif rescue_mode == "handoff_k4_open_loop_baseline_plan_replay":
        dominant_interpretation = "failure_is_cumulative_across_the_whole_self_induced_closed_loop_plan"
        next_single_narrow_move = (
            "run one narrow two-phase rollout diagnostic that compares early-only versus late-only live control "
            "to see whether the cumulative failure is front-loaded or uniformly distributed"
        )
    else:
        dominant_interpretation = "no_partial_handoff_rescue_despite_full_replay_endpoint"
        next_single_narrow_move = (
            "run one narrow handoff-discontinuity diagnostic that keeps the replay prefix but smooths the switch "
            "back to live rollout before reopening any broader TF2 mechanism search"
        )

    summary = {
        "phase": "Phase Incremental Bridge",
        "stage": "partial_open_loop_handoff_diagnostic",
        "diagnostic_only": True,
        "end_to_end_summary": {
            "baseline_plain_raw": {
                "closed_loop_live_plan": baseline_live,
                "open_loop_baseline_plan_replay": baseline_open,
            },
            "residualized_local_field_poly_rt2": {
                "handoff_k0_closed_loop_live_plan": challenger_k0,
                "handoff_k1_baseline_prefix_then_live": challenger_k1,
                "handoff_k2_baseline_prefix_then_live": challenger_k2,
                "handoff_k3_baseline_prefix_then_live": challenger_k3,
                "handoff_k4_open_loop_baseline_plan_replay": challenger_k4,
            },
        },
        "stepwise_plan_drift_summary": stepwise_drift_summary,
        "pairwise_delta_vs_baseline_closed_loop_run": pairwise_vs_baseline_live,
        "pairwise_delta_vs_challenger_k0_closed_loop_run": pairwise_vs_challenger_k0,
        "pairwise_delta_vs_challenger_k4_open_loop_replay_run": pairwise_vs_challenger_k4,
        "rescue_onset_mode": rescue_mode,
        "dominant_interpretation": dominant_interpretation,
        "next_single_narrow_move": next_single_narrow_move,
        "artifacts": {
            "end_to_end_runs_csv": "end_to_end_runs.csv",
            "stepwise_handoff_drift_epoch_metrics_csv": "stepwise_handoff_drift_epoch_metrics.csv",
        },
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF2PartialOpenLoopHandoffSuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config, candidates),
        run_rows=run_rows,
        step_drift_rows=step_drift_rows,
        summary=summary,
    )
