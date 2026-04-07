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
from .datasets import load_digits_split
from .fmpc_tf1 import _select_tf1_checkpoint_epoch, build_tf1_epoch_selection_diagnostics
from .fmpc_tf2 import FMPCTF2Config, FMPCTF2EpochMetrics, build_tf2_corrective_transport_default_config
from .metrics import majority_class_baseline_accuracy

TrajectoryMode = Literal["closed_loop_live_plan", "open_loop_baseline_plan_replay"]


@dataclass(frozen=True)
class FMPCTF2OpenClosedCandidate:
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


@dataclass
class FMPCTF2OpenVsClosedLoopSuiteConfig:
    """Diagnostic-only open-vs-closed-loop trajectory coupling suite."""

    experiment_name: str = "fmpc_tf2_open_vs_closed_loop_suite"
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
    nontrivial_plan_drift_threshold: float = 0.01

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"


@dataclass
class FMPCTF2OpenVsClosedLoopSuiteRunResult:
    run_dir: Path
    config: dict[str, Any]
    run_rows: list[dict[str, Any]]
    plan_drift_rows: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass
class _RunArtifacts:
    mode: TrajectoryMode
    candidate: FMPCTF2OpenClosedCandidate
    seed: int
    config: FMPCTF2Config
    run_dir: Path
    epoch_rows: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass
class _CellRuntime:
    mode: TrajectoryMode
    candidate: FMPCTF2OpenClosedCandidate
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


def _candidate_registry() -> list[FMPCTF2OpenClosedCandidate]:
    return [
        FMPCTF2OpenClosedCandidate(
            key="baseline_plain_raw",
            psi_family="baseline_plain",
            time_encoding_variant="raw",
            notes="current corrective default baseline",
        ),
        FMPCTF2OpenClosedCandidate(
            key="residualized_local_field_poly_rt2",
            psi_family="residualized_local_field",
            time_encoding_variant="poly_rt2",
            notes="offline-better challenger from the psi-expressivity suite",
        ),
    ]


def _suite_config_payload(
    config: FMPCTF2OpenVsClosedLoopSuiteConfig,
    candidates: list[FMPCTF2OpenClosedCandidate],
) -> dict[str, Any]:
    return {
        "phase": "Phase TF2",
        "stage": "open_vs_closed_loop_trajectory_coupling_suite",
        "seeds": [int(seed) for seed in config.seeds],
        "candidates": [
            {
                "key": candidate.key,
                "psi_family": candidate.psi_family,
                "time_encoding_variant": candidate.time_encoding_variant,
                "notes": candidate.notes,
            }
            for candidate in candidates
        ],
        "modes": ["closed_loop_live_plan", "open_loop_baseline_plan_replay"],
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
        "nontrivial_plan_drift_threshold": float(config.nontrivial_plan_drift_threshold),
        "replay_definition": (
            "open_loop_baseline_plan_replay caches the batch-start micro-step plan from the baseline "
            "closed-loop candidate and reuses that same plan for both baseline and challenger without "
            "letting the challenger regenerate its own intermediate trajectory."
        ),
    }


def _build_candidate_config(
    candidate: FMPCTF2OpenClosedCandidate,
    *,
    seed: int,
    suite_config: FMPCTF2OpenVsClosedLoopSuiteConfig,
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
        raise ValueError("Open-vs-closed-loop diagnostic is defined only for local_only supervision.")
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


def _plan_drift_from_slots(
    candidate_config: FMPCTF2Config,
    baseline_slots: list[_ReplayPlanSlot],
    self_slots: list[_ReplayPlanSlot],
) -> dict[str, float]:
    if len(baseline_slots) != len(self_slots):
        raise ValueError("Baseline and self-induced plan lengths must match.")
    bootstrap_deltas: list[float] = []
    identity_deltas: list[float] = []
    psi_input_deltas: list[float] = []
    bootstrap_cosines: list[float] = []
    identity_cosines: list[float] = []
    state_displacements: list[float] = []
    for baseline_slot, self_slot in zip(baseline_slots, self_slots, strict=True):
        replay_inputs = _candidate_replay_inputs(candidate_config, baseline_slot)
        bootstrap_deltas.append(
            _relative_target_delta(baseline_slot.boot_targets, self_slot.boot_targets)
        )
        identity_deltas.append(
            _relative_target_delta(baseline_slot.identity_targets, self_slot.identity_targets)
        )
        psi_input_deltas.append(_relative_target_delta(replay_inputs, self_slot.psi_inputs))
        bootstrap_cosines.append(
            _mean_cosine_similarity(baseline_slot.boot_targets, self_slot.boot_targets)
        )
        identity_cosines.append(
            _mean_cosine_similarity(baseline_slot.identity_targets, self_slot.identity_targets)
        )
        state_displacements.append(
            _mean_state_slot_displacement(baseline_slot.z_on_next, self_slot.z_on_next)
        )
    return {
        "mean_relative_bootstrap_target_delta": _mean(bootstrap_deltas),
        "mean_relative_identity_target_delta": _mean(identity_deltas),
        "mean_relative_psi_input_delta": _mean(psi_input_deltas),
        "mean_bootstrap_target_cosine_similarity": _mean(bootstrap_cosines),
        "mean_identity_target_cosine_similarity": _mean(identity_cosines),
        "mean_state_slot_displacement": _mean(state_displacements),
    }


def _train_one_batch_open_loop_replay(
    model: Any,
    psi_network: Any,
    config: FMPCTF2Config,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    *,
    baseline_slots: list[_ReplayPlanSlot],
) -> tuple[float, float, float, float]:
    if config.supervision_policy != "local_only":
        raise ValueError("Open-loop replay diagnostic is defined only for local_only supervision.")
    if fmpc_tf2_module._resolved_theta_update_cadence(config) != "terminal_only":
        raise ValueError("Open-loop replay diagnostic expects terminal_only theta cadence.")
    if not baseline_slots:
        raise ValueError("baseline_slots must contain at least one micro-step.")

    context = fmpc_tf2_module.build_tf1_context(model, x_batch, y_batch)
    total_losses: list[float] = []
    boot_losses: list[float] = []
    identity_losses: list[float] = []
    for slot in baseline_slots:
        psi_inputs = _candidate_replay_inputs(config, slot)
        psi_predictions = fmpc_tf2_module._psi_predict(psi_network, psi_inputs, config)
        boot_loss = float(np.mean((psi_predictions - slot.boot_targets) ** 2))
        identity_loss = float(np.mean((psi_predictions - slot.identity_targets) ** 2))
        if slot.lambda_id > 0.0:
            combined_target = (
                slot.boot_targets + (float(slot.lambda_id) * slot.identity_targets)
            ) / (1.0 + float(slot.lambda_id))
            loss_scale = 1.0 + float(slot.lambda_id)
        else:
            combined_target = slot.boot_targets
            loss_scale = 1.0
        total_loss = boot_loss + (float(slot.lambda_id) * identity_loss)
        fmpc_tf2_module._weighted_mse_step(
            psi_network,
            psi_inputs,
            combined_target,
            loss_scale=loss_scale,
        )
        total_losses.append(total_loss)
        boot_losses.append(boot_loss)
        identity_losses.append(identity_loss)

    transported_z = baseline_slots[-1].z_on_next
    transported_final_energy = fmpc_tf2_module.hidden_energy_from_state(context, transported_z)
    fmpc_tf2_module._theta_update_from_transported_state(
        model,
        context,
        transported_z,
        eta_w=float(config.eta_w),
        eta_b=fmpc_tf2_module._resolved_eta_b(config),
    )
    return (
        float(np.mean(total_losses)),
        float(np.mean(boot_losses)),
        float(np.mean(identity_losses)),
        float(transported_final_energy),
    )


def _write_run_artifacts(
    run_dir: Path,
    config: FMPCTF2Config,
    *,
    mode: TrajectoryMode,
    epoch_rows: list[dict[str, Any]],
    selection_diagnostics: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    config_payload = fmpc_tf2_module._config_payload(config)
    config_payload["trajectory_mode"] = mode
    config_payload["diagnostic_only"] = True
    if mode == "open_loop_baseline_plan_replay":
        config_payload["replay_reference_candidate"] = "baseline_plain_raw"
    fmpc_tf2_module._write_json(run_dir / "config.json", config_payload)
    fmpc_tf2_module._write_epoch_metrics(run_dir / "epoch_metrics.csv", epoch_rows)
    fmpc_tf2_module._write_json(run_dir / "selection_diagnostics.json", selection_diagnostics)
    fmpc_tf2_module._write_json(run_dir / "summary.json", summary)


def _make_cell(
    base_run_dir: Path,
    candidate: FMPCTF2OpenClosedCandidate,
    mode: TrajectoryMode,
    *,
    seed: int,
    suite_config: FMPCTF2OpenVsClosedLoopSuiteConfig,
) -> _CellRuntime:
    config = _build_candidate_config(candidate, seed=seed, suite_config=suite_config)
    run_dir = base_run_dir / "runs" / mode / candidate.key / f"seed_{int(seed)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return _CellRuntime(
        mode=mode,
        candidate=candidate,
        config=config,
        run_dir=run_dir,
        model=fmpc_tf2_module._make_pc_model(config),
        psi_network=fmpc_tf2_module._make_psi_network(config),
    )


def _success_run_row(*, artifact: _RunArtifacts, base_run_dir: Path) -> dict[str, Any]:
    timing = dict(artifact.summary.get("timing", {}))
    return {
        "trajectory_mode": artifact.mode,
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
    mode: TrajectoryMode,
    candidate_key: str,
) -> dict[str, Any]:
    relevant = [
        row
        for row in run_rows
        if row["trajectory_mode"] == mode
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


def _aggregate_plan_drift_group(
    drift_rows: list[dict[str, Any]],
    *,
    candidate_key: str,
) -> dict[str, Any]:
    relevant = [row for row in drift_rows if row["candidate_key"] == candidate_key]
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
            [float(row["mean_relative_bootstrap_target_delta"]) for row in relevant]
        ),
        "mean_relative_identity_target_delta": _mean(
            [float(row["mean_relative_identity_target_delta"]) for row in relevant]
        ),
        "mean_relative_psi_input_delta": _mean(
            [float(row["mean_relative_psi_input_delta"]) for row in relevant]
        ),
        "mean_bootstrap_target_cosine_similarity": _mean(
            [float(row["mean_bootstrap_target_cosine_similarity"]) for row in relevant]
        ),
        "mean_identity_target_cosine_similarity": _mean(
            [float(row["mean_identity_target_cosine_similarity"]) for row in relevant]
        ),
        "mean_state_slot_displacement": _mean(
            [float(row["mean_state_slot_displacement"]) for row in relevant]
        ),
    }


def _pairwise_vs_reference(
    run_rows: list[dict[str, Any]],
    *,
    candidate_key: str,
    mode: TrajectoryMode,
    reference_candidate_key: str,
    reference_mode: TrajectoryMode,
) -> dict[str, Any]:
    candidate_by_seed = {
        int(row["seed"]): row
        for row in run_rows
        if row["candidate_key"] == candidate_key
        and row["trajectory_mode"] == mode
        and row["run_status"] == "success"
    }
    reference_by_seed = {
        int(row["seed"]): row
        for row in run_rows
        if row["candidate_key"] == reference_candidate_key
        and row["trajectory_mode"] == reference_mode
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


def _finalize_cell(
    base_run_dir: Path,
    cell: _CellRuntime,
    split: Any,
    candidate_drift_rows: list[dict[str, Any]],
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
    plan_drift_summary = _aggregate_plan_drift_group(candidate_drift_rows, candidate_key=cell.candidate.key)

    summary = {
        "phase": "Phase TF2",
        "stage": "open_vs_closed_loop_trajectory_coupling_diagnostic",
        "diagnostic_only": True,
        "preset_name": cell.config.preset_name,
        "candidate_key": cell.candidate.key,
        "trajectory_mode": cell.mode,
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
        "plan_drift": plan_drift_summary,
        "timing": {
            "train_wall_time_seconds": train_wall_time_seconds,
            "final_evaluation_wall_time_seconds": evaluation_wall_time_seconds,
        },
    }
    _write_run_artifacts(
        cell.run_dir,
        cell.config,
        mode=cell.mode,
        epoch_rows=cell.epoch_rows,
        selection_diagnostics=selection_diagnostics,
        summary=summary,
    )
    return _RunArtifacts(
        mode=cell.mode,
        candidate=cell.candidate,
        seed=int(seed),
        config=cell.config,
        run_dir=cell.run_dir,
        epoch_rows=cell.epoch_rows,
        summary=summary,
    )


def _run_one_seed(
    base_run_dir: Path,
    suite_config: FMPCTF2OpenVsClosedLoopSuiteConfig,
    *,
    candidates: list[FMPCTF2OpenClosedCandidate],
    seed: int,
) -> tuple[list[_RunArtifacts], list[dict[str, Any]]]:
    baseline_candidate, challenger_candidate = candidates
    cells: dict[tuple[str, TrajectoryMode], _CellRuntime] = {}
    for mode in ("closed_loop_live_plan", "open_loop_baseline_plan_replay"):
        for candidate in candidates:
            cells[(candidate.key, mode)] = _make_cell(
                base_run_dir,
                candidate,
                mode,
                seed=seed,
                suite_config=suite_config,
            )

    baseline_config = cells[(baseline_candidate.key, "closed_loop_live_plan")].config
    split = load_digits_split(
        split_seed=baseline_config.data_seed,
        train_fraction=baseline_config.train_fraction,
        val_fraction=baseline_config.val_fraction,
        test_fraction=baseline_config.test_fraction,
    )
    plan_drift_rows: list[dict[str, Any]] = []

    train_start = perf_counter()
    for epoch_index in range(int(baseline_config.epochs)):
        lambda_id = fmpc_tf2_module._lambda_id_for_epoch(baseline_config, epoch_index)
        stage = fmpc_tf2_module._stage_for_epoch(baseline_config, epoch_index)
        epoch_metric_lists: dict[tuple[str, TrajectoryMode], dict[str, list[float]]] = {
            cell_key: {
                "train_loss": [],
                "train_boot_loss": [],
                "train_identity_loss": [],
                "train_transported_final_energy": [],
            }
            for cell_key in cells
        }
        epoch_drift_lists: dict[str, dict[str, list[float]]] = {
            candidate.key: {
                "mean_relative_bootstrap_target_delta": [],
                "mean_relative_identity_target_delta": [],
                "mean_relative_psi_input_delta": [],
                "mean_bootstrap_target_cosine_similarity": [],
                "mean_identity_target_cosine_similarity": [],
                "mean_state_slot_displacement": [],
            }
            for candidate in candidates
        }
        batch_seed = baseline_config.batch_order_seed + epoch_index
        for x_batch, y_batch in fmpc_tf2_module.iter_minibatches(
            split.x_train,
            split.y_train,
            baseline_config.batch_size,
            shuffle=baseline_config.shuffle_batches,
            seed=batch_seed,
        ):
            baseline_live = cells[(baseline_candidate.key, "closed_loop_live_plan")]
            challenger_live = cells[(challenger_candidate.key, "closed_loop_live_plan")]

            baseline_context = fmpc_tf2_module.build_tf1_context(baseline_live.model, x_batch, y_batch)
            baseline_slots = _build_cached_plan(
                baseline_context,
                baseline_live.psi_network,
                baseline_live.config,
                lambda_id=float(lambda_id),
            )
            challenger_context = fmpc_tf2_module.build_tf1_context(challenger_live.model, x_batch, y_batch)
            challenger_self_slots = _build_cached_plan(
                challenger_context,
                challenger_live.psi_network,
                challenger_live.config,
                lambda_id=float(lambda_id),
            )

            baseline_drift = _plan_drift_from_slots(
                baseline_live.config,
                baseline_slots,
                baseline_slots,
            )
            challenger_drift = _plan_drift_from_slots(
                challenger_live.config,
                baseline_slots,
                challenger_self_slots,
            )
            for metric_name, metric_value in baseline_drift.items():
                epoch_drift_lists[baseline_candidate.key][metric_name].append(float(metric_value))
            for metric_name, metric_value in challenger_drift.items():
                epoch_drift_lists[challenger_candidate.key][metric_name].append(float(metric_value))

            for candidate in candidates:
                live_cell = cells[(candidate.key, "closed_loop_live_plan")]
                train_loss, boot_loss, identity_loss, transported_energy = fmpc_tf2_module._train_one_batch_tf2(
                    live_cell.model,
                    live_cell.psi_network,
                    live_cell.config,
                    x_batch,
                    y_batch,
                    lambda_id=float(lambda_id),
                    epoch_index=epoch_index,
                )
                epoch_metric_lists[(candidate.key, "closed_loop_live_plan")]["train_loss"].append(train_loss)
                epoch_metric_lists[(candidate.key, "closed_loop_live_plan")]["train_boot_loss"].append(boot_loss)
                epoch_metric_lists[(candidate.key, "closed_loop_live_plan")]["train_identity_loss"].append(identity_loss)
                epoch_metric_lists[(candidate.key, "closed_loop_live_plan")]["train_transported_final_energy"].append(
                    transported_energy
                )

            for candidate in candidates:
                replay_cell = cells[(candidate.key, "open_loop_baseline_plan_replay")]
                train_loss, boot_loss, identity_loss, transported_energy = _train_one_batch_open_loop_replay(
                    replay_cell.model,
                    replay_cell.psi_network,
                    replay_cell.config,
                    x_batch,
                    y_batch,
                    baseline_slots=baseline_slots,
                )
                epoch_metric_lists[(candidate.key, "open_loop_baseline_plan_replay")]["train_loss"].append(train_loss)
                epoch_metric_lists[(candidate.key, "open_loop_baseline_plan_replay")]["train_boot_loss"].append(
                    boot_loss
                )
                epoch_metric_lists[(candidate.key, "open_loop_baseline_plan_replay")]["train_identity_loss"].append(
                    identity_loss
                )
                epoch_metric_lists[(candidate.key, "open_loop_baseline_plan_replay")][
                    "train_transported_final_energy"
                ].append(transported_energy)

        for cell in cells.values():
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
            metric_lists = epoch_metric_lists[(cell.candidate.key, cell.mode)]
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

        for candidate in candidates:
            plan_drift_rows.append(
                {
                    "seed": int(seed),
                    "epoch": int(epoch_index + 1),
                    "candidate_key": candidate.key,
                    "mean_relative_bootstrap_target_delta": _mean(
                        epoch_drift_lists[candidate.key]["mean_relative_bootstrap_target_delta"]
                    ),
                    "mean_relative_identity_target_delta": _mean(
                        epoch_drift_lists[candidate.key]["mean_relative_identity_target_delta"]
                    ),
                    "mean_relative_psi_input_delta": _mean(
                        epoch_drift_lists[candidate.key]["mean_relative_psi_input_delta"]
                    ),
                    "mean_bootstrap_target_cosine_similarity": _mean(
                        epoch_drift_lists[candidate.key]["mean_bootstrap_target_cosine_similarity"]
                    ),
                    "mean_identity_target_cosine_similarity": _mean(
                        epoch_drift_lists[candidate.key]["mean_identity_target_cosine_similarity"]
                    ),
                    "mean_state_slot_displacement": _mean(
                        epoch_drift_lists[candidate.key]["mean_state_slot_displacement"]
                    ),
                }
            )

    train_wall_time_seconds = float(perf_counter() - train_start)
    for cell in cells.values():
        cell.summary = {"train_wall_time_seconds": train_wall_time_seconds}

    artifacts: list[_RunArtifacts] = []
    for cell in cells.values():
        candidate_drift_rows = [
            row
            for row in plan_drift_rows
            if row["candidate_key"] == cell.candidate.key and int(row["seed"]) == int(seed)
        ]
        artifacts.append(
            _finalize_cell(
                base_run_dir,
                cell,
                split,
                candidate_drift_rows,
                seed=seed,
            )
        )
    return artifacts, plan_drift_rows


def run_fmpc_tf2_open_vs_closed_loop_suite(
    config: FMPCTF2OpenVsClosedLoopSuiteConfig,
) -> FMPCTF2OpenVsClosedLoopSuiteRunResult:
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
    plan_drift_rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        artifacts, seed_drift_rows = _run_one_seed(
            run_dir,
            config,
            candidates=candidates,
            seed=int(seed),
        )
        run_rows.extend([_success_run_row(artifact=artifact, base_run_dir=run_dir) for artifact in artifacts])
        plan_drift_rows.extend(seed_drift_rows)

    _write_csv(run_dir / "end_to_end_runs.csv", run_rows)
    _write_csv(run_dir / "plan_drift_epoch_metrics.csv", plan_drift_rows)

    closed_baseline = _aggregate_run_group(
        run_rows,
        mode="closed_loop_live_plan",
        candidate_key="baseline_plain_raw",
    )
    replay_baseline = _aggregate_run_group(
        run_rows,
        mode="open_loop_baseline_plan_replay",
        candidate_key="baseline_plain_raw",
    )
    closed_challenger = _aggregate_run_group(
        run_rows,
        mode="closed_loop_live_plan",
        candidate_key="residualized_local_field_poly_rt2",
    )
    replay_challenger = _aggregate_run_group(
        run_rows,
        mode="open_loop_baseline_plan_replay",
        candidate_key="residualized_local_field_poly_rt2",
    )

    baseline_plan_drift = _aggregate_plan_drift_group(
        plan_drift_rows,
        candidate_key="baseline_plain_raw",
    )
    challenger_plan_drift = _aggregate_plan_drift_group(
        plan_drift_rows,
        candidate_key="residualized_local_field_poly_rt2",
    )

    baseline_same_family_closed_delta = _pairwise_vs_reference(
        run_rows,
        candidate_key="baseline_plain_raw",
        mode="open_loop_baseline_plan_replay",
        reference_candidate_key="baseline_plain_raw",
        reference_mode="closed_loop_live_plan",
    )
    challenger_same_family_closed_delta = _pairwise_vs_reference(
        run_rows,
        candidate_key="residualized_local_field_poly_rt2",
        mode="open_loop_baseline_plan_replay",
        reference_candidate_key="residualized_local_field_poly_rt2",
        reference_mode="closed_loop_live_plan",
    )

    pairwise_vs_baseline_closed = {
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
        "residualized_local_field_poly_rt2__closed_loop_live_plan": _pairwise_vs_reference(
            run_rows,
            candidate_key="residualized_local_field_poly_rt2",
            mode="closed_loop_live_plan",
            reference_candidate_key="baseline_plain_raw",
            reference_mode="closed_loop_live_plan",
        ),
        "residualized_local_field_poly_rt2__open_loop_baseline_plan_replay": _pairwise_vs_reference(
            run_rows,
            candidate_key="residualized_local_field_poly_rt2",
            mode="open_loop_baseline_plan_replay",
            reference_candidate_key="baseline_plain_raw",
            reference_mode="closed_loop_live_plan",
        ),
    }
    pairwise_vs_same_family_closed = {
        "baseline_plain_raw": baseline_same_family_closed_delta,
        "residualized_local_field_poly_rt2": challenger_same_family_closed_delta,
    }

    challenger_gain = float(challenger_same_family_closed_delta["mean_test_accuracy_delta"])
    baseline_gain = float(baseline_same_family_closed_delta["mean_test_accuracy_delta"])
    challenger_rescued = bool(
        challenger_gain >= float(config.material_test_gain)
        and abs(baseline_gain) <= float(config.baseline_similarity_tolerance)
    )
    replay_helps_generally = bool(
        challenger_gain >= float(config.material_test_gain)
        and baseline_gain >= float(config.material_test_gain)
        and abs(challenger_gain - baseline_gain) <= float(config.baseline_similarity_tolerance)
    )
    neither_improves_materially = bool(
        challenger_gain < float(config.material_test_gain) and baseline_gain < float(config.material_test_gain)
    )
    challenger_plan_drift_nontrivial = bool(
        float(challenger_plan_drift["mean_relative_bootstrap_target_delta"])
        >= float(config.nontrivial_plan_drift_threshold)
        or float(challenger_plan_drift["mean_relative_identity_target_delta"])
        >= float(config.nontrivial_plan_drift_threshold)
        or float(challenger_plan_drift["mean_relative_psi_input_delta"])
        >= float(config.nontrivial_plan_drift_threshold)
        or float(challenger_plan_drift["mean_state_slot_displacement"])
        >= float(config.nontrivial_plan_drift_threshold)
    )
    replay_reduces_plan_drift_but_not_behavior = bool(
        challenger_plan_drift_nontrivial and challenger_gain < float(config.material_test_gain)
    )

    if challenger_rescued:
        dominant_interpretation = "closed_loop_trajectory_state_coevolution"
        next_single_narrow_move = (
            "run one narrow partial-open-loop handoff diagnostic that replays the first corrective "
            "micro-steps but lets the final step close live, to localize where self-induced trajectory "
            "co-evolution becomes harmful"
        )
    elif replay_helps_generally:
        dominant_interpretation = "open_loop_replay_regularizes_both_candidates_not_challenger_specific"
        next_single_narrow_move = (
            "run one narrow baseline-only replay-ablation to separate generic optimization smoothing "
            "from challenger-specific trajectory coupling"
        )
    elif replay_reduces_plan_drift_but_not_behavior:
        dominant_interpretation = "deeper_optimization_or_selection_coupling_beyond_trajectory_refresh"
        next_single_narrow_move = (
            "run one narrow partial-open-loop handoff diagnostic to test whether replay only helps early "
            "trajectory slots while later live slots still break validation-selected behavior"
        )
    elif neither_improves_materially:
        dominant_interpretation = "simple_open_loop_baseline_plan_replay_not_main_limiter"
        next_single_narrow_move = (
            "run one narrow partial-open-loop handoff diagnostic before reopening any broader TF2 mechanism search"
        )
    else:
        dominant_interpretation = "mixed_or_ambiguous_open_vs_closed_loop_effect"
        next_single_narrow_move = (
            "run one narrow partial-open-loop handoff diagnostic to disambiguate whether only early or late "
            "closed-loop trajectory co-evolution matters"
        )

    summary = {
        "phase": "Phase TF2",
        "stage": "open_vs_closed_loop_trajectory_coupling_diagnostic",
        "diagnostic_only": True,
        "end_to_end_summary": {
            "baseline_plain_raw": {
                "closed_loop_live_plan": closed_baseline,
                "open_loop_baseline_plan_replay": replay_baseline,
            },
            "residualized_local_field_poly_rt2": {
                "closed_loop_live_plan": closed_challenger,
                "open_loop_baseline_plan_replay": replay_challenger,
            },
        },
        "plan_drift_summary": {
            "baseline_plain_raw": baseline_plan_drift,
            "residualized_local_field_poly_rt2": challenger_plan_drift,
        },
        "pairwise_delta_vs_baseline_closed_loop_run": pairwise_vs_baseline_closed,
        "pairwise_delta_vs_same_family_closed_loop_run": pairwise_vs_same_family_closed,
        "is_closed_loop_trajectory_coupling_present": challenger_rescued,
        "is_challenger_rescued_by_open_loop_baseline_plan_replay": challenger_rescued,
        "open_loop_replay_helps_generally_but_not_challenger_specifically": replay_helps_generally,
        "plan_drift_is_nontrivial_but_behavior_does_not_improve": replay_reduces_plan_drift_but_not_behavior,
        "dominant_interpretation": dominant_interpretation,
        "next_single_narrow_move": next_single_narrow_move,
        "artifacts": {
            "end_to_end_runs_csv": "end_to_end_runs.csv",
            "plan_drift_epoch_metrics_csv": "plan_drift_epoch_metrics.csv",
        },
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF2OpenVsClosedLoopSuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config, candidates),
        run_rows=run_rows,
        plan_drift_rows=plan_drift_rows,
        summary=summary,
    )
