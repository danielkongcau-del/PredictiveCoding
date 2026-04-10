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
    _aggregate_step_drift,
    _build_cached_plan,
    _build_candidate_config,
    _candidate_registry,
    _candidate_replay_inputs,
    _mean,
    _pairwise_vs_reference,
    _relative_posix,
    _step_drift_from_plans,
)
from ..metrics import majority_class_baseline_accuracy

MirroredMode = Literal[
    "closed_loop_live_plan",
    "open_loop_baseline_plan_replay",
    "k0_closed_loop_live_plan",
    "k4_open_loop_baseline_plan_replay",
    "suffix_live_k1",
    "suffix_live_k2",
    "suffix_live_k3",
    "prefix_live_k1",
    "prefix_live_k2",
    "prefix_live_k3",
]
LiveControlFamily = Literal["baseline_control", "full_live", "full_replay", "prefix", "suffix"]


@dataclass(frozen=True)
class _CellSpec:
    candidate_key: str
    mode: MirroredMode
    live_mask: tuple[bool, bool, bool, bool]
    live_step_count: int
    live_control_family: LiveControlFamily


@dataclass
class FMPCTF2MirroredHandoffAsymmetrySuiteConfig:
    """Diagnostic-only mirrored handoff asymmetry suite for fixed 4-step TF2 rollout."""

    experiment_name: str = "fmpc_tf2_mirrored_handoff_asymmetry_suite"
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
class FMPCTF2MirroredHandoffAsymmetrySuiteRunResult:
    run_dir: Path
    config: dict[str, Any]
    run_rows: list[dict[str, Any]]
    step_drift_rows: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass
class _RunArtifacts:
    mode: MirroredMode
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


def _baseline_specs() -> list[_CellSpec]:
    return [
        _CellSpec("baseline_plain_raw", "closed_loop_live_plan", (True, True, True, True), 4, "baseline_control"),
        _CellSpec(
            "baseline_plain_raw",
            "open_loop_baseline_plan_replay",
            (False, False, False, False),
            0,
            "baseline_control",
        ),
    ]


def _challenger_specs() -> list[_CellSpec]:
    return [
        _CellSpec(
            "residualized_local_field_poly_rt2",
            "k0_closed_loop_live_plan",
            (True, True, True, True),
            4,
            "full_live",
        ),
        _CellSpec(
            "residualized_local_field_poly_rt2",
            "k4_open_loop_baseline_plan_replay",
            (False, False, False, False),
            0,
            "full_replay",
        ),
        _CellSpec("residualized_local_field_poly_rt2", "suffix_live_k1", (False, False, False, True), 1, "suffix"),
        _CellSpec("residualized_local_field_poly_rt2", "suffix_live_k2", (False, False, True, True), 2, "suffix"),
        _CellSpec("residualized_local_field_poly_rt2", "suffix_live_k3", (False, True, True, True), 3, "suffix"),
        _CellSpec("residualized_local_field_poly_rt2", "prefix_live_k1", (True, False, False, False), 1, "prefix"),
        _CellSpec("residualized_local_field_poly_rt2", "prefix_live_k2", (True, True, False, False), 2, "prefix"),
        _CellSpec("residualized_local_field_poly_rt2", "prefix_live_k3", (True, True, True, False), 3, "prefix"),
    ]


def _cell_specs() -> list[_CellSpec]:
    return [*_baseline_specs(), *_challenger_specs()]


def _mirror_mode(mode: MirroredMode) -> MirroredMode | None:
    mirror_map: dict[MirroredMode, MirroredMode] = {
        "prefix_live_k1": "suffix_live_k1",
        "suffix_live_k1": "prefix_live_k1",
        "prefix_live_k2": "suffix_live_k2",
        "suffix_live_k2": "prefix_live_k2",
        "prefix_live_k3": "suffix_live_k3",
        "suffix_live_k3": "prefix_live_k3",
    }
    return mirror_map.get(mode)


def _suite_config_payload(
    config: FMPCTF2MirroredHandoffAsymmetrySuiteConfig,
    candidates: dict[str, FMPCTF2PartialHandoffCandidate],
) -> dict[str, Any]:
    return {
        "phase": "Phase Incremental Bridge",
        "stage": "mirrored_handoff_asymmetry_suite",
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
        "handoff_definition": (
            "Each batch caches the exact baseline corrective-default micro-step plan. "
            "Replay portions reuse the cached baseline slots. Live portions use the challenger's "
            "own generated rollout from the current handed-off state."
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
    config_payload["live_mask"] = [bool(flag) for flag in spec.live_mask]
    config_payload["live_step_count"] = int(spec.live_step_count)
    config_payload["live_control_family"] = spec.live_control_family
    config_payload["diagnostic_only"] = True
    if "baseline_plan_replay" in spec.mode:
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
    suite_config: FMPCTF2MirroredHandoffAsymmetrySuiteConfig,
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


def _train_one_batch_mirrored_handoff(
    model: Any,
    psi_network: Any,
    config: FMPCTF2Config,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    *,
    baseline_slots: list[_ReplayPlanSlot],
    live_mask: tuple[bool, bool, bool, bool],
) -> tuple[float, float, float, float, list[dict[str, float]]]:
    if config.supervision_policy != "local_only":
        raise ValueError("Mirrored handoff diagnostic is defined only for local_only supervision.")
    if fmpc_tf2_module._resolved_theta_update_cadence(config) != "terminal_only":
        raise ValueError("Mirrored handoff diagnostic expects terminal_only theta cadence.")
    if len(baseline_slots) != int(config.micro_steps):
        raise ValueError("baseline_slots must align with config.micro_steps.")
    if len(live_mask) != int(config.micro_steps):
        raise ValueError("live_mask must align with config.micro_steps.")

    context = fmpc_tf2_module.build_tf1_context(model, x_batch, y_batch)
    active_mix_ratio = fmpc_tf2_module._active_onpolicy_mix_ratio(config, None)
    z_on = context.z0.copy()
    z_lf = context.z0.copy()
    total_losses: list[float] = []
    boot_losses: list[float] = []
    identity_losses: list[float] = []
    step_drift_metrics: list[dict[str, float]] = []

    for step_index, baseline_slot in enumerate(baseline_slots):
        use_live_step = bool(live_mask[step_index])
        if use_live_step:
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
        else:
            live_plan = None
            psi_inputs = _candidate_replay_inputs(config, baseline_slot)
            boot_targets = baseline_slot.boot_targets
            identity_targets = baseline_slot.identity_targets
            z_on_next = baseline_slot.z_on_next.copy()
            z_lf_next = baseline_slot.z_lf_next.copy()

        step_drift_metrics.append(_step_drift_from_plans(config, baseline_slot, live_plan))

        psi_predictions = fmpc_tf2_module._psi_predict(psi_network, psi_inputs, config)
        boot_loss = float(np.mean((psi_predictions - boot_targets) ** 2))
        identity_loss = float(np.mean((psi_predictions - identity_targets) ** 2))
        lambda_id = float(baseline_slot.lambda_id)
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
        "stage": "mirrored_handoff_asymmetry_diagnostic",
        "diagnostic_only": True,
        "preset_name": cell.config.preset_name,
        "candidate_key": cell.candidate.key,
        "handoff_mode": cell.spec.mode,
        "live_mask": [bool(flag) for flag in cell.spec.live_mask],
        "live_step_count": int(cell.spec.live_step_count),
        "live_control_family": cell.spec.live_control_family,
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
    live_mask = artifact.summary.get("live_mask", [])
    return {
        "handoff_mode": artifact.mode,
        "candidate_key": artifact.candidate.key,
        "psi_family": artifact.candidate.psi_family,
        "time_encoding_variant": artifact.candidate.time_encoding_variant,
        "live_step_count": int(artifact.summary["live_step_count"]),
        "live_control_family": artifact.summary["live_control_family"],
        "live_mask": ",".join("1" if bool(flag) else "0" for flag in live_mask),
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


def _aggregate_mode(run_rows: list[dict[str, Any]], *, candidate_key: str, mode: MirroredMode) -> dict[str, Any]:
    return _aggregate_run_group(run_rows, mode=mode, candidate_key=candidate_key)


def _pairwise_vs_matched_mirror(
    run_rows: list[dict[str, Any]],
    *,
    mode: MirroredMode,
) -> dict[str, Any] | None:
    mirror_mode = _mirror_mode(mode)
    if mirror_mode is None:
        return None
    return _pairwise_vs_reference(
        run_rows,
        candidate_key="residualized_local_field_poly_rt2",
        mode=mode,
        reference_candidate_key="residualized_local_field_poly_rt2",
        reference_mode=mirror_mode,
    )


def _run_one_seed(
    base_run_dir: Path,
    suite_config: FMPCTF2MirroredHandoffAsymmetrySuiteConfig,
    *,
    candidates: dict[str, FMPCTF2PartialHandoffCandidate],
    seed: int,
) -> tuple[list[_RunArtifacts], list[dict[str, Any]]]:
    cells: dict[tuple[str, MirroredMode], _CellRuntime] = {}
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

    for epoch_index in range(int(baseline_control.config.epochs)):
        lambda_id = fmpc_tf2_module._lambda_id_for_epoch(baseline_control.config, epoch_index)
        stage = fmpc_tf2_module._stage_for_epoch(baseline_control.config, epoch_index)
        epoch_metric_lists: dict[tuple[str, MirroredMode], dict[str, list[float]]] = {
            cell_key: {
                "train_loss": [],
                "train_boot_loss": [],
                "train_identity_loss": [],
                "train_transported_final_energy": [],
            }
            for cell_key in cells
        }
        epoch_step_drift_lists: dict[MirroredMode, dict[int, dict[str, list[float]]]] = {
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
                train_loss, boot_loss, identity_loss, transported_energy, step_metrics = _train_one_batch_mirrored_handoff(
                    cell.model,
                    cell.psi_network,
                    cell.config,
                    x_batch,
                    y_batch,
                    baseline_slots=baseline_slots,
                    live_mask=spec.live_mask,
                )
                cell.train_wall_time_seconds += float(perf_counter() - batch_start)
                metric_lists = epoch_metric_lists[(spec.candidate_key, spec.mode)]
                metric_lists["train_loss"].append(train_loss)
                metric_lists["train_boot_loss"].append(boot_loss)
                metric_lists["train_identity_loss"].append(identity_loss)
                metric_lists["train_transported_final_energy"].append(transported_energy)
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
                        "live_step_count": int(spec.live_step_count),
                        "live_control_family": spec.live_control_family,
                        "step_index": int(step_index),
                        **aggregated,
                    }
                )

    artifacts: list[_RunArtifacts] = []
    for spec in _cell_specs():
        cell = cells[(spec.candidate_key, spec.mode)]
        artifacts.append(_finalize_cell(base_run_dir, cell, split, seed=seed))
    return artifacts, step_drift_rows


def _mean_test_accuracy(summary: dict[str, Any]) -> float:
    value = summary["mean_test_accuracy"]
    if value is None:
        raise ValueError("mean_test_accuracy missing from aggregated summary.")
    return float(value)


def run_fmpc_tf2_mirrored_handoff_asymmetry_suite(
    config: FMPCTF2MirroredHandoffAsymmetrySuiteConfig,
) -> FMPCTF2MirroredHandoffAsymmetrySuiteRunResult:
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
    _write_csv(run_dir / "stepwise_asymmetry_drift_epoch_metrics.csv", step_drift_rows)

    baseline_live = _aggregate_mode(run_rows, candidate_key="baseline_plain_raw", mode="closed_loop_live_plan")
    baseline_open = _aggregate_mode(
        run_rows,
        candidate_key="baseline_plain_raw",
        mode="open_loop_baseline_plan_replay",
    )
    challenger_modes: list[MirroredMode] = [
        "k0_closed_loop_live_plan",
        "k4_open_loop_baseline_plan_replay",
        "suffix_live_k1",
        "suffix_live_k2",
        "suffix_live_k3",
        "prefix_live_k1",
        "prefix_live_k2",
        "prefix_live_k3",
    ]
    challenger_summary = {
        mode: _aggregate_mode(
            run_rows,
            candidate_key="residualized_local_field_poly_rt2",
            mode=mode,
        )
        for mode in challenger_modes
    }

    stepwise_asymmetry_summary = {
        mode: {
            f"step_{step_index}": _aggregate_step_drift(
                step_drift_rows,
                candidate_key="residualized_local_field_poly_rt2",
                mode=mode,
                step_index=step_index,
            )
            for step_index in range(4)
        }
        for mode in challenger_modes
    }

    matched_live_step_asymmetry_summary: dict[str, Any] = {}
    for live_step_count in (1, 2, 3):
        prefix_mode = f"prefix_live_k{live_step_count}"
        suffix_mode = f"suffix_live_k{live_step_count}"
        prefix_summary = challenger_summary[prefix_mode]
        suffix_summary = challenger_summary[suffix_mode]
        matched_live_step_asymmetry_summary[f"k{live_step_count}"] = {
            "prefix_mode": prefix_mode,
            "suffix_mode": suffix_mode,
            "prefix_summary": prefix_summary,
            "suffix_summary": suffix_summary,
            "mean_test_accuracy_gap_prefix_minus_suffix": (
                _mean_test_accuracy(prefix_summary) - _mean_test_accuracy(suffix_summary)
            ),
            "mean_val_accuracy_gap_prefix_minus_suffix": (
                float(prefix_summary["mean_val_accuracy"]) - float(suffix_summary["mean_val_accuracy"])
            ),
            "pairwise_prefix_vs_suffix": _pairwise_vs_reference(
                run_rows,
                candidate_key="residualized_local_field_poly_rt2",
                mode=prefix_mode,
                reference_candidate_key="residualized_local_field_poly_rt2",
                reference_mode=suffix_mode,
            ),
            "pairwise_suffix_vs_prefix": _pairwise_vs_reference(
                run_rows,
                candidate_key="residualized_local_field_poly_rt2",
                mode=suffix_mode,
                reference_candidate_key="residualized_local_field_poly_rt2",
                reference_mode=prefix_mode,
            ),
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
    }
    pairwise_vs_baseline_live.update(
        {
            f"residualized_local_field_poly_rt2__{mode}": _pairwise_vs_reference(
                run_rows,
                candidate_key="residualized_local_field_poly_rt2",
                mode=mode,
                reference_candidate_key="baseline_plain_raw",
                reference_mode="closed_loop_live_plan",
            )
            for mode in challenger_modes
        }
    )

    pairwise_vs_challenger_k0 = {
        mode: _pairwise_vs_reference(
            run_rows,
            candidate_key="residualized_local_field_poly_rt2",
            mode=mode,
            reference_candidate_key="residualized_local_field_poly_rt2",
            reference_mode="k0_closed_loop_live_plan",
        )
        for mode in challenger_modes
    }
    pairwise_vs_challenger_k4 = {
        mode: _pairwise_vs_reference(
            run_rows,
            candidate_key="residualized_local_field_poly_rt2",
            mode=mode,
            reference_candidate_key="residualized_local_field_poly_rt2",
            reference_mode="k4_open_loop_baseline_plan_replay",
        )
        for mode in challenger_modes
    }
    pairwise_vs_matched_mirror = {
        mode: _pairwise_vs_matched_mirror(run_rows, mode=mode)
        for mode in challenger_modes
        if _mirror_mode(mode) is not None
    }

    asymmetry_deltas = {
        live_step_count: float(
            matched_live_step_asymmetry_summary[f"k{live_step_count}"]["mean_test_accuracy_gap_prefix_minus_suffix"]
        )
        for live_step_count in (1, 2, 3)
    }
    tolerance = float(config.baseline_similarity_tolerance)
    material = float(config.material_test_gain)
    prefix_worse = all(delta <= -material for delta in asymmetry_deltas.values())
    suffix_worse = all(delta >= material for delta in asymmetry_deltas.values())
    approximately_symmetric = all(abs(delta) <= tolerance for delta in asymmetry_deltas.values())
    pair_averages = {
        live_step_count: _mean(
            [
                _mean_test_accuracy(challenger_summary[f"prefix_live_k{live_step_count}"]),
                _mean_test_accuracy(challenger_summary[f"suffix_live_k{live_step_count}"]),
            ]
        )
        for live_step_count in (1, 2, 3)
    }
    k0_test = _mean_test_accuracy(challenger_summary["k0_closed_loop_live_plan"])
    k4_test = _mean_test_accuracy(challenger_summary["k4_open_loop_baseline_plan_replay"])
    cumulative_monotone = k4_test >= pair_averages[1] >= pair_averages[2] >= pair_averages[3] >= k0_test
    any_partial_near_k4 = any(abs(pair_averages[k] - k4_test) <= tolerance for k in (1, 2, 3))

    if prefix_worse:
        dominant_interpretation = "early_self_induced_control_is_more_damaging"
        next_single_narrow_move = (
            "run one narrow first-live-step decomposition diagnostic that separates step-0-only live control "
            "from step-1/2 follow-on propagation without reopening the broader TF2 mechanism search"
        )
    elif suffix_worse:
        dominant_interpretation = "late_self_induced_control_is_more_damaging"
        next_single_narrow_move = (
            "run one narrow terminal-live-step decomposition diagnostic that separates the final live handoff "
            "from the earlier replayed scaffold while keeping the same fixed 4-step rollout"
        )
    elif approximately_symmetric and cumulative_monotone:
        dominant_interpretation = "failure_is_approximately_symmetric_and_cumulative_across_the_full_plan"
        next_single_narrow_move = (
            "run one narrow step-to-step scaffolding diagnostic that keeps the challenger live but injects "
            "only the baseline intermediate state bridge between adjacent micro-steps"
        )
    elif not any_partial_near_k4 and abs(k4_test - k0_test) >= material:
        dominant_interpretation = "only_full_replay_rescues_so_the_mechanism_is_global"
        next_single_narrow_move = (
            "run one narrow step-to-step scaffolding diagnostic to test whether global rescue requires "
            "continuous intermediate guidance rather than any isolated prefix or suffix repair"
        )
    else:
        dominant_interpretation = "asymmetry_is_mixed_but_still_points_to_global_closed_loop_coupling"
        next_single_narrow_move = (
            "run one narrow step-bridge diagnostic that freezes only the inter-step state handoff while "
            "leaving the challenger responsible for its own local target predictions"
        )

    summary = {
        "phase": "Phase Incremental Bridge",
        "stage": "mirrored_handoff_asymmetry_diagnostic",
        "diagnostic_only": True,
        "end_to_end_summary": {
            "baseline_plain_raw": {
                "closed_loop_live_plan": baseline_live,
                "open_loop_baseline_plan_replay": baseline_open,
            },
            "residualized_local_field_poly_rt2": challenger_summary,
        },
        "stepwise_asymmetry_summary": stepwise_asymmetry_summary,
        "matched_live_step_asymmetry_summary": matched_live_step_asymmetry_summary,
        "pairwise_delta_vs_baseline_closed_loop_run": pairwise_vs_baseline_live,
        "pairwise_delta_vs_challenger_k0_closed_loop_run": pairwise_vs_challenger_k0,
        "pairwise_delta_vs_challenger_k4_open_loop_replay_run": pairwise_vs_challenger_k4,
        "pairwise_delta_vs_matched_mirrored_mode": pairwise_vs_matched_mirror,
        "is_early_live_control_more_damaging": bool(prefix_worse),
        "is_late_live_control_more_damaging": bool(suffix_worse),
        "is_failure_approximately_symmetric_and_cumulative": bool(
            approximately_symmetric and cumulative_monotone
        ),
        "dominant_interpretation": dominant_interpretation,
        "next_single_narrow_move": next_single_narrow_move,
        "artifacts": {
            "end_to_end_runs_csv": "end_to_end_runs.csv",
            "stepwise_asymmetry_drift_epoch_metrics_csv": "stepwise_asymmetry_drift_epoch_metrics.csv",
        },
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF2MirroredHandoffAsymmetrySuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config, candidates),
        run_rows=run_rows,
        step_drift_rows=step_drift_rows,
        summary=summary,
    )
