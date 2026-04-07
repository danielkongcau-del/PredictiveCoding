from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np

from . import fmpc_tf2 as fmpc_tf2_module
from .datasets import load_digits_split
from .fmpc_tf1 import _select_tf1_checkpoint_epoch, build_tf1_epoch_selection_diagnostics
from .fmpc_tf2 import FMPCTF2Config, FMPCTF2EpochMetrics
from .fmpc_tf2_partial_open_loop_handoff_suite import (
    FMPCTF2PartialHandoffCandidate,
    _ReplayPlanSlot,
    _build_cached_plan,
    _build_candidate_config,
    _candidate_registry,
    _relative_posix,
)
from .fmpc_tf2_terminal_localfield_trust_region_suite import (
    _prepare_run_dir,
    _resolve_run_dir,
    _train_one_batch_terminal_localfield_trust_region,
    _write_csv,
    _write_json,
)
from .metrics import majority_class_baseline_accuracy

TerminalLocalFieldAdoptionMode = Literal[
    "baseline_closed_loop_live_plan",
    "terminal_live_raw_closed_loop",
    "terminal_local_field_direction_hard_replace_keep_live_norm",
    "terminal_local_field_direction_strong_mix_keep_live_norm_w075",
    "terminal_local_field_direction_strong_mix_keep_live_norm_w090",
    "terminal_local_field_direction_angle_clip_keep_live_norm_deg30",
    "terminal_local_field_direction_angle_clip_keep_live_norm_deg20",
]
LocalFieldDirectionInterventionMode = Literal[
    "closed_loop_live_plan",
    "live_raw_closed_loop",
    "local_field_direction_hard_replace_keep_live_norm",
    "local_field_direction_strong_mix_keep_live_norm",
    "local_field_direction_angle_clip_keep_live_norm",
]


@dataclass(frozen=True)
class _CellSpec:
    candidate_key: str
    stabilizer_mode: TerminalLocalFieldAdoptionMode
    terminal_direction_intervention: LocalFieldDirectionInterventionMode
    strong_mix_weight_toward_local_field: float | None = None
    angle_clip_degrees: float | None = None


@dataclass(frozen=True)
class _TrainSpec:
    direction_anchor_mode: LocalFieldDirectionInterventionMode


@dataclass
class FMPCTF2TerminalLocalFieldAdoptionSuiteConfig:
    """Adoption-oriented terminal local-field stabilizer selection suite."""

    experiment_name: str = "fmpc_tf2_terminal_localfield_adoption_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: Literal["single_dir", "run_id_subdir"] = "single_dir"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 60
    batch_size: int = 128
    eval_steps: int = 15
    layer_dims: tuple[int, ...] = (64, 64, 10)
    adoption_test_gain: float = 0.005
    required_gate_seed_majority_rate: float = 0.5
    selected_epoch_passes_gate_rate_floor: float = 0.4
    modest_runtime_overhead_seconds: float = 2.0

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"


@dataclass
class FMPCTF2TerminalLocalFieldAdoptionSuiteRunResult:
    run_dir: Path
    config: dict[str, Any]
    run_rows: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass
class _RunArtifacts:
    stabilizer_mode: TerminalLocalFieldAdoptionMode
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


def _as_rate(values: list[bool]) -> float:
    if not values:
        raise ValueError("Rate requires at least one value.")
    return float(sum(1.0 for value in values if bool(value)) / float(len(values)))


def _baseline_specs() -> list[_CellSpec]:
    return [
        _CellSpec(
            candidate_key="baseline_plain_raw",
            stabilizer_mode="baseline_closed_loop_live_plan",
            terminal_direction_intervention="closed_loop_live_plan",
        )
    ]


def _challenger_specs() -> list[_CellSpec]:
    return [
        _CellSpec(
            candidate_key="residualized_local_field_poly_rt2",
            stabilizer_mode="terminal_live_raw_closed_loop",
            terminal_direction_intervention="live_raw_closed_loop",
        ),
        _CellSpec(
            candidate_key="residualized_local_field_poly_rt2",
            stabilizer_mode="terminal_local_field_direction_hard_replace_keep_live_norm",
            terminal_direction_intervention="local_field_direction_hard_replace_keep_live_norm",
        ),
        _CellSpec(
            candidate_key="residualized_local_field_poly_rt2",
            stabilizer_mode="terminal_local_field_direction_strong_mix_keep_live_norm_w075",
            terminal_direction_intervention="local_field_direction_strong_mix_keep_live_norm",
            strong_mix_weight_toward_local_field=0.75,
        ),
        _CellSpec(
            candidate_key="residualized_local_field_poly_rt2",
            stabilizer_mode="terminal_local_field_direction_strong_mix_keep_live_norm_w090",
            terminal_direction_intervention="local_field_direction_strong_mix_keep_live_norm",
            strong_mix_weight_toward_local_field=0.90,
        ),
        _CellSpec(
            candidate_key="residualized_local_field_poly_rt2",
            stabilizer_mode="terminal_local_field_direction_angle_clip_keep_live_norm_deg30",
            terminal_direction_intervention="local_field_direction_angle_clip_keep_live_norm",
            angle_clip_degrees=30.0,
        ),
        _CellSpec(
            candidate_key="residualized_local_field_poly_rt2",
            stabilizer_mode="terminal_local_field_direction_angle_clip_keep_live_norm_deg20",
            terminal_direction_intervention="local_field_direction_angle_clip_keep_live_norm",
            angle_clip_degrees=20.0,
        ),
    ]


def _cell_specs() -> list[_CellSpec]:
    return [*_baseline_specs(), *_challenger_specs()]


def _suite_config_payload(
    config: FMPCTF2TerminalLocalFieldAdoptionSuiteConfig,
    candidates: dict[str, FMPCTF2PartialHandoffCandidate],
) -> dict[str, Any]:
    return {
        "phase": "Phase TF2",
        "stage": "terminal_localfield_adoption_selection_suite",
        "adoption_oriented": True,
        "changes_mainline_default": False,
        "seeds": [int(seed) for seed in config.seeds],
        "baseline_reference": "baseline_plain_raw__baseline_closed_loop_live_plan",
        "candidates": [
            {
                "key": candidate.key,
                "psi_family": candidate.psi_family,
                "time_encoding_variant": candidate.time_encoding_variant,
                "notes": candidate.notes,
            }
            for candidate in candidates.values()
            if candidate.key in {"baseline_plain_raw", "residualized_local_field_poly_rt2"}
        ],
        "challenger_modes": [spec.stabilizer_mode for spec in _challenger_specs()],
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
        "adoption_thresholds": {
            "mean_test_accuracy_gain_vs_terminal_live_raw_closed_loop": float(config.adoption_test_gain),
            "required_gate_seed_majority_rate": float(config.required_gate_seed_majority_rate),
            "selected_epoch_passes_gate_rate_floor": float(config.selected_epoch_passes_gate_rate_floor),
            "modest_runtime_overhead_seconds": float(config.modest_runtime_overhead_seconds),
        },
        "intervention_strength_order_assumption": [
            "terminal_local_field_direction_angle_clip_keep_live_norm_deg30",
            "terminal_local_field_direction_angle_clip_keep_live_norm_deg20",
            "terminal_local_field_direction_strong_mix_keep_live_norm_w075",
            "terminal_local_field_direction_strong_mix_keep_live_norm_w090",
            "terminal_local_field_direction_hard_replace_keep_live_norm",
        ],
        "terminal_definition": (
            "The challenger remains in the true harmful closed-loop regime through all micro-steps. "
            "Only the final-step action direction may be stabilized toward the teacher-free local-field "
            "direction, while the live supervision bundle remains unchanged."
        ),
    }


def _make_cell(
    base_run_dir: Path,
    candidate: FMPCTF2PartialHandoffCandidate,
    spec: _CellSpec,
    *,
    seed: int,
    suite_config: FMPCTF2TerminalLocalFieldAdoptionSuiteConfig,
) -> _CellRuntime:
    config = _build_candidate_config(candidate, seed=seed, suite_config=suite_config)
    run_dir = base_run_dir / "runs" / spec.stabilizer_mode / candidate.key / f"seed_{int(seed)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return _CellRuntime(
        spec=spec,
        candidate=candidate,
        config=config,
        run_dir=run_dir,
        model=fmpc_tf2_module._make_pc_model(config),
        psi_network=fmpc_tf2_module._make_psi_network(config),
    )


def _train_one_batch_terminal_localfield_adoption(
    model: Any,
    psi_network: Any,
    config: FMPCTF2Config,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    *,
    baseline_slots: list[_ReplayPlanSlot],
    spec: _CellSpec,
) -> tuple[float, float, float, float]:
    train_spec = _TrainSpec(direction_anchor_mode=spec.terminal_direction_intervention)
    train_loss, boot_loss, identity_loss, transported_energy, _ = (
        _train_one_batch_terminal_localfield_trust_region(
            model,
            psi_network,
            config,
            x_batch,
            y_batch,
            baseline_slots=baseline_slots,
            spec=train_spec,
            strong_mix_weight_toward_local_field=(
                0.75
                if spec.strong_mix_weight_toward_local_field is None
                else float(spec.strong_mix_weight_toward_local_field)
            ),
            angle_clip_degrees=30.0 if spec.angle_clip_degrees is None else float(spec.angle_clip_degrees),
        )
    )
    return train_loss, boot_loss, identity_loss, transported_energy


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
    config_payload["stabilizer_mode"] = spec.stabilizer_mode
    config_payload["terminal_direction_intervention"] = spec.terminal_direction_intervention
    config_payload["diagnostic_only"] = True
    config_payload["adoption_oriented"] = True
    config_payload["strong_mix_weight_toward_local_field"] = (
        None
        if spec.strong_mix_weight_toward_local_field is None
        else float(spec.strong_mix_weight_toward_local_field)
    )
    config_payload["angle_clip_degrees"] = (
        None if spec.angle_clip_degrees is None else float(spec.angle_clip_degrees)
    )
    fmpc_tf2_module._write_json(run_dir / "config.json", config_payload)
    fmpc_tf2_module._write_epoch_metrics(run_dir / "epoch_metrics.csv", epoch_rows)
    fmpc_tf2_module._write_json(run_dir / "selection_diagnostics.json", selection_diagnostics)
    fmpc_tf2_module._write_json(run_dir / "summary.json", summary)


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
        "phase": "Phase TF2",
        "stage": "terminal_localfield_adoption_selection",
        "diagnostic_only": True,
        "adoption_oriented": True,
        "preset_name": cell.config.preset_name,
        "candidate_key": cell.candidate.key,
        "stabilizer_mode": cell.spec.stabilizer_mode,
        "terminal_direction_intervention": cell.spec.terminal_direction_intervention,
        "strong_mix_weight_toward_local_field": (
            None
            if cell.spec.strong_mix_weight_toward_local_field is None
            else float(cell.spec.strong_mix_weight_toward_local_field)
        ),
        "angle_clip_degrees": None if cell.spec.angle_clip_degrees is None else float(cell.spec.angle_clip_degrees),
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
        stabilizer_mode=cell.spec.stabilizer_mode,
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
        "stabilizer_mode": artifact.stabilizer_mode,
        "candidate_key": artifact.candidate.key,
        "psi_family": artifact.candidate.psi_family,
        "time_encoding_variant": artifact.candidate.time_encoding_variant,
        "terminal_direction_intervention": artifact.summary["terminal_direction_intervention"],
        "strong_mix_weight_toward_local_field": artifact.summary["strong_mix_weight_toward_local_field"],
        "angle_clip_degrees": artifact.summary["angle_clip_degrees"],
        "seed": int(artifact.seed),
        "val_accuracy": float(artifact.summary["val_accuracy"]),
        "test_accuracy": float(artifact.summary["test_accuracy"]),
        "gate_passing_epoch_count": int(artifact.summary["gate_passing_epoch_count"]),
        "selected_epoch_passes_gate": bool(artifact.summary["selected_epoch_passes_gate"]),
        "selector_fallback_used": bool(artifact.summary["selector_fallback_used"]),
        "val_transported_final_energy": float(artifact.summary["val_transported_final_energy"]),
        "selected_epoch": int(artifact.summary["selected_epoch"]),
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
    stabilizer_mode: TerminalLocalFieldAdoptionMode,
    candidate_key: str,
) -> dict[str, Any]:
    relevant = [
        row
        for row in run_rows
        if row["stabilizer_mode"] == stabilizer_mode
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
            "selected_epoch_passes_gate_rate": None,
            "selector_fallback_used_rate": None,
            "seed_gate_positive_rate": None,
            "mean_val_transported_final_energy": None,
            "mean_total_wall_time_seconds": None,
            "std_total_wall_time_seconds": None,
        }
    val_values = [float(row["val_accuracy"]) for row in relevant]
    test_values = [float(row["test_accuracy"]) for row in relevant]
    gate_values = [float(row["gate_passing_epoch_count"]) for row in relevant]
    energy_values = [float(row["val_transported_final_energy"]) for row in relevant]
    wall_values = [float(row["total_wall_time_seconds"]) for row in relevant]
    selected_epoch_passes_gate = [bool(row["selected_epoch_passes_gate"]) for row in relevant]
    selector_fallback_used = [bool(row["selector_fallback_used"]) for row in relevant]
    seed_gate_positive = [float(row["gate_passing_epoch_count"]) > 0.0 for row in relevant]
    return {
        "mean_val_accuracy": _mean(val_values),
        "std_val_accuracy": _std(val_values),
        "mean_test_accuracy": _mean(test_values),
        "std_test_accuracy": _std(test_values),
        "mean_gate_passing_epoch_count": _mean(gate_values),
        "selected_epoch_passes_gate_rate": _as_rate(selected_epoch_passes_gate),
        "selector_fallback_used_rate": _as_rate(selector_fallback_used),
        "seed_gate_positive_rate": _as_rate(seed_gate_positive),
        "mean_val_transported_final_energy": _mean(energy_values),
        "mean_total_wall_time_seconds": _mean(wall_values),
        "std_total_wall_time_seconds": _std(wall_values),
    }


def _pairwise_vs_reference(
    run_rows: list[dict[str, Any]],
    *,
    candidate_key: str,
    stabilizer_mode: TerminalLocalFieldAdoptionMode,
    reference_candidate_key: str,
    reference_stabilizer_mode: TerminalLocalFieldAdoptionMode,
) -> dict[str, Any]:
    candidate_by_seed = {
        int(row["seed"]): row
        for row in run_rows
        if row["candidate_key"] == candidate_key
        and row["stabilizer_mode"] == stabilizer_mode
        and row["run_status"] == "success"
    }
    reference_by_seed = {
        int(row["seed"]): row
        for row in run_rows
        if row["candidate_key"] == reference_candidate_key
        and row["stabilizer_mode"] == reference_stabilizer_mode
        and row["run_status"] == "success"
    }
    shared_seeds = sorted(set(candidate_by_seed).intersection(reference_by_seed))
    if not shared_seeds:
        return {
            "mean_val_accuracy_delta": None,
            "mean_test_accuracy_delta": None,
            "mean_gate_passing_epoch_count_delta": None,
            "selected_epoch_passes_gate_rate_delta": None,
            "selector_fallback_used_rate_delta": None,
            "mean_runtime_delta": None,
        }
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
        "mean_gate_passing_epoch_count_delta": _mean(
            [
                float(candidate_by_seed[seed]["gate_passing_epoch_count"])
                - float(reference_by_seed[seed]["gate_passing_epoch_count"])
                for seed in shared_seeds
            ]
        ),
        "selected_epoch_passes_gate_rate_delta": _mean(
            [
                float(bool(candidate_by_seed[seed]["selected_epoch_passes_gate"]))
                - float(bool(reference_by_seed[seed]["selected_epoch_passes_gate"]))
                for seed in shared_seeds
            ]
        ),
        "selector_fallback_used_rate_delta": _mean(
            [
                float(bool(candidate_by_seed[seed]["selector_fallback_used"]))
                - float(bool(reference_by_seed[seed]["selector_fallback_used"]))
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


def _run_one_seed(
    base_run_dir: Path,
    suite_config: FMPCTF2TerminalLocalFieldAdoptionSuiteConfig,
    *,
    candidates: dict[str, FMPCTF2PartialHandoffCandidate],
    seed: int,
) -> list[_RunArtifacts]:
    cells: dict[tuple[str, TerminalLocalFieldAdoptionMode], _CellRuntime] = {}
    for spec in _cell_specs():
        cells[(spec.candidate_key, spec.stabilizer_mode)] = _make_cell(
            base_run_dir,
            candidates[spec.candidate_key],
            spec,
            seed=seed,
            suite_config=suite_config,
        )

    baseline_control = cells[("baseline_plain_raw", "baseline_closed_loop_live_plan")]
    split = load_digits_split(
        split_seed=baseline_control.config.data_seed,
        train_fraction=baseline_control.config.train_fraction,
        val_fraction=baseline_control.config.val_fraction,
        test_fraction=baseline_control.config.test_fraction,
    )

    for epoch_index in range(int(baseline_control.config.epochs)):
        lambda_id = fmpc_tf2_module._lambda_id_for_epoch(baseline_control.config, epoch_index)
        stage = fmpc_tf2_module._stage_for_epoch(baseline_control.config, epoch_index)
        epoch_metric_lists: dict[tuple[str, TerminalLocalFieldAdoptionMode], dict[str, list[float]]] = {
            cell_key: {
                "train_loss": [],
                "train_boot_loss": [],
                "train_identity_loss": [],
                "train_transported_final_energy": [],
            }
            for cell_key in cells
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
                cell = cells[(spec.candidate_key, spec.stabilizer_mode)]
                batch_start = perf_counter()
                train_loss, boot_loss, identity_loss, transported_energy = (
                    _train_one_batch_terminal_localfield_adoption(
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
                metric_lists = epoch_metric_lists[(spec.candidate_key, spec.stabilizer_mode)]
                metric_lists["train_loss"].append(train_loss)
                metric_lists["train_boot_loss"].append(boot_loss)
                metric_lists["train_identity_loss"].append(identity_loss)
                metric_lists["train_transported_final_energy"].append(transported_energy)

        for spec in _cell_specs():
            cell = cells[(spec.candidate_key, spec.stabilizer_mode)]
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
            metric_lists = epoch_metric_lists[(spec.candidate_key, spec.stabilizer_mode)]
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

    artifacts: list[_RunArtifacts] = []
    for spec in _cell_specs():
        cell = cells[(spec.candidate_key, spec.stabilizer_mode)]
        artifacts.append(_finalize_cell(base_run_dir, cell, split, seed=seed))
    return artifacts


def run_fmpc_tf2_terminal_localfield_adoption_suite(
    config: FMPCTF2TerminalLocalFieldAdoptionSuiteConfig,
) -> FMPCTF2TerminalLocalFieldAdoptionSuiteRunResult:
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
    for seed in config.seeds:
        artifacts = _run_one_seed(
            run_dir,
            config,
            candidates=candidates,
            seed=int(seed),
        )
        run_rows.extend([_success_run_row(artifact=artifact, base_run_dir=run_dir) for artifact in artifacts])

    _write_csv(run_dir / "end_to_end_runs.csv", run_rows)

    baseline_summary = _aggregate_run_group(
        run_rows,
        stabilizer_mode="baseline_closed_loop_live_plan",
        candidate_key="baseline_plain_raw",
    )
    challenger_modes: list[TerminalLocalFieldAdoptionMode] = [
        "terminal_live_raw_closed_loop",
        "terminal_local_field_direction_hard_replace_keep_live_norm",
        "terminal_local_field_direction_strong_mix_keep_live_norm_w075",
        "terminal_local_field_direction_strong_mix_keep_live_norm_w090",
        "terminal_local_field_direction_angle_clip_keep_live_norm_deg30",
        "terminal_local_field_direction_angle_clip_keep_live_norm_deg20",
    ]
    challenger_summary = {
        mode: _aggregate_run_group(
            run_rows,
            stabilizer_mode=mode,
            candidate_key="residualized_local_field_poly_rt2",
        )
        for mode in challenger_modes
    }
    pairwise_vs_baseline_closed_loop = {
        "baseline_plain_raw__baseline_closed_loop_live_plan": _pairwise_vs_reference(
            run_rows,
            candidate_key="baseline_plain_raw",
            stabilizer_mode="baseline_closed_loop_live_plan",
            reference_candidate_key="baseline_plain_raw",
            reference_stabilizer_mode="baseline_closed_loop_live_plan",
        ),
        **{
            f"residualized_local_field_poly_rt2__{mode}": _pairwise_vs_reference(
                run_rows,
                candidate_key="residualized_local_field_poly_rt2",
                stabilizer_mode=mode,
                reference_candidate_key="baseline_plain_raw",
                reference_stabilizer_mode="baseline_closed_loop_live_plan",
            )
            for mode in challenger_modes
        },
    }
    pairwise_vs_terminal_live_raw = {
        mode: _pairwise_vs_reference(
            run_rows,
            candidate_key="residualized_local_field_poly_rt2",
            stabilizer_mode=mode,
            reference_candidate_key="residualized_local_field_poly_rt2",
            reference_stabilizer_mode="terminal_live_raw_closed_loop",
        )
        for mode in challenger_modes
    }

    strength_order = [
        "terminal_local_field_direction_angle_clip_keep_live_norm_deg30",
        "terminal_local_field_direction_angle_clip_keep_live_norm_deg20",
        "terminal_local_field_direction_strong_mix_keep_live_norm_w075",
        "terminal_local_field_direction_strong_mix_keep_live_norm_w090",
        "terminal_local_field_direction_hard_replace_keep_live_norm",
    ]
    candidate_adoption_assessment: dict[str, Any] = {}
    qualifying_modes: list[str] = []
    for strength_rank, mode in enumerate(strength_order, start=1):
        mode_summary = challenger_summary[mode]
        pairwise_raw = pairwise_vs_terminal_live_raw[mode]
        mean_test_gain = float(pairwise_raw["mean_test_accuracy_delta"])
        mean_val_gain = float(pairwise_raw["mean_val_accuracy_delta"])
        gate_rate = float(mode_summary["seed_gate_positive_rate"])
        selected_gate_rate = float(mode_summary["selected_epoch_passes_gate_rate"])
        runtime_overhead = float(pairwise_raw["mean_runtime_delta"])
        meets_test_gain = mean_test_gain >= float(config.adoption_test_gain)
        has_majority_seed_gates = gate_rate > float(config.required_gate_seed_majority_rate)
        selected_gate_not_rare = selected_gate_rate >= float(config.selected_epoch_passes_gate_rate_floor)
        runtime_overhead_modest = runtime_overhead <= float(config.modest_runtime_overhead_seconds)
        mean_val_not_worse = mean_val_gain >= 0.0
        qualifies = bool(
            meets_test_gain
            and has_majority_seed_gates
            and selected_gate_not_rare
            and runtime_overhead_modest
            and mean_val_not_worse
        )
        if qualifies:
            qualifying_modes.append(mode)
        candidate_adoption_assessment[mode] = {
            "intervention_strength_rank": int(strength_rank),
            "mean_val_accuracy_gain_vs_terminal_live_raw_closed_loop": mean_val_gain,
            "mean_test_accuracy_gain_vs_terminal_live_raw_closed_loop": mean_test_gain,
            "mean_gate_passing_epoch_count": float(mode_summary["mean_gate_passing_epoch_count"]),
            "selected_epoch_passes_gate_rate": selected_gate_rate,
            "selector_fallback_used_rate": float(mode_summary["selector_fallback_used_rate"]),
            "seed_gate_positive_rate": gate_rate,
            "mean_runtime_overhead_seconds_vs_terminal_live_raw_closed_loop": runtime_overhead,
            "meets_mean_test_gain_threshold": bool(meets_test_gain),
            "has_majority_seed_gate_epochs": bool(has_majority_seed_gates),
            "selected_epoch_passes_gate_not_rare": bool(selected_gate_not_rare),
            "runtime_overhead_modest": bool(runtime_overhead_modest),
            "mean_val_accuracy_not_worse": bool(mean_val_not_worse),
            "qualifies_for_adoption": bool(qualifies),
        }

    recommended_mode = qualifying_modes[0] if qualifying_modes else None
    best_mean_test_candidate = max(
        strength_order,
        key=lambda mode: float(challenger_summary[mode]["mean_test_accuracy"]),
    )
    softer_viable_candidates = [mode for mode in qualifying_modes if mode != "terminal_local_field_direction_hard_replace_keep_live_norm"]
    hard_replace_necessary = bool(
        recommended_mode == "terminal_local_field_direction_hard_replace_keep_live_norm"
        and not softer_viable_candidates
    )

    if recommended_mode is None:
        dominant_interpretation = "no_terminal_local_field_stabilizer_is_yet_strong_enough_for_adoption"
        recommendation_reason = (
            "No candidate cleared the adoption threshold under the canonical selector in the true harmful closed-loop regime."
        )
        next_single_narrow_move = (
            "target one narrow late-step coupling follow-up rather than adopting a terminal local-field stabilizer into the mainline"
        )
    else:
        dominant_interpretation = "at_least_one_terminal_local_field_stabilizer_is_strong_enough_for_adoption"
        recommendation_reason = (
            "Select the weakest intervention that clears the adoption threshold while preserving nontrivial gate robustness."
        )
        next_single_narrow_move = (
            "run one narrow confirmation pass that compares the selected terminal local-field stabilizer against the current corrective default before broader TF2 changes"
        )

    if recommended_mode is not None and best_mean_test_candidate != recommended_mode:
        tradeoff_note = (
            f"{best_mean_test_candidate} has the strongest mean test accuracy, but {recommended_mode} is the "
            "weakest intervention that clears the adoption threshold."
        )
    elif recommended_mode is not None:
        tradeoff_note = "The best mean-test candidate and the weakest qualifying candidate coincide."
    else:
        tradeoff_note = "No candidate qualified, so there is no adoption tradeoff to resolve."

    summary = {
        "phase": "Phase TF2",
        "stage": "terminal_localfield_adoption_selection",
        "diagnostic_only": True,
        "adoption_oriented": True,
        "end_to_end_summary": {
            "baseline_plain_raw": {
                "baseline_closed_loop_live_plan": baseline_summary,
            },
            "residualized_local_field_poly_rt2": challenger_summary,
        },
        "pairwise_delta_vs_baseline_closed_loop_run": pairwise_vs_baseline_closed_loop,
        "pairwise_delta_vs_terminal_live_raw_closed_loop": pairwise_vs_terminal_live_raw,
        "adoption_thresholds": {
            "mean_test_accuracy_gain_vs_terminal_live_raw_closed_loop": float(config.adoption_test_gain),
            "required_gate_seed_majority_rate": float(config.required_gate_seed_majority_rate),
            "selected_epoch_passes_gate_rate_floor": float(config.selected_epoch_passes_gate_rate_floor),
            "modest_runtime_overhead_seconds": float(config.modest_runtime_overhead_seconds),
            "mean_val_accuracy_not_worse_than_terminal_live_raw_closed_loop": True,
        },
        "candidate_adoption_assessment": candidate_adoption_assessment,
        "best_mean_test_candidate": best_mean_test_candidate,
        "recommended_next_tf2_experimental_default": recommended_mode,
        "recommendation_reason": recommendation_reason,
        "is_any_terminal_local_field_stabilizer_strong_enough_for_adoption": bool(recommended_mode is not None),
        "is_hard_replace_necessary": bool(hard_replace_necessary),
        "tradeoff_note": tradeoff_note,
        "dominant_interpretation": dominant_interpretation,
        "next_single_narrow_move": next_single_narrow_move,
        "artifacts": {
            "end_to_end_runs_csv": "end_to_end_runs.csv",
        },
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF2TerminalLocalFieldAdoptionSuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config, candidates),
        run_rows=run_rows,
        summary=summary,
    )
