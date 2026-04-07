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
from .datasets import load_digits_split
from .fmpc_tf1 import _select_tf1_checkpoint_epoch, build_tf1_epoch_selection_diagnostics
from .fmpc_tf2 import FMPCTF2Config, FMPCTF2EpochMetrics, build_tf2_corrective_transport_default_config
from .metrics import majority_class_baseline_accuracy

BatchTargetStateMode = Literal["live_target_recompute", "batch_frozen_target_state_cache"]


@dataclass(frozen=True)
class FMPCTF2BatchFrozenCandidate:
    key: str
    psi_family: Literal["baseline_plain", "residualized_local_field"]
    time_encoding_variant: Literal["raw", "poly_rt2"]
    notes: str


@dataclass(frozen=True)
class _BatchFrozenMicroStepCache:
    step_index: int
    t_k: float
    r_k: float
    dt: float
    psi_inputs: np.ndarray
    boot_targets: np.ndarray
    identity_targets: np.ndarray
    z_on_next: np.ndarray
    z_lf_next: np.ndarray
    lambda_id: float
    identity_tangent_mode: str
    source_counts: dict[str, int]


@dataclass
class FMPCTF2BatchFrozenTargetStateSuiteConfig:
    """Diagnostic-only batch-frozen target/state coupling suite for the TF2 corrective default."""

    experiment_name: str = "fmpc_tf2_batch_frozen_target_state_suite"
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
    drift_reduction_fraction: float = 0.10

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"


@dataclass
class FMPCTF2BatchFrozenTargetStateSuiteRunResult:
    run_dir: Path
    config: dict[str, Any]
    run_rows: list[dict[str, Any]]
    drift_rows: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass
class _RunArtifacts:
    mode: BatchTargetStateMode
    candidate: FMPCTF2BatchFrozenCandidate
    seed: int
    config: FMPCTF2Config
    run_dir: Path
    epoch_rows: list[dict[str, Any]]
    drift_epoch_rows: list[dict[str, Any]]
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


def _candidate_registry() -> list[FMPCTF2BatchFrozenCandidate]:
    return [
        FMPCTF2BatchFrozenCandidate(
            key="baseline_plain_raw",
            psi_family="baseline_plain",
            time_encoding_variant="raw",
            notes="current corrective default baseline",
        ),
        FMPCTF2BatchFrozenCandidate(
            key="residualized_local_field_poly_rt2",
            psi_family="residualized_local_field",
            time_encoding_variant="poly_rt2",
            notes="offline-better challenger from the psi-expressivity suite",
        ),
    ]


def _suite_config_payload(
    config: FMPCTF2BatchFrozenTargetStateSuiteConfig,
    candidates: list[FMPCTF2BatchFrozenCandidate],
) -> dict[str, Any]:
    return {
        "phase": "Phase TF2",
        "stage": "batch_frozen_target_state_coupling_suite",
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
        "modes": ["live_target_recompute", "batch_frozen_target_state_cache"],
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
        "drift_reduction_fraction": float(config.drift_reduction_fraction),
        "cache_definition": (
            "batch_frozen_target_state_cache precomputes the current corrective-default TF2 micro-step "
            "plans at batch start and reuses their psi_inputs, boot_targets, identity_targets, and next states "
            "throughout that batch."
        ),
    }


def _build_candidate_config(
    candidate: FMPCTF2BatchFrozenCandidate,
    *,
    seed: int,
    suite_config: FMPCTF2BatchFrozenTargetStateSuiteConfig,
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


def _batch_drift_metrics(
    cached_inputs: np.ndarray,
    live_inputs: np.ndarray,
    cached_boot: np.ndarray,
    live_boot: np.ndarray,
    cached_identity: np.ndarray,
    live_identity: np.ndarray,
) -> dict[str, float]:
    return {
        "mean_relative_bootstrap_target_delta": _relative_target_delta(cached_boot, live_boot),
        "mean_relative_identity_target_delta": _relative_target_delta(cached_identity, live_identity),
        "mean_bootstrap_target_cosine_similarity": _mean_cosine_similarity(cached_boot, live_boot),
        "mean_identity_target_cosine_similarity": _mean_cosine_similarity(cached_identity, live_identity),
        "mean_relative_psi_input_delta": _relative_target_delta(cached_inputs, live_inputs),
    }


def _cached_batch_plans(
    context: Any,
    psi_network: Any,
    config: FMPCTF2Config,
    *,
    lambda_id: float,
) -> list[_BatchFrozenMicroStepCache]:
    knots = np.linspace(0.0, 1.0, int(config.micro_steps) + 1, dtype=np.float64)
    active_mix_ratio = fmpc_tf2_module._active_onpolicy_mix_ratio(config, None)
    z_on = context.z0.copy()
    z_lf = context.z0.copy()
    caches: list[_BatchFrozenMicroStepCache] = []
    for step_index in range(int(config.micro_steps)):
        t_k = float(knots[step_index])
        r_k = 1.0 - t_k
        dt = float(knots[step_index + 1] - knots[step_index])
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
        caches.append(
            _BatchFrozenMicroStepCache(
                step_index=int(step_index),
                t_k=t_k,
                r_k=r_k,
                dt=dt,
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
    return caches


def _train_one_batch_batch_frozen(
    model: Any,
    psi_network: Any,
    config: FMPCTF2Config,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    *,
    lambda_id: float,
    mode: BatchTargetStateMode,
) -> tuple[float, float, float, float, dict[str, float]]:
    if config.supervision_policy != "local_only":
        raise ValueError("Batch-frozen diagnostic is defined only for the corrective local_only regime.")
    if fmpc_tf2_module._resolved_theta_update_cadence(config) != "terminal_only":
        raise ValueError("Batch-frozen diagnostic expects terminal_only theta cadence.")

    context = fmpc_tf2_module.build_tf1_context(model, x_batch, y_batch)
    active_mix_ratio = fmpc_tf2_module._active_onpolicy_mix_ratio(config, None)
    cached_steps = _cached_batch_plans(context, psi_network, config, lambda_id=lambda_id)
    knots = np.linspace(0.0, 1.0, int(config.micro_steps) + 1, dtype=np.float64)
    z_on = context.z0.copy()
    z_lf = context.z0.copy()
    total_losses: list[float] = []
    boot_losses: list[float] = []
    identity_losses: list[float] = []
    drift_lists: dict[str, list[float]] = {
        "mean_relative_bootstrap_target_delta": [],
        "mean_relative_identity_target_delta": [],
        "mean_bootstrap_target_cosine_similarity": [],
        "mean_identity_target_cosine_similarity": [],
        "mean_relative_psi_input_delta": [],
    }

    for step_index in range(int(config.micro_steps)):
        t_k = float(knots[step_index])
        r_k = 1.0 - t_k
        dt = float(knots[step_index + 1] - knots[step_index])
        live_plan = fmpc_tf2_module._plan_tf2_micro_step(
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
        cached_plan = cached_steps[step_index]
        drift = _batch_drift_metrics(
            cached_plan.psi_inputs,
            live_plan.psi_inputs,
            cached_plan.boot_targets,
            live_plan.boot_targets,
            cached_plan.identity_targets,
            live_plan.identity_targets,
        )
        for metric_name, metric_value in drift.items():
            drift_lists[metric_name].append(float(metric_value))

        if mode == "live_target_recompute":
            selected_inputs = live_plan.psi_inputs
            selected_boot = live_plan.boot_targets
            selected_identity = live_plan.identity_targets
            z_on_next = live_plan.z_on_next.copy()
            z_lf_next = live_plan.z_lf_next.copy()
        else:
            selected_inputs = cached_plan.psi_inputs
            selected_boot = cached_plan.boot_targets
            selected_identity = cached_plan.identity_targets
            z_on_next = cached_plan.z_on_next.copy()
            z_lf_next = cached_plan.z_lf_next.copy()

        psi_predictions = fmpc_tf2_module._psi_predict(psi_network, selected_inputs, config)
        boot_loss = float(np.mean((psi_predictions - selected_boot) ** 2))
        identity_loss = float(np.mean((psi_predictions - selected_identity) ** 2))
        if lambda_id > 0.0:
            combined_target = (selected_boot + (lambda_id * selected_identity)) / (1.0 + lambda_id)
            loss_scale = 1.0 + lambda_id
        else:
            combined_target = selected_boot
            loss_scale = 1.0
        total_loss = boot_loss + (lambda_id * identity_loss)
        fmpc_tf2_module._weighted_mse_step(
            psi_network,
            selected_inputs,
            combined_target,
            loss_scale=loss_scale,
        )
        z_on = z_on_next
        z_lf = z_lf_next
        total_losses.append(total_loss)
        boot_losses.append(boot_loss)
        identity_losses.append(identity_loss)

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
        {metric_name: _mean(values) for metric_name, values in drift_lists.items()},
    )


def _write_run_artifacts(
    run_dir: Path,
    config: FMPCTF2Config,
    *,
    mode: BatchTargetStateMode,
    epoch_rows: list[dict[str, Any]],
    drift_epoch_rows: list[dict[str, Any]],
    selection_diagnostics: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    config_payload = fmpc_tf2_module._config_payload(config)
    config_payload["batch_target_state_mode"] = mode
    fmpc_tf2_module._write_json(run_dir / "config.json", config_payload)
    fmpc_tf2_module._write_epoch_metrics(run_dir / "epoch_metrics.csv", epoch_rows)
    _write_csv(run_dir / "within_batch_drift_epoch_metrics.csv", drift_epoch_rows)
    fmpc_tf2_module._write_json(run_dir / "selection_diagnostics.json", selection_diagnostics)
    fmpc_tf2_module._write_json(run_dir / "summary.json", summary)


def _run_one_candidate(
    base_run_dir: Path,
    suite_config: FMPCTF2BatchFrozenTargetStateSuiteConfig,
    *,
    candidate: FMPCTF2BatchFrozenCandidate,
    mode: BatchTargetStateMode,
    seed: int,
) -> _RunArtifacts:
    config = _build_candidate_config(candidate, seed=seed, suite_config=suite_config)
    run_dir = base_run_dir / "runs" / mode / candidate.key / f"seed_{int(seed)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    split = load_digits_split(
        split_seed=config.data_seed,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
    )
    model = fmpc_tf2_module._make_pc_model(config)
    psi_network = fmpc_tf2_module._make_psi_network(config)

    epoch_rows: list[dict[str, Any]] = []
    drift_epoch_rows: list[dict[str, Any]] = []
    epoch_snapshots: list[fmpc_tf2_module.FMPCTF2EpochSnapshot] = []
    train_start = perf_counter()
    for epoch_index in range(int(config.epochs)):
        lambda_id = fmpc_tf2_module._lambda_id_for_epoch(config, epoch_index)
        stage = fmpc_tf2_module._stage_for_epoch(config, epoch_index)
        batch_losses: list[float] = []
        batch_boot_losses: list[float] = []
        batch_identity_losses: list[float] = []
        batch_transport_energies: list[float] = []
        epoch_drift_lists: dict[str, list[float]] = {
            "mean_relative_bootstrap_target_delta": [],
            "mean_relative_identity_target_delta": [],
            "mean_bootstrap_target_cosine_similarity": [],
            "mean_identity_target_cosine_similarity": [],
            "mean_relative_psi_input_delta": [],
        }
        batch_seed = config.batch_order_seed + epoch_index
        for x_batch, y_batch in fmpc_tf2_module.iter_minibatches(
            split.x_train,
            split.y_train,
            config.batch_size,
            shuffle=config.shuffle_batches,
            seed=batch_seed,
        ):
            train_loss, boot_loss, identity_loss, transported_energy, batch_drift = _train_one_batch_batch_frozen(
                model,
                psi_network,
                config,
                x_batch,
                y_batch,
                lambda_id=float(lambda_id),
                mode=mode,
            )
            batch_losses.append(train_loss)
            batch_boot_losses.append(boot_loss)
            batch_identity_losses.append(identity_loss)
            batch_transport_energies.append(transported_energy)
            for metric_name, metric_value in batch_drift.items():
                epoch_drift_lists[metric_name].append(float(metric_value))

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
        drift_epoch_rows.append(
            {
                "mode": mode,
                "candidate_key": candidate.key,
                "psi_family": candidate.psi_family,
                "time_encoding_variant": candidate.time_encoding_variant,
                "seed": int(seed),
                "epoch": int(epoch_index + 1),
                "mean_relative_bootstrap_target_delta": _mean(
                    epoch_drift_lists["mean_relative_bootstrap_target_delta"]
                ),
                "mean_relative_identity_target_delta": _mean(
                    epoch_drift_lists["mean_relative_identity_target_delta"]
                ),
                "mean_bootstrap_target_cosine_similarity": _mean(
                    epoch_drift_lists["mean_bootstrap_target_cosine_similarity"]
                ),
                "mean_identity_target_cosine_similarity": _mean(
                    epoch_drift_lists["mean_identity_target_cosine_similarity"]
                ),
                "mean_relative_psi_input_delta": _mean(
                    epoch_drift_lists["mean_relative_psi_input_delta"]
                ),
            }
        )
        epoch_snapshots.append(
            fmpc_tf2_module.FMPCTF2EpochSnapshot(
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
    selected_snapshot = next(snapshot for snapshot in epoch_snapshots if int(snapshot.epoch) == selected_epoch)
    fmpc_tf2_module._restore_pc_parameters(model, selected_snapshot.model_snapshot)
    fmpc_tf2_module._restore_mlp_parameters(psi_network, selected_snapshot.psi_snapshot)

    evaluation_start = perf_counter()
    val_transport = fmpc_tf2_module._evaluate_transport_split(model, psi_network, config, split.x_val, split.y_val)
    test_transport = fmpc_tf2_module._evaluate_transport_split(model, psi_network, config, split.x_test, split.y_test)
    val_loss, val_accuracy = fmpc_tf2_module._evaluate_slow_pc_accuracy(model, split.x_val, split.y_val)
    test_loss, test_accuracy = fmpc_tf2_module._evaluate_slow_pc_accuracy(model, split.x_test, split.y_test)
    evaluation_wall_time_seconds = float(perf_counter() - evaluation_start)
    val_baseline_accuracy = majority_class_baseline_accuracy(split.y_val)
    test_baseline_accuracy = majority_class_baseline_accuracy(split.y_test)

    resolved_theta_update_cadence = fmpc_tf2_module._resolved_theta_update_cadence(config)
    theta_micro_lr, theta_micro_bias_lr = fmpc_tf2_module._theta_micro_learning_rates(
        config,
        resolved_theta_update_cadence,
    )
    mean_boot_delta = _mean([float(row["mean_relative_bootstrap_target_delta"]) for row in drift_epoch_rows])
    mean_identity_delta = _mean([float(row["mean_relative_identity_target_delta"]) for row in drift_epoch_rows])
    mean_boot_cos = _mean([float(row["mean_bootstrap_target_cosine_similarity"]) for row in drift_epoch_rows])
    mean_identity_cos = _mean([float(row["mean_identity_target_cosine_similarity"]) for row in drift_epoch_rows])
    mean_input_delta = _mean([float(row["mean_relative_psi_input_delta"]) for row in drift_epoch_rows])

    summary = {
        "phase": "Phase TF2",
        "stage": "batch_frozen_target_state_coupling_diagnostic",
        "diagnostic_only": True,
        "preset_name": config.preset_name,
        "candidate_key": candidate.key,
        "mode": mode,
        "teacher_free": True,
        "uses_teacher_artifacts": False,
        "feature_aware_tangents": bool(config.feature_aware_tangents),
        "identity_tangent_mode": fmpc_tf2_module._identity_tangent_mode(config),
        "micro_steps": int(config.micro_steps),
        "supervision_policy": config.supervision_policy,
        "theta_update_cadence": resolved_theta_update_cadence,
        "theta_update_budget": config.theta_update_budget,
        "theta_micro_lr": float(theta_micro_lr),
        "theta_micro_bias_lr": float(theta_micro_bias_lr),
        "bootstrap_integrator": config.bootstrap_integrator,
        "bootstrap_substeps": int(config.bootstrap_substeps),
        "identity_loss_weight": float(config.identity_loss_weight),
        "warmup_epochs": int(config.warmup_epochs),
        "hybrid_ramp_epochs": int(config.hybrid_ramp_epochs),
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
        "within_batch_drift": {
            "mean_relative_bootstrap_target_delta": float(mean_boot_delta),
            "mean_relative_identity_target_delta": float(mean_identity_delta),
            "mean_bootstrap_target_cosine_similarity": float(mean_boot_cos),
            "mean_identity_target_cosine_similarity": float(mean_identity_cos),
            "mean_relative_psi_input_delta": float(mean_input_delta),
        },
        "timing": {
            "train_wall_time_seconds": train_wall_time_seconds,
            "final_evaluation_wall_time_seconds": evaluation_wall_time_seconds,
        },
    }
    _write_run_artifacts(
        run_dir,
        config,
        mode=mode,
        epoch_rows=epoch_rows,
        drift_epoch_rows=drift_epoch_rows,
        selection_diagnostics=selection_diagnostics,
        summary=summary,
    )
    return _RunArtifacts(
        mode=mode,
        candidate=candidate,
        seed=int(seed),
        config=config,
        run_dir=run_dir,
        epoch_rows=epoch_rows,
        drift_epoch_rows=drift_epoch_rows,
        summary=summary,
    )


def _success_run_row(*, artifact: _RunArtifacts, base_run_dir: Path) -> dict[str, Any]:
    timing = dict(artifact.summary.get("timing", {}))
    return {
        "mode": artifact.mode,
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
    mode: BatchTargetStateMode,
    candidate_key: str,
) -> dict[str, Any]:
    relevant = [
        row
        for row in run_rows
        if row["mode"] == mode and row["candidate_key"] == candidate_key and row["run_status"] == "success"
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


def _aggregate_drift_group(
    drift_rows: list[dict[str, Any]],
    *,
    mode: BatchTargetStateMode,
    candidate_key: str,
) -> dict[str, Any]:
    relevant = [
        row for row in drift_rows if row["mode"] == mode and row["candidate_key"] == candidate_key
    ]
    if not relevant:
        return {
            "mean_relative_bootstrap_target_delta": None,
            "mean_relative_identity_target_delta": None,
            "mean_bootstrap_target_cosine_similarity": None,
            "mean_identity_target_cosine_similarity": None,
            "mean_relative_psi_input_delta": None,
        }
    return {
        "mean_relative_bootstrap_target_delta": _mean(
            [float(row["mean_relative_bootstrap_target_delta"]) for row in relevant]
        ),
        "mean_relative_identity_target_delta": _mean(
            [float(row["mean_relative_identity_target_delta"]) for row in relevant]
        ),
        "mean_bootstrap_target_cosine_similarity": _mean(
            [float(row["mean_bootstrap_target_cosine_similarity"]) for row in relevant]
        ),
        "mean_identity_target_cosine_similarity": _mean(
            [float(row["mean_identity_target_cosine_similarity"]) for row in relevant]
        ),
        "mean_relative_psi_input_delta": _mean(
            [float(row["mean_relative_psi_input_delta"]) for row in relevant]
        ),
    }


def _pairwise_vs_reference(
    run_rows: list[dict[str, Any]],
    *,
    candidate_key: str,
    mode: BatchTargetStateMode,
    reference_candidate_key: str,
    reference_mode: BatchTargetStateMode,
) -> dict[str, Any]:
    candidate_by_seed = {
        int(row["seed"]): row
        for row in run_rows
        if row["candidate_key"] == candidate_key and row["mode"] == mode and row["run_status"] == "success"
    }
    reference_by_seed = {
        int(row["seed"]): row
        for row in run_rows
        if row["candidate_key"] == reference_candidate_key
        and row["mode"] == reference_mode
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


def run_fmpc_tf2_batch_frozen_target_state_suite(
    config: FMPCTF2BatchFrozenTargetStateSuiteConfig,
) -> FMPCTF2BatchFrozenTargetStateSuiteRunResult:
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
    drift_rows: list[dict[str, Any]] = []
    for mode in ("live_target_recompute", "batch_frozen_target_state_cache"):
        for candidate in candidates:
            for seed in config.seeds:
                artifact = _run_one_candidate(
                    run_dir,
                    config,
                    candidate=candidate,
                    mode=mode,
                    seed=int(seed),
                )
                run_rows.append(_success_run_row(artifact=artifact, base_run_dir=run_dir))
                drift_rows.extend(artifact.drift_epoch_rows)

    _write_csv(run_dir / "end_to_end_runs.csv", run_rows)
    _write_csv(run_dir / "within_batch_drift_epoch_metrics.csv", drift_rows)

    live_baseline = _aggregate_run_group(
        run_rows,
        mode="live_target_recompute",
        candidate_key="baseline_plain_raw",
    )
    frozen_baseline = _aggregate_run_group(
        run_rows,
        mode="batch_frozen_target_state_cache",
        candidate_key="baseline_plain_raw",
    )
    live_challenger = _aggregate_run_group(
        run_rows,
        mode="live_target_recompute",
        candidate_key="residualized_local_field_poly_rt2",
    )
    frozen_challenger = _aggregate_run_group(
        run_rows,
        mode="batch_frozen_target_state_cache",
        candidate_key="residualized_local_field_poly_rt2",
    )

    live_baseline_drift = _aggregate_drift_group(
        drift_rows,
        mode="live_target_recompute",
        candidate_key="baseline_plain_raw",
    )
    frozen_baseline_drift = _aggregate_drift_group(
        drift_rows,
        mode="batch_frozen_target_state_cache",
        candidate_key="baseline_plain_raw",
    )
    live_challenger_drift = _aggregate_drift_group(
        drift_rows,
        mode="live_target_recompute",
        candidate_key="residualized_local_field_poly_rt2",
    )
    frozen_challenger_drift = _aggregate_drift_group(
        drift_rows,
        mode="batch_frozen_target_state_cache",
        candidate_key="residualized_local_field_poly_rt2",
    )

    baseline_same_family_live_delta = _pairwise_vs_reference(
        run_rows,
        candidate_key="baseline_plain_raw",
        mode="batch_frozen_target_state_cache",
        reference_candidate_key="baseline_plain_raw",
        reference_mode="live_target_recompute",
    )
    challenger_same_family_live_delta = _pairwise_vs_reference(
        run_rows,
        candidate_key="residualized_local_field_poly_rt2",
        mode="batch_frozen_target_state_cache",
        reference_candidate_key="residualized_local_field_poly_rt2",
        reference_mode="live_target_recompute",
    )

    pairwise_vs_baseline_live = {
        "baseline_plain_raw__live_target_recompute": _pairwise_vs_reference(
            run_rows,
            candidate_key="baseline_plain_raw",
            mode="live_target_recompute",
            reference_candidate_key="baseline_plain_raw",
            reference_mode="live_target_recompute",
        ),
        "baseline_plain_raw__batch_frozen_target_state_cache": _pairwise_vs_reference(
            run_rows,
            candidate_key="baseline_plain_raw",
            mode="batch_frozen_target_state_cache",
            reference_candidate_key="baseline_plain_raw",
            reference_mode="live_target_recompute",
        ),
        "residualized_local_field_poly_rt2__live_target_recompute": _pairwise_vs_reference(
            run_rows,
            candidate_key="residualized_local_field_poly_rt2",
            mode="live_target_recompute",
            reference_candidate_key="baseline_plain_raw",
            reference_mode="live_target_recompute",
        ),
        "residualized_local_field_poly_rt2__batch_frozen_target_state_cache": _pairwise_vs_reference(
            run_rows,
            candidate_key="residualized_local_field_poly_rt2",
            mode="batch_frozen_target_state_cache",
            reference_candidate_key="baseline_plain_raw",
            reference_mode="live_target_recompute",
        ),
    }
    pairwise_vs_same_family_live = {
        "baseline_plain_raw": baseline_same_family_live_delta,
        "residualized_local_field_poly_rt2": challenger_same_family_live_delta,
    }

    challenger_gain = float(challenger_same_family_live_delta["mean_test_accuracy_delta"])
    baseline_gain = float(baseline_same_family_live_delta["mean_test_accuracy_delta"])
    challenger_rescued = bool(
        challenger_gain >= float(config.material_test_gain)
        and abs(baseline_gain) <= float(config.baseline_similarity_tolerance)
    )
    cache_helps_generally = bool(
        challenger_gain >= float(config.material_test_gain)
        and baseline_gain >= float(config.material_test_gain)
        and abs(challenger_gain - baseline_gain) <= float(config.baseline_similarity_tolerance)
    )
    neither_improves_materially = bool(
        challenger_gain < float(config.material_test_gain) and baseline_gain < float(config.material_test_gain)
    )
    challenger_drift_reduction = bool(
        float(frozen_challenger_drift["mean_relative_bootstrap_target_delta"])
        <= (1.0 - float(config.drift_reduction_fraction))
        * float(live_challenger_drift["mean_relative_bootstrap_target_delta"])
        and float(frozen_challenger_drift["mean_relative_identity_target_delta"])
        <= (1.0 - float(config.drift_reduction_fraction))
        * float(live_challenger_drift["mean_relative_identity_target_delta"])
    )
    drift_nontrivial_but_no_behavior_gain = bool(
        challenger_drift_reduction and challenger_gain < float(config.material_test_gain)
    )
    within_batch_live_target_state_coupling_present = bool(challenger_rescued)

    if challenger_rescued:
        dominant_interpretation = "within_batch_live_target_state_coupling"
        next_single_narrow_move = (
            "run one narrow cached-state-only versus cached-target-only split diagnostic to separate "
            "which cached object class actually drives the rescue"
        )
    elif cache_helps_generally:
        dominant_interpretation = "batch_frozen_cache_regularizes_both_candidates_not_challenger_specific"
        next_single_narrow_move = (
            "run one narrow baseline-only cache-ablation to separate generic stabilization from challenger-specific coupling"
        )
    elif drift_nontrivial_but_no_behavior_gain:
        dominant_interpretation = "deeper_structural_downstream_coupling_beyond_simple_batch_cache"
        next_single_narrow_move = (
            "run one narrow cached-target-only versus cached-state-only diagnostic to identify whether "
            "state evolution or supervision reuse is the remaining structural bottleneck"
        )
    elif neither_improves_materially:
        dominant_interpretation = "simple_batch_frozen_target_state_cache_not_main_limiter"
        next_single_narrow_move = (
            "run one narrow cached-target-only versus cached-state-only split diagnostic before reopening any broader TF2 mechanism search"
        )
    else:
        dominant_interpretation = "mixed_or_ambiguous_batch_cache_effect"
        next_single_narrow_move = (
            "run one narrow cached-target-only versus cached-state-only split diagnostic to disambiguate the remaining coupling"
        )

    summary = {
        "phase": "Phase TF2",
        "stage": "batch_frozen_target_state_coupling_diagnostic",
        "diagnostic_only": True,
        "end_to_end_summary": {
            "baseline_plain_raw": {
                "live_target_recompute": live_baseline,
                "batch_frozen_target_state_cache": frozen_baseline,
            },
            "residualized_local_field_poly_rt2": {
                "live_target_recompute": live_challenger,
                "batch_frozen_target_state_cache": frozen_challenger,
            },
        },
        "within_batch_drift_summary": {
            "baseline_plain_raw": {
                "live_target_recompute": live_baseline_drift,
                "batch_frozen_target_state_cache": frozen_baseline_drift,
            },
            "residualized_local_field_poly_rt2": {
                "live_target_recompute": live_challenger_drift,
                "batch_frozen_target_state_cache": frozen_challenger_drift,
            },
        },
        "pairwise_delta_vs_baseline_live_target_run": pairwise_vs_baseline_live,
        "pairwise_delta_vs_same_family_live_target_run": pairwise_vs_same_family_live,
        "is_within_batch_live_target_state_coupling_present": within_batch_live_target_state_coupling_present,
        "is_challenger_rescued_by_batch_frozen_target_state_cache": challenger_rescued,
        "batch_frozen_cache_helps_generally_but_not_challenger_specifically": cache_helps_generally,
        "drift_is_nontrivial_but_behavior_does_not_improve": drift_nontrivial_but_no_behavior_gain,
        "dominant_interpretation": dominant_interpretation,
        "next_single_narrow_move": next_single_narrow_move,
        "artifacts": {
            "end_to_end_runs_csv": "end_to_end_runs.csv",
            "within_batch_drift_epoch_metrics_csv": "within_batch_drift_epoch_metrics.csv",
        },
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF2BatchFrozenTargetStateSuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config, candidates),
        run_rows=run_rows,
        drift_rows=drift_rows,
        summary=summary,
    )
