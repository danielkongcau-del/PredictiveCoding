from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np

from .datasets import load_digits_split
from .fmpc_tf1_flow import bootstrap_average_velocity_target, build_tf1_context, hidden_energy_from_state
from .fmpc_tf2 import (
    FMPCTF2Config,
    _active_onpolicy_mix_ratio,
    _active_theta_update_cadence,
    _lambda_id_for_epoch,
    _make_pc_model,
    _make_psi_network,
    _run_tf2_micro_step,
    _theta_micro_learning_rates,
    _theta_update_due_for_step,
    _theta_update_from_transported_state,
    build_tf2_corrective_transport_default_config,
    run_fmpc_tf2_experiment,
)
from .minibatch import iter_minibatches


BootstrapIntegrator = Literal["euler", "rk2"]


@dataclass(frozen=True)
class FMPCTF2BootstrapCandidate:
    key: str
    integrator: BootstrapIntegrator
    substeps: int


@dataclass
class FMPCTF2BootstrapFidelitySuiteConfig:
    """Offline-first bootstrap-target fidelity study for the corrective TF2 default."""

    experiment_name: str = "fmpc_tf2_bootstrap_fidelity_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    offline_probe_seeds: tuple[int, ...] = (0, 1, 2)
    end_to_end_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 60
    batch_size: int = 128
    eval_steps: int = 15
    layer_dims: tuple[int, ...] = (64, 64, 10)
    offline_probe_epoch_indices: tuple[int, ...] = (0, 4, 14, 29, 59)
    sample_batches_per_probe_epoch: int = 1
    bootstrap_integrators: tuple[BootstrapIntegrator, ...] = ("euler", "rk2")
    bootstrap_substeps_options: tuple[int, ...] = (1, 2, 4, 8, 16)
    reference_integrator: BootstrapIntegrator = "rk2"
    reference_substeps: int = 64
    max_pruned_nondefault_candidates: int = 2
    material_test_gain: float = 0.005
    material_val_gain: float = 0.0
    slow_pc_reference_runs_path: str | Path = "outputs/fmpc_tf2_default_adoption_suite/aggregate_runs.csv"
    slow_pc_reference_name: str = "canonical_slow_pc_digits_baseline"

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"


@dataclass
class FMPCTF2BootstrapFidelitySuiteRunResult:
    run_dir: Path
    config: dict[str, Any]
    offline_rows: list[dict[str, Any]]
    end_to_end_rows: list[dict[str, Any]]
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


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Required TF2 bootstrap-fidelity reference artifact is missing: {path_obj}")
    with path_obj.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _relative_posix(base_dir: Path, target: Path) -> str:
    return target.relative_to(base_dir).as_posix()


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


def _candidate_key(integrator: BootstrapIntegrator, substeps: int) -> str:
    return f"{integrator}_s{int(substeps)}"


def _candidate_registry(config: FMPCTF2BootstrapFidelitySuiteConfig) -> list[FMPCTF2BootstrapCandidate]:
    candidates: list[FMPCTF2BootstrapCandidate] = []
    for integrator in config.bootstrap_integrators:
        for substeps in config.bootstrap_substeps_options:
            candidates.append(
                FMPCTF2BootstrapCandidate(
                    key=_candidate_key(integrator, int(substeps)),
                    integrator=integrator,
                    substeps=int(substeps),
                )
            )
    if not any(candidate.key == "rk2_s4" for candidate in candidates):
        candidates.append(
            FMPCTF2BootstrapCandidate(
                key="rk2_s4",
                integrator="rk2",
                substeps=4,
            )
        )
    return candidates


def _resolve_probe_epoch_indices(config: FMPCTF2BootstrapFidelitySuiteConfig) -> tuple[int, ...]:
    resolved = sorted(
        {
            int(epoch_index)
            for epoch_index in config.offline_probe_epoch_indices
            if 0 <= int(epoch_index) < int(config.epochs)
        }
    )
    if not resolved:
        return (0,)
    return tuple(resolved)


def _suite_config_payload(
    config: FMPCTF2BootstrapFidelitySuiteConfig,
    candidates: list[FMPCTF2BootstrapCandidate],
) -> dict[str, Any]:
    return {
        "phase": "Phase TF2",
        "stage": "corrective_bootstrap_fidelity_suite",
        "candidate_keys": [candidate.key for candidate in candidates],
        "offline_probe_seeds": [int(seed) for seed in config.offline_probe_seeds],
        "end_to_end_seeds": [int(seed) for seed in config.end_to_end_seeds],
        "offline_probe_epoch_indices": [int(value) for value in _resolve_probe_epoch_indices(config)],
        "sample_batches_per_probe_epoch": int(config.sample_batches_per_probe_epoch),
        "fixed": {
            "preset_name": "tf2_corrective_transport_default",
            "use_teacher_free_features": True,
            "feature_aware_tangents": False,
            "micro_steps": 4,
            "incremental_weight_updates": False,
            "supervision_policy": "local_only",
            "theta_update_cadence": "terminal_only",
            "theta_update_budget": "matched",
            "identity_loss_weight": 0.2,
            "warmup_epochs": 5,
            "hybrid_ramp_epochs": 10,
            "selector": "gate_constrained_accuracy_then_val_accuracy",
        },
        "offline_reference_definition": {
            "integrator": config.reference_integrator,
            "substeps": int(config.reference_substeps),
            "same_vector_field": "hidden_local_flow(context, z)",
            "same_horizon": True,
            "sampled_stream": "z_lf_k",
        },
        "pruning_rule": {
            "always_keep_default": "rk2_s4",
            "rank_nondefault_candidates_by": [
                "mean_relative_mse_to_reference",
                "mean_endpoint_displacement_error",
                "mean_wall_time_seconds_per_eval",
            ],
            "max_pruned_nondefault_candidates": int(config.max_pruned_nondefault_candidates),
        },
        "slow_pc_reference_runs_path": str(config.slow_pc_reference_runs_path),
        "slow_pc_reference_name": str(config.slow_pc_reference_name),
    }


def _load_slow_pc_reference_by_seed(
    rows: list[dict[str, str]],
    *,
    reference_name: str,
    seeds: tuple[int, ...],
) -> dict[int, dict[str, float]]:
    filtered = [row for row in rows if str(row.get("preset_name", "")) == reference_name]
    if not filtered:
        raise ValueError(f"No slow-PC reference rows found for '{reference_name}'.")
    by_seed = {
        int(row["seed"]): {
            "val_accuracy": float(row["val_accuracy"]),
            "test_accuracy": float(row["test_accuracy"]),
        }
        for row in filtered
    }
    missing = [seed for seed in seeds if seed not in by_seed]
    if missing:
        raise ValueError(f"Slow-PC reference rows are missing seeds: {missing}.")
    return {int(seed): by_seed[int(seed)] for seed in seeds}


def _cosine_mean(a: np.ndarray, b: np.ndarray) -> float:
    a_array = np.asarray(a, dtype=np.float64)
    b_array = np.asarray(b, dtype=np.float64)
    numerator = np.sum(a_array * b_array, axis=1)
    denom = np.linalg.norm(a_array, axis=1) * np.linalg.norm(b_array, axis=1)
    denom = np.maximum(denom, 1e-12)
    return float(np.mean(numerator / denom))


def _offline_metric_row(
    *,
    probe_seed: int,
    probe_epoch: int,
    batch_index: int,
    step_index: int,
    t_k: float,
    r_k: float,
    candidate: FMPCTF2BootstrapCandidate,
    mse_to_reference: float,
    relative_mse_to_reference: float,
    cosine_similarity_to_reference: float,
    endpoint_displacement_error: float,
    hidden_energy_after_bootstrap_step: float,
    wall_time_seconds_per_eval: float,
) -> dict[str, Any]:
    return {
        "probe_seed": int(probe_seed),
        "probe_epoch": int(probe_epoch),
        "batch_index": int(batch_index),
        "step_index": int(step_index),
        "t_k": float(t_k),
        "r_k": float(r_k),
        "candidate_key": candidate.key,
        "bootstrap_integrator": candidate.integrator,
        "bootstrap_substeps": int(candidate.substeps),
        "mse_to_reference_average_velocity": float(mse_to_reference),
        "relative_mse_to_reference_average_velocity": float(relative_mse_to_reference),
        "cosine_similarity_to_reference_average_velocity": float(cosine_similarity_to_reference),
        "endpoint_displacement_error": float(endpoint_displacement_error),
        "hidden_energy_after_bootstrap_step": float(hidden_energy_after_bootstrap_step),
        "wall_time_seconds_per_eval": float(wall_time_seconds_per_eval),
    }


def _reference_target(
    context: Any,
    z_t: np.ndarray,
    *,
    t_k: float,
    r_k: float,
    config: FMPCTF2BootstrapFidelitySuiteConfig,
) -> tuple[np.ndarray, np.ndarray]:
    start = perf_counter()
    u_reference = bootstrap_average_velocity_target(
        context,
        z_t,
        t=t_k,
        r=r_k,
        integrator=config.reference_integrator,
        substeps=config.reference_substeps,
    )
    elapsed = perf_counter() - start
    return u_reference, np.full((int(z_t.shape[0]),), float(elapsed), dtype=np.float64)


def _candidate_target_metrics(
    context: Any,
    z_t: np.ndarray,
    *,
    t_k: float,
    r_k: float,
    candidate: FMPCTF2BootstrapCandidate,
    u_reference: np.ndarray,
) -> dict[str, float]:
    start = perf_counter()
    u_candidate = bootstrap_average_velocity_target(
        context,
        z_t,
        t=t_k,
        r=r_k,
        integrator=candidate.integrator,
        substeps=candidate.substeps,
    )
    elapsed = perf_counter() - start
    diff = np.asarray(u_candidate, dtype=np.float64) - np.asarray(u_reference, dtype=np.float64)
    mse = float(np.mean(diff**2))
    reference_sq = float(np.mean(np.asarray(u_reference, dtype=np.float64) ** 2))
    relative_mse = float(mse / max(reference_sq, 1e-12))
    endpoint_candidate = np.asarray(z_t, dtype=np.float64) + float(r_k) * np.asarray(u_candidate, dtype=np.float64)
    endpoint_reference = np.asarray(z_t, dtype=np.float64) + float(r_k) * np.asarray(u_reference, dtype=np.float64)
    endpoint_error = float(np.mean(np.linalg.norm(endpoint_candidate - endpoint_reference, axis=1)))
    hidden_energy = hidden_energy_from_state(context, endpoint_candidate)
    return {
        "mse": mse,
        "relative_mse": relative_mse,
        "cosine": _cosine_mean(u_candidate, u_reference),
        "endpoint_error": endpoint_error,
        "hidden_energy": float(hidden_energy),
        "wall_time": float(elapsed),
    }


def _collect_offline_rows(
    config: FMPCTF2BootstrapFidelitySuiteConfig,
    candidates: list[FMPCTF2BootstrapCandidate],
) -> list[dict[str, Any]]:
    probe_epoch_indices = set(_resolve_probe_epoch_indices(config))
    offline_rows: list[dict[str, Any]] = []
    for seed in config.offline_probe_seeds:
        run_config = build_tf2_corrective_transport_default_config(
            run_seed=seed,
            data_seed=seed,
            model_init_seed=seed,
            psi_init_seed=seed,
            batch_order_seed=seed,
            epochs=config.epochs,
            batch_size=config.batch_size,
            eval_steps=config.eval_steps,
            layer_dims=config.layer_dims,
        )
        split = load_digits_split(
            split_seed=run_config.data_seed,
            train_fraction=run_config.train_fraction,
            val_fraction=run_config.val_fraction,
            test_fraction=run_config.test_fraction,
        )
        model = _make_pc_model(run_config)
        psi_network = _make_psi_network(run_config)

        for epoch_index in range(run_config.epochs):
            lambda_id = _lambda_id_for_epoch(run_config, epoch_index)
            batch_seed = run_config.batch_order_seed + epoch_index
            active_cadence = _active_theta_update_cadence(run_config, epoch_index)
            micro_eta_w, micro_eta_b = _theta_micro_learning_rates(run_config, active_cadence)
            active_mix_ratio = _active_onpolicy_mix_ratio(run_config, epoch_index)
            for batch_index, (x_batch, y_batch) in enumerate(
                iter_minibatches(
                    split.x_train,
                    split.y_train,
                    run_config.batch_size,
                    shuffle=run_config.shuffle_batches,
                    seed=batch_seed,
                )
            ):
                should_probe = epoch_index in probe_epoch_indices and batch_index < int(config.sample_batches_per_probe_epoch)
                if not should_probe:
                    from .fmpc_tf2 import _train_one_batch_tf2

                    _train_one_batch_tf2(
                        model,
                        psi_network,
                        run_config,
                        x_batch,
                        y_batch,
                        lambda_id=lambda_id,
                        epoch_index=epoch_index,
                    )
                    continue

                context = build_tf1_context(model, x_batch, y_batch)
                knots = np.linspace(0.0, 1.0, int(run_config.micro_steps) + 1, dtype=np.float64)
                z_on = context.z0.copy()
                z_lf = context.z0.copy()
                for step_index in range(run_config.micro_steps):
                    t_k = float(knots[step_index])
                    r_k = 1.0 - t_k
                    dt = float(knots[step_index + 1] - knots[step_index])
                    u_reference, _ = _reference_target(
                        context,
                        z_lf,
                        t_k=t_k,
                        r_k=r_k,
                        config=config,
                    )
                    for candidate in candidates:
                        metrics = _candidate_target_metrics(
                            context,
                            z_lf,
                            t_k=t_k,
                            r_k=r_k,
                            candidate=candidate,
                            u_reference=u_reference,
                        )
                        offline_rows.append(
                            _offline_metric_row(
                                probe_seed=seed,
                                probe_epoch=epoch_index,
                                batch_index=batch_index,
                                step_index=step_index,
                                t_k=t_k,
                                r_k=r_k,
                                candidate=candidate,
                                mse_to_reference=metrics["mse"],
                                relative_mse_to_reference=metrics["relative_mse"],
                                cosine_similarity_to_reference=metrics["cosine"],
                                endpoint_displacement_error=metrics["endpoint_error"],
                                hidden_energy_after_bootstrap_step=metrics["hidden_energy"],
                                wall_time_seconds_per_eval=metrics["wall_time"],
                            )
                        )
                    z_on, z_lf, _, _, _, _, _ = _run_tf2_micro_step(
                        model,
                        psi_network,
                        run_config,
                        context,
                        z_on,
                        z_lf,
                        t_k=t_k,
                        dt=dt,
                        r_k=r_k,
                        lambda_id=lambda_id,
                        apply_theta_update=_theta_update_due_for_step(active_cadence, step_index),
                        theta_eta_w=micro_eta_w,
                        theta_eta_b=micro_eta_b,
                        onpolicy_mix_ratio=active_mix_ratio,
                    )
                if active_cadence == "terminal_only":
                    _theta_update_from_transported_state(
                        model,
                        context,
                        z_on,
                        eta_w=float(run_config.eta_w),
                        eta_b=float(run_config.eta_b if run_config.eta_b is not None else run_config.eta_w),
                    )
    return offline_rows


def _offline_summary(rows: list[dict[str, Any]], candidate: FMPCTF2BootstrapCandidate) -> dict[str, Any]:
    candidate_rows = [row for row in rows if str(row["candidate_key"]) == candidate.key]
    if not candidate_rows:
        raise ValueError(f"No offline rows found for candidate '{candidate.key}'.")
    mse_values = [float(row["mse_to_reference_average_velocity"]) for row in candidate_rows]
    rel_mse_values = [float(row["relative_mse_to_reference_average_velocity"]) for row in candidate_rows]
    cosine_values = [float(row["cosine_similarity_to_reference_average_velocity"]) for row in candidate_rows]
    endpoint_values = [float(row["endpoint_displacement_error"]) for row in candidate_rows]
    energy_values = [float(row["hidden_energy_after_bootstrap_step"]) for row in candidate_rows]
    wall_times = [float(row["wall_time_seconds_per_eval"]) for row in candidate_rows]
    return {
        "candidate_key": candidate.key,
        "bootstrap_integrator": candidate.integrator,
        "bootstrap_substeps": int(candidate.substeps),
        "num_probe_evaluations": int(len(candidate_rows)),
        "mean_mse_to_reference_average_velocity": _mean(mse_values),
        "mean_relative_mse_to_reference_average_velocity": _mean(rel_mse_values),
        "mean_cosine_similarity_to_reference_average_velocity": _mean(cosine_values),
        "mean_endpoint_displacement_error": _mean(endpoint_values),
        "mean_hidden_energy_after_bootstrap_step": _mean(energy_values),
        "mean_wall_time_seconds_per_eval": _mean(wall_times),
    }


def _pruned_candidates(
    candidates: list[FMPCTF2BootstrapCandidate],
    offline_summaries: dict[str, dict[str, Any]],
    *,
    max_nondefault: int,
) -> list[FMPCTF2BootstrapCandidate]:
    default_candidate = next(candidate for candidate in candidates if candidate.key == "rk2_s4")
    nondefault = [candidate for candidate in candidates if candidate.key != default_candidate.key]
    ranked = sorted(
        nondefault,
        key=lambda candidate: (
            float(offline_summaries[candidate.key]["mean_relative_mse_to_reference_average_velocity"]),
            float(offline_summaries[candidate.key]["mean_endpoint_displacement_error"]),
            float(offline_summaries[candidate.key]["mean_wall_time_seconds_per_eval"]),
        ),
    )
    return [default_candidate, *ranked[: int(max_nondefault)]]


def _success_end_to_end_row(
    *,
    run_index: int,
    candidate: FMPCTF2BootstrapCandidate,
    seed: int,
    result: Any,
    run_dir: Path,
) -> dict[str, Any]:
    summary = result.summary
    total_wall_time_seconds = float(summary["timing"]["train_wall_time_seconds"]) + float(
        summary["timing"]["final_evaluation_wall_time_seconds"]
    )
    return {
        "run_index": int(run_index),
        "candidate_key": candidate.key,
        "bootstrap_integrator": candidate.integrator,
        "bootstrap_substeps": int(candidate.substeps),
        "seed": int(seed),
        "checkpoint_selector": str(summary["checkpoint_selector"]),
        "val_accuracy": float(summary["val_accuracy"]),
        "test_accuracy": float(summary["test_accuracy"]),
        "gate_passing_epoch_count": int(summary["gate_passing_epoch_count"]),
        "val_transported_final_energy": float(summary["val_transported_final_energy"]),
        "selected_epoch": int(summary["best_epoch"]),
        "selected_epoch_passes_gate": bool(summary["selected_epoch_passes_gate"]),
        "selector_fallback_used": bool(summary["selector_fallback_used"]),
        "total_wall_time_seconds": float(total_wall_time_seconds),
        "run_status": "completed",
        "nan_or_inf_failure": False,
        "failure_type": "",
        "failure_message": "",
        "run_summary_path": _relative_posix(run_dir, result.run_dir / "summary.json"),
    }


def _failure_end_to_end_row(
    *,
    run_index: int,
    candidate: FMPCTF2BootstrapCandidate,
    seed: int,
    error: Exception,
) -> dict[str, Any]:
    message = str(error)
    lowered = message.lower()
    return {
        "run_index": int(run_index),
        "candidate_key": candidate.key,
        "bootstrap_integrator": candidate.integrator,
        "bootstrap_substeps": int(candidate.substeps),
        "seed": int(seed),
        "checkpoint_selector": "",
        "val_accuracy": None,
        "test_accuracy": None,
        "gate_passing_epoch_count": None,
        "val_transported_final_energy": None,
        "selected_epoch": None,
        "selected_epoch_passes_gate": None,
        "selector_fallback_used": None,
        "total_wall_time_seconds": None,
        "run_status": "failed",
        "nan_or_inf_failure": bool("nan" in lowered or "inf" in lowered),
        "failure_type": type(error).__name__,
        "failure_message": message,
        "run_summary_path": "",
    }


def _end_to_end_summary(rows: list[dict[str, Any]], candidate: FMPCTF2BootstrapCandidate, *, slow_pc_ref: dict[int, dict[str, float]]) -> dict[str, Any]:
    candidate_rows = [row for row in rows if str(row["candidate_key"]) == candidate.key]
    successes = [row for row in candidate_rows if str(row["run_status"]) == "completed"]
    payload: dict[str, Any] = {
        "candidate_key": candidate.key,
        "bootstrap_integrator": candidate.integrator,
        "bootstrap_substeps": int(candidate.substeps),
        "num_runs": int(len(candidate_rows)),
        "num_completed_runs": int(len(successes)),
        "num_failures": int(sum(1 for row in candidate_rows if str(row["run_status"]) == "failed")),
        "nan_or_inf_failure_count": int(sum(1 for row in candidate_rows if bool(row["nan_or_inf_failure"]))),
    }
    if not successes:
        payload.update(
            {
                "mean_val_accuracy": None,
                "std_val_accuracy": None,
                "mean_test_accuracy": None,
                "std_test_accuracy": None,
                "mean_gate_passing_epoch_count": None,
                "mean_val_transported_final_energy": None,
                "mean_total_wall_time_seconds": None,
                "std_total_wall_time_seconds": None,
                "mean_test_accuracy_gap_to_slow_pc": None,
            }
        )
        return payload
    val_accuracies = [float(row["val_accuracy"]) for row in successes]
    test_accuracies = [float(row["test_accuracy"]) for row in successes]
    gate_counts = [float(row["gate_passing_epoch_count"]) for row in successes]
    energies = [float(row["val_transported_final_energy"]) for row in successes]
    wall_times = [float(row["total_wall_time_seconds"]) for row in successes]
    slow_test_accuracies = [float(slow_pc_ref[int(row["seed"])]["test_accuracy"]) for row in successes]
    payload.update(
        {
            "mean_val_accuracy": _mean(val_accuracies),
            "std_val_accuracy": _std(val_accuracies),
            "mean_test_accuracy": _mean(test_accuracies),
            "std_test_accuracy": _std(test_accuracies),
            "mean_gate_passing_epoch_count": _mean(gate_counts),
            "mean_val_transported_final_energy": _mean(energies),
            "mean_total_wall_time_seconds": _mean(wall_times),
            "std_total_wall_time_seconds": _std(wall_times),
            "mean_test_accuracy_gap_to_slow_pc": _mean(test_accuracies) - _mean(slow_test_accuracies),
        }
    )
    return payload


def _pairwise_delta(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if reference["mean_test_accuracy"] is None or candidate["mean_test_accuracy"] is None:
        return {
            "mean_val_accuracy_delta": None,
            "mean_test_accuracy_delta": None,
            "mean_gate_passing_epoch_count_delta": None,
            "mean_val_transported_final_energy_delta": None,
            "mean_total_wall_time_seconds_delta": None,
            "mean_test_gap_to_slow_pc_delta": None,
        }
    return {
        "mean_val_accuracy_delta": float(candidate["mean_val_accuracy"]) - float(reference["mean_val_accuracy"]),
        "mean_test_accuracy_delta": float(candidate["mean_test_accuracy"]) - float(reference["mean_test_accuracy"]),
        "mean_gate_passing_epoch_count_delta": float(candidate["mean_gate_passing_epoch_count"])
        - float(reference["mean_gate_passing_epoch_count"]),
        "mean_val_transported_final_energy_delta": float(candidate["mean_val_transported_final_energy"])
        - float(reference["mean_val_transported_final_energy"]),
        "mean_total_wall_time_seconds_delta": float(candidate["mean_total_wall_time_seconds"])
        - float(reference["mean_total_wall_time_seconds"]),
        "mean_test_gap_to_slow_pc_delta": float(candidate["mean_test_accuracy_gap_to_slow_pc"])
        - float(reference["mean_test_accuracy_gap_to_slow_pc"]),
    }


def run_fmpc_tf2_bootstrap_fidelity_suite(
    config: FMPCTF2BootstrapFidelitySuiteConfig,
) -> FMPCTF2BootstrapFidelitySuiteRunResult:
    """Run an offline-first bootstrap-target fidelity study for the corrective TF2 default."""

    candidates = _candidate_registry(config)
    slow_pc_rows = _read_csv(config.slow_pc_reference_runs_path)
    slow_pc_ref = _load_slow_pc_reference_by_seed(
        slow_pc_rows,
        reference_name=config.slow_pc_reference_name,
        seeds=config.end_to_end_seeds,
    )

    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, config.resolved_run_id(), config.output_layout)
    )
    _write_json(run_dir / "config.json", _suite_config_payload(config, candidates))

    offline_rows = _collect_offline_rows(config, candidates)
    offline_summaries = {candidate.key: _offline_summary(offline_rows, candidate) for candidate in candidates}
    pruned_candidates = _pruned_candidates(
        candidates,
        offline_summaries,
        max_nondefault=config.max_pruned_nondefault_candidates,
    )

    offline_csv_rows = [
        {
            "candidate_key": summary["candidate_key"],
            "bootstrap_integrator": summary["bootstrap_integrator"],
            "bootstrap_substeps": int(summary["bootstrap_substeps"]),
            "num_probe_evaluations": int(summary["num_probe_evaluations"]),
            "mean_mse_to_reference_average_velocity": float(summary["mean_mse_to_reference_average_velocity"]),
            "mean_relative_mse_to_reference_average_velocity": float(
                summary["mean_relative_mse_to_reference_average_velocity"]
            ),
            "mean_cosine_similarity_to_reference_average_velocity": float(
                summary["mean_cosine_similarity_to_reference_average_velocity"]
            ),
            "mean_endpoint_displacement_error": float(summary["mean_endpoint_displacement_error"]),
            "mean_hidden_energy_after_bootstrap_step": float(summary["mean_hidden_energy_after_bootstrap_step"]),
            "mean_wall_time_seconds_per_eval": float(summary["mean_wall_time_seconds_per_eval"]),
        }
        for summary in sorted(
            offline_summaries.values(),
            key=lambda item: (
                float(item["mean_relative_mse_to_reference_average_velocity"]),
                float(item["mean_endpoint_displacement_error"]),
                float(item["mean_wall_time_seconds_per_eval"]),
            ),
        )
    ]
    _write_csv(run_dir / "offline_target_fidelity.csv", offline_csv_rows)

    end_to_end_rows: list[dict[str, Any]] = []
    run_index = 0
    runs_root = run_dir / "runs"
    for candidate in pruned_candidates:
        for seed in config.end_to_end_seeds:
            run_index += 1
            run_config = build_tf2_corrective_transport_default_config(
                output_root=runs_root,
                output_layout="run_id_subdir",
                run_id=f"{candidate.key}_seed{seed}",
                run_seed=seed,
                data_seed=seed,
                model_init_seed=seed,
                psi_init_seed=seed,
                batch_order_seed=seed,
                epochs=config.epochs,
                batch_size=config.batch_size,
                eval_steps=config.eval_steps,
                layer_dims=config.layer_dims,
                bootstrap_integrator=candidate.integrator,
                bootstrap_substeps=candidate.substeps,
            )
            try:
                result = run_fmpc_tf2_experiment(run_config)
            except Exception as error:  # pragma: no cover - failure path
                end_to_end_rows.append(
                    _failure_end_to_end_row(
                        run_index=run_index,
                        candidate=candidate,
                        seed=seed,
                        error=error,
                    )
                )
            else:
                end_to_end_rows.append(
                    _success_end_to_end_row(
                        run_index=run_index,
                        candidate=candidate,
                        seed=seed,
                        result=result,
                        run_dir=run_dir,
                    )
                )

    end_to_end_csv_rows = [
        {
            **row,
            "selected_epoch_passes_gate": ""
            if row["selected_epoch_passes_gate"] is None
            else str(bool(row["selected_epoch_passes_gate"])),
            "selector_fallback_used": ""
            if row["selector_fallback_used"] is None
            else str(bool(row["selector_fallback_used"])),
            "nan_or_inf_failure": str(bool(row["nan_or_inf_failure"])),
        }
        for row in end_to_end_rows
    ]
    _write_csv(run_dir / "end_to_end_runs.csv", end_to_end_csv_rows)

    end_to_end_summaries = {
        candidate.key: _end_to_end_summary(end_to_end_rows, candidate, slow_pc_ref=slow_pc_ref)
        for candidate in pruned_candidates
    }
    default_summary = end_to_end_summaries["rk2_s4"]
    best_summary = max(
        end_to_end_summaries.values(),
        key=lambda item: (
            float("-inf") if item["mean_test_accuracy"] is None else float(item["mean_test_accuracy"]),
            float("-inf") if item["mean_val_accuracy"] is None else float(item["mean_val_accuracy"]),
        ),
    )
    material_gain = bool(
        str(best_summary["candidate_key"]) != "rk2_s4"
        and best_summary["mean_test_accuracy"] is not None
        and float(best_summary["mean_test_accuracy"]) - float(default_summary["mean_test_accuracy"]) >= float(config.material_test_gain)
        and float(best_summary["mean_val_accuracy"]) - float(default_summary["mean_val_accuracy"]) >= float(config.material_val_gain)
        and int(best_summary["num_failures"]) == 0
        and int(best_summary["nan_or_inf_failure_count"]) == 0
    )

    if material_gain:
        bootstrap_is_bottleneck = True
        should_change_default = True
        next_move = "adopt the higher-fidelity bootstrap target inside tf2_corrective_transport_default and confirm it with a narrow multiseed validation pass"
    else:
        bootstrap_is_bottleneck = False
        should_change_default = False
        next_move = "keep the current corrective default and narrow the next move beyond curriculum/bootstrap fidelity, because u_boot fidelity is not the current limiter"

    summary = {
        "phase": "Phase TF2",
        "stage": "corrective_bootstrap_fidelity_suite",
        "offline_reference_definition": _suite_config_payload(config, candidates)["offline_reference_definition"],
        "pruned_candidates_for_end_to_end": [
            {
                "candidate_key": candidate.key,
                "bootstrap_integrator": candidate.integrator,
                "bootstrap_substeps": int(candidate.substeps),
            }
            for candidate in pruned_candidates
        ],
        "offline_target_fidelity_by_candidate": {
            key: {
                "mean_mse_to_reference_average_velocity": float(value["mean_mse_to_reference_average_velocity"]),
                "mean_relative_mse_to_reference_average_velocity": float(value["mean_relative_mse_to_reference_average_velocity"]),
                "mean_cosine_similarity_to_reference_average_velocity": float(value["mean_cosine_similarity_to_reference_average_velocity"]),
                "mean_endpoint_displacement_error": float(value["mean_endpoint_displacement_error"]),
                "mean_hidden_energy_after_bootstrap_step": float(value["mean_hidden_energy_after_bootstrap_step"]),
                "mean_wall_time_seconds_per_eval": float(value["mean_wall_time_seconds_per_eval"]),
            }
            for key, value in offline_summaries.items()
        },
        "mean_std_val_accuracy_by_candidate": {
            key: {
                "mean": value["mean_val_accuracy"],
                "std": value["std_val_accuracy"],
            }
            for key, value in end_to_end_summaries.items()
        },
        "mean_std_test_accuracy_by_candidate": {
            key: {
                "mean": value["mean_test_accuracy"],
                "std": value["std_test_accuracy"],
            }
            for key, value in end_to_end_summaries.items()
        },
        "mean_gate_passing_epoch_count_by_candidate": {
            key: value["mean_gate_passing_epoch_count"] for key, value in end_to_end_summaries.items()
        },
        "mean_val_transported_final_energy_by_candidate": {
            key: value["mean_val_transported_final_energy"] for key, value in end_to_end_summaries.items()
        },
        "gap_to_canonical_slow_pc_by_candidate": {
            key: {"mean_test_accuracy_gap": value["mean_test_accuracy_gap_to_slow_pc"]}
            for key, value in end_to_end_summaries.items()
        },
        "mean_wall_clock_runtime_by_candidate": {
            key: {
                "mean_total_wall_time_seconds": value["mean_total_wall_time_seconds"],
                "std_total_wall_time_seconds": value["std_total_wall_time_seconds"],
            }
            for key, value in end_to_end_summaries.items()
        },
        "pairwise_delta_vs_current_corrective_default": {
            key: _pairwise_delta(default_summary, value) for key, value in end_to_end_summaries.items()
        },
        "best_end_to_end_candidate": best_summary,
        "does_bootstrap_target_fidelity_materially_improve_end_to_end_behavior": bool(material_gain),
        "should_corrective_default_change": bool(should_change_default),
        "bootstrap_target_fidelity_is_current_bottleneck": bool(bootstrap_is_bottleneck),
        "next_single_narrow_research_move": next_move,
        "offline_csv_path": "offline_target_fidelity.csv",
        "end_to_end_csv_path": "end_to_end_runs.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF2BootstrapFidelitySuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config, candidates),
        offline_rows=offline_csv_rows,
        end_to_end_rows=end_to_end_rows,
        summary=summary,
    )
