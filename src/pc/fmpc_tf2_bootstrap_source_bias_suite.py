from __future__ import annotations

import csv
import json
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator, Literal

import numpy as np

from . import fmpc_tf2 as fmpc_tf2_module
from .datasets import load_digits_split
from .energy import compute_cache
from .fmpc_tf1_flow import (
    bootstrap_average_velocity_target,
    build_tf1_context,
    hidden_energy_from_state,
    hidden_states_from_state,
    validate_tf1_time_pair,
)
from .fmpc_tf2 import (
    _active_onpolicy_mix_ratio,
    _active_theta_update_cadence,
    _lambda_id_for_epoch,
    _make_pc_model,
    _make_psi_network,
    _run_tf2_micro_step,
    _theta_micro_learning_rates,
    _theta_update_due_for_step,
    _theta_update_from_transported_state,
    _train_one_batch_tf2,
    build_tf2_corrective_transport_default_config,
    run_fmpc_tf2_experiment,
)
from .inference import run_inference
from .metrics import classification_accuracy
from .minibatch import iter_minibatches
from .state_io import flatten_hidden_states


BootstrapSourceFamily = Literal["local_field", "diagnostic_detached_slow_pc"]


@dataclass(frozen=True)
class FMPCTF2BootstrapSourceCandidate:
    """One bootstrap terminal-source family for the TF2 source-bias study."""

    key: str
    source_family: BootstrapSourceFamily
    diagnostic_only: bool
    slow_pc_steps: int | None = None


@dataclass
class FMPCTF2BootstrapSourceBiasSuiteConfig:
    """Offline-first source-bias study for the TF2 corrective transport default."""

    experiment_name: str = "fmpc_tf2_bootstrap_source_bias_suite"
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
    diagnostic_slow_pc_steps: tuple[int, ...] = (4, 8, 16)
    max_pruned_diagnostic_challengers: int = 1
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
class FMPCTF2BootstrapSourceBiasSuiteRunResult:
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
        raise FileNotFoundError(f"Required TF2 bootstrap-source reference artifact is missing: {path_obj}")
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


def _cosine_mean(a: np.ndarray, b: np.ndarray) -> float:
    a_array = np.asarray(a, dtype=np.float64)
    b_array = np.asarray(b, dtype=np.float64)
    numerator = np.sum(a_array * b_array, axis=1)
    denom = np.linalg.norm(a_array, axis=1) * np.linalg.norm(b_array, axis=1)
    denom = np.maximum(denom, 1e-12)
    return float(np.mean(numerator / denom))


def _candidate_registry(config: FMPCTF2BootstrapSourceBiasSuiteConfig) -> list[FMPCTF2BootstrapSourceCandidate]:
    candidates = [
        FMPCTF2BootstrapSourceCandidate(
            key="local_field_source",
            source_family="local_field",
            diagnostic_only=False,
            slow_pc_steps=None,
        )
    ]
    for slow_pc_steps in config.diagnostic_slow_pc_steps:
        candidates.append(
            FMPCTF2BootstrapSourceCandidate(
                key=f"diagnostic_slow_pc_k{int(slow_pc_steps)}",
                source_family="diagnostic_detached_slow_pc",
                diagnostic_only=True,
                slow_pc_steps=int(slow_pc_steps),
            )
        )
    return candidates


def _resolve_probe_epoch_indices(config: FMPCTF2BootstrapSourceBiasSuiteConfig) -> tuple[int, ...]:
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
    config: FMPCTF2BootstrapSourceBiasSuiteConfig,
    candidates: list[FMPCTF2BootstrapSourceCandidate],
) -> dict[str, Any]:
    return {
        "phase": "Phase TF2",
        "stage": "corrective_bootstrap_source_bias_suite",
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
            "bootstrap_integrator": "rk2",
            "bootstrap_substeps": 4,
            "checkpoint_selector": "gate_constrained_accuracy_then_val_accuracy",
        },
        "diagnostic_constraint": {
            "detached_slow_pc_source_is_diagnostic_only": True,
            "mainline_promotion_requires_teacher_free_surrogate": True,
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


def _local_field_target_and_endpoint(
    context: Any,
    z_t: np.ndarray,
    *,
    t_k: float,
    r_k: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    start = perf_counter()
    u_lf = bootstrap_average_velocity_target(
        context,
        z_t,
        t=t_k,
        r=r_k,
        integrator="rk2",
        substeps=4,
    )
    elapsed = perf_counter() - start
    endpoint = np.asarray(z_t, dtype=np.float64) + float(r_k) * np.asarray(u_lf, dtype=np.float64)
    return np.asarray(u_lf, dtype=np.float64), endpoint, float(elapsed)


def _detached_slow_pc_endpoint(
    context: Any,
    z_t: np.ndarray,
    *,
    slow_pc_steps: int,
    eta_x: float,
    backend: str,
) -> np.ndarray:
    states = hidden_states_from_state(context, z_t)
    inference_result = run_inference(
        states,
        context.layers,
        context.clamped_mask,
        eta_x=eta_x,
        steps=int(slow_pc_steps),
        backend=backend,
        record_trace=False,
        record_state_trajectory=False,
    )
    return flatten_hidden_states(inference_result.states, context.clamped_mask)


def _candidate_target_and_endpoint(
    context: Any,
    z_t: np.ndarray,
    *,
    t_k: float,
    r_k: float,
    candidate: FMPCTF2BootstrapSourceCandidate,
    eta_x: float,
    backend: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    if candidate.source_family == "local_field":
        return _local_field_target_and_endpoint(context, z_t, t_k=t_k, r_k=r_k)
    if candidate.source_family != "diagnostic_detached_slow_pc":
        raise ValueError(f"Unsupported source_family '{candidate.source_family}'.")
    if candidate.slow_pc_steps is None:
        raise ValueError("diagnostic_detached_slow_pc requires slow_pc_steps.")
    validate_tf1_time_pair(t_k, r_k)
    start = perf_counter()
    endpoint = _detached_slow_pc_endpoint(
        context,
        z_t,
        slow_pc_steps=int(candidate.slow_pc_steps),
        eta_x=float(eta_x),
        backend=backend,
    )
    elapsed = perf_counter() - start
    target = (np.asarray(endpoint, dtype=np.float64) - np.asarray(z_t, dtype=np.float64)) / float(r_k)
    return target, endpoint, float(elapsed)


def _endpoint_metrics(context: Any, endpoint_z: np.ndarray) -> dict[str, float]:
    states = hidden_states_from_state(context, endpoint_z)
    cache = compute_cache(states, context.layers)
    predictions = np.asarray(cache.predictions[-1], dtype=np.float64)
    targets = np.asarray(context.targets, dtype=np.float64)
    output_mse = float(np.mean((predictions - targets) ** 2))
    accuracy = classification_accuracy(predictions, targets)
    return {
        "hidden_energy": float(hidden_energy_from_state(context, endpoint_z)),
        "output_mse": float(output_mse),
        "accuracy": float(accuracy),
    }


def _offline_metric_row(
    *,
    probe_seed: int,
    probe_epoch: int,
    batch_index: int,
    step_index: int,
    t_k: float,
    r_k: float,
    candidate: FMPCTF2BootstrapSourceCandidate,
    mse_to_local_field: float,
    cosine_to_local_field: float,
    endpoint_hidden_energy: float,
    endpoint_output_mse: float,
    endpoint_accuracy: float,
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
        "source_family": candidate.source_family,
        "diagnostic_only": bool(candidate.diagnostic_only),
        "slow_pc_steps": "" if candidate.slow_pc_steps is None else int(candidate.slow_pc_steps),
        "mse_to_local_field_average_velocity": float(mse_to_local_field),
        "cosine_similarity_to_local_field_average_velocity": float(cosine_to_local_field),
        "endpoint_hidden_energy": float(endpoint_hidden_energy),
        "endpoint_output_mse": float(endpoint_output_mse),
        "endpoint_accuracy": float(endpoint_accuracy),
        "wall_time_seconds_per_eval": float(wall_time_seconds_per_eval),
    }


def _collect_offline_rows(
    config: FMPCTF2BootstrapSourceBiasSuiteConfig,
    candidates: list[FMPCTF2BootstrapSourceCandidate],
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

                    u_lf, endpoint_lf, wall_lf = _local_field_target_and_endpoint(
                        context,
                        z_lf,
                        t_k=t_k,
                        r_k=r_k,
                    )
                    local_metrics = _endpoint_metrics(context, endpoint_lf)
                    offline_rows.append(
                        _offline_metric_row(
                            probe_seed=seed,
                            probe_epoch=epoch_index,
                            batch_index=batch_index,
                            step_index=step_index,
                            t_k=t_k,
                            r_k=r_k,
                            candidate=candidates[0],
                            mse_to_local_field=0.0,
                            cosine_to_local_field=1.0,
                            endpoint_hidden_energy=local_metrics["hidden_energy"],
                            endpoint_output_mse=local_metrics["output_mse"],
                            endpoint_accuracy=local_metrics["accuracy"],
                            wall_time_seconds_per_eval=wall_lf,
                        )
                    )

                    for candidate in candidates[1:]:
                        u_candidate, endpoint_candidate, wall_time = _candidate_target_and_endpoint(
                            context,
                            z_lf,
                            t_k=t_k,
                            r_k=r_k,
                            candidate=candidate,
                            eta_x=float(run_config.eta_x),
                            backend=str(model.inference_backend),
                        )
                        diff = np.asarray(u_candidate, dtype=np.float64) - np.asarray(u_lf, dtype=np.float64)
                        mse = float(np.mean(diff**2))
                        cosine = _cosine_mean(u_candidate, u_lf)
                        metrics = _endpoint_metrics(context, endpoint_candidate)
                        offline_rows.append(
                            _offline_metric_row(
                                probe_seed=seed,
                                probe_epoch=epoch_index,
                                batch_index=batch_index,
                                step_index=step_index,
                                t_k=t_k,
                                r_k=r_k,
                                candidate=candidate,
                                mse_to_local_field=mse,
                                cosine_to_local_field=cosine,
                                endpoint_hidden_energy=metrics["hidden_energy"],
                                endpoint_output_mse=metrics["output_mse"],
                                endpoint_accuracy=metrics["accuracy"],
                                wall_time_seconds_per_eval=wall_time,
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


def _offline_summary(rows: list[dict[str, Any]], candidate: FMPCTF2BootstrapSourceCandidate) -> dict[str, Any]:
    candidate_rows = [row for row in rows if str(row["candidate_key"]) == candidate.key]
    if not candidate_rows:
        raise ValueError(f"No offline rows found for candidate '{candidate.key}'.")
    mse_values = [float(row["mse_to_local_field_average_velocity"]) for row in candidate_rows]
    cosine_values = [float(row["cosine_similarity_to_local_field_average_velocity"]) for row in candidate_rows]
    energy_values = [float(row["endpoint_hidden_energy"]) for row in candidate_rows]
    output_mse_values = [float(row["endpoint_output_mse"]) for row in candidate_rows]
    accuracy_values = [float(row["endpoint_accuracy"]) for row in candidate_rows]
    wall_values = [float(row["wall_time_seconds_per_eval"]) for row in candidate_rows]
    return {
        "candidate_key": candidate.key,
        "source_family": candidate.source_family,
        "diagnostic_only": bool(candidate.diagnostic_only),
        "slow_pc_steps": None if candidate.slow_pc_steps is None else int(candidate.slow_pc_steps),
        "num_probe_evaluations": int(len(candidate_rows)),
        "mean_mse_to_local_field_average_velocity": _mean(mse_values),
        "mean_cosine_similarity_to_local_field_average_velocity": _mean(cosine_values),
        "mean_endpoint_hidden_energy": _mean(energy_values),
        "mean_endpoint_output_mse": _mean(output_mse_values),
        "mean_endpoint_accuracy": _mean(accuracy_values),
        "mean_wall_time_seconds_per_eval": _mean(wall_values),
    }


def _pruned_candidates(
    candidates: list[FMPCTF2BootstrapSourceCandidate],
    offline_summaries: dict[str, dict[str, Any]],
    *,
    max_diagnostic: int,
) -> list[FMPCTF2BootstrapSourceCandidate]:
    local_field = next(candidate for candidate in candidates if candidate.source_family == "local_field")
    diagnostic = [candidate for candidate in candidates if candidate.source_family != "local_field"]
    ranked = sorted(
        diagnostic,
        key=lambda candidate: (
            float(offline_summaries[candidate.key]["mean_endpoint_output_mse"]),
            float(offline_summaries[candidate.key]["mean_endpoint_hidden_energy"]),
            -float(offline_summaries[candidate.key]["mean_endpoint_accuracy"]),
            float(offline_summaries[candidate.key]["mean_wall_time_seconds_per_eval"]),
        ),
    )
    return [local_field, *ranked[: int(max_diagnostic)]]


def _success_end_to_end_row(
    *,
    run_index: int,
    candidate: FMPCTF2BootstrapSourceCandidate,
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
        "source_family": candidate.source_family,
        "diagnostic_only": bool(candidate.diagnostic_only),
        "slow_pc_steps": "" if candidate.slow_pc_steps is None else int(candidate.slow_pc_steps),
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
    candidate: FMPCTF2BootstrapSourceCandidate,
    seed: int,
    error: Exception,
) -> dict[str, Any]:
    message = str(error)
    lowered = message.lower()
    return {
        "run_index": int(run_index),
        "candidate_key": candidate.key,
        "source_family": candidate.source_family,
        "diagnostic_only": bool(candidate.diagnostic_only),
        "slow_pc_steps": "" if candidate.slow_pc_steps is None else int(candidate.slow_pc_steps),
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


def _end_to_end_summary(
    rows: list[dict[str, Any]],
    candidate: FMPCTF2BootstrapSourceCandidate,
    *,
    slow_pc_ref: dict[int, dict[str, float]],
) -> dict[str, Any]:
    candidate_rows = [row for row in rows if str(row["candidate_key"]) == candidate.key]
    successes = [row for row in candidate_rows if str(row["run_status"]) == "completed"]
    payload: dict[str, Any] = {
        "candidate_key": candidate.key,
        "source_family": candidate.source_family,
        "diagnostic_only": bool(candidate.diagnostic_only),
        "slow_pc_steps": None if candidate.slow_pc_steps is None else int(candidate.slow_pc_steps),
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

    val_values = [float(row["val_accuracy"]) for row in successes]
    test_values = [float(row["test_accuracy"]) for row in successes]
    gate_values = [float(row["gate_passing_epoch_count"]) for row in successes]
    energy_values = [float(row["val_transported_final_energy"]) for row in successes]
    wall_values = [float(row["total_wall_time_seconds"]) for row in successes]
    gap_values = [
        float(row["test_accuracy"]) - float(slow_pc_ref[int(row["seed"])]["test_accuracy"])
        for row in successes
    ]
    payload.update(
        {
            "mean_val_accuracy": _mean(val_values),
            "std_val_accuracy": _std(val_values),
            "mean_test_accuracy": _mean(test_values),
            "std_test_accuracy": _std(test_values),
            "mean_gate_passing_epoch_count": _mean(gate_values),
            "mean_val_transported_final_energy": _mean(energy_values),
            "mean_total_wall_time_seconds": _mean(wall_values),
            "std_total_wall_time_seconds": _std(wall_values),
            "mean_test_accuracy_gap_to_slow_pc": _mean(gap_values),
        }
    )
    return payload


def _pairwise_delta(
    challenger_summary: dict[str, Any],
    default_summary: dict[str, Any],
) -> dict[str, Any]:
    if challenger_summary["mean_test_accuracy"] is None or default_summary["mean_test_accuracy"] is None:
        return {
            "mean_val_accuracy_delta": None,
            "mean_test_accuracy_delta": None,
            "mean_gate_passing_epoch_count_delta": None,
            "mean_val_transported_final_energy_delta": None,
            "mean_total_wall_time_seconds_delta": None,
            "mean_test_gap_to_slow_pc_delta": None,
        }
    return {
        "mean_val_accuracy_delta": float(challenger_summary["mean_val_accuracy"] - default_summary["mean_val_accuracy"]),
        "mean_test_accuracy_delta": float(challenger_summary["mean_test_accuracy"] - default_summary["mean_test_accuracy"]),
        "mean_gate_passing_epoch_count_delta": float(
            challenger_summary["mean_gate_passing_epoch_count"] - default_summary["mean_gate_passing_epoch_count"]
        ),
        "mean_val_transported_final_energy_delta": float(
            challenger_summary["mean_val_transported_final_energy"] - default_summary["mean_val_transported_final_energy"]
        ),
        "mean_total_wall_time_seconds_delta": float(
            challenger_summary["mean_total_wall_time_seconds"] - default_summary["mean_total_wall_time_seconds"]
        ),
        "mean_test_gap_to_slow_pc_delta": float(
            challenger_summary["mean_test_accuracy_gap_to_slow_pc"] - default_summary["mean_test_accuracy_gap_to_slow_pc"]
        ),
    }


def _diagnostic_target_fn(
    candidate: FMPCTF2BootstrapSourceCandidate,
    *,
    eta_x: float,
    backend: str,
):
    if candidate.source_family != "diagnostic_detached_slow_pc":
        raise ValueError("diagnostic target fn only supports detached slow-PC challengers.")
    if candidate.slow_pc_steps is None:
        raise ValueError("detached slow-PC challenger requires slow_pc_steps.")

    def _target(
        context: Any,
        z: np.ndarray,
        *,
        t: float,
        r: float,
        integrator: Literal["euler", "rk2"] = "rk2",
        substeps: int = 4,
    ) -> np.ndarray:
        del integrator, substeps
        validate_tf1_time_pair(t, r)
        endpoint = _detached_slow_pc_endpoint(
            context,
            np.asarray(z, dtype=np.float64),
            slow_pc_steps=int(candidate.slow_pc_steps),
            eta_x=float(eta_x),
            backend=backend,
        )
        return (endpoint - np.asarray(z, dtype=np.float64)) / float(r)

    return _target


@contextmanager
def _patched_bootstrap_source(
    candidate: FMPCTF2BootstrapSourceCandidate,
    *,
    eta_x: float,
    backend: str,
) -> Iterator[None]:
    if candidate.source_family == "local_field":
        yield
        return
    original = fmpc_tf2_module.bootstrap_average_velocity_target
    fmpc_tf2_module.bootstrap_average_velocity_target = _diagnostic_target_fn(
        candidate,
        eta_x=eta_x,
        backend=backend,
    )
    try:
        yield
    finally:
        fmpc_tf2_module.bootstrap_average_velocity_target = original


def run_fmpc_tf2_bootstrap_source_bias_suite(
    config: FMPCTF2BootstrapSourceBiasSuiteConfig,
) -> FMPCTF2BootstrapSourceBiasSuiteRunResult:
    candidates = _candidate_registry(config)
    run_dir = _prepare_run_dir(
        _resolve_run_dir(
            config.output_root,
            config.experiment_name,
            config.resolved_run_id(),
            config.output_layout,
        )
    )
    config_payload = _suite_config_payload(config, candidates)
    _write_json(run_dir / "config.json", config_payload)

    slow_pc_rows = _read_csv(config.slow_pc_reference_runs_path)
    slow_pc_ref = _load_slow_pc_reference_by_seed(
        slow_pc_rows,
        reference_name=config.slow_pc_reference_name,
        seeds=config.end_to_end_seeds,
    )

    offline_rows = _collect_offline_rows(config, candidates)
    _write_csv(run_dir / "offline_source_bias.csv", offline_rows)
    offline_summaries = {candidate.key: _offline_summary(offline_rows, candidate) for candidate in candidates}
    pruned_candidates = _pruned_candidates(
        candidates,
        offline_summaries,
        max_diagnostic=int(config.max_pruned_diagnostic_challengers),
    )

    end_to_end_rows: list[dict[str, Any]] = []
    run_index = 0
    for candidate in pruned_candidates:
        for seed in config.end_to_end_seeds:
            run_index += 1
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
            run_config.experiment_name = f"{config.experiment_name}/runs/{candidate.key}"
            run_config.output_root = str(config.output_root)
            run_config.output_layout = "run_id_subdir"
            run_config.run_id = f"seed_{int(seed)}"
            try:
                with _patched_bootstrap_source(
                    candidate,
                    eta_x=float(run_config.eta_x),
                    backend="pc_euler",
                ):
                    result = run_fmpc_tf2_experiment(run_config)
                end_to_end_rows.append(
                    _success_end_to_end_row(
                        run_index=run_index,
                        candidate=candidate,
                        seed=seed,
                        result=result,
                        run_dir=run_dir,
                    )
                )
            except Exception as error:
                end_to_end_rows.append(
                    _failure_end_to_end_row(
                        run_index=run_index,
                        candidate=candidate,
                        seed=seed,
                        error=error,
                    )
                )
    _write_csv(run_dir / "end_to_end_runs.csv", end_to_end_rows)

    end_to_end_summaries = {
        candidate.key: _end_to_end_summary(end_to_end_rows, candidate, slow_pc_ref=slow_pc_ref)
        for candidate in pruned_candidates
    }
    default_candidate = next(candidate for candidate in pruned_candidates if candidate.source_family == "local_field")
    default_summary = end_to_end_summaries[default_candidate.key]
    challenger = next((candidate for candidate in pruned_candidates if candidate.source_family != "local_field"), None)
    challenger_summary = None if challenger is None else end_to_end_summaries[challenger.key]
    pairwise_delta = (
        {}
        if challenger_summary is None
        else {challenger.key: _pairwise_delta(challenger_summary, default_summary)}
    )
    challenger_material = False
    if challenger_summary is not None and challenger_summary["mean_test_accuracy"] is not None:
        challenger_material = bool(
            float(challenger_summary["mean_test_accuracy"] - default_summary["mean_test_accuracy"])
            >= float(config.material_test_gain)
            and float(challenger_summary["mean_val_accuracy"] - default_summary["mean_val_accuracy"])
            >= float(config.material_val_gain)
            and int(challenger_summary["nan_or_inf_failure_count"]) <= int(default_summary["nan_or_inf_failure_count"])
        )

    best_candidate = max(
        end_to_end_summaries.values(),
        key=lambda payload: (
            float("-inf") if payload["mean_test_accuracy"] is None else float(payload["mean_test_accuracy"]),
            float("-inf") if payload["mean_val_accuracy"] is None else float(payload["mean_val_accuracy"]),
        ),
    )

    summary = {
        "phase": "Phase TF2",
        "stage": "corrective_bootstrap_source_bias_suite",
        "offline_source_bias_by_candidate": offline_summaries,
        "pruned_candidates_for_end_to_end": [
            {
                "candidate_key": candidate.key,
                "source_family": candidate.source_family,
                "diagnostic_only": bool(candidate.diagnostic_only),
                "slow_pc_steps": None if candidate.slow_pc_steps is None else int(candidate.slow_pc_steps),
            }
            for candidate in pruned_candidates
        ],
        "mean_std_val_accuracy_by_candidate": {
            key: {"mean": value["mean_val_accuracy"], "std": value["std_val_accuracy"]}
            for key, value in end_to_end_summaries.items()
        },
        "mean_std_test_accuracy_by_candidate": {
            key: {"mean": value["mean_test_accuracy"], "std": value["std_test_accuracy"]}
            for key, value in end_to_end_summaries.items()
        },
        "mean_gate_passing_epoch_count_by_candidate": {
            key: value["mean_gate_passing_epoch_count"]
            for key, value in end_to_end_summaries.items()
        },
        "mean_val_transported_final_energy_by_candidate": {
            key: value["mean_val_transported_final_energy"]
            for key, value in end_to_end_summaries.items()
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
        "pairwise_delta_vs_current_corrective_default": pairwise_delta,
        "best_end_to_end_candidate": best_candidate,
        "is_bootstrap_target_bottlenecked_by_terminal_source_bias": bool(challenger_material),
        "does_detached_slow_pc_source_materially_beat_local_field_source": bool(challenger_material),
        "current_mainline_safe_result": "keep tf2_corrective_transport_default with the local-field bootstrap source",
        "diagnostic_only_finding": (
            None
            if challenger is None
            else {
                "challenger_key": challenger.key,
                "source_family": challenger.source_family,
                "diagnostic_only": True,
                "slow_pc_steps": int(challenger.slow_pc_steps) if challenger.slow_pc_steps is not None else None,
            }
        ),
        "next_teacher_free_surrogate_if_source_bias_wins": (
            "explore a teacher-free surrogate that predicts a stronger detached endpoint correction "
            "from current state without regressing directly to slow-PC fixed-point targets"
        ),
        "next_single_narrow_research_move": (
            "if source bias is not the limiter, target psi-side transport expressivity under the fixed "
            "teacher-free local-field source rather than changing target construction again"
        ),
        "offline_csv_path": "offline_source_bias.csv",
        "end_to_end_csv_path": "end_to_end_runs.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF2BootstrapSourceBiasSuiteRunResult(
        run_dir=run_dir,
        config=config_payload,
        offline_rows=offline_rows,
        end_to_end_rows=end_to_end_rows,
        summary=summary,
    )
