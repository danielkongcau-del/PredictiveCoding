from __future__ import annotations

import csv
import json
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np

from . import fmpc_tf2 as fmpc_tf2_module
from ..datasets import load_digits_split
from ..energy import compute_cache
from ..transport_core_v1.fmpc_tf1_flow import build_tf1_context, hidden_energy_from_state, hidden_states_from_state
from .fmpc_tf2 import (
    FMPCTF2Config,
    FMPCTF2RunResult,
    _evaluate_slow_pc_accuracy,
    _restore_mlp_parameters,
    _restore_pc_parameters,
    _single_source_supervision,
    build_tf2_corrective_transport_default_config,
    run_fmpc_tf2_experiment,
)
from ..metrics import classification_accuracy

ThetaMode = Literal["live_theta", "frozen_theta"]
SelectorName = Literal["canonical_selector", "oracle_transported_energy", "oracle_hybrid_loss"]


@dataclass(frozen=True)
class FMPCTF2DownstreamCandidate:
    key: str
    psi_family: Literal["baseline_plain", "residualized_local_field"]
    time_encoding_variant: Literal["raw", "poly_rt2"]
    notes: str


@dataclass
class FMPCTF2DownstreamCouplingSuiteConfig:
    """Diagnostic-only downstream coupling suite for the TF2 corrective default."""

    experiment_name: str = "fmpc_tf2_downstream_coupling_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    seeds: tuple[int, ...] = (0, 1, 2)
    epochs: int = 60
    batch_size: int = 128
    eval_steps: int = 15
    layer_dims: tuple[int, ...] = (64, 64, 10)
    drift_probe_split: Literal["val"] = "val"
    material_test_gain: float = 0.005
    theta_rescue_test_gain: float = 0.01

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"


@dataclass
class FMPCTF2DownstreamCouplingSuiteRunResult:
    run_dir: Path
    config: dict[str, Any]
    run_rows: list[dict[str, Any]]
    drift_rows: list[dict[str, Any]]
    selector_epoch_rows: list[dict[str, Any]]
    selector_rows: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass
class _ExecutedRun:
    theta_mode: ThetaMode
    candidate: FMPCTF2DownstreamCandidate
    seed: int
    config: FMPCTF2Config
    result: FMPCTF2RunResult


@dataclass(frozen=True)
class _StepState:
    step_index: int
    t_k: float
    r_k: float
    dt: float
    z_t: np.ndarray


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


def _candidate_registry() -> list[FMPCTF2DownstreamCandidate]:
    return [
        FMPCTF2DownstreamCandidate(
            key="baseline_plain_raw",
            psi_family="baseline_plain",
            time_encoding_variant="raw",
            notes="current corrective default baseline",
        ),
        FMPCTF2DownstreamCandidate(
            key="residualized_local_field_poly_rt2",
            psi_family="residualized_local_field",
            time_encoding_variant="poly_rt2",
            notes="offline-better challenger from the psi-expressivity suite",
        ),
    ]


def _suite_config_payload(
    config: FMPCTF2DownstreamCouplingSuiteConfig,
    candidates: list[FMPCTF2DownstreamCandidate],
) -> dict[str, Any]:
    return {
        "phase": "Phase Incremental Bridge",
        "stage": "corrective_downstream_coupling_suite",
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
        },
        "diagnostic_only": True,
        "material_test_gain": float(config.material_test_gain),
        "theta_rescue_test_gain": float(config.theta_rescue_test_gain),
    }


def _build_candidate_config(
    candidate: FMPCTF2DownstreamCandidate,
    *,
    seed: int,
    suite_config: FMPCTF2DownstreamCouplingSuiteConfig,
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


@contextmanager
def _patched_frozen_theta_updates() -> Iterator[None]:
    original = fmpc_tf2_module._theta_update_from_transported_state

    def _frozen_theta_update(model: Any, context: Any, transported_z: np.ndarray, *, eta_w: float, eta_b: float) -> float:
        del model, eta_w, eta_b
        return float(hidden_energy_from_state(context, transported_z))

    fmpc_tf2_module._theta_update_from_transported_state = _frozen_theta_update
    try:
        yield
    finally:
        fmpc_tf2_module._theta_update_from_transported_state = original


def _run_one_candidate(
    run_dir: Path,
    suite_config: FMPCTF2DownstreamCouplingSuiteConfig,
    *,
    theta_mode: ThetaMode,
    candidate: FMPCTF2DownstreamCandidate,
    seed: int,
) -> _ExecutedRun:
    config = _build_candidate_config(candidate, seed=seed, suite_config=suite_config)
    config.output_root = str(run_dir)
    config.experiment_name = f"runs/{theta_mode}/{candidate.key}"
    config.output_layout = "run_id_subdir"
    config.run_id = f"seed_{int(seed)}"
    if theta_mode == "frozen_theta":
        with _patched_frozen_theta_updates():
            result = run_fmpc_tf2_experiment(config)
    else:
        result = run_fmpc_tf2_experiment(config)
    return _ExecutedRun(theta_mode=theta_mode, candidate=candidate, seed=int(seed), config=config, result=result)


def _success_run_row(
    *,
    executed: _ExecutedRun,
    run_dir: Path,
) -> dict[str, Any]:
    summary = executed.result.summary
    timing = dict(summary.get("timing", {}))
    return {
        "theta_mode": executed.theta_mode,
        "candidate_key": executed.candidate.key,
        "psi_family": executed.candidate.psi_family,
        "time_encoding_variant": executed.candidate.time_encoding_variant,
        "seed": int(executed.seed),
        "val_accuracy": float(summary["val_accuracy"]),
        "test_accuracy": float(summary["test_accuracy"]),
        "gate_passing_epoch_count": int(summary["gate_passing_epoch_count"]),
        "val_transported_final_energy": float(summary["val_transported_final_energy"]),
        "selected_epoch": int(summary["best_epoch"]),
        "selected_epoch_passes_gate": bool(summary["selected_epoch_passes_gate"]),
        "selector_fallback_used": bool(summary["selector_fallback_used"]),
        "total_wall_time_seconds": float(
            timing.get("train_wall_time_seconds", 0.0)
            + timing.get("final_evaluation_wall_time_seconds", 0.0)
        ),
        "run_status": "success",
        "run_summary_path": _relative_posix(run_dir, executed.result.run_dir / "summary.json"),
    }


def _load_split(config: FMPCTF2Config) -> Any:
    return load_digits_split(
        split_seed=config.data_seed,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
    )


def _gate_pass(epoch_row: dict[str, Any]) -> bool:
    return bool(
        float(epoch_row["val_transported_final_energy"]) < float(epoch_row["val_identity_final_energy"])
        and float(epoch_row["val_transported_final_energy"])
        <= float(epoch_row["val_local_field_only_final_energy"])
        and float(epoch_row["val_accuracy"]) > float(epoch_row["val_baseline_accuracy"])
    )


def _restore_epoch_snapshot(executed: _ExecutedRun, epoch_number: int) -> None:
    snapshots = executed.result.epoch_snapshots or []
    snapshot = next(snapshot for snapshot in snapshots if int(snapshot.epoch) == int(epoch_number))
    if executed.result.model is None or executed.result.psi_network is None:
        raise ValueError("TF2 run result must carry model and psi snapshots for diagnostics.")
    _restore_pc_parameters(executed.result.model, snapshot.model_snapshot)
    _restore_mlp_parameters(executed.result.psi_network, snapshot.psi_snapshot)


def _epoch_analysis_rows(
    executed: _ExecutedRun,
    *,
    run_dir: Path,
) -> list[dict[str, Any]]:
    split = _load_split(executed.config)
    rows: list[dict[str, Any]] = []
    for epoch_row in executed.result.epoch_metrics:
        epoch_number = int(epoch_row["epoch"])
        _restore_epoch_snapshot(executed, epoch_number)
        if executed.result.model is None:
            raise ValueError("TF2 run result model must be present.")
        _, test_accuracy = _evaluate_slow_pc_accuracy(executed.result.model, split.x_test, split.y_test)
        rows.append(
            {
                "theta_mode": executed.theta_mode,
                "candidate_key": executed.candidate.key,
                "seed": int(executed.seed),
                "epoch": epoch_number,
                "train_loss": float(epoch_row["train_loss"]),
                "val_accuracy": float(epoch_row["val_accuracy"]),
                "test_accuracy": float(test_accuracy),
                "gate_pass": bool(_gate_pass(epoch_row)),
                "val_transported_final_energy": float(epoch_row["val_transported_final_energy"]),
                "run_summary_path": _relative_posix(run_dir, executed.result.run_dir / "summary.json"),
            }
        )
    return rows


def _selector_choice(
    selector_name: SelectorName,
    epoch_rows: list[dict[str, Any]],
    *,
    canonical_epoch: int,
) -> dict[str, Any]:
    if selector_name == "canonical_selector":
        row = next(row for row in epoch_rows if int(row["epoch"]) == int(canonical_epoch))
    elif selector_name == "oracle_transported_energy":
        row = min(
            epoch_rows,
            key=lambda payload: (
                float(payload["val_transported_final_energy"]),
                -float(payload["val_accuracy"]),
                int(payload["epoch"]),
            ),
        )
    elif selector_name == "oracle_hybrid_loss":
        row = min(
            epoch_rows,
            key=lambda payload: (
                float(payload["train_loss"]),
                -float(payload["val_accuracy"]),
                int(payload["epoch"]),
            ),
        )
    else:
        raise ValueError(f"Unsupported selector '{selector_name}'.")
    return {
        "selector_name": selector_name,
        "selected_epoch": int(row["epoch"]),
        "val_accuracy": float(row["val_accuracy"]),
        "test_accuracy": float(row["test_accuracy"]),
        "gate_pass": bool(row["gate_pass"]),
        "val_transported_final_energy": float(row["val_transported_final_energy"]),
        "train_loss": float(row["train_loss"]),
    }


def _selector_rows(
    executed: _ExecutedRun,
    epoch_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    canonical_epoch = int(executed.result.summary["best_epoch"])
    rows: list[dict[str, Any]] = []
    for selector_name in (
        "canonical_selector",
        "oracle_transported_energy",
        "oracle_hybrid_loss",
    ):
        choice = _selector_choice(
            selector_name,
            epoch_rows,
            canonical_epoch=canonical_epoch,
        )
        rows.append(
            {
                "theta_mode": executed.theta_mode,
                "candidate_key": executed.candidate.key,
                "seed": int(executed.seed),
                **choice,
            }
        )
    return rows


def _rollout_states(
    context: Any,
    psi_network: Any,
    config: FMPCTF2Config,
) -> tuple[list[_StepState], np.ndarray]:
    knots = np.linspace(0.0, 1.0, int(config.micro_steps) + 1, dtype=np.float64)
    z_t = context.z0.copy()
    states: list[_StepState] = []
    for step_index in range(int(config.micro_steps)):
        t_k = float(knots[step_index])
        r_k = 1.0 - t_k
        dt = float(knots[step_index + 1] - knots[step_index])
        states.append(
            _StepState(
                step_index=int(step_index),
                t_k=t_k,
                r_k=r_k,
                dt=dt,
                z_t=z_t.copy(),
            )
        )
        _, _, _, velocity = _single_source_supervision(
            context,
            psi_network,
            config,
            z_t,
            t_k=t_k,
            r_k=r_k,
        )
        z_t = np.asarray(z_t, dtype=np.float64) + float(dt) * np.asarray(velocity, dtype=np.float64)
    return states, z_t


def _output_metrics(context: Any, z_state: np.ndarray) -> dict[str, float]:
    states = hidden_states_from_state(context, z_state)
    cache = compute_cache(states, context.layers)
    predictions = np.asarray(cache.predictions[-1], dtype=np.float64)
    targets = np.asarray(context.targets, dtype=np.float64)
    return {
        "hidden_energy": float(hidden_energy_from_state(context, z_state)),
        "output_mse": float(np.mean((predictions - targets) ** 2)),
        "output_accuracy": float(classification_accuracy(predictions, targets)),
    }


def _state_dataset_metrics(
    context: Any,
    psi_network: Any,
    config: FMPCTF2Config,
    state_entries: list[_StepState],
    *,
    final_rollout_z: np.ndarray,
) -> dict[str, float]:
    boot_losses: list[float] = []
    identity_losses: list[float] = []
    identity_residuals: list[float] = []
    hybrid_losses: list[float] = []
    step_hidden_energies: list[float] = []
    step_output_mse: list[float] = []
    step_output_accuracy: list[float] = []
    lambda_eval = float(config.identity_loss_weight)
    for entry in state_entries:
        _, u_boot, u_identity, u_pred = _single_source_supervision(
            context,
            psi_network,
            config,
            entry.z_t,
            t_k=entry.t_k,
            r_k=entry.r_k,
        )
        boot_loss = float(np.mean((u_pred - u_boot) ** 2))
        identity_loss = float(np.mean((u_pred - u_identity) ** 2))
        identity_residual = float(np.mean(np.linalg.norm(u_pred - u_identity, axis=1)))
        next_z = np.asarray(entry.z_t, dtype=np.float64) + float(entry.dt) * np.asarray(u_pred, dtype=np.float64)
        next_metrics = _output_metrics(context, next_z)
        boot_losses.append(boot_loss)
        identity_losses.append(identity_loss)
        identity_residuals.append(identity_residual)
        hybrid_losses.append(boot_loss + lambda_eval * identity_loss)
        step_hidden_energies.append(next_metrics["hidden_energy"])
        step_output_mse.append(next_metrics["output_mse"])
        step_output_accuracy.append(next_metrics["output_accuracy"])
    final_metrics = _output_metrics(context, final_rollout_z)
    return {
        "bootstrap_target_mse": _mean(boot_losses),
        "identity_target_mse": _mean(identity_losses),
        "identity_residual_error": _mean(identity_residuals),
        "hybrid_loss": _mean(hybrid_losses),
        "mean_step_transported_hidden_energy": _mean(step_hidden_energies),
        "mean_step_output_mse": _mean(step_output_mse),
        "mean_step_output_accuracy": _mean(step_output_accuracy),
        "rollout_final_hidden_energy": float(final_metrics["hidden_energy"]),
        "rollout_final_output_mse": float(final_metrics["output_mse"]),
        "rollout_final_output_accuracy": float(final_metrics["output_accuracy"]),
    }


def _drift_rows_for_seed(
    baseline_run: _ExecutedRun,
    challenger_run: _ExecutedRun,
) -> list[dict[str, Any]]:
    if baseline_run.result.model is None or baseline_run.result.psi_network is None:
        raise ValueError("Baseline diagnostic run must expose model and psi network.")
    if challenger_run.result.model is None or challenger_run.result.psi_network is None:
        raise ValueError("Challenger diagnostic run must expose model and psi network.")
    _restore_epoch_snapshot(baseline_run, int(baseline_run.result.summary["best_epoch"]))
    _restore_epoch_snapshot(challenger_run, int(challenger_run.result.summary["best_epoch"]))
    split = _load_split(baseline_run.config)
    x_batch = split.x_val[: baseline_run.config.batch_size]
    y_batch = split.y_val[: baseline_run.config.batch_size]
    baseline_context = build_tf1_context(baseline_run.result.model, x_batch, y_batch)
    challenger_context = build_tf1_context(challenger_run.result.model, x_batch, y_batch)
    baseline_replay_states, baseline_final_z = _rollout_states(
        baseline_context,
        baseline_run.result.psi_network,
        baseline_run.config,
    )
    baseline_self_states, baseline_self_final_z = baseline_replay_states, baseline_final_z
    challenger_self_states, challenger_self_final_z = _rollout_states(
        challenger_context,
        challenger_run.result.psi_network,
        challenger_run.config,
    )
    diagnostics = [
        ("baseline_plain_raw", "replay_states", baseline_run, baseline_context, baseline_replay_states, baseline_self_final_z),
        ("baseline_plain_raw", "self_induced_states", baseline_run, baseline_context, baseline_self_states, baseline_self_final_z),
        (
            "residualized_local_field_poly_rt2",
            "replay_states",
            challenger_run,
            challenger_context,
            baseline_replay_states,
            challenger_self_final_z,
        ),
        (
            "residualized_local_field_poly_rt2",
            "self_induced_states",
            challenger_run,
            challenger_context,
            challenger_self_states,
            challenger_self_final_z,
        ),
    ]
    rows: list[dict[str, Any]] = []
    for candidate_key, state_mode, executed, context, states, final_z in diagnostics:
        metrics = _state_dataset_metrics(
            context,
            executed.result.psi_network,
            executed.config,
            states,
            final_rollout_z=final_z,
        )
        rows.append(
            {
                "seed": int(executed.seed),
                "theta_mode": executed.theta_mode,
                "candidate_key": candidate_key,
                "state_mode": state_mode,
                **metrics,
            }
        )
    return rows


def _aggregate_run_mode(
    run_rows: list[dict[str, Any]],
    *,
    theta_mode: ThetaMode,
    candidate_key: str,
) -> dict[str, Any]:
    relevant = [
        row
        for row in run_rows
        if row["theta_mode"] == theta_mode and row["candidate_key"] == candidate_key and row["run_status"] == "success"
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


def _selector_summary(
    selector_rows: list[dict[str, Any]],
    *,
    theta_mode: ThetaMode,
    candidate_key: str,
    selector_name: SelectorName,
) -> dict[str, Any]:
    relevant = [
        row
        for row in selector_rows
        if row["theta_mode"] == theta_mode
        and row["candidate_key"] == candidate_key
        and row["selector_name"] == selector_name
    ]
    if not relevant:
        return {"mean_val_accuracy": None, "mean_test_accuracy": None, "mean_gate_pass_rate": None}
    return {
        "mean_val_accuracy": _mean([float(row["val_accuracy"]) for row in relevant]),
        "mean_test_accuracy": _mean([float(row["test_accuracy"]) for row in relevant]),
        "mean_gate_pass_rate": _mean([1.0 if bool(row["gate_pass"]) else 0.0 for row in relevant]),
    }


def run_fmpc_tf2_downstream_coupling_suite(
    config: FMPCTF2DownstreamCouplingSuiteConfig,
) -> FMPCTF2DownstreamCouplingSuiteRunResult:
    candidates = _candidate_registry()
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

    executed_runs: list[_ExecutedRun] = []
    run_rows: list[dict[str, Any]] = []
    selector_epoch_rows: list[dict[str, Any]] = []
    selector_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []

    for theta_mode in ("live_theta", "frozen_theta"):
        for candidate in candidates:
            for seed in config.seeds:
                executed = _run_one_candidate(
                    run_dir,
                    config,
                    theta_mode=theta_mode,
                    candidate=candidate,
                    seed=int(seed),
                )
                executed_runs.append(executed)
                run_rows.append(_success_run_row(executed=executed, run_dir=run_dir))
                epoch_rows = _epoch_analysis_rows(executed, run_dir=run_dir)
                selector_epoch_rows.extend(epoch_rows)
                selector_rows.extend(_selector_rows(executed, epoch_rows))

    frozen_by_seed = {
        (run.candidate.key, int(run.seed)): run
        for run in executed_runs
        if run.theta_mode == "frozen_theta"
    }
    for seed in config.seeds:
        drift_rows.extend(
            _drift_rows_for_seed(
                frozen_by_seed[("baseline_plain_raw", int(seed))],
                frozen_by_seed[("residualized_local_field_poly_rt2", int(seed))],
            )
        )

    _write_csv(run_dir / "end_to_end_runs.csv", run_rows)
    _write_csv(run_dir / "selector_epoch_metrics.csv", selector_epoch_rows)
    _write_csv(run_dir / "selector_reselection.csv", selector_rows)
    _write_csv(run_dir / "replay_vs_self_drift.csv", drift_rows)

    live_baseline = _aggregate_run_mode(run_rows, theta_mode="live_theta", candidate_key="baseline_plain_raw")
    live_challenger = _aggregate_run_mode(
        run_rows,
        theta_mode="live_theta",
        candidate_key="residualized_local_field_poly_rt2",
    )
    frozen_baseline = _aggregate_run_mode(run_rows, theta_mode="frozen_theta", candidate_key="baseline_plain_raw")
    frozen_challenger = _aggregate_run_mode(
        run_rows,
        theta_mode="frozen_theta",
        candidate_key="residualized_local_field_poly_rt2",
    )

    drift_summary = {
        key: {
            mode: {
                metric: _mean(
                    [
                        float(row[metric])
                        for row in drift_rows
                        if row["candidate_key"] == key and row["state_mode"] == mode
                    ]
                )
                for metric in (
                    "bootstrap_target_mse",
                    "identity_target_mse",
                    "identity_residual_error",
                    "hybrid_loss",
                    "mean_step_transported_hidden_energy",
                    "mean_step_output_mse",
                    "mean_step_output_accuracy",
                    "rollout_final_hidden_energy",
                    "rollout_final_output_mse",
                    "rollout_final_output_accuracy",
                )
            }
            for mode in ("replay_states", "self_induced_states")
        }
        for key in ("baseline_plain_raw", "residualized_local_field_poly_rt2")
    }

    challenger_replay_win = bool(
        float(drift_summary["residualized_local_field_poly_rt2"]["replay_states"]["hybrid_loss"])
        < float(drift_summary["baseline_plain_raw"]["replay_states"]["hybrid_loss"])
        and float(drift_summary["residualized_local_field_poly_rt2"]["replay_states"]["identity_residual_error"])
        <= float(drift_summary["baseline_plain_raw"]["replay_states"]["identity_residual_error"])
    )
    challenger_self_loss = bool(
        float(drift_summary["residualized_local_field_poly_rt2"]["self_induced_states"]["hybrid_loss"])
        > float(drift_summary["baseline_plain_raw"]["self_induced_states"]["hybrid_loss"])
        and float(
            drift_summary["residualized_local_field_poly_rt2"]["self_induced_states"]["rollout_final_output_accuracy"]
        )
        <= float(drift_summary["baseline_plain_raw"]["self_induced_states"]["rollout_final_output_accuracy"])
    )
    rollout_state_distribution_drift_present = bool(challenger_replay_win and challenger_self_loss)

    live_test_gap = float(live_challenger["mean_test_accuracy"] - live_baseline["mean_test_accuracy"])
    frozen_test_gap = float(frozen_challenger["mean_test_accuracy"] - frozen_baseline["mean_test_accuracy"])
    live_val_gap = float(live_challenger["mean_val_accuracy"] - live_baseline["mean_val_accuracy"])
    frozen_val_gap = float(frozen_challenger["mean_val_accuracy"] - frozen_baseline["mean_val_accuracy"])
    moving_target_theta_coupling_present = bool(
        float(frozen_challenger["mean_test_accuracy"] - live_challenger["mean_test_accuracy"])
        >= float(config.theta_rescue_test_gain)
        and float(frozen_test_gap - live_test_gap) >= float(config.theta_rescue_test_gain)
        and float(frozen_val_gap) >= float(live_val_gap)
    )

    selector_summary = {
        selector_name: {
            candidate_key: _selector_summary(
                selector_rows,
                theta_mode="live_theta",
                candidate_key=candidate_key,
                selector_name=selector_name,
            )
            for candidate_key in ("baseline_plain_raw", "residualized_local_field_poly_rt2")
        }
        for selector_name in ("canonical_selector", "oracle_transported_energy", "oracle_hybrid_loss")
    }
    challenger_oracle_energy_gain = float(
        selector_summary["oracle_transported_energy"]["residualized_local_field_poly_rt2"]["mean_test_accuracy"]
        - selector_summary["canonical_selector"]["residualized_local_field_poly_rt2"]["mean_test_accuracy"]
    )
    selector_sensitivity_main_limiter = bool(
        challenger_oracle_energy_gain >= float(config.material_test_gain)
        and float(
            selector_summary["oracle_transported_energy"]["residualized_local_field_poly_rt2"]["mean_test_accuracy"]
            - selector_summary["canonical_selector"]["baseline_plain_raw"]["mean_test_accuracy"]
        )
        >= -float(config.material_test_gain)
    )
    selector_is_not_main_limiter = bool(not selector_sensitivity_main_limiter)

    if moving_target_theta_coupling_present:
        dominant_bottleneck = "moving_target_theta_state_live_target_coupling"
        next_move = "run one narrow target-lag diagnostic that keeps theta live but trains psi against one-batch-stale teacher-free targets"
    elif rollout_state_distribution_drift_present:
        dominant_bottleneck = "rollout_state_distribution_drift"
        next_move = "run one narrow rollout-stability pass that measures candidate performance under short replay-refresh schedules without changing target semantics"
    elif selector_sensitivity_main_limiter:
        dominant_bottleneck = "selector_sensitivity"
        next_move = "run one narrow selector-analysis confirmation pass before any semantic changes"
    else:
        dominant_bottleneck = "structural_downstream_coupling_not_yet_isolated"
        next_move = "run one narrow target-lag coupling diagnostic to separate live-target timing drift from residual rollout mismatch"

    summary = {
        "phase": "Phase Incremental Bridge",
        "stage": "corrective_downstream_coupling_suite",
        "live_theta_end_to_end": {
            "baseline_plain_raw": live_baseline,
            "residualized_local_field_poly_rt2": live_challenger,
        },
        "frozen_theta_end_to_end": {
            "baseline_plain_raw": frozen_baseline,
            "residualized_local_field_poly_rt2": frozen_challenger,
        },
        "live_vs_frozen_pairwise_gap": {
            "live_val_gap_challenger_minus_baseline": float(live_val_gap),
            "live_test_gap_challenger_minus_baseline": float(live_test_gap),
            "frozen_val_gap_challenger_minus_baseline": float(frozen_val_gap),
            "frozen_test_gap_challenger_minus_baseline": float(frozen_test_gap),
        },
        "replay_vs_self_drift_summary": drift_summary,
        "selector_reselection_summary_live_theta": selector_summary,
        "is_rollout_state_distribution_drift_present": bool(rollout_state_distribution_drift_present),
        "is_moving_target_theta_coupling_present": bool(moving_target_theta_coupling_present),
        "is_selector_sensitivity_the_main_limiter": bool(selector_sensitivity_main_limiter),
        "oracle_reselection_still_fails_to_rescue_challenger": bool(selector_is_not_main_limiter),
        "dominant_bottleneck": dominant_bottleneck,
        "next_single_narrow_move": next_move,
        "artifacts": {
            "run_rows_csv": "end_to_end_runs.csv",
            "selector_epoch_metrics_csv": "selector_epoch_metrics.csv",
            "selector_reselection_csv": "selector_reselection.csv",
            "replay_vs_self_drift_csv": "replay_vs_self_drift.csv",
        },
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF2DownstreamCouplingSuiteRunResult(
        run_dir=run_dir,
        config=config_payload,
        run_rows=run_rows,
        drift_rows=drift_rows,
        selector_epoch_rows=selector_epoch_rows,
        selector_rows=selector_rows,
        summary=summary,
    )
