from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..datasets import load_digits_split
from ..stage_03_transport_core_v1.fmpc_tf1_flow import (
    build_tf1_context,
    hidden_energy_from_state,
    hidden_local_flow,
    rollout_hidden_transport,
)
from .fmpc_ef_exploratory_probe import (
    FMPCEFExploratoryProbeConfig,
    Stage05ResidualCoreNetworks,
    _learned_velocity_fn,
    _predict_residual_from_inputs,
    _residual_core_inputs_for_state,
    build_corrected_residual_identity_target,
    build_fmpc_ef_exploratory_probe_config,
    run_fmpc_ef_exploratory_probe,
)


OutputLayout = Literal["single_dir", "run_id_subdir"]
Stage05V2DiagnosisLabel = Literal[
    "likely_undertrained",
    "state_branch_underutilized",
    "configured_step_rollout_accumulation_is_primary_gap",
    "selection_pressure_misaligned_with_report_accuracy",
]


@dataclass
class Stage05V2DiagnosticsConfig:
    """Narrow diagnostics for the current Stage 05 v2 exploratory reference."""

    experiment_name: str = "stage05_v2_diagnostics"
    output_root: str | Path = "outputs/stage_05_ef_core_probe"
    run_id: str | None = None
    output_layout: OutputLayout = "single_dir"
    source_comparison_dir: str | Path = (
        "outputs/stage_05_ef_core_probe/frozen_bridge_vs_two_branch_corrected_core_comparison"
    )
    seeds: tuple[int, ...] = (0, 1, 2)
    near_final_window: int = 3
    selection_accuracy_gap_threshold: float = 0.05
    state_branch_material_ratio_threshold: float = 0.25
    state_branch_weak_ratio_threshold: float = 0.05
    rollout_extra_gain_fraction_threshold: float = 0.20
    reproduction_output_subdir: str = "reproduced_stage05_v2"

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("seeds must contain at least one seed.")
        if self.near_final_window < 2:
            raise ValueError("near_final_window must be at least 2.")
        if self.selection_accuracy_gap_threshold < 0.0:
            raise ValueError("selection_accuracy_gap_threshold must be non-negative.")
        if self.state_branch_weak_ratio_threshold < 0.0:
            raise ValueError("state_branch_weak_ratio_threshold must be non-negative.")
        if self.state_branch_material_ratio_threshold < self.state_branch_weak_ratio_threshold:
            raise ValueError(
                "state_branch_material_ratio_threshold must be >= state_branch_weak_ratio_threshold."
            )
        if self.rollout_extra_gain_fraction_threshold < 0.0:
            raise ValueError("rollout_extra_gain_fraction_threshold must be non-negative.")

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_{self.seeds[0]}"


@dataclass
class Stage05V2DiagnosticsRunResult:
    run_dir: Path
    config: dict[str, Any]
    summary: dict[str, Any]


def _resolve_run_dir(
    output_root: str | Path,
    experiment_name: str,
    run_id: str,
    output_layout: OutputLayout,
) -> Path:
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
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_epoch_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric_fields = {
        "epoch",
        "lambda_id",
        "train_total_loss",
        "train_boot_loss",
        "train_identity_loss",
        "train_transported_final_energy",
        "val_one_step_transported_final_energy",
        "val_one_step_energy_delta_vs_identity",
        "val_one_step_fixed_point_residual_delta_vs_identity",
        "val_configured_transported_final_energy",
        "val_configured_energy_delta_vs_identity",
        "val_configured_fixed_point_residual_delta_vs_identity",
        "val_accuracy",
        "val_output_mse",
    }
    parsed: list[dict[str, Any]] = []
    for row in rows:
        payload: dict[str, Any] = {}
        for key, value in row.items():
            if key == "stage":
                payload[key] = value
            elif key in numeric_fields:
                payload[key] = float(value)
            else:
                payload[key] = value
        payload["epoch"] = int(payload["epoch"])
        parsed.append(payload)
    return parsed


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value.")
    return float(sum(values) / float(len(values)))


def _std(values: list[float]) -> float:
    if not values:
        raise ValueError("std requires at least one value.")
    mean_value = _mean(values)
    variance = sum((float(value) - mean_value) ** 2 for value in values) / float(len(values))
    return float(variance**0.5)


def _rms(array: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(array, dtype=np.float64) ** 2)))


def _mean_l2(array: np.ndarray) -> float:
    array_float = np.asarray(array, dtype=np.float64)
    return float(np.mean(np.linalg.norm(array_float, axis=1)))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) <= 1e-12:
        return 0.0
    return float(numerator / denominator)


def _hidden_residual_rms(context: Any, z: np.ndarray) -> float:
    flow = hidden_local_flow(context, z)
    return float(np.sqrt(np.mean(flow * flow)))


def _source_run_dir(source_comparison_dir: Path, seed: int) -> Path:
    return (
        source_comparison_dir
        / "runs"
        / "stage_05_two_branch_corrected_residual_core_v2"
        / f"seed_{seed}"
    )


def _load_source_seed_artifacts(
    source_comparison_dir: Path,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], Path]:
    run_dir = _source_run_dir(source_comparison_dir, seed)
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing Stage 05 v2 source run directory: {run_dir}")
    config = _read_json(run_dir / "config.json")
    summary = _read_json(run_dir / "summary.json")
    epoch_rows = _read_epoch_rows(run_dir / "epoch_metrics.csv")
    return config, summary, epoch_rows, run_dir


def _sigma2_from_payload(payload: Any) -> float | tuple[float, ...]:
    if isinstance(payload, list):
        return tuple(float(value) for value in payload)
    return float(payload)


def _build_probe_config_from_source_payload(
    payload: dict[str, Any],
    *,
    output_root: Path,
    experiment_name: str,
    run_id: str,
) -> FMPCEFExploratoryProbeConfig:
    dataset = payload["dataset"]
    model = payload["model"]
    transport = payload["transport"]
    psi_network = payload["psi_network"]
    run = payload["run"]
    return build_fmpc_ef_exploratory_probe_config(
        output_root=output_root,
        experiment_name=experiment_name,
        output_layout="run_id_subdir",
        run_id=run_id,
        dataset_name=str(dataset["dataset_name"]),
        run_seed=int(run["run_seed"]),
        data_seed=int(dataset["data_seed"]),
        model_init_seed=int(run["model_init_seed"]),
        psi_init_seed=int(psi_network["psi_init_seed"]),
        batch_order_seed=int(run["batch_order_seed"]),
        train_fraction=float(dataset["train_fraction"]),
        val_fraction=float(dataset["val_fraction"]),
        test_fraction=float(dataset["test_fraction"]),
        layer_dims=tuple(int(value) for value in model["layer_dims"]),
        hidden_activation=str(model["hidden_activation"]),
        output_activation=str(model["output_activation"]),
        weight_scale=float(model["weight_scale"]),
        sigma2=_sigma2_from_payload(model["sigma2"]),
        eta_x=float(model["eta_x"]),
        eta_w=float(model["eta_w"]),
        eta_b=float(model["eta_b"]),
        eval_steps=int(model["eval_steps"]),
        state_init=str(model["state_init"]),
        epochs=int(run["epochs"]),
        batch_size=int(run["batch_size"]),
        shuffle_batches=bool(run["shuffle_batches"]),
        transport_steps=int(transport["transport_steps"]),
        lambda_id_warmup_epochs=int(transport["lambda_id_warmup_epochs"]),
        lambda_id_ramp_epochs=int(transport["lambda_id_ramp_epochs"]),
        identity_loss_weight=float(transport["identity_loss_weight"]),
        tangent_epsilon=float(transport["tangent_epsilon"]),
        use_two_branch_residual_core=bool(transport["use_two_branch_residual_core"]),
        feature_aware_state_branch_tangents=bool(
            transport["feature_aware_state_branch_tangents"]
        ),
        psi_hidden_dims=tuple(int(value) for value in psi_network["hidden_dims"]),
        psi_weight_scale=float(psi_network["weight_scale"]),
        psi_eta_w=float(psi_network["eta_w"]),
        psi_eta_b=float(psi_network["eta_b"]),
        bootstrap_integrator=str(transport["bootstrap_integrator"]),
        bootstrap_substeps=int(transport["bootstrap_substeps"]),
    )


def _max_accuracy_epoch(epoch_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(epoch_rows, key=lambda row: (float(row["val_accuracy"]), -int(row["epoch"])))


def _selected_epoch_row(epoch_rows: list[dict[str, Any]], selected_epoch: int) -> dict[str, Any]:
    for row in epoch_rows:
        if int(row["epoch"]) == int(selected_epoch):
            return row
    raise ValueError(f"selected_epoch {selected_epoch} not found in epoch metrics.")


def _near_final_trend(epoch_rows: list[dict[str, Any]], window: int) -> dict[str, Any]:
    rows = epoch_rows[-int(window) :]
    first = rows[0]
    last = rows[-1]
    return {
        "window_epochs": [int(row["epoch"]) for row in rows],
        "train_total_loss_delta": float(last["train_total_loss"] - first["train_total_loss"]),
        "train_boot_loss_delta": float(last["train_boot_loss"] - first["train_boot_loss"]),
        "train_identity_loss_delta": float(last["train_identity_loss"] - first["train_identity_loss"]),
        "val_configured_energy_delta_vs_identity_delta": float(
            last["val_configured_energy_delta_vs_identity"]
            - first["val_configured_energy_delta_vs_identity"]
        ),
        "val_configured_fixed_point_residual_delta_vs_identity_delta": float(
            last["val_configured_fixed_point_residual_delta_vs_identity"]
            - first["val_configured_fixed_point_residual_delta_vs_identity"]
        ),
        "val_accuracy_delta": float(last["val_accuracy"] - first["val_accuracy"]),
        "configured_energy_still_improving": bool(
            last["val_configured_energy_delta_vs_identity"]
            < first["val_configured_energy_delta_vs_identity"]
        ),
        "configured_residual_still_improving": bool(
            last["val_configured_fixed_point_residual_delta_vs_identity"]
            < first["val_configured_fixed_point_residual_delta_vs_identity"]
        ),
        "val_accuracy_still_improving": bool(last["val_accuracy"] > first["val_accuracy"]),
    }


def _branch_contribution_label(
    *,
    state_to_traj_ratio: float,
    state_term_to_traj_term_ratio: float,
    material_threshold: float,
    weak_threshold: float,
) -> str:
    composite_ratio = max(float(state_to_traj_ratio), float(state_term_to_traj_term_ratio))
    if composite_ratio >= float(material_threshold):
        return "clearly_material"
    if composite_ratio >= float(weak_threshold):
        return "weak_but_nonzero"
    return "effectively_negligible"


def _rollout_primary_gap_label(
    *,
    step1_delta: float,
    step2_delta: float,
    extra_gain_fraction_threshold: float,
) -> str:
    if float(step2_delta) > float(step1_delta):
        return "degradation_or_compounding_after_step1"
    step1_gain = abs(float(step1_delta))
    extra_gain = abs(float(step2_delta) - float(step1_delta))
    if step1_gain <= 1e-12:
        return "insufficient_extra_gain_after_step1"
    if extra_gain <= float(extra_gain_fraction_threshold) * step1_gain:
        return "insufficient_extra_gain_after_step1"
    return "meaningful_extra_gain_after_step1"


def _stepwise_branch_rows(
    *,
    seed: int,
    model: Any,
    config: FMPCEFExploratoryProbeConfig,
    psi_network: Stage05ResidualCoreNetworks,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    context = build_tf1_context(model, x_val, y_val)
    identity_rollout = rollout_hidden_transport(
        context,
        context.z0,
        transport_steps=config.transport_steps,
        mode="identity",
    )
    local_field_rollout = rollout_hidden_transport(
        context,
        context.z0,
        transport_steps=config.transport_steps,
        mode="local_field_only",
    )
    learned_rollout = rollout_hidden_transport(
        context,
        context.z0,
        transport_steps=config.transport_steps,
        mode="learned",
        velocity_fn=_learned_velocity_fn(context, psi_network, config),
    )

    rows: list[dict[str, Any]] = []
    for step_index in range(1, config.transport_steps + 1):
        t_k = float(learned_rollout.knot_times[step_index - 1])
        r_k = 1.0 - t_k
        z_k = learned_rollout.z_knots[step_index - 1]
        trajectory_input, state_input, _ = _residual_core_inputs_for_state(
            context,
            config,
            z_k,
            context.targets,
            t=t_k,
            r=r_k,
        )
        predictions = _predict_residual_from_inputs(
            psi_network,
            trajectory_input,
            state_inputs=state_input,
        )
        corrected_identity = build_corrected_residual_identity_target(
            context,
            psi_network,
            z_k,
            context.targets,
            t=t_k,
            r=r_k,
            tangent_epsilon=config.tangent_epsilon,
            feature_aware_state_branch_tangents=config.feature_aware_state_branch_tangents,
        )

        learned_z = learned_rollout.z_knots[step_index]
        identity_z = identity_rollout.z_knots[step_index]
        local_field_z = local_field_rollout.z_knots[step_index]
        learned_energy = hidden_energy_from_state(context, learned_z)
        identity_energy = hidden_energy_from_state(context, identity_z)
        local_field_energy = hidden_energy_from_state(context, local_field_z)
        learned_residual_rms = _hidden_residual_rms(context, learned_z)
        identity_residual_rms = _hidden_residual_rms(context, identity_z)
        local_field_residual_rms = _hidden_residual_rms(context, local_field_z)

        rows.append(
            {
                "seed": int(seed),
                "step_index": int(step_index),
                "t_k": float(t_k),
                "r_k": float(r_k),
                "m_traj_rms": _rms(predictions.trajectory_residual),
                "m_state_rms": _rms(predictions.state_residual),
                "m_total_rms": _rms(predictions.total_residual),
                "m_traj_mean_l2": _mean_l2(predictions.trajectory_residual),
                "m_state_mean_l2": _mean_l2(predictions.state_residual),
                "m_total_mean_l2": _mean_l2(predictions.total_residual),
                "state_over_traj_ratio": _safe_ratio(
                    _mean_l2(predictions.state_residual),
                    _mean_l2(predictions.trajectory_residual),
                ),
                "anchor_term_rms": _rms(corrected_identity.anchor_term),
                "trajectory_term_rms": _rms(corrected_identity.trajectory_term),
                "state_term_rms": _rms(corrected_identity.state_term),
                "anchor_term_mean_l2": _mean_l2(corrected_identity.anchor_term),
                "trajectory_term_mean_l2": _mean_l2(corrected_identity.trajectory_term),
                "state_term_mean_l2": _mean_l2(corrected_identity.state_term),
                "state_term_over_trajectory_term_ratio": _safe_ratio(
                    _mean_l2(corrected_identity.state_term),
                    _mean_l2(corrected_identity.trajectory_term),
                ),
                "step_energy": float(learned_energy),
                "step_energy_delta_vs_identity": float(learned_energy - identity_energy),
                "step_energy_delta_vs_local_field_only": float(
                    learned_energy - local_field_energy
                ),
                "step_fixed_point_residual_rms": float(learned_residual_rms),
                "step_fixed_point_residual_delta_vs_identity": float(
                    learned_residual_rms - identity_residual_rms
                ),
                "step_fixed_point_residual_delta_vs_local_field_only": float(
                    learned_residual_rms - local_field_residual_rms
                ),
            }
        )

    step1_row = rows[0]
    stepN_row = rows[-1]
    branch_summary = {
        "m_traj_rms_mean": _mean([float(row["m_traj_rms"]) for row in rows]),
        "m_state_rms_mean": _mean([float(row["m_state_rms"]) for row in rows]),
        "m_traj_mean_l2": _mean([float(row["m_traj_mean_l2"]) for row in rows]),
        "m_state_mean_l2": _mean([float(row["m_state_mean_l2"]) for row in rows]),
        "state_over_traj_ratio_mean": _mean(
            [float(row["state_over_traj_ratio"]) for row in rows]
        ),
        "anchor_term_rms_mean": _mean([float(row["anchor_term_rms"]) for row in rows]),
        "trajectory_term_rms_mean": _mean(
            [float(row["trajectory_term_rms"]) for row in rows]
        ),
        "state_term_rms_mean": _mean([float(row["state_term_rms"]) for row in rows]),
        "anchor_term_mean_l2": _mean([float(row["anchor_term_mean_l2"]) for row in rows]),
        "trajectory_term_mean_l2": _mean(
            [float(row["trajectory_term_mean_l2"]) for row in rows]
        ),
        "state_term_mean_l2": _mean([float(row["state_term_mean_l2"]) for row in rows]),
        "state_term_over_trajectory_term_ratio_mean": _mean(
            [float(row["state_term_over_trajectory_term_ratio"]) for row in rows]
        ),
        "step1_energy_delta_vs_identity": float(step1_row["step_energy_delta_vs_identity"]),
        "stepN_energy_delta_vs_identity": float(stepN_row["step_energy_delta_vs_identity"]),
        "step1_fixed_point_residual_delta_vs_identity": float(
            step1_row["step_fixed_point_residual_delta_vs_identity"]
        ),
        "stepN_fixed_point_residual_delta_vs_identity": float(
            stepN_row["step_fixed_point_residual_delta_vs_identity"]
        ),
    }
    return rows, branch_summary


def _reproduction_match(
    *,
    source_summary: dict[str, Any],
    reproduced_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "selected_epoch_matches_source": bool(
            int(reproduced_summary["selected_epoch"]) == int(source_summary["selected_epoch"])
        ),
        "val_accuracy_matches_source": bool(
            np.isclose(
                float(reproduced_summary["val_accuracy"]),
                float(source_summary["val_accuracy"]),
                atol=1e-12,
                rtol=0.0,
            )
        ),
        "configured_step_energy_delta_matches_source": bool(
            np.isclose(
                float(
                    reproduced_summary["mechanism_metrics"]["configured_steps"][
                        "energy_delta_vs_identity"
                    ]
                ),
                float(
                    source_summary["mechanism_metrics"]["configured_steps"][
                        "energy_delta_vs_identity"
                    ]
                ),
                atol=1e-12,
                rtol=0.0,
            )
        ),
    }


def _seed_diagnostic(
    *,
    config: Stage05V2DiagnosticsConfig,
    run_dir: Path,
    source_comparison_dir: Path,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    source_config, source_summary, source_epoch_rows, source_run_dir = _load_source_seed_artifacts(
        source_comparison_dir,
        seed,
    )

    total_epochs = int(source_config["run"]["epochs"])
    selected_epoch = int(source_summary["selected_epoch"])
    selected_row = _selected_epoch_row(source_epoch_rows, selected_epoch)
    best_accuracy_row = _max_accuracy_epoch(source_epoch_rows)
    near_final = _near_final_trend(source_epoch_rows, config.near_final_window)

    reproduction_output_root = run_dir / config.reproduction_output_subdir
    probe_config = _build_probe_config_from_source_payload(
        source_config,
        output_root=reproduction_output_root,
        experiment_name="stage_05_two_branch_corrected_residual_core_v2",
        run_id=f"seed_{seed}",
    )
    reproduced = run_fmpc_ef_exploratory_probe(probe_config)
    reproduction_match = _reproduction_match(
        source_summary=source_summary,
        reproduced_summary=reproduced.summary,
    )

    split = load_digits_split(
        split_seed=probe_config.data_seed,
        train_fraction=probe_config.train_fraction,
        val_fraction=probe_config.val_fraction,
        test_fraction=probe_config.test_fraction,
    )
    branch_rows, branch_summary = _stepwise_branch_rows(
        seed=seed,
        model=reproduced.model,
        config=probe_config,
        psi_network=reproduced.psi_network,
        x_val=split.x_val,
        y_val=split.y_val,
    )
    branch_label = _branch_contribution_label(
        state_to_traj_ratio=float(branch_summary["state_over_traj_ratio_mean"]),
        state_term_to_traj_term_ratio=float(
            branch_summary["state_term_over_trajectory_term_ratio_mean"]
        ),
        material_threshold=config.state_branch_material_ratio_threshold,
        weak_threshold=config.state_branch_weak_ratio_threshold,
    )
    rollout_label = _rollout_primary_gap_label(
        step1_delta=float(branch_summary["step1_energy_delta_vs_identity"]),
        step2_delta=float(branch_summary["stepN_energy_delta_vs_identity"]),
        extra_gain_fraction_threshold=config.rollout_extra_gain_fraction_threshold,
    )

    epoch_rows = []
    for row in source_epoch_rows:
        epoch_rows.append(
            {
                "seed": int(seed),
                "epoch": int(row["epoch"]),
                "stage": str(row["stage"]),
                "lambda_id": float(row["lambda_id"]),
                "train_total_loss": float(row["train_total_loss"]),
                "train_boot_loss": float(row["train_boot_loss"]),
                "train_identity_loss": float(row["train_identity_loss"]),
                "val_one_step_energy_delta_vs_identity": float(
                    row["val_one_step_energy_delta_vs_identity"]
                ),
                "val_configured_step_energy_delta_vs_identity": float(
                    row["val_configured_energy_delta_vs_identity"]
                ),
                "val_configured_step_fixed_point_residual_delta_vs_identity": float(
                    row["val_configured_fixed_point_residual_delta_vs_identity"]
                ),
                "val_accuracy": float(row["val_accuracy"]),
            }
        )

    seed_summary = {
        "seed": int(seed),
        "source_run_dir": source_run_dir.as_posix(),
        "selected_epoch": int(selected_epoch),
        "total_epochs": int(total_epochs),
        "selection_hits_final_training_boundary": bool(selected_epoch == total_epochs),
        "selected_epoch_stage": str(source_summary["selected_epoch_stage"]),
        "selected_epoch_lambda_id": float(source_summary["selected_epoch_lambda_id"]),
        "highest_val_accuracy_epoch": int(best_accuracy_row["epoch"]),
        "highest_val_accuracy": float(best_accuracy_row["val_accuracy"]),
        "selected_epoch_val_accuracy": float(selected_row["val_accuracy"]),
        "selected_vs_best_accuracy_gap": float(
            best_accuracy_row["val_accuracy"] - selected_row["val_accuracy"]
        ),
        "selected_epoch_val_configured_step_energy_delta_vs_identity": float(
            selected_row["val_configured_energy_delta_vs_identity"]
        ),
        "selected_epoch_val_configured_step_fixed_point_residual_delta_vs_identity": float(
            selected_row["val_configured_fixed_point_residual_delta_vs_identity"]
        ),
        "selected_epoch_val_one_step_energy_delta_vs_identity": float(
            selected_row["val_one_step_energy_delta_vs_identity"]
        ),
        "selected_epoch_val_accuracy": float(selected_row["val_accuracy"]),
        "near_final_trend": near_final,
        "reproduction_match": reproduction_match,
        "branch_summary": {
            **branch_summary,
            "state_branch_contribution_label": branch_label,
        },
        "rollout_summary": {
            "rollout_gap_label": rollout_label,
            "one_step_energy_delta_vs_identity": float(
                source_summary["mechanism_metrics"]["one_step"]["energy_delta_vs_identity"]
            ),
            "configured_step_energy_delta_vs_identity": float(
                source_summary["mechanism_metrics"]["configured_steps"][
                    "energy_delta_vs_identity"
                ]
            ),
            "one_step_fixed_point_residual_delta_vs_identity": float(
                source_summary["mechanism_metrics"]["one_step"][
                    "fixed_point_residual_delta_vs_identity"
                ]
            ),
            "configured_step_fixed_point_residual_delta_vs_identity": float(
                source_summary["mechanism_metrics"]["configured_steps"][
                    "fixed_point_residual_delta_vs_identity"
                ]
            ),
            "incremental_configured_step_energy_gain_over_step1": float(
                branch_summary["stepN_energy_delta_vs_identity"]
                - branch_summary["step1_energy_delta_vs_identity"]
            ),
            "incremental_configured_step_residual_gain_over_step1": float(
                branch_summary["stepN_fixed_point_residual_delta_vs_identity"]
                - branch_summary["step1_fixed_point_residual_delta_vs_identity"]
            ),
        },
        "selection_rule_summary": {
            "selected_epoch": int(selected_epoch),
            "best_accuracy_epoch": int(best_accuracy_row["epoch"]),
            "best_accuracy_epoch_is_selected_epoch": bool(
                int(best_accuracy_row["epoch"]) == int(selected_epoch)
            ),
            "accuracy_gap_vs_selected": float(
                best_accuracy_row["val_accuracy"] - selected_row["val_accuracy"]
            ),
            "best_accuracy_still_low": bool(float(best_accuracy_row["val_accuracy"]) < 0.5),
        },
    }
    return seed_summary, epoch_rows, branch_rows


def _aggregate_epoch_diagnostics(epoch_rows: list[dict[str, Any]]) -> dict[str, Any]:
    last_epoch_by_seed: list[dict[str, Any]] = []
    for seed in sorted({int(row["seed"]) for row in epoch_rows}):
        seed_rows = [row for row in epoch_rows if int(row["seed"]) == seed]
        last_epoch_by_seed.append(seed_rows[-1])
    return {
        "final_epoch_train_total_loss_mean": _mean(
            [float(row["train_total_loss"]) for row in last_epoch_by_seed]
        ),
        "final_epoch_train_boot_loss_mean": _mean(
            [float(row["train_boot_loss"]) for row in last_epoch_by_seed]
        ),
        "final_epoch_train_identity_loss_mean": _mean(
            [float(row["train_identity_loss"]) for row in last_epoch_by_seed]
        ),
        "final_epoch_val_configured_step_energy_delta_vs_identity_mean": _mean(
            [
                float(row["val_configured_step_energy_delta_vs_identity"])
                for row in last_epoch_by_seed
            ]
        ),
        "final_epoch_val_configured_step_fixed_point_residual_delta_vs_identity_mean": _mean(
            [
                float(row["val_configured_step_fixed_point_residual_delta_vs_identity"])
                for row in last_epoch_by_seed
            ]
        ),
        "final_epoch_val_accuracy_mean": _mean(
            [float(row["val_accuracy"]) for row in last_epoch_by_seed]
        ),
    }


def _aggregate_branch_diagnostics(seed_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = [
        float(summary["branch_summary"]["state_over_traj_ratio_mean"])
        for summary in seed_summaries
    ]
    term_ratios = [
        float(summary["branch_summary"]["state_term_over_trajectory_term_ratio_mean"])
        for summary in seed_summaries
    ]
    labels = [
        str(summary["branch_summary"]["state_branch_contribution_label"])
        for summary in seed_summaries
    ]
    label_counts = {label: labels.count(label) for label in sorted(set(labels))}
    dominant_label = max(label_counts.items(), key=lambda item: item[1])[0]
    return {
        "m_traj_rms_mean": _mean(
            [float(summary["branch_summary"]["m_traj_rms_mean"]) for summary in seed_summaries]
        ),
        "m_state_rms_mean": _mean(
            [float(summary["branch_summary"]["m_state_rms_mean"]) for summary in seed_summaries]
        ),
        "m_traj_mean_l2": _mean(
            [float(summary["branch_summary"]["m_traj_mean_l2"]) for summary in seed_summaries]
        ),
        "m_state_mean_l2": _mean(
            [float(summary["branch_summary"]["m_state_mean_l2"]) for summary in seed_summaries]
        ),
        "state_over_traj_ratio_mean": _mean(ratios),
        "state_over_traj_ratio_std": _std(ratios),
        "anchor_term_rms_mean": _mean(
            [float(summary["branch_summary"]["anchor_term_rms_mean"]) for summary in seed_summaries]
        ),
        "trajectory_term_rms_mean": _mean(
            [
                float(summary["branch_summary"]["trajectory_term_rms_mean"])
                for summary in seed_summaries
            ]
        ),
        "state_term_rms_mean": _mean(
            [float(summary["branch_summary"]["state_term_rms_mean"]) for summary in seed_summaries]
        ),
        "state_term_over_trajectory_term_ratio_mean": _mean(term_ratios),
        "state_term_over_trajectory_term_ratio_std": _std(term_ratios),
        "state_branch_contribution_label_counts": label_counts,
        "state_branch_contribution_label": dominant_label,
        "state_branch_forward_contribution_material_but_identity_term_small": bool(
            _mean(ratios) >= 0.25 and _mean(term_ratios) < 0.01
        ),
    }


def _aggregate_rollout_diagnostics(seed_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [str(summary["rollout_summary"]["rollout_gap_label"]) for summary in seed_summaries]
    label_counts = {label: labels.count(label) for label in sorted(set(labels))}
    dominant_label = max(label_counts.items(), key=lambda item: item[1])[0]
    return {
        "one_step_energy_delta_vs_identity_mean": _mean(
            [
                float(summary["rollout_summary"]["one_step_energy_delta_vs_identity"])
                for summary in seed_summaries
            ]
        ),
        "configured_step_energy_delta_vs_identity_mean": _mean(
            [
                float(summary["rollout_summary"]["configured_step_energy_delta_vs_identity"])
                for summary in seed_summaries
            ]
        ),
        "one_step_fixed_point_residual_delta_vs_identity_mean": _mean(
            [
                float(
                    summary["rollout_summary"]["one_step_fixed_point_residual_delta_vs_identity"]
                )
                for summary in seed_summaries
            ]
        ),
        "configured_step_fixed_point_residual_delta_vs_identity_mean": _mean(
            [
                float(
                    summary["rollout_summary"][
                        "configured_step_fixed_point_residual_delta_vs_identity"
                    ]
                )
                for summary in seed_summaries
            ]
        ),
        "incremental_configured_step_energy_gain_over_step1_mean": _mean(
            [
                float(summary["rollout_summary"]["incremental_configured_step_energy_gain_over_step1"])
                for summary in seed_summaries
            ]
        ),
        "incremental_configured_step_residual_gain_over_step1_mean": _mean(
            [
                float(
                    summary["rollout_summary"][
                        "incremental_configured_step_residual_gain_over_step1"
                    ]
                )
                for summary in seed_summaries
            ]
        ),
        "rollout_gap_label_counts": label_counts,
        "rollout_gap_label": dominant_label,
    }


def _aggregate_selection_diagnostics(
    *,
    config: Stage05V2DiagnosticsConfig,
    seed_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    gaps = [
        float(summary["selection_rule_summary"]["accuracy_gap_vs_selected"])
        for summary in seed_summaries
    ]
    best_is_selected_rate = _mean(
        [
            1.0
            if bool(summary["selection_rule_summary"]["best_accuracy_epoch_is_selected_epoch"])
            else 0.0
            for summary in seed_summaries
        ]
    )
    sizable_gap_rate = _mean(
        [
            1.0
            if float(summary["selection_rule_summary"]["accuracy_gap_vs_selected"])
            >= float(config.selection_accuracy_gap_threshold)
            else 0.0
            for summary in seed_summaries
        ]
    )
    best_accuracy_low_rate = _mean(
        [
            1.0 if bool(summary["selection_rule_summary"]["best_accuracy_still_low"]) else 0.0
            for summary in seed_summaries
        ]
    )
    return {
        "selected_vs_best_accuracy_gap_mean": _mean(gaps),
        "selected_vs_best_accuracy_gap_std": _std(gaps),
        "best_accuracy_epoch_is_selected_epoch_rate": best_is_selected_rate,
        "sizable_accuracy_gap_rate": sizable_gap_rate,
        "best_accuracy_still_low_rate": best_accuracy_low_rate,
        "selection_rule_likely_primary_issue": bool(
            sizable_gap_rate >= 0.5 and best_accuracy_low_rate < 0.5
        ),
    }


def _select_diagnosis(
    *,
    seed_summaries: list[dict[str, Any]],
    branch_aggregate: dict[str, Any],
    selection_aggregate: dict[str, Any],
) -> tuple[Stage05V2DiagnosisLabel, str, str]:
    boundary_hit_rate = _mean(
        [
            1.0 if bool(summary["selection_hits_final_training_boundary"]) else 0.0
            for summary in seed_summaries
        ]
    )
    configured_energy_improving_rate = _mean(
        [
            1.0 if bool(summary["near_final_trend"]["configured_energy_still_improving"]) else 0.0
            for summary in seed_summaries
        ]
    )
    configured_residual_improving_rate = _mean(
        [
            1.0
            if bool(summary["near_final_trend"]["configured_residual_still_improving"])
            else 0.0
            for summary in seed_summaries
        ]
    )
    accuracy_improving_rate = _mean(
        [
            1.0 if bool(summary["near_final_trend"]["val_accuracy_still_improving"]) else 0.0
            for summary in seed_summaries
        ]
    )
    if bool(selection_aggregate["selection_rule_likely_primary_issue"]):
        return (
            "selection_pressure_misaligned_with_report_accuracy",
            "selection_rule_refinement",
            "Validation accuracy peaks materially above the energy-selected epoch on enough seeds to make selection pressure the main report-accuracy issue.",
        )
    if str(branch_aggregate["state_branch_contribution_label"]) == "effectively_negligible":
        return (
            "state_branch_underutilized",
            "branch_strengthening",
            "The state branch stays effectively negligible relative to the trajectory branch and its identity-term contribution is also negligible.",
        )
    if (
        boundary_hit_rate >= 1.0
        and configured_energy_improving_rate >= 1.0
        and configured_residual_improving_rate >= 1.0
        and accuracy_improving_rate >= 1.0
    ):
        return (
            "likely_undertrained",
            "longer_training_or_budget",
            "All Stage 05 v2 seeds select the final training epoch and configured-step mechanism plus validation accuracy are still improving at the training boundary.",
        )
    return (
        "configured_step_rollout_accumulation_is_primary_gap",
        "rollout_aware_multi_step_strengthening",
        "Configured-step gain stays much weaker than one-step gain, and the extra gain after step 1 remains too small to close the Stage 04 gap.",
    )


def _report_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Stage 05 V2 Diagnostics",
        "",
        "## Scope",
        f"- source comparison: `{summary['source_artifacts']['source_comparison_dir']}`",
        f"- seeds: `{summary['source_artifacts']['seeds']}`",
        "- this pass keeps Stage 04 frozen and keeps the Stage 05 v2 transport family unchanged",
        "",
        "## Training Boundary",
        f"- selected epochs: `{summary['training_boundary']['selected_epochs']}`",
        f"- total epochs: `{summary['training_boundary']['total_epochs']}`",
        f"- selection hits final training boundary on all seeds: `{summary['training_boundary']['selection_hits_final_boundary_on_all_seeds']}`",
        "",
        "## Epoch-Level Diagnosis",
        f"- configured-step energy still improving near final epoch rate: `{summary['epoch_level']['configured_energy_still_improving_rate']}`",
        f"- configured-step residual still improving near final epoch rate: `{summary['epoch_level']['configured_residual_still_improving_rate']}`",
        f"- validation accuracy still improving near final epoch rate: `{summary['epoch_level']['val_accuracy_still_improving_rate']}`",
        f"- final-epoch mean validation accuracy: `{summary['epoch_level']['aggregate']['final_epoch_val_accuracy_mean']:.6f}`",
        "",
        "## Branch Contribution",
        f"- state branch contribution label: `{summary['branch_contribution']['state_branch_contribution_label']}`",
        f"- mean `||m_state|| / ||m_traj||`: `{summary['branch_contribution']['state_over_traj_ratio_mean']:.6f}`",
        f"- mean state-term / trajectory-term ratio: `{summary['branch_contribution']['state_term_over_trajectory_term_ratio_mean']:.6f}`",
        f"- forward state contribution material but identity-term contribution small: `{summary['branch_contribution']['state_branch_forward_contribution_material_but_identity_term_small']}`",
        "",
        "## Rollout Diagnosis",
        f"- rollout gap label: `{summary['rollout_diagnosis']['rollout_gap_label']}`",
        f"- mean one-step energy delta vs identity: `{summary['rollout_diagnosis']['one_step_energy_delta_vs_identity_mean']:.12f}`",
        f"- mean configured-step energy delta vs identity: `{summary['rollout_diagnosis']['configured_step_energy_delta_vs_identity_mean']:.12f}`",
        f"- mean incremental configured-step energy gain over step 1: `{summary['rollout_diagnosis']['incremental_configured_step_energy_gain_over_step1_mean']:.12f}`",
        "",
        "## Selection-Rule Diagnosis",
        f"- best-accuracy epoch is selected-epoch rate: `{summary['selection_rule_diagnosis']['best_accuracy_epoch_is_selected_epoch_rate']}`",
        f"- mean selected-vs-best accuracy gap: `{summary['selection_rule_diagnosis']['selected_vs_best_accuracy_gap_mean']:.6f}`",
        f"- selection rule likely primary issue: `{summary['selection_rule_diagnosis']['selection_rule_likely_primary_issue']}`",
        "",
        "## Final Diagnosis",
        f"- selected label: `{summary['selected_diagnosis_label']}`",
        f"- rationale: `{summary['diagnosis_rationale']}`",
        f"- next Stage 05 v3 target: `{summary['recommended_next_stage05_v3_target']}`",
        "",
    ]
    return "\n".join(lines)


def _config_payload(config: Stage05V2DiagnosticsConfig) -> dict[str, Any]:
    return {
        "phase": "FMPC Stage 05 EF Core Probe",
        "stage": "stage05_v2_diagnostics",
        "source_comparison_dir": str(Path(config.source_comparison_dir).as_posix()),
        "seeds": [int(seed) for seed in config.seeds],
        "near_final_window": int(config.near_final_window),
        "selection_accuracy_gap_threshold": float(config.selection_accuracy_gap_threshold),
        "state_branch_material_ratio_threshold": float(
            config.state_branch_material_ratio_threshold
        ),
        "state_branch_weak_ratio_threshold": float(config.state_branch_weak_ratio_threshold),
        "rollout_extra_gain_fraction_threshold": float(
            config.rollout_extra_gain_fraction_threshold
        ),
        "artifacts": {
            "diagnostic_summary_json": "diagnostic_summary.json",
            "diagnostic_report_md": "diagnostic_report.md",
            "epoch_diagnostics_csv": "epoch_diagnostics.csv",
            "branch_diagnostics_csv": "branch_diagnostics.csv",
        },
    }


def run_stage05_v2_diagnostics(
    config: Stage05V2DiagnosticsConfig,
) -> Stage05V2DiagnosticsRunResult:
    """Run a narrow evidence-backed diagnostic pass for the current Stage 05 v2 reference."""

    run_dir = _prepare_run_dir(
        _resolve_run_dir(
            config.output_root,
            config.experiment_name,
            config.resolved_run_id(),
            config.output_layout,
        )
    )
    source_comparison_dir = Path(config.source_comparison_dir)
    if not source_comparison_dir.exists():
        raise FileNotFoundError(
            f"Missing source comparison directory: {source_comparison_dir}"
        )

    config_payload = _config_payload(config)
    _write_json(run_dir / "config.json", config_payload)

    seed_summaries: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        seed_summary, seed_epoch_rows, seed_branch_rows = _seed_diagnostic(
            config=config,
            run_dir=run_dir,
            source_comparison_dir=source_comparison_dir,
            seed=int(seed),
        )
        seed_summaries.append(seed_summary)
        epoch_rows.extend(seed_epoch_rows)
        branch_rows.extend(seed_branch_rows)

    training_boundary = {
        "selected_epochs": [int(summary["selected_epoch"]) for summary in seed_summaries],
        "total_epochs": [int(summary["total_epochs"]) for summary in seed_summaries],
        "selection_hits_final_boundary_on_all_seeds": bool(
            all(bool(summary["selection_hits_final_training_boundary"]) for summary in seed_summaries)
        ),
    }
    epoch_level = {
        "configured_energy_still_improving_rate": _mean(
            [
                1.0 if bool(summary["near_final_trend"]["configured_energy_still_improving"]) else 0.0
                for summary in seed_summaries
            ]
        ),
        "configured_residual_still_improving_rate": _mean(
            [
                1.0
                if bool(summary["near_final_trend"]["configured_residual_still_improving"])
                else 0.0
                for summary in seed_summaries
            ]
        ),
        "val_accuracy_still_improving_rate": _mean(
            [
                1.0 if bool(summary["near_final_trend"]["val_accuracy_still_improving"]) else 0.0
                for summary in seed_summaries
            ]
        ),
        "aggregate": _aggregate_epoch_diagnostics(epoch_rows),
    }
    branch_aggregate = _aggregate_branch_diagnostics(seed_summaries)
    rollout_aggregate = _aggregate_rollout_diagnostics(seed_summaries)
    selection_aggregate = _aggregate_selection_diagnostics(
        config=config,
        seed_summaries=seed_summaries,
    )
    diagnosis_label, recommendation_target, rationale = _select_diagnosis(
        seed_summaries=seed_summaries,
        branch_aggregate=branch_aggregate,
        selection_aggregate=selection_aggregate,
    )

    summary = {
        "phase": "FMPC Stage 05 EF Core Probe",
        "stage": "stage05_v2_diagnostics",
        "source_artifacts": {
            "source_comparison_dir": source_comparison_dir.as_posix(),
            "seeds": [int(seed) for seed in config.seeds],
        },
        "training_boundary": training_boundary,
        "epoch_level": epoch_level,
        "branch_contribution": branch_aggregate,
        "rollout_diagnosis": rollout_aggregate,
        "selection_rule_diagnosis": selection_aggregate,
        "per_seed": seed_summaries,
        "selected_diagnosis_label": diagnosis_label,
        "diagnosis_rationale": rationale,
        "recommended_next_stage05_v3_target": recommendation_target,
        "artifacts": {
            "config_json": "config.json",
            "diagnostic_summary_json": "diagnostic_summary.json",
            "diagnostic_report_md": "diagnostic_report.md",
            "epoch_diagnostics_csv": "epoch_diagnostics.csv",
            "branch_diagnostics_csv": "branch_diagnostics.csv",
        },
    }
    _write_csv(run_dir / "epoch_diagnostics.csv", epoch_rows)
    _write_csv(run_dir / "branch_diagnostics.csv", branch_rows)
    _write_json(run_dir / "diagnostic_summary.json", summary)
    _write_text(run_dir / "diagnostic_report.md", _report_markdown(summary))
    scratch_dir = run_dir / config.reproduction_output_subdir
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)

    return Stage05V2DiagnosticsRunResult(
        run_dir=run_dir,
        config=config_payload,
        summary=summary,
    )
