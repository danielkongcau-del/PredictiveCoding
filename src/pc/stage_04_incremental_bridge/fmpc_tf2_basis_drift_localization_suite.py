from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..datasets import load_digits_split
from ..stage_03_transport_core_v1.fmpc_tf1_flow import build_tf1_context, hidden_states_from_state
from .fmpc_tf2 import (
    build_tf2_corrective_transport_terminal_angleclip_default_config,
    run_fmpc_tf2_experiment,
)
from .fmpc_tf2_endpoint_basis_suite import (
    _delta_geometry,
    _interface_geometry,
    _mean,
    _mean_list,
    _prefixed_metric_stats,
    _region_from_peak_index,
    _relative_posix,
    _rowspace_basis,
    _runtime_proxy_seconds,
    _std,
    _transport_penultimate_by_knot,
)
from ..inference import run_teacher_inference_export

_KNOT_GAP_METRICS = (
    "hidden_state_rms_gap_to_slow_pc",
    "output_state_rms_gap_to_slow_pc",
    "hidden_state_rms_gap_rowspace",
    "hidden_state_rms_gap_orthogonal",
    "hidden_state_gap_rowspace_fraction",
    "centroid_displacement_rms_rowspace",
    "centroid_displacement_rms_total",
)

_KNOT_INTERFACE_METRICS = (
    "frozen_head_accuracy",
    "frozen_head_output_mse",
    "fisher_separability_ratio",
    "nearest_centroid_accuracy",
    "mean_true_class_margin",
    "mean_within_class_rms",
    "between_class_centroid_margin",
)


@dataclass
class FMPCTF2BasisDriftLocalizationSuiteConfig:
    """Run a narrow adopted-package late-rollout basis-drift source-localization diagnostic."""

    experiment_name: str = "fmpc_tf2_basis_drift_localization_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 60
    batch_size: int = 128
    eval_steps: int = 15
    layer_dims: tuple[int, ...] = (64, 64, 10)
    dominance_share_threshold: float = 0.60
    dominance_margin: float = 0.10

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"


@dataclass
class FMPCTF2BasisDriftLocalizationSuiteRunResult:
    run_dir: Path
    config: dict[str, Any]
    aggregate_rows: list[dict[str, Any]]
    knot_rows: list[dict[str, Any]]
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


def _suite_config_payload(config: FMPCTF2BasisDriftLocalizationSuiteConfig) -> dict[str, Any]:
    return {
        "phase": "FMPC Stage 04 Incremental Bridge",
        "stage": "adopted_package_basis_drift_source_localization",
        "base_preset_name": "tf2_corrective_transport_terminal_angleclip_default",
        "transport_family_fixed": True,
        "reference_alignment": "same_model_target_clamped_slow_pc_trajectory_sampled_at_transport_knot_times",
        "seeds": [int(seed) for seed in config.seeds],
        "dominance_thresholds": {
            "dominance_share_threshold": float(config.dominance_share_threshold),
            "dominance_margin": float(config.dominance_margin),
        },
        "required_localization_metrics": {
            "hidden_state_rms_gap_to_slow_pc": True,
            "output_state_rms_gap_to_slow_pc": True,
            "rowspace_component_breakdown": True,
            "centroid_rowspace_displacement": True,
            "nearest_centroid_accuracy": True,
            "fisher_separability_ratio": True,
            "frozen_head_output_mse": True,
            "class_margin_and_within_class_spread": True,
        },
    }


def _run_id(seed: int) -> str:
    return f"adopted_s{seed}"


def _apply_readout(features: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    features_array = np.asarray(features, dtype=np.float64)
    return features_array @ np.asarray(weight, dtype=np.float64).T + np.asarray(bias, dtype=np.float64)


def _rms_gap(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.sqrt(np.mean(delta * delta)))


def _slow_pc_penultimate_by_knot(
    model: Any,
    context: Any,
    knot_times: list[float],
) -> tuple[list[np.ndarray], list[int]]:
    teacher_export = run_teacher_inference_export(
        model.layers,
        np.asarray(context.inputs, dtype=np.float64),
        y=np.asarray(context.targets, dtype=np.float64),
        init=model.state_init,
        mode="train",
        eta_x=model.eta_x,
        steps=int(model.eval_steps),
        backend=str(model.inference_backend),
        record_trace=False,
        record_trajectory=True,
    )
    if teacher_export.z_trajectory is None:
        raise ValueError("Slow-PC trajectory export is required for basis-drift localization.")
    aligned_step_indices: list[int] = []
    penultimate_by_knot: list[np.ndarray] = []
    max_index = int(teacher_export.steps)
    for knot_time in knot_times:
        step_index = int(np.clip(np.rint(float(knot_time) * float(max_index)), 0, max_index))
        aligned_step_indices.append(step_index)
        penultimate_by_knot.append(
            np.asarray(hidden_states_from_state(context, teacher_export.z_trajectory[step_index])[-2], dtype=np.float64)
        )
    return penultimate_by_knot, aligned_step_indices


def _knot_metric_stats(knot_rows: list[dict[str, Any]], split_name: str, metric_names: tuple[str, ...]) -> dict[str, list[float]]:
    split_rows = [row for row in knot_rows if str(row["split"]) == split_name]
    if not split_rows:
        raise ValueError(f"No knot rows found for split '{split_name}'.")
    knot_indices = sorted({int(row["knot_index"]) for row in split_rows})
    stats: dict[str, list[float]] = {}
    for metric_name in metric_names:
        stats[f"mean_{metric_name}_by_knot"] = [
            _mean([float(row[metric_name]) for row in split_rows if int(row["knot_index"]) == knot_index])
            for knot_index in knot_indices
        ]
    stats["rollout_knot_indices"] = [int(index) for index in knot_indices]
    stats["rollout_knot_times"] = [
        _mean([float(row["knot_time"]) for row in split_rows if int(row["knot_index"]) == knot_index])
        for knot_index in knot_indices
    ]
    stats["aligned_slow_pc_step_indices"] = [
        int(round(_mean([float(row["slow_pc_step_index"]) for row in split_rows if int(row["knot_index"]) == knot_index])))
        for knot_index in knot_indices
    ]
    return stats


def _contribution_from_series(series: list[float]) -> dict[str, float]:
    if len(series) < 2:
        raise ValueError("Contribution analysis requires at least two knots.")
    start_value = float(series[0])
    preterminal_value = float(series[-2])
    terminal_value = float(series[-1])
    preterminal_contribution = max(0.0, preterminal_value - start_value)
    terminal_jump_contribution = max(0.0, terminal_value - preterminal_value)
    total_positive_contribution = preterminal_contribution + terminal_jump_contribution
    if total_positive_contribution <= 1e-12:
        preterminal_share = 0.5
        terminal_share = 0.5
    else:
        preterminal_share = float(preterminal_contribution / total_positive_contribution)
        terminal_share = float(terminal_jump_contribution / total_positive_contribution)
    return {
        "start_value": start_value,
        "preterminal_value": preterminal_value,
        "terminal_value": terminal_value,
        "preterminal_contribution": preterminal_contribution,
        "terminal_jump_contribution": terminal_jump_contribution,
        "preterminal_share": preterminal_share,
        "terminal_share": terminal_share,
    }


def _split_contribution_breakdown(knot_breakdown: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    return {
        "rowspace_hidden_gap": _contribution_from_series(knot_breakdown["mean_hidden_state_rms_gap_rowspace_by_knot"]),
        "output_state_gap": _contribution_from_series(knot_breakdown["mean_output_state_rms_gap_to_slow_pc_by_knot"]),
        "rowspace_centroid_displacement": _contribution_from_series(
            knot_breakdown["mean_centroid_displacement_rms_rowspace_by_knot"]
        ),
    }


def _diagnose_split_contributions(
    breakdown: dict[str, dict[str, float]],
    *,
    share_threshold: float,
    margin_threshold: float,
) -> tuple[str | None, float, float]:
    preterminal_share = _mean([float(payload["preterminal_share"]) for payload in breakdown.values()])
    terminal_share = _mean([float(payload["terminal_share"]) for payload in breakdown.values()])
    if preterminal_share >= share_threshold and (preterminal_share - terminal_share) >= margin_threshold:
        return "preterminal", preterminal_share, terminal_share
    if terminal_share >= share_threshold and (terminal_share - preterminal_share) >= margin_threshold:
        return "terminal", preterminal_share, terminal_share
    return None, preterminal_share, terminal_share


def _recommend_next_move(diagnosis: str) -> str:
    if diagnosis == "terminal_jump_injection_dominates":
        return "run one narrow adopted-package terminal-jump formulation diagnostic next"
    if diagnosis == "preterminal_accumulation_dominates":
        return "run one narrow adopted-package late-rollout drift-control diagnostic next"
    if diagnosis == "mixed_but_terminal_dominant":
        return "run one narrow adopted-package terminal-jump formulation diagnostic next because the mixed evidence still leans terminal and that is the narrower follow-up"
    return "run one narrow adopted-package late-rollout drift-control diagnostic next because the mixed evidence still leans preterminal accumulation"


def _diagnose_and_recommend(
    config: FMPCTF2BasisDriftLocalizationSuiteConfig,
    validation_breakdown: dict[str, list[float]],
    test_breakdown: dict[str, list[float]],
) -> tuple[str, dict[str, Any], str]:
    validation_contributions = _split_contribution_breakdown(validation_breakdown)
    test_contributions = _split_contribution_breakdown(test_breakdown)
    validation_winner, validation_pre, validation_terminal = _diagnose_split_contributions(
        validation_contributions,
        share_threshold=float(config.dominance_share_threshold),
        margin_threshold=float(config.dominance_margin),
    )
    test_winner, test_pre, test_terminal = _diagnose_split_contributions(
        test_contributions,
        share_threshold=float(config.dominance_share_threshold),
        margin_threshold=float(config.dominance_margin),
    )
    if validation_winner == "preterminal" and test_winner == "preterminal":
        diagnosis = "preterminal_accumulation_dominates"
    elif validation_winner == "terminal" and test_winner == "terminal":
        diagnosis = "terminal_jump_injection_dominates"
    else:
        overall_pre = _mean([validation_pre, test_pre])
        overall_terminal = _mean([validation_terminal, test_terminal])
        diagnosis = "mixed_but_preterminal_dominant" if overall_pre >= overall_terminal else "mixed_but_terminal_dominant"
    evidence = {
        "validation_contribution_breakdown": validation_contributions,
        "test_contribution_breakdown": test_contributions,
        "validation_preterminal_share_mean": float(validation_pre),
        "validation_terminal_share_mean": float(validation_terminal),
        "test_preterminal_share_mean": float(test_pre),
        "test_terminal_share_mean": float(test_terminal),
    }
    return diagnosis, evidence, _recommend_next_move(diagnosis)


def run_fmpc_tf2_basis_drift_localization_suite(
    config: FMPCTF2BasisDriftLocalizationSuiteConfig,
) -> FMPCTF2BasisDriftLocalizationSuiteRunResult:
    run_dir = _prepare_run_dir(
        _resolve_run_dir(
            config.output_root,
            config.experiment_name,
            config.resolved_run_id(),
            config.output_layout,
        )
    )
    _write_json(run_dir / "config.json", _suite_config_payload(config))

    tf2_root = run_dir / "tf2_runs"
    tf2_root.mkdir(parents=True, exist_ok=True)

    aggregate_rows: list[dict[str, Any]] = []
    knot_rows: list[dict[str, Any]] = []

    for seed in config.seeds:
        tf2_config = build_tf2_corrective_transport_terminal_angleclip_default_config(
            experiment_name="tf2",
            output_root=tf2_root,
            output_layout="run_id_subdir",
            run_id=_run_id(int(seed)),
            run_seed=int(seed),
            data_seed=int(seed),
            model_init_seed=int(seed),
            psi_init_seed=int(seed),
            batch_order_seed=int(seed),
            epochs=int(config.epochs),
            batch_size=int(config.batch_size),
            eval_steps=int(config.eval_steps),
            layer_dims=tuple(config.layer_dims),
        )
        result = run_fmpc_tf2_experiment(tf2_config)
        if result.model is None or result.psi_network is None:
            raise ValueError("Basis-drift localization suite requires runtime model and psi network objects.")

        split = load_digits_split(
            split_seed=int(tf2_config.data_seed),
            train_fraction=float(tf2_config.train_fraction),
            val_fraction=float(tf2_config.val_fraction),
            test_fraction=float(tf2_config.test_fraction),
        )

        weight = np.asarray(result.model.layers[-1].weight, dtype=np.float64)
        bias = np.asarray(result.model.layers[-1].bias, dtype=np.float64)
        basis = _rowspace_basis(weight)

        seed_row: dict[str, Any] = {
            "seed": int(seed),
            "run_id": str(result.run_dir.name),
            "run_summary_path": _relative_posix(run_dir, result.run_dir / "summary.json"),
            "selected_epoch": int(result.summary["best_epoch"]),
            "val_accuracy": float(result.summary["val_accuracy"]),
            "test_accuracy": float(result.summary["test_accuracy"]),
            "gate_passing_epoch_count": int(result.summary["gate_passing_epoch_count"]),
            "selected_epoch_passes_gate": bool(result.summary["selected_epoch_passes_gate"]),
            "selector_fallback_used": bool(result.summary["selector_fallback_used"]),
            "val_report_output_mse": float(result.summary["val_loss"]),
            "test_report_output_mse": float(result.summary["test_loss"]),
            "val_transported_final_energy": float(result.summary["val_transported_final_energy"]),
            "runtime_proxy_seconds": _runtime_proxy_seconds(result.summary),
        }

        for split_name, x_split, y_split in (
            ("validation", split.x_val, split.y_val),
            ("test", split.x_test, split.y_test),
        ):
            context = build_tf1_context(result.model, x_split, y_split)
            transported_knots, knot_times = _transport_penultimate_by_knot(
                result.model,
                result.psi_network,
                tf2_config,
                x_split,
                y_split,
            )
            slow_pc_knots, slow_pc_step_indices = _slow_pc_penultimate_by_knot(result.model, context, knot_times)
            if len(transported_knots) != len(slow_pc_knots):
                raise ValueError("Transported and slow-PC knot lists must align.")

            rowspace_gap_series: list[float] = []
            output_gap_series: list[float] = []
            centroid_rowspace_series: list[float] = []

            for knot_index, (knot_time, slow_pc_step_index, transported_features, slow_pc_features) in enumerate(
                zip(knot_times, slow_pc_step_indices, transported_knots, slow_pc_knots, strict=True)
            ):
                transported_geometry = _interface_geometry(transported_features, y_split, weight, bias)
                slow_pc_geometry = _interface_geometry(slow_pc_features, y_split, weight, bias)
                delta_geometry = _delta_geometry(transported_features, slow_pc_features, y_split, basis)
                output_state_rms_gap = _rms_gap(
                    _apply_readout(transported_features, weight, bias),
                    _apply_readout(slow_pc_features, weight, bias),
                )
                rowspace_gap_series.append(float(delta_geometry["delta_h_rms_rowspace"]))
                output_gap_series.append(float(output_state_rms_gap))
                centroid_rowspace_series.append(float(delta_geometry["centroid_displacement_rms_rowspace"]))

                knot_rows.append(
                    {
                        "seed": int(seed),
                        "split": split_name,
                        "knot_index": int(knot_index),
                        "knot_time": float(knot_time),
                        "slow_pc_step_index": int(slow_pc_step_index),
                        "hidden_state_rms_gap_to_slow_pc": float(delta_geometry["delta_h_rms_total"]),
                        "output_state_rms_gap_to_slow_pc": float(output_state_rms_gap),
                        "hidden_state_rms_gap_rowspace": float(delta_geometry["delta_h_rms_rowspace"]),
                        "hidden_state_rms_gap_orthogonal": float(delta_geometry["delta_h_rms_orthogonal"]),
                        "hidden_state_gap_rowspace_fraction": float(delta_geometry["delta_h_rowspace_fraction"]),
                        "centroid_displacement_rms_rowspace": float(delta_geometry["centroid_displacement_rms_rowspace"]),
                        "centroid_displacement_rms_total": float(delta_geometry["centroid_displacement_rms_total"]),
                        "transported_frozen_head_accuracy": float(transported_geometry["frozen_head_accuracy"]),
                        "slow_pc_frozen_head_accuracy": float(slow_pc_geometry["frozen_head_accuracy"]),
                        "transported_frozen_head_output_mse": float(transported_geometry["frozen_head_output_mse"]),
                        "slow_pc_frozen_head_output_mse": float(slow_pc_geometry["frozen_head_output_mse"]),
                        "transported_fisher_separability_ratio": float(
                            transported_geometry["fisher_separability_ratio"]
                        ),
                        "slow_pc_fisher_separability_ratio": float(slow_pc_geometry["fisher_separability_ratio"]),
                        "transported_nearest_centroid_accuracy": float(
                            transported_geometry["nearest_centroid_accuracy"]
                        ),
                        "slow_pc_nearest_centroid_accuracy": float(slow_pc_geometry["nearest_centroid_accuracy"]),
                        "transported_between_class_centroid_margin": float(
                            transported_geometry["between_class_centroid_margin"]
                        ),
                        "slow_pc_between_class_centroid_margin": float(
                            slow_pc_geometry["between_class_centroid_margin"]
                        ),
                        "transported_mean_within_class_rms": float(transported_geometry["mean_within_class_rms"]),
                        "slow_pc_mean_within_class_rms": float(slow_pc_geometry["mean_within_class_rms"]),
                        "transported_mean_true_class_margin": float(transported_geometry["mean_true_class_margin"]),
                        "slow_pc_mean_true_class_margin": float(slow_pc_geometry["mean_true_class_margin"]),
                    }
                )

            split_prefix = "val" if split_name == "validation" else "test"
            rowspace_contrib = _contribution_from_series(rowspace_gap_series)
            output_contrib = _contribution_from_series(output_gap_series)
            centroid_row_contrib = _contribution_from_series(centroid_rowspace_series)
            seed_row[f"{split_prefix}_preterminal_hidden_gap_rowspace"] = float(rowspace_contrib["preterminal_value"])
            seed_row[f"{split_prefix}_terminal_hidden_gap_rowspace"] = float(rowspace_contrib["terminal_value"])
            seed_row[f"{split_prefix}_preterminal_rowspace_share"] = float(rowspace_contrib["preterminal_share"])
            seed_row[f"{split_prefix}_terminal_rowspace_share"] = float(rowspace_contrib["terminal_share"])
            seed_row[f"{split_prefix}_preterminal_output_gap"] = float(output_contrib["preterminal_value"])
            seed_row[f"{split_prefix}_terminal_output_gap"] = float(output_contrib["terminal_value"])
            seed_row[f"{split_prefix}_preterminal_output_share"] = float(output_contrib["preterminal_share"])
            seed_row[f"{split_prefix}_terminal_output_share"] = float(output_contrib["terminal_share"])
            seed_row[f"{split_prefix}_preterminal_centroid_rowspace_gap"] = float(
                centroid_row_contrib["preterminal_value"]
            )
            seed_row[f"{split_prefix}_terminal_centroid_rowspace_gap"] = float(centroid_row_contrib["terminal_value"])
            seed_row[f"{split_prefix}_preterminal_centroid_share"] = float(
                centroid_row_contrib["preterminal_share"]
            )
            seed_row[f"{split_prefix}_terminal_centroid_share"] = float(centroid_row_contrib["terminal_share"])

        aggregate_rows.append(seed_row)

    _write_csv(run_dir / "aggregate_runs.csv", aggregate_rows)
    _write_csv(run_dir / "knotwise_localization.csv", knot_rows)

    integrated_behavior = {
        "mean_val_accuracy": _mean([float(row["val_accuracy"]) for row in aggregate_rows]),
        "std_val_accuracy": _std([float(row["val_accuracy"]) for row in aggregate_rows]),
        "mean_test_accuracy": _mean([float(row["test_accuracy"]) for row in aggregate_rows]),
        "std_test_accuracy": _std([float(row["test_accuracy"]) for row in aggregate_rows]),
        "mean_gate_passing_epoch_count": _mean([float(row["gate_passing_epoch_count"]) for row in aggregate_rows]),
        "selected_epoch_passes_gate_rate": _mean(
            [1.0 if bool(row["selected_epoch_passes_gate"]) else 0.0 for row in aggregate_rows]
        ),
        "selector_fallback_used_rate": _mean(
            [1.0 if bool(row["selector_fallback_used"]) else 0.0 for row in aggregate_rows]
        ),
        "mean_val_report_output_mse": _mean([float(row["val_report_output_mse"]) for row in aggregate_rows]),
        "std_val_report_output_mse": _std([float(row["val_report_output_mse"]) for row in aggregate_rows]),
        "mean_test_report_output_mse": _mean([float(row["test_report_output_mse"]) for row in aggregate_rows]),
        "std_test_report_output_mse": _std([float(row["test_report_output_mse"]) for row in aggregate_rows]),
        "mean_val_transported_final_energy": _mean(
            [float(row["val_transported_final_energy"]) for row in aggregate_rows]
        ),
        "mean_selected_epoch": _mean([float(row["selected_epoch"]) for row in aggregate_rows]),
        "mean_runtime_proxy_seconds": _mean([float(row["runtime_proxy_seconds"]) for row in aggregate_rows]),
    }

    validation_breakdown = _knot_metric_stats(
        knot_rows,
        "validation",
        _KNOT_GAP_METRICS
        + tuple(f"transported_{metric}" for metric in _KNOT_INTERFACE_METRICS)
        + tuple(f"slow_pc_{metric}" for metric in _KNOT_INTERFACE_METRICS),
    )
    test_breakdown = _knot_metric_stats(
        knot_rows,
        "test",
        _KNOT_GAP_METRICS
        + tuple(f"transported_{metric}" for metric in _KNOT_INTERFACE_METRICS)
        + tuple(f"slow_pc_{metric}" for metric in _KNOT_INTERFACE_METRICS),
    )
    validation_breakdown["peak_rowspace_gap_knot_index"] = int(
        np.argmax(np.asarray(validation_breakdown["mean_hidden_state_rms_gap_rowspace_by_knot"], dtype=np.float64))
    )
    validation_breakdown["peak_rowspace_gap_region"] = _region_from_peak_index(
        int(validation_breakdown["peak_rowspace_gap_knot_index"]),
        len(validation_breakdown["rollout_knot_indices"]) - 1,
    )
    validation_breakdown["peak_output_gap_knot_index"] = int(
        np.argmax(np.asarray(validation_breakdown["mean_output_state_rms_gap_to_slow_pc_by_knot"], dtype=np.float64))
    )
    validation_breakdown["peak_output_gap_region"] = _region_from_peak_index(
        int(validation_breakdown["peak_output_gap_knot_index"]),
        len(validation_breakdown["rollout_knot_indices"]) - 1,
    )
    test_breakdown["peak_rowspace_gap_knot_index"] = int(
        np.argmax(np.asarray(test_breakdown["mean_hidden_state_rms_gap_rowspace_by_knot"], dtype=np.float64))
    )
    test_breakdown["peak_rowspace_gap_region"] = _region_from_peak_index(
        int(test_breakdown["peak_rowspace_gap_knot_index"]),
        len(test_breakdown["rollout_knot_indices"]) - 1,
    )
    test_breakdown["peak_output_gap_knot_index"] = int(
        np.argmax(np.asarray(test_breakdown["mean_output_state_rms_gap_to_slow_pc_by_knot"], dtype=np.float64))
    )
    test_breakdown["peak_output_gap_region"] = _region_from_peak_index(
        int(test_breakdown["peak_output_gap_knot_index"]),
        len(test_breakdown["rollout_knot_indices"]) - 1,
    )

    diagnosis, diagnosis_evidence, recommended_next_move = _diagnose_and_recommend(
        config,
        validation_breakdown,
        test_breakdown,
    )

    summary = {
        "phase": "FMPC Stage 04 Incremental Bridge",
        "stage": "adopted_package_basis_drift_source_localization",
        "num_runs": len(aggregate_rows),
        "integrated_behavior": integrated_behavior,
        "validation_knot_breakdown": validation_breakdown,
        "test_knot_breakdown": test_breakdown,
        "diagnosis": diagnosis,
        "diagnosis_evidence": diagnosis_evidence,
        "recommended_next_narrow_tf2_move": recommended_next_move,
        "aggregate_runs_csv_path": "aggregate_runs.csv",
        "knotwise_localization_csv_path": "knotwise_localization.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF2BasisDriftLocalizationSuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config),
        aggregate_rows=aggregate_rows,
        knot_rows=knot_rows,
        summary=summary,
    )
