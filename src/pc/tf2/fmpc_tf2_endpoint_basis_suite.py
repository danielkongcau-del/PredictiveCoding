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
from ..tf1.fmpc_tf1_flow import build_tf1_context, hidden_states_from_state, rollout_hidden_transport
from .fmpc_tf2 import (
    FMPCTF2Config,
    _learned_velocity_fn,
    build_tf2_corrective_transport_terminal_angleclip_default_config,
    run_fmpc_tf2_experiment,
)
from .fmpc_tf2_readout_refit_suite import _build_feature_bundle, _load_reference_context
from ..metrics import classification_accuracy, regression_mse


_GEOMETRY_METRICS = (
    "frozen_head_accuracy",
    "frozen_head_output_mse",
    "mean_pairwise_centroid_distance",
    "min_pairwise_centroid_distance",
    "mean_within_class_rms",
    "mean_within_class_variance",
    "between_class_centroid_margin",
    "fisher_separability_ratio",
    "nearest_centroid_accuracy",
    "mean_top1_margin",
    "std_top1_margin",
    "mean_true_class_margin",
    "std_true_class_margin",
)

_DELTA_METRICS = (
    "delta_h_rms_total",
    "delta_h_rms_rowspace",
    "delta_h_rms_orthogonal",
    "delta_h_rowspace_fraction",
    "delta_h_orthogonal_fraction",
    "centroid_displacement_rms_total",
    "centroid_displacement_rms_rowspace",
    "centroid_displacement_rms_orthogonal",
    "centroid_displacement_rowspace_fraction",
    "centroid_displacement_orthogonal_fraction",
)


@dataclass
class FMPCTF2EndpointBasisSuiteConfig:
    """Run a narrow adopted-package endpoint-basis / separability diagnostic."""

    experiment_name: str = "fmpc_tf2_endpoint_basis_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 60
    batch_size: int = 128
    eval_steps: int = 15
    layer_dims: tuple[int, ...] = (64, 64, 10)
    reference_summary_path: str | Path = "outputs/tf2/fmpc_tf2_gap_decomposition_suite/aggregate_summary.json"
    rowspace_fraction_threshold: float = 0.55
    centroid_rowspace_fraction_threshold: float = 0.55
    between_margin_ratio_threshold: float = 0.90
    within_spread_ratio_threshold: float = 1.10

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"


@dataclass
class FMPCTF2EndpointBasisSuiteRunResult:
    run_dir: Path
    config: dict[str, Any]
    aggregate_rows: list[dict[str, Any]]
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


def _mean_list(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("vectors must contain at least one element.")
    width = len(vectors[0])
    if any(len(vector) != width for vector in vectors):
        raise ValueError("All vectors must share the same width.")
    array = np.asarray(vectors, dtype=np.float64)
    return [float(value) for value in np.mean(array, axis=0)]


def _suite_config_payload(config: FMPCTF2EndpointBasisSuiteConfig) -> dict[str, Any]:
    return {
        "phase": "Phase TF2",
        "stage": "adopted_package_endpoint_basis_separability",
        "base_preset_name": "tf2_corrective_transport_terminal_angleclip_default",
        "transport_family_fixed": True,
        "validation_only_selection": True,
        "report_scope": {
            "integrated_behavior_metrics": True,
            "hidden_to_output_interface_geometry": True,
            "readout_rowspace_delta_decomposition": True,
            "validation_knot_breakdown": True,
        },
        "seeds": [int(seed) for seed in config.seeds],
        "thresholds": {
            "rowspace_fraction_threshold": float(config.rowspace_fraction_threshold),
            "centroid_rowspace_fraction_threshold": float(config.centroid_rowspace_fraction_threshold),
            "between_margin_ratio_threshold": float(config.between_margin_ratio_threshold),
            "within_spread_ratio_threshold": float(config.within_spread_ratio_threshold),
        },
    }


def _run_id(seed: int) -> str:
    return f"adopted_s{seed}"


def _class_indices(y: np.ndarray) -> np.ndarray:
    return np.argmax(np.asarray(y, dtype=np.float64), axis=1)


def _apply_readout(features: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.asarray(features, dtype=np.float64) @ np.asarray(weight, dtype=np.float64).T + np.asarray(
        bias, dtype=np.float64
    )


def _centroids(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features_array = np.asarray(features, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)
    classes = np.unique(labels_array)
    centroids = np.stack([features_array[labels_array == cls].mean(axis=0) for cls in classes], axis=0)
    return classes, centroids


def _pairwise_centroid_distances(centroids: np.ndarray) -> np.ndarray:
    if centroids.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    distances: list[float] = []
    for i in range(centroids.shape[0]):
        for j in range(i + 1, centroids.shape[0]):
            distances.append(float(np.linalg.norm(centroids[i] - centroids[j])))
    return np.asarray(distances, dtype=np.float64)


def _within_class_rms(
    features: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    centroids: np.ndarray,
) -> tuple[np.ndarray, float]:
    per_class: list[float] = []
    features_array = np.asarray(features, dtype=np.float64)
    for class_index, cls in enumerate(classes):
        class_points = features_array[labels == cls]
        deltas = class_points - centroids[class_index]
        rms = float(np.sqrt(np.mean(np.sum(deltas * deltas, axis=1))))
        per_class.append(rms)
    mean_rms = float(np.mean(per_class)) if per_class else 0.0
    return np.asarray(per_class, dtype=np.float64), mean_rms


def _nearest_centroid_accuracy(features: np.ndarray, labels: np.ndarray, classes: np.ndarray, centroids: np.ndarray) -> float:
    features_array = np.asarray(features, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)
    distances = np.linalg.norm(features_array[:, None, :] - centroids[None, :, :], axis=2)
    predictions = classes[np.argmin(distances, axis=1)]
    return float(np.mean(predictions == labels_array))


def _top1_top2_margins(logits: np.ndarray) -> np.ndarray:
    logits_array = np.asarray(logits, dtype=np.float64)
    sorted_logits = np.sort(logits_array, axis=1)
    return sorted_logits[:, -1] - sorted_logits[:, -2]


def _true_class_margins(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    logits_array = np.asarray(logits, dtype=np.float64)
    label_indices = np.asarray(labels, dtype=np.int64)
    target_logits = logits_array[np.arange(logits_array.shape[0]), label_indices]
    masked = logits_array.copy()
    masked[np.arange(masked.shape[0]), label_indices] = -np.inf
    other_max = np.max(masked, axis=1)
    return target_logits - other_max


def _rowspace_basis(weight: np.ndarray) -> np.ndarray:
    weight_array = np.asarray(weight, dtype=np.float64)
    _, singular_values, vh = np.linalg.svd(weight_array, full_matrices=False)
    rank = int(np.sum(singular_values > 1e-12))
    if rank == 0:
        return np.zeros((weight_array.shape[1], 0), dtype=np.float64)
    return vh[:rank].T.copy()


def _project_rowspace(vectors: np.ndarray, basis: np.ndarray) -> np.ndarray:
    vectors_array = np.asarray(vectors, dtype=np.float64)
    if basis.shape[1] == 0:
        return np.zeros_like(vectors_array)
    return (vectors_array @ basis) @ basis.T


def _rms_norm(vectors: np.ndarray) -> float:
    vectors_array = np.asarray(vectors, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(vectors_array * vectors_array, axis=1))))


def _interface_geometry(features: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> dict[str, float]:
    labels = _class_indices(y)
    classes, centroids = _centroids(features, labels)
    pairwise = _pairwise_centroid_distances(centroids)
    _, mean_within_rms = _within_class_rms(features, labels, classes, centroids)
    logits = _apply_readout(features, weight, bias)
    top1_margins = _top1_top2_margins(logits)
    true_margins = _true_class_margins(logits, labels)
    mean_pairwise = float(np.mean(pairwise)) if pairwise.size > 0 else 0.0
    min_pairwise = float(np.min(pairwise)) if pairwise.size > 0 else 0.0
    fisher_ratio = 0.0 if mean_within_rms <= 0.0 else float(mean_pairwise / mean_within_rms)
    return {
        "frozen_head_accuracy": float(classification_accuracy(logits, y)),
        "frozen_head_output_mse": float(regression_mse(logits, y)),
        "mean_pairwise_centroid_distance": mean_pairwise,
        "min_pairwise_centroid_distance": min_pairwise,
        "mean_within_class_rms": float(mean_within_rms),
        "mean_within_class_variance": float(mean_within_rms**2),
        "between_class_centroid_margin": min_pairwise,
        "fisher_separability_ratio": fisher_ratio,
        "nearest_centroid_accuracy": _nearest_centroid_accuracy(features, labels, classes, centroids),
        "mean_top1_margin": float(np.mean(top1_margins)),
        "std_top1_margin": float(np.std(top1_margins)),
        "mean_true_class_margin": float(np.mean(true_margins)),
        "std_true_class_margin": float(np.std(true_margins)),
    }


def _delta_geometry(
    transported: np.ndarray,
    slow_pc: np.ndarray,
    y: np.ndarray,
    basis: np.ndarray,
) -> dict[str, float]:
    transported_array = np.asarray(transported, dtype=np.float64)
    slow_pc_array = np.asarray(slow_pc, dtype=np.float64)
    labels = _class_indices(y)
    delta = transported_array - slow_pc_array
    delta_row = _project_rowspace(delta, basis)
    delta_orth = delta - delta_row
    delta_total_rms = _rms_norm(delta)
    delta_row_rms = _rms_norm(delta_row)
    delta_orth_rms = _rms_norm(delta_orth)

    classes_t, centroids_t = _centroids(transported_array, labels)
    classes_s, centroids_s = _centroids(slow_pc_array, labels)
    if not np.array_equal(classes_t, classes_s):
        raise ValueError("Transported and slow-PC centroid classes must align.")
    centroid_delta = centroids_t - centroids_s
    centroid_row = _project_rowspace(centroid_delta, basis)
    centroid_orth = centroid_delta - centroid_row
    centroid_total_rms = _rms_norm(centroid_delta)
    centroid_row_rms = _rms_norm(centroid_row)
    centroid_orth_rms = _rms_norm(centroid_orth)

    return {
        "delta_h_rms_total": delta_total_rms,
        "delta_h_rms_rowspace": delta_row_rms,
        "delta_h_rms_orthogonal": delta_orth_rms,
        "delta_h_rowspace_fraction": 0.0 if delta_total_rms <= 0.0 else float(delta_row_rms / delta_total_rms),
        "delta_h_orthogonal_fraction": 0.0
        if delta_total_rms <= 0.0
        else float(delta_orth_rms / delta_total_rms),
        "centroid_displacement_rms_total": centroid_total_rms,
        "centroid_displacement_rms_rowspace": centroid_row_rms,
        "centroid_displacement_rms_orthogonal": centroid_orth_rms,
        "centroid_displacement_rowspace_fraction": (
            0.0 if centroid_total_rms <= 0.0 else float(centroid_row_rms / centroid_total_rms)
        ),
        "centroid_displacement_orthogonal_fraction": (
            0.0 if centroid_total_rms <= 0.0 else float(centroid_orth_rms / centroid_total_rms)
        ),
    }


def _transport_penultimate_by_knot(
    model: Any,
    psi_network: Any,
    config: FMPCTF2Config,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[list[np.ndarray], list[float]]:
    context = build_tf1_context(model, x, y)
    velocity_fn = _learned_velocity_fn(context, psi_network, config)
    rollout = rollout_hidden_transport(
        context,
        context.z0,
        transport_steps=int(config.micro_steps),
        mode="learned",
        velocity_fn=velocity_fn,
    )
    penultimate_by_knot = [
        np.asarray(hidden_states_from_state(context, z_knot)[-2], dtype=np.float64) for z_knot in rollout.z_knots
    ]
    return penultimate_by_knot, [float(value) for value in rollout.knot_times.tolist()]


def _runtime_proxy_seconds(summary: dict[str, Any]) -> float:
    timing = dict(summary.get("timing", {}))
    return float(timing.get("train_wall_time_seconds", 0.0)) + float(
        timing.get("final_evaluation_wall_time_seconds", 0.0)
    )


def _flatten_prefixed(prefix: str, payload: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in payload.items()}


def _summary_stats(values: list[float]) -> dict[str, float]:
    return {"mean": _mean(values), "std": _std(values)}


def _prefixed_metric_stats(rows: list[dict[str, Any]], prefix: str, metrics: tuple[str, ...]) -> dict[str, dict[str, float]]:
    return {
        metric: _summary_stats([float(row[f"{prefix}_{metric}"]) for row in rows])
        for metric in metrics
    }


def _pairwise_prefix_delta(
    rows: list[dict[str, Any]],
    left_prefix: str,
    right_prefix: str,
    metrics: tuple[str, ...],
) -> dict[str, float]:
    return {
        f"mean_{metric}_delta": _mean(
            [float(row[f"{left_prefix}_{metric}"]) - float(row[f"{right_prefix}_{metric}"]) for row in rows]
        )
        for metric in metrics
    }


def _region_from_peak_index(peak_index: int, last_index: int) -> str:
    if peak_index <= 1:
        return "early_knot"
    if peak_index >= last_index:
        return "terminal_knot"
    return "mid_knot"


def _report_only_delta_vs_reference(control_summary: dict[str, Any], reference_summary: dict[str, Any]) -> dict[str, float]:
    return {
        "mean_val_accuracy_delta": float(control_summary["mean_val_accuracy"] - reference_summary["mean_val_accuracy"]),
        "mean_test_accuracy_delta": float(control_summary["mean_test_accuracy"] - reference_summary["mean_test_accuracy"]),
        "mean_val_report_output_mse_delta": float(
            control_summary["mean_val_report_output_mse"] - reference_summary["mean_val_report_output_mse"]
        ),
        "mean_test_report_output_mse_delta": float(
            control_summary["mean_test_report_output_mse"] - reference_summary["mean_test_report_output_mse"]
        ),
        "mean_runtime_proxy_seconds_delta": float(
            control_summary["mean_runtime_proxy_seconds"] - reference_summary["mean_runtime_proxy_seconds"]
        ),
    }


def _diagnose_and_recommend(
    config: FMPCTF2EndpointBasisSuiteConfig,
    transported_val: dict[str, dict[str, float]],
    slow_pc_val: dict[str, dict[str, float]],
    delta_val: dict[str, dict[str, float]],
) -> tuple[str, dict[str, float], str]:
    between_ratio = float(
        transported_val["between_class_centroid_margin"]["mean"]
        / max(1e-12, slow_pc_val["between_class_centroid_margin"]["mean"])
    )
    within_ratio = float(
        transported_val["mean_within_class_rms"]["mean"] / max(1e-12, slow_pc_val["mean_within_class_rms"]["mean"])
    )
    fisher_ratio = float(
        transported_val["fisher_separability_ratio"]["mean"]
        / max(1e-12, slow_pc_val["fisher_separability_ratio"]["mean"])
    )
    rowspace_fraction = float(delta_val["delta_h_rowspace_fraction"]["mean"])
    centroid_rowspace_fraction = float(delta_val["centroid_displacement_rowspace_fraction"]["mean"])
    nc_gap = float(
        transported_val["nearest_centroid_accuracy"]["mean"] - slow_pc_val["nearest_centroid_accuracy"]["mean"]
    )

    evidence = {
        "between_margin_ratio_transport_vs_slow_pc": between_ratio,
        "within_class_spread_ratio_transport_vs_slow_pc": within_ratio,
        "fisher_ratio_transport_vs_slow_pc": fisher_ratio,
        "delta_h_rowspace_fraction": rowspace_fraction,
        "centroid_displacement_rowspace_fraction": centroid_rowspace_fraction,
        "nearest_centroid_accuracy_gap_transport_minus_slow_pc": nc_gap,
    }

    rowspace_dominant = (
        rowspace_fraction >= float(config.rowspace_fraction_threshold)
        or centroid_rowspace_fraction >= float(config.centroid_rowspace_fraction_threshold)
    )
    reduced_between_margin = between_ratio <= float(config.between_margin_ratio_threshold)
    inflated_within_spread = within_ratio >= float(config.within_spread_ratio_threshold)

    if rowspace_dominant and not inflated_within_spread:
        return (
            "distortion_in_readout_relevant_row_space",
            evidence,
            "run one narrow adopted-package readout-sensitive / output-sensitive terminal direction diagnostic inside the current package",
        )
    if reduced_between_margin and not inflated_within_spread and not rowspace_dominant:
        return (
            "reduced_between_class_margin",
            evidence,
            "run one narrow adopted-package terminal class-margin preservation diagnostic without changing the transport family",
        )
    if inflated_within_spread and not reduced_between_margin and not rowspace_dominant:
        return (
            "inflated_within_class_spread",
            evidence,
            "run one narrow adopted-package terminal representation contraction / separability stabilization diagnostic without changing the transport family",
        )
    return (
        "mixed_picture_between_margin_within_class_spread_and_row_space_distortion",
        evidence,
        "run one narrow adopted-package readout-sensitive terminal separability diagnostic, starting from row-space-sensitive terminal direction control",
    )


def run_fmpc_tf2_endpoint_basis_suite(
    config: FMPCTF2EndpointBasisSuiteConfig,
) -> FMPCTF2EndpointBasisSuiteRunResult:
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
    val_knot_times: list[list[float]] = []
    val_knot_series: dict[str, list[list[float]]] = {
        "fisher_separability_ratio": [],
        "mean_within_class_rms": [],
        "between_class_centroid_margin": [],
        "nearest_centroid_accuracy": [],
        "mean_true_class_margin": [],
        "delta_h_rms_rowspace": [],
        "delta_h_rowspace_fraction": [],
        "delta_h_rms_total": [],
        "fisher_gap_to_slow_pc": [],
        "within_class_rms_delta_vs_slow_pc": [],
        "between_margin_gap_to_slow_pc": [],
        "nearest_centroid_accuracy_gap_to_slow_pc": [],
    }
    val_peak_degradation_indices: list[float] = []

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
            raise ValueError("Endpoint-basis suite requires runtime model and psi network objects.")

        split = load_digits_split(
            split_seed=int(tf2_config.data_seed),
            train_fraction=float(tf2_config.train_fraction),
            val_fraction=float(tf2_config.val_fraction),
            test_fraction=float(tf2_config.test_fraction),
        )

        val_bundle = _build_feature_bundle(result.model, result.psi_network, tf2_config, split.x_val, split.y_val)
        test_bundle = _build_feature_bundle(result.model, result.psi_network, tf2_config, split.x_test, split.y_test)

        weight = np.asarray(result.model.layers[-1].weight, dtype=np.float64)
        bias = np.asarray(result.model.layers[-1].bias, dtype=np.float64)
        basis = _rowspace_basis(weight)

        transported_val = _interface_geometry(val_bundle.transported_penultimate, split.y_val, weight, bias)
        slow_pc_val = _interface_geometry(val_bundle.slow_pc_penultimate, split.y_val, weight, bias)
        transported_test = _interface_geometry(test_bundle.transported_penultimate, split.y_test, weight, bias)
        slow_pc_test = _interface_geometry(test_bundle.slow_pc_penultimate, split.y_test, weight, bias)
        delta_val = _delta_geometry(val_bundle.transported_penultimate, val_bundle.slow_pc_penultimate, split.y_val, basis)
        delta_test = _delta_geometry(
            test_bundle.transported_penultimate,
            test_bundle.slow_pc_penultimate,
            split.y_test,
            basis,
        )

        knot_features, knot_times = _transport_penultimate_by_knot(
            result.model,
            result.psi_network,
            tf2_config,
            split.x_val,
            split.y_val,
        )
        knot_geometries = [_interface_geometry(features, split.y_val, weight, bias) for features in knot_features]
        knot_deltas = [
            _delta_geometry(features, val_bundle.slow_pc_penultimate, split.y_val, basis) for features in knot_features
        ]
        fisher_gaps_to_slow_pc = [
            float(slow_pc_val["fisher_separability_ratio"] - geometry["fisher_separability_ratio"])
            for geometry in knot_geometries
        ]
        within_deltas_vs_slow_pc = [
            float(geometry["mean_within_class_rms"] - slow_pc_val["mean_within_class_rms"]) for geometry in knot_geometries
        ]
        between_gaps_to_slow_pc = [
            float(slow_pc_val["between_class_centroid_margin"] - geometry["between_class_centroid_margin"])
            for geometry in knot_geometries
        ]
        nc_gaps_to_slow_pc = [
            float(geometry["nearest_centroid_accuracy"] - slow_pc_val["nearest_centroid_accuracy"])
            for geometry in knot_geometries
        ]
        peak_fisher_gap_index = int(np.argmax(np.asarray(fisher_gaps_to_slow_pc, dtype=np.float64)))

        val_knot_times.append([float(value) for value in knot_times])
        val_knot_series["fisher_separability_ratio"].append(
            [float(geometry["fisher_separability_ratio"]) for geometry in knot_geometries]
        )
        val_knot_series["mean_within_class_rms"].append(
            [float(geometry["mean_within_class_rms"]) for geometry in knot_geometries]
        )
        val_knot_series["between_class_centroid_margin"].append(
            [float(geometry["between_class_centroid_margin"]) for geometry in knot_geometries]
        )
        val_knot_series["nearest_centroid_accuracy"].append(
            [float(geometry["nearest_centroid_accuracy"]) for geometry in knot_geometries]
        )
        val_knot_series["mean_true_class_margin"].append(
            [float(geometry["mean_true_class_margin"]) for geometry in knot_geometries]
        )
        val_knot_series["delta_h_rms_rowspace"].append(
            [float(delta["delta_h_rms_rowspace"]) for delta in knot_deltas]
        )
        val_knot_series["delta_h_rowspace_fraction"].append(
            [float(delta["delta_h_rowspace_fraction"]) for delta in knot_deltas]
        )
        val_knot_series["delta_h_rms_total"].append([float(delta["delta_h_rms_total"]) for delta in knot_deltas])
        val_knot_series["fisher_gap_to_slow_pc"].append(fisher_gaps_to_slow_pc)
        val_knot_series["within_class_rms_delta_vs_slow_pc"].append(within_deltas_vs_slow_pc)
        val_knot_series["between_margin_gap_to_slow_pc"].append(between_gaps_to_slow_pc)
        val_knot_series["nearest_centroid_accuracy_gap_to_slow_pc"].append(nc_gaps_to_slow_pc)
        val_peak_degradation_indices.append(float(peak_fisher_gap_index))

        summary = result.summary
        row = {
            "seed": int(seed),
            "run_id": str(result.run_dir.name),
            "run_summary_path": _relative_posix(run_dir, result.run_dir / "summary.json"),
            "selected_epoch": int(summary["best_epoch"]),
            "integrated_val_accuracy": float(summary["val_accuracy"]),
            "integrated_test_accuracy": float(summary["test_accuracy"]),
            "integrated_val_report_output_mse": float(summary["val_loss"]),
            "integrated_test_report_output_mse": float(summary["test_loss"]),
            "integrated_gate_passing_epoch_count": int(summary["gate_passing_epoch_count"]),
            "integrated_selected_epoch_passes_gate": bool(summary["selected_epoch_passes_gate"]),
            "integrated_selector_fallback_used": bool(summary["selector_fallback_used"]),
            "integrated_val_transported_final_energy": float(summary["val_transported_final_energy"]),
            "integrated_runtime_proxy_seconds": _runtime_proxy_seconds(summary),
            "val_knot_peak_fisher_gap_knot_index": float(peak_fisher_gap_index),
            "val_knot_peak_fisher_gap_region": _region_from_peak_index(peak_fisher_gap_index, len(knot_times) - 1),
        }
        row.update(_flatten_prefixed("transported_val", transported_val))
        row.update(_flatten_prefixed("slow_pc_val", slow_pc_val))
        row.update(_flatten_prefixed("transported_test", transported_test))
        row.update(_flatten_prefixed("slow_pc_test", slow_pc_test))
        row.update(_flatten_prefixed("val", delta_val))
        row.update(_flatten_prefixed("test", delta_test))
        aggregate_rows.append(row)

    _write_csv(run_dir / "aggregate_runs.csv", aggregate_rows)

    integrated_summary = {
        "mean_val_accuracy": _mean([float(row["integrated_val_accuracy"]) for row in aggregate_rows]),
        "std_val_accuracy": _std([float(row["integrated_val_accuracy"]) for row in aggregate_rows]),
        "mean_test_accuracy": _mean([float(row["integrated_test_accuracy"]) for row in aggregate_rows]),
        "std_test_accuracy": _std([float(row["integrated_test_accuracy"]) for row in aggregate_rows]),
        "mean_val_report_output_mse": _mean([float(row["integrated_val_report_output_mse"]) for row in aggregate_rows]),
        "std_val_report_output_mse": _std([float(row["integrated_val_report_output_mse"]) for row in aggregate_rows]),
        "mean_test_report_output_mse": _mean([float(row["integrated_test_report_output_mse"]) for row in aggregate_rows]),
        "std_test_report_output_mse": _std([float(row["integrated_test_report_output_mse"]) for row in aggregate_rows]),
        "mean_gate_passing_epoch_count": _mean(
            [float(row["integrated_gate_passing_epoch_count"]) for row in aggregate_rows]
        ),
        "selected_epoch_passes_gate_rate": _mean(
            [1.0 if bool(row["integrated_selected_epoch_passes_gate"]) else 0.0 for row in aggregate_rows]
        ),
        "selector_fallback_used_rate": _mean(
            [1.0 if bool(row["integrated_selector_fallback_used"]) else 0.0 for row in aggregate_rows]
        ),
        "mean_selected_epoch": _mean([float(row["selected_epoch"]) for row in aggregate_rows]),
        "mean_val_transported_final_energy": _mean(
            [float(row["integrated_val_transported_final_energy"]) for row in aggregate_rows]
        ),
        "mean_runtime_proxy_seconds": _mean([float(row["integrated_runtime_proxy_seconds"]) for row in aggregate_rows]),
    }

    endpoint_geometry = {
        "transported": {
            "validation": _prefixed_metric_stats(aggregate_rows, "transported_val", _GEOMETRY_METRICS),
            "test": _prefixed_metric_stats(aggregate_rows, "transported_test", _GEOMETRY_METRICS),
        },
        "slow_pc": {
            "validation": _prefixed_metric_stats(aggregate_rows, "slow_pc_val", _GEOMETRY_METRICS),
            "test": _prefixed_metric_stats(aggregate_rows, "slow_pc_test", _GEOMETRY_METRICS),
        },
    }
    delta_geometry = {
        "validation": _prefixed_metric_stats(aggregate_rows, "val", _DELTA_METRICS),
        "test": _prefixed_metric_stats(aggregate_rows, "test", _DELTA_METRICS),
    }
    pairwise_endpoint_deltas = {
        "validation_transported_minus_slow_pc": _pairwise_prefix_delta(
            aggregate_rows,
            "transported_val",
            "slow_pc_val",
            _GEOMETRY_METRICS,
        ),
        "test_transported_minus_slow_pc": _pairwise_prefix_delta(
            aggregate_rows,
            "transported_test",
            "slow_pc_test",
            _GEOMETRY_METRICS,
        ),
    }

    knot_breakdown = {
        "rollout_knot_indices": list(range(len(val_knot_times[0]))),
        "rollout_knot_times": _mean_list(val_knot_times),
        "mean_fisher_separability_ratio_by_knot": _mean_list(val_knot_series["fisher_separability_ratio"]),
        "mean_within_class_rms_by_knot": _mean_list(val_knot_series["mean_within_class_rms"]),
        "mean_between_class_centroid_margin_by_knot": _mean_list(val_knot_series["between_class_centroid_margin"]),
        "mean_nearest_centroid_accuracy_by_knot": _mean_list(val_knot_series["nearest_centroid_accuracy"]),
        "mean_true_class_margin_by_knot": _mean_list(val_knot_series["mean_true_class_margin"]),
        "mean_delta_h_rms_rowspace_by_knot": _mean_list(val_knot_series["delta_h_rms_rowspace"]),
        "mean_delta_h_rowspace_fraction_by_knot": _mean_list(val_knot_series["delta_h_rowspace_fraction"]),
        "mean_delta_h_rms_total_by_knot": _mean_list(val_knot_series["delta_h_rms_total"]),
        "mean_fisher_gap_to_slow_pc_by_knot": _mean_list(val_knot_series["fisher_gap_to_slow_pc"]),
        "mean_within_class_rms_delta_vs_slow_pc_by_knot": _mean_list(
            val_knot_series["within_class_rms_delta_vs_slow_pc"]
        ),
        "mean_between_margin_gap_to_slow_pc_by_knot": _mean_list(val_knot_series["between_margin_gap_to_slow_pc"]),
        "mean_nearest_centroid_accuracy_gap_to_slow_pc_by_knot": _mean_list(
            val_knot_series["nearest_centroid_accuracy_gap_to_slow_pc"]
        ),
        "mean_peak_fisher_gap_knot_index": _mean(val_peak_degradation_indices),
    }
    knot_breakdown["peak_fisher_gap_knot_index"] = int(
        np.argmax(np.asarray(knot_breakdown["mean_fisher_gap_to_slow_pc_by_knot"], dtype=np.float64))
    )
    knot_breakdown["dominant_degradation_region"] = _region_from_peak_index(
        int(knot_breakdown["peak_fisher_gap_knot_index"]),
        len(knot_breakdown["rollout_knot_indices"]) - 1,
    )
    knot_breakdown["peak_endpoint_divergence_knot_index"] = int(
        np.argmax(np.asarray(knot_breakdown["mean_delta_h_rms_total_by_knot"], dtype=np.float64))
    )
    knot_breakdown["dominant_divergence_region"] = _region_from_peak_index(
        int(knot_breakdown["peak_endpoint_divergence_knot_index"]),
        len(knot_breakdown["rollout_knot_indices"]) - 1,
    )

    diagnosis, diagnosis_evidence, recommended_next_move = _diagnose_and_recommend(
        config=config,
        transported_val=endpoint_geometry["transported"]["validation"],
        slow_pc_val=endpoint_geometry["slow_pc"]["validation"],
        delta_val=delta_geometry["validation"],
    )

    reference_context = _load_reference_context(config.reference_summary_path)
    report_only_reference: dict[str, Any] | None = None
    if reference_context is not None:
        by_method = dict(reference_context.get("by_method", {}))
        slow_pc_reference = by_method.get("canonical_slow_pc_digits_baseline")
        historical_reference = by_method.get("tf2_corrective_transport_default")
        report_only_reference = {
            "canonical_slow_pc_digits_baseline": slow_pc_reference,
            "historical_corrective_reference": historical_reference,
            "control_vs_canonical_slow_pc_digits_baseline": (
                None if slow_pc_reference is None else _report_only_delta_vs_reference(integrated_summary, slow_pc_reference)
            ),
            "control_vs_historical_corrective_reference": (
                None
                if historical_reference is None
                else _report_only_delta_vs_reference(integrated_summary, historical_reference)
            ),
        }

    summary = {
        "phase": "Phase TF2",
        "stage": "adopted_package_endpoint_basis_separability",
        "num_runs": len(aggregate_rows),
        "reference_context_reused": reference_context is not None,
        "reference_context_path": None if reference_context is None else str(Path(config.reference_summary_path).as_posix()),
        "integrated_behavior": integrated_summary,
        "endpoint_cases": endpoint_geometry,
        "delta_geometry": delta_geometry,
        "pairwise_endpoint_deltas": pairwise_endpoint_deltas,
        "validation_knot_breakdown": knot_breakdown,
        "diagnosis": diagnosis,
        "diagnosis_evidence": diagnosis_evidence,
        "recommended_next_narrow_tf2_move": recommended_next_move,
        "report_only_external_reference": report_only_reference,
        "aggregate_csv_path": "aggregate_runs.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF2EndpointBasisSuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config),
        aggregate_rows=aggregate_rows,
        summary=summary,
    )
