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

from ..datasets import load_digits_split
from ..stage_03_transport_core_v1.fmpc_tf1_flow import build_tf1_context, hidden_states_from_state, rollout_hidden_transport
from .fmpc_tf2 import (
    FMPCTF2Config,
    _evaluate_slow_pc_accuracy,
    _learned_velocity_fn,
    _restore_pc_parameters,
    _snapshot_pc_parameters,
    build_tf2_corrective_transport_terminal_angleclip_default_config,
    run_fmpc_tf2_experiment,
)
from .fmpc_tf2_gap_decomposition_suite import _evaluate_tf2_supervised_alignment
from ..inference import run_teacher_inference_export
from ..metrics import classification_accuracy, regression_mse

_RefitBasis = Literal["none", "transported_endpoints", "slow_pc_endpoints"]


@dataclass(frozen=True)
class _CaseSpec:
    case_name: str
    description: str
    refit_basis: _RefitBasis


@dataclass(frozen=True)
class _ReadoutSelection:
    ridge: float
    val_accuracy: float
    val_output_mse: float
    solver: str


@dataclass(frozen=True)
class _ReadoutParameters:
    weight: np.ndarray
    bias: np.ndarray
    selection: _ReadoutSelection


@dataclass(frozen=True)
class _EndpointFeatureBundle:
    predict_penultimate: np.ndarray
    transported_penultimate: np.ndarray
    slow_pc_penultimate: np.ndarray


@dataclass
class FMPCTF2ReadoutRefitSuiteConfig:
    """Run a narrow adopted-package readout-refit / endpoint-separability diagnostic."""

    experiment_name: str = "fmpc_tf2_readout_refit_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 60
    batch_size: int = 128
    eval_steps: int = 15
    layer_dims: tuple[int, ...] = (64, 64, 10)
    readout_ridge_grid: tuple[float, ...] = (0.0, 1e-4, 1e-2)
    material_test_gain_threshold: float = 0.005
    material_supervised_output_mse_gain_threshold: float = 0.001
    reference_summary_path: str | Path = "outputs/stage_04_incremental_bridge/fmpc_tf2_gap_decomposition_suite/aggregate_summary.json"

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"

    def case_specs(self) -> tuple[_CaseSpec, ...]:
        return (
            _CaseSpec(
                case_name="adopted_control",
                description="Current adopted package with the integrated readout head unchanged.",
                refit_basis="none",
            ),
            _CaseSpec(
                case_name="transported_endpoint_readout_refit",
                description="Refit only the output layer on frozen transported endpoints from the selected adopted model.",
                refit_basis="transported_endpoints",
            ),
            _CaseSpec(
                case_name="slow_pc_endpoint_readout_refit",
                description="Refit only the output layer on frozen target-clamped slow-PC endpoints from the same selected adopted model.",
                refit_basis="slow_pc_endpoints",
            ),
        )


@dataclass
class FMPCTF2ReadoutRefitSuiteRunResult:
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


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _suite_config_payload(config: FMPCTF2ReadoutRefitSuiteConfig) -> dict[str, Any]:
    return {
        "phase": "FMPC Stage 04 Incremental Bridge",
        "stage": "adopted_package_readout_refit_endpoint_separability",
        "base_preset_name": "tf2_corrective_transport_terminal_angleclip_default",
        "diagnostic_protocol": {
            "transport_family_fixed": True,
            "hidden_parameters_frozen_during_readout_refit": True,
            "validation_only_readout_selection": True,
            "test_report_only": True,
            "report_metrics_for_refit_cases": "integrated_prediction_mode_with_refit_head",
            "supervised_metrics_for_refit_cases": "frozen_endpoint_basis_with_refit_head",
        },
        "case_specs": [
            {
                "case_name": spec.case_name,
                "description": spec.description,
                "refit_basis": spec.refit_basis,
            }
            for spec in config.case_specs()
        ],
        "seeds": [int(seed) for seed in config.seeds],
        "readout_ridge_grid": [float(value) for value in config.readout_ridge_grid],
        "thresholds": {
            "material_test_gain_threshold": float(config.material_test_gain_threshold),
            "material_supervised_output_mse_gain_threshold": float(config.material_supervised_output_mse_gain_threshold),
        },
    }


def _candidate_run_id(seed: int) -> str:
    return f"adopted_s{seed}"


def _load_reference_context(reference_summary_path: str | Path) -> dict[str, Any] | None:
    path = Path(reference_summary_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _penultimate_state(states: list[np.ndarray]) -> np.ndarray:
    if len(states) < 3:
        raise ValueError("TF2 readout-refit diagnostic expects at least one hidden layer.")
    return np.asarray(states[-2], dtype=np.float64)


def _predict_penultimate_state(model: Any, x: np.ndarray) -> np.ndarray:
    inference_result = model.infer(x, mode="predict", record_trace=False)
    return _penultimate_state(inference_result.states)


def _transport_penultimate_state(
    model: Any,
    psi_network: Any,
    config: FMPCTF2Config,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    context = build_tf1_context(model, x, y)
    velocity_fn = _learned_velocity_fn(context, psi_network, config)
    rollout = rollout_hidden_transport(
        context,
        context.z0,
        transport_steps=int(config.micro_steps),
        mode="learned",
        velocity_fn=velocity_fn,
    )
    return _penultimate_state(hidden_states_from_state(context, rollout.z_knots[-1]))


def _slow_pc_penultimate_state(model: Any, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    teacher_export = run_teacher_inference_export(
        model.layers,
        np.asarray(x, dtype=np.float64),
        y=np.asarray(y, dtype=np.float64),
        init=model.state_init,
        mode="train",
        eta_x=model.eta_x,
        steps=int(model.eval_steps),
        backend=str(model.inference_backend),
        record_trace=False,
        record_trajectory=False,
    )
    return _penultimate_state(teacher_export.final_states)


def _build_feature_bundle(
    model: Any,
    psi_network: Any,
    config: FMPCTF2Config,
    x: np.ndarray,
    y: np.ndarray,
) -> _EndpointFeatureBundle:
    return _EndpointFeatureBundle(
        predict_penultimate=_predict_penultimate_state(model, x),
        transported_penultimate=_transport_penultimate_state(model, psi_network, config, x, y),
        slow_pc_penultimate=_slow_pc_penultimate_state(model, x, y),
    )


def _design_matrix(features: np.ndarray) -> np.ndarray:
    features_array = np.asarray(features, dtype=np.float64)
    ones = np.ones((features_array.shape[0], 1), dtype=np.float64)
    return np.concatenate([features_array, ones], axis=1)


def _solve_linear_readout(features: np.ndarray, targets: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, str]:
    x_aug = _design_matrix(features)
    y_array = np.asarray(targets, dtype=np.float64)
    if ridge <= 0.0:
        beta, _, _, _ = np.linalg.lstsq(x_aug, y_array, rcond=None)
        solver = "least_squares"
    else:
        regularizer = np.eye(x_aug.shape[1], dtype=np.float64)
        regularizer[-1, -1] = 0.0
        beta = np.linalg.solve(x_aug.T @ x_aug + float(ridge) * regularizer, x_aug.T @ y_array)
        solver = "ridge_closed_form"
    weight = np.asarray(beta[:-1, :].T, dtype=np.float64)
    bias = np.asarray(beta[-1, :], dtype=np.float64)
    return weight, bias, solver


def _apply_linear_readout(features: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    features_array = np.asarray(features, dtype=np.float64)
    return features_array @ np.asarray(weight, dtype=np.float64).T + np.asarray(bias, dtype=np.float64)


def _fit_readout_with_validation(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    ridge_grid: tuple[float, ...],
) -> _ReadoutParameters:
    if not ridge_grid:
        raise ValueError("ridge_grid must contain at least one value.")
    best_tuple: tuple[float, float, float] | None = None
    best_payload: tuple[np.ndarray, np.ndarray, _ReadoutSelection] | None = None
    for ridge in ridge_grid:
        weight, bias, solver = _solve_linear_readout(train_features, train_targets, float(ridge))
        val_predictions = _apply_linear_readout(val_features, weight, bias)
        val_accuracy = classification_accuracy(val_predictions, val_targets)
        val_output_mse = regression_mse(val_predictions, val_targets)
        candidate_tuple = (-float(val_accuracy), float(val_output_mse), float(ridge))
        if best_tuple is None or candidate_tuple < best_tuple:
            best_tuple = candidate_tuple
            best_payload = (
                weight,
                bias,
                _ReadoutSelection(
                    ridge=float(ridge),
                    val_accuracy=float(val_accuracy),
                    val_output_mse=float(val_output_mse),
                    solver=solver,
                ),
            )
    if best_payload is None:
        raise ValueError("Expected at least one fitted readout candidate.")
    weight, bias, selection = best_payload
    return _ReadoutParameters(weight=weight, bias=bias, selection=selection)


def _supervised_basis_features(bundle: _EndpointFeatureBundle, basis: _RefitBasis) -> np.ndarray:
    if basis == "transported_endpoints":
        return bundle.transported_penultimate
    if basis == "slow_pc_endpoints":
        return bundle.slow_pc_penultimate
    raise ValueError(f"Unsupported supervised refit basis '{basis}'.")


def _runtime_proxy_seconds(summary: dict[str, Any]) -> float:
    timing = dict(summary.get("timing", {}))
    return float(timing.get("train_wall_time_seconds", 0.0)) + float(
        timing.get("final_evaluation_wall_time_seconds", 0.0)
    )


def _control_row(
    run_index: int,
    run_dir: Path,
    seed: int,
    result: Any,
    val_alignment: Any,
    test_alignment: Any,
) -> dict[str, Any]:
    summary = result.summary
    return {
        "run_index": int(run_index),
        "case_name": "adopted_control",
        "seed": int(seed),
        "run_id": str(result.run_dir.name),
        "run_summary_path": _relative_posix(run_dir, result.run_dir / "summary.json"),
        "refit_basis": "none",
        "readout_refit_selected_ridge": "",
        "readout_refit_selection_val_accuracy": "",
        "readout_refit_selection_val_output_mse": "",
        "readout_refit_solver": "",
        "selected_epoch": int(summary["best_epoch"]),
        "val_accuracy": float(summary["val_accuracy"]),
        "test_accuracy": float(summary["test_accuracy"]),
        "val_report_output_mse": float(summary["val_loss"]),
        "test_report_output_mse": float(summary["test_loss"]),
        "val_supervised_output_mse": float(val_alignment.transport_output_mse),
        "test_supervised_output_mse": float(test_alignment.transport_output_mse),
        "readout_refit_runtime_seconds": 0.0,
        "runtime_proxy_seconds": _runtime_proxy_seconds(summary),
    }


def _refit_row(
    run_index: int,
    run_dir: Path,
    case: _CaseSpec,
    seed: int,
    result: Any,
    readout: _ReadoutParameters,
    val_report_predictions: np.ndarray,
    test_report_predictions: np.ndarray,
    val_supervised_predictions: np.ndarray,
    test_supervised_predictions: np.ndarray,
    val_targets: np.ndarray,
    test_targets: np.ndarray,
    refit_runtime_seconds: float,
) -> dict[str, Any]:
    summary = result.summary
    return {
        "run_index": int(run_index),
        "case_name": case.case_name,
        "seed": int(seed),
        "run_id": str(result.run_dir.name),
        "run_summary_path": _relative_posix(run_dir, result.run_dir / "summary.json"),
        "refit_basis": case.refit_basis,
        "readout_refit_selected_ridge": float(readout.selection.ridge),
        "readout_refit_selection_val_accuracy": float(readout.selection.val_accuracy),
        "readout_refit_selection_val_output_mse": float(readout.selection.val_output_mse),
        "readout_refit_solver": str(readout.selection.solver),
        "selected_epoch": int(summary["best_epoch"]),
        "val_accuracy": float(classification_accuracy(val_report_predictions, val_targets)),
        "test_accuracy": float(classification_accuracy(test_report_predictions, test_targets)),
        "val_report_output_mse": float(regression_mse(val_report_predictions, val_targets)),
        "test_report_output_mse": float(regression_mse(test_report_predictions, test_targets)),
        "val_supervised_output_mse": float(regression_mse(val_supervised_predictions, val_targets)),
        "test_supervised_output_mse": float(regression_mse(test_supervised_predictions, test_targets)),
        "readout_refit_runtime_seconds": float(refit_runtime_seconds),
        "runtime_proxy_seconds": _runtime_proxy_seconds(summary) + float(refit_runtime_seconds),
    }


def _case_rows(rows: list[dict[str, Any]], case_name: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["case_name"]) == case_name]


def _case_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Case summary requires at least one row.")
    val_accuracies = [float(row["val_accuracy"]) for row in rows]
    test_accuracies = [float(row["test_accuracy"]) for row in rows]
    val_report_mse = [float(row["val_report_output_mse"]) for row in rows]
    test_report_mse = [float(row["test_report_output_mse"]) for row in rows]
    val_supervised_mse = [float(row["val_supervised_output_mse"]) for row in rows]
    test_supervised_mse = [float(row["test_supervised_output_mse"]) for row in rows]
    selected_epochs = [float(row["selected_epoch"]) for row in rows]
    runtime_values = [float(row["runtime_proxy_seconds"]) for row in rows]
    refit_runtime_values = [float(row["readout_refit_runtime_seconds"]) for row in rows]
    ridge_values = [
        float(value)
        for value in (_float_or_none(row["readout_refit_selected_ridge"]) for row in rows)
        if value is not None
    ]
    selection_val_acc = [
        float(value)
        for value in (_float_or_none(row["readout_refit_selection_val_accuracy"]) for row in rows)
        if value is not None
    ]
    selection_val_mse = [
        float(value)
        for value in (_float_or_none(row["readout_refit_selection_val_output_mse"]) for row in rows)
        if value is not None
    ]
    return {
        "num_runs": len(rows),
        "mean_val_accuracy": _mean(val_accuracies),
        "std_val_accuracy": _std(val_accuracies),
        "mean_test_accuracy": _mean(test_accuracies),
        "std_test_accuracy": _std(test_accuracies),
        "mean_val_report_output_mse": _mean(val_report_mse),
        "std_val_report_output_mse": _std(val_report_mse),
        "mean_test_report_output_mse": _mean(test_report_mse),
        "std_test_report_output_mse": _std(test_report_mse),
        "mean_val_supervised_output_mse": _mean(val_supervised_mse),
        "std_val_supervised_output_mse": _std(val_supervised_mse),
        "mean_test_supervised_output_mse": _mean(test_supervised_mse),
        "std_test_supervised_output_mse": _std(test_supervised_mse),
        "mean_selected_epoch": _mean(selected_epochs),
        "mean_runtime_proxy_seconds": _mean(runtime_values),
        "std_runtime_proxy_seconds": _std(runtime_values),
        "mean_readout_refit_runtime_seconds": _mean(refit_runtime_values),
        "std_readout_refit_runtime_seconds": _std(refit_runtime_values),
        "mean_selected_readout_ridge": None if not ridge_values else _mean(ridge_values),
        "selected_readout_ridges": [float(value) for value in ridge_values],
        "mean_readout_selection_val_accuracy": None if not selection_val_acc else _mean(selection_val_acc),
        "mean_readout_selection_val_output_mse": None if not selection_val_mse else _mean(selection_val_mse),
    }


def _pairwise_delta(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]) -> dict[str, Any]:
    left_by_seed = {int(row["seed"]): row for row in left_rows}
    right_by_seed = {int(row["seed"]): row for row in right_rows}
    shared_seeds = sorted(set(left_by_seed) & set(right_by_seed))
    if not shared_seeds:
        raise ValueError("Pairwise comparison requires at least one shared seed.")

    def mean_delta(metric_name: str) -> float:
        return _mean(
            [float(left_by_seed[seed][metric_name]) - float(right_by_seed[seed][metric_name]) for seed in shared_seeds]
        )

    return {
        "shared_seeds": [int(seed) for seed in shared_seeds],
        "mean_val_accuracy_delta": mean_delta("val_accuracy"),
        "mean_test_accuracy_delta": mean_delta("test_accuracy"),
        "mean_val_report_output_mse_delta": mean_delta("val_report_output_mse"),
        "mean_test_report_output_mse_delta": mean_delta("test_report_output_mse"),
        "mean_val_supervised_output_mse_delta": mean_delta("val_supervised_output_mse"),
        "mean_test_supervised_output_mse_delta": mean_delta("test_supervised_output_mse"),
        "mean_runtime_proxy_seconds_delta": mean_delta("runtime_proxy_seconds"),
    }


def _report_only_delta_vs_reference(case_summary: dict[str, Any], reference_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean_val_accuracy_delta": float(case_summary["mean_val_accuracy"] - reference_summary["mean_val_accuracy"]),
        "mean_test_accuracy_delta": float(case_summary["mean_test_accuracy"] - reference_summary["mean_test_accuracy"]),
        "mean_val_report_output_mse_delta": float(
            case_summary["mean_val_report_output_mse"] - reference_summary["mean_val_report_output_mse"]
        ),
        "mean_test_report_output_mse_delta": float(
            case_summary["mean_test_report_output_mse"] - reference_summary["mean_test_report_output_mse"]
        ),
        "mean_runtime_proxy_seconds_delta": float(
            case_summary["mean_runtime_proxy_seconds"] - reference_summary["mean_runtime_proxy_seconds"]
        ),
    }


def _materially_improves(
    pairwise: dict[str, Any],
    config: FMPCTF2ReadoutRefitSuiteConfig,
) -> bool:
    return (
        float(pairwise["mean_test_accuracy_delta"]) >= float(config.material_test_gain_threshold)
        and float(pairwise["mean_val_accuracy_delta"]) >= 0.0
        and float(pairwise["mean_val_supervised_output_mse_delta"])
        <= -float(config.material_supervised_output_mse_gain_threshold)
        and float(pairwise["mean_test_supervised_output_mse_delta"])
        <= -float(config.material_supervised_output_mse_gain_threshold)
    )


def _materially_improves_integrated_report_behavior(
    pairwise: dict[str, Any],
    config: FMPCTF2ReadoutRefitSuiteConfig,
) -> bool:
    return (
        float(pairwise["mean_test_accuracy_delta"]) >= float(config.material_test_gain_threshold)
        and float(pairwise["mean_val_accuracy_delta"]) >= 0.0
        and float(pairwise["mean_val_report_output_mse_delta"])
        <= -float(config.material_supervised_output_mse_gain_threshold)
        and float(pairwise["mean_test_report_output_mse_delta"])
        <= -float(config.material_supervised_output_mse_gain_threshold)
    )


def _diagnosis_and_next_move(
    transported_vs_control: dict[str, Any],
    slowpc_vs_control: dict[str, Any],
    slowpc_vs_transported: dict[str, Any],
    config: FMPCTF2ReadoutRefitSuiteConfig,
) -> tuple[str, str]:
    transported_improves = _materially_improves(transported_vs_control, config)
    slowpc_beats_transported = _materially_improves_integrated_report_behavior(slowpc_vs_transported, config)
    slowpc_improves_control = _materially_improves_integrated_report_behavior(slowpc_vs_control, config)
    if slowpc_beats_transported:
        return (
            "endpoint_basis_mismatch_between_transported_and_slow_pc_endpoints",
            "run one narrow adopted-package endpoint-basis / separability diagnostic at the hidden-to-output interface without changing the TF2 transport family",
        )
    if slowpc_improves_control and not transported_improves:
        return (
            "endpoint_basis_mismatch_between_transported_and_slow_pc_endpoints",
            "run one narrow adopted-package endpoint-basis / separability diagnostic at the hidden-to-output interface without changing the TF2 transport family",
        )
    if transported_improves:
        return (
            "head_fit_or_readout_optimization_problem",
            "run one narrow adopted-package head-only optimization confirmation pass without changing the TF2 transport family",
        )
    return (
        "remaining_issue_lies_elsewhere_inside_the_adopted_package",
        "shift away from readout refit and target a different package-internal issue, most likely prediction-mode output coupling rather than transported-head weighting",
    )


def run_fmpc_tf2_readout_refit_suite(
    config: FMPCTF2ReadoutRefitSuiteConfig,
) -> FMPCTF2ReadoutRefitSuiteRunResult:
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
    run_index = 0
    base_case = next(spec for spec in config.case_specs() if spec.case_name == "adopted_control")
    refit_cases = [spec for spec in config.case_specs() if spec.case_name != "adopted_control"]

    for seed in config.seeds:
        tf2_config = build_tf2_corrective_transport_terminal_angleclip_default_config(
            experiment_name="tf2",
            output_root=tf2_root,
            output_layout="run_id_subdir",
            run_id=_candidate_run_id(int(seed)),
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
            raise ValueError("Readout-refit suite requires runtime model and psi network objects.")

        split = load_digits_split(
            split_seed=int(tf2_config.data_seed),
            train_fraction=float(tf2_config.train_fraction),
            val_fraction=float(tf2_config.val_fraction),
            test_fraction=float(tf2_config.test_fraction),
        )

        train_features = _build_feature_bundle(result.model, result.psi_network, tf2_config, split.x_train, split.y_train)
        val_features = _build_feature_bundle(result.model, result.psi_network, tf2_config, split.x_val, split.y_val)
        test_features = _build_feature_bundle(result.model, result.psi_network, tf2_config, split.x_test, split.y_test)

        val_alignment = _evaluate_tf2_supervised_alignment(
            result.model,
            result.psi_network,
            tf2_config,
            split.x_val,
            split.y_val,
        )
        test_alignment = _evaluate_tf2_supervised_alignment(
            result.model,
            result.psi_network,
            tf2_config,
            split.x_test,
            split.y_test,
        )
        aggregate_rows.append(
            _control_row(
                run_index=run_index,
                run_dir=run_dir,
                seed=int(seed),
                result=result,
                val_alignment=val_alignment,
                test_alignment=test_alignment,
            )
        )
        run_index += 1

        base_snapshot = _snapshot_pc_parameters(result.model)
        for case in refit_cases:
            refit_start = perf_counter()
            train_basis = _supervised_basis_features(train_features, case.refit_basis)
            val_basis = _supervised_basis_features(val_features, case.refit_basis)
            test_basis = _supervised_basis_features(test_features, case.refit_basis)
            readout = _fit_readout_with_validation(
                train_features=train_basis,
                train_targets=split.y_train,
                val_features=val_basis,
                val_targets=split.y_val,
                ridge_grid=tuple(config.readout_ridge_grid),
            )

            _restore_pc_parameters(result.model, base_snapshot)
            result.model.layers[-1].weight = readout.weight.copy()
            result.model.layers[-1].bias = readout.bias.copy()
            val_loss, val_accuracy = _evaluate_slow_pc_accuracy(result.model, split.x_val, split.y_val)
            test_loss, test_accuracy = _evaluate_slow_pc_accuracy(result.model, split.x_test, split.y_test)
            val_report_predictions = result.model.predict(split.x_val)
            test_report_predictions = result.model.predict(split.x_test)
            _restore_pc_parameters(result.model, base_snapshot)

            val_supervised_predictions = _apply_linear_readout(val_basis, readout.weight, readout.bias)
            test_supervised_predictions = _apply_linear_readout(test_basis, readout.weight, readout.bias)
            refit_runtime_seconds = float(perf_counter() - refit_start)

            if abs(float(classification_accuracy(val_report_predictions, split.y_val)) - float(val_accuracy)) > 1e-12:
                raise ValueError("Report prediction accuracy mismatch on validation split.")
            if abs(float(classification_accuracy(test_report_predictions, split.y_test)) - float(test_accuracy)) > 1e-12:
                raise ValueError("Report prediction accuracy mismatch on test split.")
            if abs(float(regression_mse(val_report_predictions, split.y_val)) - float(val_loss)) > 1e-12:
                raise ValueError("Report prediction MSE mismatch on validation split.")
            if abs(float(regression_mse(test_report_predictions, split.y_test)) - float(test_loss)) > 1e-12:
                raise ValueError("Report prediction MSE mismatch on test split.")

            aggregate_rows.append(
                _refit_row(
                    run_index=run_index,
                    run_dir=run_dir,
                    case=case,
                    seed=int(seed),
                    result=result,
                    readout=readout,
                    val_report_predictions=val_report_predictions,
                    test_report_predictions=test_report_predictions,
                    val_supervised_predictions=val_supervised_predictions,
                    test_supervised_predictions=test_supervised_predictions,
                    val_targets=split.y_val,
                    test_targets=split.y_test,
                    refit_runtime_seconds=refit_runtime_seconds,
                )
            )
            run_index += 1
        _restore_pc_parameters(result.model, base_snapshot)

    _write_csv(run_dir / "aggregate_runs.csv", aggregate_rows)

    by_case = {
        case.case_name: _case_summary(_case_rows(aggregate_rows, case.case_name))
        for case in config.case_specs()
    }
    control_rows = _case_rows(aggregate_rows, base_case.case_name)
    transported_rows = _case_rows(aggregate_rows, "transported_endpoint_readout_refit")
    slowpc_rows = _case_rows(aggregate_rows, "slow_pc_endpoint_readout_refit")

    pairwise_vs_control = {
        "transported_endpoint_readout_refit": _pairwise_delta(transported_rows, control_rows),
        "slow_pc_endpoint_readout_refit": _pairwise_delta(slowpc_rows, control_rows),
    }
    pairwise_slowpc_vs_transported = _pairwise_delta(slowpc_rows, transported_rows)

    reference_context = _load_reference_context(config.reference_summary_path)
    report_only_vs_slow_pc: dict[str, Any] | None = None
    if reference_context is not None:
        slow_pc_reference = dict(reference_context.get("by_method", {})).get("canonical_slow_pc_digits_baseline")
        if slow_pc_reference is not None:
            report_only_vs_slow_pc = {
                case_name: _report_only_delta_vs_reference(summary, slow_pc_reference)
                for case_name, summary in by_case.items()
            }

    diagnosis, recommended_next_move = _diagnosis_and_next_move(
        transported_vs_control=pairwise_vs_control["transported_endpoint_readout_refit"],
        slowpc_vs_control=pairwise_vs_control["slow_pc_endpoint_readout_refit"],
        slowpc_vs_transported=pairwise_slowpc_vs_transported,
        config=config,
    )

    summary = {
        "phase": "FMPC Stage 04 Incremental Bridge",
        "stage": "adopted_package_readout_refit_endpoint_separability",
        "num_runs": len(aggregate_rows),
        "reference_context_reused": reference_context is not None,
        "reference_context_path": None if reference_context is None else str(Path(config.reference_summary_path).as_posix()),
        "mean_std_val_accuracy_by_case": {
            case_name: {"mean": payload["mean_val_accuracy"], "std": payload["std_val_accuracy"]}
            for case_name, payload in by_case.items()
        },
        "mean_std_test_accuracy_by_case": {
            case_name: {"mean": payload["mean_test_accuracy"], "std": payload["std_test_accuracy"]}
            for case_name, payload in by_case.items()
        },
        "mean_std_report_output_mse_by_case": {
            case_name: {
                "mean_val": payload["mean_val_report_output_mse"],
                "std_val": payload["std_val_report_output_mse"],
                "mean_test": payload["mean_test_report_output_mse"],
                "std_test": payload["std_test_report_output_mse"],
            }
            for case_name, payload in by_case.items()
        },
        "mean_std_supervised_output_mse_by_case": {
            case_name: {
                "mean_val": payload["mean_val_supervised_output_mse"],
                "std_val": payload["std_val_supervised_output_mse"],
                "mean_test": payload["mean_test_supervised_output_mse"],
                "std_test": payload["std_test_supervised_output_mse"],
            }
            for case_name, payload in by_case.items()
        },
        "mean_selected_epoch_by_case": {
            case_name: payload["mean_selected_epoch"] for case_name, payload in by_case.items()
        },
        "mean_runtime_proxy_seconds_by_case": {
            case_name: payload["mean_runtime_proxy_seconds"] for case_name, payload in by_case.items()
        },
        "mean_selected_readout_ridge_by_case": {
            case_name: payload["mean_selected_readout_ridge"] for case_name, payload in by_case.items()
        },
        "by_case": by_case,
        "pairwise_transported_endpoint_readout_refit_vs_adopted_control": pairwise_vs_control[
            "transported_endpoint_readout_refit"
        ],
        "pairwise_slow_pc_endpoint_readout_refit_vs_adopted_control": pairwise_vs_control[
            "slow_pc_endpoint_readout_refit"
        ],
        "pairwise_slow_pc_endpoint_readout_refit_vs_transported_endpoint_readout_refit": pairwise_slowpc_vs_transported,
        "report_only_vs_canonical_slow_pc_digits_baseline": report_only_vs_slow_pc,
        "transported_endpoint_refit_materially_improves_over_control": _materially_improves(
            pairwise_vs_control["transported_endpoint_readout_refit"],
            config,
        ),
        "slow_pc_endpoint_refit_materially_improves_over_control": _materially_improves_integrated_report_behavior(
            pairwise_vs_control["slow_pc_endpoint_readout_refit"],
            config,
        ),
        "slow_pc_endpoint_refit_materially_improves_over_transported_endpoint_refit": _materially_improves_integrated_report_behavior(
            pairwise_slowpc_vs_transported,
            config,
        ),
        "diagnosis": diagnosis,
        "recommended_next_narrow_tf2_move": recommended_next_move,
        "aggregate_csv_path": "aggregate_runs.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF2ReadoutRefitSuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config),
        aggregate_rows=aggregate_rows,
        summary=summary,
    )
