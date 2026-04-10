from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .fmpc_student import (
    evaluate_fmpc_delta_predictions,
    evaluate_fmpc_identity_baseline,
    fmpc_split_evaluation_metrics_payload,
    load_fmpc_student_teacher_runtime,
    prepare_fmpc_student_teacher_references,
)
from .fmpc_student_baselines import (
    ClassMeanDeltaStudent,
    RidgeDeltaStudent,
    RidgeDeltaStudentConfig,
    StandardizedMLPStudent,
    StandardizedMLPStudentConfig,
)
from .fmpc_student_data import FMPCStudentDataset, FMPCStudentSplit, load_fmpc_student_dataset
from .fmpc_student_normalization import FMPCStudentNormalizationStats, fit_fmpc_student_normalization
from ..minibatch import iter_minibatches
from ..real_pc import OutputLayout
from ..utils import set_seed


@dataclass
class FMPCStudentSuiteConfig:
    """Configuration for the Phase 5A endpoint student baseline suite on digits."""

    experiment_name: str = "fmpc_v0_student_suite"
    dataset_name: str = "digits"
    teacher_preparation_path: str | Path = "outputs/fmpc_v0_prepare_digits"
    run_seed: int = 0
    model_init_seed: int = 0
    batch_order_seed: int = 0
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: OutputLayout = "single_dir"
    normalization_eps: float = 1e-8
    ridge_alphas: tuple[float, ...] = (1e-4, 1e-2, 1.0, 100.0)
    mlp_hidden_dims_candidates: tuple[tuple[int, ...], ...] = ((64,), (128,), (128, 128))
    mlp_epochs_candidates: tuple[int, ...] = (20, 40)
    mlp_eta_w_candidates: tuple[float, ...] = (0.01, 0.05)
    mlp_weight_scale: float = 0.02
    mlp_batch_size: int = 64
    mlp_hidden_activation: str = "tanh"
    mlp_output_activation: str = "identity"
    mlp_shuffle_batches: bool = True
    allow_teacher_retrain: bool = False

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_{self.run_seed}"


@dataclass
class FMPCStudentSuiteRunResult:
    run_dir: Path
    config: dict[str, Any]
    candidates: list[dict[str, Any]]
    summary: dict[str, Any]


def _resolve_run_dir(
    output_root: str | Path,
    experiment_name: str,
    run_id: str,
    output_layout: OutputLayout,
) -> Path:
    root = Path(output_root)
    if output_layout == "single_dir":
        return root / experiment_name
    if output_layout == "run_id_subdir":
        return root / experiment_name / run_id
    raise ValueError(f"Unsupported output_layout '{output_layout}'.")


def _prepare_run_dir(run_dir: Path) -> Path:
    if run_dir.exists():
        import shutil

        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, indent=2)


def _relative_artifact_reference(from_dir: Path, target: str | Path | None) -> str | None:
    if target is None:
        return None
    return Path(
        os.path.relpath(Path(target).resolve(), start=from_dir.resolve())
    ).as_posix()


def _write_candidates_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("candidates.csv requires at least one candidate row.")
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _serialize_hidden_dims(hidden_dims: tuple[int, ...] | None) -> str | None:
    if hidden_dims is None:
        return None
    return "x".join(str(value) for value in hidden_dims)


def _candidate_metrics_columns(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _evaluate_model_split(
    model: Any,
    split: FMPCStudentSplit,
    reference: Any,
    teacher_model: Any,
) -> tuple[Any, dict[str, Any]]:
    transport_start = perf_counter()
    delta_z_hat = np.asarray(model.predict_delta_z(split), dtype=np.float64)
    transport_wall_time_seconds = float(perf_counter() - transport_start)
    evaluation = evaluate_fmpc_delta_predictions(
        delta_z_hat,
        split,
        reference,
        teacher_model,
        transport_wall_time_seconds=transport_wall_time_seconds,
    )
    return evaluation, fmpc_split_evaluation_metrics_payload(evaluation)


def _summary_family_payload(
    *,
    family: str,
    config_id: str,
    model_config: dict[str, Any],
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    test_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "family": family,
        "config_id": config_id,
        "selection_metric_source": "val_metric",
        "report_metric_source": "test_metric",
        "model_config": model_config,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }


def _identity_summary_payload(dataset: FMPCStudentDataset, references: dict[str, Any], teacher_model: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    train_eval = evaluate_fmpc_identity_baseline(dataset.train, references["train"], teacher_model)
    val_eval = evaluate_fmpc_identity_baseline(dataset.val, references["val"], teacher_model)
    test_eval = evaluate_fmpc_identity_baseline(dataset.test, references["test"], teacher_model)
    train_metrics = fmpc_split_evaluation_metrics_payload(train_eval)
    val_metrics = fmpc_split_evaluation_metrics_payload(val_eval)
    test_metrics = fmpc_split_evaluation_metrics_payload(test_eval)
    row = {
        "config_id": "identity",
        "family": "identity",
        "normalization": "none",
        "alpha": None,
        "hidden_dims": None,
        "epochs": None,
        "eta_w": None,
        "is_family_best": True,
        "is_learned_family": False,
        "is_learned_winner": False,
        "evaluated_on_test": True,
        **_candidate_metrics_columns("train", train_metrics),
        **_candidate_metrics_columns("val", val_metrics),
        **_candidate_metrics_columns("test", test_metrics),
    }
    summary = _summary_family_payload(
        family="identity",
        config_id="identity",
        model_config={"family": "identity", "student_output_definition": "delta_z_hat = 0"},
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )
    return row, summary


def _class_mean_summary_payload(dataset: FMPCStudentDataset, references: dict[str, Any], teacher_model: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    model = ClassMeanDeltaStudent.fit(dataset.train)
    train_eval, train_metrics = _evaluate_model_split(model, dataset.train, references["train"], teacher_model)
    val_eval, val_metrics = _evaluate_model_split(model, dataset.val, references["val"], teacher_model)
    test_eval, test_metrics = _evaluate_model_split(model, dataset.test, references["test"], teacher_model)
    row = {
        "config_id": "class_mean_delta",
        "family": "class_mean_delta",
        "normalization": "none",
        "alpha": None,
        "hidden_dims": None,
        "epochs": None,
        "eta_w": None,
        "is_family_best": True,
        "is_learned_family": False,
        "is_learned_winner": False,
        "evaluated_on_test": True,
        **_candidate_metrics_columns("train", train_metrics),
        **_candidate_metrics_columns("val", val_metrics),
        **_candidate_metrics_columns("test", test_metrics),
    }
    summary = _summary_family_payload(
        family="class_mean_delta",
        config_id="class_mean_delta",
        model_config=model.to_jsonable(),
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )
    return row, summary


def _run_ridge_family(
    config: FMPCStudentSuiteConfig,
    dataset: FMPCStudentDataset,
    references: dict[str, Any],
    teacher_model: Any,
    normalization: FMPCStudentNormalizationStats,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    best_index = -1
    best_val_metric = np.inf
    best_model: RidgeDeltaStudent | None = None
    best_train_metrics: dict[str, Any] | None = None
    best_val_metrics: dict[str, Any] | None = None

    for candidate_index, alpha in enumerate(config.ridge_alphas):
        model = RidgeDeltaStudent.fit(
            dataset.train,
            normalization=normalization,
            config=RidgeDeltaStudentConfig(alpha=float(alpha)),
        )
        train_eval, train_metrics = _evaluate_model_split(model, dataset.train, references["train"], teacher_model)
        val_eval, val_metrics = _evaluate_model_split(model, dataset.val, references["val"], teacher_model)
        row = {
            "config_id": f"ridge_alpha_{alpha:g}",
            "family": "ridge",
            "normalization": "train_stats",
            "alpha": float(alpha),
            "hidden_dims": None,
            "epochs": None,
            "eta_w": None,
            "is_family_best": False,
            "is_learned_family": True,
            "is_learned_winner": False,
            "evaluated_on_test": False,
            **_candidate_metrics_columns("train", train_metrics),
            **_candidate_metrics_columns("val", val_metrics),
        }
        rows.append(row)
        if val_eval.state_rms_gap < best_val_metric:
            best_val_metric = val_eval.state_rms_gap
            best_index = candidate_index
            best_model = model
            best_train_metrics = train_metrics
            best_val_metrics = val_metrics

    if best_model is None or best_train_metrics is None or best_val_metrics is None:
        raise RuntimeError("Ridge family did not produce a valid candidate.")

    test_eval, test_metrics = _evaluate_model_split(best_model, dataset.test, references["test"], teacher_model)
    rows[best_index]["is_family_best"] = True
    rows[best_index]["evaluated_on_test"] = True
    rows[best_index].update(_candidate_metrics_columns("test", test_metrics))

    summary = _summary_family_payload(
        family="ridge",
        config_id=str(rows[best_index]["config_id"]),
        model_config=best_model.to_jsonable(),
        train_metrics=best_train_metrics,
        val_metrics=best_val_metrics,
        test_metrics=test_metrics,
    )
    return rows, summary, rows[best_index]


def _run_standardized_mlp_family(
    config: FMPCStudentSuiteConfig,
    dataset: FMPCStudentDataset,
    references: dict[str, Any],
    teacher_model: Any,
    normalization: FMPCStudentNormalizationStats,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    best_val_metric = np.inf
    best_model: StandardizedMLPStudent | None = None
    best_row_index = -1
    best_train_metrics: dict[str, Any] | None = None
    best_val_metrics: dict[str, Any] | None = None

    candidate_counter = 0
    for hidden_dims in config.mlp_hidden_dims_candidates:
        for epochs in config.mlp_epochs_candidates:
            for eta_w in config.mlp_eta_w_candidates:
                mlp_config = StandardizedMLPStudentConfig(
                    hidden_dims=tuple(int(value) for value in hidden_dims),
                    hidden_activation=config.mlp_hidden_activation,
                    output_activation=config.mlp_output_activation,
                    weight_scale=config.mlp_weight_scale,
                    eta_w=float(eta_w),
                    eta_b=float(eta_w),
                    epochs=int(epochs),
                    batch_size=int(config.mlp_batch_size),
                    shuffle_batches=bool(config.mlp_shuffle_batches),
                )
                model = StandardizedMLPStudent.initialize(
                    z_dim=dataset.z_dim,
                    target_dim=dataset.target_dim,
                    normalization=normalization,
                    config=mlp_config,
                    seed=config.model_init_seed + candidate_counter,
                )
                candidate_id = (
                    f"mlp_standardized_h{_serialize_hidden_dims(mlp_config.hidden_dims)}"
                    f"_e{mlp_config.epochs}_lr{mlp_config.eta_w:g}"
                )
                best_snapshot = None
                candidate_best_val_metric = np.inf

                for epoch in range(1, mlp_config.epochs + 1):
                    for x_batch, y_batch in iter_minibatches(
                        dataset.train.student_inputs,
                        dataset.train.delta_z,
                        mlp_config.batch_size,
                        shuffle=mlp_config.shuffle_batches,
                        seed=config.batch_order_seed + candidate_counter * 1000 + (epoch - 1),
                    ):
                        model.train_batch(x_batch, y_batch)
                    val_eval, _ = _evaluate_model_split(model, dataset.val, references["val"], teacher_model)
                    if val_eval.state_rms_gap < candidate_best_val_metric:
                        candidate_best_val_metric = val_eval.state_rms_gap
                        best_snapshot = model.snapshot()

                if best_snapshot is None:
                    raise RuntimeError(f"{candidate_id} did not record a best checkpoint.")
                model.restore(best_snapshot)

                train_eval, train_metrics = _evaluate_model_split(model, dataset.train, references["train"], teacher_model)
                val_eval, val_metrics = _evaluate_model_split(model, dataset.val, references["val"], teacher_model)
                row = {
                    "config_id": candidate_id,
                    "family": "mlp_standardized",
                    "normalization": "train_stats",
                    "alpha": None,
                    "hidden_dims": _serialize_hidden_dims(mlp_config.hidden_dims),
                    "epochs": int(mlp_config.epochs),
                    "eta_w": float(mlp_config.eta_w),
                    "is_family_best": False,
                    "is_learned_family": True,
                    "is_learned_winner": False,
                    "evaluated_on_test": False,
                    **_candidate_metrics_columns("train", train_metrics),
                    **_candidate_metrics_columns("val", val_metrics),
                }
                rows.append(row)
                if val_eval.state_rms_gap < best_val_metric:
                    best_val_metric = val_eval.state_rms_gap
                    best_model = model
                    best_row_index = len(rows) - 1
                    best_train_metrics = train_metrics
                    best_val_metrics = val_metrics
                candidate_counter += 1

    if best_model is None or best_train_metrics is None or best_val_metrics is None:
        raise RuntimeError("Standardized MLP family did not produce a valid candidate.")

    test_eval, test_metrics = _evaluate_model_split(best_model, dataset.test, references["test"], teacher_model)
    rows[best_row_index]["is_family_best"] = True
    rows[best_row_index]["evaluated_on_test"] = True
    rows[best_row_index].update(_candidate_metrics_columns("test", test_metrics))

    summary = _summary_family_payload(
        family="mlp_standardized",
        config_id=str(rows[best_row_index]["config_id"]),
        model_config=best_model.to_jsonable(),
        train_metrics=best_train_metrics,
        val_metrics=best_val_metrics,
        test_metrics=test_metrics,
    )
    return rows, summary, rows[best_row_index]


def _suite_config_payload(
    config: FMPCStudentSuiteConfig,
    *,
    run_id: str,
    run_dir: Path,
    dataset: FMPCStudentDataset,
    normalization: FMPCStudentNormalizationStats,
    teacher_checkpoint_loaded: bool,
    comparison_atol: float,
) -> dict[str, Any]:
    return {
        "experiment_name": config.experiment_name,
        "run_id": run_id,
        "phase": "Phase 5",
        "stage": "phase5a_student_signal_rescue",
        "dataset_name": config.dataset_name,
        "teacher_artifact_dir": _relative_artifact_reference(run_dir, config.teacher_preparation_path),
        "teacher_manifest_path": _relative_artifact_reference(run_dir, dataset.teacher_manifest_path),
        "teacher_checkpoint_path": _relative_artifact_reference(run_dir, dataset.teacher_checkpoint_path),
        "selection_metric_name": "state_rms_gap",
        "selection_metric_source": "val_metric",
        "report_metric_source": "test_metric",
        "run_seed": config.run_seed,
        "model_init_seed": config.model_init_seed,
        "batch_order_seed": config.batch_order_seed,
        "teacher_checkpoint_loaded": bool(teacher_checkpoint_loaded),
        "teacher_reference_comparison_atol": float(comparison_atol),
        "teacher_recovery": {
            "allow_teacher_retrain": bool(config.allow_teacher_retrain),
            "teacher_checkpoint_required_by_default": True,
        },
        "student_input_definition": dataset.student_input_definition,
        "student_target_definition": dataset.student_target_definition,
        "normalization": normalization.to_jsonable(),
        "search_space": {
            "families": ["class_mean_delta", "ridge", "mlp_standardized"],
            "ridge_alphas": [float(value) for value in config.ridge_alphas],
            "mlp_hidden_dims_candidates": [list(candidate) for candidate in config.mlp_hidden_dims_candidates],
            "mlp_epochs_candidates": [int(value) for value in config.mlp_epochs_candidates],
            "mlp_eta_w_candidates": [float(value) for value in config.mlp_eta_w_candidates],
            "mlp_weight_scale": float(config.mlp_weight_scale),
            "mlp_batch_size": int(config.mlp_batch_size),
            "mlp_hidden_activation": config.mlp_hidden_activation,
            "mlp_output_activation": config.mlp_output_activation,
        },
    }


def run_fmpc_student_suite(config: FMPCStudentSuiteConfig) -> FMPCStudentSuiteRunResult:
    """Run the Phase 5A endpoint student baseline suite on `digits`."""

    if config.dataset_name != "digits":
        raise ValueError("Phase 5A student-signal rescue currently supports digits only.")
    set_seed(config.run_seed)

    dataset = load_fmpc_student_dataset(config.teacher_preparation_path, expected_dataset_name="digits")
    teacher_model, teacher_split, used_teacher_retrain_fallback, comparison_atol = load_fmpc_student_teacher_runtime(
        dataset,
        allow_teacher_retrain=config.allow_teacher_retrain,
    )
    references = prepare_fmpc_student_teacher_references(
        dataset,
        teacher_model,
        teacher_split,
        comparison_atol=comparison_atol,
    )
    normalization = fit_fmpc_student_normalization(dataset.train, eps=config.normalization_eps)

    run_id = config.resolved_run_id()
    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, run_id, config.output_layout)
    )

    identity_row, identity_summary = _identity_summary_payload(dataset, references, teacher_model)
    class_mean_row, class_mean_summary = _class_mean_summary_payload(dataset, references, teacher_model)
    ridge_rows, ridge_summary, ridge_best_row = _run_ridge_family(
        config,
        dataset,
        references,
        teacher_model,
        normalization,
    )
    mlp_rows, mlp_summary, mlp_best_row = _run_standardized_mlp_family(
        config,
        dataset,
        references,
        teacher_model,
        normalization,
    )

    learned_family_rows = [ridge_best_row, mlp_best_row]
    learned_winner_row = min(
        learned_family_rows,
        key=lambda row: float(row["val_state_rms_gap"]),
    )
    for row in ridge_rows + mlp_rows:
        if row["config_id"] == learned_winner_row["config_id"]:
            row["is_learned_winner"] = True

    overall_family_rows = [identity_row, class_mean_row, ridge_best_row, mlp_best_row]
    overall_best_row = min(
        overall_family_rows,
        key=lambda row: float(row["val_state_rms_gap"]),
    )

    summary = {
        "experiment_name": config.experiment_name,
        "run_id": run_id,
        "phase": "Phase 5",
        "stage": "phase5a_student_signal_rescue",
        "dataset_name": config.dataset_name,
        "teacher_artifact_dir": _relative_artifact_reference(run_dir, config.teacher_preparation_path),
        "teacher_manifest_path": _relative_artifact_reference(run_dir, dataset.teacher_manifest_path),
        "teacher_checkpoint_path": _relative_artifact_reference(run_dir, dataset.teacher_checkpoint_path),
        "teacher_checkpoint_loaded": not used_teacher_retrain_fallback,
        "selection_metric_source": "val_metric",
        "report_metric_source": "test_metric",
        "identity_baseline": identity_summary,
        "class_mean_delta": class_mean_summary,
        "ridge": ridge_summary,
        "mlp_standardized": mlp_summary,
        "winner": {
            "overall_best_family_by_val": str(overall_best_row["family"]),
            "overall_best_config_id_by_val": str(overall_best_row["config_id"]),
            "learned_winner_family_by_val": str(learned_winner_row["family"]),
            "learned_winner_config_id_by_val": str(learned_winner_row["config_id"]),
            "learned_winner_val_state_rms_gap": float(learned_winner_row["val_state_rms_gap"]),
            "learned_winner_test_state_rms_gap": float(learned_winner_row["test_state_rms_gap"]),
            "learned_winner_beats_identity_on_val_metric": bool(
                float(learned_winner_row["val_state_rms_gap"]) < float(identity_row["val_state_rms_gap"])
            ),
            "learned_winner_beats_identity_on_test_metric": bool(
                float(learned_winner_row["test_state_rms_gap"]) < float(identity_row["test_state_rms_gap"])
            ),
            "val_state_rms_gap_delta_vs_identity": float(
                float(learned_winner_row["val_state_rms_gap"]) - float(identity_row["val_state_rms_gap"])
            ),
            "test_state_rms_gap_delta_vs_identity": float(
                float(learned_winner_row["test_state_rms_gap"]) - float(identity_row["test_state_rms_gap"])
            ),
        },
        "teacher_target_stats": {
            "train": {
                "delta_z_l2_mean": dataset.train.metadata["delta_z_l2_mean"],
                "delta_z_rms": dataset.train.metadata["delta_z_rms"],
                "delta_z_max_abs": dataset.train.metadata["delta_z_max_abs"],
            },
            "val": {
                "delta_z_l2_mean": dataset.val.metadata["delta_z_l2_mean"],
                "delta_z_rms": dataset.val.metadata["delta_z_rms"],
                "delta_z_max_abs": dataset.val.metadata["delta_z_max_abs"],
            },
            "test": {
                "delta_z_l2_mean": dataset.test.metadata["delta_z_l2_mean"],
                "delta_z_rms": dataset.test.metadata["delta_z_rms"],
                "delta_z_max_abs": dataset.test.metadata["delta_z_max_abs"],
            },
        },
    }

    config_payload = _suite_config_payload(
        config,
        run_id=run_id,
        run_dir=run_dir,
        dataset=dataset,
        normalization=normalization,
        teacher_checkpoint_loaded=not used_teacher_retrain_fallback,
        comparison_atol=comparison_atol,
    )
    candidate_rows = [identity_row, class_mean_row, *ridge_rows, *mlp_rows]

    _write_json(run_dir / "config.json", config_payload)
    _write_candidates_csv(run_dir / "candidates.csv", candidate_rows)
    _write_json(run_dir / "summary.json", summary)

    return FMPCStudentSuiteRunResult(
        run_dir=run_dir,
        config=config_payload,
        candidates=candidate_rows,
        summary=summary,
    )
