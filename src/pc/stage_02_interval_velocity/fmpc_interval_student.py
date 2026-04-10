from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .fmpc_interval_data import (
    FMPCIntervalBatch,
    FMPCIntervalDataset,
    FMPCIntervalSplit,
    acceptance_schedule_focus_pairs,
    build_fmpc_interval_inputs,
    compute_interval_velocity_target,
    iter_all_interval_blocks,
    iter_balanced_interval_batches,
    iter_weighted_interval_blocks,
    load_fmpc_interval_dataset,
    teacher_step_aligned_rollout_schedules,
)
from .fmpc_interval_features import (
    FMPCIntervalTeacherFeatureBundle,
    prepare_interval_teacher_feature_context,
    precompute_interval_teacher_trajectory_feature_tangents,
    precompute_interval_teacher_trajectory_features,
)
from .fmpc_interval_normalization import (
    FMPCIntervalNormalizationStats,
    fit_fmpc_interval_normalization,
    fit_fmpc_interval_augmented_normalization,
)
from ..stage_01_reference_prep.fmpc_student import (
    evaluate_fmpc_delta_predictions,
    evaluate_fmpc_identity_baseline,
    fmpc_split_evaluation_metrics_payload,
    load_fmpc_student_teacher_runtime,
    prepare_fmpc_student_teacher_references,
)
from ..stage_01_reference_prep.fmpc_student_baselines import RidgeDeltaStudent, RidgeDeltaStudentConfig
from ..stage_01_reference_prep.fmpc_student_data import FMPCStudentDataset, load_fmpc_student_dataset
from ..stage_01_reference_prep.fmpc_student_normalization import fit_fmpc_student_normalization
from ..mlp_baseline import MLPNetwork, init_mlp_baseline_layers
from ..real_pc import OutputLayout
from ..utils import set_seed


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, indent=2)


def _write_candidates_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("candidates.csv requires at least one row.")
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


def _relative_artifact_reference(from_dir: Path, target: str | Path | None) -> str | None:
    if target is None:
        return None
    return Path(os.path.relpath(Path(target).resolve(), start=from_dir.resolve())).as_posix()


def _serialize_hidden_dims(hidden_dims: tuple[int, ...] | None) -> str | None:
    if hidden_dims is None:
        return None
    return "x".join(str(value) for value in hidden_dims)


def _validate_rollout_knots(knots: tuple[int, ...], teacher_steps: int) -> tuple[int, ...]:
    if knots[0] != 0 or knots[-1] != teacher_steps:
        raise ValueError("Rollout knots must start at 0 and end at teacher_steps.")
    if tuple(sorted(knots)) != knots or len(set(knots)) != len(knots):
        raise ValueError("Rollout knots must be strictly increasing.")
    return tuple(int(value) for value in knots)


def _interval_metrics_from_endpoint(
    endpoint_metrics: dict[str, Any],
    *,
    mean_knot_state_rms_gap: float | None,
) -> dict[str, Any]:
    return {
        "final_state_l2_gap": endpoint_metrics["state_l2_gap"],
        "final_state_rms_gap": endpoint_metrics["state_rms_gap"],
        "teacher_energy": endpoint_metrics["teacher_energy"],
        "predicted_energy": endpoint_metrics["predicted_energy"],
        "energy_gap_to_teacher": endpoint_metrics["energy_gap_to_teacher"],
        "update_direction_cosine": endpoint_metrics["update_direction_cosine"],
        "transport_wall_time_seconds": endpoint_metrics["transport_wall_time_seconds"],
        "teacher_inference_wall_time_seconds": endpoint_metrics["teacher_inference_wall_time_seconds"],
        "speedup_vs_teacher": endpoint_metrics["speedup_vs_teacher"],
        "mean_knot_state_rms_gap": mean_knot_state_rms_gap,
    }


def _candidate_metric_columns(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def build_interval_residual_target(
    u_star: np.ndarray,
    g_s: np.ndarray,
) -> np.ndarray:
    """Return the residual interval target `u_res = u_star - g_s`.

    Shape contract:
    - `u_star`: `(batch, z_dim)`
    - `g_s`: `(batch, z_dim)`
    - returns `(batch, z_dim)`
    """

    u_star_array = np.asarray(u_star, dtype=np.float64)
    g_s_array = np.asarray(g_s, dtype=np.float64)
    if u_star_array.shape != g_s_array.shape:
        raise ValueError("u_star and g_s must share the same shape.")
    return (u_star_array - g_s_array).astype(np.float64, copy=False)


def reconstruct_interval_velocity_from_residual(
    g_s: np.ndarray,
    u_res_hat: np.ndarray,
) -> np.ndarray:
    """Return `u_hat = g_s + u_res_hat` in batch-first form."""

    g_s_array = np.asarray(g_s, dtype=np.float64)
    u_res_hat_array = np.asarray(u_res_hat, dtype=np.float64)
    if g_s_array.shape != u_res_hat_array.shape:
        raise ValueError("g_s and u_res_hat must share the same shape.")
    return (g_s_array + u_res_hat_array).astype(np.float64, copy=False)


@dataclass(frozen=True)
class IntervalRidgeStudentConfig:
    alpha: float


@dataclass(frozen=True)
class IntervalRidgeStudent:
    """Deterministic closed-form multi-output ridge regression on interval velocity targets."""

    config: IntervalRidgeStudentConfig
    normalization: FMPCIntervalNormalizationStats
    coefficients: np.ndarray
    bias: np.ndarray
    z_dim: int
    target_dim: int

    @classmethod
    def fit(
        cls,
        train_split: FMPCIntervalSplit,
        *,
        normalization: FMPCIntervalNormalizationStats,
        config: IntervalRidgeStudentConfig,
    ) -> IntervalRidgeStudent:
        if config.alpha <= 0.0:
            raise ValueError("Ridge alpha must be positive.")
        input_dim = int(train_split.z_dim + train_split.target_dim + 2)
        design_dim = input_dim + 1
        xtx = np.zeros((design_dim, design_dim), dtype=np.float64)
        xty = np.zeros((design_dim, train_split.z_dim), dtype=np.float64)
        for block in iter_all_interval_blocks(train_split):
            inputs = normalization.transform_inputs(
                block["z_s"],
                block["target_onehot"],
                block["tau_s"],
                block["tau_t"],
            )
            targets = normalization.transform_u(block["u_star"])
            design = np.concatenate([inputs, np.ones((inputs.shape[0], 1), dtype=np.float64)], axis=1)
            pair_weight = float(block["pair_weight"])
            xtx += pair_weight * (design.T @ design)
            xty += pair_weight * (design.T @ targets)
        reg = config.alpha * np.eye(design_dim, dtype=np.float64)
        solution = np.linalg.solve(xtx + reg, xty)
        return cls(
            config=config,
            normalization=normalization,
            coefficients=np.asarray(solution[:-1, :], dtype=np.float64),
            bias=np.asarray(solution[-1, :], dtype=np.float64),
            z_dim=int(train_split.z_dim),
            target_dim=int(train_split.target_dim),
        )

    def predict_u(
        self,
        z_s: np.ndarray,
        target_onehot: np.ndarray,
        tau_s: np.ndarray | float,
        tau_t: np.ndarray | float,
    ) -> np.ndarray:
        inputs = self.normalization.transform_inputs(z_s, target_onehot, tau_s, tau_t)
        u_normalized = inputs @ self.coefficients + self.bias
        return self.normalization.inverse_u(u_normalized)

    def predict_u_batch(self, batch: FMPCIntervalBatch) -> np.ndarray:
        return self.predict_u(batch.z_s, batch.target_onehot, batch.tau_s, batch.tau_t)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "family": "interval_ridge",
            "alpha": float(self.config.alpha),
            "normalization": self.normalization.to_jsonable(),
            "student_input_definition": "concat([z_s, target_onehot, tau_s, tau_t])",
            "student_target_definition": "u_star = (z_t - z_s) / (tau_t - tau_s)",
        }


@dataclass(frozen=True)
class IntervalAugmentedRidgeStudentConfig:
    alpha: float
    selected_teacher_feature_names: tuple[str, ...] = ("g_s", "e_out_s", "F_s")
    target_mode: Literal["u_star", "u_residual_local_field"] = "u_star"
    knot_focused_schedule_names: tuple[str, ...] = ("2-step", "3-step")
    knot_focus_mixture: float = 0.0


@dataclass(frozen=True)
class IntervalAugmentedRidgeStudent:
    """Deterministic ridge student augmented with frozen-teacher current-state features."""

    family_name: str
    config: IntervalAugmentedRidgeStudentConfig
    normalization: FMPCIntervalNormalizationStats
    coefficients: np.ndarray
    bias: np.ndarray
    z_dim: int
    target_dim: int
    teacher_model: Any
    teacher_feature_bundle: FMPCIntervalTeacherFeatureBundle

    @classmethod
    def fit(
        cls,
        train_split: FMPCIntervalSplit,
        *,
        normalization: FMPCIntervalNormalizationStats,
        config: IntervalAugmentedRidgeStudentConfig,
        family_name: str,
        teacher_model: Any,
        teacher_feature_bundle: FMPCIntervalTeacherFeatureBundle,
    ) -> "IntervalAugmentedRidgeStudent":
        if config.alpha <= 0.0:
            raise ValueError("Ridge alpha must be positive.")
        if family_name not in {"interval_ridge_aug", "interval_ridge_residual"}:
            raise ValueError("family_name must be 'interval_ridge_aug' or 'interval_ridge_residual'.")
        input_dim = int(
            train_split.z_dim
            + train_split.target_dim
            + 2
            + normalization.teacher_feature_mean.shape[0]
        )
        design_dim = input_dim + 1
        xtx = np.zeros((design_dim, design_dim), dtype=np.float64)
        xty = np.zeros((design_dim, train_split.z_dim), dtype=np.float64)
        train_trajectory_features = teacher_feature_bundle.trajectory_features("train")
        for block in iter_weighted_interval_blocks(
            train_split,
            knot_focused_schedule_names=config.knot_focused_schedule_names,
            knot_focus_mixture=config.knot_focus_mixture,
        ):
            state_features = train_trajectory_features.step_features(
                int(block["source_index"]),
                selected_feature_names=config.selected_teacher_feature_names,
            )
            teacher_feature_matrix = state_features.feature_matrix(config.selected_teacher_feature_names)
            inputs = normalization.transform_inputs(
                block["z_s"],
                block["target_onehot"],
                block["tau_s"],
                block["tau_t"],
                teacher_features=teacher_feature_matrix,
            )
            targets = np.asarray(block["u_star"], dtype=np.float64)
            if config.target_mode == "u_residual_local_field":
                targets = build_interval_residual_target(targets, state_features.g_s)
            targets = normalization.transform_u(targets)
            design = np.concatenate([inputs, np.ones((inputs.shape[0], 1), dtype=np.float64)], axis=1)
            pair_weight = float(block["pair_weight"])
            xtx += pair_weight * (design.T @ design)
            xty += pair_weight * (design.T @ targets)
        reg = config.alpha * np.eye(design_dim, dtype=np.float64)
        solution = np.linalg.solve(xtx + reg, xty)
        return cls(
            family_name=family_name,
            config=config,
            normalization=normalization,
            coefficients=np.asarray(solution[:-1, :], dtype=np.float64),
            bias=np.asarray(solution[-1, :], dtype=np.float64),
            z_dim=int(train_split.z_dim),
            target_dim=int(train_split.target_dim),
            teacher_model=teacher_model,
            teacher_feature_bundle=teacher_feature_bundle,
        )

    def _predict_target_from_features(
        self,
        *,
        z_s: np.ndarray,
        target_onehot: np.ndarray,
        tau_s: np.ndarray | float,
        tau_t: np.ndarray | float,
        teacher_feature_matrix: np.ndarray,
        g_s: np.ndarray,
    ) -> np.ndarray:
        inputs = self.normalization.transform_inputs(
            z_s,
            target_onehot,
            tau_s,
            tau_t,
            teacher_features=teacher_feature_matrix,
        )
        predicted_target = self.normalization.inverse_u(inputs @ self.coefficients + self.bias)
        if self.config.target_mode == "u_residual_local_field":
            return reconstruct_interval_velocity_from_residual(g_s, predicted_target)
        return np.asarray(predicted_target, dtype=np.float64)

    def predict_u_for_rollout(
        self,
        split_name: str,
        z_s: np.ndarray,
        target_onehot: np.ndarray,
        tau_s: np.ndarray | float,
        tau_t: np.ndarray | float,
    ) -> np.ndarray:
        teacher_feature_matrix, state_features = self.teacher_feature_bundle.current_state_feature_matrix(
            self.teacher_model,
            split_name=split_name,
            z_s=np.asarray(z_s, dtype=np.float64),
            target_onehot=np.asarray(target_onehot, dtype=np.float64),
            tau_s=tau_s,
            tau_t=tau_t,
            selected_feature_names=self.config.selected_teacher_feature_names,
        )
        return self._predict_target_from_features(
            z_s=np.asarray(z_s, dtype=np.float64),
            target_onehot=np.asarray(target_onehot, dtype=np.float64),
            tau_s=tau_s,
            tau_t=tau_t,
            teacher_feature_matrix=teacher_feature_matrix,
            g_s=state_features.g_s,
        )

    def to_jsonable(self) -> dict[str, Any]:
        target_definition = (
            "u_star = (z_t - z_s) / (tau_t - tau_s)"
            if self.config.target_mode == "u_star"
            else "u_res = u_star - g_s"
        )
        return {
            "family": self.family_name,
            "alpha": float(self.config.alpha),
            "teacher_feature_names": list(self.config.selected_teacher_feature_names),
            "target_mode": self.config.target_mode,
            "knot_focused_schedule_names": list(self.config.knot_focused_schedule_names),
            "knot_focus_mixture": float(self.config.knot_focus_mixture),
            "normalization": self.normalization.to_jsonable(),
            "student_input_definition": (
                "concat([z_s, target_onehot, tau_s, tau_t, teacher_state_features])"
            ),
            "teacher_feature_contract": {
                "y_hat_s_supported": True,
                "e_out_s_supported": True,
                "g_s_supported": True,
                "F_s_supported": True,
                "selected_teacher_feature_names": list(self.config.selected_teacher_feature_names),
            },
            "student_target_definition": target_definition,
            "prediction_reconstruction": (
                "u_hat = g_s + u_res_hat"
                if self.config.target_mode == "u_residual_local_field"
                else "u_hat = u_star_hat"
            ),
        }


@dataclass(frozen=True)
class IntervalStandardizedMLPStudentConfig:
    hidden_dims: tuple[int, ...]
    hidden_activation: str = "tanh"
    output_activation: str = "identity"
    weight_scale: float = 0.02
    eta_w: float = 0.01
    eta_b: float | None = None
    epochs: int = 20
    batch_size: int = 64
    batches_per_epoch: int | None = None
    rollout_aux_weight: float = 0.0
    rollout_aux_schedule_names: tuple[str, ...] = ()


@dataclass
class IntervalStandardizedMLPStudent:
    """NumPy MLP student trained on normalized interval velocity targets."""

    config: IntervalStandardizedMLPStudentConfig
    normalization: FMPCIntervalNormalizationStats
    network: MLPNetwork
    z_dim: int
    target_dim: int

    @classmethod
    def initialize(
        cls,
        *,
        z_dim: int,
        target_dim: int,
        normalization: FMPCIntervalNormalizationStats,
        config: IntervalStandardizedMLPStudentConfig,
        seed: int,
    ) -> IntervalStandardizedMLPStudent:
        input_dim = int(z_dim + target_dim + 2)
        network = MLPNetwork(
            layers=init_mlp_baseline_layers(
                (input_dim, *config.hidden_dims, int(z_dim)),
                hidden_activation=config.hidden_activation,
                output_activation=config.output_activation,
                weight_scale=config.weight_scale,
                seed=seed,
                dtype=np.float64,
            ),
            eta_w=config.eta_w,
            eta_b=config.eta_b,
        )
        return cls(
            config=config,
            normalization=normalization,
            network=network,
            z_dim=int(z_dim),
            target_dim=int(target_dim),
        )

    def predict_u(
        self,
        z_s: np.ndarray,
        target_onehot: np.ndarray,
        tau_s: np.ndarray | float,
        tau_t: np.ndarray | float,
    ) -> np.ndarray:
        inputs = self.normalization.transform_inputs(z_s, target_onehot, tau_s, tau_t)
        u_normalized = self.network.predict(inputs)
        return self.normalization.inverse_u(u_normalized)

    def predict_u_batch(self, batch: FMPCIntervalBatch) -> np.ndarray:
        return self.predict_u(batch.z_s, batch.target_onehot, batch.tau_s, batch.tau_t)

    def train_batch(self, batch: FMPCIntervalBatch, *, lr_scale: float = 1.0) -> float:
        if lr_scale < 0.0:
            raise ValueError("lr_scale must be non-negative.")
        inputs = self.normalization.transform_inputs(batch.z_s, batch.target_onehot, batch.tau_s, batch.tau_t)
        targets = self.normalization.transform_u(batch.u_star)
        if lr_scale == 0.0:
            predictions = self.network.predict(inputs)
            return float(np.mean((predictions - targets) ** 2))
        original_eta_w = float(self.network.eta_w)
        original_eta_b = float(self.network.eta_b)
        self.network.eta_w = original_eta_w * lr_scale
        self.network.eta_b = original_eta_b * lr_scale
        try:
            result = self.network.train_batch(inputs, targets)
        finally:
            self.network.eta_w = original_eta_w
            self.network.eta_b = original_eta_b
        return float(result.loss)

    def snapshot(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(layer.weight.copy(), layer.bias.copy()) for layer in self.network.layers]

    def restore(self, snapshot: list[tuple[np.ndarray, np.ndarray]]) -> None:
        if len(snapshot) != len(self.network.layers):
            raise ValueError("Parameter snapshot must align with interval student layers.")
        for layer, (weight, bias) in zip(self.network.layers, snapshot, strict=True):
            layer.weight = weight.copy()
            layer.bias = bias.copy()

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "family": "interval_mlp_standardized",
            "hidden_dims": list(self.config.hidden_dims),
            "hidden_activation": self.config.hidden_activation,
            "output_activation": self.config.output_activation,
            "weight_scale": float(self.config.weight_scale),
            "eta_w": float(self.config.eta_w),
            "eta_b": float(self.config.eta_b if self.config.eta_b is not None else self.config.eta_w),
            "epochs": int(self.config.epochs),
            "batch_size": int(self.config.batch_size),
            "batches_per_epoch": self.config.batches_per_epoch,
            "rollout_aux_weight": float(self.config.rollout_aux_weight),
            "rollout_aux_schedule_names": list(self.config.rollout_aux_schedule_names),
            "normalization": self.normalization.to_jsonable(),
            "student_input_definition": "concat([z_s, target_onehot, tau_s, tau_t])",
            "student_target_definition": "u_star = (z_t - z_s) / (tau_t - tau_s)",
        }


@dataclass(frozen=True)
class FMPCIntervalRolloutResult:
    rollout_schedule_name: str
    knots: tuple[int, ...]
    predicted_knot_states: list[np.ndarray]
    final_state: np.ndarray
    mean_knot_state_rms_gap: float | None
    transport_wall_time_seconds: float


@dataclass(frozen=True)
class FMPCIntervalRolloutAuxiliaryBatches:
    """Teacher-supervised auxiliary batches built from a self-fed rollout.

    Shape contract:
    - each batch inside `batches` has the same contract as `FMPCIntervalBatch`
    - `batches[i].z_s` is the student-predicted source state for that rollout segment
    - `batches[i].z_t` is the teacher knot state for the corresponding target knot
    """

    rollout_schedule_name: str
    knots: tuple[int, ...]
    batches: tuple[FMPCIntervalBatch, ...]
    rollout_aux_velocity_mse: float
    intermediate_state_rms_gap: float | None
    final_state_rms_gap: float


def _full_split_interval_batch(
    split: FMPCIntervalSplit,
    *,
    source_index: int,
    target_index: int,
    z_s: np.ndarray,
    z_t: np.ndarray,
) -> FMPCIntervalBatch:
    batch_size = int(split.num_samples)
    tau_s = np.full((batch_size, 1), source_index / split.teacher_steps, dtype=np.float64)
    tau_t = np.full((batch_size, 1), target_index / split.teacher_steps, dtype=np.float64)
    sample_row_indices = np.arange(batch_size, dtype=np.int64)
    source_step_indices = np.full(batch_size, source_index, dtype=np.int64)
    target_step_indices = np.full(batch_size, target_index, dtype=np.int64)
    span_lengths = np.full(batch_size, target_index - source_index, dtype=np.int64)
    student_inputs = build_fmpc_interval_inputs(z_s, split.target_onehot, tau_s, tau_t)
    u_star = compute_interval_velocity_target(z_s, z_t, tau_s, tau_t)
    return FMPCIntervalBatch(
        sample_row_indices=sample_row_indices,
        source_step_indices=source_step_indices,
        target_step_indices=target_step_indices,
        span_lengths=span_lengths,
        target_onehot=np.asarray(split.target_onehot, dtype=np.float64),
        z_s=np.asarray(z_s, dtype=np.float64),
        z_t=np.asarray(z_t, dtype=np.float64),
        tau_s=tau_s,
        tau_t=tau_t,
        delta_tau=(tau_t - tau_s).astype(np.float64, copy=False),
        u_star=np.asarray(u_star, dtype=np.float64),
        student_inputs=student_inputs,
    )


def _predict_rollout_velocity(
    model: Any,
    split: FMPCIntervalSplit,
    current_state: np.ndarray,
    tau_s: np.ndarray,
    tau_t: np.ndarray,
) -> np.ndarray:
    if hasattr(model, "predict_u_for_rollout"):
        return np.asarray(
            model.predict_u_for_rollout(
                split.split_name,
                current_state,
                split.target_onehot,
                tau_s,
                tau_t,
            ),
            dtype=np.float64,
        )
    return np.asarray(
        model.predict_u(current_state, split.target_onehot, tau_s, tau_t),
        dtype=np.float64,
    )


def rollout_interval_student(
    model: Any,
    split: FMPCIntervalSplit,
    *,
    rollout_schedule_name: str,
    knots: tuple[int, ...],
) -> FMPCIntervalRolloutResult:
    """Run self-fed interval rollout on one split using teacher-step-aligned knots."""

    resolved_knots = _validate_rollout_knots(knots, split.teacher_steps)
    current_state = np.asarray(split.z0, dtype=np.float64)
    predicted_knot_states = [current_state.copy()]
    rollout_start = perf_counter()
    for source_index, target_index in zip(resolved_knots[:-1], resolved_knots[1:], strict=True):
        tau_s = np.full((split.num_samples, 1), source_index / split.teacher_steps, dtype=np.float64)
        tau_t = np.full((split.num_samples, 1), target_index / split.teacher_steps, dtype=np.float64)
        u_hat = _predict_rollout_velocity(model, split, current_state, tau_s, tau_t)
        current_state = (current_state + (tau_t - tau_s) * np.asarray(u_hat, dtype=np.float64)).astype(
            np.float64,
            copy=False,
        )
        predicted_knot_states.append(current_state.copy())
    transport_wall_time_seconds = float(perf_counter() - rollout_start)
    knot_state_rms_gaps = [
        float(np.sqrt(np.mean((predicted_knot_states[index] - split.z_trajectory[:, knot, :]) ** 2)))
        for index, knot in enumerate(resolved_knots[1:], start=1)
    ]
    mean_knot_state_rms_gap = (
        None if not knot_state_rms_gaps else float(np.mean(np.asarray(knot_state_rms_gaps, dtype=np.float64)))
    )
    return FMPCIntervalRolloutResult(
        rollout_schedule_name=rollout_schedule_name,
        knots=resolved_knots,
        predicted_knot_states=predicted_knot_states,
        final_state=np.asarray(current_state, dtype=np.float64),
        mean_knot_state_rms_gap=mean_knot_state_rms_gap,
        transport_wall_time_seconds=transport_wall_time_seconds,
    )


def build_rollout_auxiliary_batches(
    model: Any,
    split: FMPCIntervalSplit,
    *,
    rollout_schedule_name: str,
    knots: tuple[int, ...],
) -> FMPCIntervalRolloutAuxiliaryBatches:
    """Build teacher-supervised auxiliary batches from a self-fed rollout.

    The rollout starts from teacher `z_0`, but every later segment consumes the student-predicted
    source state from the previous segment. Each auxiliary batch then uses a corrective target
    velocity that would move that predicted source state toward the corresponding teacher knot.
    """

    resolved_knots = _validate_rollout_knots(knots, split.teacher_steps)
    current_state = np.asarray(split.z0, dtype=np.float64)
    batches: list[FMPCIntervalBatch] = []
    velocity_mses: list[float] = []
    intermediate_state_rms_gaps: list[float] = []
    final_state_rms_gap: float | None = None

    for segment_index, (source_index, target_index) in enumerate(
        zip(resolved_knots[:-1], resolved_knots[1:], strict=True),
        start=1,
    ):
        tau_s = np.full((split.num_samples, 1), source_index / split.teacher_steps, dtype=np.float64)
        tau_t = np.full((split.num_samples, 1), target_index / split.teacher_steps, dtype=np.float64)
        u_hat = _predict_rollout_velocity(model, split, current_state, tau_s, tau_t)
        teacher_target_state = np.asarray(split.z_trajectory[:, target_index, :], dtype=np.float64)
        corrective_u = compute_interval_velocity_target(current_state, teacher_target_state, tau_s, tau_t)
        predicted_next_state = (current_state + (tau_t - tau_s) * u_hat).astype(np.float64, copy=False)
        state_rms_gap = float(np.sqrt(np.mean((predicted_next_state - teacher_target_state) ** 2)))
        velocity_mses.append(float(np.mean((u_hat - corrective_u) ** 2)))
        if segment_index < len(resolved_knots) - 1:
            intermediate_state_rms_gaps.append(state_rms_gap)
        else:
            final_state_rms_gap = state_rms_gap
        batches.append(
            _full_split_interval_batch(
                split,
                source_index=source_index,
                target_index=target_index,
                z_s=current_state,
                z_t=teacher_target_state,
            )
        )
        current_state = predicted_next_state

    if final_state_rms_gap is None:
        raise RuntimeError("Rollout auxiliary batches require at least one rollout segment.")
    return FMPCIntervalRolloutAuxiliaryBatches(
        rollout_schedule_name=rollout_schedule_name,
        knots=resolved_knots,
        batches=tuple(batches),
        rollout_aux_velocity_mse=float(np.mean(np.asarray(velocity_mses, dtype=np.float64))),
        intermediate_state_rms_gap=(
            None
            if not intermediate_state_rms_gaps
            else float(np.mean(np.asarray(intermediate_state_rms_gaps, dtype=np.float64)))
        ),
        final_state_rms_gap=float(final_state_rms_gap),
    )


def _evaluate_interval_rollout(
    model: Any,
    interval_split: FMPCIntervalSplit,
    endpoint_split: Any,
    reference: Any,
    teacher_model: Any,
    *,
    rollout_schedule_name: str,
    knots: tuple[int, ...],
) -> tuple[FMPCIntervalRolloutResult, dict[str, Any]]:
    rollout = rollout_interval_student(
        model,
        interval_split,
        rollout_schedule_name=rollout_schedule_name,
        knots=knots,
    )
    delta_z_hat = np.asarray(rollout.final_state - endpoint_split.z0, dtype=np.float64)
    endpoint_evaluation = evaluate_fmpc_delta_predictions(
        delta_z_hat,
        endpoint_split,
        reference,
        teacher_model,
        transport_wall_time_seconds=rollout.transport_wall_time_seconds,
    )
    metrics = _interval_metrics_from_endpoint(
        fmpc_split_evaluation_metrics_payload(endpoint_evaluation),
        mean_knot_state_rms_gap=rollout.mean_knot_state_rms_gap,
    )
    return rollout, metrics


@dataclass
class FMPCIntervalSuiteConfig:
    """Configuration for the Phase 5B offline interval-conditioned transporter suite."""

    experiment_name: str = "fmpc_interval_suite"
    dataset_name: str = "digits"
    teacher_preparation_path: str | Path = "outputs/fmpc_v0_prepare_digits"
    run_seed: int = 0
    model_init_seed: int = 0
    batch_order_seed: int = 0
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: OutputLayout = "single_dir"
    normalization_eps: float = 1e-8
    carried_forward_endpoint_ridge_alpha: float = 1e-4
    interval_ridge_alphas: tuple[float, ...] = (1e-4, 1e-2, 1.0, 100.0)
    gradient_augmented_feature_names: tuple[str, ...] = ("g_s", "e_out_s", "F_s")
    augmented_knot_focused_schedule_names: tuple[str, ...] = ("2-step", "3-step")
    augmented_knot_focus_mixture_candidates: tuple[float, ...] = (0.0, 0.5)
    mlp_hidden_dims_candidates: tuple[tuple[int, ...], ...] = ((64,), (128,), (128, 128))
    mlp_epochs_candidates: tuple[int, ...] = (20, 40)
    mlp_eta_w_candidates: tuple[float, ...] = (0.01, 0.05)
    mlp_weight_scale: float = 0.02
    mlp_batch_size: int = 64
    mlp_batches_per_epoch: int | None = None
    mlp_hidden_activation: str = "tanh"
    mlp_output_activation: str = "identity"
    mlp_rollout_aux_schedule_names: tuple[str, ...] = ("2-step", "3-step")
    mlp_rollout_aux_weight_candidates: tuple[float, ...] = (0.0, 0.25, 0.5)
    rollout_schedule_names: tuple[str, ...] = ("1-step", "2-step", "3-step")
    allow_teacher_retrain: bool = False

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_{self.run_seed}"


@dataclass
class FMPCIntervalSuiteRunResult:
    run_dir: Path
    config: dict[str, Any]
    candidates: list[dict[str, Any]]
    summary: dict[str, Any]


def _teacher_export_batch_size_from_endpoint_dataset(endpoint_dataset: FMPCStudentDataset) -> int:
    return int(endpoint_dataset.metadata["teacher_export_batch_size"])


def _prepare_interval_teacher_feature_bundle(
    *,
    interval_dataset: FMPCIntervalDataset,
    teacher_split: Any,
    teacher_model: Any,
    teacher_export_batch_size: int,
) -> FMPCIntervalTeacherFeatureBundle:
    train_context = prepare_interval_teacher_feature_context(
        interval_dataset.train,
        teacher_split,
        teacher_export_batch_size=teacher_export_batch_size,
    )
    val_context = prepare_interval_teacher_feature_context(
        interval_dataset.val,
        teacher_split,
        teacher_export_batch_size=teacher_export_batch_size,
    )
    test_context = prepare_interval_teacher_feature_context(
        interval_dataset.test,
        teacher_split,
        teacher_export_batch_size=teacher_export_batch_size,
    )
    return FMPCIntervalTeacherFeatureBundle(
        teacher_steps=int(interval_dataset.teacher_steps),
        train=precompute_interval_teacher_trajectory_features(
            teacher_model,
            interval_dataset.train,
            train_context,
        ),
        val=precompute_interval_teacher_trajectory_features(
            teacher_model,
            interval_dataset.val,
            val_context,
        ),
        test=precompute_interval_teacher_trajectory_features(
            teacher_model,
            interval_dataset.test,
            test_context,
        ),
        train_tangents=precompute_interval_teacher_trajectory_feature_tangents(
            teacher_model,
            interval_dataset.train,
            train_context,
            fd_epsilon=1e-3,
        ),
        val_tangents=precompute_interval_teacher_trajectory_feature_tangents(
            teacher_model,
            interval_dataset.val,
            val_context,
            fd_epsilon=1e-3,
        ),
        test_tangents=precompute_interval_teacher_trajectory_feature_tangents(
            teacher_model,
            interval_dataset.test,
            test_context,
            fd_epsilon=1e-3,
        ),
        split_contexts={
            "train": train_context,
            "val": val_context,
            "test": test_context,
        },
    )


def _endpoint_baseline_summary_payload(
    *,
    family: str,
    config_id: str,
    model_config: dict[str, Any],
    val_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "family": family,
        "config_id": config_id,
        "selection_metric_source": "val_metric",
        "report_metric_source": "test_metric",
        "model_config": model_config,
        "val": val_metrics,
        "test": test_metrics,
    }


def _identity_baseline_row_and_summary(
    endpoint_dataset: FMPCStudentDataset,
    references: dict[str, Any],
    teacher_model: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    val_eval = evaluate_fmpc_identity_baseline(endpoint_dataset.val, references["val"], teacher_model)
    test_eval = evaluate_fmpc_identity_baseline(endpoint_dataset.test, references["test"], teacher_model)
    val_metrics = _interval_metrics_from_endpoint(
        fmpc_split_evaluation_metrics_payload(val_eval),
        mean_knot_state_rms_gap=None,
    )
    test_metrics = _interval_metrics_from_endpoint(
        fmpc_split_evaluation_metrics_payload(test_eval),
        mean_knot_state_rms_gap=None,
    )
    row = {
        "config_id": "identity",
        "family": "identity",
        "rollout_schedule": "endpoint",
        "schedule_knots": "",
        "normalization": "none",
        "alpha": None,
        "hidden_dims": None,
        "epochs": None,
        "eta_w": None,
        "feature_contract": "none",
        "target_contract": "delta_z_hat = 0",
        "knot_focused_sampling_mixture": None,
        "knot_focused_schedule_names": "",
        "is_family_best": True,
        "is_learned_family": False,
        "is_overall_winner": False,
        "evaluated_on_test": True,
        **_candidate_metric_columns("val", val_metrics),
        **_candidate_metric_columns("test", test_metrics),
    }
    summary = _endpoint_baseline_summary_payload(
        family="identity",
        config_id="identity",
        model_config={"family": "identity", "student_output_definition": "delta_z_hat = 0"},
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )
    return row, summary


def _endpoint_ridge_baseline_row_and_summary(
    *,
    config: FMPCIntervalSuiteConfig,
    endpoint_dataset: FMPCStudentDataset,
    references: dict[str, Any],
    teacher_model: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    endpoint_normalization = fit_fmpc_student_normalization(endpoint_dataset.train, eps=config.normalization_eps)
    baseline_model = RidgeDeltaStudent.fit(
        endpoint_dataset.train,
        normalization=endpoint_normalization,
        config=RidgeDeltaStudentConfig(alpha=float(config.carried_forward_endpoint_ridge_alpha)),
    )
    val_delta = baseline_model.predict_delta_z(endpoint_dataset.val)
    val_eval = evaluate_fmpc_delta_predictions(
        val_delta,
        endpoint_dataset.val,
        references["val"],
        teacher_model,
        transport_wall_time_seconds=0.0,
    )
    test_delta = baseline_model.predict_delta_z(endpoint_dataset.test)
    test_eval = evaluate_fmpc_delta_predictions(
        test_delta,
        endpoint_dataset.test,
        references["test"],
        teacher_model,
        transport_wall_time_seconds=0.0,
    )
    val_metrics = _interval_metrics_from_endpoint(
        fmpc_split_evaluation_metrics_payload(val_eval),
        mean_knot_state_rms_gap=None,
    )
    test_metrics = _interval_metrics_from_endpoint(
        fmpc_split_evaluation_metrics_payload(test_eval),
        mean_knot_state_rms_gap=None,
    )
    row = {
        "config_id": f"phase5a_endpoint_ridge_alpha_{config.carried_forward_endpoint_ridge_alpha:g}",
        "family": "phase5a_endpoint_ridge",
        "rollout_schedule": "endpoint",
        "schedule_knots": "",
        "normalization": "train_stats_endpoint",
        "alpha": float(config.carried_forward_endpoint_ridge_alpha),
        "hidden_dims": None,
        "epochs": None,
        "eta_w": None,
        "feature_contract": endpoint_dataset.student_input_definition,
        "target_contract": endpoint_dataset.student_target_definition,
        "knot_focused_sampling_mixture": None,
        "knot_focused_schedule_names": "",
        "is_family_best": True,
        "is_learned_family": False,
        "is_overall_winner": False,
        "evaluated_on_test": True,
        **_candidate_metric_columns("val", val_metrics),
        **_candidate_metric_columns("test", test_metrics),
    }
    summary = _endpoint_baseline_summary_payload(
        family="phase5a_endpoint_ridge",
        config_id=str(row["config_id"]),
        model_config={
            "family": "phase5a_endpoint_ridge",
            "alpha": float(config.carried_forward_endpoint_ridge_alpha),
            "student_input_definition": endpoint_dataset.student_input_definition,
            "student_target_definition": endpoint_dataset.student_target_definition,
        },
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )
    return row, summary


def _family_summary_payload(
    *,
    family: str,
    config_id: str,
    rollout_schedule: str,
    schedule_knots: tuple[int, ...],
    model_config: dict[str, Any],
    val_metrics: dict[str, Any],
    test_metrics: dict[str, Any] | None,
    best_epoch: int | None = None,
    training_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "family": family,
        "config_id": config_id,
        "rollout_schedule": rollout_schedule,
        "schedule_knots": list(schedule_knots),
        "selection_metric_source": "val_metric",
        "report_metric_source": "test_metric",
        "model_config": model_config,
        "best_epoch": best_epoch,
        "training_diagnostics": training_diagnostics,
        "val": val_metrics,
        "test": test_metrics,
    }


def _build_interval_ridge_rows(
    *,
    config: FMPCIntervalSuiteConfig,
    interval_dataset: FMPCIntervalDataset,
    endpoint_dataset: FMPCStudentDataset,
    references: dict[str, Any],
    teacher_model: Any,
    normalization: FMPCIntervalNormalizationStats,
    rollout_schedules: dict[str, tuple[int, ...]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], IntervalRidgeStudent]:
    rows: list[dict[str, Any]] = []
    family_best_row: dict[str, Any] | None = None
    family_best_metrics: dict[str, Any] | None = None
    family_best_model: IntervalRidgeStudent | None = None
    for alpha in config.interval_ridge_alphas:
        model = IntervalRidgeStudent.fit(
            interval_dataset.train,
            normalization=normalization,
            config=IntervalRidgeStudentConfig(alpha=float(alpha)),
        )
        for schedule_name, knots in rollout_schedules.items():
            _, val_metrics = _evaluate_interval_rollout(
                model,
                interval_dataset.val,
                endpoint_dataset.val,
                references["val"],
                teacher_model,
                rollout_schedule_name=schedule_name,
                knots=knots,
            )
            row = {
                "config_id": f"interval_ridge_alpha_{alpha:g}",
                "family": "interval_ridge",
                "rollout_schedule": schedule_name,
                "schedule_knots": "-".join(str(knot) for knot in knots),
                "normalization": "train_stats_interval",
                "alpha": float(alpha),
                "hidden_dims": None,
                "epochs": None,
                "eta_w": None,
                "feature_contract": interval_dataset.interval_input_definition,
                "target_contract": interval_dataset.interval_target_definition,
                "knot_focused_sampling_mixture": 0.0,
                "knot_focused_schedule_names": "",
                "is_family_best": False,
                "is_learned_family": True,
                "is_overall_winner": False,
                "evaluated_on_test": False,
                **_candidate_metric_columns("val", val_metrics),
            }
            rows.append(row)
            if (
                family_best_row is None
                or float(val_metrics["final_state_rms_gap"]) < float(family_best_metrics["final_state_rms_gap"])
            ):
                family_best_row = row
                family_best_metrics = val_metrics
                family_best_model = model
    if family_best_row is None or family_best_metrics is None or family_best_model is None:
        raise RuntimeError("interval_ridge family did not produce a valid candidate.")
    family_best_row["is_family_best"] = True
    summary = _family_summary_payload(
        family="interval_ridge",
        config_id=str(family_best_row["config_id"]),
        rollout_schedule=str(family_best_row["rollout_schedule"]),
        schedule_knots=tuple(
            int(value) for value in str(family_best_row["schedule_knots"]).split("-") if value != ""
        ),
        model_config=family_best_model.to_jsonable(),
        val_metrics=family_best_metrics,
        test_metrics=None,
    )
    return rows, summary, family_best_row, family_best_model


def _build_interval_augmented_ridge_rows(
    *,
    family_name: str,
    target_mode: Literal["u_star", "u_residual_local_field"],
    config: FMPCIntervalSuiteConfig,
    interval_dataset: FMPCIntervalDataset,
    endpoint_dataset: FMPCStudentDataset,
    references: dict[str, Any],
    teacher_model: Any,
    teacher_feature_bundle: FMPCIntervalTeacherFeatureBundle,
    rollout_schedules: dict[str, tuple[int, ...]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], IntervalAugmentedRidgeStudent]:
    rows: list[dict[str, Any]] = []
    family_best_row: dict[str, Any] | None = None
    family_best_metrics: dict[str, Any] | None = None
    family_best_model: IntervalAugmentedRidgeStudent | None = None

    for alpha in config.interval_ridge_alphas:
        for knot_focus_mixture in config.augmented_knot_focus_mixture_candidates:
            normalization = fit_fmpc_interval_augmented_normalization(
                interval_dataset.train,
                trajectory_features=teacher_feature_bundle.train,
                selected_feature_names=config.gradient_augmented_feature_names,
                target_mode=target_mode,
                knot_focused_schedule_names=config.augmented_knot_focused_schedule_names,
                knot_focus_mixture=float(knot_focus_mixture),
                eps=config.normalization_eps,
            )
            model = IntervalAugmentedRidgeStudent.fit(
                interval_dataset.train,
                normalization=normalization,
                config=IntervalAugmentedRidgeStudentConfig(
                    alpha=float(alpha),
                    selected_teacher_feature_names=tuple(config.gradient_augmented_feature_names),
                    target_mode=target_mode,
                    knot_focused_schedule_names=tuple(config.augmented_knot_focused_schedule_names),
                    knot_focus_mixture=float(knot_focus_mixture),
                ),
                family_name=family_name,
                teacher_model=teacher_model,
                teacher_feature_bundle=teacher_feature_bundle,
            )
            for schedule_name, knots in rollout_schedules.items():
                _, val_metrics = _evaluate_interval_rollout(
                    model,
                    interval_dataset.val,
                    endpoint_dataset.val,
                    references["val"],
                    teacher_model,
                    rollout_schedule_name=schedule_name,
                    knots=knots,
                )
                row = {
                    "config_id": f"{family_name}_alpha_{alpha:g}_focus{knot_focus_mixture:g}",
                    "family": family_name,
                    "rollout_schedule": schedule_name,
                    "schedule_knots": "-".join(str(knot) for knot in knots),
                    "normalization": "train_stats_interval_augmented",
                    "alpha": float(alpha),
                    "hidden_dims": None,
                    "epochs": None,
                    "eta_w": None,
                    "feature_contract": ",".join(config.gradient_augmented_feature_names),
                    "target_contract": "u_star" if target_mode == "u_star" else "u_res = u_star - g_s",
                    "knot_focused_sampling_mixture": float(knot_focus_mixture),
                    "knot_focused_schedule_names": ",".join(config.augmented_knot_focused_schedule_names),
                    "is_family_best": False,
                    "is_learned_family": True,
                    "is_overall_winner": False,
                    "evaluated_on_test": False,
                    **_candidate_metric_columns("val", val_metrics),
                }
                rows.append(row)
                if (
                    family_best_row is None
                    or float(val_metrics["final_state_rms_gap"]) < float(family_best_metrics["final_state_rms_gap"])
                ):
                    family_best_row = row
                    family_best_metrics = val_metrics
                    family_best_model = model
    if family_best_row is None or family_best_metrics is None or family_best_model is None:
        raise RuntimeError(f"{family_name} did not produce a valid candidate.")
    family_best_row["is_family_best"] = True
    summary = _family_summary_payload(
        family=family_name,
        config_id=str(family_best_row["config_id"]),
        rollout_schedule=str(family_best_row["rollout_schedule"]),
        schedule_knots=tuple(
            int(value) for value in str(family_best_row["schedule_knots"]).split("-") if value != ""
        ),
        model_config=family_best_model.to_jsonable(),
        val_metrics=family_best_metrics,
        test_metrics=None,
    )
    return rows, summary, family_best_row, family_best_model


def _build_interval_mlp_candidate_model(
    *,
    interval_dataset: FMPCIntervalDataset,
    normalization: FMPCIntervalNormalizationStats,
    config: IntervalStandardizedMLPStudentConfig,
    seed: int,
) -> IntervalStandardizedMLPStudent:
    return IntervalStandardizedMLPStudent.initialize(
        z_dim=interval_dataset.z_dim,
        target_dim=interval_dataset.target_dim,
        normalization=normalization,
        config=config,
        seed=seed,
    )


def _rollout_aux_training_diagnostics(
    model: IntervalStandardizedMLPStudent,
    train_split: FMPCIntervalSplit,
    *,
    aux_rollout_schedules: dict[str, tuple[int, ...]],
) -> tuple[list[FMPCIntervalBatch], dict[str, Any]]:
    aux_batches: list[FMPCIntervalBatch] = []
    velocity_mses: list[float] = []
    intermediate_state_rms_gaps: list[float] = []
    final_state_rms_gaps: list[float] = []
    per_schedule: dict[str, Any] = {}

    for schedule_name, knots in aux_rollout_schedules.items():
        auxiliary = build_rollout_auxiliary_batches(
            model,
            train_split,
            rollout_schedule_name=schedule_name,
            knots=knots,
        )
        aux_batches.extend(auxiliary.batches)
        velocity_mses.append(float(auxiliary.rollout_aux_velocity_mse))
        if auxiliary.intermediate_state_rms_gap is not None:
            intermediate_state_rms_gaps.append(float(auxiliary.intermediate_state_rms_gap))
        final_state_rms_gaps.append(float(auxiliary.final_state_rms_gap))
        per_schedule[schedule_name] = {
            "rollout_aux_velocity_mse": float(auxiliary.rollout_aux_velocity_mse),
            "intermediate_state_rms_gap": auxiliary.intermediate_state_rms_gap,
            "final_state_rms_gap": float(auxiliary.final_state_rms_gap),
        }

    diagnostics = {
        "rollout_aux_velocity_mse": _mean_or_none(velocity_mses),
        "rollout_aux_intermediate_state_rms_gap": _mean_or_none(intermediate_state_rms_gaps),
        "rollout_aux_final_state_rms_gap": _mean_or_none(final_state_rms_gaps),
        "rollout_aux_per_schedule": per_schedule,
    }
    return aux_batches, diagnostics


def _build_interval_mlp_rows(
    *,
    config: FMPCIntervalSuiteConfig,
    interval_dataset: FMPCIntervalDataset,
    endpoint_dataset: FMPCStudentDataset,
    references: dict[str, Any],
    teacher_model: Any,
    normalization: FMPCIntervalNormalizationStats,
    rollout_schedules: dict[str, tuple[int, ...]],
    aux_rollout_schedules: dict[str, tuple[int, ...]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    family_best_row: dict[str, Any] | None = None
    family_best_metrics: dict[str, Any] | None = None
    family_best_record: dict[str, Any] | None = None

    candidate_counter = 0
    batches_per_epoch = int(
        config.mlp_batches_per_epoch
        if config.mlp_batches_per_epoch is not None
        else int(np.ceil(interval_dataset.train.num_samples / config.mlp_batch_size))
    )

    for hidden_dims in config.mlp_hidden_dims_candidates:
        for epochs in config.mlp_epochs_candidates:
            for eta_w in config.mlp_eta_w_candidates:
                for rollout_aux_weight in config.mlp_rollout_aux_weight_candidates:
                    rollout_aux_schedule_names = (
                        tuple(config.mlp_rollout_aux_schedule_names) if rollout_aux_weight > 0.0 else ()
                    )
                    mlp_config = IntervalStandardizedMLPStudentConfig(
                        hidden_dims=tuple(int(value) for value in hidden_dims),
                        hidden_activation=config.mlp_hidden_activation,
                        output_activation=config.mlp_output_activation,
                        weight_scale=config.mlp_weight_scale,
                        eta_w=float(eta_w),
                        eta_b=float(eta_w),
                        epochs=int(epochs),
                        batch_size=int(config.mlp_batch_size),
                        batches_per_epoch=batches_per_epoch,
                        rollout_aux_weight=float(rollout_aux_weight),
                        rollout_aux_schedule_names=rollout_aux_schedule_names,
                    )
                    model = _build_interval_mlp_candidate_model(
                        interval_dataset=interval_dataset,
                        normalization=normalization,
                        config=mlp_config,
                        seed=config.model_init_seed + candidate_counter,
                    )
                    candidate_id = (
                        f"interval_mlp_h{_serialize_hidden_dims(mlp_config.hidden_dims)}"
                        f"_e{mlp_config.epochs}_lr{mlp_config.eta_w:g}"
                        f"_aux{mlp_config.rollout_aux_weight:g}"
                    )
                    best_by_schedule: dict[str, dict[str, Any]] = {
                        schedule_name: {
                            "val_metric": np.inf,
                            "snapshot": None,
                        }
                        for schedule_name in rollout_schedules
                    }
                    for epoch in range(1, mlp_config.epochs + 1):
                        primary_losses: list[float] = []
                        for batch in iter_balanced_interval_batches(
                            interval_dataset.train,
                            mlp_config.batch_size,
                            num_batches=batches_per_epoch,
                            seed=config.batch_order_seed + candidate_counter * 1000 + epoch,
                        ):
                            primary_losses.append(model.train_batch(batch))
                        rollout_aux_velocity_losses: list[float] = []
                        rollout_aux_intermediate_state_rms_gaps: list[float] = []
                        rollout_aux_final_state_rms_gaps: list[float] = []
                        rollout_aux_per_schedule: dict[str, Any] = {}
                        if mlp_config.rollout_aux_weight > 0.0:
                            aux_batches, aux_diagnostics = _rollout_aux_training_diagnostics(
                                model,
                                interval_dataset.train,
                                aux_rollout_schedules=aux_rollout_schedules,
                            )
                            if aux_diagnostics["rollout_aux_velocity_mse"] is not None:
                                rollout_aux_velocity_losses.append(
                                    float(aux_diagnostics["rollout_aux_velocity_mse"])
                                )
                            if aux_diagnostics["rollout_aux_intermediate_state_rms_gap"] is not None:
                                rollout_aux_intermediate_state_rms_gaps.append(
                                    float(aux_diagnostics["rollout_aux_intermediate_state_rms_gap"])
                                )
                            if aux_diagnostics["rollout_aux_final_state_rms_gap"] is not None:
                                rollout_aux_final_state_rms_gaps.append(
                                    float(aux_diagnostics["rollout_aux_final_state_rms_gap"])
                                )
                            rollout_aux_per_schedule = dict(aux_diagnostics["rollout_aux_per_schedule"])
                            for aux_batch in aux_batches:
                                model.train_batch(aux_batch, lr_scale=mlp_config.rollout_aux_weight)
                        epoch_training_metrics = {
                            "primary_interval_loss": _mean_or_none(primary_losses),
                            "rollout_aux_velocity_mse": _mean_or_none(rollout_aux_velocity_losses),
                            "rollout_aux_intermediate_state_rms_gap": _mean_or_none(
                                rollout_aux_intermediate_state_rms_gaps
                            ),
                            "rollout_aux_final_state_rms_gap": _mean_or_none(
                                rollout_aux_final_state_rms_gaps
                            ),
                            "rollout_aux_per_schedule": rollout_aux_per_schedule,
                        }
                        for schedule_name, knots in rollout_schedules.items():
                            _, val_metrics = _evaluate_interval_rollout(
                                model,
                                interval_dataset.val,
                                endpoint_dataset.val,
                                references["val"],
                                teacher_model,
                                rollout_schedule_name=schedule_name,
                                knots=knots,
                            )
                            if float(val_metrics["final_state_rms_gap"]) < float(
                                best_by_schedule[schedule_name]["val_metric"]
                            ):
                                best_by_schedule[schedule_name] = {
                                    "epoch": epoch,
                                    "val_metric": float(val_metrics["final_state_rms_gap"]),
                                    "snapshot": model.snapshot(),
                                    "val_metrics": val_metrics,
                                    "training_metrics": epoch_training_metrics,
                                }

                    for schedule_name, knots in rollout_schedules.items():
                        schedule_best = best_by_schedule[schedule_name]
                        if schedule_best["snapshot"] is None:
                            raise RuntimeError(f"{candidate_id} did not record a best snapshot for {schedule_name}.")
                        row = {
                            "config_id": candidate_id,
                            "family": "interval_mlp_standardized",
                            "rollout_schedule": schedule_name,
                            "schedule_knots": "-".join(str(knot) for knot in knots),
                            "normalization": "train_stats_interval",
                            "alpha": None,
                            "hidden_dims": _serialize_hidden_dims(mlp_config.hidden_dims),
                            "epochs": int(mlp_config.epochs),
                            "eta_w": float(mlp_config.eta_w),
                            "feature_contract": interval_dataset.interval_input_definition,
                            "target_contract": interval_dataset.interval_target_definition,
                            "knot_focused_sampling_mixture": 0.0,
                            "knot_focused_schedule_names": "",
                            "rollout_aux_weight": float(mlp_config.rollout_aux_weight),
                            "rollout_aux_schedules": ",".join(mlp_config.rollout_aux_schedule_names),
                            "best_epoch": int(schedule_best["epoch"]),
                            "train_primary_interval_loss": schedule_best["training_metrics"]["primary_interval_loss"],
                            "train_rollout_aux_velocity_mse": schedule_best["training_metrics"][
                                "rollout_aux_velocity_mse"
                            ],
                            "train_rollout_aux_intermediate_state_rms_gap": schedule_best["training_metrics"][
                                "rollout_aux_intermediate_state_rms_gap"
                            ],
                            "train_rollout_aux_final_state_rms_gap": schedule_best["training_metrics"][
                                "rollout_aux_final_state_rms_gap"
                            ],
                            "is_family_best": False,
                            "is_learned_family": True,
                            "is_overall_winner": False,
                            "evaluated_on_test": False,
                            **_candidate_metric_columns("val", schedule_best["val_metrics"]),
                        }
                        rows.append(row)
                        if (
                            family_best_row is None
                            or float(schedule_best["val_metrics"]["final_state_rms_gap"])
                            < float(family_best_metrics["final_state_rms_gap"])
                        ):
                            family_best_row = row
                            family_best_metrics = schedule_best["val_metrics"]
                            family_best_record = {
                                "config": mlp_config,
                                "snapshot": schedule_best["snapshot"],
                                "seed": config.model_init_seed + candidate_counter,
                                "best_epoch": int(schedule_best["epoch"]),
                                "training_metrics": dict(schedule_best["training_metrics"]),
                            }
                    candidate_counter += 1

    if family_best_row is None or family_best_metrics is None or family_best_record is None:
        raise RuntimeError("interval_mlp_standardized family did not produce a valid candidate.")
    family_best_row["is_family_best"] = True
    summary = _family_summary_payload(
        family="interval_mlp_standardized",
        config_id=str(family_best_row["config_id"]),
        rollout_schedule=str(family_best_row["rollout_schedule"]),
        schedule_knots=tuple(
            int(value) for value in str(family_best_row["schedule_knots"]).split("-") if value != ""
        ),
        model_config=_build_interval_mlp_candidate_model(
            interval_dataset=interval_dataset,
            normalization=normalization,
            config=family_best_record["config"],
            seed=int(family_best_record["seed"]),
        ).to_jsonable(),
        val_metrics=family_best_metrics,
        test_metrics=None,
        best_epoch=int(family_best_record["best_epoch"]),
        training_diagnostics=dict(family_best_record["training_metrics"]),
    )
    return rows, summary, family_best_row, family_best_record


def _suite_config_payload(
    *,
    config: FMPCIntervalSuiteConfig,
    run_id: str,
    run_dir: Path,
    endpoint_dataset: FMPCStudentDataset,
    interval_dataset: FMPCIntervalDataset,
    normalization: FMPCIntervalNormalizationStats,
    teacher_checkpoint_loaded: bool,
    comparison_atol: float,
    rollout_schedules: dict[str, tuple[int, ...]],
    aux_rollout_schedules: dict[str, tuple[int, ...]],
) -> dict[str, Any]:
    return {
        "experiment_name": config.experiment_name,
        "run_id": run_id,
        "phase": "Phase 5",
        "stage": "phase5b_interval_conditioned_transporter",
        "dataset_name": config.dataset_name,
        "teacher_artifact_dir": _relative_artifact_reference(run_dir, config.teacher_preparation_path),
        "teacher_manifest_path": _relative_artifact_reference(run_dir, endpoint_dataset.teacher_manifest_path),
        "teacher_checkpoint_path": _relative_artifact_reference(run_dir, endpoint_dataset.teacher_checkpoint_path),
        "teacher_checkpoint_loaded": bool(teacher_checkpoint_loaded),
        "teacher_reference_comparison_atol": float(comparison_atol),
        "teacher_steps": int(interval_dataset.teacher_steps),
        "rollout_schedules": {name: list(knots) for name, knots in rollout_schedules.items()},
        "rollout_aware_auxiliary_schedules": {name: list(knots) for name, knots in aux_rollout_schedules.items()},
        "selection_metric_name": "final_state_rms_gap",
        "selection_metric_source": "val_metric",
        "report_metric_source": "test_metric",
        "student_input_definition": interval_dataset.interval_input_definition,
        "student_target_definition": interval_dataset.interval_target_definition,
        "run_seed": config.run_seed,
        "model_init_seed": config.model_init_seed,
        "batch_order_seed": config.batch_order_seed,
        "teacher_recovery": {
            "allow_teacher_retrain": bool(config.allow_teacher_retrain),
            "teacher_checkpoint_required_by_default": True,
        },
        "normalization": normalization.to_jsonable(),
        "search_space": {
            "interval_families": [
                "interval_ridge",
                "interval_ridge_aug",
                "interval_ridge_residual",
                "interval_mlp_standardized",
            ],
            "interval_ridge_alphas": [float(value) for value in config.interval_ridge_alphas],
            "gradient_augmented_feature_names": list(config.gradient_augmented_feature_names),
            "augmented_knot_focused_schedule_names": list(config.augmented_knot_focused_schedule_names),
            "augmented_knot_focus_mixture_candidates": [
                float(value) for value in config.augmented_knot_focus_mixture_candidates
            ],
            "mlp_hidden_dims_candidates": [list(candidate) for candidate in config.mlp_hidden_dims_candidates],
            "mlp_epochs_candidates": [int(value) for value in config.mlp_epochs_candidates],
            "mlp_eta_w_candidates": [float(value) for value in config.mlp_eta_w_candidates],
            "mlp_rollout_aux_weight_candidates": [
                float(value) for value in config.mlp_rollout_aux_weight_candidates
            ],
            "mlp_rollout_aux_schedule_names": list(config.mlp_rollout_aux_schedule_names),
            "mlp_weight_scale": float(config.mlp_weight_scale),
            "mlp_batch_size": int(config.mlp_batch_size),
            "mlp_batches_per_epoch": config.mlp_batches_per_epoch,
            "mlp_hidden_activation": config.mlp_hidden_activation,
            "mlp_output_activation": config.mlp_output_activation,
            "carried_forward_phase5a_endpoint_ridge_alpha": float(config.carried_forward_endpoint_ridge_alpha),
        },
        "pair_sampling": {
            "policy": "span_balanced_uniform_span_then_uniform_start_then_uniform_sample",
            "short_interval_dominance_avoided": True,
            "gradient_augmented_rescue": {
                "knot_focused_schedule_names": list(config.augmented_knot_focused_schedule_names),
                "knot_focus_mixture_candidates": [float(value) for value in config.augmented_knot_focus_mixture_candidates],
            },
        },
        "training_objectives": {
            "primary_interval_loss": "MSE(u_hat, u_star)",
            "rollout_auxiliary_loss": {
                "enabled_for_family": "interval_mlp_standardized",
                "teacher_supervised": True,
                "self_fed_between_knots": True,
                "meanflow_or_jvp_used": False,
                "intermediate_and_final_losses_recorded_separately": True,
            },
        },
    }


def run_fmpc_interval_suite(config: FMPCIntervalSuiteConfig) -> FMPCIntervalSuiteRunResult:
    """Run the Phase 5B interval-conditioned offline transporter suite on `digits`."""

    if config.dataset_name != "digits":
        raise ValueError("Phase 5B interval suite currently supports digits only.")
    set_seed(config.run_seed)

    endpoint_dataset = load_fmpc_student_dataset(config.teacher_preparation_path, expected_dataset_name="digits")
    interval_dataset = load_fmpc_interval_dataset(config.teacher_preparation_path, expected_dataset_name="digits")
    teacher_model, teacher_split, used_teacher_retrain_fallback, comparison_atol = load_fmpc_student_teacher_runtime(
        endpoint_dataset,
        allow_teacher_retrain=config.allow_teacher_retrain,
    )
    references = prepare_fmpc_student_teacher_references(
        endpoint_dataset,
        teacher_model,
        teacher_split,
        comparison_atol=comparison_atol,
    )
    teacher_feature_bundle = _prepare_interval_teacher_feature_bundle(
        interval_dataset=interval_dataset,
        teacher_split=teacher_split,
        teacher_model=teacher_model,
        teacher_export_batch_size=_teacher_export_batch_size_from_endpoint_dataset(endpoint_dataset),
    )
    normalization = fit_fmpc_interval_normalization(interval_dataset.train, eps=config.normalization_eps)
    default_schedules = teacher_step_aligned_rollout_schedules(interval_dataset.teacher_steps)
    missing_schedule_names = [name for name in config.rollout_schedule_names if name not in default_schedules]
    if missing_schedule_names:
        raise ValueError(
            f"Unsupported rollout schedule names {missing_schedule_names}; "
            f"expected a subset of {list(default_schedules.keys())}."
        )
    invalid_aux_schedule_names = [name for name in config.mlp_rollout_aux_schedule_names if name not in {"2-step", "3-step"}]
    if invalid_aux_schedule_names:
        raise ValueError(
            "Phase 5B rollout-aware rescue only permits 2-step and 3-step auxiliary schedules; "
            f"got {invalid_aux_schedule_names}."
        )
    invalid_aug_schedule_names = [
        name for name in config.augmented_knot_focused_schedule_names if name not in {"2-step", "3-step"}
    ]
    if invalid_aug_schedule_names:
        raise ValueError(
            "Phase 5B.2 knot-focused rescue only permits 2-step and 3-step focus schedules; "
            f"got {invalid_aug_schedule_names}."
        )
    invalid_knot_focus_mixtures = [
        value for value in config.augmented_knot_focus_mixture_candidates if not (0.0 <= float(value) <= 1.0)
    ]
    if invalid_knot_focus_mixtures:
        raise ValueError(
            "augmented_knot_focus_mixture_candidates must lie in [0, 1]; "
            f"got {invalid_knot_focus_mixtures}."
        )
    missing_aux_schedule_names = [name for name in config.mlp_rollout_aux_schedule_names if name not in default_schedules]
    if missing_aux_schedule_names:
        raise ValueError(
            f"Unsupported rollout-aware auxiliary schedule names {missing_aux_schedule_names}; "
            f"expected a subset of {list(default_schedules.keys())}."
        )
    rollout_schedules = {name: default_schedules[name] for name in config.rollout_schedule_names}
    aux_rollout_schedules = {name: default_schedules[name] for name in config.mlp_rollout_aux_schedule_names}

    run_id = config.resolved_run_id()
    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, run_id, config.output_layout)
    )

    identity_row, identity_summary = _identity_baseline_row_and_summary(endpoint_dataset, references, teacher_model)
    endpoint_ridge_row, endpoint_ridge_summary = _endpoint_ridge_baseline_row_and_summary(
        config=config,
        endpoint_dataset=endpoint_dataset,
        references=references,
        teacher_model=teacher_model,
    )
    ridge_rows, ridge_summary, ridge_family_best_row, ridge_family_best_model = _build_interval_ridge_rows(
        config=config,
        interval_dataset=interval_dataset,
        endpoint_dataset=endpoint_dataset,
        references=references,
        teacher_model=teacher_model,
        normalization=normalization,
        rollout_schedules=rollout_schedules,
    )
    ridge_aug_rows, ridge_aug_summary, ridge_aug_family_best_row, ridge_aug_family_best_model = (
        _build_interval_augmented_ridge_rows(
            family_name="interval_ridge_aug",
            target_mode="u_star",
            config=config,
            interval_dataset=interval_dataset,
            endpoint_dataset=endpoint_dataset,
            references=references,
            teacher_model=teacher_model,
            teacher_feature_bundle=teacher_feature_bundle,
            rollout_schedules=rollout_schedules,
        )
    )
    ridge_residual_rows, ridge_residual_summary, ridge_residual_family_best_row, ridge_residual_family_best_model = (
        _build_interval_augmented_ridge_rows(
            family_name="interval_ridge_residual",
            target_mode="u_residual_local_field",
            config=config,
            interval_dataset=interval_dataset,
            endpoint_dataset=endpoint_dataset,
            references=references,
            teacher_model=teacher_model,
            teacher_feature_bundle=teacher_feature_bundle,
            rollout_schedules=rollout_schedules,
        )
    )
    mlp_rows, mlp_summary, mlp_family_best_row, mlp_family_best_record = _build_interval_mlp_rows(
        config=config,
        interval_dataset=interval_dataset,
        endpoint_dataset=endpoint_dataset,
        references=references,
        teacher_model=teacher_model,
        normalization=normalization,
        rollout_schedules=rollout_schedules,
        aux_rollout_schedules=aux_rollout_schedules,
    )

    learned_family_best_rows = [
        ridge_family_best_row,
        ridge_aug_family_best_row,
        ridge_residual_family_best_row,
        mlp_family_best_row,
    ]
    learned_winner_row = min(
        learned_family_best_rows,
        key=lambda row: float(row["val_final_state_rms_gap"]),
    )
    winner_family = str(learned_winner_row["family"])
    winner_schedule_name = str(learned_winner_row["rollout_schedule"])
    winner_knots = tuple(int(value) for value in str(learned_winner_row["schedule_knots"]).split("-") if value != "")

    if winner_family == "interval_ridge":
        winner_model = ridge_family_best_model
        winner_model_config = winner_model.to_jsonable()
        winner_best_epoch = None
        winner_training_diagnostics = None
    elif winner_family == "interval_ridge_aug":
        winner_model = ridge_aug_family_best_model
        winner_model_config = winner_model.to_jsonable()
        winner_best_epoch = None
        winner_training_diagnostics = None
    elif winner_family == "interval_ridge_residual":
        winner_model = ridge_residual_family_best_model
        winner_model_config = winner_model.to_jsonable()
        winner_best_epoch = None
        winner_training_diagnostics = None
    else:
        winner_model = _build_interval_mlp_candidate_model(
            interval_dataset=interval_dataset,
            normalization=normalization,
            config=mlp_family_best_record["config"],
            seed=int(mlp_family_best_record["seed"]),
        )
        winner_model.restore(mlp_family_best_record["snapshot"])
        winner_model_config = winner_model.to_jsonable()
        winner_best_epoch = int(mlp_family_best_record["best_epoch"])
        winner_training_diagnostics = dict(mlp_family_best_record["training_metrics"])

    _, winner_val_metrics = _evaluate_interval_rollout(
        winner_model,
        interval_dataset.val,
        endpoint_dataset.val,
        references["val"],
        teacher_model,
        rollout_schedule_name=winner_schedule_name,
        knots=winner_knots,
    )
    _, winner_test_metrics = _evaluate_interval_rollout(
        winner_model,
        interval_dataset.test,
        endpoint_dataset.test,
        references["test"],
        teacher_model,
        rollout_schedule_name=winner_schedule_name,
        knots=winner_knots,
    )

    for row in ridge_rows + ridge_aug_rows + ridge_residual_rows + mlp_rows:
        if (
            str(row["family"]) == winner_family
            and str(row["config_id"]) == str(learned_winner_row["config_id"])
            and str(row["rollout_schedule"]) == winner_schedule_name
        ):
            row["is_overall_winner"] = True
            row["evaluated_on_test"] = True
            row.update(_candidate_metric_columns("test", winner_test_metrics))

    if winner_family == "interval_ridge":
        ridge_summary["test"] = winner_test_metrics
    elif winner_family == "interval_ridge_aug":
        ridge_aug_summary["test"] = winner_test_metrics
    elif winner_family == "interval_ridge_residual":
        ridge_residual_summary["test"] = winner_test_metrics
    else:
        mlp_summary["test"] = winner_test_metrics

    candidate_rows = [
        identity_row,
        endpoint_ridge_row,
        *ridge_rows,
        *ridge_aug_rows,
        *ridge_residual_rows,
        *mlp_rows,
    ]
    summary = {
        "experiment_name": config.experiment_name,
        "run_id": run_id,
        "phase": "Phase 5",
        "stage": "phase5b_interval_conditioned_transporter",
        "dataset_name": config.dataset_name,
        "teacher_artifact_dir": _relative_artifact_reference(run_dir, config.teacher_preparation_path),
        "teacher_manifest_path": _relative_artifact_reference(run_dir, endpoint_dataset.teacher_manifest_path),
        "teacher_checkpoint_path": _relative_artifact_reference(run_dir, endpoint_dataset.teacher_checkpoint_path),
        "teacher_checkpoint_loaded": not used_teacher_retrain_fallback,
        "teacher_steps": int(interval_dataset.teacher_steps),
        "schedule_knots": {name: list(knots) for name, knots in rollout_schedules.items()},
        "rollout_aware_auxiliary_schedules": {name: list(knots) for name, knots in aux_rollout_schedules.items()},
        "selection_metric_source": "val_metric",
        "report_metric_source": "test_metric",
        "rollout_aware_rescue": {
            "active": True,
            "primary_interval_loss": "MSE(u_hat, u_star)",
            "auxiliary_family": "interval_mlp_standardized",
            "auxiliary_schedule_names": list(config.mlp_rollout_aux_schedule_names),
            "auxiliary_weight_candidates": [float(value) for value in config.mlp_rollout_aux_weight_candidates],
            "self_fed_between_knots": True,
            "teacher_supervised": True,
            "meanflow_or_jvp_used": False,
            "refinement_used": False,
        },
        "identity_baseline": identity_summary,
        "phase5a_endpoint_ridge_baseline": endpoint_ridge_summary,
        "interval_ridge": ridge_summary,
        "interval_ridge_aug": ridge_aug_summary,
        "interval_ridge_residual": ridge_residual_summary,
        "interval_mlp_standardized": mlp_summary,
        "gradient_augmented_rescue": {
            "active": True,
            "teacher_feature_names": list(config.gradient_augmented_feature_names),
            "supported_teacher_feature_names": ["y_hat_s", "e_out_s", "g_s", "F_s"],
            "knot_focused_schedule_names": list(config.augmented_knot_focused_schedule_names),
            "knot_focus_mixture_candidates": [float(value) for value in config.augmented_knot_focus_mixture_candidates],
            "residual_target_enabled": True,
            "current_state_only": True,
            "no_future_teacher_state_leakage": True,
            "meanflow_or_jvp_used": False,
            "refinement_used": False,
        },
        "winner": {
            "family": winner_family,
            "config_id": str(learned_winner_row["config_id"]),
            "rollout_schedule": winner_schedule_name,
            "schedule_knots": list(winner_knots),
            "model_config": winner_model_config,
            "best_epoch": winner_best_epoch,
            "training_diagnostics": winner_training_diagnostics,
            "val": winner_val_metrics,
            "test": winner_test_metrics,
            "winner_beats_identity_on_val_metric": bool(
                float(winner_val_metrics["final_state_rms_gap"])
                < float(identity_summary["val"]["final_state_rms_gap"])
            ),
            "winner_beats_identity_on_test_metric": bool(
                float(winner_test_metrics["final_state_rms_gap"])
                < float(identity_summary["test"]["final_state_rms_gap"])
            ),
            "winner_beats_phase5a_endpoint_ridge_on_val_metric": bool(
                float(winner_val_metrics["final_state_rms_gap"])
                < float(endpoint_ridge_summary["val"]["final_state_rms_gap"])
            ),
            "winner_beats_phase5a_endpoint_ridge_on_test_metric": bool(
                float(winner_test_metrics["final_state_rms_gap"])
                < float(endpoint_ridge_summary["test"]["final_state_rms_gap"])
            ),
            "winner_is_true_multistep_learned_interval_family": bool(
                winner_family in {
                    "interval_ridge",
                    "interval_ridge_aug",
                    "interval_ridge_residual",
                    "interval_mlp_standardized",
                }
                and winner_schedule_name in {"2-step", "3-step"}
            ),
        },
        "teacher_target_stats": {
            "train": {
                "delta_z_l2_mean": endpoint_dataset.train.metadata["delta_z_l2_mean"],
                "delta_z_rms": endpoint_dataset.train.metadata["delta_z_rms"],
            },
            "val": {
                "delta_z_l2_mean": endpoint_dataset.val.metadata["delta_z_l2_mean"],
                "delta_z_rms": endpoint_dataset.val.metadata["delta_z_rms"],
            },
            "test": {
                "delta_z_l2_mean": endpoint_dataset.test.metadata["delta_z_l2_mean"],
                "delta_z_rms": endpoint_dataset.test.metadata["delta_z_rms"],
            },
        },
    }

    config_payload = _suite_config_payload(
        config=config,
        run_id=run_id,
        run_dir=run_dir,
        endpoint_dataset=endpoint_dataset,
        interval_dataset=interval_dataset,
        normalization=normalization,
        teacher_checkpoint_loaded=not used_teacher_retrain_fallback,
        comparison_atol=comparison_atol,
        rollout_schedules=rollout_schedules,
        aux_rollout_schedules=aux_rollout_schedules,
    )
    _write_json(run_dir / "config.json", config_payload)
    _write_candidates_csv(run_dir / "candidates.csv", candidate_rows)
    _write_json(run_dir / "summary.json", summary)
    return FMPCIntervalSuiteRunResult(
        run_dir=run_dir,
        config=config_payload,
        candidates=candidate_rows,
        summary=summary,
    )
