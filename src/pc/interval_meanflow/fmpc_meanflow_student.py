from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .fmpc_interval_data import (
    FMPCIntervalBatch,
    FMPCIntervalDataset,
    FMPCIntervalSplit,
    acceptance_schedule_focus_pairs,
    iter_mixed_interval_batches,
    load_fmpc_interval_dataset,
    teacher_step_aligned_rollout_schedules,
)
from .fmpc_interval_features import (
    FMPCIntervalTeacherFeatureBundle,
    FMPCIntervalTeacherStateFeatures,
    FMPCIntervalTeacherStateFeatureTangents,
    prepare_interval_teacher_feature_context,
    precompute_interval_teacher_trajectory_feature_tangents,
    precompute_interval_teacher_trajectory_features,
)
from .fmpc_interval_normalization import (
    FMPCIntervalNormalizationStats,
    fit_fmpc_interval_augmented_normalization,
)
from .fmpc_interval_student import (
    IntervalAugmentedRidgeStudent,
    IntervalAugmentedRidgeStudentConfig,
    build_rollout_auxiliary_batches,
    rollout_interval_student,
)
from .fmpc_meanflow_jvp import build_meanflow_input_tangent, forward_mlp_with_jvp
from ..fmpc_v0.fmpc_student import (
    evaluate_fmpc_delta_predictions,
    evaluate_fmpc_identity_baseline,
    fmpc_split_evaluation_metrics_payload,
    load_fmpc_student_teacher_runtime,
    prepare_fmpc_student_teacher_references,
)
from ..fmpc_v0.fmpc_student_baselines import RidgeDeltaStudent, RidgeDeltaStudentConfig
from ..fmpc_v0.fmpc_student_data import FMPCStudentDataset, load_fmpc_student_dataset
from ..fmpc_v0.fmpc_student_normalization import fit_fmpc_student_normalization
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


def _serialize_hidden_dims(hidden_dims: tuple[int, ...]) -> str:
    if len(hidden_dims) == 0:
        return "linear"
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


def reconstruct_meanflow_velocity(
    g_s: np.ndarray,
    predicted_target_raw: np.ndarray,
    *,
    target_mode: Literal["u_star", "u_residual_local_field"],
) -> np.ndarray:
    """Reconstruct raw `u_hat` from the predicted target space."""

    g_array = np.asarray(g_s, dtype=np.float64)
    predicted_array = np.asarray(predicted_target_raw, dtype=np.float64)
    if g_array.shape != predicted_array.shape:
        raise ValueError("g_s and predicted_target_raw must share the same shape.")
    if target_mode == "u_star":
        return predicted_array.astype(np.float64, copy=False)
    if target_mode == "u_residual_local_field":
        return (g_array + predicted_array).astype(np.float64, copy=False)
    raise ValueError("target_mode must be 'u_star' or 'u_residual_local_field'.")


def build_meanflow_identity_target(
    normalization: FMPCIntervalNormalizationStats,
    *,
    g_s: np.ndarray,
    dg_s: np.ndarray | None,
    delta_tau: np.ndarray,
    predicted_target_normalized: np.ndarray,
    jvp_normalized: np.ndarray,
    target_mode: Literal["u_star", "u_residual_local_field"],
) -> np.ndarray:
    """Build the stop-gradient MeanFlow identity target in normalized target space.

    Residual contract:
    - direct family: `u_hat ≈ g_s + dt * d/dtau_s u_hat`
    - residual family: `u_hat = g_s + r_hat`
      so the residual-space identity target becomes
      `r_hat ≈ dt * (d g_s / d tau_s + d r_hat / d tau_s)`
    """

    g_array = np.asarray(g_s, dtype=np.float64)
    dg_array = None if dg_s is None else np.asarray(dg_s, dtype=np.float64)
    delta_tau_array = np.asarray(delta_tau, dtype=np.float64)
    predicted_array = np.asarray(predicted_target_normalized, dtype=np.float64)
    jvp_array = np.asarray(jvp_normalized, dtype=np.float64)
    if g_array.shape != predicted_array.shape or predicted_array.shape != jvp_array.shape:
        raise ValueError("g_s, predicted_target_normalized, and jvp_normalized must share the same shape.")
    if dg_array is not None and dg_array.shape != g_array.shape:
        raise ValueError("dg_s must share the same shape as g_s when provided.")
    if delta_tau_array.shape != (g_array.shape[0], 1):
        raise ValueError("delta_tau must be shaped (batch, 1).")
    raw_jvp = jvp_array * np.maximum(normalization.u_std, normalization.eps)
    if target_mode == "u_star":
        raw_identity_target = g_array + delta_tau_array * raw_jvp
    elif target_mode == "u_residual_local_field":
        if dg_array is None:
            raise ValueError("Residual MeanFlow identity requires dg_s.")
        raw_identity_target = delta_tau_array * (dg_array + raw_jvp)
    else:
        raise ValueError("target_mode must be 'u_star' or 'u_residual_local_field'.")
    return normalization.transform_u(raw_identity_target)


def build_meanflow_full_identity_target_raw(
    *,
    g_s: np.ndarray,
    delta_tau: np.ndarray,
    combined_jvp_raw: np.ndarray,
) -> np.ndarray:
    """Build the raw-space MeanFlow identity target for a full reconstructed velocity.

    Shape contract:
    - `g_s`: `(batch, z_dim)`
    - `delta_tau`: `(batch, 1)`
    - `combined_jvp_raw`: `(batch, z_dim)`

    Formula:
    - `u_id_tgt = g_s + dt * d/dtau_s u_hat`
    """

    g_array = np.asarray(g_s, dtype=np.float64)
    delta_tau_array = np.asarray(delta_tau, dtype=np.float64)
    jvp_array = np.asarray(combined_jvp_raw, dtype=np.float64)
    if g_array.shape != jvp_array.shape:
        raise ValueError("g_s and combined_jvp_raw must share the same shape.")
    if delta_tau_array.shape != (g_array.shape[0], 1):
        raise ValueError("delta_tau must be shaped (batch, 1).")
    return (g_array + delta_tau_array * jvp_array).astype(np.float64, copy=False)


def _training_loss_weights(
    *,
    epoch: int,
    teacher_loss_weight: float,
    identity_loss_weight: float,
    teacher_warmup_epochs: int,
    identity_ramp_epochs: int,
) -> tuple[float, float]:
    if epoch <= 0:
        raise ValueError("epoch must be positive.")
    teacher_weight = float(teacher_loss_weight)
    final_identity_weight = float(identity_loss_weight)
    if final_identity_weight <= 0.0:
        return teacher_weight, 0.0
    warmup = int(max(0, teacher_warmup_epochs))
    ramp = int(max(0, identity_ramp_epochs))
    if epoch <= warmup:
        return teacher_weight, 0.0
    if ramp == 0:
        return teacher_weight, final_identity_weight
    progress = min(1.0, float(epoch - warmup) / float(ramp))
    return teacher_weight, final_identity_weight * progress


def meanflow_identity_active_mask(
    source_step_indices: np.ndarray,
    target_step_indices: np.ndarray,
    *,
    teacher_steps: int,
    identity_scope_mode: Literal["all_intervals", "acceptance_schedule_segments_only"],
    knot_focused_schedule_names: tuple[str, ...] = ("2-step", "3-step"),
) -> np.ndarray:
    """Return the per-example boolean mask for applying the MeanFlow identity loss.

    Shape contract:
    - `source_step_indices`: `(batch,)`
    - `target_step_indices`: `(batch,)`
    - returns: `(batch,)`
    """

    source_array = np.asarray(source_step_indices, dtype=np.int64)
    target_array = np.asarray(target_step_indices, dtype=np.int64)
    if source_array.shape != target_array.shape or source_array.ndim != 1:
        raise ValueError("source_step_indices and target_step_indices must be rank-1 arrays with matching shape.")
    if identity_scope_mode == "all_intervals":
        return np.ones(source_array.shape, dtype=bool)
    if identity_scope_mode != "acceptance_schedule_segments_only":
        raise ValueError(
            "identity_scope_mode must be 'all_intervals' or 'acceptance_schedule_segments_only'."
        )
    focus_pairs = acceptance_schedule_focus_pairs(
        teacher_steps,
        schedule_names=knot_focused_schedule_names,
    )
    active = np.zeros(source_array.shape, dtype=bool)
    for source_index, target_index in focus_pairs:
        active |= (source_array == int(source_index)) & (target_array == int(target_index))
    return active


@dataclass(frozen=True)
class MeanFlowMLPStudentConfig:
    hidden_dims: tuple[int, ...]
    family_name: Literal[
        "teacher_only_mlp_aug",
        "meanflow_mlp_aug",
        "meanflow_mlp_residual",
        "meanflow_linear_residual",
        "meanflow_twobranch_residual",
    ]
    target_mode: Literal["u_star", "u_residual_local_field"]
    selected_teacher_feature_names: tuple[str, ...] = ("g_s", "e_out_s", "F_s")
    hidden_activation: str = "tanh"
    output_activation: str = "identity"
    weight_scale: float = 0.02
    eta_w: float = 0.01
    eta_b: float | None = None
    epochs: int = 40
    batch_size: int = 64
    batches_per_epoch: int | None = None
    teacher_loss_weight: float = 1.0
    identity_loss_weight: float = 0.0
    rollout_aux_weight: float = 0.0
    teacher_warmup_epochs: int = 12
    identity_ramp_epochs: int = 20
    identity_scope_mode: Literal["all_intervals", "acceptance_schedule_segments_only"] = "all_intervals"
    knot_focus_probability: float = 0.0
    knot_focused_schedule_names: tuple[str, ...] = ("2-step", "3-step")
    feature_aware_teacher_tangents: bool = True
    feature_tangent_epsilon: float = 1e-3

    def __post_init__(self) -> None:
        if self.family_name == "teacher_only_mlp_aug" and self.identity_loss_weight != 0.0:
            raise ValueError("teacher_only_mlp_aug must use identity_loss_weight = 0.")
        if self.family_name != "teacher_only_mlp_aug" and self.identity_loss_weight < 0.0:
            raise ValueError("identity_loss_weight must be non-negative.")
        if self.family_name == "meanflow_linear_residual":
            if self.hidden_dims != ():
                raise ValueError("meanflow_linear_residual must use hidden_dims=().")
            if self.target_mode != "u_residual_local_field":
                raise ValueError("meanflow_linear_residual must use target_mode='u_residual_local_field'.")
        if self.family_name == "meanflow_twobranch_residual":
            raise ValueError("meanflow_twobranch_residual requires MeanFlowTwoBranchStudentConfig.")
        if self.teacher_loss_weight <= 0.0:
            raise ValueError("teacher_loss_weight must be positive.")
        if self.rollout_aux_weight < 0.0:
            raise ValueError("rollout_aux_weight must be non-negative.")
        if not (0.0 <= self.knot_focus_probability <= 1.0):
            raise ValueError("knot_focus_probability must lie in [0, 1].")
        if self.feature_tangent_epsilon <= 0.0:
            raise ValueError("feature_tangent_epsilon must be positive.")
        if self.identity_scope_mode not in {"all_intervals", "acceptance_schedule_segments_only"}:
            raise ValueError(
                "identity_scope_mode must be 'all_intervals' or 'acceptance_schedule_segments_only'."
            )


@dataclass(frozen=True)
class FMPCMeanFlowSuiteConfig:
    teacher_preparation_path: str | Path
    output_root: str | Path = "outputs"
    experiment_name: str = "fmpc_meanflow_suite"
    run_id: str | None = None
    output_layout: OutputLayout = "single_dir"
    dataset_name: str = "digits"
    run_seed: int = 0
    model_init_seed: int = 0
    batch_order_seed: int = 0
    normalization_eps: float = 1e-8
    allow_teacher_retrain: bool = False
    rollout_schedule_names: tuple[str, ...] = ("1-step", "2-step", "3-step")
    feature_names: tuple[str, ...] = ("g_s", "e_out_s", "F_s")
    feature_tangent_epsilon: float = 1e-3
    knot_focused_schedule_names: tuple[str, ...] = ("2-step", "3-step")
    knot_focus_probability_candidates: tuple[float, ...] = (0.0,)
    hidden_dims_candidates: tuple[tuple[int, ...], ...] = ((64,), (128,))
    epochs_candidates: tuple[int, ...] = (40,)
    eta_w_candidates: tuple[float, ...] = (0.01, 0.05)
    teacher_loss_weight: float = 1.0
    identity_loss_weight_candidates: tuple[float, ...] = (0.02, 0.05, 0.1)
    identity_scope_modes: tuple[Literal["all_intervals", "acceptance_schedule_segments_only"], ...] = (
        "all_intervals",
        "acceptance_schedule_segments_only",
    )
    rollout_aux_weight_candidates: tuple[float, ...] = (0.0,)
    teacher_warmup_epochs: int = 12
    identity_ramp_epochs: int = 20
    batch_size: int = 64
    batches_per_epoch: int | None = None
    weight_scale: float = 0.02
    hidden_activation: str = "tanh"
    output_activation: str = "identity"
    carried_forward_endpoint_ridge_alpha: float = 1e-4
    carried_forward_phase5b2_alpha: float = 0.01
    carried_forward_phase5b2_schedule_name: str = "3-step"
    carried_forward_phase5b2_knot_focus_mixture: float = 0.0
    carried_forward_phase6a1_linear_eta_w: float = 0.05
    carried_forward_phase6a1_linear_identity_loss_weight: float = 0.1
    carried_forward_phase6a1_linear_schedule_name: str = "3-step"
    carried_forward_phase6a1_linear_identity_scope_mode: Literal[
        "all_intervals",
        "acceptance_schedule_segments_only",
    ] = "all_intervals"
    carried_forward_phase6a2_twobranch_hidden_dims: tuple[int, ...] = (128,)
    carried_forward_phase6a2_twobranch_eta_w: float = 0.05
    carried_forward_phase6a2_twobranch_identity_loss_weight: float = 0.1
    carried_forward_phase6a2_twobranch_schedule_name: str = "3-step"
    carried_forward_phase6a2_twobranch_identity_scope_mode: Literal[
        "all_intervals",
        "acceptance_schedule_segments_only",
    ] = "all_intervals"
    twobranch_warmstart_correction_only_warmup_epochs: int = 10
    rollout_aux_schedule_names: tuple[str, ...] = ("2-step", "3-step")

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return str(self.run_id)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_{self.run_seed}"


@dataclass(frozen=True)
class FMPCMeanFlowSuiteRunResult:
    run_dir: Path
    config: dict[str, Any]
    candidates: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass(frozen=True)
class MeanFlowTwoBranchStudentConfig:
    correction_hidden_dims: tuple[int, ...]
    family_name: Literal[
        "meanflow_twobranch_residual",
        "meanflow_twobranch_residual_warmstart",
    ] = "meanflow_twobranch_residual"
    selected_teacher_feature_names: tuple[str, ...] = ("g_s", "e_out_s", "F_s")
    local_branch_feature_names: tuple[str, ...] = ("g_s", "e_out_s", "F_s")
    hidden_activation: str = "tanh"
    output_activation: str = "identity"
    weight_scale: float = 0.02
    eta_w: float = 0.01
    eta_b: float | None = None
    epochs: int = 40
    batch_size: int = 64
    batches_per_epoch: int | None = None
    teacher_loss_weight: float = 1.0
    identity_loss_weight: float = 0.0
    rollout_aux_weight: float = 0.0
    teacher_warmup_epochs: int = 12
    identity_ramp_epochs: int = 20
    identity_scope_mode: Literal["all_intervals", "acceptance_schedule_segments_only"] = "all_intervals"
    knot_focus_probability: float = 0.0
    knot_focused_schedule_names: tuple[str, ...] = ("2-step", "3-step")
    feature_aware_teacher_tangents: bool = True
    feature_tangent_epsilon: float = 1e-3
    zero_init_correction_output_head: bool = True
    correction_only_warmup_epochs: int = 0
    local_branch_warm_start: bool = False
    warm_start_source_family: str = "phase6a1_linear_residual_baseline"

    def __post_init__(self) -> None:
        if self.identity_loss_weight <= 0.0:
            raise ValueError("meanflow_twobranch_residual must use a nonzero identity_loss_weight.")
        if self.teacher_loss_weight <= 0.0:
            raise ValueError("teacher_loss_weight must be positive.")
        if self.rollout_aux_weight < 0.0:
            raise ValueError("rollout_aux_weight must be non-negative.")
        if not (0.0 <= self.knot_focus_probability <= 1.0):
            raise ValueError("knot_focus_probability must lie in [0, 1].")
        if self.feature_tangent_epsilon <= 0.0:
            raise ValueError("feature_tangent_epsilon must be positive.")
        if self.identity_scope_mode not in {"all_intervals", "acceptance_schedule_segments_only"}:
            raise ValueError(
                "identity_scope_mode must be 'all_intervals' or 'acceptance_schedule_segments_only'."
            )
        if len(self.local_branch_feature_names) == 0:
            raise ValueError("local_branch_feature_names must not be empty.")
        if "g_s" not in self.local_branch_feature_names:
            raise ValueError("local_branch_feature_names must include g_s.")
        if self.correction_only_warmup_epochs < 0:
            raise ValueError("correction_only_warmup_epochs must be non-negative.")
        if self.family_name == "meanflow_twobranch_residual_warmstart":
            if not self.local_branch_warm_start:
                raise ValueError("warm-start family requires local_branch_warm_start=True.")
            if self.correction_only_warmup_epochs <= 0:
                raise ValueError("warm-start family requires correction_only_warmup_epochs > 0.")


@dataclass
class MeanFlowMLPStudent:
    """MeanFlow-style NumPy MLP student on augmented interval inputs."""

    config: MeanFlowMLPStudentConfig
    normalization: FMPCIntervalNormalizationStats
    network: MLPNetwork
    z_dim: int
    target_dim: int
    teacher_model: Any
    teacher_feature_bundle: FMPCIntervalTeacherFeatureBundle

    @classmethod
    def initialize(
        cls,
        *,
        z_dim: int,
        target_dim: int,
        normalization: FMPCIntervalNormalizationStats,
        config: MeanFlowMLPStudentConfig,
        seed: int,
        teacher_model: Any,
        teacher_feature_bundle: FMPCIntervalTeacherFeatureBundle,
    ) -> "MeanFlowMLPStudent":
        input_dim = int(z_dim + target_dim + 2 + normalization.teacher_feature_mean.shape[0])
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
            teacher_model=teacher_model,
            teacher_feature_bundle=teacher_feature_bundle,
        )

    def snapshot(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(layer.weight.copy(), layer.bias.copy()) for layer in self.network.layers]

    def restore(self, snapshot: list[tuple[np.ndarray, np.ndarray]]) -> None:
        if len(snapshot) != len(self.network.layers):
            raise ValueError("Parameter snapshot must align with MeanFlow student layers.")
        for layer, (weight, bias) in zip(self.network.layers, snapshot, strict=True):
            layer.weight = weight.copy()
            layer.bias = bias.copy()

    def _trajectory_batch_features(
        self,
        split_name: str,
        batch: FMPCIntervalBatch,
    ) -> tuple[np.ndarray, np.ndarray, FMPCIntervalTeacherStateFeatures, FMPCIntervalTeacherStateFeatureTangents]:
        trajectory_features = self.teacher_feature_bundle.trajectory_features(split_name)
        trajectory_tangents = self.teacher_feature_bundle.trajectory_feature_tangents(split_name)
        state_features = trajectory_features.gather_batch_features(
            batch.sample_row_indices,
            batch.source_step_indices,
            selected_feature_names=self.config.selected_teacher_feature_names,
        )
        state_tangents = trajectory_tangents.gather_batch_feature_tangents(
            batch.sample_row_indices,
            batch.source_step_indices,
        )
        teacher_feature_matrix = state_features.feature_matrix(self.config.selected_teacher_feature_names)
        teacher_feature_tangent_matrix = state_tangents.feature_tangent_matrix(
            self.config.selected_teacher_feature_names
        )
        return teacher_feature_matrix, teacher_feature_tangent_matrix, state_features, state_tangents

    def _current_state_rollout_features(
        self,
        split_name: str,
        z_s: np.ndarray,
        target_onehot: np.ndarray,
        tau_s: np.ndarray | float,
        tau_t: np.ndarray | float,
    ) -> tuple[np.ndarray, FMPCIntervalTeacherStateFeatures]:
        return self.teacher_feature_bundle.current_state_feature_matrix(
            self.teacher_model,
            split_name=split_name,
            z_s=np.asarray(z_s, dtype=np.float64),
            target_onehot=np.asarray(target_onehot, dtype=np.float64),
            tau_s=tau_s,
            tau_t=tau_t,
            selected_feature_names=self.config.selected_teacher_feature_names,
        )

    def _normalized_forward(
        self,
        *,
        z_s: np.ndarray,
        target_onehot: np.ndarray,
        tau_s: np.ndarray | float,
        tau_t: np.ndarray | float,
        teacher_feature_matrix: np.ndarray,
        teacher_feature_tangent_matrix: np.ndarray,
        g_s: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        inputs = self.normalization.transform_inputs(
            z_s,
            target_onehot,
            tau_s,
            tau_t,
            teacher_features=teacher_feature_matrix,
        )
        input_tangent = build_meanflow_input_tangent(
            self.normalization,
            g_s,
            target_dim=self.target_dim,
            teacher_feature_dim=teacher_feature_matrix.shape[1],
            teacher_feature_tangent=(
                teacher_feature_tangent_matrix if self.config.feature_aware_teacher_tangents else None
            ),
            d_tau_s=1.0,
            d_tau_t=0.0,
        )
        jvp_result = forward_mlp_with_jvp(self.network, inputs, input_tangent)
        return inputs, jvp_result.output, jvp_result.jvp

    def predict_u_for_rollout(
        self,
        split_name: str,
        z_s: np.ndarray,
        target_onehot: np.ndarray,
        tau_s: np.ndarray | float,
        tau_t: np.ndarray | float,
    ) -> np.ndarray:
        teacher_feature_matrix, state_features = self._current_state_rollout_features(
            split_name,
            z_s,
            target_onehot,
            tau_s,
            tau_t,
        )
        inputs = self.normalization.transform_inputs(
            z_s,
            target_onehot,
            tau_s,
            tau_t,
            teacher_features=teacher_feature_matrix,
        )
        predicted_target_normalized = self.network.predict(inputs)
        predicted_target_raw = self.normalization.inverse_u(predicted_target_normalized)
        return reconstruct_meanflow_velocity(
            state_features.g_s,
            predicted_target_raw,
            target_mode=self.config.target_mode,
        )

    def _train_on_normalized_target(
        self,
        inputs: np.ndarray,
        target_normalized: np.ndarray,
        *,
        step_scale: float,
    ) -> float:
        if step_scale < 0.0:
            raise ValueError("step_scale must be non-negative.")
        if step_scale == 0.0:
            predictions = self.network.predict(inputs)
            return float(np.mean((predictions - target_normalized) ** 2))
        original_eta_w = float(self.network.eta_w)
        original_eta_b = float(self.network.eta_b)
        self.network.eta_w = original_eta_w * step_scale
        self.network.eta_b = original_eta_b * step_scale
        try:
            result = self.network.train_batch(inputs, target_normalized)
        finally:
            self.network.eta_w = original_eta_w
            self.network.eta_b = original_eta_b
        return float(result.loss)

    def _identity_active_mask(self, batch: FMPCIntervalBatch) -> np.ndarray:
        return meanflow_identity_active_mask(
            batch.source_step_indices,
            batch.target_step_indices,
            teacher_steps=self.teacher_feature_bundle.teacher_steps,
            identity_scope_mode=self.config.identity_scope_mode,
            knot_focused_schedule_names=self.config.knot_focused_schedule_names,
        )

    def train_batch(
        self,
        batch: FMPCIntervalBatch,
        *,
        split_name: str,
        teacher_loss_weight: float,
        identity_loss_weight: float,
        step_scale: float = 1.0,
        train_local_branch: bool = True,
        train_correction_branch: bool = True,
    ) -> dict[str, float]:
        (
            teacher_feature_matrix,
            teacher_feature_tangent_matrix,
            state_features,
            state_tangents,
        ) = self._trajectory_batch_features(split_name, batch)
        inputs, predicted_target_normalized, jvp_normalized = self._normalized_forward(
            z_s=batch.z_s,
            target_onehot=batch.target_onehot,
            tau_s=batch.tau_s,
            tau_t=batch.tau_t,
            teacher_feature_matrix=teacher_feature_matrix,
            teacher_feature_tangent_matrix=teacher_feature_tangent_matrix,
            g_s=state_features.g_s,
        )
        if self.config.target_mode == "u_star":
            teacher_target_raw = np.asarray(batch.u_star, dtype=np.float64)
        else:
            teacher_target_raw = np.asarray(batch.u_star - state_features.g_s, dtype=np.float64)
        teacher_target_normalized = self.normalization.transform_u(teacher_target_raw)
        identity_target_normalized = build_meanflow_identity_target(
            self.normalization,
            g_s=state_features.g_s,
            dg_s=state_tangents.Dg_g_s,
            delta_tau=batch.delta_tau,
            predicted_target_normalized=predicted_target_normalized,
            jvp_normalized=jvp_normalized,
            target_mode=self.config.target_mode,
        )
        teacher_loss = float(np.mean((predicted_target_normalized - teacher_target_normalized) ** 2))
        identity_active_mask = self._identity_active_mask(batch)
        identity_active_fraction = float(np.mean(identity_active_mask.astype(np.float64)))
        if np.any(identity_active_mask):
            identity_loss = float(
                np.mean(
                    (
                        predicted_target_normalized[identity_active_mask]
                        - identity_target_normalized[identity_active_mask]
                    )
                    ** 2
                )
            )
        else:
            identity_loss = 0.0

        teacher_update_loss = self._train_on_normalized_target(
            inputs,
            teacher_target_normalized,
            step_scale=float(teacher_loss_weight) * float(step_scale),
        )

        identity_update_loss = None
        if float(identity_loss_weight) > 0.0 and np.any(identity_active_mask):
            identity_inputs = inputs[identity_active_mask]
            identity_targets = identity_target_normalized[identity_active_mask]
            identity_update_loss = self._train_on_normalized_target(
                identity_inputs,
                identity_targets,
                step_scale=float(identity_loss_weight) * float(step_scale),
            )

        total_weight = float(teacher_loss_weight) + float(identity_loss_weight) * identity_active_fraction
        if total_weight > 0.0:
            total_loss = (
                float(teacher_loss_weight) * teacher_loss
                + float(identity_loss_weight) * identity_active_fraction * identity_loss
            ) / total_weight
        else:
            total_loss = teacher_loss
        return {
            "teacher_loss": teacher_loss,
            "identity_loss": identity_loss,
            "total_loss": float(total_loss),
            "teacher_update_loss": float(teacher_update_loss),
            "identity_update_loss": None if identity_update_loss is None else float(identity_update_loss),
            "identity_active_fraction": identity_active_fraction,
        }

    def train_aux_batch(
        self,
        batch: FMPCIntervalBatch,
        *,
        split_name: str,
        aux_weight: float,
    ) -> float:
        teacher_feature_matrix, state_features = self._current_state_rollout_features(
            split_name,
            batch.z_s,
            batch.target_onehot,
            batch.tau_s,
            batch.tau_t,
        )
        inputs = self.normalization.transform_inputs(
            batch.z_s,
            batch.target_onehot,
            batch.tau_s,
            batch.tau_t,
            teacher_features=teacher_feature_matrix,
        )
        if self.config.target_mode == "u_star":
            teacher_target_raw = np.asarray(batch.u_star, dtype=np.float64)
        else:
            teacher_target_raw = np.asarray(batch.u_star - state_features.g_s, dtype=np.float64)
        teacher_target_normalized = self.normalization.transform_u(teacher_target_raw)
        return self._train_on_normalized_target(
            inputs,
            teacher_target_normalized,
            step_scale=float(aux_weight),
        )

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "family": self.config.family_name,
            "model_kind": "linear" if self.config.family_name == "meanflow_linear_residual" else "mlp",
            "target_mode": self.config.target_mode,
            "hidden_dims": list(self.config.hidden_dims),
            "hidden_activation": self.config.hidden_activation,
            "output_activation": self.config.output_activation,
            "weight_scale": float(self.config.weight_scale),
            "eta_w": float(self.config.eta_w),
            "eta_b": float(self.config.eta_b if self.config.eta_b is not None else self.config.eta_w),
            "epochs": int(self.config.epochs),
            "batch_size": int(self.config.batch_size),
            "batches_per_epoch": self.config.batches_per_epoch,
            "teacher_loss_weight": float(self.config.teacher_loss_weight),
            "identity_loss_weight": float(self.config.identity_loss_weight),
            "rollout_aux_weight": float(self.config.rollout_aux_weight),
            "teacher_warmup_epochs": int(self.config.teacher_warmup_epochs),
            "identity_ramp_epochs": int(self.config.identity_ramp_epochs),
            "identity_scope_mode": self.config.identity_scope_mode,
            "knot_focus_probability": float(self.config.knot_focus_probability),
            "knot_focused_schedule_names": list(self.config.knot_focused_schedule_names),
            "selected_teacher_feature_names": list(self.config.selected_teacher_feature_names),
            "jvp_tangent_contract": {
                "dz": "g_s",
                "d_tau_s": 1.0,
                "d_tau_t": 0.0,
                "target_onehot_tangent": 0.0,
                "teacher_feature_tangent": "D_g teacher_features(z_s)"
                if self.config.feature_aware_teacher_tangents
                else 0.0,
                "feature_aware_teacher_tangents": bool(self.config.feature_aware_teacher_tangents),
                "feature_tangent_epsilon": float(self.config.feature_tangent_epsilon),
                "residual_identity_includes_dg_s": bool(
                    self.config.target_mode == "u_residual_local_field"
                ),
            },
            "normalization": self.normalization.to_jsonable(),
            "student_input_definition": (
                "concat([z_s, target_onehot, tau_s, tau_t, teacher_state_features])"
            ),
            "teacher_feature_contract": {
                "selected_teacher_feature_names": list(self.config.selected_teacher_feature_names),
                "target_onehot_is_frozen_for_jvp": True,
                "teacher_feature_block_is_feature_aware": bool(self.config.feature_aware_teacher_tangents),
            },
            "hybrid_curriculum": {
                "teacher_only_warmup_epochs": int(self.config.teacher_warmup_epochs),
                "identity_ramp_epochs": int(self.config.identity_ramp_epochs),
                "late_fixed_identity_weight": float(self.config.identity_loss_weight),
                "identity_scope_mode": self.config.identity_scope_mode,
            },
            "direct_target_definition": (
                "u_star = (z_t - z_s) / (tau_t - tau_s)"
                if self.config.target_mode == "u_star"
                else "r_star = u_star - g_s"
            ),
            "prediction_reconstruction": (
                "u_hat = u_hat_direct"
                if self.config.target_mode == "u_star"
                else "u_hat = g_s + r_hat"
            ),
            "meanflow_identity_enabled": bool(self.config.identity_loss_weight > 0.0),
        }


def _teacher_feature_name_dim(
    feature_name: str,
    *,
    z_dim: int,
    target_dim: int,
) -> int:
    if feature_name == "g_s":
        return int(z_dim)
    if feature_name in {"y_hat_s", "e_out_s"}:
        return int(target_dim)
    if feature_name == "F_s":
        return 1
    raise ValueError(f"Unsupported teacher feature name '{feature_name}'.")


def _teacher_feature_block_dim(
    feature_names: tuple[str, ...],
    *,
    z_dim: int,
    target_dim: int,
) -> int:
    return int(
        sum(
            _teacher_feature_name_dim(feature_name, z_dim=z_dim, target_dim=target_dim)
            for feature_name in feature_names
        )
    )


def _teacher_feature_slices(
    feature_names: tuple[str, ...],
    *,
    z_dim: int,
    target_dim: int,
) -> dict[str, slice]:
    slices: dict[str, slice] = {}
    offset = 0
    for feature_name in feature_names:
        feature_dim = _teacher_feature_name_dim(feature_name, z_dim=z_dim, target_dim=target_dim)
        slices[feature_name] = slice(offset, offset + feature_dim)
        offset += feature_dim
    return slices


def _extract_local_branch_warm_start_from_linear_model(
    *,
    model: "MeanFlowMLPStudent",
    normalization: FMPCIntervalNormalizationStats,
    z_dim: int,
    target_dim: int,
    selected_teacher_feature_names: tuple[str, ...],
    local_branch_feature_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Project the carried-forward linear MeanFlow solution onto the local branch.

    The warm-start uses only the teacher-feature block of the monolithic linear winner.
    Inputs are converted back to the raw local-branch feature space:
    `feat_norm = (feat_raw - mean) / std`.
    """

    if len(model.network.layers) != 1:
        raise ValueError("Local warm-start extraction requires a single-layer linear MeanFlow model.")
    layer = model.network.layers[0]
    feature_slices = _teacher_feature_slices(
        selected_teacher_feature_names,
        z_dim=z_dim,
        target_dim=target_dim,
    )
    feature_block_offset = int(z_dim + target_dim + 2)
    local_weight_blocks: list[np.ndarray] = []
    local_bias = layer.bias.copy().astype(np.float64, copy=False)
    safe_teacher_feature_std = np.maximum(normalization.teacher_feature_std, normalization.eps)
    for feature_name in local_branch_feature_names:
        feature_slice = feature_slices[feature_name]
        feature_columns = slice(
            feature_block_offset + int(feature_slice.start),
            feature_block_offset + int(feature_slice.stop),
        )
        weight_block_normalized = np.asarray(layer.weight[:, feature_columns], dtype=np.float64)
        feature_mean = np.asarray(normalization.teacher_feature_mean[feature_slice], dtype=np.float64)
        feature_std = np.asarray(safe_teacher_feature_std[feature_slice], dtype=np.float64)
        local_weight_blocks.append(weight_block_normalized / feature_std[np.newaxis, :])
        local_bias = local_bias - np.matmul(feature_mean / feature_std, weight_block_normalized.T)
    local_weight = np.concatenate(local_weight_blocks, axis=1).astype(np.float64, copy=False)
    return local_weight, local_bias.astype(np.float64, copy=False)


@dataclass
class MeanFlowTwoBranchResidualStudent:
    """Two-branch feature-aware MeanFlow residual student.

    Decomposition:
    - `u_hat = u_local + u_corr`
    - local branch is a simple linear current-dynamics branch on `[g_s, e_out_s, F_s]`
    - correction branch is a neural residual branch on the full augmented Phase 6A.1 input
    """

    config: MeanFlowTwoBranchStudentConfig
    normalization: FMPCIntervalNormalizationStats
    local_network: MLPNetwork
    correction_network: MLPNetwork
    z_dim: int
    target_dim: int
    teacher_model: Any
    teacher_feature_bundle: FMPCIntervalTeacherFeatureBundle

    @classmethod
    def initialize(
        cls,
        *,
        z_dim: int,
        target_dim: int,
        normalization: FMPCIntervalNormalizationStats,
        config: MeanFlowTwoBranchStudentConfig,
        seed: int,
        teacher_model: Any,
        teacher_feature_bundle: FMPCIntervalTeacherFeatureBundle,
        local_branch_warm_start: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "MeanFlowTwoBranchResidualStudent":
        correction_input_dim = int(z_dim + target_dim + 2 + normalization.teacher_feature_mean.shape[0])
        correction_network = MLPNetwork(
            layers=init_mlp_baseline_layers(
                (correction_input_dim, *config.correction_hidden_dims, int(z_dim)),
                hidden_activation=config.hidden_activation,
                output_activation=config.output_activation,
                weight_scale=config.weight_scale,
                seed=seed,
                dtype=np.float64,
            ),
            eta_w=config.eta_w,
            eta_b=config.eta_b,
        )
        if config.zero_init_correction_output_head:
            correction_network.layers[-1].weight.fill(0.0)
            correction_network.layers[-1].bias.fill(0.0)

        local_input_dim = _teacher_feature_block_dim(
            config.local_branch_feature_names,
            z_dim=z_dim,
            target_dim=target_dim,
        )
        local_network = MLPNetwork(
            layers=init_mlp_baseline_layers(
                (local_input_dim, int(z_dim)),
                hidden_activation="identity",
                output_activation="identity",
                weight_scale=config.weight_scale,
                seed=seed + 7919,
                dtype=np.float64,
            ),
            eta_w=config.eta_w,
            eta_b=config.eta_b,
        )
        local_layer = local_network.layers[0]
        local_layer.weight.fill(0.0)
        local_layer.bias.fill(0.0)
        offset = 0
        for feature_name in config.local_branch_feature_names:
            feature_dim = _teacher_feature_name_dim(feature_name, z_dim=z_dim, target_dim=target_dim)
            if feature_name == "g_s":
                local_layer.weight[:, offset : offset + int(z_dim)] = np.eye(int(z_dim), dtype=np.float64)
                break
            offset += feature_dim
        if local_branch_warm_start is not None:
            warm_weight, warm_bias = local_branch_warm_start
            warm_weight_array = np.asarray(warm_weight, dtype=np.float64)
            warm_bias_array = np.asarray(warm_bias, dtype=np.float64)
            if warm_weight_array.shape != local_layer.weight.shape:
                raise ValueError("Local branch warm-start weight shape must match the local branch layer.")
            if warm_bias_array.shape != local_layer.bias.shape:
                raise ValueError("Local branch warm-start bias shape must match the local branch layer.")
            local_layer.weight = warm_weight_array.copy()
            local_layer.bias = warm_bias_array.copy()

        return cls(
            config=config,
            normalization=normalization,
            local_network=local_network,
            correction_network=correction_network,
            z_dim=int(z_dim),
            target_dim=int(target_dim),
            teacher_model=teacher_model,
            teacher_feature_bundle=teacher_feature_bundle,
        )

    def snapshot(self) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
        return {
            "local": [(layer.weight.copy(), layer.bias.copy()) for layer in self.local_network.layers],
            "correction": [
                (layer.weight.copy(), layer.bias.copy()) for layer in self.correction_network.layers
            ],
        }

    def restore(self, snapshot: dict[str, list[tuple[np.ndarray, np.ndarray]]]) -> None:
        local_snapshot = snapshot["local"]
        correction_snapshot = snapshot["correction"]
        if len(local_snapshot) != len(self.local_network.layers):
            raise ValueError("Local snapshot must align with local branch layers.")
        if len(correction_snapshot) != len(self.correction_network.layers):
            raise ValueError("Correction snapshot must align with correction branch layers.")
        for layer, (weight, bias) in zip(self.local_network.layers, local_snapshot, strict=True):
            layer.weight = weight.copy()
            layer.bias = bias.copy()
        for layer, (weight, bias) in zip(self.correction_network.layers, correction_snapshot, strict=True):
            layer.weight = weight.copy()
            layer.bias = bias.copy()

    def _trajectory_batch_features(
        self,
        split_name: str,
        batch: FMPCIntervalBatch,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        FMPCIntervalTeacherStateFeatures,
        FMPCIntervalTeacherStateFeatureTangents,
    ]:
        trajectory_features = self.teacher_feature_bundle.trajectory_features(split_name)
        trajectory_tangents = self.teacher_feature_bundle.trajectory_feature_tangents(split_name)
        state_features = trajectory_features.gather_batch_features(
            batch.sample_row_indices,
            batch.source_step_indices,
            selected_feature_names=self.config.selected_teacher_feature_names,
        )
        state_tangents = trajectory_tangents.gather_batch_feature_tangents(
            batch.sample_row_indices,
            batch.source_step_indices,
        )
        correction_feature_matrix = state_features.feature_matrix(self.config.selected_teacher_feature_names)
        correction_feature_tangent_matrix = state_tangents.feature_tangent_matrix(
            self.config.selected_teacher_feature_names
        )
        local_feature_matrix = state_features.feature_matrix(self.config.local_branch_feature_names)
        local_feature_tangent_matrix = state_tangents.feature_tangent_matrix(
            self.config.local_branch_feature_names
        )
        return (
            correction_feature_matrix,
            correction_feature_tangent_matrix,
            local_feature_matrix,
            local_feature_tangent_matrix,
            state_features,
            state_tangents,
        )

    def _current_state_rollout_features(
        self,
        split_name: str,
        z_s: np.ndarray,
        target_onehot: np.ndarray,
        tau_s: np.ndarray | float,
        tau_t: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray, FMPCIntervalTeacherStateFeatures]:
        correction_feature_matrix, state_features = self.teacher_feature_bundle.current_state_feature_matrix(
            self.teacher_model,
            split_name=split_name,
            z_s=np.asarray(z_s, dtype=np.float64),
            target_onehot=np.asarray(target_onehot, dtype=np.float64),
            tau_s=tau_s,
            tau_t=tau_t,
            selected_feature_names=self.config.selected_teacher_feature_names,
        )
        local_feature_matrix = state_features.feature_matrix(self.config.local_branch_feature_names)
        return correction_feature_matrix, local_feature_matrix, state_features

    def _forward_branches(
        self,
        *,
        z_s: np.ndarray,
        target_onehot: np.ndarray,
        tau_s: np.ndarray | float,
        tau_t: np.ndarray | float,
        correction_feature_matrix: np.ndarray,
        correction_feature_tangent_matrix: np.ndarray,
        local_feature_matrix: np.ndarray,
        local_feature_tangent_matrix: np.ndarray,
        g_s: np.ndarray,
    ) -> dict[str, np.ndarray]:
        correction_inputs = self.normalization.transform_inputs(
            z_s,
            target_onehot,
            tau_s,
            tau_t,
            teacher_features=correction_feature_matrix,
        )
        correction_input_tangent = build_meanflow_input_tangent(
            self.normalization,
            g_s,
            target_dim=self.target_dim,
            teacher_feature_dim=correction_feature_matrix.shape[1],
            teacher_feature_tangent=(
                correction_feature_tangent_matrix if self.config.feature_aware_teacher_tangents else None
            ),
            d_tau_s=1.0,
            d_tau_t=0.0,
        )
        correction_forward = forward_mlp_with_jvp(
            self.correction_network,
            correction_inputs,
            correction_input_tangent,
        )
        local_input_tangent = (
            np.asarray(local_feature_tangent_matrix, dtype=np.float64)
            if self.config.feature_aware_teacher_tangents
            else np.zeros_like(local_feature_matrix, dtype=np.float64)
        )
        local_forward = forward_mlp_with_jvp(
            self.local_network,
            np.asarray(local_feature_matrix, dtype=np.float64),
            local_input_tangent,
        )
        u_local = np.asarray(local_forward.output, dtype=np.float64)
        u_corr = np.asarray(correction_forward.output, dtype=np.float64)
        return {
            "correction_inputs": correction_inputs,
            "local_inputs": np.asarray(local_feature_matrix, dtype=np.float64),
            "u_local": u_local,
            "u_corr": u_corr,
            "u_hat": (u_local + u_corr).astype(np.float64, copy=False),
            "u_local_jvp": np.asarray(local_forward.jvp, dtype=np.float64),
            "u_corr_jvp": np.asarray(correction_forward.jvp, dtype=np.float64),
            "u_hat_jvp": (local_forward.jvp + correction_forward.jvp).astype(np.float64, copy=False),
        }

    def predict_u_for_rollout(
        self,
        split_name: str,
        z_s: np.ndarray,
        target_onehot: np.ndarray,
        tau_s: np.ndarray | float,
        tau_t: np.ndarray | float,
    ) -> np.ndarray:
        correction_feature_matrix, local_feature_matrix, _ = self._current_state_rollout_features(
            split_name,
            z_s,
            target_onehot,
            tau_s,
            tau_t,
        )
        correction_inputs = self.normalization.transform_inputs(
            z_s,
            target_onehot,
            tau_s,
            tau_t,
            teacher_features=correction_feature_matrix,
        )
        u_local = self.local_network.predict(local_feature_matrix)
        u_corr = self.correction_network.predict(correction_inputs)
        return (u_local + u_corr).astype(np.float64, copy=False)

    def _train_on_raw_target(
        self,
        network: MLPNetwork,
        inputs: np.ndarray,
        target_raw: np.ndarray,
        *,
        step_scale: float,
    ) -> float:
        if step_scale < 0.0:
            raise ValueError("step_scale must be non-negative.")
        if step_scale == 0.0:
            predictions = network.predict(inputs)
            return float(np.mean((predictions - target_raw) ** 2))
        original_eta_w = float(network.eta_w)
        original_eta_b = float(network.eta_b)
        network.eta_w = original_eta_w * step_scale
        network.eta_b = original_eta_b * step_scale
        try:
            result = network.train_batch(inputs, target_raw)
        finally:
            network.eta_w = original_eta_w
            network.eta_b = original_eta_b
        return float(result.loss)

    def _identity_active_mask(self, batch: FMPCIntervalBatch) -> np.ndarray:
        return meanflow_identity_active_mask(
            batch.source_step_indices,
            batch.target_step_indices,
            teacher_steps=self.teacher_feature_bundle.teacher_steps,
            identity_scope_mode=self.config.identity_scope_mode,
            knot_focused_schedule_names=self.config.knot_focused_schedule_names,
        )

    def train_batch(
        self,
        batch: FMPCIntervalBatch,
        *,
        split_name: str,
        teacher_loss_weight: float,
        identity_loss_weight: float,
        step_scale: float = 1.0,
        train_local_branch: bool = True,
        train_correction_branch: bool = True,
    ) -> dict[str, Any]:
        (
            correction_feature_matrix,
            correction_feature_tangent_matrix,
            local_feature_matrix,
            local_feature_tangent_matrix,
            state_features,
            _state_tangents,
        ) = self._trajectory_batch_features(split_name, batch)
        forward = self._forward_branches(
            z_s=batch.z_s,
            target_onehot=batch.target_onehot,
            tau_s=batch.tau_s,
            tau_t=batch.tau_t,
            correction_feature_matrix=correction_feature_matrix,
            correction_feature_tangent_matrix=correction_feature_tangent_matrix,
            local_feature_matrix=local_feature_matrix,
            local_feature_tangent_matrix=local_feature_tangent_matrix,
            g_s=state_features.g_s,
        )
        u_star = np.asarray(batch.u_star, dtype=np.float64)
        u_hat = forward["u_hat"]
        u_local = forward["u_local"]
        u_corr = forward["u_corr"]
        identity_target_raw = build_meanflow_full_identity_target_raw(
            g_s=state_features.g_s,
            delta_tau=np.asarray(batch.delta_tau, dtype=np.float64),
            combined_jvp_raw=forward["u_hat_jvp"],
        )
        teacher_loss = float(np.mean((u_hat - u_star) ** 2))
        identity_active_mask = self._identity_active_mask(batch)
        identity_active_fraction = float(np.mean(identity_active_mask.astype(np.float64)))
        if np.any(identity_active_mask):
            identity_loss = float(
                np.mean((u_hat[identity_active_mask] - identity_target_raw[identity_active_mask]) ** 2)
            )
        else:
            identity_loss = 0.0
        local_anchor_loss = float(np.mean((u_local - state_features.g_s) ** 2))
        correction_teacher_target = (u_star - u_local).astype(np.float64, copy=False)
        branch_teacher_scale = 0.5 * float(teacher_loss_weight) * float(step_scale)
        local_teacher_update_loss = None
        correction_teacher_update_loss = None
        if train_local_branch:
            local_teacher_update_loss = self._train_on_raw_target(
                self.local_network,
                forward["local_inputs"],
                state_features.g_s,
                step_scale=branch_teacher_scale,
            )
        if train_correction_branch:
            correction_teacher_update_loss = self._train_on_raw_target(
                self.correction_network,
                forward["correction_inputs"],
                correction_teacher_target,
                step_scale=branch_teacher_scale,
            )
        local_identity_update_loss = None
        correction_identity_update_loss = None
        if float(identity_loss_weight) > 0.0 and np.any(identity_active_mask):
            branch_identity_scale = 0.5 * float(identity_loss_weight) * float(step_scale)
            if train_local_branch:
                local_identity_update_loss = self._train_on_raw_target(
                    self.local_network,
                    forward["local_inputs"][identity_active_mask],
                    (identity_target_raw[identity_active_mask] - u_corr[identity_active_mask]).astype(
                        np.float64,
                        copy=False,
                    ),
                    step_scale=branch_identity_scale,
                )
            if train_correction_branch:
                correction_identity_update_loss = self._train_on_raw_target(
                    self.correction_network,
                    forward["correction_inputs"][identity_active_mask],
                    (identity_target_raw[identity_active_mask] - u_local[identity_active_mask]).astype(
                        np.float64,
                        copy=False,
                    ),
                    step_scale=branch_identity_scale,
                )

        total_weight = float(teacher_loss_weight) + float(identity_loss_weight) * identity_active_fraction
        if total_weight > 0.0:
            total_loss = (
                float(teacher_loss_weight) * teacher_loss
                + float(identity_loss_weight) * identity_active_fraction * identity_loss
            ) / total_weight
        else:
            total_loss = teacher_loss
        return {
            "teacher_loss": teacher_loss,
            "identity_loss": identity_loss,
            "total_loss": float(total_loss),
            "identity_active_fraction": identity_active_fraction,
            "local_anchor_loss": local_anchor_loss,
            "train_local_branch": bool(train_local_branch),
            "train_correction_branch": bool(train_correction_branch),
            "local_teacher_update_loss": None
            if local_teacher_update_loss is None
            else float(local_teacher_update_loss),
            "correction_teacher_update_loss": None
            if correction_teacher_update_loss is None
            else float(correction_teacher_update_loss),
            "local_identity_update_loss": None
            if local_identity_update_loss is None
            else float(local_identity_update_loss),
            "correction_identity_update_loss": None
            if correction_identity_update_loss is None
            else float(correction_identity_update_loss),
        }

    def train_aux_batch(
        self,
        batch: FMPCIntervalBatch,
        *,
        split_name: str,
        aux_weight: float,
    ) -> float:
        correction_feature_matrix, local_feature_matrix, _state_features = self._current_state_rollout_features(
            split_name,
            batch.z_s,
            batch.target_onehot,
            batch.tau_s,
            batch.tau_t,
        )
        correction_inputs = self.normalization.transform_inputs(
            batch.z_s,
            batch.target_onehot,
            batch.tau_s,
            batch.tau_t,
            teacher_features=correction_feature_matrix,
        )
        u_local = self.local_network.predict(local_feature_matrix)
        correction_target = (np.asarray(batch.u_star, dtype=np.float64) - u_local).astype(np.float64, copy=False)
        return self._train_on_raw_target(
            self.correction_network,
            correction_inputs,
            correction_target,
            step_scale=float(aux_weight),
        )

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "family": self.config.family_name,
            "model_kind": "two_branch",
            "branch_decomposition": "u_hat = u_local + u_corr",
            "teacher_loss_weight": float(self.config.teacher_loss_weight),
            "identity_loss_weight": float(self.config.identity_loss_weight),
            "rollout_aux_weight": float(self.config.rollout_aux_weight),
            "teacher_warmup_epochs": int(self.config.teacher_warmup_epochs),
            "identity_ramp_epochs": int(self.config.identity_ramp_epochs),
            "identity_scope_mode": self.config.identity_scope_mode,
            "knot_focus_probability": float(self.config.knot_focus_probability),
            "knot_focused_schedule_names": list(self.config.knot_focused_schedule_names),
            "selected_teacher_feature_names": list(self.config.selected_teacher_feature_names),
            "local_branch": {
                "family": "linear_affine",
                "input_feature_names": list(self.config.local_branch_feature_names),
                "initialized_near_teacher_local_field": True,
                "maps_g_s_to_u_local_at_init": True,
                "warm_started": bool(self.config.local_branch_warm_start),
                "warm_start_source_family": self.config.warm_start_source_family
                if self.config.local_branch_warm_start
                else None,
            },
            "correction_branch": {
                "family": "mlp",
                "hidden_dims": list(self.config.correction_hidden_dims),
                "hidden_activation": self.config.hidden_activation,
                "output_activation": self.config.output_activation,
                "zero_init_output_head": bool(self.config.zero_init_correction_output_head),
            },
            "staged_training": {
                "enabled": bool(self.config.family_name == "meanflow_twobranch_residual_warmstart"),
                "stage_a_name": "correction_only_warmup",
                "stage_a_epochs": int(self.config.correction_only_warmup_epochs),
                "stage_a_local_branch_frozen": bool(self.config.family_name == "meanflow_twobranch_residual_warmstart"),
                "stage_b_name": "joint_hybrid_finetune",
                "stage_b_epochs": int(max(0, self.config.epochs - self.config.correction_only_warmup_epochs)),
            },
            "feature_aware_teacher_tangents": bool(self.config.feature_aware_teacher_tangents),
            "feature_tangent_epsilon": float(self.config.feature_tangent_epsilon),
            "residual_identity_includes_dg_s": True,
            "meanflow_identity_enabled": bool(self.config.identity_loss_weight > 0.0),
            "normalization": self.normalization.to_jsonable(),
            "student_input_definition": "u_hat = u_local([g_s, e_out_s, F_s]) + u_corr([z_s, target_onehot, tau_s, tau_t, teacher_state_features])",
            "direct_target_definition": "u_star = (z_t - z_s) / (tau_t - tau_s)",
        }


def _prepare_interval_teacher_feature_bundle(
    *,
    interval_dataset: FMPCIntervalDataset,
    teacher_split: Any,
    teacher_model: Any,
    teacher_export_batch_size: int,
    feature_tangent_epsilon: float,
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
            fd_epsilon=feature_tangent_epsilon,
        ),
        val_tangents=precompute_interval_teacher_trajectory_feature_tangents(
            teacher_model,
            interval_dataset.val,
            val_context,
            fd_epsilon=feature_tangent_epsilon,
        ),
        test_tangents=precompute_interval_teacher_trajectory_feature_tangents(
            teacher_model,
            interval_dataset.test,
            test_context,
            fd_epsilon=feature_tangent_epsilon,
        ),
        split_contexts={
            "train": train_context,
            "val": val_context,
            "test": test_context,
        },
    )


def _teacher_export_batch_size_from_endpoint_dataset(dataset: FMPCStudentDataset) -> int:
    return int(dataset.metadata["teacher_export_batch_size"])


def _evaluate_rollout(
    model: Any,
    interval_split: FMPCIntervalSplit,
    endpoint_split: Any,
    reference: Any,
    teacher_model: Any,
    *,
    rollout_schedule_name: str,
    knots: tuple[int, ...],
) -> tuple[Any, dict[str, Any]]:
    rollout = rollout_interval_student(
        model,
        interval_split,
        rollout_schedule_name=rollout_schedule_name,
        knots=knots,
    )
    delta_z_hat = np.asarray(rollout.final_state - endpoint_split.z0, dtype=np.float64)
    evaluation = evaluate_fmpc_delta_predictions(
        delta_z_hat,
        endpoint_split,
        reference,
        teacher_model,
        transport_wall_time_seconds=rollout.transport_wall_time_seconds,
    )
    metrics = _interval_metrics_from_endpoint(
        fmpc_split_evaluation_metrics_payload(evaluation),
        mean_knot_state_rms_gap=rollout.mean_knot_state_rms_gap,
    )
    return rollout, metrics


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
    *,
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
        "hidden_dims": None,
        "epochs": None,
        "eta_w": None,
        "teacher_loss_weight": None,
        "identity_loss_weight": None,
        "rollout_aux_weight": None,
        "feature_contract": "identity",
        "target_contract": "delta_z_hat = 0",
        "meanflow_identity_enabled": False,
        "winner_uses_nonzero_identity_term": False,
        "knot_focus_probability": None,
        "is_family_best": True,
        "is_learned_family": False,
        "is_meanflow_family": False,
        "is_overall_winner": False,
        "evaluated_on_test": True,
        **_candidate_metric_columns("val", val_metrics),
        **_candidate_metric_columns("test", test_metrics),
    }
    summary = _endpoint_baseline_summary_payload(
        family="identity",
        config_id="identity",
        model_config={
            "family": "identity",
            "student_output_definition": "delta_z_hat = 0",
        },
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )
    return row, summary


def _endpoint_ridge_baseline_row_and_summary(
    *,
    config: FMPCMeanFlowSuiteConfig,
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
        "hidden_dims": None,
        "epochs": None,
        "eta_w": None,
        "teacher_loss_weight": None,
        "identity_loss_weight": None,
        "rollout_aux_weight": None,
        "feature_contract": endpoint_dataset.student_input_definition,
        "target_contract": endpoint_dataset.student_target_definition,
        "meanflow_identity_enabled": False,
        "winner_uses_nonzero_identity_term": False,
        "knot_focus_probability": None,
        "is_family_best": True,
        "is_learned_family": False,
        "is_meanflow_family": False,
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


def _phase5b2_baseline_row_and_summary(
    *,
    config: FMPCMeanFlowSuiteConfig,
    interval_dataset: FMPCIntervalDataset,
    endpoint_dataset: FMPCStudentDataset,
    references: dict[str, Any],
    teacher_model: Any,
    teacher_feature_bundle: FMPCIntervalTeacherFeatureBundle,
    rollout_schedules: dict[str, tuple[int, ...]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalization = fit_fmpc_interval_augmented_normalization(
        interval_dataset.train,
        trajectory_features=teacher_feature_bundle.train,
        selected_feature_names=config.feature_names,
        target_mode="u_residual_local_field",
        knot_focused_schedule_names=config.knot_focused_schedule_names,
        knot_focus_mixture=float(config.carried_forward_phase5b2_knot_focus_mixture),
        eps=config.normalization_eps,
    )
    model = IntervalAugmentedRidgeStudent.fit(
        interval_dataset.train,
        normalization=normalization,
        config=IntervalAugmentedRidgeStudentConfig(
            alpha=float(config.carried_forward_phase5b2_alpha),
            selected_teacher_feature_names=tuple(config.feature_names),
            target_mode="u_residual_local_field",
            knot_focused_schedule_names=tuple(config.knot_focused_schedule_names),
            knot_focus_mixture=float(config.carried_forward_phase5b2_knot_focus_mixture),
        ),
        family_name="interval_ridge_residual",
        teacher_model=teacher_model,
        teacher_feature_bundle=teacher_feature_bundle,
    )
    knots = rollout_schedules[config.carried_forward_phase5b2_schedule_name]
    _, val_metrics = _evaluate_rollout(
        model,
        interval_dataset.val,
        endpoint_dataset.val,
        references["val"],
        teacher_model,
        rollout_schedule_name=config.carried_forward_phase5b2_schedule_name,
        knots=knots,
    )
    _, test_metrics = _evaluate_rollout(
        model,
        interval_dataset.test,
        endpoint_dataset.test,
        references["test"],
        teacher_model,
        rollout_schedule_name=config.carried_forward_phase5b2_schedule_name,
        knots=knots,
    )
    row = {
        "config_id": (
            f"phase5b2_interval_ridge_residual_alpha_{config.carried_forward_phase5b2_alpha:g}"
            f"_focus{config.carried_forward_phase5b2_knot_focus_mixture:g}"
        ),
        "family": "phase5b2_interval_ridge_residual",
        "rollout_schedule": config.carried_forward_phase5b2_schedule_name,
        "schedule_knots": "-".join(str(knot) for knot in knots),
        "hidden_dims": None,
        "epochs": None,
        "eta_w": None,
        "teacher_loss_weight": None,
        "identity_loss_weight": None,
        "rollout_aux_weight": None,
        "feature_contract": ",".join(config.feature_names),
        "target_contract": "u_res = u_star - g_s",
        "meanflow_identity_enabled": False,
        "winner_uses_nonzero_identity_term": False,
        "knot_focus_probability": float(config.carried_forward_phase5b2_knot_focus_mixture),
        "is_family_best": True,
        "is_learned_family": False,
        "is_meanflow_family": False,
        "is_overall_winner": False,
        "evaluated_on_test": True,
        **_candidate_metric_columns("val", val_metrics),
        **_candidate_metric_columns("test", test_metrics),
    }
    summary = {
        "family": "phase5b2_interval_ridge_residual",
        "config_id": str(row["config_id"]),
        "rollout_schedule": config.carried_forward_phase5b2_schedule_name,
        "schedule_knots": list(knots),
        "selection_metric_source": "val_metric",
        "report_metric_source": "test_metric",
        "model_config": model.to_jsonable(),
        "val": val_metrics,
        "test": test_metrics,
    }
    return row, summary


def _phase6a1_linear_baseline_row_summary_and_record(
    *,
    config: FMPCMeanFlowSuiteConfig,
    interval_dataset: FMPCIntervalDataset,
    endpoint_dataset: FMPCStudentDataset,
    references: dict[str, Any],
    teacher_model: Any,
    teacher_feature_bundle: FMPCIntervalTeacherFeatureBundle,
    rollout_schedules: dict[str, tuple[int, ...]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    normalization = fit_fmpc_interval_augmented_normalization(
        interval_dataset.train,
        trajectory_features=teacher_feature_bundle.train,
        selected_feature_names=config.feature_names,
        target_mode="u_residual_local_field",
        knot_focused_schedule_names=config.knot_focused_schedule_names,
        knot_focus_mixture=0.0,
        eps=config.normalization_eps,
    )
    baseline_config = MeanFlowMLPStudentConfig(
        hidden_dims=(),
        family_name="meanflow_linear_residual",
        target_mode="u_residual_local_field",
        selected_teacher_feature_names=tuple(config.feature_names),
        hidden_activation=config.hidden_activation,
        output_activation=config.output_activation,
        weight_scale=config.weight_scale,
        eta_w=float(config.carried_forward_phase6a1_linear_eta_w),
        eta_b=float(config.carried_forward_phase6a1_linear_eta_w),
        epochs=40,
        batch_size=int(config.batch_size),
        batches_per_epoch=int(
            config.batches_per_epoch
            if config.batches_per_epoch is not None
            else int(np.ceil(interval_dataset.train.num_samples / config.batch_size))
        ),
        teacher_loss_weight=float(config.teacher_loss_weight),
        identity_loss_weight=float(config.carried_forward_phase6a1_linear_identity_loss_weight),
        rollout_aux_weight=0.0,
        teacher_warmup_epochs=int(config.teacher_warmup_epochs),
        identity_ramp_epochs=int(config.identity_ramp_epochs),
        identity_scope_mode=config.carried_forward_phase6a1_linear_identity_scope_mode,
        knot_focus_probability=0.0,
        knot_focused_schedule_names=tuple(config.knot_focused_schedule_names),
        feature_aware_teacher_tangents=True,
        feature_tangent_epsilon=float(config.feature_tangent_epsilon),
    )
    model = _build_meanflow_candidate_model(
        interval_dataset=interval_dataset,
        normalization=normalization,
        config=baseline_config,
        seed=config.model_init_seed + 26001,
        teacher_model=teacher_model,
        teacher_feature_bundle=teacher_feature_bundle,
    )
    best_by_schedule = _fit_meanflow_candidate_over_epochs(
        model,
        model_config=baseline_config,
        interval_dataset=interval_dataset,
        endpoint_dataset=endpoint_dataset,
        references=references,
        teacher_model=teacher_model,
        rollout_schedules=rollout_schedules,
        aux_rollout_schedules={name: rollout_schedules[name] for name in config.rollout_aux_schedule_names},
        batch_order_seed=config.batch_order_seed,
        candidate_seed_offset=26001,
    )
    schedule_name = str(config.carried_forward_phase6a1_linear_schedule_name)
    knots = rollout_schedules[schedule_name]
    schedule_best = best_by_schedule[schedule_name]
    if schedule_best["snapshot"] is None:
        raise RuntimeError("Phase 6A.1 linear baseline did not record a best snapshot.")
    model.restore(schedule_best["snapshot"])
    _, test_metrics = _evaluate_rollout(
        model,
        interval_dataset.test,
        endpoint_dataset.test,
        references["test"],
        teacher_model,
        rollout_schedule_name=schedule_name,
        knots=knots,
    )
    row = {
        "config_id": (
            f"phase6a1_linear_residual_lr{config.carried_forward_phase6a1_linear_eta_w:g}"
            f"_id{config.carried_forward_phase6a1_linear_identity_loss_weight:g}"
            f"_scope{'all' if config.carried_forward_phase6a1_linear_identity_scope_mode == 'all_intervals' else 'knot'}"
        ),
        "family": "phase6a1_linear_residual_baseline",
        "rollout_schedule": schedule_name,
        "schedule_knots": "-".join(str(knot) for knot in knots),
        "hidden_dims": "linear",
        "epochs": int(baseline_config.epochs),
        "eta_w": float(baseline_config.eta_w),
        "teacher_loss_weight": float(baseline_config.teacher_loss_weight),
        "identity_loss_weight": float(baseline_config.identity_loss_weight),
        "rollout_aux_weight": float(baseline_config.rollout_aux_weight),
        "identity_scope_mode": baseline_config.identity_scope_mode,
        "feature_contract": ",".join(baseline_config.selected_teacher_feature_names),
        "target_contract": "u_res = u_star - g_s",
        "meanflow_identity_enabled": True,
        "winner_uses_nonzero_identity_term": False,
        "knot_focus_probability": 0.0,
        "is_two_branch": False,
        "local_branch_input_set": "",
        "correction_branch_family": "",
        "is_family_best": True,
        "is_learned_family": False,
        "is_meanflow_family": True,
        "model_kind": "linear",
        "is_overall_winner": False,
        "evaluated_on_test": True,
        "feature_aware_teacher_tangents": True,
        "zero_tangent_through_teacher_feature_block": False,
        "feature_tangent_epsilon": float(baseline_config.feature_tangent_epsilon),
        "residual_identity_includes_dg_s": True,
        "best_epoch": int(schedule_best["epoch"]),
        "train_teacher_loss": schedule_best["training_metrics"]["teacher_loss"],
        "train_identity_loss": schedule_best["training_metrics"]["identity_loss"],
        "train_total_loss": schedule_best["training_metrics"]["total_loss"],
        "effective_teacher_loss_weight": schedule_best["training_metrics"]["effective_teacher_loss_weight"],
        "effective_identity_loss_weight": schedule_best["training_metrics"]["effective_identity_loss_weight"],
        "train_identity_active_fraction": schedule_best["training_metrics"]["identity_active_fraction"],
        "train_rollout_aux_loss": schedule_best["training_metrics"]["rollout_aux_loss"],
        "train_rollout_aux_velocity_mse": schedule_best["training_metrics"]["rollout_aux_velocity_mse"],
        **_candidate_metric_columns("val", schedule_best["val_metrics"]),
        **_candidate_metric_columns("test", test_metrics),
    }
    summary = {
        "family": "phase6a1_linear_residual_baseline",
        "config_id": str(row["config_id"]),
        "rollout_schedule": schedule_name,
        "schedule_knots": list(knots),
        "selection_metric_source": "val_metric",
        "report_metric_source": "test_metric",
        "model_config": model.to_jsonable(),
        "best_epoch": int(schedule_best["epoch"]),
        "training_diagnostics": dict(schedule_best["training_metrics"]),
        "val": schedule_best["val_metrics"],
        "test": test_metrics,
    }
    record = {
        "config": baseline_config,
        "normalization": normalization,
        "seed": int(config.model_init_seed + 26001),
        "snapshot": schedule_best["snapshot"],
        "best_epoch": int(schedule_best["epoch"]),
        "training_metrics": dict(schedule_best["training_metrics"]),
        "val_metrics": schedule_best["val_metrics"],
        "test_metrics": test_metrics,
        "model": model,
    }
    return row, summary, record


def _phase6a2_twobranch_baseline_row_and_summary(
    *,
    config: FMPCMeanFlowSuiteConfig,
    interval_dataset: FMPCIntervalDataset,
    endpoint_dataset: FMPCStudentDataset,
    references: dict[str, Any],
    teacher_model: Any,
    teacher_feature_bundle: FMPCIntervalTeacherFeatureBundle,
    rollout_schedules: dict[str, tuple[int, ...]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalization = fit_fmpc_interval_augmented_normalization(
        interval_dataset.train,
        trajectory_features=teacher_feature_bundle.train,
        selected_feature_names=config.feature_names,
        target_mode="u_residual_local_field",
        knot_focused_schedule_names=config.knot_focused_schedule_names,
        knot_focus_mixture=0.0,
        eps=config.normalization_eps,
    )
    baseline_config = MeanFlowTwoBranchStudentConfig(
        correction_hidden_dims=tuple(config.carried_forward_phase6a2_twobranch_hidden_dims),
        family_name="meanflow_twobranch_residual",
        selected_teacher_feature_names=tuple(config.feature_names),
        local_branch_feature_names=("g_s", "e_out_s", "F_s"),
        hidden_activation=config.hidden_activation,
        output_activation=config.output_activation,
        weight_scale=config.weight_scale,
        eta_w=float(config.carried_forward_phase6a2_twobranch_eta_w),
        eta_b=float(config.carried_forward_phase6a2_twobranch_eta_w),
        epochs=40,
        batch_size=int(config.batch_size),
        batches_per_epoch=int(
            config.batches_per_epoch
            if config.batches_per_epoch is not None
            else int(np.ceil(interval_dataset.train.num_samples / config.batch_size))
        ),
        teacher_loss_weight=float(config.teacher_loss_weight),
        identity_loss_weight=float(config.carried_forward_phase6a2_twobranch_identity_loss_weight),
        rollout_aux_weight=0.0,
        teacher_warmup_epochs=int(config.teacher_warmup_epochs),
        identity_ramp_epochs=int(config.identity_ramp_epochs),
        identity_scope_mode=config.carried_forward_phase6a2_twobranch_identity_scope_mode,
        knot_focus_probability=0.0,
        knot_focused_schedule_names=tuple(config.knot_focused_schedule_names),
        feature_aware_teacher_tangents=True,
        feature_tangent_epsilon=float(config.feature_tangent_epsilon),
    )
    model = _build_meanflow_twobranch_candidate_model(
        interval_dataset=interval_dataset,
        normalization=normalization,
        config=baseline_config,
        seed=config.model_init_seed + 27001,
        teacher_model=teacher_model,
        teacher_feature_bundle=teacher_feature_bundle,
    )
    best_by_schedule = _fit_meanflow_twobranch_candidate_over_epochs(
        model,
        model_config=baseline_config,
        interval_dataset=interval_dataset,
        endpoint_dataset=endpoint_dataset,
        references=references,
        teacher_model=teacher_model,
        rollout_schedules=rollout_schedules,
        aux_rollout_schedules={name: rollout_schedules[name] for name in config.rollout_aux_schedule_names},
        batch_order_seed=config.batch_order_seed,
        candidate_seed_offset=27001,
    )
    schedule_name = str(config.carried_forward_phase6a2_twobranch_schedule_name)
    knots = rollout_schedules[schedule_name]
    schedule_best = best_by_schedule[schedule_name]
    if schedule_best["snapshot"] is None:
        raise RuntimeError("Phase 6A.2 two-branch baseline did not record a best snapshot.")
    model.restore(schedule_best["snapshot"])
    _, test_metrics = _evaluate_rollout(
        model,
        interval_dataset.test,
        endpoint_dataset.test,
        references["test"],
        teacher_model,
        rollout_schedule_name=schedule_name,
        knots=knots,
    )
    row = {
        "config_id": (
            f"phase6a2_twobranch_h{_serialize_hidden_dims(config.carried_forward_phase6a2_twobranch_hidden_dims)}"
            f"_lr{config.carried_forward_phase6a2_twobranch_eta_w:g}"
            f"_id{config.carried_forward_phase6a2_twobranch_identity_loss_weight:g}"
            f"_scope{'all' if config.carried_forward_phase6a2_twobranch_identity_scope_mode == 'all_intervals' else 'knot'}"
        ),
        "family": "phase6a2_twobranch_residual_baseline",
        "rollout_schedule": schedule_name,
        "schedule_knots": "-".join(str(knot) for knot in knots),
        "hidden_dims": _serialize_hidden_dims(config.carried_forward_phase6a2_twobranch_hidden_dims),
        "epochs": int(baseline_config.epochs),
        "eta_w": float(baseline_config.eta_w),
        "teacher_loss_weight": float(baseline_config.teacher_loss_weight),
        "identity_loss_weight": float(baseline_config.identity_loss_weight),
        "rollout_aux_weight": float(baseline_config.rollout_aux_weight),
        "identity_scope_mode": baseline_config.identity_scope_mode,
        "feature_contract": ",".join(baseline_config.selected_teacher_feature_names),
        "target_contract": "u_hat = u_local + u_corr",
        "meanflow_identity_enabled": True,
        "winner_uses_nonzero_identity_term": False,
        "knot_focus_probability": 0.0,
        "is_two_branch": True,
        "local_branch_input_set": ",".join(baseline_config.local_branch_feature_names),
        "correction_branch_family": "mlp",
        "local_branch_warm_started": False,
        "warm_start_source": "",
        "correction_only_warmup_epochs": int(baseline_config.correction_only_warmup_epochs),
        "is_family_best": True,
        "is_learned_family": False,
        "is_meanflow_family": True,
        "model_kind": "two_branch",
        "is_overall_winner": False,
        "evaluated_on_test": True,
        "feature_aware_teacher_tangents": True,
        "zero_tangent_through_teacher_feature_block": False,
        "feature_tangent_epsilon": float(baseline_config.feature_tangent_epsilon),
        "residual_identity_includes_dg_s": True,
        "best_epoch": int(schedule_best["epoch"]),
        "train_teacher_loss": schedule_best["training_metrics"]["teacher_loss"],
        "train_identity_loss": schedule_best["training_metrics"]["identity_loss"],
        "train_total_loss": schedule_best["training_metrics"]["total_loss"],
        "effective_teacher_loss_weight": schedule_best["training_metrics"]["effective_teacher_loss_weight"],
        "effective_identity_loss_weight": schedule_best["training_metrics"]["effective_identity_loss_weight"],
        "train_identity_active_fraction": schedule_best["training_metrics"]["identity_active_fraction"],
        "train_rollout_aux_loss": schedule_best["training_metrics"]["rollout_aux_loss"],
        "train_rollout_aux_velocity_mse": schedule_best["training_metrics"]["rollout_aux_velocity_mse"],
        **_candidate_metric_columns("val", schedule_best["val_metrics"]),
        **_candidate_metric_columns("test", test_metrics),
    }
    summary = {
        "family": "phase6a2_twobranch_residual_baseline",
        "config_id": str(row["config_id"]),
        "rollout_schedule": schedule_name,
        "schedule_knots": list(knots),
        "selection_metric_source": "val_metric",
        "report_metric_source": "test_metric",
        "model_config": model.to_jsonable(),
        "best_epoch": int(schedule_best["epoch"]),
        "training_diagnostics": dict(schedule_best["training_metrics"]),
        "val": schedule_best["val_metrics"],
        "test": test_metrics,
    }
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


def _build_meanflow_candidate_model(
    *,
    interval_dataset: FMPCIntervalDataset,
    normalization: FMPCIntervalNormalizationStats,
    config: MeanFlowMLPStudentConfig,
    seed: int,
    teacher_model: Any,
    teacher_feature_bundle: FMPCIntervalTeacherFeatureBundle,
) -> MeanFlowMLPStudent:
    return MeanFlowMLPStudent.initialize(
        z_dim=interval_dataset.z_dim,
        target_dim=interval_dataset.target_dim,
        normalization=normalization,
        config=config,
        seed=seed,
        teacher_model=teacher_model,
        teacher_feature_bundle=teacher_feature_bundle,
    )


def _build_meanflow_twobranch_candidate_model(
    *,
    interval_dataset: FMPCIntervalDataset,
    normalization: FMPCIntervalNormalizationStats,
    config: MeanFlowTwoBranchStudentConfig,
    seed: int,
    teacher_model: Any,
    teacher_feature_bundle: FMPCIntervalTeacherFeatureBundle,
    local_branch_warm_start: tuple[np.ndarray, np.ndarray] | None = None,
) -> MeanFlowTwoBranchResidualStudent:
    return MeanFlowTwoBranchResidualStudent.initialize(
        z_dim=interval_dataset.z_dim,
        target_dim=interval_dataset.target_dim,
        normalization=normalization,
        config=config,
        seed=seed,
        teacher_model=teacher_model,
        teacher_feature_bundle=teacher_feature_bundle,
        local_branch_warm_start=local_branch_warm_start,
    )


def _meanflow_rollout_aux_training_diagnostics(
    model: Any,
    train_split: FMPCIntervalSplit,
    *,
    aux_rollout_schedules: dict[str, tuple[int, ...]],
    aux_weight: float,
) -> dict[str, Any]:
    velocity_mses: list[float] = []
    intermediate_state_rms_gaps: list[float] = []
    final_state_rms_gaps: list[float] = []
    aux_losses: list[float] = []
    per_schedule: dict[str, Any] = {}
    for schedule_name, knots in aux_rollout_schedules.items():
        auxiliary = build_rollout_auxiliary_batches(
            model,
            train_split,
            rollout_schedule_name=schedule_name,
            knots=knots,
        )
        velocity_mses.append(float(auxiliary.rollout_aux_velocity_mse))
        if auxiliary.intermediate_state_rms_gap is not None:
            intermediate_state_rms_gaps.append(float(auxiliary.intermediate_state_rms_gap))
        final_state_rms_gaps.append(float(auxiliary.final_state_rms_gap))
        schedule_losses: list[float] = []
        for aux_batch in auxiliary.batches:
            schedule_losses.append(
                model.train_aux_batch(
                    aux_batch,
                    split_name="train",
                    aux_weight=aux_weight,
                )
            )
        aux_losses.extend(schedule_losses)
        per_schedule[schedule_name] = {
            "rollout_aux_velocity_mse": float(auxiliary.rollout_aux_velocity_mse),
            "intermediate_state_rms_gap": auxiliary.intermediate_state_rms_gap,
            "final_state_rms_gap": float(auxiliary.final_state_rms_gap),
            "train_aux_loss": _mean_or_none(schedule_losses),
        }
    return {
        "rollout_aux_velocity_mse": _mean_or_none(velocity_mses),
        "rollout_aux_intermediate_state_rms_gap": _mean_or_none(intermediate_state_rms_gaps),
        "rollout_aux_final_state_rms_gap": _mean_or_none(final_state_rms_gaps),
        "rollout_aux_loss": _mean_or_none(aux_losses),
        "rollout_aux_per_schedule": per_schedule,
    }


def _fit_meanflow_candidate_over_epochs(
    model: Any,
    *,
    model_config: Any,
    interval_dataset: FMPCIntervalDataset,
    endpoint_dataset: FMPCStudentDataset,
    references: dict[str, Any],
    teacher_model: Any,
    rollout_schedules: dict[str, tuple[int, ...]],
    aux_rollout_schedules: dict[str, tuple[int, ...]],
    batch_order_seed: int,
    candidate_seed_offset: int,
) -> dict[str, dict[str, Any]]:
    best_by_schedule: dict[str, dict[str, Any]] = {
        schedule_name: {"val_metric": np.inf, "snapshot": None}
        for schedule_name in rollout_schedules
    }
    for epoch in range(1, int(model_config.epochs) + 1):
        teacher_weight, identity_weight = _training_loss_weights(
            epoch=epoch,
            teacher_loss_weight=float(model_config.teacher_loss_weight),
            identity_loss_weight=float(model_config.identity_loss_weight),
            teacher_warmup_epochs=int(model_config.teacher_warmup_epochs),
            identity_ramp_epochs=int(model_config.identity_ramp_epochs),
        )
        teacher_losses: list[float] = []
        identity_losses: list[float] = []
        total_losses: list[float] = []
        identity_active_fractions: list[float] = []
        local_anchor_losses: list[float] = []
        for batch in iter_mixed_interval_batches(
            interval_dataset.train,
            int(model_config.batch_size),
            num_batches=int(model_config.batches_per_epoch),
            seed=int(batch_order_seed) + int(candidate_seed_offset) * 1000 + epoch,
            knot_focused_schedule_names=tuple(model_config.knot_focused_schedule_names),
            knot_focus_probability=float(model_config.knot_focus_probability),
        ):
            diagnostics = model.train_batch(
                batch,
                split_name="train",
                teacher_loss_weight=teacher_weight,
                identity_loss_weight=identity_weight,
            )
            teacher_losses.append(float(diagnostics["teacher_loss"]))
            identity_losses.append(float(diagnostics["identity_loss"]))
            total_losses.append(float(diagnostics["total_loss"]))
            identity_active_fractions.append(float(diagnostics["identity_active_fraction"]))
            if "local_anchor_loss" in diagnostics:
                local_anchor_losses.append(float(diagnostics["local_anchor_loss"]))
        rollout_aux_metrics = {
            "rollout_aux_loss": None,
            "rollout_aux_velocity_mse": None,
            "rollout_aux_intermediate_state_rms_gap": None,
            "rollout_aux_final_state_rms_gap": None,
            "rollout_aux_per_schedule": {},
        }
        if float(model_config.rollout_aux_weight) > 0.0:
            rollout_aux_metrics = _meanflow_rollout_aux_training_diagnostics(
                model,
                interval_dataset.train,
                aux_rollout_schedules=aux_rollout_schedules,
                aux_weight=float(model_config.rollout_aux_weight),
            )
        epoch_training_metrics = {
            "teacher_loss": _mean_or_none(teacher_losses),
            "identity_loss": _mean_or_none(identity_losses),
            "total_loss": _mean_or_none(total_losses),
            "effective_teacher_loss_weight": float(teacher_weight),
            "effective_identity_loss_weight": float(identity_weight),
            "identity_scope_mode": model_config.identity_scope_mode,
            "identity_active_fraction": _mean_or_none(identity_active_fractions),
            "local_anchor_loss": _mean_or_none(local_anchor_losses),
            **rollout_aux_metrics,
        }
        for schedule_name, knots in rollout_schedules.items():
            _, val_metrics = _evaluate_rollout(
                model,
                interval_dataset.val,
                endpoint_dataset.val,
                references["val"],
                teacher_model,
                rollout_schedule_name=schedule_name,
                knots=knots,
            )
            if float(val_metrics["final_state_rms_gap"]) < float(best_by_schedule[schedule_name]["val_metric"]):
                best_by_schedule[schedule_name] = {
                    "epoch": epoch,
                    "val_metric": float(val_metrics["final_state_rms_gap"]),
                    "snapshot": model.snapshot(),
                    "val_metrics": val_metrics,
                    "training_metrics": epoch_training_metrics,
                }
    return best_by_schedule


def _meanflow_twobranch_stage_control(
    epoch: int,
    *,
    model_config: MeanFlowTwoBranchStudentConfig,
) -> dict[str, Any]:
    if model_config.family_name == "meanflow_twobranch_residual_warmstart":
        in_stage_a = epoch <= int(model_config.correction_only_warmup_epochs)
        return {
            "stage_name": "correction_only_warmup" if in_stage_a else "joint_hybrid_finetune",
            "train_local_branch": not in_stage_a,
            "train_correction_branch": True,
            "teacher_loss_weight": float(model_config.teacher_loss_weight),
            "identity_loss_weight": float(model_config.identity_loss_weight),
        }
    teacher_weight, identity_weight = _training_loss_weights(
        epoch=epoch,
        teacher_loss_weight=float(model_config.teacher_loss_weight),
        identity_loss_weight=float(model_config.identity_loss_weight),
        teacher_warmup_epochs=int(model_config.teacher_warmup_epochs),
        identity_ramp_epochs=int(model_config.identity_ramp_epochs),
    )
    return {
        "stage_name": "joint_from_scratch",
        "train_local_branch": True,
        "train_correction_branch": True,
        "teacher_loss_weight": float(teacher_weight),
        "identity_loss_weight": float(identity_weight),
    }


def _fit_meanflow_twobranch_candidate_over_epochs(
    model: MeanFlowTwoBranchResidualStudent,
    *,
    model_config: MeanFlowTwoBranchStudentConfig,
    interval_dataset: FMPCIntervalDataset,
    endpoint_dataset: FMPCStudentDataset,
    references: dict[str, Any],
    teacher_model: Any,
    rollout_schedules: dict[str, tuple[int, ...]],
    aux_rollout_schedules: dict[str, tuple[int, ...]],
    batch_order_seed: int,
    candidate_seed_offset: int,
) -> dict[str, dict[str, Any]]:
    best_by_schedule: dict[str, dict[str, Any]] = {
        schedule_name: {"val_metric": np.inf, "snapshot": None}
        for schedule_name in rollout_schedules
    }
    for epoch in range(1, int(model_config.epochs) + 1):
        stage_control = _meanflow_twobranch_stage_control(epoch, model_config=model_config)
        teacher_losses: list[float] = []
        identity_losses: list[float] = []
        total_losses: list[float] = []
        identity_active_fractions: list[float] = []
        local_anchor_losses: list[float] = []
        local_teacher_update_losses: list[float] = []
        correction_teacher_update_losses: list[float] = []
        local_identity_update_losses: list[float] = []
        correction_identity_update_losses: list[float] = []
        for batch in iter_mixed_interval_batches(
            interval_dataset.train,
            int(model_config.batch_size),
            num_batches=int(model_config.batches_per_epoch),
            seed=int(batch_order_seed) + int(candidate_seed_offset) * 1000 + epoch,
            knot_focused_schedule_names=tuple(model_config.knot_focused_schedule_names),
            knot_focus_probability=float(model_config.knot_focus_probability),
        ):
            diagnostics = model.train_batch(
                batch,
                split_name="train",
                teacher_loss_weight=float(stage_control["teacher_loss_weight"]),
                identity_loss_weight=float(stage_control["identity_loss_weight"]),
                train_local_branch=bool(stage_control["train_local_branch"]),
                train_correction_branch=bool(stage_control["train_correction_branch"]),
            )
            teacher_losses.append(float(diagnostics["teacher_loss"]))
            identity_losses.append(float(diagnostics["identity_loss"]))
            total_losses.append(float(diagnostics["total_loss"]))
            identity_active_fractions.append(float(diagnostics["identity_active_fraction"]))
            local_anchor_losses.append(float(diagnostics["local_anchor_loss"]))
            if diagnostics["local_teacher_update_loss"] is not None:
                local_teacher_update_losses.append(float(diagnostics["local_teacher_update_loss"]))
            if diagnostics["correction_teacher_update_loss"] is not None:
                correction_teacher_update_losses.append(float(diagnostics["correction_teacher_update_loss"]))
            if diagnostics["local_identity_update_loss"] is not None:
                local_identity_update_losses.append(float(diagnostics["local_identity_update_loss"]))
            if diagnostics["correction_identity_update_loss"] is not None:
                correction_identity_update_losses.append(float(diagnostics["correction_identity_update_loss"]))
        rollout_aux_metrics = {
            "rollout_aux_loss": None,
            "rollout_aux_velocity_mse": None,
            "rollout_aux_intermediate_state_rms_gap": None,
            "rollout_aux_final_state_rms_gap": None,
            "rollout_aux_per_schedule": {},
        }
        if float(model_config.rollout_aux_weight) > 0.0:
            rollout_aux_metrics = _meanflow_rollout_aux_training_diagnostics(
                model,
                interval_dataset.train,
                aux_rollout_schedules=aux_rollout_schedules,
                aux_weight=float(model_config.rollout_aux_weight),
            )
        epoch_training_metrics = {
            "teacher_loss": _mean_or_none(teacher_losses),
            "identity_loss": _mean_or_none(identity_losses),
            "total_loss": _mean_or_none(total_losses),
            "effective_teacher_loss_weight": float(stage_control["teacher_loss_weight"]),
            "effective_identity_loss_weight": float(stage_control["identity_loss_weight"]),
            "identity_scope_mode": model_config.identity_scope_mode,
            "identity_active_fraction": _mean_or_none(identity_active_fractions),
            "local_anchor_loss": _mean_or_none(local_anchor_losses),
            "stage_name": str(stage_control["stage_name"]),
            "train_local_branch": bool(stage_control["train_local_branch"]),
            "train_correction_branch": bool(stage_control["train_correction_branch"]),
            "local_teacher_update_loss": _mean_or_none(local_teacher_update_losses),
            "correction_teacher_update_loss": _mean_or_none(correction_teacher_update_losses),
            "local_identity_update_loss": _mean_or_none(local_identity_update_losses),
            "correction_identity_update_loss": _mean_or_none(correction_identity_update_losses),
            **rollout_aux_metrics,
        }
        for schedule_name, knots in rollout_schedules.items():
            _, val_metrics = _evaluate_rollout(
                model,
                interval_dataset.val,
                endpoint_dataset.val,
                references["val"],
                teacher_model,
                rollout_schedule_name=schedule_name,
                knots=knots,
            )
            if float(val_metrics["final_state_rms_gap"]) < float(best_by_schedule[schedule_name]["val_metric"]):
                best_by_schedule[schedule_name] = {
                    "epoch": epoch,
                    "val_metric": float(val_metrics["final_state_rms_gap"]),
                    "snapshot": model.snapshot(),
                    "val_metrics": val_metrics,
                    "training_metrics": epoch_training_metrics,
                }
    return best_by_schedule


def _build_meanflow_mlp_rows(
    *,
    family_name: Literal[
        "teacher_only_mlp_aug",
        "meanflow_mlp_aug",
        "meanflow_mlp_residual",
        "meanflow_linear_residual",
    ],
    target_mode: Literal["u_star", "u_residual_local_field"],
    config: FMPCMeanFlowSuiteConfig,
    interval_dataset: FMPCIntervalDataset,
    endpoint_dataset: FMPCStudentDataset,
    references: dict[str, Any],
    teacher_model: Any,
    teacher_feature_bundle: FMPCIntervalTeacherFeatureBundle,
    rollout_schedules: dict[str, tuple[int, ...]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    family_best_row: dict[str, Any] | None = None
    family_best_metrics: dict[str, Any] | None = None
    family_best_record: dict[str, Any] | None = None
    candidate_counter = 0
    batches_per_epoch = int(
        config.batches_per_epoch
        if config.batches_per_epoch is not None
        else int(np.ceil(interval_dataset.train.num_samples / config.batch_size))
    )
    aux_rollout_schedules = {name: rollout_schedules[name] for name in config.rollout_aux_schedule_names}
    identity_candidates = (0.0,) if family_name == "teacher_only_mlp_aug" else config.identity_loss_weight_candidates
    identity_scope_modes = ("all_intervals",) if family_name == "teacher_only_mlp_aug" else config.identity_scope_modes
    hidden_dims_candidates = ((),) if family_name == "meanflow_linear_residual" else config.hidden_dims_candidates

    for hidden_dims in hidden_dims_candidates:
        for epochs in config.epochs_candidates:
            for eta_w in config.eta_w_candidates:
                for knot_focus_probability in config.knot_focus_probability_candidates:
                    for identity_scope_mode in identity_scope_modes:
                        for identity_loss_weight in identity_candidates:
                            for rollout_aux_weight in config.rollout_aux_weight_candidates:
                                normalization = fit_fmpc_interval_augmented_normalization(
                                    interval_dataset.train,
                                    trajectory_features=teacher_feature_bundle.train,
                                    selected_feature_names=config.feature_names,
                                    target_mode=target_mode,
                                    knot_focused_schedule_names=config.knot_focused_schedule_names,
                                    knot_focus_mixture=float(knot_focus_probability),
                                    eps=config.normalization_eps,
                                )
                                mlp_config = MeanFlowMLPStudentConfig(
                                    hidden_dims=tuple(int(value) for value in hidden_dims),
                                    family_name=family_name,
                                    target_mode=target_mode,
                                    selected_teacher_feature_names=tuple(config.feature_names),
                                    hidden_activation=config.hidden_activation,
                                    output_activation=config.output_activation,
                                    weight_scale=config.weight_scale,
                                    eta_w=float(eta_w),
                                    eta_b=float(eta_w),
                                    epochs=int(epochs),
                                    batch_size=int(config.batch_size),
                                    batches_per_epoch=batches_per_epoch,
                                    teacher_loss_weight=float(config.teacher_loss_weight),
                                    identity_loss_weight=float(identity_loss_weight),
                                    rollout_aux_weight=float(rollout_aux_weight),
                                    teacher_warmup_epochs=int(config.teacher_warmup_epochs),
                                    identity_ramp_epochs=int(config.identity_ramp_epochs),
                                    identity_scope_mode=identity_scope_mode,
                                    knot_focus_probability=float(knot_focus_probability),
                                    knot_focused_schedule_names=tuple(config.knot_focused_schedule_names),
                                    feature_aware_teacher_tangents=True,
                                    feature_tangent_epsilon=float(config.feature_tangent_epsilon),
                                )
                                model = _build_meanflow_candidate_model(
                                    interval_dataset=interval_dataset,
                                    normalization=normalization,
                                    config=mlp_config,
                                    seed=config.model_init_seed + candidate_counter,
                                    teacher_model=teacher_model,
                                    teacher_feature_bundle=teacher_feature_bundle,
                                )
                                candidate_id = (
                                    f"{family_name}_h{_serialize_hidden_dims(mlp_config.hidden_dims)}"
                                    f"_e{mlp_config.epochs}_lr{mlp_config.eta_w:g}"
                                    f"_id{mlp_config.identity_loss_weight:g}"
                                    f"_aux{mlp_config.rollout_aux_weight:g}"
                                    f"_scope{'all' if mlp_config.identity_scope_mode == 'all_intervals' else 'knot'}"
                                    f"_focus{mlp_config.knot_focus_probability:g}"
                                )
                                candidate_counter += 1
                                best_by_schedule = _fit_meanflow_candidate_over_epochs(
                                    model,
                                    model_config=mlp_config,
                                    interval_dataset=interval_dataset,
                                    endpoint_dataset=endpoint_dataset,
                                    references=references,
                                    teacher_model=teacher_model,
                                    rollout_schedules=rollout_schedules,
                                    aux_rollout_schedules=aux_rollout_schedules,
                                    batch_order_seed=config.batch_order_seed,
                                    candidate_seed_offset=candidate_counter,
                                )
                            for schedule_name, knots in rollout_schedules.items():
                                schedule_best = best_by_schedule[schedule_name]
                                if schedule_best["snapshot"] is None:
                                    raise RuntimeError(f"{candidate_id} did not record a best snapshot for {schedule_name}.")
                                row = {
                                    "config_id": candidate_id,
                                    "family": family_name,
                                    "rollout_schedule": schedule_name,
                                    "schedule_knots": "-".join(str(knot) for knot in knots),
                                    "hidden_dims": _serialize_hidden_dims(mlp_config.hidden_dims),
                                    "epochs": int(mlp_config.epochs),
                                    "eta_w": float(mlp_config.eta_w),
                                    "teacher_loss_weight": float(mlp_config.teacher_loss_weight),
                                    "identity_loss_weight": float(mlp_config.identity_loss_weight),
                                    "rollout_aux_weight": float(mlp_config.rollout_aux_weight),
                                    "identity_scope_mode": mlp_config.identity_scope_mode,
                                    "feature_contract": ",".join(mlp_config.selected_teacher_feature_names),
                                    "target_contract": "u_star" if target_mode == "u_star" else "u_res = u_star - g_s",
                                    "meanflow_identity_enabled": bool(mlp_config.identity_loss_weight > 0.0),
                                    "winner_uses_nonzero_identity_term": False,
                                    "knot_focus_probability": float(mlp_config.knot_focus_probability),
                                    "is_two_branch": False,
                                    "local_branch_input_set": "",
                                    "correction_branch_family": "",
                                    "is_family_best": False,
                                    "is_learned_family": True,
                                    "is_meanflow_family": family_name in {
                                        "meanflow_mlp_aug",
                                        "meanflow_mlp_residual",
                                        "meanflow_linear_residual",
                                    },
                                    "model_kind": "linear" if family_name == "meanflow_linear_residual" else "mlp",
                                    "is_overall_winner": False,
                                    "evaluated_on_test": False,
                                    "feature_aware_teacher_tangents": True,
                                    "zero_tangent_through_teacher_feature_block": False,
                                    "feature_tangent_epsilon": float(mlp_config.feature_tangent_epsilon),
                                    "residual_identity_includes_dg_s": bool(
                                        target_mode == "u_residual_local_field"
                                    ),
                                    "best_epoch": int(schedule_best["epoch"]),
                                    "train_teacher_loss": schedule_best["training_metrics"]["teacher_loss"],
                                    "train_identity_loss": schedule_best["training_metrics"]["identity_loss"],
                                    "train_total_loss": schedule_best["training_metrics"]["total_loss"],
                                    "effective_teacher_loss_weight": schedule_best["training_metrics"]["effective_teacher_loss_weight"],
                                    "effective_identity_loss_weight": schedule_best["training_metrics"]["effective_identity_loss_weight"],
                                    "train_identity_active_fraction": schedule_best["training_metrics"]["identity_active_fraction"],
                                    "train_rollout_aux_loss": schedule_best["training_metrics"]["rollout_aux_loss"],
                                    "train_rollout_aux_velocity_mse": schedule_best["training_metrics"]["rollout_aux_velocity_mse"],
                                    **_candidate_metric_columns("val", schedule_best["val_metrics"]),
                                }
                                rows.append(row)
                                if (
                                    family_best_row is None
                                    or float(schedule_best["val_metrics"]["final_state_rms_gap"]) < float(family_best_metrics["final_state_rms_gap"])
                                ):
                                    family_best_row = row
                                    family_best_metrics = schedule_best["val_metrics"]
                                    family_best_record = {
                                        "config": mlp_config,
                                        "normalization": normalization,
                                        "snapshot": schedule_best["snapshot"],
                                        "seed": config.model_init_seed + candidate_counter,
                                        "training_metrics": dict(schedule_best["training_metrics"]),
                                    }
    if family_best_row is None or family_best_metrics is None or family_best_record is None:
        raise RuntimeError(f"{family_name} did not produce a valid candidate.")
    family_best_row["is_family_best"] = True
    summary = _family_summary_payload(
        family=family_name,
        config_id=str(family_best_row["config_id"]),
        rollout_schedule=str(family_best_row["rollout_schedule"]),
        schedule_knots=tuple(int(value) for value in str(family_best_row["schedule_knots"]).split("-") if value != ""),
        model_config={
            **family_best_record["config"].__dict__,
            "normalization": family_best_record["normalization"].to_jsonable(),
        },
        val_metrics=family_best_metrics,
        test_metrics=None,
        best_epoch=int(family_best_row["best_epoch"]),
        training_diagnostics=family_best_record["training_metrics"],
    )
    return rows, summary, family_best_row, family_best_record


def _build_meanflow_twobranch_rows(
    *,
    family_name: Literal["meanflow_twobranch_residual", "meanflow_twobranch_residual_warmstart"],
    config: FMPCMeanFlowSuiteConfig,
    interval_dataset: FMPCIntervalDataset,
    endpoint_dataset: FMPCStudentDataset,
    references: dict[str, Any],
    teacher_model: Any,
    teacher_feature_bundle: FMPCIntervalTeacherFeatureBundle,
    rollout_schedules: dict[str, tuple[int, ...]],
    local_branch_warm_start: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    family_best_row: dict[str, Any] | None = None
    family_best_metrics: dict[str, Any] | None = None
    family_best_record: dict[str, Any] | None = None
    candidate_counter = 0
    batches_per_epoch = int(
        config.batches_per_epoch
        if config.batches_per_epoch is not None
        else int(np.ceil(interval_dataset.train.num_samples / config.batch_size))
    )
    aux_rollout_schedules = {name: rollout_schedules[name] for name in config.rollout_aux_schedule_names}
    for hidden_dims in config.hidden_dims_candidates:
        for epochs in config.epochs_candidates:
            for eta_w in config.eta_w_candidates:
                for knot_focus_probability in config.knot_focus_probability_candidates:
                    for identity_scope_mode in config.identity_scope_modes:
                        for identity_loss_weight in config.identity_loss_weight_candidates:
                            for rollout_aux_weight in config.rollout_aux_weight_candidates:
                                normalization = fit_fmpc_interval_augmented_normalization(
                                    interval_dataset.train,
                                    trajectory_features=teacher_feature_bundle.train,
                                    selected_feature_names=config.feature_names,
                                    target_mode="u_residual_local_field",
                                    knot_focused_schedule_names=config.knot_focused_schedule_names,
                                    knot_focus_mixture=float(knot_focus_probability),
                                    eps=config.normalization_eps,
                                )
                                twobranch_config = MeanFlowTwoBranchStudentConfig(
                                    correction_hidden_dims=tuple(int(value) for value in hidden_dims),
                                    family_name=family_name,
                                    selected_teacher_feature_names=tuple(config.feature_names),
                                    local_branch_feature_names=("g_s", "e_out_s", "F_s"),
                                    hidden_activation=config.hidden_activation,
                                    output_activation=config.output_activation,
                                    weight_scale=config.weight_scale,
                                    eta_w=float(eta_w),
                                    eta_b=float(eta_w),
                                    epochs=int(epochs),
                                    batch_size=int(config.batch_size),
                                    batches_per_epoch=batches_per_epoch,
                                    teacher_loss_weight=float(config.teacher_loss_weight),
                                    identity_loss_weight=float(identity_loss_weight),
                                    rollout_aux_weight=float(rollout_aux_weight),
                                    teacher_warmup_epochs=int(config.teacher_warmup_epochs),
                                    identity_ramp_epochs=int(config.identity_ramp_epochs),
                                    identity_scope_mode=identity_scope_mode,
                                    knot_focus_probability=float(knot_focus_probability),
                                    knot_focused_schedule_names=tuple(config.knot_focused_schedule_names),
                                    feature_aware_teacher_tangents=True,
                                    feature_tangent_epsilon=float(config.feature_tangent_epsilon),
                                    zero_init_correction_output_head=True,
                                    correction_only_warmup_epochs=int(
                                        config.twobranch_warmstart_correction_only_warmup_epochs
                                        if family_name == "meanflow_twobranch_residual_warmstart"
                                        else 0
                                    ),
                                    local_branch_warm_start=bool(
                                        family_name == "meanflow_twobranch_residual_warmstart"
                                    ),
                                    warm_start_source_family="phase6a1_linear_residual_baseline",
                                )
                                model = _build_meanflow_twobranch_candidate_model(
                                    interval_dataset=interval_dataset,
                                    normalization=normalization,
                                    config=twobranch_config,
                                    seed=config.model_init_seed + 50000 + candidate_counter,
                                    teacher_model=teacher_model,
                                    teacher_feature_bundle=teacher_feature_bundle,
                                    local_branch_warm_start=local_branch_warm_start
                                    if family_name == "meanflow_twobranch_residual_warmstart"
                                    else None,
                                )
                                candidate_id = (
                                    f"{family_name}_h{_serialize_hidden_dims(twobranch_config.correction_hidden_dims)}"
                                    f"_e{twobranch_config.epochs}_lr{twobranch_config.eta_w:g}"
                                    f"_id{twobranch_config.identity_loss_weight:g}"
                                    f"_aux{twobranch_config.rollout_aux_weight:g}"
                                    f"_scope{'all' if twobranch_config.identity_scope_mode == 'all_intervals' else 'knot'}"
                                    f"_focus{twobranch_config.knot_focus_probability:g}"
                                )
                                candidate_counter += 1
                                best_by_schedule = _fit_meanflow_twobranch_candidate_over_epochs(
                                    model,
                                    model_config=twobranch_config,
                                    interval_dataset=interval_dataset,
                                    endpoint_dataset=endpoint_dataset,
                                    references=references,
                                    teacher_model=teacher_model,
                                    rollout_schedules=rollout_schedules,
                                    aux_rollout_schedules=aux_rollout_schedules,
                                    batch_order_seed=config.batch_order_seed,
                                    candidate_seed_offset=50000 + candidate_counter,
                                )
                                for schedule_name, knots in rollout_schedules.items():
                                    schedule_best = best_by_schedule[schedule_name]
                                    if schedule_best["snapshot"] is None:
                                        raise RuntimeError(
                                            f"{candidate_id} did not record a best snapshot for {schedule_name}."
                                        )
                                    row = {
                                        "config_id": candidate_id,
                                        "family": family_name,
                                        "rollout_schedule": schedule_name,
                                        "schedule_knots": "-".join(str(knot) for knot in knots),
                                        "hidden_dims": _serialize_hidden_dims(twobranch_config.correction_hidden_dims),
                                        "epochs": int(twobranch_config.epochs),
                                        "eta_w": float(twobranch_config.eta_w),
                                        "teacher_loss_weight": float(twobranch_config.teacher_loss_weight),
                                        "identity_loss_weight": float(twobranch_config.identity_loss_weight),
                                        "rollout_aux_weight": float(twobranch_config.rollout_aux_weight),
                                        "identity_scope_mode": twobranch_config.identity_scope_mode,
                                        "feature_contract": ",".join(twobranch_config.selected_teacher_feature_names),
                                        "target_contract": "u_hat = u_local + u_corr; teacher target on full u_hat",
                                        "meanflow_identity_enabled": True,
                                        "winner_uses_nonzero_identity_term": False,
                                        "knot_focus_probability": float(twobranch_config.knot_focus_probability),
                                        "is_two_branch": True,
                                        "local_branch_input_set": ",".join(twobranch_config.local_branch_feature_names),
                                        "correction_branch_family": "mlp",
                                        "local_branch_warm_started": bool(twobranch_config.local_branch_warm_start),
                                        "warm_start_source": (
                                            twobranch_config.warm_start_source_family
                                            if twobranch_config.local_branch_warm_start
                                            else ""
                                        ),
                                        "correction_only_warmup_epochs": int(twobranch_config.correction_only_warmup_epochs),
                                        "is_family_best": False,
                                        "is_learned_family": True,
                                        "is_meanflow_family": True,
                                        "model_kind": "two_branch",
                                        "is_overall_winner": False,
                                        "evaluated_on_test": False,
                                        "feature_aware_teacher_tangents": True,
                                        "zero_tangent_through_teacher_feature_block": False,
                                        "feature_tangent_epsilon": float(twobranch_config.feature_tangent_epsilon),
                                        "residual_identity_includes_dg_s": True,
                                        "best_epoch": int(schedule_best["epoch"]),
                                        "train_teacher_loss": schedule_best["training_metrics"]["teacher_loss"],
                                        "train_identity_loss": schedule_best["training_metrics"]["identity_loss"],
                                        "train_total_loss": schedule_best["training_metrics"]["total_loss"],
                                        "train_local_anchor_loss": schedule_best["training_metrics"]["local_anchor_loss"],
                                        "effective_teacher_loss_weight": schedule_best["training_metrics"]["effective_teacher_loss_weight"],
                                        "effective_identity_loss_weight": schedule_best["training_metrics"]["effective_identity_loss_weight"],
                                        "train_identity_active_fraction": schedule_best["training_metrics"]["identity_active_fraction"],
                                        "train_stage_name": schedule_best["training_metrics"]["stage_name"],
                                        "train_local_branch_flag": schedule_best["training_metrics"]["train_local_branch"],
                                        "train_correction_branch_flag": schedule_best["training_metrics"]["train_correction_branch"],
                                        "train_rollout_aux_loss": schedule_best["training_metrics"]["rollout_aux_loss"],
                                        "train_rollout_aux_velocity_mse": schedule_best["training_metrics"]["rollout_aux_velocity_mse"],
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
                                            "config": twobranch_config,
                                            "normalization": normalization,
                                            "snapshot": schedule_best["snapshot"],
                                            "seed": config.model_init_seed + 50000 + candidate_counter,
                                            "training_metrics": dict(schedule_best["training_metrics"]),
                                        }
    if family_best_row is None or family_best_metrics is None or family_best_record is None:
        raise RuntimeError(f"{family_name} did not produce a valid candidate.")
    family_best_row["is_family_best"] = True
    summary = _family_summary_payload(
        family=family_name,
        config_id=str(family_best_row["config_id"]),
        rollout_schedule=str(family_best_row["rollout_schedule"]),
        schedule_knots=tuple(
            int(value) for value in str(family_best_row["schedule_knots"]).split("-") if value != ""
        ),
        model_config={
            **family_best_record["config"].__dict__,
            "normalization": family_best_record["normalization"].to_jsonable(),
        },
        val_metrics=family_best_metrics,
        test_metrics=None,
        best_epoch=int(family_best_row["best_epoch"]),
        training_diagnostics=family_best_record["training_metrics"],
    )
    return rows, summary, family_best_row, family_best_record


def _suite_config_payload(
    *,
    config: FMPCMeanFlowSuiteConfig,
    run_id: str,
    run_dir: Path,
    endpoint_dataset: FMPCStudentDataset,
    interval_dataset: FMPCIntervalDataset,
    teacher_checkpoint_loaded: bool,
    comparison_atol: float,
    rollout_schedules: dict[str, tuple[int, ...]],
) -> dict[str, Any]:
    return {
        "experiment_name": config.experiment_name,
        "run_id": run_id,
        "phase": "Phase 6",
        "stage": "phase6a3_warmstarted_two_branch_meanflow",
        "dataset_name": config.dataset_name,
        "teacher_artifact_dir": _relative_artifact_reference(run_dir, config.teacher_preparation_path),
        "teacher_manifest_path": _relative_artifact_reference(run_dir, endpoint_dataset.teacher_manifest_path),
        "teacher_checkpoint_path": _relative_artifact_reference(run_dir, endpoint_dataset.teacher_checkpoint_path),
        "teacher_checkpoint_loaded": bool(teacher_checkpoint_loaded),
        "teacher_reference_comparison_atol": float(comparison_atol),
        "teacher_steps": int(interval_dataset.teacher_steps),
        "rollout_schedules": {name: list(knots) for name, knots in rollout_schedules.items()},
        "selection_metric_name": "final_state_rms_gap",
        "selection_metric_source": "val_metric",
        "report_metric_source": "test_metric",
        "search_space": {
            "families": [
                "teacher_only_mlp_aug",
                "meanflow_mlp_aug",
                "meanflow_mlp_residual",
                "meanflow_linear_residual",
                "meanflow_twobranch_residual",
                "meanflow_twobranch_residual_warmstart",
            ],
            "feature_names": list(config.feature_names),
            "feature_tangent_epsilon": float(config.feature_tangent_epsilon),
            "knot_focus_probability_candidates": [float(value) for value in config.knot_focus_probability_candidates],
            "hidden_dims_candidates": [list(candidate) for candidate in config.hidden_dims_candidates],
            "epochs_candidates": [int(value) for value in config.epochs_candidates],
            "eta_w_candidates": [float(value) for value in config.eta_w_candidates],
            "teacher_loss_weight": float(config.teacher_loss_weight),
            "identity_loss_weight_candidates": [float(value) for value in config.identity_loss_weight_candidates],
            "identity_scope_modes": list(config.identity_scope_modes),
            "rollout_aux_weight_candidates": [float(value) for value in config.rollout_aux_weight_candidates],
            "teacher_warmup_epochs": int(config.teacher_warmup_epochs),
            "identity_ramp_epochs": int(config.identity_ramp_epochs),
            "carried_forward_phase6a1_linear_schedule_name": config.carried_forward_phase6a1_linear_schedule_name,
            "carried_forward_phase6a2_twobranch_schedule_name": config.carried_forward_phase6a2_twobranch_schedule_name,
            "twobranch_warmstart_correction_only_warmup_epochs": int(config.twobranch_warmstart_correction_only_warmup_epochs),
        },
        "jvp_contract": {
            "implementation": "manual_forward_mode_numpy",
            "tangent": {
                "dz": "g_s",
                "d_tau_s": 1.0,
                "d_tau_t": 0.0,
                "target_onehot": 0.0,
                "teacher_feature_block": "D_g teacher_features(z_s)",
            },
            "teacher_feature_block_tangent_zero": False,
            "feature_aware_teacher_tangents": True,
            "residual_identity_includes_dg_s": True,
            "two_branch_identity_applies_to_full_u_hat": True,
        },
        "teacher_recovery": {
            "allow_teacher_retrain": bool(config.allow_teacher_retrain),
            "teacher_checkpoint_required_by_default": True,
        },
    }


def run_fmpc_meanflow_suite(config: FMPCMeanFlowSuiteConfig) -> FMPCMeanFlowSuiteRunResult:
    """Run the Phase 6A.3 MeanFlow-style teacher-supervised suite on `digits`."""

    if config.dataset_name != "digits":
        raise ValueError("Phase 6A MeanFlow suite currently supports digits only.")
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
        feature_tangent_epsilon=float(config.feature_tangent_epsilon),
    )
    default_schedules = teacher_step_aligned_rollout_schedules(interval_dataset.teacher_steps)
    missing_schedule_names = [name for name in config.rollout_schedule_names if name not in default_schedules]
    if missing_schedule_names:
        raise ValueError(f"Unsupported rollout schedule names {missing_schedule_names}; expected a subset of {list(default_schedules.keys())}.")
    rollout_schedules = {name: default_schedules[name] for name in config.rollout_schedule_names}
    if config.carried_forward_phase5b2_schedule_name not in rollout_schedules:
        raise ValueError("Phase 6A compare suite must include the carried-forward Phase 5B.2 schedule.")
    if config.carried_forward_phase6a1_linear_schedule_name not in rollout_schedules:
        raise ValueError("Phase 6A.3 compare suite must include the carried-forward Phase 6A.1 linear schedule.")
    if config.carried_forward_phase6a2_twobranch_schedule_name not in rollout_schedules:
        raise ValueError("Phase 6A.3 compare suite must include the carried-forward Phase 6A.2 two-branch schedule.")

    run_id = config.resolved_run_id()
    run_dir = _prepare_run_dir(_resolve_run_dir(config.output_root, config.experiment_name, run_id, config.output_layout))

    identity_row, identity_summary = _identity_baseline_row_and_summary(
        endpoint_dataset=endpoint_dataset,
        references=references,
        teacher_model=teacher_model,
    )
    endpoint_ridge_row, endpoint_ridge_summary = _endpoint_ridge_baseline_row_and_summary(
        config=config,
        endpoint_dataset=endpoint_dataset,
        references=references,
        teacher_model=teacher_model,
    )
    phase5b2_row, phase5b2_summary = _phase5b2_baseline_row_and_summary(
        config=config,
        interval_dataset=interval_dataset,
        endpoint_dataset=endpoint_dataset,
        references=references,
        teacher_model=teacher_model,
        teacher_feature_bundle=teacher_feature_bundle,
        rollout_schedules=rollout_schedules,
    )
    phase6a1_linear_row, phase6a1_linear_summary, phase6a1_linear_record = _phase6a1_linear_baseline_row_summary_and_record(
        config=config,
        interval_dataset=interval_dataset,
        endpoint_dataset=endpoint_dataset,
        references=references,
        teacher_model=teacher_model,
        teacher_feature_bundle=teacher_feature_bundle,
        rollout_schedules=rollout_schedules,
    )
    phase6a2_twobranch_row, phase6a2_twobranch_summary = _phase6a2_twobranch_baseline_row_and_summary(
        config=config,
        interval_dataset=interval_dataset,
        endpoint_dataset=endpoint_dataset,
        references=references,
        teacher_model=teacher_model,
        teacher_feature_bundle=teacher_feature_bundle,
        rollout_schedules=rollout_schedules,
    )
    phase6a1_local_warm_start = _extract_local_branch_warm_start_from_linear_model(
        model=phase6a1_linear_record["model"],
        normalization=phase6a1_linear_record["normalization"],
        z_dim=interval_dataset.z_dim,
        target_dim=interval_dataset.target_dim,
        selected_teacher_feature_names=tuple(config.feature_names),
        local_branch_feature_names=("g_s", "e_out_s", "F_s"),
    )
    teacher_only_rows, teacher_only_summary, teacher_only_best_row, teacher_only_best_record = _build_meanflow_mlp_rows(
        family_name="teacher_only_mlp_aug",
        target_mode="u_star",
        config=config,
        interval_dataset=interval_dataset,
        endpoint_dataset=endpoint_dataset,
        references=references,
        teacher_model=teacher_model,
        teacher_feature_bundle=teacher_feature_bundle,
        rollout_schedules=rollout_schedules,
    )
    meanflow_aug_rows, meanflow_aug_summary, meanflow_aug_best_row, meanflow_aug_best_record = _build_meanflow_mlp_rows(
        family_name="meanflow_mlp_aug",
        target_mode="u_star",
        config=config,
        interval_dataset=interval_dataset,
        endpoint_dataset=endpoint_dataset,
        references=references,
        teacher_model=teacher_model,
        teacher_feature_bundle=teacher_feature_bundle,
        rollout_schedules=rollout_schedules,
    )
    meanflow_residual_rows, meanflow_residual_summary, meanflow_residual_best_row, meanflow_residual_best_record = _build_meanflow_mlp_rows(
        family_name="meanflow_mlp_residual",
        target_mode="u_residual_local_field",
        config=config,
        interval_dataset=interval_dataset,
        endpoint_dataset=endpoint_dataset,
        references=references,
        teacher_model=teacher_model,
        teacher_feature_bundle=teacher_feature_bundle,
        rollout_schedules=rollout_schedules,
    )
    meanflow_linear_rows, meanflow_linear_summary, meanflow_linear_best_row, meanflow_linear_best_record = _build_meanflow_mlp_rows(
        family_name="meanflow_linear_residual",
        target_mode="u_residual_local_field",
        config=config,
        interval_dataset=interval_dataset,
        endpoint_dataset=endpoint_dataset,
        references=references,
        teacher_model=teacher_model,
        teacher_feature_bundle=teacher_feature_bundle,
        rollout_schedules=rollout_schedules,
    )
    twobranch_rows, twobranch_summary, twobranch_best_row, twobranch_best_record = _build_meanflow_twobranch_rows(
        family_name="meanflow_twobranch_residual",
        config=config,
        interval_dataset=interval_dataset,
        endpoint_dataset=endpoint_dataset,
        references=references,
        teacher_model=teacher_model,
        teacher_feature_bundle=teacher_feature_bundle,
        rollout_schedules=rollout_schedules,
    )
    twobranch_warm_rows, twobranch_warm_summary, twobranch_warm_best_row, twobranch_warm_best_record = _build_meanflow_twobranch_rows(
        family_name="meanflow_twobranch_residual_warmstart",
        config=config,
        interval_dataset=interval_dataset,
        endpoint_dataset=endpoint_dataset,
        references=references,
        teacher_model=teacher_model,
        teacher_feature_bundle=teacher_feature_bundle,
        rollout_schedules=rollout_schedules,
        local_branch_warm_start=phase6a1_local_warm_start,
    )

    learned_family_best_rows = [
        teacher_only_best_row,
        meanflow_aug_best_row,
        meanflow_residual_best_row,
        meanflow_linear_best_row,
        twobranch_best_row,
        twobranch_warm_best_row,
    ]
    learned_winner_row = min(learned_family_best_rows, key=lambda row: float(row["val_final_state_rms_gap"]))
    winner_family = str(learned_winner_row["family"])
    winner_schedule_name = str(learned_winner_row["rollout_schedule"])
    winner_knots = tuple(int(value) for value in str(learned_winner_row["schedule_knots"]).split("-") if value != "")
    winner_record = (
        teacher_only_best_record
        if winner_family == "teacher_only_mlp_aug"
        else meanflow_aug_best_record
        if winner_family == "meanflow_mlp_aug"
        else meanflow_residual_best_record
        if winner_family == "meanflow_mlp_residual"
        else meanflow_linear_best_record
        if winner_family == "meanflow_linear_residual"
        else twobranch_best_record
        if winner_family == "meanflow_twobranch_residual"
        else twobranch_warm_best_record
    )
    if winner_family in {"meanflow_twobranch_residual", "meanflow_twobranch_residual_warmstart"}:
        winner_model = _build_meanflow_twobranch_candidate_model(
            interval_dataset=interval_dataset,
            normalization=winner_record["normalization"],
            config=winner_record["config"],
            seed=int(winner_record["seed"]),
            teacher_model=teacher_model,
            teacher_feature_bundle=teacher_feature_bundle,
            local_branch_warm_start=phase6a1_local_warm_start
            if winner_family == "meanflow_twobranch_residual_warmstart"
            else None,
        )
    else:
        winner_model = _build_meanflow_candidate_model(
            interval_dataset=interval_dataset,
            normalization=winner_record["normalization"],
            config=winner_record["config"],
            seed=int(winner_record["seed"]),
            teacher_model=teacher_model,
            teacher_feature_bundle=teacher_feature_bundle,
        )
    winner_model.restore(winner_record["snapshot"])
    _, winner_val_metrics = _evaluate_rollout(
        winner_model,
        interval_dataset.val,
        endpoint_dataset.val,
        references["val"],
        teacher_model,
        rollout_schedule_name=winner_schedule_name,
        knots=winner_knots,
    )
    _, winner_test_metrics = _evaluate_rollout(
        winner_model,
        interval_dataset.test,
        endpoint_dataset.test,
        references["test"],
        teacher_model,
        rollout_schedule_name=winner_schedule_name,
        knots=winner_knots,
    )

    all_rows = [
        identity_row,
        endpoint_ridge_row,
        phase5b2_row,
        phase6a1_linear_row,
        phase6a2_twobranch_row,
        *teacher_only_rows,
        *meanflow_aug_rows,
        *meanflow_residual_rows,
        *meanflow_linear_rows,
        *twobranch_rows,
        *twobranch_warm_rows,
    ]
    for row in all_rows:
        if str(row["family"]) == winner_family and str(row["config_id"]) == str(learned_winner_row["config_id"]) and str(row["rollout_schedule"]) == winner_schedule_name:
            row["is_overall_winner"] = True
            row["evaluated_on_test"] = True
            row["winner_uses_nonzero_identity_term"] = bool(float(row["identity_loss_weight"] or 0.0) > 0.0)
            row.update(_candidate_metric_columns("test", winner_test_metrics))

    if winner_family == "teacher_only_mlp_aug":
        teacher_only_summary["test"] = winner_test_metrics
    elif winner_family == "meanflow_mlp_aug":
        meanflow_aug_summary["test"] = winner_test_metrics
    elif winner_family == "meanflow_mlp_residual":
        meanflow_residual_summary["test"] = winner_test_metrics
    elif winner_family == "meanflow_linear_residual":
        meanflow_linear_summary["test"] = winner_test_metrics
    elif winner_family == "meanflow_twobranch_residual":
        twobranch_summary["test"] = winner_test_metrics
    else:
        twobranch_warm_summary["test"] = winner_test_metrics

    summary = {
        "experiment_name": config.experiment_name,
        "run_id": run_id,
        "phase": "Phase 6",
        "stage": "phase6a3_warmstarted_two_branch_meanflow",
        "dataset_name": config.dataset_name,
        "teacher_artifact_dir": _relative_artifact_reference(run_dir, config.teacher_preparation_path),
        "teacher_manifest_path": _relative_artifact_reference(run_dir, endpoint_dataset.teacher_manifest_path),
        "teacher_checkpoint_path": _relative_artifact_reference(run_dir, endpoint_dataset.teacher_checkpoint_path),
        "teacher_checkpoint_loaded": not used_teacher_retrain_fallback,
        "teacher_steps": int(interval_dataset.teacher_steps),
        "schedule_knots": {name: list(knots) for name, knots in rollout_schedules.items()},
        "selection_metric_source": "val_metric",
        "report_metric_source": "test_metric",
        "meanflow_contract": {
            "u_star_definition": "u_star = (z_t - z_s) / (tau_t - tau_s)",
            "g_s_definition": "current teacher local field at z_s in normalized-time units",
            "identity_formula": "u = g_s + dt * d/dtau_s u",
            "d_dtaus_formula": "J_z u * g_s + partial_tau_s u",
            "manual_numpy_jvp": True,
            "identity_target_uses_stop_gradient": True,
            "feature_aware_teacher_tangents": True,
            "teacher_feature_block_zero_tangent": False,
            "residual_identity_includes_dg_s": True,
            "feature_tangent_epsilon": float(config.feature_tangent_epsilon),
            "two_branch_identity_applies_to_full_u_hat": True,
        },
        "identity_baseline": identity_summary,
        "phase5a_endpoint_ridge_baseline": endpoint_ridge_summary,
        "phase5b2_interval_ridge_residual_baseline": phase5b2_summary,
        "phase6a1_linear_residual_baseline": phase6a1_linear_summary,
        "phase6a2_twobranch_residual_baseline": phase6a2_twobranch_summary,
        "teacher_only_mlp_aug": teacher_only_summary,
        "meanflow_mlp_aug": meanflow_aug_summary,
        "meanflow_mlp_residual": meanflow_residual_summary,
        "meanflow_linear_residual": meanflow_linear_summary,
        "meanflow_twobranch_residual": twobranch_summary,
        "meanflow_twobranch_residual_warmstart": twobranch_warm_summary,
        "winner": {
            "family": winner_family,
            "config_id": str(learned_winner_row["config_id"]),
            "rollout_schedule": winner_schedule_name,
            "schedule_knots": list(winner_knots),
            "model_config": winner_model.to_jsonable(),
            "best_epoch": int(learned_winner_row["best_epoch"]),
            "training_diagnostics": winner_record["training_metrics"],
            "val": winner_val_metrics,
            "test": winner_test_metrics,
            "winner_is_meanflow_family": bool(
                winner_family
                in {
                    "meanflow_mlp_aug",
                    "meanflow_mlp_residual",
                    "meanflow_linear_residual",
                    "meanflow_twobranch_residual",
                    "meanflow_twobranch_residual_warmstart",
                }
            ),
            "winner_is_linear_family": bool(winner_family == "meanflow_linear_residual"),
            "winner_is_mlp_family": bool(
                winner_family in {"teacher_only_mlp_aug", "meanflow_mlp_aug", "meanflow_mlp_residual"}
            ),
            "winner_is_two_branch_family": bool(
                winner_family in {"meanflow_twobranch_residual", "meanflow_twobranch_residual_warmstart"}
            ),
            "winner_uses_nonzero_identity_term": bool(float(winner_record["config"].identity_loss_weight) > 0.0),
            "winner_is_true_multistep_meanflow_candidate": bool(
                winner_family
                in {
                    "meanflow_mlp_aug",
                    "meanflow_mlp_residual",
                    "meanflow_linear_residual",
                    "meanflow_twobranch_residual",
                    "meanflow_twobranch_residual_warmstart",
                }
                and winner_schedule_name in {"2-step", "3-step"}
            ),
            "winner_is_monolithic": bool(
                winner_family in {"teacher_only_mlp_aug", "meanflow_mlp_aug", "meanflow_mlp_residual", "meanflow_linear_residual"}
            ),
            "winner_is_two_branch": bool(
                winner_family in {"meanflow_twobranch_residual", "meanflow_twobranch_residual_warmstart"}
            ),
            "winner_local_branch_input_set": (
                list(winner_record["config"].local_branch_feature_names)
                if winner_family in {"meanflow_twobranch_residual", "meanflow_twobranch_residual_warmstart"}
                else []
            ),
            "winner_correction_branch_family": "mlp"
            if winner_family in {"meanflow_twobranch_residual", "meanflow_twobranch_residual_warmstart"}
            else "",
            "winner_local_branch_warm_started": bool(
                winner_family == "meanflow_twobranch_residual_warmstart"
            ),
            "winner_local_branch_warm_start_source": (
                winner_record["config"].warm_start_source_family
                if winner_family == "meanflow_twobranch_residual_warmstart"
                else ""
            ),
            "winner_feature_aware_teacher_tangents_enabled": bool(winner_record["config"].feature_aware_teacher_tangents),
            "winner_residual_identity_includes_dg_s": True,
            "winner_beats_identity_on_val_metric": bool(float(winner_val_metrics["final_state_rms_gap"]) < float(identity_summary["val"]["final_state_rms_gap"])),
            "winner_beats_identity_on_test_metric": bool(float(winner_test_metrics["final_state_rms_gap"]) < float(identity_summary["test"]["final_state_rms_gap"])),
            "winner_beats_phase5a_endpoint_ridge_on_val_metric": bool(float(winner_val_metrics["final_state_rms_gap"]) < float(endpoint_ridge_summary["val"]["final_state_rms_gap"])),
            "winner_beats_phase5a_endpoint_ridge_on_test_metric": bool(float(winner_test_metrics["final_state_rms_gap"]) < float(endpoint_ridge_summary["test"]["final_state_rms_gap"])),
            "winner_beats_phase5b2_on_val_metric": bool(float(winner_val_metrics["final_state_rms_gap"]) < float(phase5b2_summary["val"]["final_state_rms_gap"])),
            "winner_beats_phase5b2_on_test_metric": bool(float(winner_test_metrics["final_state_rms_gap"]) < float(phase5b2_summary["test"]["final_state_rms_gap"])),
            "winner_beats_phase6a1_linear_on_val_metric": bool(float(winner_val_metrics["final_state_rms_gap"]) < float(phase6a1_linear_summary["val"]["final_state_rms_gap"])),
            "winner_beats_phase6a1_linear_on_test_metric": bool(float(winner_test_metrics["final_state_rms_gap"]) < float(phase6a1_linear_summary["test"]["final_state_rms_gap"])),
            "winner_beats_phase6a2_twobranch_on_val_metric": bool(float(winner_val_metrics["final_state_rms_gap"]) < float(phase6a2_twobranch_summary["val"]["final_state_rms_gap"])),
            "winner_beats_phase6a2_twobranch_on_test_metric": bool(float(winner_test_metrics["final_state_rms_gap"]) < float(phase6a2_twobranch_summary["test"]["final_state_rms_gap"])),
        },
    }

    config_payload = _suite_config_payload(
        config=config,
        run_id=run_id,
        run_dir=run_dir,
        endpoint_dataset=endpoint_dataset,
        interval_dataset=interval_dataset,
        teacher_checkpoint_loaded=not used_teacher_retrain_fallback,
        comparison_atol=comparison_atol,
        rollout_schedules=rollout_schedules,
    )
    _write_json(run_dir / "config.json", config_payload)
    _write_candidates_csv(run_dir / "candidates.csv", all_rows)
    _write_json(run_dir / "summary.json", summary)
    return FMPCMeanFlowSuiteRunResult(run_dir=run_dir, config=config_payload, candidates=all_rows, summary=summary)
