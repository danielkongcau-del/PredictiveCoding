from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .fmpc_interval_data import (
    FMPCIntervalSplit,
    build_fmpc_interval_inputs,
    compute_interval_velocity_target,
    iter_all_interval_blocks,
    iter_weighted_interval_blocks,
)
from .fmpc_interval_features import FMPCIntervalTeacherTrajectoryFeatures


def _as_batch_first(name: str, array: np.ndarray) -> np.ndarray:
    array_float = np.asarray(array, dtype=np.float64)
    if array_float.ndim != 2:
        raise ValueError(f"{name} must be shaped (batch, features).")
    return array_float


@dataclass(frozen=True)
class FMPCIntervalNormalizationStats:
    """Train-split normalization statistics for Phase 5B interval students.

    Shape contract:
    - `z_state_mean`: `(z_dim,)`
    - `z_state_std`: `(z_dim,)`
    - `u_mean`: `(z_dim,)`
    - `u_std`: `(z_dim,)`
    """

    z_state_mean: np.ndarray
    z_state_std: np.ndarray
    u_mean: np.ndarray
    u_std: np.ndarray
    teacher_feature_mean: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float64)
    )
    teacher_feature_std: np.ndarray = field(
        default_factory=lambda: np.ones((0,), dtype=np.float64)
    )
    teacher_feature_names: tuple[str, ...] = ()
    eps: float = 1e-8

    def __post_init__(self) -> None:
        object.__setattr__(self, "z_state_mean", np.asarray(self.z_state_mean, dtype=np.float64))
        object.__setattr__(self, "z_state_std", np.asarray(self.z_state_std, dtype=np.float64))
        object.__setattr__(self, "u_mean", np.asarray(self.u_mean, dtype=np.float64))
        object.__setattr__(self, "u_std", np.asarray(self.u_std, dtype=np.float64))
        object.__setattr__(self, "teacher_feature_mean", np.asarray(self.teacher_feature_mean, dtype=np.float64))
        object.__setattr__(self, "teacher_feature_std", np.asarray(self.teacher_feature_std, dtype=np.float64))
        object.__setattr__(self, "teacher_feature_names", tuple(self.teacher_feature_names))
        if self.z_state_mean.ndim != 1 or self.z_state_std.ndim != 1:
            raise ValueError("z_state normalization statistics must be rank-1 arrays.")
        if self.u_mean.ndim != 1 or self.u_std.ndim != 1:
            raise ValueError("u normalization statistics must be rank-1 arrays.")
        if self.teacher_feature_mean.ndim != 1 or self.teacher_feature_std.ndim != 1:
            raise ValueError("teacher_feature normalization statistics must be rank-1 arrays.")
        if self.z_state_mean.shape != self.z_state_std.shape:
            raise ValueError("z_state_mean and z_state_std must share the same shape.")
        if self.u_mean.shape != self.u_std.shape:
            raise ValueError("u_mean and u_std must share the same shape.")
        if self.teacher_feature_mean.shape != self.teacher_feature_std.shape:
            raise ValueError("teacher_feature_mean and teacher_feature_std must share the same shape.")
        if self.teacher_feature_mean.shape[0] > 0 and len(self.teacher_feature_names) == 0:
            raise ValueError(
                "teacher_feature_names must be provided when teacher_feature normalization statistics are non-empty."
            )
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")

    @property
    def z_dim(self) -> int:
        return int(self.z_state_mean.shape[0])

    def _safe_z_state_std(self) -> np.ndarray:
        return np.maximum(self.z_state_std, self.eps)

    def _safe_u_std(self) -> np.ndarray:
        return np.maximum(self.u_std, self.eps)

    def _safe_teacher_feature_std(self) -> np.ndarray:
        return np.maximum(self.teacher_feature_std, self.eps)

    def transform_z_state(self, z_state: np.ndarray) -> np.ndarray:
        """Standardize interval states shaped `(batch, z_dim)` using train-only stats."""

        z_state_array = _as_batch_first("z_state", z_state)
        if z_state_array.shape[1] != self.z_dim:
            raise ValueError(f"z_state feature dimension must be {self.z_dim}.")
        return ((z_state_array - self.z_state_mean) / self._safe_z_state_std()).astype(np.float64, copy=False)

    def transform_u(self, u: np.ndarray) -> np.ndarray:
        """Standardize interval targets shaped `(batch, z_dim)` using train-only stats."""

        u_array = _as_batch_first("u", u)
        if u_array.shape[1] != self.u_mean.shape[0]:
            raise ValueError(f"u feature dimension must be {self.u_mean.shape[0]}.")
        return ((u_array - self.u_mean) / self._safe_u_std()).astype(np.float64, copy=False)

    def inverse_u(self, u_normalized: np.ndarray) -> np.ndarray:
        """Inverse-transform normalized interval targets shaped `(batch, z_dim)`."""

        u_array = _as_batch_first("u_normalized", u_normalized)
        if u_array.shape[1] != self.u_mean.shape[0]:
            raise ValueError(f"u_normalized feature dimension must be {self.u_mean.shape[0]}.")
        return (u_array * self._safe_u_std() + self.u_mean).astype(np.float64, copy=False)

    def transform_teacher_features(self, teacher_features: np.ndarray) -> np.ndarray:
        """Standardize optional teacher-state feature blocks shaped `(batch, feature_dim)`."""

        teacher_features_array = _as_batch_first("teacher_features", teacher_features)
        if teacher_features_array.shape[1] != self.teacher_feature_mean.shape[0]:
            raise ValueError(
                f"teacher_features feature dimension must be {self.teacher_feature_mean.shape[0]}."
            )
        if self.teacher_feature_mean.shape[0] == 0:
            return teacher_features_array.astype(np.float64, copy=False)
        return (
            (teacher_features_array - self.teacher_feature_mean) / self._safe_teacher_feature_std()
        ).astype(np.float64, copy=False)

    def transform_inputs(
        self,
        z_s: np.ndarray,
        target_onehot: np.ndarray,
        tau_s: np.ndarray | float,
        tau_t: np.ndarray | float,
        *,
        teacher_features: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return normalized interval inputs.

        Shape contract:
        - without `teacher_features`:
          `concat([normalized_z_s, target_onehot, tau_s, tau_t])`
        - with `teacher_features`:
          `concat([normalized_z_s, target_onehot, tau_s, tau_t, normalized_teacher_features])`
        """

        z_state_normalized = self.transform_z_state(z_s)
        base_inputs = build_fmpc_interval_inputs(z_state_normalized, target_onehot, tau_s, tau_t)
        if teacher_features is None:
            return base_inputs
        normalized_teacher_features = self.transform_teacher_features(teacher_features)
        return np.concatenate([base_inputs, normalized_teacher_features], axis=1).astype(np.float64, copy=False)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "z_state_mean": self.z_state_mean.tolist(),
            "z_state_std": self.z_state_std.tolist(),
            "u_mean": self.u_mean.tolist(),
            "u_std": self.u_std.tolist(),
            "teacher_feature_mean": self.teacher_feature_mean.tolist(),
            "teacher_feature_std": self.teacher_feature_std.tolist(),
            "teacher_feature_names": list(self.teacher_feature_names),
            "teacher_feature_dim": int(self.teacher_feature_mean.shape[0]),
            "eps": float(self.eps),
            "z_state_normalized": True,
            "target_onehot_normalized": False,
            "tau_normalized": False,
            "teacher_features_normalized": bool(self.teacher_feature_mean.shape[0] > 0),
            "u_normalized": True,
        }


def fit_fmpc_interval_normalization(
    train_split: FMPCIntervalSplit,
    *,
    eps: float = 1e-8,
) -> FMPCIntervalNormalizationStats:
    """Fit train-only normalization statistics for interval states and interval targets."""

    states = np.asarray(train_split.z_trajectory, dtype=np.float64)
    if states.ndim != 3:
        raise ValueError("train_split.z_trajectory must be shaped (batch, step, z_dim).")
    z_state_matrix = states.reshape(-1, states.shape[-1])

    weighted_sum_u = np.zeros(train_split.z_dim, dtype=np.float64)
    weighted_sum_sq_u = np.zeros(train_split.z_dim, dtype=np.float64)
    total_weight = 0.0
    for block in iter_all_interval_blocks(train_split):
        u_star = np.asarray(block["u_star"], dtype=np.float64)
        pair_weight = float(block["pair_weight"])
        weighted_sum_u += pair_weight * np.sum(u_star, axis=0)
        weighted_sum_sq_u += pair_weight * np.sum(u_star**2, axis=0)
        total_weight += pair_weight * float(u_star.shape[0])
    if total_weight <= 0.0:
        raise RuntimeError("Interval normalization requires positive total pair weight.")
    u_mean = weighted_sum_u / total_weight
    u_variance = np.maximum(weighted_sum_sq_u / total_weight - u_mean**2, 0.0)
    return FMPCIntervalNormalizationStats(
        z_state_mean=np.mean(z_state_matrix, axis=0),
        z_state_std=np.std(z_state_matrix, axis=0),
        u_mean=u_mean,
        u_std=np.sqrt(u_variance).astype(np.float64, copy=False),
        eps=float(eps),
    )


def fit_fmpc_interval_augmented_normalization(
    train_split: FMPCIntervalSplit,
    *,
    trajectory_features: FMPCIntervalTeacherTrajectoryFeatures,
    selected_feature_names: tuple[str, ...],
    target_mode: str,
    knot_focused_schedule_names: tuple[str, ...] = ("2-step", "3-step"),
    knot_focus_mixture: float = 0.0,
    eps: float = 1e-8,
) -> FMPCIntervalNormalizationStats:
    """Fit train-only normalization for augmented interval students.

    Supported target semantics:
    - `u_star`
    - `u_res = u_star - g_s`
    """

    if target_mode not in {"u_star", "u_residual_local_field"}:
        raise ValueError(
            "target_mode must be 'u_star' or 'u_residual_local_field'."
        )
    states = np.asarray(train_split.z_trajectory, dtype=np.float64)
    if states.ndim != 3:
        raise ValueError("train_split.z_trajectory must be shaped (batch, step, z_dim).")
    z_state_matrix = states.reshape(-1, states.shape[-1])

    weighted_sum_u = np.zeros(train_split.z_dim, dtype=np.float64)
    weighted_sum_sq_u = np.zeros(train_split.z_dim, dtype=np.float64)
    weighted_sum_features: np.ndarray | None = None
    weighted_sum_sq_features: np.ndarray | None = None
    total_weight = 0.0

    for block in iter_weighted_interval_blocks(
        train_split,
        knot_focused_schedule_names=knot_focused_schedule_names,
        knot_focus_mixture=knot_focus_mixture,
    ):
        state_features = trajectory_features.step_features(
            int(block["source_index"]),
            selected_feature_names=selected_feature_names,
        )
        feature_matrix = state_features.feature_matrix(selected_feature_names)
        if weighted_sum_features is None:
            weighted_sum_features = np.zeros(feature_matrix.shape[1], dtype=np.float64)
            weighted_sum_sq_features = np.zeros(feature_matrix.shape[1], dtype=np.float64)
        pair_weight = float(block["pair_weight"])
        u_target = np.asarray(block["u_star"], dtype=np.float64)
        if target_mode == "u_residual_local_field":
            u_target = (u_target - state_features.g_s).astype(np.float64, copy=False)
        weighted_sum_u += pair_weight * np.sum(u_target, axis=0)
        weighted_sum_sq_u += pair_weight * np.sum(u_target**2, axis=0)
        weighted_sum_features += pair_weight * np.sum(feature_matrix, axis=0)
        weighted_sum_sq_features += pair_weight * np.sum(feature_matrix**2, axis=0)
        total_weight += pair_weight * float(u_target.shape[0])

    if total_weight <= 0.0:
        raise RuntimeError("Augmented interval normalization requires positive total pair weight.")
    if weighted_sum_features is None or weighted_sum_sq_features is None:
        raise RuntimeError("Augmented interval normalization requires at least one teacher feature.")
    u_mean = weighted_sum_u / total_weight
    u_variance = np.maximum(weighted_sum_sq_u / total_weight - u_mean**2, 0.0)
    teacher_feature_mean = weighted_sum_features / total_weight
    teacher_feature_variance = np.maximum(
        weighted_sum_sq_features / total_weight - teacher_feature_mean**2,
        0.0,
    )
    return FMPCIntervalNormalizationStats(
        z_state_mean=np.mean(z_state_matrix, axis=0),
        z_state_std=np.std(z_state_matrix, axis=0),
        u_mean=u_mean,
        u_std=np.sqrt(u_variance).astype(np.float64, copy=False),
        teacher_feature_mean=teacher_feature_mean.astype(np.float64, copy=False),
        teacher_feature_std=np.sqrt(teacher_feature_variance).astype(np.float64, copy=False),
        teacher_feature_names=tuple(selected_feature_names),
        eps=float(eps),
    )


def transform_interval_block_targets(
    normalization: FMPCIntervalNormalizationStats,
    *,
    z_s: np.ndarray,
    z_t: np.ndarray,
    tau_s: np.ndarray | float,
    tau_t: np.ndarray | float,
) -> np.ndarray:
    """Return normalized `u_star` for one interval block in batch-first form."""

    return normalization.transform_u(compute_interval_velocity_target(z_s, z_t, tau_s, tau_t))
