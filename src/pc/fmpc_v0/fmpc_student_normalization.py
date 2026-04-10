from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .fmpc_student_data import FMPCStudentSplit, build_fmpc_student_inputs


def _as_batch_first(name: str, array: np.ndarray) -> np.ndarray:
    array_float = np.asarray(array, dtype=np.float64)
    if array_float.ndim != 2:
        raise ValueError(f"{name} must be shaped (batch, features).")
    return array_float


@dataclass(frozen=True)
class FMPCStudentNormalizationStats:
    """Train-split normalization statistics for the endpoint FMPC student.

    Shape contract:
    - `z0_mean`: `(z_dim,)`
    - `z0_std`: `(z_dim,)`
    - `delta_z_mean`: `(z_dim,)`
    - `delta_z_std`: `(z_dim,)`

    All transforms are batch-first and preserve `float64`.
    """

    z0_mean: np.ndarray
    z0_std: np.ndarray
    delta_z_mean: np.ndarray
    delta_z_std: np.ndarray
    eps: float = 1e-8

    def __post_init__(self) -> None:
        object.__setattr__(self, "z0_mean", np.asarray(self.z0_mean, dtype=np.float64))
        object.__setattr__(self, "z0_std", np.asarray(self.z0_std, dtype=np.float64))
        object.__setattr__(self, "delta_z_mean", np.asarray(self.delta_z_mean, dtype=np.float64))
        object.__setattr__(self, "delta_z_std", np.asarray(self.delta_z_std, dtype=np.float64))
        if self.z0_mean.ndim != 1 or self.z0_std.ndim != 1:
            raise ValueError("z0 normalization statistics must be rank-1 arrays.")
        if self.delta_z_mean.ndim != 1 or self.delta_z_std.ndim != 1:
            raise ValueError("delta_z normalization statistics must be rank-1 arrays.")
        if self.z0_mean.shape != self.z0_std.shape:
            raise ValueError("z0_mean and z0_std must share the same shape.")
        if self.delta_z_mean.shape != self.delta_z_std.shape:
            raise ValueError("delta_z_mean and delta_z_std must share the same shape.")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")

    @property
    def z_dim(self) -> int:
        return int(self.z0_mean.shape[0])

    def _safe_z0_std(self) -> np.ndarray:
        return np.maximum(self.z0_std, self.eps)

    def _safe_delta_z_std(self) -> np.ndarray:
        return np.maximum(self.delta_z_std, self.eps)

    def transform_z0(self, z0: np.ndarray) -> np.ndarray:
        """Standardize `z0` shaped `(batch, z_dim)` using train-split statistics."""
        z0_array = _as_batch_first("z0", z0)
        if z0_array.shape[1] != self.z_dim:
            raise ValueError(f"z0 feature dimension must be {self.z_dim}.")
        return ((z0_array - self.z0_mean) / self._safe_z0_std()).astype(np.float64, copy=False)

    def transform_delta_z(self, delta_z: np.ndarray) -> np.ndarray:
        """Standardize `delta_z` shaped `(batch, z_dim)` using train-split statistics."""
        delta_z_array = _as_batch_first("delta_z", delta_z)
        if delta_z_array.shape[1] != self.delta_z_mean.shape[0]:
            raise ValueError(f"delta_z feature dimension must be {self.delta_z_mean.shape[0]}.")
        return ((delta_z_array - self.delta_z_mean) / self._safe_delta_z_std()).astype(
            np.float64,
            copy=False,
        )

    def inverse_delta_z(self, delta_z_normalized: np.ndarray) -> np.ndarray:
        """Inverse-transform normalized `delta_z` predictions shaped `(batch, z_dim)`."""
        delta_z_array = _as_batch_first("delta_z_normalized", delta_z_normalized)
        if delta_z_array.shape[1] != self.delta_z_mean.shape[0]:
            raise ValueError(f"delta_z_normalized feature dimension must be {self.delta_z_mean.shape[0]}.")
        return (delta_z_array * self._safe_delta_z_std() + self.delta_z_mean).astype(
            np.float64,
            copy=False,
        )

    def transform_inputs(self, z0: np.ndarray, target_onehot: np.ndarray) -> np.ndarray:
        """Return `concat([normalized_z0, target_onehot])` with batch-first `float64` semantics."""
        z0_normalized = self.transform_z0(z0)
        targets = _as_batch_first("target_onehot", target_onehot)
        if z0_normalized.shape[0] != targets.shape[0]:
            raise ValueError("z0 and target_onehot must share the same batch dimension.")
        return build_fmpc_student_inputs(z0_normalized, targets)

    def transform_split_inputs(self, split: FMPCStudentSplit) -> np.ndarray:
        """Return normalized student inputs for one validated FMPC student split."""
        return self.transform_inputs(split.z0, split.target_onehot)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "z0_mean": self.z0_mean.tolist(),
            "z0_std": self.z0_std.tolist(),
            "delta_z_mean": self.delta_z_mean.tolist(),
            "delta_z_std": self.delta_z_std.tolist(),
            "eps": float(self.eps),
            "z0_normalized": True,
            "target_onehot_normalized": False,
            "delta_z_normalized": True,
        }


def fit_fmpc_student_normalization(
    train_split: FMPCStudentSplit,
    *,
    eps: float = 1e-8,
) -> FMPCStudentNormalizationStats:
    """Fit train-split normalization statistics for `z0` and `delta_z`.

    Shape contract:
    - `train_split.z0`: `(batch, z_dim)`
    - `train_split.delta_z`: `(batch, z_dim)`
    """

    z0 = _as_batch_first("train_split.z0", train_split.z0)
    delta_z = _as_batch_first("train_split.delta_z", train_split.delta_z)
    if z0.shape != delta_z.shape:
        raise ValueError("train_split.z0 and train_split.delta_z must share the same shape.")
    return FMPCStudentNormalizationStats(
        z0_mean=np.mean(z0, axis=0),
        z0_std=np.std(z0, axis=0),
        delta_z_mean=np.mean(delta_z, axis=0),
        delta_z_std=np.std(delta_z, axis=0),
        eps=float(eps),
    )
