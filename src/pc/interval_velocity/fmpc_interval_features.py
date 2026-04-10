from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..energy import compute_cache
from ..inference import build_clamped_mask, initialize_states, run_inference
from ..models import PCNetwork
from ..state_io import flatten_hidden_states, unflatten_hidden_states
from ..toy_data import SupervisedDataSplit


_SUPPORTED_TEACHER_FEATURE_NAMES = ("y_hat_s", "e_out_s", "g_s", "F_s")


def _as_batch_first(name: str, array: np.ndarray) -> np.ndarray:
    array_float = np.asarray(array, dtype=np.float64)
    if array_float.ndim != 2:
        raise ValueError(f"{name} must be shaped (batch, features).")
    return array_float


def _teacher_split_arrays(
    split: SupervisedDataSplit,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if split_name == "train":
        return (
            np.asarray(split.x_train, dtype=np.float64),
            np.asarray(split.y_train, dtype=np.float64),
            np.asarray(split.metadata["train_indices"], dtype=np.int64),
        )
    if split_name == "val":
        return (
            np.asarray(split.x_val, dtype=np.float64),
            np.asarray(split.y_val, dtype=np.float64),
            np.asarray(split.metadata["val_indices"], dtype=np.int64),
        )
    if split_name == "test":
        return (
            np.asarray(split.x_test, dtype=np.float64),
            np.asarray(split.y_test, dtype=np.float64),
            np.asarray(split.metadata["test_indices"], dtype=np.int64),
        )
    raise ValueError(f"Unsupported split_name '{split_name}'.")


def _per_sample_energy_column(cache: Any, teacher_model: PCNetwork) -> np.ndarray:
    batch_size = int(cache.errors[1].shape[0])
    energies = np.zeros((batch_size,), dtype=np.float64)
    for layer_index, layer in enumerate(teacher_model.layers, start=1):
        error = np.asarray(cache.errors[layer_index], dtype=np.float64)
        energies += 0.5 * np.sum(error * error, axis=1) / float(layer.sigma2)
    return energies.reshape(batch_size, 1)


def _state_list_from_hidden_state(
    teacher_model: PCNetwork,
    x: np.ndarray,
    y: np.ndarray,
    z_s: np.ndarray,
) -> tuple[list[np.ndarray], list[bool]]:
    clamped_mask = build_clamped_mask(len(teacher_model.layers) + 1, mode="train")
    states_template = initialize_states(
        teacher_model.layers,
        np.asarray(x, dtype=np.float64),
        y=np.asarray(y, dtype=np.float64),
        init=teacher_model.state_init,
        mode="train",
    )
    states = unflatten_hidden_states(
        np.asarray(z_s, dtype=np.float64),
        states_template,
        clamped_mask,
    )
    return states, clamped_mask


def _validate_feature_names(selected_feature_names: tuple[str, ...]) -> tuple[str, ...]:
    invalid = [name for name in selected_feature_names if name not in _SUPPORTED_TEACHER_FEATURE_NAMES]
    if invalid:
        raise ValueError(
            f"Unsupported teacher feature names {invalid}; expected a subset of {_SUPPORTED_TEACHER_FEATURE_NAMES}."
        )
    if len(selected_feature_names) != len(set(selected_feature_names)):
        raise ValueError("selected_feature_names must not contain duplicates.")
    return tuple(selected_feature_names)


def _validate_fd_epsilon(fd_epsilon: float) -> float:
    epsilon = float(fd_epsilon)
    if epsilon <= 0.0:
        raise ValueError("fd_epsilon must be positive.")
    return epsilon


def _feature_chunks(
    *,
    selected_feature_names: tuple[str, ...],
    y_hat_s: np.ndarray,
    e_out_s: np.ndarray,
    g_s: np.ndarray,
    F_s: np.ndarray,
) -> list[np.ndarray]:
    feature_names = _validate_feature_names(selected_feature_names)
    chunks: list[np.ndarray] = []
    for name in feature_names:
        if name == "y_hat_s":
            chunks.append(y_hat_s)
        elif name == "e_out_s":
            chunks.append(e_out_s)
        elif name == "g_s":
            chunks.append(g_s)
        elif name == "F_s":
            chunks.append(F_s)
    return chunks


@dataclass(frozen=True)
class FMPCIntervalTeacherStateFeatures:
    """Frozen-teacher local dynamical features at a current hidden state `z_s`.

    Shape contract:
    - `y_hat_s`: `(batch, output_dim)`
    - `e_out_s`: `(batch, output_dim)`
    - `g_s`: `(batch, z_dim)`
    - `F_s`: `(batch, 1)`

    Notes:
    - `g_s` is the frozen teacher's one-step hidden-state inference field at `z_s`,
      expressed in normalized-time units compatible with
      `u_star = (z_t - z_s) / (tau_t - tau_s)`.
    - under Euler inference, this is a scaled negative energy gradient field:
      `g_s = K * (z_after_one_teacher_step - z_s)`.
    """

    y_hat_s: np.ndarray
    e_out_s: np.ndarray
    g_s: np.ndarray
    F_s: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "y_hat_s", _as_batch_first("y_hat_s", self.y_hat_s))
        object.__setattr__(self, "e_out_s", _as_batch_first("e_out_s", self.e_out_s))
        object.__setattr__(self, "g_s", _as_batch_first("g_s", self.g_s))
        object.__setattr__(self, "F_s", _as_batch_first("F_s", self.F_s))
        batch_size = int(self.y_hat_s.shape[0])
        if self.e_out_s.shape[0] != batch_size or self.g_s.shape[0] != batch_size or self.F_s.shape[0] != batch_size:
            raise ValueError("All teacher state features must share the same batch dimension.")
        if self.y_hat_s.shape != self.e_out_s.shape:
            raise ValueError("y_hat_s and e_out_s must share the same shape.")
        if self.F_s.shape[1] != 1:
            raise ValueError("F_s must be shaped (batch, 1).")

    def feature_matrix(
        self,
        selected_feature_names: tuple[str, ...] = ("g_s", "e_out_s", "F_s"),
    ) -> np.ndarray:
        """Concatenate selected teacher-state features in the given order.

        Shape contract:
        - returns `(batch, total_selected_feature_dim)`
        """

        if len(selected_feature_names) == 0:
            return np.zeros((self.y_hat_s.shape[0], 0), dtype=np.float64)
        chunks = _feature_chunks(
            selected_feature_names=selected_feature_names,
            y_hat_s=self.y_hat_s,
            e_out_s=self.e_out_s,
            g_s=self.g_s,
            F_s=self.F_s,
        )
        return np.concatenate(chunks, axis=1).astype(np.float64, copy=False)


@dataclass(frozen=True)
class FMPCIntervalTeacherStateFeatureTangents:
    """Directional derivatives of teacher-state features along the current field `g_s`.

    Shape contract:
    - `Dg_y_hat_s`: `(batch, output_dim)`
    - `Dg_e_out_s`: `(batch, output_dim)`
    - `Dg_g_s`: `(batch, z_dim)`
    - `Dg_F_s`: `(batch, 1)`
    """

    Dg_y_hat_s: np.ndarray
    Dg_e_out_s: np.ndarray
    Dg_g_s: np.ndarray
    Dg_F_s: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "Dg_y_hat_s", _as_batch_first("Dg_y_hat_s", self.Dg_y_hat_s))
        object.__setattr__(self, "Dg_e_out_s", _as_batch_first("Dg_e_out_s", self.Dg_e_out_s))
        object.__setattr__(self, "Dg_g_s", _as_batch_first("Dg_g_s", self.Dg_g_s))
        object.__setattr__(self, "Dg_F_s", _as_batch_first("Dg_F_s", self.Dg_F_s))
        batch_size = int(self.Dg_y_hat_s.shape[0])
        if (
            self.Dg_e_out_s.shape[0] != batch_size
            or self.Dg_g_s.shape[0] != batch_size
            or self.Dg_F_s.shape[0] != batch_size
        ):
            raise ValueError("All teacher-state feature tangents must share the same batch dimension.")
        if self.Dg_y_hat_s.shape != self.Dg_e_out_s.shape:
            raise ValueError("Dg_y_hat_s and Dg_e_out_s must share the same shape.")
        if self.Dg_F_s.shape[1] != 1:
            raise ValueError("Dg_F_s must be shaped (batch, 1).")

    def feature_tangent_matrix(
        self,
        selected_feature_names: tuple[str, ...] = ("g_s", "e_out_s", "F_s"),
    ) -> np.ndarray:
        """Concatenate selected feature tangents in the same order as the feature block."""

        if len(selected_feature_names) == 0:
            return np.zeros((self.Dg_y_hat_s.shape[0], 0), dtype=np.float64)
        chunks = _feature_chunks(
            selected_feature_names=selected_feature_names,
            y_hat_s=self.Dg_y_hat_s,
            e_out_s=self.Dg_e_out_s,
            g_s=self.Dg_g_s,
            F_s=self.Dg_F_s,
        )
        return np.concatenate(chunks, axis=1).astype(np.float64, copy=False)


@dataclass(frozen=True)
class FMPCIntervalTeacherFeatureSplitContext:
    """Teacher-backed split context for current-state feature computation."""

    split_name: str
    x: np.ndarray
    y: np.ndarray
    sample_indices: np.ndarray
    teacher_steps: int
    teacher_export_batch_size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _as_batch_first("x", self.x))
        object.__setattr__(self, "y", _as_batch_first("y", self.y))
        object.__setattr__(self, "sample_indices", np.asarray(self.sample_indices, dtype=np.int64))
        if self.x.shape[0] != self.y.shape[0] or self.x.shape[0] != self.sample_indices.shape[0]:
            raise ValueError("x, y, and sample_indices must share the same batch dimension.")
        if self.teacher_steps <= 0:
            raise ValueError("teacher_steps must be positive.")
        if self.teacher_export_batch_size <= 0:
            raise ValueError("teacher_export_batch_size must be positive.")


@dataclass(frozen=True)
class FMPCIntervalTeacherTrajectoryFeatures:
    """Precomputed teacher-state features along a saved teacher trajectory.

    Shape contract:
    - `y_hat_trajectory`: `(batch, teacher_steps + 1, output_dim)`
    - `e_out_trajectory`: `(batch, teacher_steps + 1, output_dim)`
    - `g_trajectory`: `(batch, teacher_steps + 1, z_dim)`
    - `F_trajectory`: `(batch, teacher_steps + 1, 1)`
    """

    split_name: str
    teacher_steps: int
    y_hat_trajectory: np.ndarray
    e_out_trajectory: np.ndarray
    g_trajectory: np.ndarray
    F_trajectory: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "y_hat_trajectory", np.asarray(self.y_hat_trajectory, dtype=np.float64))
        object.__setattr__(self, "e_out_trajectory", np.asarray(self.e_out_trajectory, dtype=np.float64))
        object.__setattr__(self, "g_trajectory", np.asarray(self.g_trajectory, dtype=np.float64))
        object.__setattr__(self, "F_trajectory", np.asarray(self.F_trajectory, dtype=np.float64))
        expected_steps = int(self.teacher_steps) + 1
        for name, array in (
            ("y_hat_trajectory", self.y_hat_trajectory),
            ("e_out_trajectory", self.e_out_trajectory),
            ("g_trajectory", self.g_trajectory),
            ("F_trajectory", self.F_trajectory),
        ):
            if array.ndim != 3:
                raise ValueError(f"{name} must be shaped (batch, step, features).")
            if array.shape[1] != expected_steps:
                raise ValueError(f"{name} step axis must have length {expected_steps}.")
        if self.y_hat_trajectory.shape != self.e_out_trajectory.shape:
            raise ValueError("y_hat_trajectory and e_out_trajectory must share the same shape.")
        if self.y_hat_trajectory.shape[0] != self.g_trajectory.shape[0] or self.y_hat_trajectory.shape[0] != self.F_trajectory.shape[0]:
            raise ValueError("All trajectory feature tensors must share the same batch dimension.")
        if self.F_trajectory.shape[2] != 1:
            raise ValueError("F_trajectory must have trailing feature dimension 1.")

    def step_features(
        self,
        source_index: int,
        *,
        selected_feature_names: tuple[str, ...] = ("g_s", "e_out_s", "F_s"),
    ) -> FMPCIntervalTeacherStateFeatures:
        if source_index < 0 or source_index > self.teacher_steps:
            raise ValueError(f"source_index must lie in [0, {self.teacher_steps}].")
        return FMPCIntervalTeacherStateFeatures(
            y_hat_s=self.y_hat_trajectory[:, source_index, :],
            e_out_s=self.e_out_trajectory[:, source_index, :],
            g_s=self.g_trajectory[:, source_index, :],
            F_s=self.F_trajectory[:, source_index, :],
        )

    def gather_batch_features(
        self,
        sample_row_indices: np.ndarray,
        source_step_indices: np.ndarray,
        *,
        selected_feature_names: tuple[str, ...] = ("g_s", "e_out_s", "F_s"),
    ) -> FMPCIntervalTeacherStateFeatures:
        """Gather teacher-state features for per-sample source knots.

        Shape contract:
        - `sample_row_indices`: `(batch,)`
        - `source_step_indices`: `(batch,)`
        - returns batch-first teacher-state features aligned to those indices
        """

        sample_rows = np.asarray(sample_row_indices, dtype=np.int64)
        source_steps = np.asarray(source_step_indices, dtype=np.int64)
        if sample_rows.ndim != 1 or source_steps.ndim != 1:
            raise ValueError("sample_row_indices and source_step_indices must be rank-1 arrays.")
        if sample_rows.shape != source_steps.shape:
            raise ValueError("sample_row_indices and source_step_indices must share the same shape.")
        if np.any(sample_rows < 0) or np.any(sample_rows >= self.y_hat_trajectory.shape[0]):
            raise ValueError("sample_row_indices are out of range for this trajectory feature tensor.")
        if np.any(source_steps < 0) or np.any(source_steps > self.teacher_steps):
            raise ValueError(f"source_step_indices must lie in [0, {self.teacher_steps}].")
        _validate_feature_names(selected_feature_names)
        return FMPCIntervalTeacherStateFeatures(
            y_hat_s=self.y_hat_trajectory[sample_rows, source_steps, :],
            e_out_s=self.e_out_trajectory[sample_rows, source_steps, :],
            g_s=self.g_trajectory[sample_rows, source_steps, :],
            F_s=self.F_trajectory[sample_rows, source_steps, :],
        )


@dataclass(frozen=True)
class FMPCIntervalTeacherTrajectoryFeatureTangents:
    """Precomputed directional derivatives of teacher features along `g_s`.

    Shape contract:
    - `Dg_y_hat_trajectory`: `(batch, teacher_steps + 1, output_dim)`
    - `Dg_e_out_trajectory`: `(batch, teacher_steps + 1, output_dim)`
    - `Dg_g_trajectory`: `(batch, teacher_steps + 1, z_dim)`
    - `Dg_F_trajectory`: `(batch, teacher_steps + 1, 1)`
    """

    split_name: str
    teacher_steps: int
    Dg_y_hat_trajectory: np.ndarray
    Dg_e_out_trajectory: np.ndarray
    Dg_g_trajectory: np.ndarray
    Dg_F_trajectory: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "Dg_y_hat_trajectory", np.asarray(self.Dg_y_hat_trajectory, dtype=np.float64))
        object.__setattr__(self, "Dg_e_out_trajectory", np.asarray(self.Dg_e_out_trajectory, dtype=np.float64))
        object.__setattr__(self, "Dg_g_trajectory", np.asarray(self.Dg_g_trajectory, dtype=np.float64))
        object.__setattr__(self, "Dg_F_trajectory", np.asarray(self.Dg_F_trajectory, dtype=np.float64))
        expected_steps = int(self.teacher_steps) + 1
        for name, array in (
            ("Dg_y_hat_trajectory", self.Dg_y_hat_trajectory),
            ("Dg_e_out_trajectory", self.Dg_e_out_trajectory),
            ("Dg_g_trajectory", self.Dg_g_trajectory),
            ("Dg_F_trajectory", self.Dg_F_trajectory),
        ):
            if array.ndim != 3:
                raise ValueError(f"{name} must be shaped (batch, step, features).")
            if array.shape[1] != expected_steps:
                raise ValueError(f"{name} step axis must have length {expected_steps}.")
        if self.Dg_y_hat_trajectory.shape != self.Dg_e_out_trajectory.shape:
            raise ValueError("Dg_y_hat_trajectory and Dg_e_out_trajectory must share the same shape.")
        if (
            self.Dg_y_hat_trajectory.shape[0] != self.Dg_g_trajectory.shape[0]
            or self.Dg_y_hat_trajectory.shape[0] != self.Dg_F_trajectory.shape[0]
        ):
            raise ValueError("All trajectory tangent tensors must share the same batch dimension.")
        if self.Dg_F_trajectory.shape[2] != 1:
            raise ValueError("Dg_F_trajectory must have trailing feature dimension 1.")

    def step_feature_tangents(self, source_index: int) -> FMPCIntervalTeacherStateFeatureTangents:
        if source_index < 0 or source_index > self.teacher_steps:
            raise ValueError(f"source_index must lie in [0, {self.teacher_steps}].")
        return FMPCIntervalTeacherStateFeatureTangents(
            Dg_y_hat_s=self.Dg_y_hat_trajectory[:, source_index, :],
            Dg_e_out_s=self.Dg_e_out_trajectory[:, source_index, :],
            Dg_g_s=self.Dg_g_trajectory[:, source_index, :],
            Dg_F_s=self.Dg_F_trajectory[:, source_index, :],
        )

    def gather_batch_feature_tangents(
        self,
        sample_row_indices: np.ndarray,
        source_step_indices: np.ndarray,
    ) -> FMPCIntervalTeacherStateFeatureTangents:
        sample_rows = np.asarray(sample_row_indices, dtype=np.int64)
        source_steps = np.asarray(source_step_indices, dtype=np.int64)
        if sample_rows.ndim != 1 or source_steps.ndim != 1:
            raise ValueError("sample_row_indices and source_step_indices must be rank-1 arrays.")
        if sample_rows.shape != source_steps.shape:
            raise ValueError("sample_row_indices and source_step_indices must share the same shape.")
        if np.any(sample_rows < 0) or np.any(sample_rows >= self.Dg_y_hat_trajectory.shape[0]):
            raise ValueError("sample_row_indices are out of range for this trajectory tangent tensor.")
        if np.any(source_steps < 0) or np.any(source_steps > self.teacher_steps):
            raise ValueError(f"source_step_indices must lie in [0, {self.teacher_steps}].")
        return FMPCIntervalTeacherStateFeatureTangents(
            Dg_y_hat_s=self.Dg_y_hat_trajectory[sample_rows, source_steps, :],
            Dg_e_out_s=self.Dg_e_out_trajectory[sample_rows, source_steps, :],
            Dg_g_s=self.Dg_g_trajectory[sample_rows, source_steps, :],
            Dg_F_s=self.Dg_F_trajectory[sample_rows, source_steps, :],
        )


@dataclass(frozen=True)
class FMPCIntervalTeacherFeatureBundle:
    """Exact-checkpoint-backed teacher feature contexts and train/val/test trajectory features."""

    teacher_steps: int
    train: FMPCIntervalTeacherTrajectoryFeatures
    val: FMPCIntervalTeacherTrajectoryFeatures
    test: FMPCIntervalTeacherTrajectoryFeatures
    train_tangents: FMPCIntervalTeacherTrajectoryFeatureTangents
    val_tangents: FMPCIntervalTeacherTrajectoryFeatureTangents
    test_tangents: FMPCIntervalTeacherTrajectoryFeatureTangents
    split_contexts: dict[str, FMPCIntervalTeacherFeatureSplitContext]
    feature_names_supported: tuple[str, ...] = _SUPPORTED_TEACHER_FEATURE_NAMES

    def trajectory_features(self, split_name: str) -> FMPCIntervalTeacherTrajectoryFeatures:
        if split_name == "train":
            return self.train
        if split_name == "val":
            return self.val
        if split_name == "test":
            return self.test
        raise ValueError(f"Unsupported split_name '{split_name}'.")

    def trajectory_feature_tangents(self, split_name: str) -> FMPCIntervalTeacherTrajectoryFeatureTangents:
        if split_name == "train":
            return self.train_tangents
        if split_name == "val":
            return self.val_tangents
        if split_name == "test":
            return self.test_tangents
        raise ValueError(f"Unsupported split_name '{split_name}'.")

    def current_state_feature_matrix(
        self,
        teacher_model: PCNetwork,
        *,
        split_name: str,
        z_s: np.ndarray,
        target_onehot: np.ndarray,
        tau_s: np.ndarray | float,
        tau_t: np.ndarray | float,
        selected_feature_names: tuple[str, ...],
    ) -> tuple[np.ndarray, FMPCIntervalTeacherStateFeatures]:
        """Compute current-state feature matrix for one split in its canonical sample order."""

        _ = target_onehot
        _ = tau_s
        _ = tau_t
        context = self.split_contexts[split_name]
        state_features = compute_interval_teacher_state_features(
            teacher_model,
            context.x,
            context.y,
            np.asarray(z_s, dtype=np.float64),
            teacher_steps=context.teacher_steps,
            batch_size=context.teacher_export_batch_size,
        )
        return state_features.feature_matrix(selected_feature_names), state_features

    def current_state_feature_matrix_and_tangents(
        self,
        teacher_model: PCNetwork,
        *,
        split_name: str,
        z_s: np.ndarray,
        target_onehot: np.ndarray,
        tau_s: np.ndarray | float,
        tau_t: np.ndarray | float,
        selected_feature_names: tuple[str, ...],
        fd_epsilon: float,
    ) -> tuple[np.ndarray, FMPCIntervalTeacherStateFeatures, FMPCIntervalTeacherStateFeatureTangents]:
        """Compute current-state teacher features and their directional derivatives.

        Shape contract:
        - returns `(feature_matrix, state_features, feature_tangents)`
        - the feature tangent contract follows `D_g feat(z_s)` along the current field `g_s`
        """

        _ = target_onehot
        _ = tau_s
        _ = tau_t
        context = self.split_contexts[split_name]
        state_features = compute_interval_teacher_state_features(
            teacher_model,
            context.x,
            context.y,
            np.asarray(z_s, dtype=np.float64),
            teacher_steps=context.teacher_steps,
            batch_size=context.teacher_export_batch_size,
        )
        state_tangents = compute_interval_teacher_state_feature_tangents(
            teacher_model,
            context.x,
            context.y,
            np.asarray(z_s, dtype=np.float64),
            teacher_steps=context.teacher_steps,
            batch_size=context.teacher_export_batch_size,
            fd_epsilon=fd_epsilon,
            state_features=state_features,
        )
        return (
            state_features.feature_matrix(selected_feature_names),
            state_features,
            state_tangents,
        )


def compute_interval_teacher_state_features(
    teacher_model: PCNetwork,
    x: np.ndarray,
    y: np.ndarray,
    z_s: np.ndarray,
    *,
    teacher_steps: int,
    batch_size: int | None = None,
) -> FMPCIntervalTeacherStateFeatures:
    """Compute frozen-teacher local dynamical features at current state `z_s`.

    Shape contract:
    - `x`: `(batch, input_dim)`
    - `y`: `(batch, output_dim)`
    - `z_s`: `(batch, z_dim)`
    - returns batch-first teacher-state features at the same batch size

    Notes:
    - batching is explicit because the baseline teacher dynamics scale with batch size
    - to match saved teacher trajectory semantics, callers should normally pass the same
      `teacher_export_batch_size` that was used when teacher targets were exported
    """

    x_array = _as_batch_first("x", x)
    y_array = _as_batch_first("y", y)
    z_array = _as_batch_first("z_s", z_s)
    if x_array.shape[0] != y_array.shape[0] or x_array.shape[0] != z_array.shape[0]:
        raise ValueError("x, y, and z_s must share the same batch dimension.")
    if teacher_steps <= 0:
        raise ValueError("teacher_steps must be positive.")
    effective_batch_size = int(x_array.shape[0] if batch_size is None else batch_size)
    if effective_batch_size <= 0:
        raise ValueError("batch_size must be positive when provided.")

    y_hat_batches: list[np.ndarray] = []
    e_out_batches: list[np.ndarray] = []
    g_batches: list[np.ndarray] = []
    energy_batches: list[np.ndarray] = []

    for start in range(0, x_array.shape[0], effective_batch_size):
        stop = min(start + effective_batch_size, x_array.shape[0])
        x_batch = x_array[start:stop]
        y_batch = y_array[start:stop]
        z_batch = z_array[start:stop]
        states, clamped_mask = _state_list_from_hidden_state(teacher_model, x_batch, y_batch, z_batch)
        cache = compute_cache(states, teacher_model.layers)
        one_step = run_inference(
            states,
            teacher_model.layers,
            clamped_mask,
            eta_x=teacher_model.eta_x,
            steps=1,
            backend=str(teacher_model.inference_backend),
            record_trace=False,
            record_state_trajectory=False,
        )
        z_next = flatten_hidden_states(one_step.states, clamped_mask)
        y_hat_batches.append(np.asarray(cache.predictions[-1], dtype=np.float64))
        e_out_batches.append(np.asarray(cache.errors[-1], dtype=np.float64))
        g_batches.append((teacher_steps * (z_next - z_batch)).astype(np.float64, copy=False))
        energy_batches.append(_per_sample_energy_column(cache, teacher_model))

    return FMPCIntervalTeacherStateFeatures(
        y_hat_s=np.concatenate(y_hat_batches, axis=0),
        e_out_s=np.concatenate(e_out_batches, axis=0),
        g_s=np.concatenate(g_batches, axis=0),
        F_s=np.concatenate(energy_batches, axis=0),
    )


def compute_interval_teacher_state_feature_tangents(
    teacher_model: PCNetwork,
    x: np.ndarray,
    y: np.ndarray,
    z_s: np.ndarray,
    *,
    teacher_steps: int,
    batch_size: int | None = None,
    fd_epsilon: float,
    state_features: FMPCIntervalTeacherStateFeatures | None = None,
) -> FMPCIntervalTeacherStateFeatureTangents:
    """Compute directional derivatives of teacher features along the current field `g_s`.

    Shape contract:
    - `x`: `(batch, input_dim)`
    - `y`: `(batch, output_dim)`
    - `z_s`: `(batch, z_dim)`
    - returns batch-first directional derivatives for `y_hat_s`, `e_out_s`, `g_s`, and `F_s`

    Finite-difference contract:
    - `D_g feat(z_s) ≈ [feat(z_s + eps * g_s) - feat(z_s - eps * g_s)] / (2 * eps)`
    - `eps` is the explicit normalized-time probe size
    """

    epsilon = _validate_fd_epsilon(fd_epsilon)
    z_array = _as_batch_first("z_s", z_s)
    base_features = (
        compute_interval_teacher_state_features(
            teacher_model,
            x,
            y,
            z_array,
            teacher_steps=teacher_steps,
            batch_size=batch_size,
        )
        if state_features is None
        else state_features
    )
    if base_features.g_s.shape != z_array.shape:
        raise ValueError("state_features.g_s must share the same shape as z_s.")
    z_plus = (z_array + epsilon * base_features.g_s).astype(np.float64, copy=False)
    z_minus = (z_array - epsilon * base_features.g_s).astype(np.float64, copy=False)
    plus_features = compute_interval_teacher_state_features(
        teacher_model,
        x,
        y,
        z_plus,
        teacher_steps=teacher_steps,
        batch_size=batch_size,
    )
    minus_features = compute_interval_teacher_state_features(
        teacher_model,
        x,
        y,
        z_minus,
        teacher_steps=teacher_steps,
        batch_size=batch_size,
    )
    scale = 2.0 * epsilon
    return FMPCIntervalTeacherStateFeatureTangents(
        Dg_y_hat_s=((plus_features.y_hat_s - minus_features.y_hat_s) / scale).astype(np.float64, copy=False),
        Dg_e_out_s=((plus_features.e_out_s - minus_features.e_out_s) / scale).astype(np.float64, copy=False),
        Dg_g_s=((plus_features.g_s - minus_features.g_s) / scale).astype(np.float64, copy=False),
        Dg_F_s=((plus_features.F_s - minus_features.F_s) / scale).astype(np.float64, copy=False),
    )


def prepare_interval_teacher_feature_context(
    interval_split: Any,
    teacher_split: SupervisedDataSplit,
    *,
    teacher_export_batch_size: int,
) -> FMPCIntervalTeacherFeatureSplitContext:
    """Align one interval split with the exact teacher split arrays used for feature computation."""

    x, y, sample_indices = _teacher_split_arrays(teacher_split, interval_split.split_name)
    if not np.array_equal(np.asarray(interval_split.sample_indices, dtype=np.int64), sample_indices):
        raise ValueError(
            f"{interval_split.split_name} sample_indices do not match the exact teacher split ordering."
        )
    if not np.allclose(np.asarray(interval_split.target_onehot, dtype=np.float64), y):
        raise ValueError(f"{interval_split.split_name} targets do not match the exact teacher split.")
    return FMPCIntervalTeacherFeatureSplitContext(
        split_name=interval_split.split_name,
        x=x,
        y=y,
        sample_indices=sample_indices,
        teacher_steps=int(interval_split.teacher_steps),
        teacher_export_batch_size=int(teacher_export_batch_size),
    )


def precompute_interval_teacher_trajectory_features(
    teacher_model: PCNetwork,
    interval_split: Any,
    context: FMPCIntervalTeacherFeatureSplitContext,
) -> FMPCIntervalTeacherTrajectoryFeatures:
    """Precompute current-state teacher features for every saved teacher knot in one split."""

    if context.teacher_steps != int(interval_split.teacher_steps):
        raise ValueError("context.teacher_steps must match interval_split.teacher_steps.")
    y_hat_steps: list[np.ndarray] = []
    e_out_steps: list[np.ndarray] = []
    g_steps: list[np.ndarray] = []
    F_steps: list[np.ndarray] = []
    for source_index in range(context.teacher_steps + 1):
        features = compute_interval_teacher_state_features(
            teacher_model,
            context.x,
            context.y,
            np.asarray(interval_split.z_trajectory[:, source_index, :], dtype=np.float64),
            teacher_steps=context.teacher_steps,
            batch_size=context.teacher_export_batch_size,
        )
        y_hat_steps.append(features.y_hat_s)
        e_out_steps.append(features.e_out_s)
        g_steps.append(features.g_s)
        F_steps.append(features.F_s)
    return FMPCIntervalTeacherTrajectoryFeatures(
        split_name=context.split_name,
        teacher_steps=context.teacher_steps,
        y_hat_trajectory=np.stack(y_hat_steps, axis=1),
        e_out_trajectory=np.stack(e_out_steps, axis=1),
        g_trajectory=np.stack(g_steps, axis=1),
        F_trajectory=np.stack(F_steps, axis=1),
    )


def precompute_interval_teacher_trajectory_feature_tangents(
    teacher_model: PCNetwork,
    interval_split: Any,
    context: FMPCIntervalTeacherFeatureSplitContext,
    *,
    fd_epsilon: float,
) -> FMPCIntervalTeacherTrajectoryFeatureTangents:
    """Precompute `D_g` teacher feature tangents for every saved teacher knot in one split."""

    if context.teacher_steps != int(interval_split.teacher_steps):
        raise ValueError("context.teacher_steps must match interval_split.teacher_steps.")
    Dg_y_hat_steps: list[np.ndarray] = []
    Dg_e_out_steps: list[np.ndarray] = []
    Dg_g_steps: list[np.ndarray] = []
    Dg_F_steps: list[np.ndarray] = []
    for source_index in range(context.teacher_steps + 1):
        z_source = np.asarray(interval_split.z_trajectory[:, source_index, :], dtype=np.float64)
        features = compute_interval_teacher_state_features(
            teacher_model,
            context.x,
            context.y,
            z_source,
            teacher_steps=context.teacher_steps,
            batch_size=context.teacher_export_batch_size,
        )
        tangents = compute_interval_teacher_state_feature_tangents(
            teacher_model,
            context.x,
            context.y,
            z_source,
            teacher_steps=context.teacher_steps,
            batch_size=context.teacher_export_batch_size,
            fd_epsilon=fd_epsilon,
            state_features=features,
        )
        Dg_y_hat_steps.append(tangents.Dg_y_hat_s)
        Dg_e_out_steps.append(tangents.Dg_e_out_s)
        Dg_g_steps.append(tangents.Dg_g_s)
        Dg_F_steps.append(tangents.Dg_F_s)
    return FMPCIntervalTeacherTrajectoryFeatureTangents(
        split_name=context.split_name,
        teacher_steps=context.teacher_steps,
        Dg_y_hat_trajectory=np.stack(Dg_y_hat_steps, axis=1),
        Dg_e_out_trajectory=np.stack(Dg_e_out_steps, axis=1),
        Dg_g_trajectory=np.stack(Dg_g_steps, axis=1),
        Dg_F_trajectory=np.stack(Dg_F_steps, axis=1),
    )
