from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..stage_01_reference_prep.fmpc_student_data import FMPCStudentDataset, FMPCStudentSplit, load_fmpc_student_dataset


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _as_batch_first(name: str, array: np.ndarray) -> np.ndarray:
    array_float = np.asarray(array, dtype=np.float64)
    if array_float.ndim != 2:
        raise ValueError(f"{name} must be shaped (batch, features).")
    return array_float


def _as_time_column(name: str, tau: np.ndarray | float, *, batch_size: int) -> np.ndarray:
    tau_array = np.asarray(tau, dtype=np.float64)
    if tau_array.ndim == 0:
        return np.full((batch_size, 1), float(tau_array), dtype=np.float64)
    if tau_array.ndim == 1:
        if tau_array.shape[0] != batch_size:
            raise ValueError(f"{name} must have batch dimension {batch_size}.")
        return tau_array.reshape(batch_size, 1).astype(np.float64, copy=False)
    if tau_array.ndim == 2 and tau_array.shape == (batch_size, 1):
        return tau_array.astype(np.float64, copy=False)
    raise ValueError(f"{name} must be a scalar, (batch,), or (batch, 1).")


@dataclass(frozen=True)
class FMPCIntervalBatch:
    """One batch of interval-conditioned teacher supervision.

    Shape contract:
    - `sample_row_indices`: `(batch,)`
    - `source_step_indices`: `(batch,)`
    - `target_step_indices`: `(batch,)`
    - `span_lengths`: `(batch,)`
    - `target_onehot`: `(batch, target_dim)`
    - `z_s`: `(batch, z_dim)`
    - `z_t`: `(batch, z_dim)`
    - `tau_s`: `(batch, 1)`
    - `tau_t`: `(batch, 1)`
    - `delta_tau`: `(batch, 1)`
    - `u_star`: `(batch, z_dim)`
    - `student_inputs`: `(batch, z_dim + target_dim + 2)`
    """

    sample_row_indices: np.ndarray
    source_step_indices: np.ndarray
    target_step_indices: np.ndarray
    span_lengths: np.ndarray
    target_onehot: np.ndarray
    z_s: np.ndarray
    z_t: np.ndarray
    tau_s: np.ndarray
    tau_t: np.ndarray
    delta_tau: np.ndarray
    u_star: np.ndarray
    student_inputs: np.ndarray


@dataclass(frozen=True)
class FMPCIntervalSplit:
    """Trajectory-backed interval supervision for one split.

    Shape contract:
    - `sample_indices`: `(batch,)`
    - `target_onehot`: `(batch, target_dim)`
    - `z0`: `(batch, z_dim)`
    - `z_star`: `(batch, z_dim)`
    - `z_trajectory`: `(batch, teacher_steps + 1, z_dim)`
    """

    split_name: str
    sample_indices: np.ndarray
    target_onehot: np.ndarray
    z0: np.ndarray
    z_star: np.ndarray
    z_trajectory: np.ndarray
    teacher_steps: int
    metadata: dict[str, Any]

    @property
    def num_samples(self) -> int:
        return int(self.z0.shape[0])

    @property
    def z_dim(self) -> int:
        return int(self.z0.shape[1])

    @property
    def target_dim(self) -> int:
        return int(self.target_onehot.shape[1])


@dataclass(frozen=True)
class FMPCIntervalDataset:
    """Validated trajectory-backed dataset for Phase 5B interval students."""

    dataset_name: str
    schema_version: str
    teacher_manifest_path: Path
    teacher_checkpoint_path: Path | None
    teacher_mode: str
    teacher_target_semantics: str
    interval_input_definition: str
    interval_target_definition: str
    teacher_steps: int
    z_dim: int
    target_dim: int
    train: FMPCIntervalSplit
    val: FMPCIntervalSplit
    test: FMPCIntervalSplit
    metadata: dict[str, Any]


def build_fmpc_interval_inputs(
    z_s: np.ndarray,
    target_onehot: np.ndarray,
    tau_s: np.ndarray | float,
    tau_t: np.ndarray | float,
) -> np.ndarray:
    """Return `concat([z_s, target_onehot, tau_s, tau_t])`.

    Shape contract:
    - `z_s`: `(batch, z_dim)`
    - `target_onehot`: `(batch, target_dim)`
    - `tau_s`: scalar, `(batch,)`, or `(batch, 1)`
    - `tau_t`: scalar, `(batch,)`, or `(batch, 1)`
    - returns: `(batch, z_dim + target_dim + 2)`
    """

    z_s_array = _as_batch_first("z_s", z_s)
    targets = _as_batch_first("target_onehot", target_onehot)
    if z_s_array.shape[0] != targets.shape[0]:
        raise ValueError("z_s and target_onehot must share the same batch dimension.")
    tau_s_column = _as_time_column("tau_s", tau_s, batch_size=z_s_array.shape[0])
    tau_t_column = _as_time_column("tau_t", tau_t, batch_size=z_s_array.shape[0])
    return np.concatenate([z_s_array, targets, tau_s_column, tau_t_column], axis=1).astype(
        np.float64,
        copy=False,
    )


def compute_interval_velocity_target(
    z_s: np.ndarray,
    z_t: np.ndarray,
    tau_s: np.ndarray | float,
    tau_t: np.ndarray | float,
) -> np.ndarray:
    """Compute `u_star = (z_t - z_s) / (tau_t - tau_s)` in batch-first form."""

    z_s_array = _as_batch_first("z_s", z_s)
    z_t_array = _as_batch_first("z_t", z_t)
    if z_s_array.shape != z_t_array.shape:
        raise ValueError("z_s and z_t must share the same shape.")
    tau_s_column = _as_time_column("tau_s", tau_s, batch_size=z_s_array.shape[0])
    tau_t_column = _as_time_column("tau_t", tau_t, batch_size=z_s_array.shape[0])
    delta_tau = tau_t_column - tau_s_column
    if np.any(delta_tau <= 0.0):
        raise ValueError("tau_t must be strictly greater than tau_s for every interval pair.")
    return ((z_t_array - z_s_array) / delta_tau).astype(np.float64, copy=False)


def teacher_step_aligned_rollout_schedules(teacher_steps: int) -> dict[str, tuple[int, ...]]:
    """Return default teacher-step-aligned rollout knots for 1/2/3-step evaluation."""

    if teacher_steps <= 0:
        raise ValueError("teacher_steps must be positive.")
    one_step = (0, int(teacher_steps))
    two_step = (0, int(round(teacher_steps / 2.0)), int(teacher_steps))
    three_step = (
        0,
        int(round(teacher_steps / 3.0)),
        int(round(2.0 * teacher_steps / 3.0)),
        int(teacher_steps),
    )
    schedules = {
        "1-step": one_step,
        "2-step": two_step,
        "3-step": three_step,
    }
    for name, knots in schedules.items():
        if knots[0] != 0 or knots[-1] != teacher_steps:
            raise ValueError(f"{name} rollout schedule must start at 0 and end at teacher_steps.")
        if tuple(sorted(knots)) != knots or len(set(knots)) != len(knots):
            raise ValueError(f"{name} rollout schedule must use strictly increasing teacher-aligned knots.")
    return schedules


def acceptance_schedule_focus_pairs(
    teacher_steps: int,
    *,
    schedule_names: tuple[str, ...] = ("2-step", "3-step"),
) -> tuple[tuple[int, int], ...]:
    """Return the unique teacher-step-aligned interval pairs used by acceptance rollouts."""

    schedules = teacher_step_aligned_rollout_schedules(teacher_steps)
    invalid = [name for name in schedule_names if name not in schedules]
    if invalid:
        raise ValueError(
            f"Unsupported schedule_names {invalid}; expected a subset of {list(schedules.keys())}."
        )
    focus_pairs: list[tuple[int, int]] = []
    for name in schedule_names:
        knots = schedules[name]
        for source_index, target_index in zip(knots[:-1], knots[1:], strict=True):
            pair = (int(source_index), int(target_index))
            if pair not in focus_pairs:
                focus_pairs.append(pair)
    return tuple(focus_pairs)


def _balanced_pair_weight(teacher_steps: int, span_length: int) -> float:
    if not 1 <= span_length <= teacher_steps:
        raise ValueError("span_length must be in [1, teacher_steps].")
    return 1.0 / float(teacher_steps * (teacher_steps - span_length + 1))


def iter_all_interval_blocks(split: FMPCIntervalSplit) -> Iterator[dict[str, Any]]:
    """Yield all interval blocks with span-balanced weights.

    Each yielded block contains every sample for one `(s, t)` pair and a weight that makes
    the aggregate span mass uniform across interval lengths.
    """

    for span_length in range(1, split.teacher_steps + 1):
        pair_weight = _balanced_pair_weight(split.teacher_steps, span_length)
        for source_index in range(0, split.teacher_steps - span_length + 1):
            target_index = source_index + span_length
            tau_s = np.full((split.num_samples, 1), source_index / split.teacher_steps, dtype=np.float64)
            tau_t = np.full((split.num_samples, 1), target_index / split.teacher_steps, dtype=np.float64)
            z_s = np.asarray(split.z_trajectory[:, source_index, :], dtype=np.float64)
            z_t = np.asarray(split.z_trajectory[:, target_index, :], dtype=np.float64)
            yield {
                "span_length": int(span_length),
                "source_index": int(source_index),
                "target_index": int(target_index),
                "pair_weight": float(pair_weight),
                "z_s": z_s,
                "z_t": z_t,
                "tau_s": tau_s,
                "tau_t": tau_t,
                "delta_tau": tau_t - tau_s,
                "target_onehot": split.target_onehot,
                "u_star": compute_interval_velocity_target(z_s, z_t, tau_s, tau_t),
            }


def iter_weighted_interval_blocks(
    split: FMPCIntervalSplit,
    *,
    knot_focused_schedule_names: tuple[str, ...] = (),
    knot_focus_mixture: float = 0.0,
) -> Iterator[dict[str, Any]]:
    """Yield weighted interval blocks under a base-plus-knot-focused mixture.

    The default `knot_focus_mixture=0.0` recovers the original span-balanced weighting.
    When `knot_focus_mixture > 0`, the weighted distribution becomes:

    - `(1 - knot_focus_mixture) * span_balanced_distribution`
    - `+ knot_focus_mixture * uniform_distribution_over_acceptance_schedule_pairs`
    """

    if not (0.0 <= knot_focus_mixture <= 1.0):
        raise ValueError("knot_focus_mixture must lie in [0, 1].")
    focus_pairs = acceptance_schedule_focus_pairs(
        split.teacher_steps,
        schedule_names=knot_focused_schedule_names,
    ) if knot_focused_schedule_names else ()
    focus_pair_weight = 0.0 if len(focus_pairs) == 0 else float(knot_focus_mixture) / float(len(focus_pairs))

    for block in iter_all_interval_blocks(split):
        pair = (int(block["source_index"]), int(block["target_index"]))
        pair_weight = (1.0 - float(knot_focus_mixture)) * float(block["pair_weight"])
        if pair in focus_pairs:
            pair_weight += focus_pair_weight
        if pair_weight <= 0.0:
            continue
        yield {
            **block,
            "pair_weight": float(pair_weight),
            "knot_focused_pair": pair in focus_pairs,
            "knot_focused_schedule_names": tuple(knot_focused_schedule_names),
            "knot_focus_mixture": float(knot_focus_mixture),
        }


def sample_balanced_interval_batch(
    split: FMPCIntervalSplit,
    batch_size: int,
    *,
    seed: int,
) -> FMPCIntervalBatch:
    """Sample one span-balanced interval batch from a trajectory-backed split."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    rng = np.random.default_rng(seed)
    return _sample_balanced_interval_batch_rng(split, batch_size, rng)


def _sample_balanced_interval_batch_rng(
    split: FMPCIntervalSplit,
    batch_size: int,
    rng: np.random.Generator,
) -> FMPCIntervalBatch:
    span_lengths = rng.integers(1, split.teacher_steps + 1, size=batch_size, endpoint=False)
    sample_row_indices = rng.integers(0, split.num_samples, size=batch_size, endpoint=False)
    source_step_indices = np.zeros(batch_size, dtype=np.int64)
    for span_length in np.unique(span_lengths):
        span_mask = span_lengths == span_length
        upper = split.teacher_steps - int(span_length) + 1
        source_step_indices[span_mask] = rng.integers(0, upper, size=int(np.sum(span_mask)), endpoint=False)
    target_step_indices = source_step_indices + span_lengths

    z_s = split.z_trajectory[sample_row_indices, source_step_indices, :].astype(np.float64, copy=False)
    z_t = split.z_trajectory[sample_row_indices, target_step_indices, :].astype(np.float64, copy=False)
    target_onehot = split.target_onehot[sample_row_indices].astype(np.float64, copy=False)
    tau_s = (source_step_indices.astype(np.float64) / split.teacher_steps).reshape(batch_size, 1)
    tau_t = (target_step_indices.astype(np.float64) / split.teacher_steps).reshape(batch_size, 1)
    student_inputs = build_fmpc_interval_inputs(z_s, target_onehot, tau_s, tau_t)
    u_star = compute_interval_velocity_target(z_s, z_t, tau_s, tau_t)

    return FMPCIntervalBatch(
        sample_row_indices=sample_row_indices.astype(np.int64, copy=False),
        source_step_indices=source_step_indices.astype(np.int64, copy=False),
        target_step_indices=target_step_indices.astype(np.int64, copy=False),
        span_lengths=span_lengths.astype(np.int64, copy=False),
        target_onehot=target_onehot,
        z_s=z_s,
        z_t=z_t,
        tau_s=tau_s.astype(np.float64, copy=False),
        tau_t=tau_t.astype(np.float64, copy=False),
        delta_tau=(tau_t - tau_s).astype(np.float64, copy=False),
        u_star=u_star,
        student_inputs=student_inputs,
    )


def _sample_focus_interval_batch_rng(
    split: FMPCIntervalSplit,
    batch_size: int,
    rng: np.random.Generator,
    *,
    focus_pairs: tuple[tuple[int, int], ...],
) -> FMPCIntervalBatch:
    if len(focus_pairs) == 0:
        raise ValueError("focus_pairs must not be empty.")
    pair_indices = rng.integers(0, len(focus_pairs), size=batch_size, endpoint=False)
    chosen_pairs = np.asarray([focus_pairs[index] for index in pair_indices], dtype=np.int64)
    source_step_indices = chosen_pairs[:, 0].astype(np.int64, copy=False)
    target_step_indices = chosen_pairs[:, 1].astype(np.int64, copy=False)
    span_lengths = (target_step_indices - source_step_indices).astype(np.int64, copy=False)
    sample_row_indices = rng.integers(0, split.num_samples, size=batch_size, endpoint=False)

    z_s = split.z_trajectory[sample_row_indices, source_step_indices, :].astype(np.float64, copy=False)
    z_t = split.z_trajectory[sample_row_indices, target_step_indices, :].astype(np.float64, copy=False)
    target_onehot = split.target_onehot[sample_row_indices].astype(np.float64, copy=False)
    tau_s = (source_step_indices.astype(np.float64) / split.teacher_steps).reshape(batch_size, 1)
    tau_t = (target_step_indices.astype(np.float64) / split.teacher_steps).reshape(batch_size, 1)
    student_inputs = build_fmpc_interval_inputs(z_s, target_onehot, tau_s, tau_t)
    u_star = compute_interval_velocity_target(z_s, z_t, tau_s, tau_t)

    return FMPCIntervalBatch(
        sample_row_indices=sample_row_indices.astype(np.int64, copy=False),
        source_step_indices=source_step_indices,
        target_step_indices=target_step_indices,
        span_lengths=span_lengths,
        target_onehot=target_onehot,
        z_s=z_s,
        z_t=z_t,
        tau_s=tau_s.astype(np.float64, copy=False),
        tau_t=tau_t.astype(np.float64, copy=False),
        delta_tau=(tau_t - tau_s).astype(np.float64, copy=False),
        u_star=u_star,
        student_inputs=student_inputs,
    )


def iter_balanced_interval_batches(
    split: FMPCIntervalSplit,
    batch_size: int,
    *,
    num_batches: int,
    seed: int,
) -> Iterator[FMPCIntervalBatch]:
    """Yield deterministic span-balanced interval batches from one split."""

    if num_batches <= 0:
        raise ValueError("num_batches must be positive.")
    rng = np.random.default_rng(seed)
    for _ in range(num_batches):
        yield _sample_balanced_interval_batch_rng(split, batch_size, rng)


def iter_mixed_interval_batches(
    split: FMPCIntervalSplit,
    batch_size: int,
    *,
    num_batches: int,
    seed: int,
    knot_focused_schedule_names: tuple[str, ...] = ("2-step", "3-step"),
    knot_focus_probability: float = 0.0,
) -> Iterator[FMPCIntervalBatch]:
    """Yield deterministic batches from a span-balanced plus knot-focused mixture."""

    if not (0.0 <= knot_focus_probability <= 1.0):
        raise ValueError("knot_focus_probability must lie in [0, 1].")
    if num_batches <= 0:
        raise ValueError("num_batches must be positive.")
    rng = np.random.default_rng(seed)
    focus_pairs = acceptance_schedule_focus_pairs(
        split.teacher_steps,
        schedule_names=knot_focused_schedule_names,
    )
    for _ in range(num_batches):
        if knot_focus_probability == 0.0:
            yield _sample_balanced_interval_batch_rng(split, batch_size, rng)
            continue
        if knot_focus_probability == 1.0:
            yield _sample_focus_interval_batch_rng(
                split,
                batch_size,
                rng,
                focus_pairs=focus_pairs,
            )
            continue

        focus_mask = rng.random(batch_size) < float(knot_focus_probability)
        base_batch = _sample_balanced_interval_batch_rng(split, batch_size, rng)
        if not np.any(focus_mask):
            yield base_batch
            continue
        focus_batch = _sample_focus_interval_batch_rng(
            split,
            int(np.sum(focus_mask)),
            rng,
            focus_pairs=focus_pairs,
        )

        sample_row_indices = base_batch.sample_row_indices.copy()
        source_step_indices = base_batch.source_step_indices.copy()
        target_step_indices = base_batch.target_step_indices.copy()
        span_lengths = base_batch.span_lengths.copy()
        target_onehot = base_batch.target_onehot.copy()
        z_s = base_batch.z_s.copy()
        z_t = base_batch.z_t.copy()
        tau_s = base_batch.tau_s.copy()
        tau_t = base_batch.tau_t.copy()
        delta_tau = base_batch.delta_tau.copy()
        u_star = base_batch.u_star.copy()
        student_inputs = base_batch.student_inputs.copy()

        sample_row_indices[focus_mask] = focus_batch.sample_row_indices
        source_step_indices[focus_mask] = focus_batch.source_step_indices
        target_step_indices[focus_mask] = focus_batch.target_step_indices
        span_lengths[focus_mask] = focus_batch.span_lengths
        target_onehot[focus_mask] = focus_batch.target_onehot
        z_s[focus_mask] = focus_batch.z_s
        z_t[focus_mask] = focus_batch.z_t
        tau_s[focus_mask] = focus_batch.tau_s
        tau_t[focus_mask] = focus_batch.tau_t
        delta_tau[focus_mask] = focus_batch.delta_tau
        u_star[focus_mask] = focus_batch.u_star
        student_inputs[focus_mask] = focus_batch.student_inputs

        yield FMPCIntervalBatch(
            sample_row_indices=sample_row_indices.astype(np.int64, copy=False),
            source_step_indices=source_step_indices.astype(np.int64, copy=False),
            target_step_indices=target_step_indices.astype(np.int64, copy=False),
            span_lengths=span_lengths.astype(np.int64, copy=False),
            target_onehot=target_onehot.astype(np.float64, copy=False),
            z_s=z_s.astype(np.float64, copy=False),
            z_t=z_t.astype(np.float64, copy=False),
            tau_s=tau_s.astype(np.float64, copy=False),
            tau_t=tau_t.astype(np.float64, copy=False),
            delta_tau=delta_tau.astype(np.float64, copy=False),
            u_star=u_star.astype(np.float64, copy=False),
            student_inputs=student_inputs.astype(np.float64, copy=False),
        )


def _load_interval_split(
    endpoint_split: FMPCStudentSplit,
    *,
    manifest_path: Path,
    split_payload: dict[str, Any],
    expected_teacher_steps: int,
) -> FMPCIntervalSplit:
    split_path_value = split_payload.get("relative_path", split_payload.get("path"))
    if split_path_value is None:
        raise ValueError("Trajectory split payload must contain relative_path or path.")
    split_path = Path(str(split_path_value))
    if not split_path.is_absolute():
        split_path = (manifest_path.parent / split_path).resolve()
    with np.load(split_path) as arrays:
        if "z_trajectory" not in arrays:
            raise ValueError(
                f"{endpoint_split.split_name} teacher artifact is missing z_trajectory; "
                "Phase 5B requires export_trajectory=True."
            )
        z_trajectory = np.asarray(arrays["z_trajectory"], dtype=np.float64)

    if z_trajectory.ndim != 3:
        raise ValueError(f"{endpoint_split.split_name}.z_trajectory must be shaped (batch, steps, z_dim).")
    if z_trajectory.shape[0] != endpoint_split.z0.shape[0]:
        raise ValueError(f"{endpoint_split.split_name}.z_trajectory batch dimension must match z0.")
    if z_trajectory.shape[2] != endpoint_split.z0.shape[1]:
        raise ValueError(f"{endpoint_split.split_name}.z_trajectory z_dim must match z0.")
    if z_trajectory.shape[1] != expected_teacher_steps + 1:
        raise ValueError(
            f"{endpoint_split.split_name}.z_trajectory step axis must have length {expected_teacher_steps + 1}."
        )
    np.testing.assert_allclose(z_trajectory[:, 0, :], endpoint_split.z0, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(z_trajectory[:, -1, :], endpoint_split.z_star, atol=1e-12, rtol=1e-12)

    return FMPCIntervalSplit(
        split_name=endpoint_split.split_name,
        sample_indices=endpoint_split.sample_indices,
        target_onehot=endpoint_split.target_onehot,
        z0=endpoint_split.z0,
        z_star=endpoint_split.z_star,
        z_trajectory=z_trajectory.astype(np.float64, copy=False),
        teacher_steps=int(expected_teacher_steps),
        metadata={
            **endpoint_split.metadata,
            "trajectory_shape": list(z_trajectory.shape),
            "trajectory_includes_endpoints": True,
            "trajectory_axis_semantics": "(batch, step, z_dim)",
            "tau_definition": "tau_k = k / teacher_steps",
        },
    )


def load_fmpc_interval_dataset(
    path: str | Path,
    *,
    expected_dataset_name: str = "digits",
) -> FMPCIntervalDataset:
    """Load a trajectory-enabled Phase 5B interval dataset from teacher artifacts."""

    endpoint_dataset = load_fmpc_student_dataset(path, expected_dataset_name=expected_dataset_name)
    manifest = _read_json(endpoint_dataset.teacher_manifest_path)
    if not bool(manifest.get("export_trajectory", False)):
        raise ValueError("Phase 5B interval loading requires export_trajectory=True teacher artifacts.")
    teacher_steps = int(manifest["teacher_steps"])
    train_split = _load_interval_split(
        endpoint_dataset.train,
        manifest_path=endpoint_dataset.teacher_manifest_path,
        split_payload=manifest["splits"]["train"],
        expected_teacher_steps=teacher_steps,
    )
    val_split = _load_interval_split(
        endpoint_dataset.val,
        manifest_path=endpoint_dataset.teacher_manifest_path,
        split_payload=manifest["splits"]["val"],
        expected_teacher_steps=teacher_steps,
    )
    test_split = _load_interval_split(
        endpoint_dataset.test,
        manifest_path=endpoint_dataset.teacher_manifest_path,
        split_payload=manifest["splits"]["test"],
        expected_teacher_steps=teacher_steps,
    )

    return FMPCIntervalDataset(
        dataset_name=endpoint_dataset.dataset_name,
        schema_version=endpoint_dataset.schema_version,
        teacher_manifest_path=endpoint_dataset.teacher_manifest_path,
        teacher_checkpoint_path=endpoint_dataset.teacher_checkpoint_path,
        teacher_mode=endpoint_dataset.teacher_mode,
        teacher_target_semantics=endpoint_dataset.teacher_target_semantics,
        interval_input_definition="concat([z_s, target_onehot, tau_s, tau_t])",
        interval_target_definition="u_star = (z_t - z_s) / (tau_t - tau_s)",
        teacher_steps=teacher_steps,
        z_dim=endpoint_dataset.z_dim,
        target_dim=endpoint_dataset.target_dim,
        train=train_split,
        val=val_split,
        test=test_split,
        metadata={
            **endpoint_dataset.metadata,
            "trajectory_includes_endpoints": bool(manifest.get("trajectory_includes_endpoints", True)),
            "trajectory_axis_semantics": str(
                manifest.get("trajectory_axis_semantics", "(batch, step, z_dim)")
            ),
            "tau_definition": str(manifest.get("tau_definition", "tau_k = k / teacher_steps")),
        },
    )
