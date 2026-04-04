from __future__ import annotations

import numpy as np

from pc.fmpc_interval_data import FMPCIntervalSplit, teacher_step_aligned_rollout_schedules
from pc.fmpc_interval_student import rollout_interval_student


class _ClassVelocityStudent:
    def __init__(self, class_velocity: np.ndarray) -> None:
        self.class_velocity = np.asarray(class_velocity, dtype=np.float64)

    def predict_u(
        self,
        z_s: np.ndarray,
        target_onehot: np.ndarray,
        tau_s: np.ndarray | float,
        tau_t: np.ndarray | float,
    ) -> np.ndarray:
        labels = np.argmax(np.asarray(target_onehot, dtype=np.float64), axis=1)
        return np.asarray(self.class_velocity[labels], dtype=np.float64)


def _linear_interval_split() -> FMPCIntervalSplit:
    sample_indices = np.asarray([0, 1], dtype=np.int64)
    target_onehot = np.eye(2, dtype=np.float64)
    z0 = np.zeros((2, 2), dtype=np.float64)
    class_velocity = np.asarray([[1.0, -0.5], [-0.25, 0.75]], dtype=np.float64)

    trajectory = []
    for step in range(4):
        tau = step / 3.0
        trajectory.append(z0 + tau * class_velocity)
    z_trajectory = np.stack(trajectory, axis=1)

    return FMPCIntervalSplit(
        split_name="test",
        sample_indices=sample_indices,
        target_onehot=target_onehot,
        z0=z0,
        z_star=z_trajectory[:, -1, :],
        z_trajectory=z_trajectory,
        teacher_steps=3,
        metadata={},
    )


def test_rollout_interval_student_is_shape_correct_and_deterministic() -> None:
    split = _linear_interval_split()
    student = _ClassVelocityStudent(class_velocity=np.asarray([[1.0, -0.5], [-0.25, 0.75]], dtype=np.float64))
    knots = teacher_step_aligned_rollout_schedules(split.teacher_steps)["3-step"]

    first = rollout_interval_student(student, split, rollout_schedule_name="3-step", knots=knots)
    second = rollout_interval_student(student, split, rollout_schedule_name="3-step", knots=knots)

    assert first.final_state.shape == split.z_star.shape
    assert first.final_state.dtype == np.float64
    assert len(first.predicted_knot_states) == len(knots)
    np.testing.assert_allclose(first.final_state, split.z_star, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(second.final_state, first.final_state, atol=1e-12, rtol=1e-12)
    assert first.mean_knot_state_rms_gap == 0.0
    assert first.transport_wall_time_seconds >= 0.0
