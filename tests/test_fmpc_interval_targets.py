from __future__ import annotations

import numpy as np

from pc.fmpc_interval_data import FMPCIntervalSplit, compute_interval_velocity_target, sample_balanced_interval_batch
from pc.fmpc_interval_normalization import fit_fmpc_interval_normalization
from pc.fmpc_interval_student import (
    IntervalRidgeStudent,
    IntervalRidgeStudentConfig,
    IntervalStandardizedMLPStudent,
    IntervalStandardizedMLPStudentConfig,
    build_interval_residual_target,
    build_rollout_auxiliary_batches,
    reconstruct_interval_velocity_from_residual,
)


def _synthetic_interval_split() -> FMPCIntervalSplit:
    sample_indices = np.asarray([0, 1, 2], dtype=np.int64)
    target_onehot = np.eye(3, dtype=np.float64)
    z0 = np.asarray(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [2.0, -1.0],
        ],
        dtype=np.float64,
    )
    velocity = np.asarray(
        [
            [0.6, -0.3],
            [0.2, 0.4],
            [-0.5, 0.1],
        ],
        dtype=np.float64,
    )
    z_trajectory = []
    for step in range(4):
        tau = step / 3.0
        z_trajectory.append(z0 + tau * velocity)
    return FMPCIntervalSplit(
        split_name="train",
        sample_indices=sample_indices,
        target_onehot=target_onehot,
        z0=z0,
        z_star=z_trajectory[-1],
        z_trajectory=np.stack(z_trajectory, axis=1),
        teacher_steps=3,
        metadata={},
    )


def test_compute_interval_velocity_target_matches_average_velocity_formula() -> None:
    z_s = np.asarray([[0.0, 0.0], [1.0, 2.0]], dtype=np.float64)
    z_t = np.asarray([[1.0, 3.0], [3.0, 6.0]], dtype=np.float64)
    tau_s = np.asarray([0.0, 0.25], dtype=np.float64)
    tau_t = np.asarray([0.5, 1.0], dtype=np.float64)

    u_star = compute_interval_velocity_target(z_s, z_t, tau_s, tau_t)

    expected = np.asarray([[2.0, 6.0], [8.0 / 3.0, 16.0 / 3.0]], dtype=np.float64)
    np.testing.assert_allclose(u_star, expected, atol=1e-12, rtol=1e-12)


def test_sampled_interval_batch_uses_average_velocity_target() -> None:
    split = _synthetic_interval_split()
    batch = sample_balanced_interval_batch(split, batch_size=12, seed=42)

    expected = (batch.z_t - batch.z_s) / batch.delta_tau
    np.testing.assert_allclose(batch.u_star, expected, atol=1e-12, rtol=1e-12)


def test_interval_ridge_fit_predict_shape_and_dtype() -> None:
    split = _synthetic_interval_split()
    normalization = fit_fmpc_interval_normalization(split, eps=1e-8)
    model = IntervalRidgeStudent.fit(
        split,
        normalization=normalization,
        config=IntervalRidgeStudentConfig(alpha=1e-4),
    )

    tau_s = np.asarray([0.0, 1.0 / 3.0, 2.0 / 3.0], dtype=np.float64)
    tau_t = np.asarray([1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=np.float64)
    prediction = model.predict_u(split.z_trajectory[:, 0, :], split.target_onehot, tau_s, tau_t)

    assert prediction.shape == (3, split.z_dim)
    assert prediction.dtype == np.float64


def test_interval_standardized_mlp_predict_shape_and_dtype() -> None:
    split = _synthetic_interval_split()
    normalization = fit_fmpc_interval_normalization(split, eps=1e-8)
    model = IntervalStandardizedMLPStudent.initialize(
        z_dim=split.z_dim,
        target_dim=split.target_dim,
        normalization=normalization,
        config=IntervalStandardizedMLPStudentConfig(
            hidden_dims=(8,),
            epochs=1,
            batch_size=3,
            eta_w=0.01,
        ),
        seed=123,
    )

    tau_s = np.asarray([0.0, 1.0 / 3.0, 2.0 / 3.0], dtype=np.float64)
    tau_t = np.asarray([1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=np.float64)
    prediction = model.predict_u(split.z_trajectory[:, 0, :], split.target_onehot, tau_s, tau_t)

    assert prediction.shape == (3, split.z_dim)
    assert prediction.dtype == np.float64


class _ZeroVelocityStudent:
    def predict_u(
        self,
        z_s: np.ndarray,
        target_onehot: np.ndarray,
        tau_s: np.ndarray | float,
        tau_t: np.ndarray | float,
    ) -> np.ndarray:
        return np.zeros_like(np.asarray(z_s, dtype=np.float64))


def test_rollout_auxiliary_batches_use_predicted_states_for_later_segments() -> None:
    split = _synthetic_interval_split()
    auxiliary = build_rollout_auxiliary_batches(
        _ZeroVelocityStudent(),
        split,
        rollout_schedule_name="2-step",
        knots=(0, 2, 3),
    )

    assert len(auxiliary.batches) == 2
    first_batch, second_batch = auxiliary.batches
    np.testing.assert_allclose(first_batch.z_s, split.z0, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(second_batch.z_s, split.z0, atol=1e-12, rtol=1e-12)
    assert not np.allclose(second_batch.z_s, split.z_trajectory[:, 2, :], atol=1e-12, rtol=1e-12)
    expected_second_target = (split.z_trajectory[:, 3, :] - split.z0) / (1.0 / 3.0)
    np.testing.assert_allclose(second_batch.u_star, expected_second_target, atol=1e-12, rtol=1e-12)
    assert auxiliary.intermediate_state_rms_gap is not None
    assert auxiliary.final_state_rms_gap >= 0.0


def test_interval_residual_target_roundtrip_is_exact() -> None:
    u_star = np.asarray([[1.0, 2.0], [3.0, -1.0]], dtype=np.float64)
    g_s = np.asarray([[0.25, 0.5], [1.0, -0.25]], dtype=np.float64)

    u_res = build_interval_residual_target(u_star, g_s)
    reconstructed = reconstruct_interval_velocity_from_residual(g_s, u_res)

    np.testing.assert_allclose(u_res, u_star - g_s, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(reconstructed, u_star, atol=1e-12, rtol=1e-12)
