from __future__ import annotations

import numpy as np

from pc.fmpc_interval_data import FMPCIntervalSplit
from pc.fmpc_interval_features import FMPCIntervalTeacherTrajectoryFeatures
from pc.fmpc_interval_normalization import (
    fit_fmpc_interval_augmented_normalization,
    fit_fmpc_interval_normalization,
)


def _synthetic_interval_split() -> FMPCIntervalSplit:
    sample_indices = np.asarray([0, 1], dtype=np.int64)
    target_onehot = np.eye(2, dtype=np.float64)
    z0 = np.asarray(
        [
            [0.0, 1.0],
            [2.0, 3.0],
        ],
        dtype=np.float64,
    )
    z1 = z0 + np.asarray([[1.0, -1.0], [1.0, -1.0]], dtype=np.float64)
    z2 = z0 + 2.0 * np.asarray([[1.0, -1.0], [1.0, -1.0]], dtype=np.float64)
    z3 = z0 + 3.0 * np.asarray([[1.0, -1.0], [1.0, -1.0]], dtype=np.float64)
    z_trajectory = np.stack([z0, z1, z2, z3], axis=1)
    return FMPCIntervalSplit(
        split_name="train",
        sample_indices=sample_indices,
        target_onehot=target_onehot,
        z0=z0,
        z_star=z3,
        z_trajectory=z_trajectory,
        teacher_steps=3,
        metadata={},
    )


def test_fmpc_interval_normalization_roundtrip() -> None:
    split = _synthetic_interval_split()
    stats = fit_fmpc_interval_normalization(split, eps=1e-8)

    z_s = split.z_trajectory[:, 1, :]
    u = np.asarray([[3.0, -3.0], [3.0, -3.0]], dtype=np.float64)

    z_transformed = stats.transform_z_state(z_s)
    u_transformed = stats.transform_u(u)
    u_roundtrip = stats.inverse_u(u_transformed)
    transformed_inputs = stats.transform_inputs(
        z_s,
        split.target_onehot,
        tau_s=np.asarray([1.0 / 3.0, 1.0 / 3.0], dtype=np.float64),
        tau_t=np.asarray([1.0, 1.0], dtype=np.float64),
    )

    assert z_transformed.shape == z_s.shape
    assert z_transformed.dtype == np.float64
    assert u_transformed.shape == u.shape
    assert u_transformed.dtype == np.float64
    np.testing.assert_allclose(u_roundtrip, u, atol=1e-12, rtol=1e-12)
    assert transformed_inputs.shape == (2, split.z_dim + split.target_dim + 2)
    np.testing.assert_allclose(transformed_inputs[:, -2], np.asarray([1.0 / 3.0, 1.0 / 3.0]), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(transformed_inputs[:, -1], np.asarray([1.0, 1.0]), atol=0.0, rtol=0.0)


def test_fmpc_interval_augmented_normalization_roundtrip() -> None:
    split = _synthetic_interval_split()
    trajectory_features = FMPCIntervalTeacherTrajectoryFeatures(
        split_name="train",
        teacher_steps=split.teacher_steps,
        y_hat_trajectory=np.stack([split.target_onehot] * (split.teacher_steps + 1), axis=1),
        e_out_trajectory=np.stack([np.full_like(split.target_onehot, 0.5)] * (split.teacher_steps + 1), axis=1),
        g_trajectory=np.stack([np.full_like(split.z0, 2.0)] * (split.teacher_steps + 1), axis=1),
        F_trajectory=np.stack(
            [np.full((split.num_samples, 1), float(step + 1), dtype=np.float64) for step in range(split.teacher_steps + 1)],
            axis=1,
        ),
    )
    stats = fit_fmpc_interval_augmented_normalization(
        split,
        trajectory_features=trajectory_features,
        selected_feature_names=("g_s", "e_out_s", "F_s"),
        target_mode="u_residual_local_field",
        knot_focus_mixture=0.5,
        eps=1e-8,
    )

    z_s = split.z_trajectory[:, 1, :]
    teacher_features = np.concatenate(
        [
            trajectory_features.g_trajectory[:, 1, :],
            trajectory_features.e_out_trajectory[:, 1, :],
            trajectory_features.F_trajectory[:, 1, :],
        ],
        axis=1,
    )
    u_res = np.asarray([[1.0, -1.0], [1.0, -1.0]], dtype=np.float64)

    transformed_inputs = stats.transform_inputs(
        z_s,
        split.target_onehot,
        tau_s=np.asarray([1.0 / 3.0, 1.0 / 3.0], dtype=np.float64),
        tau_t=np.asarray([1.0, 1.0], dtype=np.float64),
        teacher_features=teacher_features,
    )
    u_res_transformed = stats.transform_u(u_res)
    u_res_roundtrip = stats.inverse_u(u_res_transformed)

    assert transformed_inputs.shape == (2, split.z_dim + split.target_dim + 2 + teacher_features.shape[1])
    assert transformed_inputs.dtype == np.float64
    assert stats.teacher_feature_mean.shape[0] == teacher_features.shape[1]
    np.testing.assert_allclose(u_res_roundtrip, u_res, atol=1e-12, rtol=1e-12)
