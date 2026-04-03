from __future__ import annotations

import numpy as np

from pc.fmpc_student_data import FMPCStudentSplit, build_fmpc_student_inputs
from pc.fmpc_student_normalization import fit_fmpc_student_normalization


def _make_split() -> FMPCStudentSplit:
    z0 = np.asarray(
        [
            [1.0, 10.0],
            [3.0, 14.0],
            [5.0, 18.0],
        ],
        dtype=np.float64,
    )
    target_onehot = np.eye(3, dtype=np.float64)
    z_star = z0 + np.asarray(
        [
            [0.5, -1.0],
            [1.0, -0.5],
            [1.5, 0.0],
        ],
        dtype=np.float64,
    )
    return FMPCStudentSplit(
        split_name="train",
        sample_indices=np.arange(3, dtype=np.int64),
        target_onehot=target_onehot,
        z0=z0,
        z_star=z_star,
        delta_z=z_star - z0,
        student_inputs=build_fmpc_student_inputs(z0, target_onehot),
        metadata={},
    )


def test_fmpc_student_normalization_roundtrip_recovers_original_delta_z() -> None:
    train_split = _make_split()
    normalization = fit_fmpc_student_normalization(train_split, eps=1e-8)

    delta_z_normalized = normalization.transform_delta_z(train_split.delta_z)
    delta_z_recovered = normalization.inverse_delta_z(delta_z_normalized)

    np.testing.assert_allclose(delta_z_recovered, train_split.delta_z, atol=1e-12, rtol=1e-12)


def test_fmpc_student_normalization_only_normalizes_z0_part_of_inputs() -> None:
    train_split = _make_split()
    normalization = fit_fmpc_student_normalization(train_split, eps=1e-8)

    transformed_inputs = normalization.transform_split_inputs(train_split)
    expected_z0 = normalization.transform_z0(train_split.z0)

    np.testing.assert_allclose(transformed_inputs[:, : train_split.z0.shape[1]], expected_z0)
    np.testing.assert_allclose(
        transformed_inputs[:, train_split.z0.shape[1] :],
        train_split.target_onehot,
    )
    assert transformed_inputs.dtype == np.float64
