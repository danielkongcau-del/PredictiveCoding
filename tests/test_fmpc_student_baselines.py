from __future__ import annotations

import numpy as np

from pc.fmpc_student_baselines import ClassMeanDeltaStudent, RidgeDeltaStudent, RidgeDeltaStudentConfig
from pc.fmpc_student_data import FMPCStudentSplit, build_fmpc_student_inputs
from pc.fmpc_student_normalization import fit_fmpc_student_normalization


def _make_split(
    *,
    split_name: str,
    z0: np.ndarray,
    target_onehot: np.ndarray,
    delta_z: np.ndarray,
) -> FMPCStudentSplit:
    z0_array = np.asarray(z0, dtype=np.float64)
    targets = np.asarray(target_onehot, dtype=np.float64)
    delta = np.asarray(delta_z, dtype=np.float64)
    z_star = z0_array + delta
    return FMPCStudentSplit(
        split_name=split_name,
        sample_indices=np.arange(z0_array.shape[0], dtype=np.int64),
        target_onehot=targets,
        z0=z0_array,
        z_star=z_star,
        delta_z=delta,
        student_inputs=build_fmpc_student_inputs(z0_array, targets),
        metadata={},
    )


def test_class_mean_delta_student_uses_train_class_prototypes() -> None:
    train_split = _make_split(
        split_name="train",
        z0=np.zeros((4, 2), dtype=np.float64),
        target_onehot=np.eye(2, dtype=np.float64)[[0, 0, 1, 1]],
        delta_z=np.asarray(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [10.0, 20.0],
                [14.0, 24.0],
            ],
            dtype=np.float64,
        ),
    )
    val_split = _make_split(
        split_name="val",
        z0=np.zeros((2, 2), dtype=np.float64),
        target_onehot=np.eye(2, dtype=np.float64)[[1, 0]],
        delta_z=np.zeros((2, 2), dtype=np.float64),
    )

    model = ClassMeanDeltaStudent.fit(train_split)
    predictions = model.predict_delta_z(val_split)

    expected = np.asarray(
        [
            [12.0, 22.0],
            [2.0, 3.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(predictions, expected)
    assert predictions.dtype == np.float64


def test_ridge_delta_student_fit_predict_preserves_shape_and_dtype() -> None:
    train_z0 = np.asarray(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=np.float64,
    )
    train_targets = np.eye(2, dtype=np.float64)[[0, 1, 0, 1]]
    train_delta = np.asarray(
        [
            [1.0, 0.5],
            [0.5, 1.0],
            [3.0, 0.5],
            [0.5, 3.0],
        ],
        dtype=np.float64,
    )
    train_split = _make_split(
        split_name="train",
        z0=train_z0,
        target_onehot=train_targets,
        delta_z=train_delta,
    )
    val_split = _make_split(
        split_name="val",
        z0=np.asarray([[3.0, 1.0], [1.0, 3.0]], dtype=np.float64),
        target_onehot=np.eye(2, dtype=np.float64)[[0, 1]],
        delta_z=np.zeros((2, 2), dtype=np.float64),
    )

    normalization = fit_fmpc_student_normalization(train_split)
    model = RidgeDeltaStudent.fit(
        train_split,
        normalization=normalization,
        config=RidgeDeltaStudentConfig(alpha=1e-4),
    )
    predictions = model.predict_delta_z(val_split)

    assert predictions.shape == val_split.delta_z.shape
    assert predictions.dtype == np.float64
