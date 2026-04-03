from __future__ import annotations

import numpy as np

from pc.fmpc_student import init_fmpc_student_transporter
from pc.fmpc_student_data import build_fmpc_student_inputs


def test_fmpc_student_transporter_predicts_batch_first_float64() -> None:
    transporter = init_fmpc_student_transporter(
        z_dim=4,
        target_dim=3,
        hidden_dims=(5,),
        hidden_activation="tanh",
        output_activation="identity",
        weight_scale=0.05,
        eta_w=0.01,
        eta_b=0.01,
        seed=7,
    )
    z0 = np.arange(8, dtype=np.float64).reshape(2, 4)
    targets = np.eye(3, dtype=np.float64)[[0, 1]]
    student_inputs = build_fmpc_student_inputs(z0, targets)

    delta_z_hat = transporter.predict_delta_z(student_inputs)
    z_hat = transporter.predict_z_hat(
        type(
            "SplitStub",
            (),
            {"z0": z0, "target_onehot": targets, "student_inputs": student_inputs},
        )()
    )

    assert delta_z_hat.shape == (2, 4)
    assert delta_z_hat.dtype == np.float64
    assert z_hat.shape == (2, 4)
    assert z_hat.dtype == np.float64
    np.testing.assert_allclose(z_hat, z0 + delta_z_hat)


def test_fmpc_student_transporter_init_is_deterministic() -> None:
    first = init_fmpc_student_transporter(
        z_dim=4,
        target_dim=3,
        hidden_dims=(5,),
        hidden_activation="tanh",
        output_activation="identity",
        weight_scale=0.05,
        eta_w=0.01,
        eta_b=0.01,
        seed=11,
    )
    second = init_fmpc_student_transporter(
        z_dim=4,
        target_dim=3,
        hidden_dims=(5,),
        hidden_activation="tanh",
        output_activation="identity",
        weight_scale=0.05,
        eta_w=0.01,
        eta_b=0.01,
        seed=11,
    )
    z0 = np.arange(12, dtype=np.float64).reshape(3, 4)
    targets = np.eye(3, dtype=np.float64)
    inputs = build_fmpc_student_inputs(z0, targets)

    np.testing.assert_allclose(first.predict_delta_z(inputs), second.predict_delta_z(inputs))
