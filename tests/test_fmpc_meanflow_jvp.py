from __future__ import annotations

import numpy as np

from pc.fmpc_interval_normalization import FMPCIntervalNormalizationStats
from pc.fmpc_meanflow_jvp import (
    build_meanflow_input_tangent,
    forward_mlp_with_jvp,
)
from pc.mlp_baseline import MLPNetwork, init_mlp_baseline_layers


def test_forward_mlp_with_jvp_matches_finite_difference() -> None:
    network = MLPNetwork(
        layers=init_mlp_baseline_layers(
            (5, 4, 3),
            hidden_activation="tanh",
            output_activation="identity",
            weight_scale=0.1,
            seed=7,
            dtype=np.float64,
        ),
        eta_w=0.01,
    )
    inputs = np.asarray(
        [[0.2, -0.1, 0.3, 0.5, -0.4], [-0.3, 0.4, -0.2, 0.1, 0.25]],
        dtype=np.float64,
    )
    tangent = np.asarray(
        [[0.05, -0.02, 0.03, -0.04, 0.01], [-0.01, 0.03, -0.02, 0.05, -0.04]],
        dtype=np.float64,
    )
    result = forward_mlp_with_jvp(network, inputs, tangent)

    eps = 1e-6
    forward_plus = network.predict(inputs + eps * tangent)
    forward_minus = network.predict(inputs - eps * tangent)
    finite_difference = (forward_plus - forward_minus) / (2.0 * eps)

    np.testing.assert_allclose(result.jvp, finite_difference, atol=1e-6, rtol=1e-5)


def test_build_meanflow_input_tangent_can_include_feature_aware_teacher_block() -> None:
    normalization = FMPCIntervalNormalizationStats(
        z_state_mean=np.zeros((2,), dtype=np.float64),
        z_state_std=np.asarray([2.0, 4.0], dtype=np.float64),
        u_mean=np.zeros((2,), dtype=np.float64),
        u_std=np.ones((2,), dtype=np.float64),
        teacher_feature_mean=np.zeros((3,), dtype=np.float64),
        teacher_feature_std=np.asarray([2.0, 5.0, 10.0], dtype=np.float64),
        teacher_feature_names=("g_s", "e_out_s", "F_s"),
        eps=1e-8,
    )
    g_s = np.asarray([[2.0, -4.0], [1.0, 8.0]], dtype=np.float64)
    teacher_feature_tangent = np.asarray(
        [[0.2, -0.5, 1.0], [0.4, 1.5, -2.0]],
        dtype=np.float64,
    )
    tangent = build_meanflow_input_tangent(
        normalization,
        g_s,
        target_dim=3,
        teacher_feature_dim=3,
        teacher_feature_tangent=teacher_feature_tangent,
        d_tau_s=1.0,
        d_tau_t=0.0,
    )

    assert tangent.shape == (2, 10)
    np.testing.assert_allclose(tangent[:, :2], np.asarray([[1.0, -1.0], [0.5, 2.0]], dtype=np.float64))
    np.testing.assert_allclose(tangent[:, 2:5], 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(tangent[:, 5:6], 1.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(tangent[:, 6:7], 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        tangent[:, 7:],
        teacher_feature_tangent / np.asarray([2.0, 5.0, 10.0], dtype=np.float64),
        atol=1e-12,
        rtol=1e-12,
    )
