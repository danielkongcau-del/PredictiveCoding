from __future__ import annotations

import numpy as np

from pc.fmpc_tf1_jvp import (
    build_tf1_input,
    build_tf1_input_tangent,
    forward_tf1_mlp_with_jvp,
)
from pc.fmpc_tf1_flow import FMPCTF1StateFeatures, FMPCTF1StateFeatureTangents
from pc.mlp_baseline import MLPNetwork, init_mlp_baseline_layers


def test_tf1_core_input_tangent_uses_fixed_terminal_time_direction() -> None:
    z_t = np.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]], dtype=np.float64)
    target_onehot = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    g_t = np.array([[0.01, -0.02, 0.03], [-0.04, 0.05, -0.06]], dtype=np.float64)

    inputs = build_tf1_input(
        z_t,
        target_onehot,
        t=0.25,
        r=0.75,
        use_teacher_free_features=False,
    )
    tangent = build_tf1_input_tangent(
        g_t,
        target_dim=2,
        use_teacher_free_features=False,
    )

    assert inputs.shape == (2, 7)
    assert tangent.shape == (2, 7)
    np.testing.assert_allclose(tangent[:, :3], g_t)
    np.testing.assert_allclose(tangent[:, 3:5], 0.0)
    np.testing.assert_allclose(tangent[:, 5], 1.0)
    np.testing.assert_allclose(tangent[:, 6], -1.0)


def test_forward_tf1_mlp_with_jvp_matches_finite_difference() -> None:
    network = MLPNetwork(
        layers=init_mlp_baseline_layers((7, 5, 3), seed=11, weight_scale=0.05),
        eta_w=0.01,
        eta_b=0.01,
    )
    z_t = np.array([[0.2, -0.1, 0.4], [0.1, 0.3, -0.2]], dtype=np.float64)
    target_onehot = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    g_t = np.array([[0.03, -0.01, 0.02], [-0.02, 0.01, 0.04]], dtype=np.float64)

    inputs = build_tf1_input(
        z_t,
        target_onehot,
        t=0.2,
        r=0.8,
        use_teacher_free_features=False,
    )
    tangent = build_tf1_input_tangent(
        g_t,
        target_dim=2,
        use_teacher_free_features=False,
    )
    result = forward_tf1_mlp_with_jvp(network, inputs, tangent)

    epsilon = 1e-6
    forward_plus = network.predict(inputs + (epsilon * tangent))
    forward_minus = network.predict(inputs - (epsilon * tangent))
    finite_difference = (forward_plus - forward_minus) / (2.0 * epsilon)
    np.testing.assert_allclose(result.jvp, finite_difference, atol=1e-7)


def test_tf1_augmented_input_and_tangent_include_feature_blocks() -> None:
    z_t = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)
    target_onehot = np.array([[1.0, 0.0]], dtype=np.float64)
    features = FMPCTF1StateFeatures(
        g_t=np.array([[0.01, 0.02, 0.03]], dtype=np.float64),
        e_out_t=np.array([[0.4, -0.5]], dtype=np.float64),
        F_t=np.array([[0.6]], dtype=np.float64),
        y_hat_t=np.array([[0.6, 0.4]], dtype=np.float64),
    )
    tangents = FMPCTF1StateFeatureTangents(
        Dg_g_t=np.array([[0.7, 0.8, 0.9]], dtype=np.float64),
        Dg_e_out_t=np.array([[1.0, 1.1]], dtype=np.float64),
        Dg_F_t=np.array([[1.2]], dtype=np.float64),
        Dg_y_hat_t=np.array([[-1.0, -1.1]], dtype=np.float64),
    )

    inputs = build_tf1_input(
        z_t,
        target_onehot,
        t=0.0,
        r=1.0,
        use_teacher_free_features=True,
        features=features,
    )
    tangent = build_tf1_input_tangent(
        features.g_t,
        target_dim=2,
        use_teacher_free_features=True,
        feature_aware_tangents=True,
        feature_tangents=tangents,
    )

    assert inputs.shape == (1, 13)
    assert tangent.shape == (1, 13)
    np.testing.assert_allclose(inputs[:, -6:], np.array([[0.01, 0.02, 0.03, 0.4, -0.5, 0.6]]))
    np.testing.assert_allclose(tangent[:, -6:], np.array([[0.7, 0.8, 0.9, 1.0, 1.1, 1.2]]))
