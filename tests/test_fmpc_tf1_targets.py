from __future__ import annotations

import numpy as np
import pytest

from pc.fmpc_tf1 import build_tf1_identity_target
from pc.fmpc_tf1_flow import (
    bootstrap_average_velocity_target,
    build_tf1_context,
)
from pc.layers import init_mlp_layers
from pc.models import PCNetwork


def _make_model() -> PCNetwork:
    return PCNetwork(
        layers=init_mlp_layers((4, 3, 2), seed=19, weight_scale=0.05),
        eta_x=0.1,
        eta_w=0.02,
        eta_b=0.02,
        train_steps=0,
        eval_steps=3,
        state_init="forward",
    )


def test_bootstrap_average_velocity_target_is_deterministic_and_finite() -> None:
    model = _make_model()
    x = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], dtype=np.float64)
    y = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    context = build_tf1_context(model, x, y)

    first = bootstrap_average_velocity_target(context, context.z0, t=0.0, r=1.0)
    second = bootstrap_average_velocity_target(context, context.z0, t=0.0, r=1.0)

    assert first.shape == context.z0.shape
    assert first.dtype == np.float64
    assert np.all(np.isfinite(first))
    np.testing.assert_allclose(first, second, atol=1e-12)


def test_bootstrap_average_velocity_target_rejects_invalid_remaining_horizon() -> None:
    model = _make_model()
    x = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float64)
    y = np.array([[1.0, 0.0]], dtype=np.float64)
    context = build_tf1_context(model, x, y)

    with pytest.raises(ValueError):
        bootstrap_average_velocity_target(context, context.z0, t=0.5, r=0.0)
    with pytest.raises(ValueError):
        bootstrap_average_velocity_target(context, context.z0, t=0.5, r=0.6)


def test_tf1_identity_target_matches_g_plus_remaining_horizon_times_jvp() -> None:
    g_t = np.array([[0.1, -0.2], [0.3, 0.4]], dtype=np.float64)
    jvp = np.array([[1.0, 2.0], [-1.0, 0.5]], dtype=np.float64)

    target = build_tf1_identity_target(g_t, 0.75, jvp)
    expected = g_t + (0.75 * jvp)
    np.testing.assert_allclose(target, expected, atol=1e-12)
