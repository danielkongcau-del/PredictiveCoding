from __future__ import annotations

import numpy as np

from pc.energy import compute_cache
from pc.fmpc_tf1_flow import (
    build_tf1_context,
    hidden_energy_from_state,
    hidden_local_flow,
    teacher_free_state_features,
)
from pc.inference import compute_state_gradients
from pc.layers import init_mlp_layers
from pc.models import PCNetwork


def _make_model() -> PCNetwork:
    return PCNetwork(
        layers=init_mlp_layers((4, 3, 2), seed=7, weight_scale=0.05),
        eta_x=0.1,
        eta_w=0.02,
        eta_b=0.02,
        train_steps=0,
        eval_steps=3,
        state_init="forward",
    )


def test_hidden_local_flow_matches_negative_flattened_hidden_gradient() -> None:
    model = _make_model()
    x = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], dtype=np.float64)
    y = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    context = build_tf1_context(model, x, y)
    flow = hidden_local_flow(context, context.z0)

    cache = compute_cache(context.states_template, model.layers)
    gradients = compute_state_gradients(
        context.states_template,
        cache,
        model.layers,
        context.clamped_mask,
    )
    expected = -np.asarray(gradients[1], dtype=np.float64)
    np.testing.assert_allclose(flow, expected, atol=1e-12)


def test_hidden_energy_and_teacher_free_state_features_have_expected_shapes() -> None:
    model = _make_model()
    x = np.array([[0.0, 0.1, 0.2, 0.3], [0.3, 0.2, 0.1, 0.0]], dtype=np.float64)
    y = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    context = build_tf1_context(model, x, y)
    energy = hidden_energy_from_state(context, context.z0)
    assert np.isfinite(energy)

    features = teacher_free_state_features(context, context.z0)
    assert features.g_t.shape == context.z0.shape
    assert features.e_out_t.shape == y.shape
    assert features.F_t.shape == (x.shape[0], 1)
    assert features.y_hat_t.shape == y.shape
    assert features.g_t.dtype == np.float64
    assert features.e_out_t.dtype == np.float64
    assert features.F_t.dtype == np.float64
