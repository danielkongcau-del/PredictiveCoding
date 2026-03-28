from __future__ import annotations

from typing import get_type_hints

import numpy as np
import pytest

from pc.energy import compute_cache
from pc.inference import build_clamped_mask, compute_state_gradients, initialize_states
from pc.layers import init_mlp_layers
from pc.training import fit as fit_model, train_batch as train_batch_fn
from pc.training import apply_parameter_updates, parameter_gradients


@pytest.mark.parametrize("batch_size", [1, 4])
def test_cache_and_gradient_shapes(batch_size: int) -> None:
    layers = init_mlp_layers([3, 4, 2], seed=7)
    x = np.arange(batch_size * 3, dtype=np.float64).reshape(batch_size, 3) / 10.0
    y = np.arange(batch_size * 2, dtype=np.float64).reshape(batch_size, 2) / 10.0
    states = initialize_states(layers, x, y=y, init="forward", mode="train")
    cache = compute_cache(states, layers)
    clamped_mask = build_clamped_mask(len(layers) + 1, mode="train")
    state_gradients = compute_state_gradients(states, cache, layers, clamped_mask)
    weight_gradients, bias_gradients = parameter_gradients(states, cache, layers)

    assert states[0].shape == (batch_size, 3)
    assert states[1].shape == (batch_size, 4)
    assert states[2].shape == (batch_size, 2)
    assert cache.pre_activations[1].shape == (batch_size, 4)
    assert cache.predictions[1].shape == (batch_size, 4)
    assert cache.errors[1].shape == (batch_size, 4)
    assert cache.pre_activations[2].shape == (batch_size, 2)
    assert cache.predictions[2].shape == (batch_size, 2)
    assert cache.errors[2].shape == (batch_size, 2)
    assert state_gradients[0] is None
    assert state_gradients[1].shape == (batch_size, 4)
    assert state_gradients[2] is None
    assert weight_gradients[0].shape == (4, 3)
    assert bias_gradients[0].shape == (4,)
    assert weight_gradients[1].shape == (2, 4)
    assert bias_gradients[1].shape == (2,)

    original_shapes = [(layer.weight.shape, layer.bias.shape) for layer in layers]
    apply_parameter_updates(layers, weight_gradients, bias_gradients, eta_w=0.05, eta_b=0.05)
    updated_shapes = [(layer.weight.shape, layer.bias.shape) for layer in layers]
    assert updated_shapes == original_shapes


def test_bias_broadcasting_matches_manual_formula() -> None:
    layers = init_mlp_layers([2, 3], hidden_activation="identity", output_activation="identity", seed=3)
    x = np.array([[1.0, -2.0], [0.5, 1.5]], dtype=np.float64)
    states = initialize_states(layers, x, init="forward", mode="predict")
    cache = compute_cache(states, layers)
    manual = x @ layers[0].weight.T + layers[0].bias
    np.testing.assert_allclose(cache.pre_activations[1], manual)


def test_fixed_seed_reproduces_parameters() -> None:
    layers_a = init_mlp_layers([2, 3, 1], seed=11)
    layers_b = init_mlp_layers([2, 3, 1], seed=11)

    for layer_a, layer_b in zip(layers_a, layers_b, strict=True):
        np.testing.assert_allclose(layer_a.weight, layer_b.weight)
        np.testing.assert_allclose(layer_a.bias, layer_b.bias)
        assert layer_a.activation_name == layer_b.activation_name


def test_player_params_validation_rejects_invalid_shapes_and_settings() -> None:
    from pc.layers import PCLayerParams

    with pytest.raises(ValueError, match="weight must be"):
        PCLayerParams(
            weight=np.array([1.0, 2.0], dtype=np.float64),
            bias=np.array([0.0], dtype=np.float64),
            sigma2=1.0,
            activation_name="identity",
        )

    with pytest.raises(ValueError, match="bias must be"):
        PCLayerParams(
            weight=np.array([[1.0, 2.0]], dtype=np.float64),
            bias=np.array([[0.0]], dtype=np.float64),
            sigma2=1.0,
            activation_name="identity",
        )

    with pytest.raises(ValueError, match="bias.shape\\[0\\] must equal weight.shape\\[0\\]"):
        PCLayerParams(
            weight=np.array([[1.0, 2.0]], dtype=np.float64),
            bias=np.array([0.0, 1.0], dtype=np.float64),
            sigma2=1.0,
            activation_name="identity",
        )

    with pytest.raises(ValueError, match="sigma2 must be positive"):
        PCLayerParams(
            weight=np.array([[1.0, 2.0]], dtype=np.float64),
            bias=np.array([0.0], dtype=np.float64),
            sigma2=0.0,
            activation_name="identity",
        )

    with pytest.raises(ValueError, match="Unsupported activation"):
        PCLayerParams(
            weight=np.array([[1.0, 2.0]], dtype=np.float64),
            bias=np.array([0.0], dtype=np.float64),
            sigma2=1.0,
            activation_name="unsupported",
        )


def test_training_type_hints_can_be_resolved() -> None:
    train_batch_hints = get_type_hints(train_batch_fn)
    fit_hints = get_type_hints(fit_model)

    assert "model" in train_batch_hints
    assert "model" in fit_hints
