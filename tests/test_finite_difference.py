from __future__ import annotations

import numpy as np

from pc.energy import compute_cache, total_energy
from pc.inference import compute_state_gradients
from pc.layers import PCLayerParams
from pc.training import parameter_gradients


def make_identity_setup() -> tuple[list[PCLayerParams], list[np.ndarray]]:
    layers = [
        PCLayerParams(
            weight=np.array([[0.7]], dtype=np.float64),
            bias=np.array([0.05], dtype=np.float64),
            sigma2=1.0,
            activation_name="identity",
        ),
        PCLayerParams(
            weight=np.array([[-0.3]], dtype=np.float64),
            bias=np.array([-0.02], dtype=np.float64),
            sigma2=1.0,
            activation_name="identity",
        ),
    ]
    states = [
        np.array([[0.2]], dtype=np.float64),
        np.array([[0.1]], dtype=np.float64),
        np.array([[-0.4]], dtype=np.float64),
    ]
    return layers, states


def test_finite_difference_matches_hidden_state_and_weight_gradients() -> None:
    layers, states = make_identity_setup()
    cache = compute_cache(states, layers)
    clamped_mask = [True, False, False]
    state_gradients = compute_state_gradients(states, cache, layers, clamped_mask)
    weight_gradients, _ = parameter_gradients(states, cache, layers)

    epsilon = 1e-6

    def energy_with_hidden(hidden_value: float) -> float:
        perturbed_states = [state.copy() for state in states]
        perturbed_states[1][0, 0] = hidden_value
        perturbed_cache = compute_cache(perturbed_states, layers)
        return total_energy(perturbed_cache, layers, batch_size=1)

    hidden_fd = (
        energy_with_hidden(states[1][0, 0] + epsilon) - energy_with_hidden(states[1][0, 0] - epsilon)
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(state_gradients[1][0, 0], hidden_fd, rtol=1e-6, atol=1e-6)

    def energy_with_weight(weight_value: float) -> float:
        perturbed_layers = [
            PCLayerParams(
                weight=layer.weight.copy(),
                bias=layer.bias.copy(),
                sigma2=layer.sigma2,
                activation_name=layer.activation_name,
            )
            for layer in layers
        ]
        perturbed_layers[0].weight[0, 0] = weight_value
        perturbed_cache = compute_cache(states, perturbed_layers)
        return total_energy(perturbed_cache, perturbed_layers, batch_size=1)

    weight_fd = (
        energy_with_weight(layers[0].weight[0, 0] + epsilon)
        - energy_with_weight(layers[0].weight[0, 0] - epsilon)
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(weight_gradients[0][0, 0], weight_fd, rtol=1e-6, atol=1e-6)
