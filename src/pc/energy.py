from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .activations import get_activation
from .layers import PCLayerParams
from .utils import assert_shape, ensure_finite_array, ensure_finite_collection


@dataclass
class PCCache:
    """Cached a^l, mu^l, and e^l lists aligned to layer indices 0..L."""

    pre_activations: list[np.ndarray | None]
    predictions: list[np.ndarray | None]
    errors: list[np.ndarray | None]


def compute_cache(states: list[np.ndarray], layers: list[PCLayerParams]) -> PCCache:
    """Compute a^l, mu^l, and e^l from states x^0..x^L using batch-first arrays."""
    if len(states) != len(layers) + 1:
        raise ValueError(
            f"states must have length {len(layers) + 1}, received {len(states)}."
        )

    batch_size = states[0].shape[0]
    pre_activations: list[np.ndarray | None] = [None] * len(states)
    predictions: list[np.ndarray | None] = [None] * len(states)
    errors: list[np.ndarray | None] = [None] * len(states)

    for layer_index, layer in enumerate(layers, start=1):
        previous_state = states[layer_index - 1]
        current_state = states[layer_index]
        if previous_state.ndim != 2 or current_state.ndim != 2:
            raise ValueError("All states must be rank-2 arrays shaped (batch, features).")
        if previous_state.shape[0] != batch_size or current_state.shape[0] != batch_size:
            raise ValueError("All states must share the same batch size.")
        assert_shape(
            previous_state,
            (batch_size, layer.weight.shape[1]),
            f"x^{layer_index - 1}",
        )
        assert_shape(
            current_state,
            (batch_size, layer.weight.shape[0]),
            f"x^{layer_index}",
        )

        activation, _ = get_activation(layer.activation_name)
        a_l = previous_state @ layer.weight.T + layer.bias
        mu_l = activation(a_l)
        e_l = current_state - mu_l

        ensure_finite_array(a_l, f"a^{layer_index}")
        ensure_finite_array(mu_l, f"mu^{layer_index}")
        ensure_finite_array(e_l, f"e^{layer_index}")

        pre_activations[layer_index] = a_l
        predictions[layer_index] = mu_l
        errors[layer_index] = e_l

    ensure_finite_collection(pre_activations, "pre_activations")
    ensure_finite_collection(predictions, "predictions")
    ensure_finite_collection(errors, "errors")
    return PCCache(
        pre_activations=pre_activations,
        predictions=predictions,
        errors=errors,
    )


def total_energy(cache: PCCache, layers: list[PCLayerParams], batch_size: int) -> float:
    """Compute the scalar baseline energy for a batch of size B."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    energy = 0.0
    for layer_index, layer in enumerate(layers, start=1):
        if layer.sigma2 <= 0.0:
            raise ValueError("sigma2 must be positive for every layer.")
        error = cache.errors[layer_index]
        if error is None:
            raise ValueError(f"cache.errors[{layer_index}] is missing.")
        energy += np.sum(error * error) / (2.0 * batch_size * layer.sigma2)
    return float(energy)
