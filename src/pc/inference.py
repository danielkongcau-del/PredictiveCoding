from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .activations import get_activation
from .energy import PCCache, compute_cache, total_energy
from .layers import PCLayerParams
from .utils import ensure_finite_array


@dataclass
class InferenceResult:
    """Final inference states, cache, and energy trace for x^0..x^L."""

    states: list[np.ndarray]
    cache: PCCache
    energy_trace: list[float]
    final_energy: float


def build_clamped_mask(num_states: int, mode: str) -> list[bool]:
    """Build the Phase 0 layer-level clamped mask for states x^0..x^L."""
    if num_states < 2:
        raise ValueError("num_states must be at least 2.")
    if mode == "train":
        return [True] + [False] * (num_states - 2) + [True]
    if mode == "predict":
        return [True] + [False] * (num_states - 1)
    raise ValueError(f"Unsupported mode '{mode}'.")


def initialize_states(
    layers: list[PCLayerParams],
    x: np.ndarray,
    y: np.ndarray | None = None,
    init: str = "forward",
    mode: str = "train",
) -> list[np.ndarray]:
    """Initialize states x^0..x^L with x shaped (B, d_0) and optional y shaped (B, d_L)."""
    if x.ndim != 2:
        raise ValueError("x must be shaped (batch, features).")
    if len(layers) == 0:
        raise ValueError("Phase 0 requires at least one predictive layer.")

    x_array = np.asarray(x, dtype=np.float64)
    batch_size = x_array.shape[0]
    clamped_mask = build_clamped_mask(len(layers) + 1, mode)
    states: list[np.ndarray] = [x_array.copy()]

    if mode == "train":
        if y is None:
            raise ValueError("y must be provided in training mode.")
        y_array = np.asarray(y, dtype=np.float64)
        expected_output_dim = layers[-1].weight.shape[0]
        if y_array.shape != (batch_size, expected_output_dim):
            raise ValueError(
                f"y must have shape {(batch_size, expected_output_dim)}, "
                f"received {y_array.shape}."
            )
    else:
        y_array = None

    previous_state = states[0]
    for layer_index, layer in enumerate(layers, start=1):
        output_dim = layer.weight.shape[0]
        if init == "zeros":
            state = np.zeros((batch_size, output_dim), dtype=np.float64)
        elif init == "forward":
            activation, _ = get_activation(layer.activation_name)
            state = activation(previous_state @ layer.weight.T + layer.bias)
        else:
            raise ValueError(f"Unsupported initialization mode '{init}'.")

        if clamped_mask[layer_index]:
            if mode == "train" and layer_index == len(layers):
                state = y_array.copy()
            else:
                raise ValueError(f"Unexpected clamped state at layer {layer_index}.")

        ensure_finite_array(state, f"x^{layer_index}")
        states.append(state.astype(np.float64, copy=False))
        previous_state = states[layer_index]

    return states


def compute_state_gradients(
    states: list[np.ndarray],
    cache: PCCache,
    layers: list[PCLayerParams],
    clamped_mask: list[bool],
) -> list[np.ndarray | None]:
    """Compute dF/dx^l for Phase 0 states x^0..x^L using batch-first arrays."""
    if len(states) != len(layers) + 1:
        raise ValueError("states and layers have inconsistent lengths.")
    if len(clamped_mask) != len(states):
        raise ValueError("clamped_mask must align with states.")
    if not clamped_mask[0]:
        raise ValueError("Phase 0 requires x^0 to remain clamped.")

    batch_size = states[0].shape[0]
    gradients: list[np.ndarray | None] = [None] * len(states)

    for layer_index in range(1, len(states)):
        if clamped_mask[layer_index]:
            continue

        current_error = cache.errors[layer_index]
        if current_error is None:
            raise ValueError(f"cache.errors[{layer_index}] is missing.")
        gradient = current_error / (batch_size * layers[layer_index - 1].sigma2)

        if layer_index < len(layers):
            next_error = cache.errors[layer_index + 1]
            next_pre_activation = cache.pre_activations[layer_index + 1]
            if next_error is None or next_pre_activation is None:
                raise ValueError(f"Cache is missing layer {layer_index + 1} values.")
            _, next_activation_prime = get_activation(layers[layer_index].activation_name)
            top_down = (
                next_error * next_activation_prime(next_pre_activation)
            ) @ layers[layer_index].weight
            gradient = gradient - top_down / (batch_size * layers[layer_index].sigma2)

        ensure_finite_array(gradient, f"dF/dx^{layer_index}")
        gradients[layer_index] = gradient

    return gradients


def run_inference(
    states: list[np.ndarray],
    layers: list[PCLayerParams],
    clamped_mask: list[bool],
    eta_x: float,
    steps: int,
    record_trace: bool = True,
) -> InferenceResult:
    """Run synchronous Phase 0 inference on states x^0..x^L for a fixed parameter set."""
    if eta_x <= 0.0:
        raise ValueError("eta_x must be positive.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    current_states = [state.copy() for state in states]
    batch_size = current_states[0].shape[0]
    cache = compute_cache(current_states, layers)
    energy_trace: list[float] = []

    if record_trace:
        energy_trace.append(total_energy(cache, layers, batch_size))

    for _ in range(steps):
        gradients = compute_state_gradients(current_states, cache, layers, clamped_mask)
        next_states = [state.copy() for state in current_states]
        for layer_index in range(1, len(current_states)):
            if clamped_mask[layer_index]:
                continue
            gradient = gradients[layer_index]
            if gradient is None:
                raise ValueError(f"Missing gradient for free layer {layer_index}.")
            next_states[layer_index] = current_states[layer_index] - eta_x * gradient
            ensure_finite_array(next_states[layer_index], f"x^{layer_index}")
        current_states = next_states
        cache = compute_cache(current_states, layers)
        if record_trace:
            energy_trace.append(total_energy(cache, layers, batch_size))

    final_energy = total_energy(cache, layers, batch_size)
    return InferenceResult(
        states=current_states,
        cache=cache,
        energy_trace=energy_trace,
        final_energy=final_energy,
    )
