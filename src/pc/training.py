from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .activations import get_activation
from .energy import PCCache
from .inference import build_clamped_mask, initialize_states, run_inference
from .layers import PCLayerParams
from .utils import ensure_finite_array, set_seed


@dataclass
class TrainBatchResult:
    """Training outputs for one batch with batch-first x and y arrays."""

    train_steps: int
    energy_trace: list[float]
    pre_update_energy: float
    post_update_energy: float | None
    parameter_norms: dict[str, list[float]]


def parameter_gradients(
    states: list[np.ndarray],
    cache: PCCache,
    layers: list[PCLayerParams],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Compute dF/dW^l and dF/db^l after inference for states x^0..x^L."""
    batch_size = states[0].shape[0]
    weight_gradients: list[np.ndarray] = []
    bias_gradients: list[np.ndarray] = []

    for layer_index, layer in enumerate(layers, start=1):
        pre_activation = cache.pre_activations[layer_index]
        error = cache.errors[layer_index]
        if pre_activation is None or error is None:
            raise ValueError(f"Cache is missing layer {layer_index} values.")
        _, activation_prime = get_activation(layer.activation_name)
        local_term = error * activation_prime(pre_activation)
        grad_w = -(local_term.T @ states[layer_index - 1]) / (batch_size * layer.sigma2)
        grad_b = -np.sum(local_term, axis=0) / (batch_size * layer.sigma2)
        ensure_finite_array(grad_w, f"dF/dW^{layer_index}")
        ensure_finite_array(grad_b, f"dF/db^{layer_index}")
        weight_gradients.append(grad_w)
        bias_gradients.append(grad_b)

    return weight_gradients, bias_gradients


def apply_parameter_updates(
    layers: list[PCLayerParams],
    weight_gradients: list[np.ndarray],
    bias_gradients: list[np.ndarray],
    eta_w: float,
    eta_b: float,
) -> None:
    """Apply explicit gradient descent to W^l and b^l."""
    if eta_w <= 0.0 or eta_b <= 0.0:
        raise ValueError("eta_w and eta_b must be positive.")
    if len(weight_gradients) != len(layers) or len(bias_gradients) != len(layers):
        raise ValueError("Gradient lists must align with layers.")

    for layer_index, layer in enumerate(layers):
        layer.weight = layer.weight - eta_w * weight_gradients[layer_index]
        layer.bias = layer.bias - eta_b * bias_gradients[layer_index]
        ensure_finite_array(layer.weight, f"W^{layer_index + 1}")
        ensure_finite_array(layer.bias, f"b^{layer_index + 1}")


def _parameter_norms(layers: list[PCLayerParams]) -> dict[str, list[float]]:
    return {
        "weight_norms": [float(np.linalg.norm(layer.weight)) for layer in layers],
        "bias_norms": [float(np.linalg.norm(layer.bias)) for layer in layers],
    }


def train_batch(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    compute_post_update_energy: bool = False,
) -> TrainBatchResult:
    """Run one Phase 0 training batch with x shaped (B, d_0) and y shaped (B, d_L)."""
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    clamped_mask = build_clamped_mask(len(model.layers) + 1, mode="train")

    states = initialize_states(
        model.layers,
        x_array,
        y=y_array,
        init=model.state_init,
        mode="train",
    )
    inference_result = run_inference(
        states,
        model.layers,
        clamped_mask,
        eta_x=model.eta_x,
        steps=model.train_steps,
        record_trace=True,
    )
    weight_gradients, bias_gradients = parameter_gradients(
        inference_result.states,
        inference_result.cache,
        model.layers,
    )
    apply_parameter_updates(
        model.layers,
        weight_gradients,
        bias_gradients,
        eta_w=model.eta_w,
        eta_b=model.eta_b,
    )

    post_update_energy: float | None = None
    if compute_post_update_energy:
        post_states = initialize_states(
            model.layers,
            x_array,
            y=y_array,
            init=model.state_init,
            mode="train",
        )
        post_result = run_inference(
            post_states,
            model.layers,
            clamped_mask,
            eta_x=model.eta_x,
            steps=model.train_steps,
            record_trace=True,
        )
        post_update_energy = post_result.final_energy

    return TrainBatchResult(
        train_steps=model.train_steps,
        energy_trace=inference_result.energy_trace,
        pre_update_energy=inference_result.final_energy,
        post_update_energy=post_update_energy,
        parameter_norms=_parameter_norms(model.layers),
    )


def fit(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    seed: int | None = None,
) -> dict[str, list[float]]:
    """Fit a Phase 0 model on full-batch x and y arrays for a fixed number of epochs."""
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    set_seed(seed)

    history: dict[str, list[float]] = {
        "pre_update_energy": [],
        "post_update_energy": [],
    }

    for _ in range(epochs):
        batch_result = train_batch(model, x, y, compute_post_update_energy=True)
        history["pre_update_energy"].append(batch_result.pre_update_energy)
        if batch_result.post_update_energy is None:
            raise ValueError("post_update_energy must be present during fit.")
        history["post_update_energy"].append(batch_result.post_update_energy)

    return history
