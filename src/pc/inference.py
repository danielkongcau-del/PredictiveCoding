from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .activations import get_activation
from .energy import PCCache, compute_cache, total_energy
from .layers import PCLayerParams
from .state_io import flatten_hidden_states
from .utils import ensure_finite_array

InferenceMethod = Literal["euler", "rk2"]
InferenceBackendName = Literal["pc_euler", "pc_rk2", "fmpc"]


@dataclass
class InferenceResult:
    """Final inference states, cache, optional state trajectory, and energy trace for x^0..x^L."""

    states: list[np.ndarray]
    cache: PCCache
    energy_trace: list[float]
    final_energy: float
    inference_backend: InferenceBackendName
    state_trajectory: list[list[np.ndarray]] | None = None


@dataclass
class TeacherInferenceExport:
    """Teacher-export payload derived from the current slow predictive-coding inference path.

    Shape contract:
    - `initial_states[i]`: `(batch, features_i)`
    - `final_states[i]`: `(batch, features_i)`
    - `z0`: `(batch, total_hidden_dim)`
    - `z_star`: `(batch, total_hidden_dim)`
    - `z_trajectory[t]`: `(batch, total_hidden_dim)` for `t = 0..steps` when present

    Notes:
    - `z0` is the flattened free hidden state before any inference update.
    - `z_star` is the flattened free hidden state after the configured inference steps.
    - `z_trajectory`, when requested, includes both endpoints:
      `z_trajectory[0] == z0` and `z_trajectory[-1] == z_star`.
    """

    mode: Literal["train", "predict"]
    clamped_mask: list[bool]
    initial_states: list[np.ndarray]
    final_states: list[np.ndarray]
    z0: np.ndarray
    z_star: np.ndarray
    z_trajectory: list[np.ndarray] | None
    energy_trace: list[float]
    steps: int
    inference_backend: InferenceBackendName
    inference_method: InferenceMethod | None


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


def _validate_inference_method(method: str) -> InferenceMethod:
    if method not in {"euler", "rk2"}:
        raise ValueError(f"Unsupported inference method '{method}'.")
    return method


def _legacy_method_from_backend(backend: InferenceBackendName) -> InferenceMethod:
    if backend == "pc_euler":
        return "euler"
    if backend == "pc_rk2":
        return "rk2"
    raise NotImplementedError("The reserved 'fmpc' backend does not define a legacy PC integrator label yet.")


def resolve_inference_backend_name(
    backend: str | None = None,
    *,
    method: str | None = None,
) -> InferenceBackendName:
    """Resolve the explicit inference backend name.

    Accepted backend labels:
    - `pc_euler`
    - `pc_rk2`
    - `fmpc` (reserved placeholder; not implemented here)

    Backward-compatible aliases:
    - `euler` -> `pc_euler`
    - `rk2` -> `pc_rk2`
    """
    if backend is not None and method is not None:
        normalized_backend = resolve_inference_backend_name(backend, method=None)
        normalized_method_backend = resolve_inference_backend_name(method, method=None)
        if normalized_backend != normalized_method_backend:
            raise ValueError(
                "inference_backend and inference_method refer to different backends."
            )
        return normalized_backend

    label = "pc_euler" if backend is None and method is None else (backend if backend is not None else method)
    if label == "euler":
        return "pc_euler"
    if label == "rk2":
        return "pc_rk2"
    if label in {"pc_euler", "pc_rk2", "fmpc"}:
        return label  # type: ignore[return-value]
    raise ValueError(f"Unsupported inference backend '{label}'.")


def _copy_states(states: list[np.ndarray]) -> list[np.ndarray]:
    return [np.asarray(state, dtype=np.float64).copy() for state in states]


def _apply_euler_step(
    current_states: list[np.ndarray],
    gradients: list[np.ndarray | None],
    clamped_mask: list[bool],
    eta_x: float,
) -> list[np.ndarray]:
    next_states = [state.copy() for state in current_states]
    for layer_index in range(1, len(current_states)):
        if clamped_mask[layer_index]:
            continue
        gradient = gradients[layer_index]
        if gradient is None:
            raise ValueError(f"Missing gradient for free layer {layer_index}.")
        next_states[layer_index] = current_states[layer_index] - eta_x * gradient
        ensure_finite_array(next_states[layer_index], f"x^{layer_index}")
    return next_states


def _apply_rk2_step(
    current_states: list[np.ndarray],
    gradients_0: list[np.ndarray | None],
    layers: list[PCLayerParams],
    clamped_mask: list[bool],
    eta_x: float,
) -> list[np.ndarray]:
    proposal_states = _apply_euler_step(current_states, gradients_0, clamped_mask, eta_x)
    proposal_cache = compute_cache(proposal_states, layers)
    gradients_1 = compute_state_gradients(proposal_states, proposal_cache, layers, clamped_mask)

    next_states = [state.copy() for state in current_states]
    for layer_index in range(1, len(current_states)):
        if clamped_mask[layer_index]:
            continue
        gradient_0 = gradients_0[layer_index]
        gradient_1 = gradients_1[layer_index]
        if gradient_0 is None or gradient_1 is None:
            raise ValueError(f"Missing RK2 gradients for free layer {layer_index}.")
        next_states[layer_index] = current_states[layer_index] - 0.5 * eta_x * (gradient_0 + gradient_1)
        ensure_finite_array(next_states[layer_index], f"x^{layer_index}")
    return next_states


def run_inference(
    states: list[np.ndarray],
    layers: list[PCLayerParams],
    clamped_mask: list[bool],
    eta_x: float,
    steps: int,
    backend: str | None = None,
    method: InferenceMethod | None = None,
    record_trace: bool = True,
    record_state_trajectory: bool = False,
) -> InferenceResult:
    """Run synchronous Phase 0 inference on states x^0..x^L for a fixed parameter set."""
    if eta_x <= 0.0:
        raise ValueError("eta_x must be positive.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    resolved_backend = resolve_inference_backend_name(backend, method=method)

    current_states = [state.copy() for state in states]
    batch_size = current_states[0].shape[0]
    cache = compute_cache(current_states, layers)
    energy_trace: list[float] = []
    state_trajectory: list[list[np.ndarray]] | None = None

    if record_trace:
        energy_trace.append(total_energy(cache, layers, batch_size))
    if record_state_trajectory:
        state_trajectory = [_copy_states(current_states)]

    for _ in range(steps):
        gradients = compute_state_gradients(current_states, cache, layers, clamped_mask)
        if resolved_backend == "pc_euler":
            next_states = _apply_euler_step(current_states, gradients, clamped_mask, eta_x)
        elif resolved_backend == "pc_rk2":
            next_states = _apply_rk2_step(current_states, gradients, layers, clamped_mask, eta_x)
        else:
            raise NotImplementedError(
                "The reserved 'fmpc' inference backend is not implemented yet."
            )
        current_states = next_states
        cache = compute_cache(current_states, layers)
        if record_trace:
            energy_trace.append(total_energy(cache, layers, batch_size))
        if state_trajectory is not None:
            state_trajectory.append(_copy_states(current_states))

    final_energy = total_energy(cache, layers, batch_size)
    return InferenceResult(
        states=current_states,
        cache=cache,
        energy_trace=energy_trace,
        final_energy=final_energy,
        inference_backend=resolved_backend,
        state_trajectory=state_trajectory,
    )


def build_teacher_inference_export(
    initial_states: list[np.ndarray],
    inference_result: InferenceResult,
    clamped_mask: list[bool],
    *,
    mode: Literal["train", "predict"],
    steps: int,
    inference_backend: InferenceBackendName,
    inference_method: InferenceMethod | None,
) -> TeacherInferenceExport:
    """Materialize teacher targets from an already-run inference pass.

    This helper is intended for callers that already executed the slow iterative teacher and
    want a reusable export payload without rerunning inference.
    """
    initial_state_copies = _copy_states(initial_states)
    final_state_copies = _copy_states(inference_result.states)
    z0 = flatten_hidden_states(initial_state_copies, clamped_mask)
    z_star = flatten_hidden_states(final_state_copies, clamped_mask)
    z_trajectory = None
    if inference_result.state_trajectory is not None:
        z_trajectory = [
            flatten_hidden_states(state_snapshot, clamped_mask)
            for state_snapshot in inference_result.state_trajectory
        ]

    return TeacherInferenceExport(
        mode=mode,
        clamped_mask=list(clamped_mask),
        initial_states=initial_state_copies,
        final_states=final_state_copies,
        z0=z0,
        z_star=z_star,
        z_trajectory=z_trajectory,
        energy_trace=list(inference_result.energy_trace),
        steps=steps,
        inference_backend=inference_backend,
        inference_method=inference_method,
    )


def run_teacher_inference_export(
    layers: list[PCLayerParams],
    x: np.ndarray,
    *,
    y: np.ndarray | None = None,
    init: str = "forward",
    mode: Literal["train", "predict"] = "train",
    eta_x: float,
    steps: int,
    backend: str | None = None,
    method: InferenceMethod | None = None,
    record_trace: bool = True,
    record_trajectory: bool = False,
) -> TeacherInferenceExport:
    """Run the current slow teacher inference and export hidden-state supervision targets.

    Shape contract:
    - `x`: `(batch, input_dim)`
    - `y`: `(batch, output_dim)` when `mode="train"`
    - returned `z0` and `z_star`: `(batch, total_hidden_dim)`

    Teacher semantics:
    - the teacher is exactly the current iterative predictive-coding inference path
    - no student/transporter approximation is used here
    """
    x_array = np.asarray(x, dtype=np.float64)
    y_array = None if y is None else np.asarray(y, dtype=np.float64)
    resolved_backend = resolve_inference_backend_name(backend, method=method)
    initial_states = initialize_states(
        layers,
        x_array,
        y=y_array,
        init=init,
        mode=mode,
    )
    clamped_mask = build_clamped_mask(len(layers) + 1, mode=mode)
    inference_result = run_inference(
        initial_states,
        layers,
        clamped_mask,
        eta_x=eta_x,
        steps=steps,
        backend=resolved_backend,
        record_trace=record_trace,
        record_state_trajectory=record_trajectory,
    )
    return build_teacher_inference_export(
        initial_states,
        inference_result,
        clamped_mask,
        mode=mode,
        steps=steps,
        inference_backend=resolved_backend,
        inference_method=_legacy_method_from_backend(resolved_backend),
    )
