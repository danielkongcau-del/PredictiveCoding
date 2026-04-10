from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from ..energy import compute_cache, total_energy
from ..inference import build_clamped_mask, compute_state_gradients, initialize_states
from ..layers import PCLayerParams
from ..models import PCNetwork
from ..state_io import flatten_hidden_states, unflatten_hidden_states
from ..utils import ensure_finite_array

TF1TransportMode = Literal["identity", "local_field_only", "learned"]


def _as_batch_first(name: str, array: np.ndarray) -> np.ndarray:
    array_float = np.asarray(array, dtype=np.float64)
    if array_float.ndim != 2:
        raise ValueError(f"{name} must be shaped (batch, features).")
    return array_float


def _hidden_indices(clamped_mask: list[bool]) -> list[int]:
    if len(clamped_mask) <= 2:
        return []
    return [
        layer_index
        for layer_index in range(1, len(clamped_mask) - 1)
        if not clamped_mask[layer_index]
    ]


def _per_sample_energy(
    states: list[np.ndarray],
    layers: list[PCLayerParams],
) -> np.ndarray:
    cache = compute_cache(states, layers)
    batch_size = int(states[0].shape[0])
    energy = np.zeros((batch_size, 1), dtype=np.float64)
    for layer_index, layer in enumerate(layers, start=1):
        error = cache.errors[layer_index]
        if error is None:
            raise ValueError(f"cache.errors[{layer_index}] is missing.")
        energy += np.sum(error * error, axis=1, keepdims=True) / (2.0 * layer.sigma2)
    ensure_finite_array(energy, "per_sample_energy")
    return energy


def validate_tf1_time_pair(t: float, r: float) -> None:
    """Validate the TF1 `(t, r)` contract.

    Shape contract:
    - scalar `t`
    - scalar `r`
    """

    if not (0.0 <= float(t) < 1.0):
        raise ValueError("t must satisfy 0 <= t < 1.")
    if not (0.0 < float(r) <= 1.0 - float(t) + 1e-12):
        raise ValueError("r must satisfy 0 < r <= 1 - t.")


def uniform_rollout_knots(transport_steps: int) -> np.ndarray:
    """Return uniform batch-shared rollout knots on `[0, 1]`."""
    if transport_steps <= 0:
        raise ValueError("transport_steps must be positive.")
    knots = np.linspace(0.0, 1.0, int(transport_steps) + 1, dtype=np.float64)
    return knots


@dataclass(frozen=True)
class FMPCTF1Context:
    """Teacher-free TF1 training context for one batch.

    Shape contract:
    - `inputs`: `(batch, input_dim)`
    - `targets`: `(batch, target_dim)`
    - `z0`: `(batch, hidden_dim)`
    """

    inputs: np.ndarray
    targets: np.ndarray
    layers: list[PCLayerParams]
    states_template: list[np.ndarray]
    clamped_mask: list[bool]
    z0: np.ndarray

    def __post_init__(self) -> None:
        inputs = _as_batch_first("inputs", self.inputs)
        targets = _as_batch_first("targets", self.targets)
        z0 = _as_batch_first("z0", self.z0)
        if inputs.shape[0] != targets.shape[0] or inputs.shape[0] != z0.shape[0]:
            raise ValueError("inputs, targets, and z0 must share the same batch size.")
        if len(self.states_template) != len(self.layers) + 1:
            raise ValueError("states_template must align with the layer list.")
        if len(self.clamped_mask) != len(self.states_template):
            raise ValueError("clamped_mask must align with states_template.")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "z0", z0)

    @property
    def batch_size(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.z0.shape[1])

    @property
    def target_dim(self) -> int:
        return int(self.targets.shape[1])


@dataclass(frozen=True)
class FMPCTF1StateFeatures:
    """Current-state teacher-free TF1 feature block.

    Shape contract:
    - `g_t`: `(batch, hidden_dim)`
    - `e_out_t`: `(batch, target_dim)`
    - `F_t`: `(batch, 1)`
    - `y_hat_t`: `(batch, target_dim)`
    """

    g_t: np.ndarray
    e_out_t: np.ndarray
    F_t: np.ndarray
    y_hat_t: np.ndarray

    def __post_init__(self) -> None:
        g_t = _as_batch_first("g_t", self.g_t)
        e_out_t = _as_batch_first("e_out_t", self.e_out_t)
        F_t = _as_batch_first("F_t", self.F_t)
        y_hat_t = _as_batch_first("y_hat_t", self.y_hat_t)
        batch_size = int(g_t.shape[0])
        if e_out_t.shape[0] != batch_size or F_t.shape[0] != batch_size or y_hat_t.shape[0] != batch_size:
            raise ValueError("All TF1 feature blocks must share the same batch size.")
        if F_t.shape[1] != 1:
            raise ValueError("F_t must be shaped (batch, 1).")
        object.__setattr__(self, "g_t", g_t)
        object.__setattr__(self, "e_out_t", e_out_t)
        object.__setattr__(self, "F_t", F_t)
        object.__setattr__(self, "y_hat_t", y_hat_t)


@dataclass(frozen=True)
class FMPCTF1StateFeatureTangents:
    """Directional derivatives of TF1 current-state features along `g_t`.

    Shape contract:
    - `Dg_g_t`: `(batch, hidden_dim)`
    - `Dg_e_out_t`: `(batch, target_dim)`
    - `Dg_F_t`: `(batch, 1)`
    - `Dg_y_hat_t`: `(batch, target_dim)`
    """

    Dg_g_t: np.ndarray
    Dg_e_out_t: np.ndarray
    Dg_F_t: np.ndarray
    Dg_y_hat_t: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "Dg_g_t", _as_batch_first("Dg_g_t", self.Dg_g_t))
        object.__setattr__(self, "Dg_e_out_t", _as_batch_first("Dg_e_out_t", self.Dg_e_out_t))
        object.__setattr__(self, "Dg_F_t", _as_batch_first("Dg_F_t", self.Dg_F_t))
        object.__setattr__(self, "Dg_y_hat_t", _as_batch_first("Dg_y_hat_t", self.Dg_y_hat_t))
        if self.Dg_F_t.shape[1] != 1:
            raise ValueError("Dg_F_t must be shaped (batch, 1).")


@dataclass(frozen=True)
class FMPCTF1TransportRolloutResult:
    """Coarse hidden-state transport rollout result.

    Shape contract:
    - `z_knots[k]`: `(batch, hidden_dim)` for `k = 0..transport_steps`
    - `knot_times`: `(transport_steps + 1,)`
    """

    z_knots: list[np.ndarray]
    knot_times: np.ndarray
    final_energy: float

    def __post_init__(self) -> None:
        if len(self.z_knots) != int(self.knot_times.shape[0]):
            raise ValueError("z_knots must align with knot_times.")
        for knot_index, z_knot in enumerate(self.z_knots):
            self.z_knots[knot_index] = _as_batch_first(f"z_knots[{knot_index}]", z_knot)


def build_tf1_context(
    model: PCNetwork,
    inputs: np.ndarray,
    targets: np.ndarray,
) -> FMPCTF1Context:
    """Build one TF1 training context from batch-first supervised arrays."""

    inputs_array = _as_batch_first("inputs", inputs)
    targets_array = _as_batch_first("targets", targets)
    states_template = initialize_states(
        model.layers,
        inputs_array,
        y=targets_array,
        init=model.state_init,
        mode="train",
    )
    clamped_mask = build_clamped_mask(len(model.layers) + 1, mode="train")
    z0 = flatten_hidden_states(states_template, clamped_mask)
    return FMPCTF1Context(
        inputs=inputs_array,
        targets=targets_array,
        layers=model.layers,
        states_template=states_template,
        clamped_mask=clamped_mask,
        z0=z0,
    )


def hidden_states_from_state(
    context: FMPCTF1Context,
    z: np.ndarray,
) -> list[np.ndarray]:
    """Reconstruct full states from a batch-first hidden latent `z`."""

    return unflatten_hidden_states(_as_batch_first("z", z), context.states_template, context.clamped_mask)


def hidden_energy_from_state(
    context: FMPCTF1Context,
    z: np.ndarray,
) -> float:
    """Return the batch-mean baseline PC energy at hidden state `z`."""

    states = hidden_states_from_state(context, z)
    cache = compute_cache(states, context.layers)
    return total_energy(cache, context.layers, context.batch_size)


def hidden_local_flow(
    context: FMPCTF1Context,
    z: np.ndarray,
) -> np.ndarray:
    """Return the exact teacher-free hidden-state flow `g_theta(z; c)`.

    Shape contract:
    - `z`: `(batch, hidden_dim)`
    - returns `(batch, hidden_dim)`
    """

    z_array = _as_batch_first("z", z)
    states = hidden_states_from_state(context, z_array)
    cache = compute_cache(states, context.layers)
    gradients = compute_state_gradients(states, cache, context.layers, context.clamped_mask)
    chunks: list[np.ndarray] = []
    for layer_index in _hidden_indices(context.clamped_mask):
        gradient = gradients[layer_index]
        if gradient is None:
            raise ValueError(f"Missing hidden gradient for layer {layer_index}.")
        chunks.append((-np.asarray(gradient, dtype=np.float64)).copy())
    if len(chunks) == 0:
        return np.zeros_like(z_array)
    flow = np.concatenate(chunks, axis=1).astype(np.float64, copy=False)
    ensure_finite_array(flow, "g_theta")
    return flow


def teacher_free_state_features(
    context: FMPCTF1Context,
    z: np.ndarray,
) -> FMPCTF1StateFeatures:
    """Return the current teacher-free feature block at hidden state `z`."""

    z_array = _as_batch_first("z", z)
    states = hidden_states_from_state(context, z_array)
    cache = compute_cache(states, context.layers)
    y_hat_t = np.asarray(cache.predictions[-1], dtype=np.float64)
    e_out_t = np.asarray(cache.errors[-1], dtype=np.float64)
    if y_hat_t.ndim != 2 or e_out_t.ndim != 2:
        raise ValueError("Output prediction and error blocks must be batch-first arrays.")
    features = FMPCTF1StateFeatures(
        g_t=hidden_local_flow(context, z_array),
        e_out_t=e_out_t,
        F_t=_per_sample_energy(states, context.layers),
        y_hat_t=y_hat_t,
    )
    return features


def teacher_free_feature_tangents(
    context: FMPCTF1Context,
    z: np.ndarray,
    *,
    epsilon: float,
) -> FMPCTF1StateFeatureTangents:
    """Return directional derivatives of the TF1 feature block along `g_t`."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    z_array = _as_batch_first("z", z)
    features = teacher_free_state_features(context, z_array)
    if z_array.shape[1] == 0:
        zeros_hidden = np.zeros_like(z_array)
        zeros_target = np.zeros_like(features.e_out_t)
        zeros_scalar = np.zeros_like(features.F_t)
        return FMPCTF1StateFeatureTangents(
            Dg_g_t=zeros_hidden,
            Dg_e_out_t=zeros_target,
            Dg_F_t=zeros_scalar,
            Dg_y_hat_t=np.zeros_like(features.y_hat_t),
        )
    z_plus = z_array + float(epsilon) * features.g_t
    z_minus = z_array - float(epsilon) * features.g_t
    plus_features = teacher_free_state_features(context, z_plus)
    minus_features = teacher_free_state_features(context, z_minus)
    scale = 2.0 * float(epsilon)
    return FMPCTF1StateFeatureTangents(
        Dg_g_t=(plus_features.g_t - minus_features.g_t) / scale,
        Dg_e_out_t=(plus_features.e_out_t - minus_features.e_out_t) / scale,
        Dg_F_t=(plus_features.F_t - minus_features.F_t) / scale,
        Dg_y_hat_t=(plus_features.y_hat_t - minus_features.y_hat_t) / scale,
    )


def bootstrap_average_velocity_target(
    context: FMPCTF1Context,
    z: np.ndarray,
    *,
    t: float,
    r: float,
    integrator: Literal["euler", "rk2"] = "rk2",
    substeps: int = 4,
) -> np.ndarray:
    """Return a self-bootstrap average-velocity target over the remaining horizon.

    Shape contract:
    - `z`: `(batch, hidden_dim)`
    - returns `(batch, hidden_dim)`
    """

    validate_tf1_time_pair(t, r)
    if integrator not in {"euler", "rk2"}:
        raise ValueError(f"Unsupported bootstrap integrator '{integrator}'.")
    if substeps <= 0:
        raise ValueError("substeps must be positive.")
    z_array = _as_batch_first("z", z)
    step_size = float(r) / float(substeps)
    current = z_array.copy()
    for _ in range(int(substeps)):
        if integrator == "euler":
            current = current + step_size * hidden_local_flow(context, current)
        else:
            k1 = hidden_local_flow(context, current)
            mid = current + 0.5 * step_size * k1
            k2 = hidden_local_flow(context, mid)
            current = current + step_size * k2
        ensure_finite_array(current, "bootstrap_z")
    return (current - z_array) / float(r)


def rollout_hidden_transport(
    context: FMPCTF1Context,
    z0: np.ndarray,
    *,
    transport_steps: int,
    mode: TF1TransportMode,
    velocity_fn: Callable[[np.ndarray, float, float], np.ndarray] | None = None,
) -> FMPCTF1TransportRolloutResult:
    """Roll out a coarse hidden-state transport path on shared uniform knots."""

    z_current = _as_batch_first("z0", z0).copy()
    knots = uniform_rollout_knots(transport_steps)
    z_knots = [z_current.copy()]
    for knot_index in range(int(transport_steps)):
        t_k = float(knots[knot_index])
        t_next = float(knots[knot_index + 1])
        dt = t_next - t_k
        r_k = 1.0 - t_k
        if mode == "identity":
            velocity = np.zeros_like(z_current)
        elif mode == "local_field_only":
            velocity = hidden_local_flow(context, z_current)
        elif mode == "learned":
            if velocity_fn is None:
                raise ValueError("velocity_fn must be provided in learned transport mode.")
            velocity = _as_batch_first("velocity", velocity_fn(z_current, t_k, r_k))
        else:
            raise ValueError(f"Unsupported transport mode '{mode}'.")
        if velocity.shape != z_current.shape:
            raise ValueError("Transport velocity must match the hidden-state shape.")
        z_current = z_current + float(dt) * velocity
        ensure_finite_array(z_current, "transported_hidden_state")
        z_knots.append(z_current.copy())
    return FMPCTF1TransportRolloutResult(
        z_knots=z_knots,
        knot_times=knots,
        final_energy=hidden_energy_from_state(context, z_current),
    )
