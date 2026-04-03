from __future__ import annotations

import numpy as np


def _validate_states_and_mask(
    states: list[np.ndarray],
    clamped_mask: list[bool],
) -> tuple[int, np.dtype[np.float64]]:
    """Validate batch-first state lists and aligned clamped-mask metadata."""
    if len(states) == 0:
        raise ValueError("states must contain at least x^0.")
    if len(states) != len(clamped_mask):
        raise ValueError("clamped_mask must align with states.")

    batch_size = int(np.asarray(states[0], dtype=np.float64).shape[0])
    for layer_index, state in enumerate(states):
        state_array = np.asarray(state, dtype=np.float64)
        if state_array.ndim != 2:
            raise ValueError(f"states[{layer_index}] must be shaped (batch, features).")
        if state_array.shape[0] != batch_size:
            raise ValueError("All states must share the same batch dimension.")

    return batch_size, np.float64


def _free_hidden_state_indices(
    states: list[np.ndarray],
    clamped_mask: list[bool],
) -> list[int]:
    """Return layer indices for free hidden states, excluding x^0 and x^L."""
    _validate_states_and_mask(states, clamped_mask)
    if len(states) <= 2:
        return []
    return [
        layer_index
        for layer_index in range(1, len(states) - 1)
        if not clamped_mask[layer_index]
    ]


def flatten_hidden_states(
    states: list[np.ndarray],
    clamped_mask: list[bool],
) -> np.ndarray:
    """Flatten free hidden states into a batch-first latent array.

    Shape contract:
    - input `states[i]`: `(batch, features_i)` for `i = 0..L`
    - output `z`: `(batch, total_hidden_dim)`

    Flattening rules:
    - never flatten `x^0`
    - never flatten the final output state `x^L`
    - only flatten hidden layers `x^1 .. x^(L-1)` whose `clamped_mask[i]` is `False`
    - concatenate hidden features in ascending layer order
    - output dtype is always `float64`
    """
    batch_size, _ = _validate_states_and_mask(states, clamped_mask)
    free_indices = _free_hidden_state_indices(states, clamped_mask)
    if len(free_indices) == 0:
        return np.zeros((batch_size, 0), dtype=np.float64)

    chunks = [np.asarray(states[layer_index], dtype=np.float64) for layer_index in free_indices]
    return np.concatenate(chunks, axis=1).astype(np.float64, copy=False)


def unflatten_hidden_states(
    z: np.ndarray,
    states_template: list[np.ndarray],
    clamped_mask: list[bool],
) -> list[np.ndarray]:
    """Reconstruct a full state list from a flattened hidden-state representation.

    Shape contract:
    - input `z`: `(batch, total_hidden_dim)`
    - input `states_template[i]`: `(batch, features_i)` for `i = 0..L`
    - output `states[i]`: `(batch, features_i)` for `i = 0..L`

    Reconstruction rules:
    - `x^0` is copied from `states_template`
    - the final output state `x^L` is copied from `states_template`
    - clamped hidden states are copied from `states_template`
    - only free hidden states `x^1 .. x^(L-1)` are read from `z`
    - hidden features are consumed in the same ascending-layer concatenation order used by
      `flatten_hidden_states(...)`
    - reconstructed arrays are `float64`
    """
    batch_size, _ = _validate_states_and_mask(states_template, clamped_mask)
    z_array = np.asarray(z, dtype=np.float64)
    if z_array.ndim != 2:
        raise ValueError("z must be shaped (batch, total_hidden_dim).")
    if z_array.shape[0] != batch_size:
        raise ValueError("z batch dimension must match states_template.")

    free_indices = _free_hidden_state_indices(states_template, clamped_mask)
    expected_hidden_dim = sum(
        int(np.asarray(states_template[layer_index], dtype=np.float64).shape[1])
        for layer_index in free_indices
    )
    if z_array.shape[1] != expected_hidden_dim:
        raise ValueError(
            f"z feature dimension must be {expected_hidden_dim}, received {z_array.shape[1]}."
        )

    reconstructed = [np.asarray(state, dtype=np.float64).copy() for state in states_template]
    offset = 0
    for layer_index in free_indices:
        state_width = int(np.asarray(states_template[layer_index], dtype=np.float64).shape[1])
        reconstructed[layer_index] = z_array[:, offset : offset + state_width].copy()
        offset += state_width

    return reconstructed
