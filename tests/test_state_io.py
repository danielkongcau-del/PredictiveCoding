from __future__ import annotations

import numpy as np
import pytest

from pc.inference import build_clamped_mask
from pc.state_io import flatten_hidden_states, unflatten_hidden_states


def test_flatten_unflatten_round_trip_recovers_free_hidden_states() -> None:
    states = [
        np.array([[10.0, 11.0], [12.0, 13.0]], dtype=np.float64),  # x^0
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),  # x^1
        np.array([[7.0, 8.0], [9.0, 10.0]], dtype=np.float64),  # x^2
        np.array([[20.0], [21.0]], dtype=np.float64),  # x^3
    ]
    clamped_mask = [True, False, False, True]

    z = flatten_hidden_states(states, clamped_mask)
    reconstructed = unflatten_hidden_states(z, states, clamped_mask)

    assert z.shape == (2, 5)
    np.testing.assert_allclose(reconstructed[1], states[1])
    np.testing.assert_allclose(reconstructed[2], states[2])
    np.testing.assert_allclose(reconstructed[0], states[0])
    np.testing.assert_allclose(reconstructed[3], states[3])


def test_flatten_hidden_states_uses_ascending_contiguous_hidden_concatenation_order() -> None:
    states = [
        np.zeros((2, 2), dtype=np.float64),
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        np.array([[5.0], [6.0]], dtype=np.float64),
        np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float64),
        np.zeros((2, 1), dtype=np.float64),
    ]
    clamped_mask = [True, False, True, False, False]

    z = flatten_hidden_states(states, clamped_mask)

    expected = np.array(
        [
            [1.0, 2.0, 7.0, 8.0, 9.0],
            [3.0, 4.0, 10.0, 11.0, 12.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(z, expected)


def test_flatten_hidden_states_excludes_output_state_even_when_predict_output_is_free() -> None:
    states = [
        np.zeros((3, 2), dtype=np.float64),
        np.ones((3, 4), dtype=np.float64),
        np.full((3, 5), 2.0, dtype=np.float64),
    ]
    clamped_mask = build_clamped_mask(num_states=3, mode="predict")

    z = flatten_hidden_states(states, clamped_mask)
    reconstructed = unflatten_hidden_states(z, states, clamped_mask)

    assert z.shape == (3, 4)
    np.testing.assert_allclose(reconstructed[1], states[1])
    np.testing.assert_allclose(reconstructed[2], states[2])


def test_flatten_hidden_states_returns_empty_latent_when_no_free_hidden_states() -> None:
    states = [
        np.zeros((4, 3), dtype=np.float64),
        np.ones((4, 2), dtype=np.float64),
    ]
    clamped_mask = [True, False]

    z = flatten_hidden_states(states, clamped_mask)
    reconstructed = unflatten_hidden_states(z, states, clamped_mask)

    assert z.shape == (4, 0)
    np.testing.assert_allclose(reconstructed[0], states[0])
    np.testing.assert_allclose(reconstructed[1], states[1])


def test_unflatten_hidden_states_raises_on_feature_mismatch() -> None:
    states = [
        np.zeros((2, 2), dtype=np.float64),
        np.ones((2, 3), dtype=np.float64),
        np.zeros((2, 1), dtype=np.float64),
    ]
    clamped_mask = [True, False, True]
    bad_z = np.zeros((2, 4), dtype=np.float64)

    with pytest.raises(ValueError, match="z feature dimension must be 3"):
        unflatten_hidden_states(bad_z, states, clamped_mask)


def test_unflatten_hidden_states_raises_on_batch_mismatch() -> None:
    states = [
        np.zeros((2, 2), dtype=np.float64),
        np.ones((2, 3), dtype=np.float64),
        np.zeros((2, 1), dtype=np.float64),
    ]
    clamped_mask = [True, False, True]
    bad_z = np.zeros((3, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="z batch dimension must match states_template"):
        unflatten_hidden_states(bad_z, states, clamped_mask)
