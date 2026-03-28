from __future__ import annotations

import numpy as np

from pc.energy import compute_cache, total_energy
from pc.layers import PCLayerParams


def test_manual_single_layer_energy_matches_expected_value() -> None:
    layers = [
        PCLayerParams(
            weight=np.array([[1.5]], dtype=np.float64),
            bias=np.array([-0.5], dtype=np.float64),
            sigma2=1.0,
            activation_name="identity",
        )
    ]
    states = [
        np.array([[1.0], [2.0]], dtype=np.float64),
        np.array([[0.0], [4.5]], dtype=np.float64),
    ]
    cache = compute_cache(states, layers)
    energy = total_energy(cache, layers, batch_size=2)
    assert energy == 1.25


def test_zero_error_layer_contributes_zero_energy() -> None:
    layers = [
        PCLayerParams(
            weight=np.array([[2.0]], dtype=np.float64),
            bias=np.array([1.0], dtype=np.float64),
            sigma2=1.0,
            activation_name="identity",
        )
    ]
    states = [
        np.array([[0.5], [-1.0]], dtype=np.float64),
        np.array([[2.0], [-1.0]], dtype=np.float64),
    ]
    cache = compute_cache(states, layers)
    assert total_energy(cache, layers, batch_size=2) == 0.0


def test_multi_layer_energy_equals_sum_of_layer_terms() -> None:
    layers = [
        PCLayerParams(
            weight=np.array([[1.0]], dtype=np.float64),
            bias=np.array([0.0], dtype=np.float64),
            sigma2=2.0,
            activation_name="identity",
        ),
        PCLayerParams(
            weight=np.array([[2.0]], dtype=np.float64),
            bias=np.array([0.5], dtype=np.float64),
            sigma2=4.0,
            activation_name="identity",
        ),
    ]
    states = [
        np.array([[1.0], [3.0]], dtype=np.float64),
        np.array([[2.0], [1.0]], dtype=np.float64),
        np.array([[3.0], [1.0]], dtype=np.float64),
    ]
    cache = compute_cache(states, layers)
    expected_layer_1 = np.sum(np.array([[1.0], [-2.0]], dtype=np.float64) ** 2) / (2.0 * 2 * 2.0)
    expected_layer_2 = np.sum(np.array([[-1.5], [-1.5]], dtype=np.float64) ** 2) / (2.0 * 2 * 4.0)
    expected_total = expected_layer_1 + expected_layer_2
    np.testing.assert_allclose(total_energy(cache, layers, batch_size=2), expected_total)
