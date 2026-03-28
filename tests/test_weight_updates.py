from __future__ import annotations

import numpy as np

from pc.layers import init_mlp_layers
from pc.models import PCNetwork


def test_parameter_update_changes_values_and_preserves_shapes() -> None:
    x = np.linspace(-1.0, 1.0, 8, dtype=np.float64).reshape(-1, 1)
    y = (0.5 * x) - 0.1
    layers = init_mlp_layers([1, 3, 1], seed=5)
    model = PCNetwork(layers=layers, eta_x=0.2, eta_w=0.05, eta_b=0.05, train_steps=15)

    before_weights = [layer.weight.copy() for layer in model.layers]
    before_biases = [layer.bias.copy() for layer in model.layers]

    result = model.train_batch(x, y, compute_post_update_energy=True)

    assert result.pre_update_energy > 0.0
    assert result.post_update_energy is not None
    changed = False
    for layer, before_weight, before_bias in zip(model.layers, before_weights, before_biases, strict=True):
        assert layer.weight.shape == before_weight.shape
        assert layer.bias.shape == before_bias.shape
        assert np.all(np.isfinite(layer.weight))
        assert np.all(np.isfinite(layer.bias))
        if not np.allclose(layer.weight, before_weight) or not np.allclose(layer.bias, before_bias):
            changed = True
    assert changed
