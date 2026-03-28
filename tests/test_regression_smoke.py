from __future__ import annotations

import numpy as np

from pc.layers import init_mlp_layers
from pc.models import PCNetwork


def test_toy_regression_training_improves_energy_and_beats_constant_baseline() -> None:
    x = np.linspace(-1.0, 1.0, 16, dtype=np.float64).reshape(-1, 1)
    y = (0.75 * x) - 0.1
    layers = init_mlp_layers(
        [1, 4, 1],
        hidden_activation="tanh",
        output_activation="identity",
        weight_scale=0.15,
        seed=0,
    )
    model = PCNetwork(
        layers=layers,
        eta_x=0.2,
        eta_w=0.05,
        eta_b=0.05,
        train_steps=25,
        eval_steps=25,
        state_init="forward",
    )

    history = model.fit(x, y, epochs=40, seed=0)
    predictions = model.predict(x)

    baseline_prediction = np.full_like(y, np.mean(y))
    baseline_mse = float(np.mean((baseline_prediction - y) ** 2))
    model_mse = float(np.mean((predictions - y) ** 2))

    assert history["pre_update_energy"][0] > history["post_update_energy"][-1]
    assert model_mse < baseline_mse
