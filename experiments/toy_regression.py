from __future__ import annotations

import numpy as np

from pc.layers import init_mlp_layers
from pc.models import PCNetwork
from pc.utils import set_seed


def make_toy_regression_data() -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic tiny regression dataset with x and y shaped (B, 1)."""
    x = np.linspace(-1.0, 1.0, 16, dtype=np.float64).reshape(-1, 1)
    y = (0.75 * x) - 0.1
    return x, y


def main() -> None:
    """Run the Phase 0 toy regression experiment and print scalar energy history."""
    set_seed(0)
    x, y = make_toy_regression_data()
    layers = init_mlp_layers(
        layer_dims=[1, 4, 1],
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
    history = model.fit(x, y, epochs=60, seed=0)
    predictions = model.predict(x)
    mse = float(np.mean((predictions - y) ** 2))

    print("Toy regression completed.")
    print(f"Initial pre-update energy: {history['pre_update_energy'][0]:.6f}")
    print(f"Final pre-update energy: {history['pre_update_energy'][-1]:.6f}")
    print(f"Final post-update energy: {history['post_update_energy'][-1]:.6f}")
    print(f"Final prediction MSE: {mse:.6f}")


if __name__ == "__main__":
    main()
