from __future__ import annotations

import numpy as np

from pc.inference import build_clamped_mask, initialize_states, run_inference
from pc.layers import PCLayerParams, init_mlp_layers
from pc.models import PCNetwork


def make_stable_layers() -> list[PCLayerParams]:
    return [
        PCLayerParams(
            weight=np.array([[0.4], [-0.3]], dtype=np.float64),
            bias=np.array([0.05, -0.02], dtype=np.float64),
            sigma2=1.0,
            activation_name="tanh",
        ),
        PCLayerParams(
            weight=np.array([[0.6, -0.2]], dtype=np.float64),
            bias=np.array([0.1], dtype=np.float64),
            sigma2=1.0,
            activation_name="identity",
        ),
    ]


def test_clamped_layers_remain_unchanged_and_hidden_state_moves() -> None:
    layers = make_stable_layers()
    x = np.array([[1.0], [-0.5]], dtype=np.float64)
    y = np.array([[0.3], [-0.2]], dtype=np.float64)
    initial_states = initialize_states(layers, x, y=y, init="zeros", mode="train")
    clamped_mask = build_clamped_mask(len(layers) + 1, mode="train")

    result = run_inference(
        initial_states,
        layers,
        clamped_mask,
        eta_x=0.2,
        steps=10,
        record_trace=True,
    )

    np.testing.assert_allclose(result.states[0], initial_states[0])
    np.testing.assert_allclose(result.states[2], initial_states[2])
    assert not np.allclose(result.states[1], initial_states[1])


def test_training_inference_reduces_energy_net() -> None:
    layers = make_stable_layers()
    x = np.array([[1.0], [-0.5], [0.25]], dtype=np.float64)
    y = np.array([[0.3], [-0.2], [0.15]], dtype=np.float64)
    initial_states = initialize_states(layers, x, y=y, init="forward", mode="train")
    clamped_mask = build_clamped_mask(len(layers) + 1, mode="train")

    result = run_inference(
        initial_states,
        layers,
        clamped_mask,
        eta_x=0.2,
        steps=15,
        record_trace=True,
    )

    assert result.energy_trace[0] > result.energy_trace[-1]


def test_deterministic_inference_trace_under_fixed_seed() -> None:
    x = np.linspace(-1.0, 1.0, 6, dtype=np.float64).reshape(-1, 1)
    y = 0.5 * x
    layers_a = init_mlp_layers([1, 3, 1], seed=13)
    layers_b = init_mlp_layers([1, 3, 1], seed=13)
    model_a = PCNetwork(layers=layers_a, eta_x=0.2, eta_w=0.05, train_steps=12, eval_steps=12)
    model_b = PCNetwork(layers=layers_b, eta_x=0.2, eta_w=0.05, train_steps=12, eval_steps=12)

    result_a = model_a.infer(x, y=y, mode="train", record_trace=True)
    result_b = model_b.infer(x, y=y, mode="train", record_trace=True)

    np.testing.assert_allclose(result_a.energy_trace, result_b.energy_trace)


def test_predict_omits_trace_computation() -> None:
    layers = init_mlp_layers([1, 3, 1], seed=17)
    model = PCNetwork(layers=layers, eta_x=0.2, eta_w=0.05, train_steps=5, eval_steps=5)
    x = np.linspace(-1.0, 1.0, 4, dtype=np.float64).reshape(-1, 1)

    result = model.infer(x, mode="predict", record_trace=False)
    prediction = model.predict(x)

    assert result.energy_trace == []
    np.testing.assert_allclose(prediction, result.states[-1])
