from __future__ import annotations

import numpy as np

from pc.benchmark_specs import get_benchmark_spec
from pc.metrics import majority_class_baseline_accuracy
from pc.mlp_baseline import init_mlp_baseline_layers


def test_mlp_initialization_is_deterministic() -> None:
    layers_a = init_mlp_baseline_layers([1, 4, 1], seed=17, weight_scale=0.1)
    layers_b = init_mlp_baseline_layers([1, 4, 1], seed=17, weight_scale=0.1)

    for layer_a, layer_b in zip(layers_a, layers_b, strict=True):
        assert np.allclose(layer_a.weight, layer_b.weight)
        assert np.allclose(layer_a.bias, layer_b.bias)
        assert layer_a.activation_name == layer_b.activation_name


def test_mlp_predict_returns_expected_shape() -> None:
    spec = get_benchmark_spec("toy_regression")
    x, _ = spec.make_data()
    model = spec.build_mlp_model()

    predictions = model.predict(x)

    assert predictions.shape == (x.shape[0], spec.layer_dims[-1])


def test_mlp_train_batch_changes_parameters() -> None:
    spec = get_benchmark_spec("toy_regression")
    x, y = spec.make_data()
    model = spec.build_mlp_model()
    initial_weights = [layer.weight.copy() for layer in model.layers]
    initial_biases = [layer.bias.copy() for layer in model.layers]

    batch_result = model.train_batch(x, y)

    assert batch_result.loss >= 0.0
    assert any(
        not np.allclose(initial_weight, layer.weight)
        for initial_weight, layer in zip(initial_weights, model.layers, strict=True)
    )
    assert any(
        not np.allclose(initial_bias, layer.bias)
        for initial_bias, layer in zip(initial_biases, model.layers, strict=True)
    )


def test_mlp_regression_loss_decreases_over_epochs() -> None:
    spec = get_benchmark_spec("toy_regression")
    x, y = spec.make_data()
    model = spec.build_mlp_model()
    initial_loss = float(np.mean((model.predict(x) - y) ** 2))

    history = model.fit(x, y, epochs=20, seed=spec.run_seed)

    assert history["loss"][-1] < initial_loss


def test_mlp_blobs_accuracy_beats_majority_baseline() -> None:
    spec = get_benchmark_spec("toy_blobs_classification")
    x, y = spec.make_data()
    model = spec.build_mlp_model()
    baseline_accuracy = majority_class_baseline_accuracy(y)
    best_accuracy = 0.0

    for _ in range(spec.epochs):
        model.train_batch(x, y)
        predictions = model.predict(x)
        accuracy = spec.primary_metric_fn(predictions, y)
        best_accuracy = max(best_accuracy, accuracy)

    assert best_accuracy > baseline_accuracy
