from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .activations import get_activation
from .utils import ensure_finite_array, make_rng, set_seed


@dataclass
class MLPLayerParams:
    """Parameters for an MLP layer with W: (d_out, d_in) and b: (d_out,)."""

    weight: np.ndarray
    bias: np.ndarray
    activation_name: str

    def __post_init__(self) -> None:
        self.weight = np.asarray(self.weight, dtype=np.float64)
        self.bias = np.asarray(self.bias, dtype=np.float64)

        if self.weight.ndim != 2:
            raise ValueError("weight must be a rank-2 array shaped (d_out, d_in).")
        if self.bias.ndim != 1:
            raise ValueError("bias must be a rank-1 array shaped (d_out,).")
        if self.bias.shape[0] != self.weight.shape[0]:
            raise ValueError("bias.shape[0] must equal weight.shape[0] for each MLP layer.")
        get_activation(self.activation_name)


@dataclass
class MLPTrainBatchResult:
    """Training outputs for one batch with batch-first x and y arrays."""

    loss: float
    parameter_norms: dict[str, list[float]]


def init_mlp_baseline_layers(
    layer_dims: list[int] | tuple[int, ...],
    hidden_activation: str = "tanh",
    output_activation: str = "identity",
    weight_scale: float = 0.05,
    seed: int | None = None,
    dtype: Any = np.float64,
) -> list[MLPLayerParams]:
    """Initialize MLP layers for dimensions d_0..d_L using batch-first conventions."""
    if len(layer_dims) < 2:
        raise ValueError("layer_dims must contain at least input and output dimensions.")
    if weight_scale <= 0.0:
        raise ValueError("weight_scale must be positive.")

    rng = make_rng(seed)
    layers: list[MLPLayerParams] = []

    for layer_index in range(1, len(layer_dims)):
        out_features = int(layer_dims[layer_index])
        in_features = int(layer_dims[layer_index - 1])
        activation_name = (
            output_activation if layer_index == len(layer_dims) - 1 else hidden_activation
        )
        weight = rng.normal(
            loc=0.0,
            scale=weight_scale,
            size=(out_features, in_features),
        ).astype(dtype, copy=False)
        bias = np.zeros((out_features,), dtype=dtype)
        ensure_finite_array(weight, f"mlp_weight[{layer_index}]")
        layers.append(
            MLPLayerParams(
                weight=weight,
                bias=bias,
                activation_name=activation_name,
            )
        )
    return layers


def _forward_pass(
    layers: list[MLPLayerParams],
    x: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray | None]]:
    """Return activations h^0..h^L and pre-activations [None, a^1..a^L]."""
    activations: list[np.ndarray] = [x]
    pre_activations: list[np.ndarray | None] = [None]
    current = x

    for layer_index, layer in enumerate(layers, start=1):
        activation_fn, _ = get_activation(layer.activation_name)
        pre_activation = current @ layer.weight.T + layer.bias
        current = activation_fn(pre_activation)
        ensure_finite_array(pre_activation, f"mlp_pre_activation[{layer_index}]")
        ensure_finite_array(current, f"mlp_activation[{layer_index}]")
        pre_activations.append(pre_activation)
        activations.append(current)

    return activations, pre_activations


def _mean_squared_error_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Return mean squared error averaged over all output elements."""
    return float(np.mean((predictions - targets) ** 2))


def _backward_pass(
    layers: list[MLPLayerParams],
    activations: list[np.ndarray],
    pre_activations: list[np.ndarray | None],
    targets: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return dL/dW and dL/db for the mean squared error loss."""
    predictions = activations[-1]
    output_size = predictions.size
    delta = (2.0 / float(output_size)) * (predictions - targets)

    weight_gradients: list[np.ndarray] = [np.empty((0, 0), dtype=np.float64)] * len(layers)
    bias_gradients: list[np.ndarray] = [np.empty((0,), dtype=np.float64)] * len(layers)

    for layer_index in range(len(layers) - 1, -1, -1):
        layer = layers[layer_index]
        pre_activation = pre_activations[layer_index + 1]
        if pre_activation is None:
            raise ValueError("pre_activations must be present for every predictive layer.")
        _, activation_prime = get_activation(layer.activation_name)
        local_delta = delta * activation_prime(pre_activation)
        grad_w = local_delta.T @ activations[layer_index]
        grad_b = np.sum(local_delta, axis=0)
        ensure_finite_array(grad_w, f"mlp_dL/dW[{layer_index + 1}]")
        ensure_finite_array(grad_b, f"mlp_dL/db[{layer_index + 1}]")
        weight_gradients[layer_index] = grad_w
        bias_gradients[layer_index] = grad_b
        if layer_index > 0:
            delta = local_delta @ layer.weight

    return weight_gradients, bias_gradients


def _parameter_norms(layers: list[MLPLayerParams]) -> dict[str, list[float]]:
    return {
        "weight_norms": [float(np.linalg.norm(layer.weight)) for layer in layers],
        "bias_norms": [float(np.linalg.norm(layer.bias)) for layer in layers],
    }


@dataclass
class MLPNetwork:
    """Minimal NumPy MLP baseline with batch-first public APIs."""

    layers: list[MLPLayerParams]
    eta_w: float
    eta_b: float | None = None

    def __post_init__(self) -> None:
        if len(self.layers) == 0:
            raise ValueError("MLPNetwork requires at least one layer.")
        if self.eta_w <= 0.0:
            raise ValueError("eta_w must be positive.")
        if self.eta_b is None:
            self.eta_b = self.eta_w
        if self.eta_b <= 0.0:
            raise ValueError("eta_b must be positive.")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return predictions for x shaped (batch, d_0)."""
        x_array = np.asarray(x, dtype=np.float64)
        activations, _ = _forward_pass(self.layers, x_array)
        return activations[-1]

    def train_batch(self, x: np.ndarray, y: np.ndarray) -> MLPTrainBatchResult:
        """Train on one batch with x shaped (batch, d_0) and y shaped (batch, d_L)."""
        x_array = np.asarray(x, dtype=np.float64)
        y_array = np.asarray(y, dtype=np.float64)
        activations, pre_activations = _forward_pass(self.layers, x_array)
        weight_gradients, bias_gradients = _backward_pass(
            self.layers,
            activations,
            pre_activations,
            y_array,
        )

        for layer_index, layer in enumerate(self.layers):
            layer.weight = layer.weight - self.eta_w * weight_gradients[layer_index]
            layer.bias = layer.bias - self.eta_b * bias_gradients[layer_index]
            ensure_finite_array(layer.weight, f"mlp_weight[{layer_index + 1}]")
            ensure_finite_array(layer.bias, f"mlp_bias[{layer_index + 1}]")

        predictions = self.predict(x_array)
        loss = _mean_squared_error_loss(predictions, y_array)
        return MLPTrainBatchResult(
            loss=loss,
            parameter_norms=_parameter_norms(self.layers),
        )

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int,
        seed: int | None = None,
    ) -> dict[str, list[float]]:
        """Fit the model on full-batch x and y arrays for a fixed number of epochs."""
        if epochs <= 0:
            raise ValueError("epochs must be positive.")
        set_seed(seed)

        history: dict[str, list[float]] = {"loss": []}
        for _ in range(epochs):
            batch_result = self.train_batch(x, y)
            history["loss"].append(batch_result.loss)
        return history
