from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .activations import get_activation
from .utils import ensure_finite_array, make_rng


@dataclass
class PCLayerParams:
    """Parameters for a predictive layer with W^l: (d_l, d_(l-1)) and b^l: (d_l,)."""

    weight: np.ndarray
    bias: np.ndarray
    sigma2: float
    activation_name: str

    def __post_init__(self) -> None:
        self.weight = np.asarray(self.weight)
        self.bias = np.asarray(self.bias)
        self.sigma2 = float(self.sigma2)

        if self.weight.ndim != 2:
            raise ValueError("weight must be a rank-2 array shaped (d_l, d_(l-1)).")
        if self.bias.ndim != 1:
            raise ValueError("bias must be a rank-1 array shaped (d_l,).")
        if self.bias.shape[0] != self.weight.shape[0]:
            raise ValueError(
                "bias.shape[0] must equal weight.shape[0] for each predictive layer."
            )
        if self.sigma2 <= 0.0:
            raise ValueError("sigma2 must be positive for each predictive layer.")
        get_activation(self.activation_name)


def _expand_sigma2(
    sigma2: None | float | list[float] | tuple[float, ...],
    num_layers: int,
) -> list[float]:
    if sigma2 is None:
        return [1.0] * num_layers
    if isinstance(sigma2, (int, float)):
        return [float(sigma2)] * num_layers
    sigma2_values = [float(value) for value in sigma2]
    if len(sigma2_values) != num_layers:
        raise ValueError(
            f"sigma2 must have length {num_layers}, received {len(sigma2_values)}."
        )
    return sigma2_values


def init_mlp_layers(
    layer_dims: list[int] | tuple[int, ...],
    hidden_activation: str = "tanh",
    output_activation: str = "identity",
    weight_scale: float = 0.05,
    sigma2: None | float | list[float] | tuple[float, ...] = None,
    seed: int | None = None,
    dtype: Any = np.float64,
) -> list[PCLayerParams]:
    """Initialize MLP layers for dimensions d_0..d_L using batch-first conventions."""
    if len(layer_dims) < 2:
        raise ValueError("layer_dims must contain at least input and output dimensions.")
    if weight_scale <= 0.0:
        raise ValueError("weight_scale must be positive.")

    rng = make_rng(seed)
    sigma2_values = _expand_sigma2(sigma2, len(layer_dims) - 1)
    layers: list[PCLayerParams] = []

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
        ensure_finite_array(weight, f"weight[{layer_index}]")
        layers.append(
            PCLayerParams(
                weight=weight,
                bias=bias,
                sigma2=sigma2_values[layer_index - 1],
                activation_name=activation_name,
            )
        )
    return layers
