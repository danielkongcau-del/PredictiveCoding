from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..activations import get_activation
from .fmpc_interval_normalization import FMPCIntervalNormalizationStats
from ..mlp_baseline import MLPNetwork


def _as_batch_first(name: str, array: np.ndarray) -> np.ndarray:
    array_float = np.asarray(array, dtype=np.float64)
    if array_float.ndim != 2:
        raise ValueError(f"{name} must be shaped (batch, features).")
    return array_float


@dataclass(frozen=True)
class MeanFlowMLPJVPResult:
    """Explicit forward-mode JVP result for the NumPy MLP family.

    Shape contract:
    - `output`: `(batch, output_dim)`
    - `jvp`: `(batch, output_dim)`
    """

    output: np.ndarray
    jvp: np.ndarray

    def __post_init__(self) -> None:
        output = _as_batch_first("output", self.output)
        jvp = _as_batch_first("jvp", self.jvp)
        if output.shape != jvp.shape:
            raise ValueError("output and jvp must share the same shape.")
        object.__setattr__(self, "output", output)
        object.__setattr__(self, "jvp", jvp)


def build_meanflow_input_tangent(
    normalization: FMPCIntervalNormalizationStats,
    g_s: np.ndarray,
    *,
    target_dim: int,
    teacher_feature_dim: int,
    teacher_feature_tangent: np.ndarray | None = None,
    d_tau_s: float = 1.0,
    d_tau_t: float = 0.0,
) -> np.ndarray:
    """Build the default MeanFlow input tangent in normalized input coordinates.

    Shape contract:
    - `g_s`: `(batch, z_dim)` in raw hidden-state coordinates
    - returns `(batch, z_dim + target_dim + 2 + teacher_feature_dim)`

    Tangent convention:
    - normalized `z_s` block: `g_s / z_state_std`
    - `target_onehot` block: `0`
    - `tau_s` block: `d_tau_s`
    - `tau_t` block: `d_tau_t`
    - teacher-feature block:
      - `0` when `teacher_feature_tangent is None`
      - otherwise `teacher_feature_tangent / teacher_feature_std`
    """

    g_array = _as_batch_first("g_s", g_s)
    if g_array.shape[1] != normalization.z_dim:
        raise ValueError(f"g_s feature dimension must be {normalization.z_dim}.")
    z_tangent = g_array / np.maximum(normalization.z_state_std, normalization.eps)
    batch_size = int(g_array.shape[0])
    target_tangent = np.zeros((batch_size, int(target_dim)), dtype=np.float64)
    tau_s_tangent = np.full((batch_size, 1), float(d_tau_s), dtype=np.float64)
    tau_t_tangent = np.full((batch_size, 1), float(d_tau_t), dtype=np.float64)
    if teacher_feature_tangent is None:
        teacher_feature_tangent_block = np.zeros((batch_size, int(teacher_feature_dim)), dtype=np.float64)
    else:
        teacher_feature_tangent_array = _as_batch_first("teacher_feature_tangent", teacher_feature_tangent)
        if teacher_feature_tangent_array.shape != (batch_size, int(teacher_feature_dim)):
            raise ValueError(
                "teacher_feature_tangent must be shaped (batch, teacher_feature_dim)."
            )
        teacher_feature_tangent_block = (
            teacher_feature_tangent_array / np.maximum(normalization.teacher_feature_std, normalization.eps)
        ).astype(np.float64, copy=False)
    return np.concatenate(
        [z_tangent, target_tangent, tau_s_tangent, tau_t_tangent, teacher_feature_tangent_block],
        axis=1,
    ).astype(np.float64, copy=False)


def forward_mlp_with_jvp(
    network: MLPNetwork,
    inputs: np.ndarray,
    input_tangent: np.ndarray,
) -> MeanFlowMLPJVPResult:
    """Evaluate an explicit NumPy MLP and its forward-mode JVP.

    Shape contract:
    - `inputs`: `(batch, input_dim)`
    - `input_tangent`: `(batch, input_dim)`
    - returns batch-first `(output, jvp)`
    """

    current = _as_batch_first("inputs", inputs)
    tangent = _as_batch_first("input_tangent", input_tangent)
    if current.shape != tangent.shape:
        raise ValueError("inputs and input_tangent must share the same shape.")

    for layer_index, layer in enumerate(network.layers, start=1):
        activation_fn, activation_prime = get_activation(layer.activation_name)
        pre_activation = current @ layer.weight.T + layer.bias
        pre_activation_tangent = tangent @ layer.weight.T
        current = activation_fn(pre_activation)
        tangent = activation_prime(pre_activation) * pre_activation_tangent
        if current.ndim != 2 or tangent.ndim != 2:
            raise RuntimeError(f"Layer {layer_index} produced a non-batch-first activation or JVP.")

    return MeanFlowMLPJVPResult(output=current, jvp=tangent)
