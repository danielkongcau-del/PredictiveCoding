
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..activations import get_activation
from ..layers import init_mlp_layers
from ..mlp_baseline import MLPNetwork, init_mlp_baseline_layers
from ..models import PCNetwork
from ..stage_03_transport_core_v1.fmpc_tf1_flow import (
    FMPCTF1Context,
    FMPCTF1StateFeatures,
    FMPCTF1StateFeatureTangents,
    hidden_local_flow,
    teacher_free_state_features,
)
from ..stage_03_transport_core_v1.fmpc_tf1_jvp import build_tf1_input
from ..utils import ensure_finite_array
from .common import _as_batch_first
from .configs import FMPCEFExploratoryProbeConfig

@dataclass
class Stage05ResidualCoreNetworks:
    """Stage 05 residual transport head.

    v1 uses the trajectory branch only.
    v2 adds a current-state correction branch.
    """

    trajectory_network: MLPNetwork
    state_network: MLPNetwork | None = None

    @property
    def uses_two_branch_residual_core(self) -> bool:
        return self.state_network is not None

@dataclass(frozen=True)
class Stage05ResidualCorePredictions:
    """Explicit Stage 05 residual decomposition at one `(z_t, t, r)` point."""

    trajectory_input: np.ndarray
    state_input: np.ndarray | None
    trajectory_residual: np.ndarray
    state_residual: np.ndarray
    total_residual: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "trajectory_input", _as_batch_first("trajectory_input", self.trajectory_input)
        )
        if self.state_input is not None:
            object.__setattr__(self, "state_input", _as_batch_first("state_input", self.state_input))
        object.__setattr__(
            self,
            "trajectory_residual",
            _as_batch_first("trajectory_residual", self.trajectory_residual),
        )
        object.__setattr__(self, "state_residual", _as_batch_first("state_residual", self.state_residual))
        object.__setattr__(self, "total_residual", _as_batch_first("total_residual", self.total_residual))

def build_exploratory_probe_input(
    z_t: np.ndarray,
    target_onehot: np.ndarray,
    *,
    t: float,
    r: float,
) -> np.ndarray:
    """Return the minimal batch-first `(z_t, target_onehot, t, r)` probe input."""

    return build_tf1_input(
        z_t,
        target_onehot,
        t=t,
        r=r,
        use_teacher_free_features=False,
    )

def build_state_branch_input(
    features: FMPCTF1StateFeatures,
) -> np.ndarray:
    """Return the v2 current-state branch input block `(g_t, e_out_t, F_t)`."""

    return np.concatenate(
        [features.g_t, features.e_out_t, features.F_t],
        axis=1,
    ).astype(np.float64, copy=False)

def build_state_branch_input_tangent(
    features: FMPCTF1StateFeatures,
    *,
    feature_aware_state_branch_tangents: bool,
    feature_tangents: FMPCTF1StateFeatureTangents | None = None,
) -> np.ndarray:
    """Return the v2 current-state branch tangent block.

    When feature-aware tangents are disabled, the state branch is treated as frozen
    side information for the identity JVP and therefore receives a zero tangent.
    """

    batch_size = int(features.g_t.shape[0])
    if feature_aware_state_branch_tangents:
        if feature_tangents is None:
            raise ValueError(
                "feature_tangents must be provided when feature_aware_state_branch_tangents=True."
            )
        return np.concatenate(
            [
                feature_tangents.Dg_g_t,
                feature_tangents.Dg_e_out_t,
                feature_tangents.Dg_F_t,
            ],
            axis=1,
        ).astype(np.float64, copy=False)
    feature_dim = int(features.g_t.shape[1]) + int(features.e_out_t.shape[1]) + int(features.F_t.shape[1])
    return np.zeros((batch_size, feature_dim), dtype=np.float64)

def _make_pc_model(config: FMPCEFExploratoryProbeConfig) -> PCNetwork:
    layers = init_mlp_layers(
        config.layer_dims,
        hidden_activation=config.hidden_activation,
        output_activation=config.output_activation,
        weight_scale=config.weight_scale,
        sigma2=config.sigma2,
        seed=config.model_init_seed,
    )
    return PCNetwork(
        layers=layers,
        eta_x=config.eta_x,
        eta_w=config.eta_w,
        eta_b=config.eta_b,
        train_steps=0,
        eval_steps=config.eval_steps,
        inference_backend="pc_euler",
        state_init=config.state_init,
    )

def _make_psi_network(config: FMPCEFExploratoryProbeConfig) -> Stage05ResidualCoreNetworks:
    hidden_dim = int(sum(config.layer_dims[1:-1]))
    target_dim = int(config.layer_dims[-1])
    trajectory_dims = [hidden_dim + target_dim + 2, *config.psi_hidden_dims, hidden_dim]
    trajectory_network = MLPNetwork(
        layers=init_mlp_baseline_layers(
            trajectory_dims,
            hidden_activation="tanh",
            output_activation="identity",
            weight_scale=config.psi_weight_scale,
            seed=config.psi_init_seed,
        ),
        eta_w=config.psi_eta_w,
        eta_b=config.psi_eta_b,
    )
    state_network: MLPNetwork | None = None
    if config.use_two_branch_residual_core:
        state_dims = [hidden_dim + target_dim + 1, *config.psi_hidden_dims, hidden_dim]
        state_network = MLPNetwork(
            layers=init_mlp_baseline_layers(
                state_dims,
                hidden_activation="tanh",
                output_activation="identity",
                weight_scale=config.psi_weight_scale,
                seed=config.psi_init_seed + 1,
            ),
            eta_w=config.psi_eta_w,
            eta_b=config.psi_eta_b,
        )
    return Stage05ResidualCoreNetworks(
        trajectory_network=trajectory_network,
        state_network=state_network,
    )

def _forward_mlp(
    network: MLPNetwork,
    inputs: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray | None]]:
    activations: list[np.ndarray] = [_as_batch_first("inputs", inputs)]
    pre_activations: list[np.ndarray | None] = [None]
    current = activations[0]
    for layer_index, layer in enumerate(network.layers, start=1):
        activation_fn, _ = get_activation(layer.activation_name)
        pre_activation = current @ layer.weight.T + layer.bias
        current = activation_fn(pre_activation)
        ensure_finite_array(pre_activation, f"stage05_pre_activation[{layer_index}]")
        ensure_finite_array(current, f"stage05_activation[{layer_index}]")
        pre_activations.append(pre_activation)
        activations.append(current)
    return activations, pre_activations

def _weighted_mse_step(
    network: MLPNetwork,
    inputs: np.ndarray,
    target: np.ndarray,
    *,
    loss_scale: float,
    sample_weights: np.ndarray | None = None,
) -> None:
    activations, pre_activations = _forward_mlp(network, inputs)
    predictions = activations[-1]
    targets = _as_batch_first("target", target)
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must share the same shape.")
    if loss_scale <= 0.0:
        raise ValueError("loss_scale must be positive.")

    output_size = float(predictions.size)
    delta = predictions - targets
    if sample_weights is not None:
        weights = _as_batch_first("sample_weights", sample_weights)
        if weights.shape != (predictions.shape[0], 1):
            raise ValueError("sample_weights must be shaped (batch, 1).")
        delta = weights * delta
    delta = (2.0 * float(loss_scale) / output_size) * delta
    for layer_index in range(len(network.layers) - 1, -1, -1):
        layer = network.layers[layer_index]
        pre_activation = pre_activations[layer_index + 1]
        if pre_activation is None:
            raise ValueError("pre_activations must be present for every layer.")
        _, activation_prime = get_activation(layer.activation_name)
        local_delta = delta * activation_prime(pre_activation)
        grad_w = local_delta.T @ activations[layer_index]
        grad_b = np.sum(local_delta, axis=0)
        next_delta = local_delta @ layer.weight if layer_index > 0 else None
        layer.weight = layer.weight - network.eta_w * grad_w
        layer.bias = layer.bias - network.eta_b * grad_b
        ensure_finite_array(layer.weight, f"stage05_weight[{layer_index + 1}]")
        ensure_finite_array(layer.bias, f"stage05_bias[{layer_index + 1}]")
        if next_delta is not None:
            delta = next_delta

def _predict_residual_from_inputs(
    residual_core: Stage05ResidualCoreNetworks,
    trajectory_inputs: np.ndarray,
    *,
    state_inputs: np.ndarray | None = None,
) -> Stage05ResidualCorePredictions:
    trajectory_predictions = _as_batch_first(
        "trajectory_residual",
        residual_core.trajectory_network.predict(trajectory_inputs),
    )
    state_predictions = np.zeros_like(trajectory_predictions)
    if residual_core.state_network is not None:
        if state_inputs is None:
            raise ValueError("state_inputs are required for the two-branch residual core.")
        state_predictions = _as_batch_first(
            "state_residual",
            residual_core.state_network.predict(state_inputs),
        )
    total_predictions = trajectory_predictions + state_predictions
    return Stage05ResidualCorePredictions(
        trajectory_input=trajectory_inputs,
        state_input=state_inputs,
        trajectory_residual=trajectory_predictions,
        state_residual=state_predictions,
        total_residual=total_predictions,
    )

def _weighted_two_branch_mse_step(
    residual_core: Stage05ResidualCoreNetworks,
    trajectory_inputs: np.ndarray,
    state_inputs: np.ndarray,
    target: np.ndarray,
    *,
    loss_scale: float,
    sample_weights: np.ndarray | None = None,
) -> None:
    if residual_core.state_network is None:
        raise ValueError("Two-branch update requires state_network.")
    predictions = _predict_residual_from_inputs(
        residual_core,
        trajectory_inputs,
        state_inputs=state_inputs,
    )
    _weighted_mse_step(
        residual_core.trajectory_network,
        trajectory_inputs,
        target - predictions.state_residual,
        loss_scale=loss_scale,
        sample_weights=sample_weights,
    )
    _weighted_mse_step(
        residual_core.state_network,
        state_inputs,
        target - predictions.trajectory_residual,
        loss_scale=loss_scale,
        sample_weights=sample_weights,
    )

def _weighted_explicit_transport_drift_step(
    residual_core: Stage05ResidualCoreNetworks,
    trajectory_inputs: np.ndarray,
    state_inputs: np.ndarray,
    transport_target: np.ndarray,
    drift_target: np.ndarray,
    identity_target: np.ndarray,
    *,
    lambda_drift: float,
    lambda_id: float,
) -> None:
    if residual_core.state_network is None:
        raise ValueError("Explicit transport-drift updates require state_network.")
    if lambda_drift < 0.0:
        raise ValueError("lambda_drift must be non-negative.")
    if lambda_id < 0.0:
        raise ValueError("lambda_id must be non-negative.")

    predictions = _predict_residual_from_inputs(
        residual_core,
        trajectory_inputs,
        state_inputs=state_inputs,
    )
    trajectory_loss_scale = 1.0 + float(lambda_id)
    trajectory_target = (
        transport_target
        + (float(lambda_id) * (identity_target - predictions.state_residual))
    ) / trajectory_loss_scale
    _weighted_mse_step(
        residual_core.trajectory_network,
        trajectory_inputs,
        trajectory_target,
        loss_scale=trajectory_loss_scale,
    )

    state_loss_scale = float(lambda_drift) + float(lambda_id)
    if state_loss_scale <= 0.0:
        return
    updated_predictions = _predict_residual_from_inputs(
        residual_core,
        trajectory_inputs,
        state_inputs=state_inputs,
    )
    state_target = (
        (float(lambda_drift) * drift_target)
        + (float(lambda_id) * (identity_target - updated_predictions.trajectory_residual))
    ) / state_loss_scale
    _weighted_mse_step(
        residual_core.state_network,
        state_inputs,
        state_target,
        loss_scale=state_loss_scale,
    )

def build_residual_input_tangent(
    g_t: np.ndarray,
    *,
    target_dim: int,
) -> np.ndarray:
    """Return the Stage 05 fixed-terminal-time residual input tangent."""

    g_array = _as_batch_first("g_t", g_t)
    batch_size = int(g_array.shape[0])
    target_tangent = np.zeros((batch_size, int(target_dim)), dtype=np.float64)
    t_tangent = np.full((batch_size, 1), 1.0, dtype=np.float64)
    r_tangent = np.full((batch_size, 1), -1.0, dtype=np.float64)
    return np.concatenate([g_array, target_tangent, t_tangent, r_tangent], axis=1).astype(
        np.float64,
        copy=False,
    )

def _residual_core_inputs_for_state(
    context: FMPCTF1Context,
    config: FMPCEFExploratoryProbeConfig,
    z_t: np.ndarray,
    target_onehot: np.ndarray,
    *,
    t: float,
    r: float,
) -> tuple[np.ndarray, np.ndarray | None, FMPCTF1StateFeatures | None]:
    trajectory_input = build_exploratory_probe_input(
        z_t,
        target_onehot,
        t=t,
        r=r,
    )
    if not config.use_two_branch_residual_core:
        return trajectory_input, None, None
    features = teacher_free_state_features(context, z_t)
    state_input = build_state_branch_input(features)
    return trajectory_input, state_input, features

def _predict_total_velocity_at_state(
    context: FMPCTF1Context,
    psi_network: Stage05ResidualCoreNetworks,
    config: FMPCEFExploratoryProbeConfig,
    z_t: np.ndarray,
    *,
    t: float,
    r: float,
) -> np.ndarray:
    z_array = _as_batch_first("z_t", z_t)
    trajectory_input, state_input, _ = _residual_core_inputs_for_state(
        context,
        config,
        z_array,
        context.targets,
        t=t,
        r=r,
    )
    residual_predictions = _predict_residual_from_inputs(
        psi_network,
        trajectory_input,
        state_inputs=state_input,
    )
    return _as_batch_first(
        "total_velocity",
        hidden_local_flow(context, z_array) + residual_predictions.total_residual,
    )

def _learned_velocity_fn(
    context: FMPCTF1Context,
    psi_network: Stage05ResidualCoreNetworks,
    config: FMPCEFExploratoryProbeConfig,
):
    def _velocity(z_t: np.ndarray, t_k: float, r_k: float) -> np.ndarray:
        return _predict_total_velocity_at_state(
            context,
            psi_network,
            config,
            z_t,
            t=t_k,
            r=r_k,
        )

    return _velocity
