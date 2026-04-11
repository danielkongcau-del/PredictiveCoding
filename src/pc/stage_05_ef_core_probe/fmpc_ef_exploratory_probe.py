from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np

from ..activations import get_activation
from ..datasets import load_digits_split
from ..energy import compute_cache, total_energy
from ..stage_03_transport_core_v1.fmpc_tf1_flow import (
    FMPCTF1Context,
    bootstrap_average_velocity_target,
    build_tf1_context,
    hidden_energy_from_state,
    hidden_local_flow,
    hidden_states_from_state,
    rollout_hidden_transport,
    validate_tf1_time_pair,
)
from ..stage_03_transport_core_v1.fmpc_tf1_jvp import build_tf1_input, forward_tf1_mlp_with_jvp
from ..layers import init_mlp_layers
from ..metrics import classification_accuracy
from ..minibatch import iter_minibatches
from ..mlp_baseline import MLPNetwork, init_mlp_baseline_layers
from ..models import PCNetwork
from ..training import apply_parameter_updates, parameter_gradients
from ..utils import ensure_finite_array, set_seed

ProbeStage = Literal["warmup", "transition", "hybrid"]
OutputLayout = Literal["single_dir", "run_id_subdir"]
ResidualIdentityMode = Literal["residual_corrected_meanflow"]

RESIDUAL_MEANFLOW_TRANSPORT_FAMILY = "residual_meanflow_core"
RESIDUAL_IDENTITY_MODE = "residual_corrected_meanflow"
U_PSI_INPUT_CONTRACT = "concat([z_t, target_onehot, t, r])"
BOOTSTRAP_TARGET_CONTRACT = "m_boot = ((Phi_LF_r(z_t; c) - z_t) / r) - g_t"
RESIDUAL_IDENTITY_TARGET_CONTRACT = "m_id = r * D_T g_t + r * D_T m_psi"


def _as_batch_first(name: str, array: np.ndarray) -> np.ndarray:
    array_float = np.asarray(array, dtype=np.float64)
    if array_float.ndim != 2:
        raise ValueError(f"{name} must be shaped (batch, features).")
    return array_float


@dataclass
class FMPCEFExploratoryProbeConfig:
    """Configuration for the Stage 05 corrected residual MeanFlow core probe."""

    experiment_name: str = "fmpc_ef_exploratory_probe"
    dataset_name: str = "digits"
    run_seed: int = 0
    data_seed: int = 0
    model_init_seed: int = 0
    psi_init_seed: int = 0
    batch_order_seed: int = 0
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: OutputLayout = "single_dir"
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    layer_dims: tuple[int, ...] = (64, 16, 10)
    hidden_activation: str = "tanh"
    output_activation: str = "identity"
    weight_scale: float = 0.05
    sigma2: float | tuple[float, ...] = 1.0
    eta_x: float = 0.10
    eta_w: float = 0.02
    eta_b: float | None = 0.02
    eval_steps: int = 15
    state_init: str = "forward"
    epochs: int = 12
    batch_size: int = 128
    shuffle_batches: bool = True
    transport_steps: int = 2
    warmup_epochs: int | None = None
    lambda_id_warmup_epochs: int = 3
    lambda_id_ramp_epochs: int = 3
    identity_loss_weight: float = 0.1
    tangent_epsilon: float = 1e-3
    identity_mode: ResidualIdentityMode = RESIDUAL_IDENTITY_MODE
    psi_hidden_dims: tuple[int, ...] = (128,)
    psi_weight_scale: float = 0.01
    psi_eta_w: float = 0.01
    psi_eta_b: float | None = 0.01
    bootstrap_integrator: Literal["euler", "rk2"] = "rk2"
    bootstrap_substeps: int = 4
    selection_metric: Literal["val_configured_transported_final_energy"] = (
        "val_configured_transported_final_energy"
    )

    def __post_init__(self) -> None:
        if self.warmup_epochs is not None:
            object.__setattr__(self, "lambda_id_warmup_epochs", int(self.warmup_epochs))
        if self.dataset_name != "digits":
            raise ValueError("The first exploratory probe currently supports digits only.")
        if len(self.layer_dims) < 3:
            raise ValueError("Exploratory probe expects at least one hidden layer.")
        if self.transport_steps <= 0:
            raise ValueError("transport_steps must be positive.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.bootstrap_substeps <= 0:
            raise ValueError("bootstrap_substeps must be positive.")
        if self.lambda_id_warmup_epochs < 0:
            raise ValueError("lambda_id_warmup_epochs must be non-negative.")
        if self.lambda_id_ramp_epochs < 0:
            raise ValueError("lambda_id_ramp_epochs must be non-negative.")
        if self.identity_loss_weight < 0.0:
            raise ValueError("identity_loss_weight must be non-negative.")
        if self.tangent_epsilon <= 0.0:
            raise ValueError("tangent_epsilon must be positive.")
        if self.identity_mode != RESIDUAL_IDENTITY_MODE:
            raise ValueError(
                f"Stage 05 currently supports identity_mode='{RESIDUAL_IDENTITY_MODE}' only."
            )
        if self.selection_metric != "val_configured_transported_final_energy":
            raise ValueError("Only val_configured_transported_final_energy is supported.")

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_{self.run_seed}"


@dataclass(frozen=True)
class ProbeMechanismMetrics:
    transport_steps: int
    initial_energy: float
    identity_final_energy: float
    local_field_only_final_energy: float
    transported_final_energy: float
    energy_delta_vs_identity: float
    energy_delta_vs_local_field_only: float
    initial_fixed_point_residual_rms: float
    identity_final_fixed_point_residual_rms: float
    local_field_only_final_fixed_point_residual_rms: float
    transported_final_fixed_point_residual_rms: float
    fixed_point_residual_delta_vs_identity: float
    fixed_point_residual_delta_vs_local_field_only: float


@dataclass(frozen=True)
class CorrectedResidualIdentityTarget:
    """Corrected residual identity target and its explicit constituent terms."""

    target: np.ndarray
    anchor_derivative: np.ndarray
    residual_jvp: np.ndarray
    anchor_term: np.ndarray
    residual_term: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", _as_batch_first("target", self.target))
        object.__setattr__(
            self,
            "anchor_derivative",
            _as_batch_first("anchor_derivative", self.anchor_derivative),
        )
        object.__setattr__(self, "residual_jvp", _as_batch_first("residual_jvp", self.residual_jvp))
        object.__setattr__(self, "anchor_term", _as_batch_first("anchor_term", self.anchor_term))
        object.__setattr__(self, "residual_term", _as_batch_first("residual_term", self.residual_term))
        if self.target.shape != self.anchor_derivative.shape:
            raise ValueError("target and anchor_derivative must share the same shape.")
        if self.target.shape != self.residual_jvp.shape:
            raise ValueError("target and residual_jvp must share the same shape.")
        if self.target.shape != self.anchor_term.shape:
            raise ValueError("target and anchor_term must share the same shape.")
        if self.target.shape != self.residual_term.shape:
            raise ValueError("target and residual_term must share the same shape.")


@dataclass(frozen=True)
class FMPCEFExploratoryProbeEpochMetrics:
    epoch: int
    stage: str
    lambda_id: float
    train_total_loss: float
    train_boot_loss: float
    train_identity_loss: float
    train_transported_final_energy: float
    val_one_step_transported_final_energy: float
    val_one_step_energy_delta_vs_identity: float
    val_one_step_fixed_point_residual_delta_vs_identity: float
    val_configured_transported_final_energy: float
    val_configured_energy_delta_vs_identity: float
    val_configured_fixed_point_residual_delta_vs_identity: float
    val_accuracy: float
    val_output_mse: float


@dataclass
class FMPCEFExploratoryProbeEpochSnapshot:
    epoch: int
    model_snapshot: list[tuple[np.ndarray, np.ndarray]]
    psi_snapshot: list[tuple[np.ndarray, np.ndarray]]


@dataclass
class FMPCEFExploratoryProbeRunResult:
    run_dir: Path
    config: dict[str, Any]
    epoch_metrics: list[dict[str, Any]]
    summary: dict[str, Any]
    model: PCNetwork | None = None
    psi_network: MLPNetwork | None = None


def build_fmpc_ef_exploratory_probe_config(
    **overrides: Any,
) -> FMPCEFExploratoryProbeConfig:
    """Return the canonical Stage 05 corrected residual MeanFlow config."""

    payload: dict[str, Any] = {
        "layer_dims": (64, 16, 10),
        "transport_steps": 2,
        "lambda_id_warmup_epochs": 3,
        "lambda_id_ramp_epochs": 3,
        "identity_loss_weight": 0.1,
        "tangent_epsilon": 1e-3,
        "identity_mode": RESIDUAL_IDENTITY_MODE,
        "epochs": 12,
        "batch_size": 128,
        "eval_steps": 15,
        "psi_hidden_dims": (128,),
        "psi_weight_scale": 0.01,
    }
    payload.update(overrides)
    return FMPCEFExploratoryProbeConfig(**payload)


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


def _resolve_run_dir(
    output_root: str | Path,
    experiment_name: str,
    run_id: str,
    output_layout: OutputLayout,
) -> Path:
    if output_layout == "single_dir":
        return Path(output_root) / experiment_name
    if output_layout == "run_id_subdir":
        return Path(output_root) / experiment_name / run_id
    raise ValueError(f"Unsupported output_layout '{output_layout}'.")


def _prepare_run_dir(run_dir: Path) -> Path:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, indent=2)


def _write_epoch_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("epoch metrics must contain at least one row.")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _sigma2_payload(sigma2: float | tuple[float, ...]) -> float | list[float]:
    if isinstance(sigma2, tuple):
        return [float(value) for value in sigma2]
    return float(sigma2)


def _snapshot_pc_parameters(model: PCNetwork) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(layer.weight.copy(), layer.bias.copy()) for layer in model.layers]


def _restore_pc_parameters(model: PCNetwork, snapshot: list[tuple[np.ndarray, np.ndarray]]) -> None:
    if len(snapshot) != len(model.layers):
        raise ValueError("PC parameter snapshot must align with model layers.")
    for layer, (weight, bias) in zip(model.layers, snapshot, strict=True):
        layer.weight = weight.copy()
        layer.bias = bias.copy()


def _snapshot_mlp_parameters(network: MLPNetwork) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(layer.weight.copy(), layer.bias.copy()) for layer in network.layers]


def _restore_mlp_parameters(
    network: MLPNetwork,
    snapshot: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    if len(snapshot) != len(network.layers):
        raise ValueError("MLP parameter snapshot must align with network layers.")
    for layer, (weight, bias) in zip(network.layers, snapshot, strict=True):
        layer.weight = weight.copy()
        layer.bias = bias.copy()


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


def _make_psi_network(config: FMPCEFExploratoryProbeConfig) -> MLPNetwork:
    hidden_dim = int(sum(config.layer_dims[1:-1]))
    target_dim = int(config.layer_dims[-1])
    layer_dims = [hidden_dim + target_dim + 2, *config.psi_hidden_dims, hidden_dim]
    return MLPNetwork(
        layers=init_mlp_baseline_layers(
            layer_dims,
            hidden_activation="tanh",
            output_activation="identity",
            weight_scale=config.psi_weight_scale,
            seed=config.psi_init_seed,
        ),
        eta_w=config.psi_eta_w,
        eta_b=config.psi_eta_b,
    )


def _sigmoid_unit_interval(progress: float) -> float:
    clipped = float(np.clip(progress, 0.0, 1.0))
    lo = 1.0 / (1.0 + np.exp(6.0))
    hi = 1.0 / (1.0 + np.exp(-6.0))
    raw = 1.0 / (1.0 + np.exp(-(12.0 * clipped - 6.0)))
    return float((raw - lo) / (hi - lo))


def lambda_id_for_epoch(config: FMPCEFExploratoryProbeConfig, epoch_index: int) -> float:
    if epoch_index < config.lambda_id_warmup_epochs:
        return 0.0
    if config.identity_loss_weight <= 0.0:
        return 0.0
    if config.lambda_id_ramp_epochs <= 0:
        return float(config.identity_loss_weight)
    progress = (epoch_index - config.lambda_id_warmup_epochs + 1) / float(
        config.lambda_id_ramp_epochs
    )
    return float(config.identity_loss_weight) * _sigmoid_unit_interval(progress)


def _stage_for_epoch(config: FMPCEFExploratoryProbeConfig, epoch_index: int) -> ProbeStage:
    if epoch_index < config.lambda_id_warmup_epochs:
        return "warmup"
    if config.lambda_id_ramp_epochs > 0 and epoch_index < (
        config.lambda_id_warmup_epochs + config.lambda_id_ramp_epochs
    ):
        return "transition"
    return "hybrid"


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
) -> None:
    activations, pre_activations = _forward_mlp(network, inputs)
    predictions = activations[-1]
    targets = _as_batch_first("target", target)
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must share the same shape.")
    if loss_scale <= 0.0:
        raise ValueError("loss_scale must be positive.")

    output_size = float(predictions.size)
    delta = (2.0 * float(loss_scale) / output_size) * (predictions - targets)
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


def approximate_anchor_directional_derivative(
    context: FMPCTF1Context,
    z_t: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    """Approximate `D_T g_t` along the current exact local-flow direction."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    z_array = _as_batch_first("z_t", z_t)
    if z_array.shape[1] == 0:
        return np.zeros_like(z_array)
    g_t = hidden_local_flow(context, z_array)
    z_plus = z_array + float(epsilon) * g_t
    z_minus = z_array - float(epsilon) * g_t
    g_plus = hidden_local_flow(context, z_plus)
    g_minus = hidden_local_flow(context, z_minus)
    derivative = (g_plus - g_minus) / (2.0 * float(epsilon))
    ensure_finite_array(derivative, "stage05_anchor_directional_derivative")
    return derivative


def build_corrected_residual_identity_target(
    context: FMPCTF1Context,
    psi_network: MLPNetwork,
    z_t: np.ndarray,
    target_onehot: np.ndarray,
    *,
    t: float,
    r: float,
    tangent_epsilon: float,
) -> CorrectedResidualIdentityTarget:
    """Return the Stage 05 corrected residual identity target.

    The target is `m_id = r * D_T g_t + r * D_T m_psi`.
    """

    validate_tf1_time_pair(t, r)
    z_array = _as_batch_first("z_t", z_t)
    target_array = _as_batch_first("target_onehot", target_onehot)
    if z_array.shape[0] != target_array.shape[0]:
        raise ValueError("z_t and target_onehot must share the same batch size.")
    g_t = hidden_local_flow(context, z_array)
    inputs = build_exploratory_probe_input(z_array, target_array, t=t, r=r)
    input_tangent = build_residual_input_tangent(g_t, target_dim=target_array.shape[1])
    residual_jvp = forward_tf1_mlp_with_jvp(psi_network, inputs, input_tangent).jvp
    anchor_derivative = approximate_anchor_directional_derivative(
        context,
        z_array,
        epsilon=tangent_epsilon,
    )
    anchor_term = float(r) * anchor_derivative
    residual_term = float(r) * residual_jvp
    return CorrectedResidualIdentityTarget(
        target=anchor_term + residual_term,
        anchor_derivative=anchor_derivative,
        residual_jvp=residual_jvp,
        anchor_term=anchor_term,
        residual_term=residual_term,
    )


def _config_payload(config: FMPCEFExploratoryProbeConfig) -> dict[str, Any]:
    return {
        "phase": "post_incremental_bridge_exploratory",
        "stage": "ef_core_probe",
        "dataset": {
            "dataset_name": config.dataset_name,
            "train_fraction": float(config.train_fraction),
            "val_fraction": float(config.val_fraction),
            "test_fraction": float(config.test_fraction),
            "data_seed": int(config.data_seed),
        },
        "model": {
            "layer_dims": [int(value) for value in config.layer_dims],
            "hidden_activation": config.hidden_activation,
            "output_activation": config.output_activation,
            "weight_scale": float(config.weight_scale),
            "sigma2": _sigma2_payload(config.sigma2),
            "eta_x": float(config.eta_x),
            "eta_w": float(config.eta_w),
            "eta_b": float(config.eta_b if config.eta_b is not None else config.eta_w),
            "eval_steps": int(config.eval_steps),
            "state_init": config.state_init,
        },
        "transport": {
            "teacher_free": True,
            "uses_teacher_artifacts": False,
            "transport_family": RESIDUAL_MEANFLOW_TRANSPORT_FAMILY,
            "residual_identity_mode": config.identity_mode,
            "energy_substrate": "baseline_pc_energy",
            "local_flow_definition": "exact_negative_hidden_state_gradient",
            "direct_anchor_source": "self_bootstrap_local_field",
            "psi_family": "residual_local_flow_mlp",
            "velocity_parameterization": "u_psi = g_theta + residual_mlp(z_t, target_onehot, t, r)",
            "u_psi_input_contract": U_PSI_INPUT_CONTRACT,
            "bootstrap_target_contract": BOOTSTRAP_TARGET_CONTRACT,
            "residual_identity_target_contract": RESIDUAL_IDENTITY_TARGET_CONTRACT,
            "identity_loss_weight": float(config.identity_loss_weight),
            "tangent_epsilon": float(config.tangent_epsilon),
            "lambda_id_warmup_epochs": int(config.lambda_id_warmup_epochs),
            "lambda_id_ramp_epochs": int(config.lambda_id_ramp_epochs),
            "no_teacher_dependency_in_target_construction": True,
            "use_teacher_free_features": False,
            "transport_scope": "train_only",
            "transport_steps": int(config.transport_steps),
            "bootstrap_integrator": config.bootstrap_integrator,
            "bootstrap_substeps": int(config.bootstrap_substeps),
            "selection_metric": config.selection_metric,
            "selection_metric_source": "val_metric",
            "report_metric_source": "test_metric",
            "acceptance_contract": "mechanism_first",
        },
        "psi_network": {
            "hidden_dims": [int(value) for value in config.psi_hidden_dims],
            "weight_scale": float(config.psi_weight_scale),
            "eta_w": float(config.psi_eta_w),
            "eta_b": float(config.psi_eta_b if config.psi_eta_b is not None else config.psi_eta_w),
            "psi_init_seed": int(config.psi_init_seed),
        },
        "run": {
            "run_seed": int(config.run_seed),
            "model_init_seed": int(config.model_init_seed),
            "batch_order_seed": int(config.batch_order_seed),
            "epochs": int(config.epochs),
            "batch_size": int(config.batch_size),
            "shuffle_batches": bool(config.shuffle_batches),
            "output_layout": config.output_layout,
        },
    }


def _collect_residual_supervision(
    context: FMPCTF1Context,
    psi_network: MLPNetwork,
    config: FMPCEFExploratoryProbeConfig,
    *,
    z_knots: list[np.ndarray],
    knot_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_blocks: list[np.ndarray] = []
    boot_targets: list[np.ndarray] = []
    identity_targets: list[np.ndarray] = []
    for knot_index, t_k in enumerate(knot_times[:-1]):
        z_t = z_knots[knot_index]
        t_float = float(t_k)
        r_k = 1.0 - t_float
        inputs = build_exploratory_probe_input(
            z_t,
            context.targets,
            t=t_float,
            r=r_k,
        )
        u_boot = bootstrap_average_velocity_target(
            context,
            z_t,
            t=t_float,
            r=r_k,
            integrator=config.bootstrap_integrator,
            substeps=config.bootstrap_substeps,
        )
        g_t = hidden_local_flow(context, z_t)
        corrected_identity = build_corrected_residual_identity_target(
            context,
            psi_network,
            z_t,
            context.targets,
            t=t_float,
            r=r_k,
            tangent_epsilon=config.tangent_epsilon,
        )
        input_blocks.append(inputs)
        boot_targets.append(u_boot - g_t)
        identity_targets.append(corrected_identity.target)
    return (
        np.concatenate(input_blocks, axis=0).astype(np.float64, copy=False),
        np.concatenate(boot_targets, axis=0).astype(np.float64, copy=False),
        np.concatenate(identity_targets, axis=0).astype(np.float64, copy=False),
    )


def _learned_velocity_fn(
    context: FMPCTF1Context,
    psi_network: MLPNetwork,
):
    def _velocity(z_t: np.ndarray, t_k: float, r_k: float) -> np.ndarray:
        z_array = _as_batch_first("z_t", z_t)
        inputs = build_exploratory_probe_input(
            z_array,
            context.targets,
            t=t_k,
            r=r_k,
        )
        residual = _as_batch_first("residual_velocity", psi_network.predict(inputs))
        velocity = hidden_local_flow(context, z_array) + residual
        return _as_batch_first("velocity", velocity)

    return _velocity


def _theta_update_from_transported_state(
    model: PCNetwork,
    context: FMPCTF1Context,
    transported_z: np.ndarray,
) -> float:
    states = hidden_states_from_state(context, transported_z)
    cache = compute_cache(states, model.layers)
    pre_update_energy = total_energy(cache, model.layers, context.batch_size)
    weight_gradients, bias_gradients = parameter_gradients(states, cache, model.layers)
    apply_parameter_updates(
        model.layers,
        weight_gradients,
        bias_gradients,
        eta_w=model.eta_w,
        eta_b=model.eta_b,
    )
    return float(pre_update_energy)


def _hidden_residual_rms(context: FMPCTF1Context, z: np.ndarray) -> float:
    flow = hidden_local_flow(context, z)
    return float(np.sqrt(np.mean(flow * flow)))


def _evaluate_mechanism_metrics(
    model: PCNetwork,
    psi_network: MLPNetwork,
    config: FMPCEFExploratoryProbeConfig,
    x_split: np.ndarray,
    y_split: np.ndarray,
    *,
    transport_steps: int,
) -> ProbeMechanismMetrics:
    context = build_tf1_context(model, x_split, y_split)
    initial_energy = hidden_energy_from_state(context, context.z0)
    initial_residual_rms = _hidden_residual_rms(context, context.z0)
    identity = rollout_hidden_transport(
        context,
        context.z0,
        transport_steps=transport_steps,
        mode="identity",
    )
    local_field = rollout_hidden_transport(
        context,
        context.z0,
        transport_steps=transport_steps,
        mode="local_field_only",
    )
    learned = rollout_hidden_transport(
        context,
        context.z0,
        transport_steps=transport_steps,
        mode="learned",
        velocity_fn=_learned_velocity_fn(context, psi_network),
    )
    identity_residual = _hidden_residual_rms(context, identity.z_knots[-1])
    local_field_residual = _hidden_residual_rms(context, local_field.z_knots[-1])
    learned_residual = _hidden_residual_rms(context, learned.z_knots[-1])
    return ProbeMechanismMetrics(
        transport_steps=int(transport_steps),
        initial_energy=float(initial_energy),
        identity_final_energy=float(identity.final_energy),
        local_field_only_final_energy=float(local_field.final_energy),
        transported_final_energy=float(learned.final_energy),
        energy_delta_vs_identity=float(learned.final_energy - identity.final_energy),
        energy_delta_vs_local_field_only=float(learned.final_energy - local_field.final_energy),
        initial_fixed_point_residual_rms=float(initial_residual_rms),
        identity_final_fixed_point_residual_rms=float(identity_residual),
        local_field_only_final_fixed_point_residual_rms=float(local_field_residual),
        transported_final_fixed_point_residual_rms=float(learned_residual),
        fixed_point_residual_delta_vs_identity=float(learned_residual - identity_residual),
        fixed_point_residual_delta_vs_local_field_only=float(learned_residual - local_field_residual),
    )


def _evaluate_slow_pc_metrics(
    model: PCNetwork,
    x_split: np.ndarray,
    y_split: np.ndarray,
) -> tuple[float, float]:
    predictions = model.predict(x_split)
    loss = float(np.mean((predictions - y_split) ** 2))
    accuracy = classification_accuracy(predictions, y_split)
    return loss, accuracy


def _snapshot_for_epoch(
    epoch_snapshots: list[FMPCEFExploratoryProbeEpochSnapshot],
    epoch: int,
) -> FMPCEFExploratoryProbeEpochSnapshot:
    for snapshot in epoch_snapshots:
        if int(snapshot.epoch) == int(epoch):
            return snapshot
    raise ValueError(f"No snapshot recorded for epoch {epoch}.")


def _train_one_batch(
    model: PCNetwork,
    psi_network: MLPNetwork,
    config: FMPCEFExploratoryProbeConfig,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    *,
    lambda_id: float,
    stage: ProbeStage,
) -> tuple[float, float, float, float]:
    context = build_tf1_context(model, x_batch, y_batch)
    source_rollout = rollout_hidden_transport(
        context,
        context.z0,
        transport_steps=config.transport_steps,
        mode="local_field_only",
    )
    psi_inputs, boot_targets, identity_targets = _collect_residual_supervision(
        context,
        psi_network,
        config,
        z_knots=source_rollout.z_knots,
        knot_times=source_rollout.knot_times,
    )
    psi_predictions = psi_network.predict(psi_inputs)
    boot_loss = float(np.mean((psi_predictions - boot_targets) ** 2))
    identity_loss = float(np.mean((psi_predictions - identity_targets) ** 2))
    if lambda_id > 0.0:
        combined_target = (boot_targets + (lambda_id * identity_targets)) / (1.0 + lambda_id)
        loss_scale = 1.0 + lambda_id
    else:
        combined_target = boot_targets
        loss_scale = 1.0
    total_loss = boot_loss + (lambda_id * identity_loss)
    _weighted_mse_step(psi_network, psi_inputs, combined_target, loss_scale=loss_scale)

    if stage == "warmup":
        theta_rollout = source_rollout
    else:
        theta_rollout = rollout_hidden_transport(
            context,
            context.z0,
            transport_steps=config.transport_steps,
            mode="learned",
            velocity_fn=_learned_velocity_fn(context, psi_network),
        )

    transported_energy = _theta_update_from_transported_state(
        model,
        context,
        theta_rollout.z_knots[-1],
    )
    return total_loss, boot_loss, identity_loss, transported_energy


def run_fmpc_ef_exploratory_probe(
    config: FMPCEFExploratoryProbeConfig,
) -> FMPCEFExploratoryProbeRunResult:
    """Run the first post-bridge exploratory teacher-free core probe."""

    set_seed(config.run_seed)
    split = load_digits_split(
        split_seed=config.data_seed,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
    )
    model = _make_pc_model(config)
    psi_network = _make_psi_network(config)

    run_dir = _prepare_run_dir(
        _resolve_run_dir(
            config.output_root,
            config.experiment_name,
            config.resolved_run_id(),
            config.output_layout,
        )
    )
    _write_json(run_dir / "config.json", _config_payload(config))

    epoch_rows: list[dict[str, Any]] = []
    epoch_snapshots: list[FMPCEFExploratoryProbeEpochSnapshot] = []

    train_start = perf_counter()
    for epoch_index in range(config.epochs):
        stage = _stage_for_epoch(config, epoch_index)
        lambda_id = lambda_id_for_epoch(config, epoch_index)
        batch_total_losses: list[float] = []
        batch_boot_losses: list[float] = []
        batch_identity_losses: list[float] = []
        batch_transport_energies: list[float] = []
        batch_seed = config.batch_order_seed + epoch_index
        for x_batch, y_batch in iter_minibatches(
            split.x_train,
            split.y_train,
            config.batch_size,
            shuffle=config.shuffle_batches,
            seed=batch_seed,
        ):
            total_loss, boot_loss, identity_loss, transported_energy = _train_one_batch(
                model,
                psi_network,
                config,
                x_batch,
                y_batch,
                lambda_id=lambda_id,
                stage=stage,
            )
            batch_total_losses.append(total_loss)
            batch_boot_losses.append(boot_loss)
            batch_identity_losses.append(identity_loss)
            batch_transport_energies.append(transported_energy)

        val_one_step = _evaluate_mechanism_metrics(
            model,
            psi_network,
            config,
            split.x_val,
            split.y_val,
            transport_steps=1,
        )
        val_configured = _evaluate_mechanism_metrics(
            model,
            psi_network,
            config,
            split.x_val,
            split.y_val,
            transport_steps=config.transport_steps,
        )
        val_output_mse, val_accuracy = _evaluate_slow_pc_metrics(model, split.x_val, split.y_val)

        row = asdict(
            FMPCEFExploratoryProbeEpochMetrics(
                epoch=epoch_index + 1,
                stage=stage,
                lambda_id=float(lambda_id),
                train_total_loss=float(np.mean(batch_total_losses)),
                train_boot_loss=float(np.mean(batch_boot_losses)),
                train_identity_loss=float(np.mean(batch_identity_losses)),
                train_transported_final_energy=float(np.mean(batch_transport_energies)),
                val_one_step_transported_final_energy=val_one_step.transported_final_energy,
                val_one_step_energy_delta_vs_identity=val_one_step.energy_delta_vs_identity,
                val_one_step_fixed_point_residual_delta_vs_identity=(
                    val_one_step.fixed_point_residual_delta_vs_identity
                ),
                val_configured_transported_final_energy=val_configured.transported_final_energy,
                val_configured_energy_delta_vs_identity=val_configured.energy_delta_vs_identity,
                val_configured_fixed_point_residual_delta_vs_identity=(
                    val_configured.fixed_point_residual_delta_vs_identity
                ),
                val_accuracy=val_accuracy,
                val_output_mse=val_output_mse,
            )
        )
        epoch_rows.append(row)
        epoch_snapshots.append(
            FMPCEFExploratoryProbeEpochSnapshot(
                epoch=epoch_index + 1,
                model_snapshot=_snapshot_pc_parameters(model),
                psi_snapshot=_snapshot_mlp_parameters(psi_network),
            )
        )

    train_wall_time_seconds = float(perf_counter() - train_start)

    best_row = min(epoch_rows, key=lambda row: float(row["val_configured_transported_final_energy"]))
    selected_epoch = int(best_row["epoch"])
    selected_snapshot = _snapshot_for_epoch(epoch_snapshots, selected_epoch)
    _restore_pc_parameters(model, selected_snapshot.model_snapshot)
    _restore_mlp_parameters(psi_network, selected_snapshot.psi_snapshot)

    evaluation_start = perf_counter()
    val_one_step = _evaluate_mechanism_metrics(
        model,
        psi_network,
        config,
        split.x_val,
        split.y_val,
        transport_steps=1,
    )
    val_configured = _evaluate_mechanism_metrics(
        model,
        psi_network,
        config,
        split.x_val,
        split.y_val,
        transport_steps=config.transport_steps,
    )
    test_one_step = _evaluate_mechanism_metrics(
        model,
        psi_network,
        config,
        split.x_test,
        split.y_test,
        transport_steps=1,
    )
    test_configured = _evaluate_mechanism_metrics(
        model,
        psi_network,
        config,
        split.x_test,
        split.y_test,
        transport_steps=config.transport_steps,
    )
    val_output_mse, val_accuracy = _evaluate_slow_pc_metrics(model, split.x_val, split.y_val)
    test_output_mse, test_accuracy = _evaluate_slow_pc_metrics(model, split.x_test, split.y_test)
    evaluation_wall_time_seconds = float(perf_counter() - evaluation_start)

    mechanism_acceptance = {
        "one_step_energy_decrease_vs_identity": bool(val_one_step.energy_delta_vs_identity < 0.0),
        "one_step_energy_decrease_vs_local_field_only": bool(
            val_one_step.energy_delta_vs_local_field_only <= 0.0
        ),
        "configured_steps_energy_decrease_vs_identity": bool(
            val_configured.energy_delta_vs_identity < 0.0
        ),
        "configured_steps_fixed_point_residual_decrease_vs_identity": bool(
            val_configured.fixed_point_residual_delta_vs_identity < 0.0
        ),
        "configured_steps_fixed_point_residual_decrease_vs_local_field_only": bool(
            val_configured.fixed_point_residual_delta_vs_local_field_only <= 0.0
        ),
    }

    summary = {
        "phase": "post_incremental_bridge_exploratory",
        "stage": "ef_core_probe",
        "teacher_free": True,
        "uses_teacher_artifacts": False,
        "transport_family": RESIDUAL_MEANFLOW_TRANSPORT_FAMILY,
        "residual_identity_mode": config.identity_mode,
        "energy_substrate": "baseline_pc_energy",
        "local_flow_definition": "exact_negative_hidden_state_gradient",
        "direct_anchor_source": "self_bootstrap_local_field",
        "psi_family": "residual_local_flow_mlp",
        "transport_scope": "train_only",
        "transport_steps": int(config.transport_steps),
        "u_psi_input_contract": U_PSI_INPUT_CONTRACT,
        "bootstrap_target_contract": BOOTSTRAP_TARGET_CONTRACT,
        "residual_identity_target_contract": RESIDUAL_IDENTITY_TARGET_CONTRACT,
        "identity_loss_weight": float(config.identity_loss_weight),
        "tangent_epsilon": float(config.tangent_epsilon),
        "lambda_id_warmup_epochs": int(config.lambda_id_warmup_epochs),
        "lambda_id_ramp_epochs": int(config.lambda_id_ramp_epochs),
        "selection_metric_source": "val_metric",
        "report_metric_source": "test_metric",
        "selection_metric_name": config.selection_metric,
        "selection_metric_value": float(val_configured.transported_final_energy),
        "selection_metric_higher_is_better": False,
        "selected_epoch": int(selected_epoch),
        "selected_epoch_stage": str(best_row["stage"]),
        "selected_epoch_lambda_id": float(best_row["lambda_id"]),
        "train_wall_time_seconds": train_wall_time_seconds,
        "evaluation_wall_time_seconds": evaluation_wall_time_seconds,
        "val_accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy),
        "val_output_mse": float(val_output_mse),
        "test_output_mse": float(test_output_mse),
        "mechanism_acceptance": mechanism_acceptance,
        "mechanism_metrics": {
            "one_step": asdict(val_one_step),
            "configured_steps": asdict(val_configured),
        },
        "test_mechanism_metrics": {
            "one_step": asdict(test_one_step),
            "configured_steps": asdict(test_configured),
        },
        "identity_no_transport_baseline": {
            "one_step": {
                "final_energy": float(val_one_step.identity_final_energy),
                "fixed_point_residual_rms": float(
                    val_one_step.identity_final_fixed_point_residual_rms
                ),
            },
            "configured_steps": {
                "final_energy": float(val_configured.identity_final_energy),
                "fixed_point_residual_rms": float(
                    val_configured.identity_final_fixed_point_residual_rms
                ),
            },
        },
        "local_field_only_baseline": {
            "one_step": {
                "final_energy": float(val_one_step.local_field_only_final_energy),
                "fixed_point_residual_rms": float(
                    val_one_step.local_field_only_final_fixed_point_residual_rms
                ),
            },
            "configured_steps": {
                "final_energy": float(val_configured.local_field_only_final_energy),
                "fixed_point_residual_rms": float(
                    val_configured.local_field_only_final_fixed_point_residual_rms
                ),
            },
        },
        "deterministic_artifacts": True,
        "acceptance_contract": "mechanism_first",
        "task_accuracy_is_gate": False,
        "no_teacher_dependency_in_target_construction": True,
        "run_artifacts": {
            "config_json": "config.json",
            "epoch_metrics_csv": "epoch_metrics.csv",
            "summary_json": "summary.json",
        },
    }
    ensure_finite_array(np.asarray([summary["val_accuracy"], summary["test_accuracy"]]), "summary_metrics")

    _write_epoch_metrics(run_dir / "epoch_metrics.csv", epoch_rows)
    _write_json(run_dir / "summary.json", summary)

    return FMPCEFExploratoryProbeRunResult(
        run_dir=run_dir,
        config=_config_payload(config),
        epoch_metrics=epoch_rows,
        summary=summary,
        model=model,
        psi_network=psi_network,
    )
