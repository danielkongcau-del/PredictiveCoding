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

from .activations import get_activation
from .datasets import load_digits_split
from .energy import compute_cache, total_energy
from .fmpc_tf1_flow import (
    FMPCTF1Context,
    FMPCTF1StateFeatureTangents,
    FMPCTF1StateFeatures,
    bootstrap_average_velocity_target,
    build_tf1_context,
    hidden_states_from_state,
    rollout_hidden_transport,
    teacher_free_feature_tangents,
    teacher_free_state_features,
)
from .fmpc_tf1_jvp import (
    build_tf1_input,
    build_tf1_input_tangent,
    forward_tf1_mlp_with_jvp,
)
from .layers import init_mlp_layers
from .metrics import classification_accuracy, majority_class_baseline_accuracy
from .minibatch import iter_minibatches
from .mlp_baseline import MLPNetwork, init_mlp_baseline_layers
from .models import PCNetwork
from .training import apply_parameter_updates, parameter_gradients
from .utils import ensure_finite_array, set_seed

TF1ModelVariant = Literal["tf1_mlp_core", "tf1_mlp_aug"]
TF1PresetName = Literal["mechanism_smoke", "baseline_comparable", "baseline_working_default"]
TF1CheckpointSelector = Literal[
    "energy_only",
    "val_accuracy_only",
    "gate_constrained_accuracy_then_energy",
    "gate_constrained_accuracy_then_val_accuracy",
]
OutputLayout = Literal["single_dir", "run_id_subdir"]


@dataclass
class FMPCTF1Config:
    """Configuration for the minimal teacher-free FMPC v1 experiment."""

    experiment_name: str = "fmpc_tf1"
    dataset_name: str = "digits"
    run_seed: int = 0
    data_seed: int = 0
    model_init_seed: int = 0
    psi_init_seed: int = 0
    batch_order_seed: int = 0
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: OutputLayout = "single_dir"
    preset_name: TF1PresetName | None = "mechanism_smoke"
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
    epochs: int = 60
    batch_size: int = 128
    shuffle_batches: bool = True
    model_variant: TF1ModelVariant = "tf1_mlp_core"
    use_teacher_free_features: bool = False
    feature_aware_tangents: bool = False
    tangent_epsilon: float = 1e-3
    transport_steps: int = 2
    psi_hidden_dims: tuple[int, ...] = (128,)
    psi_weight_scale: float = 0.05
    psi_eta_w: float = 0.01
    psi_eta_b: float | None = 0.01
    bootstrap_integrator: Literal["euler", "rk2"] = "rk2"
    bootstrap_substeps: int = 4
    identity_loss_weight: float = 0.1
    warmup_epochs: int = 5
    hybrid_ramp_epochs: int = 10
    selection_metric: Literal["val_transported_final_energy"] = "val_transported_final_energy"
    checkpoint_selector: TF1CheckpointSelector = "energy_only"

    def __post_init__(self) -> None:
        if self.dataset_name != "digits":
            raise ValueError("Phase TF1 v1 currently supports digits only.")
        if len(self.layer_dims) < 3:
            raise ValueError("TF1 expects at least one hidden layer and one output layer.")
        if self.transport_steps <= 0:
            raise ValueError("transport_steps must be positive.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.bootstrap_substeps <= 0:
            raise ValueError("bootstrap_substeps must be positive.")
        if self.identity_loss_weight < 0.0:
            raise ValueError("identity_loss_weight must be non-negative.")
        if self.warmup_epochs < 0 or self.hybrid_ramp_epochs < 0:
            raise ValueError("warmup_epochs and hybrid_ramp_epochs must be non-negative.")
        if self.tangent_epsilon <= 0.0:
            raise ValueError("tangent_epsilon must be positive.")
        if self.model_variant == "tf1_mlp_core" and self.use_teacher_free_features:
            raise ValueError("tf1_mlp_core must keep use_teacher_free_features=False.")
        if self.model_variant == "tf1_mlp_aug" and not self.use_teacher_free_features:
            raise ValueError("tf1_mlp_aug requires use_teacher_free_features=True.")
        if self.preset_name not in {None, "mechanism_smoke", "baseline_comparable", "baseline_working_default"}:
            raise ValueError(
                "preset_name must be None, 'mechanism_smoke', 'baseline_comparable', or 'baseline_working_default'."
            )
        if self.checkpoint_selector not in {
            "energy_only",
            "val_accuracy_only",
            "gate_constrained_accuracy_then_energy",
            "gate_constrained_accuracy_then_val_accuracy",
        }:
            raise ValueError(f"Unsupported checkpoint_selector '{self.checkpoint_selector}'.")

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_{self.run_seed}"


@dataclass(frozen=True)
class FMPCTF1EpochMetrics:
    epoch: int
    lambda_id: float
    stage: str
    train_loss: float
    train_boot_loss: float
    train_identity_loss: float
    train_transported_final_energy: float
    val_transported_final_energy: float
    val_identity_final_energy: float
    val_local_field_only_final_energy: float
    val_energy_delta_vs_identity: float
    val_energy_delta_vs_local_field_only: float
    val_accuracy: float
    val_baseline_accuracy: float


@dataclass
class FMPCTF1EpochSnapshot:
    epoch: int
    model_snapshot: list[tuple[np.ndarray, np.ndarray]]
    psi_snapshot: list[tuple[np.ndarray, np.ndarray]]


@dataclass
class FMPCTF1RunResult:
    run_dir: Path
    config: dict[str, Any]
    epoch_metrics: list[dict[str, Any]]
    summary: dict[str, Any]
    model: PCNetwork | None = None
    psi_network: MLPNetwork | None = None
    epoch_snapshots: list[FMPCTF1EpochSnapshot] | None = None
    selection_diagnostics: dict[str, Any] | None = None


def build_tf1_mechanism_smoke_config(**overrides: Any) -> FMPCTF1Config:
    """Return the canonical small-substrate TF1 preset."""

    payload: dict[str, Any] = {
        "preset_name": "mechanism_smoke",
        "layer_dims": (64, 16, 10),
        "model_variant": "tf1_mlp_core",
        "use_teacher_free_features": False,
        "feature_aware_tangents": False,
        "transport_steps": 2,
        "warmup_epochs": 5,
        "hybrid_ramp_epochs": 10,
        "epochs": 60,
        "eval_steps": 15,
        "psi_hidden_dims": (128,),
    }
    payload.update(overrides)
    return FMPCTF1Config(**payload)


def build_tf1_baseline_comparable_config(**overrides: Any) -> FMPCTF1Config:
    """Return the baseline-sized digits TF1 preset."""

    payload: dict[str, Any] = {
        "preset_name": "baseline_comparable",
        "layer_dims": (64, 64, 10),
        "model_variant": "tf1_mlp_core",
        "use_teacher_free_features": False,
        "feature_aware_tangents": False,
        "transport_steps": 2,
        "warmup_epochs": 5,
        "hybrid_ramp_epochs": 10,
        "epochs": 60,
        "eval_steps": 15,
        "psi_hidden_dims": (128,),
    }
    payload.update(overrides)
    return FMPCTF1Config(**payload)


def build_tf1_baseline_working_default_config(**overrides: Any) -> FMPCTF1Config:
    """Return the current evidence-driven but provisional TF1 working default."""

    payload: dict[str, Any] = {
        "preset_name": "baseline_working_default",
        "layer_dims": (64, 64, 10),
        "model_variant": "tf1_mlp_aug",
        "use_teacher_free_features": True,
        "feature_aware_tangents": False,
        "transport_steps": 1,
        "warmup_epochs": 5,
        "hybrid_ramp_epochs": 10,
        "epochs": 60,
        "eval_steps": 15,
        "psi_hidden_dims": (128,),
        "identity_loss_weight": 0.2,
        "checkpoint_selector": "gate_constrained_accuracy_then_val_accuracy",
    }
    payload.update(overrides)
    return FMPCTF1Config(**payload)


def build_tf1_preset_config(
    preset_name: TF1PresetName,
    **overrides: Any,
) -> FMPCTF1Config:
    """Build a TF1 config from one of the named canonical presets."""

    if preset_name == "mechanism_smoke":
        return build_tf1_mechanism_smoke_config(**overrides)
    if preset_name == "baseline_comparable":
        return build_tf1_baseline_comparable_config(**overrides)
    if preset_name == "baseline_working_default":
        return build_tf1_baseline_working_default_config(**overrides)
    raise ValueError(f"Unsupported TF1 preset '{preset_name}'.")


@dataclass(frozen=True)
class _SplitTransportMetrics:
    transported_final_energy: float
    identity_final_energy: float
    local_field_only_final_energy: float
    rollout_knots: list[float]


def build_tf1_identity_target(
    g_t: np.ndarray,
    remaining_horizon: float,
    jvp: np.ndarray,
) -> np.ndarray:
    """Return the TF1 MeanFlow identity target under fixed terminal time."""

    g_array = np.asarray(g_t, dtype=np.float64)
    jvp_array = np.asarray(jvp, dtype=np.float64)
    if g_array.shape != jvp_array.shape:
        raise ValueError("g_t and jvp must share the same shape.")
    return g_array + float(remaining_horizon) * jvp_array


def build_tf1_epoch_selection_diagnostics(
    epoch_metrics: list[dict[str, Any]],
    *,
    significant_accuracy_gap_threshold: float = 0.01,
) -> dict[str, Any]:
    """Summarize checkpoint-selection diagnostics from TF1 epoch metrics."""

    if not epoch_metrics:
        raise ValueError("epoch_metrics must contain at least one row.")

    gate_rows = [
        row
        for row in epoch_metrics
        if (
            float(row["val_transported_final_energy"]) < float(row["val_identity_final_energy"])
            and float(row["val_transported_final_energy"]) <= float(row["val_local_field_only_final_energy"])
            and float(row["val_accuracy"]) > float(row["val_baseline_accuracy"])
        )
    ]
    selection_specs = {
        "val_transported_final_energy": False,
        "val_accuracy": True,
        "val_energy_delta_vs_local_field_only": False,
    }
    selection_rules: dict[str, dict[str, Any]] = {}
    for metric_name, higher_is_better in selection_specs.items():
        if higher_is_better:
            best_row = max(epoch_metrics, key=lambda row: float(row[metric_name]))
        else:
            best_row = min(epoch_metrics, key=lambda row: float(row[metric_name]))
        selection_rules[metric_name] = {
            "selected_epoch": int(best_row["epoch"]),
            "selection_metric_name": metric_name,
            "selection_metric_higher_is_better": bool(higher_is_better),
            "selection_metric_value": float(best_row[metric_name]),
            "val_accuracy": float(best_row["val_accuracy"]),
            "val_transported_final_energy": float(best_row["val_transported_final_energy"]),
            "val_energy_delta_vs_identity": float(best_row["val_energy_delta_vs_identity"]),
            "val_energy_delta_vs_local_field_only": float(best_row["val_energy_delta_vs_local_field_only"]),
        }

    best_energy = selection_rules["val_transported_final_energy"]
    best_accuracy = selection_rules["val_accuracy"]
    accuracy_gap = float(best_accuracy["val_accuracy"] - best_energy["val_accuracy"])
    return {
        "selection_rules": selection_rules,
        "gate_passing_epoch_count": int(len(gate_rows)),
        "has_gate_passing_epoch": bool(gate_rows),
        "best_gate_passing_epoch_by_val_accuracy": (
            None
            if not gate_rows
            else {
                "selected_epoch": int(max(gate_rows, key=lambda row: float(row["val_accuracy"]))["epoch"]),
                "val_accuracy": float(max(float(row["val_accuracy"]) for row in gate_rows)),
            }
        ),
        "accuracy_gap_best_energy_vs_best_accuracy": accuracy_gap,
        "significant_accuracy_gap_threshold": float(significant_accuracy_gap_threshold),
        "significant_validation_accuracy_left_on_table": bool(
            accuracy_gap > float(significant_accuracy_gap_threshold)
        ),
    }


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


def _restore_mlp_parameters(network: MLPNetwork, snapshot: list[tuple[np.ndarray, np.ndarray]]) -> None:
    if len(snapshot) != len(network.layers):
        raise ValueError("MLP parameter snapshot must align with network layers.")
    for layer, (weight, bias) in zip(network.layers, snapshot, strict=True):
        layer.weight = weight.copy()
        layer.bias = bias.copy()


def _forward_mlp(
    network: MLPNetwork,
    inputs: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray | None]]:
    activations: list[np.ndarray] = [np.asarray(inputs, dtype=np.float64)]
    pre_activations: list[np.ndarray | None] = [None]
    current = activations[0]
    for layer_index, layer in enumerate(network.layers, start=1):
        activation_fn, _ = get_activation(layer.activation_name)
        pre_activation = current @ layer.weight.T + layer.bias
        current = activation_fn(pre_activation)
        ensure_finite_array(pre_activation, f"tf1_pre_activation[{layer_index}]")
        ensure_finite_array(current, f"tf1_activation[{layer_index}]")
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
    targets = np.asarray(target, dtype=np.float64)
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
        ensure_finite_array(layer.weight, f"tf1_weight[{layer_index + 1}]")
        ensure_finite_array(layer.bias, f"tf1_bias[{layer_index + 1}]")
        if next_delta is not None:
            delta = next_delta


def _psi_input_dim(config: FMPCTF1Config) -> int:
    hidden_dim = int(sum(config.layer_dims[1:-1]))
    target_dim = int(config.layer_dims[-1])
    if config.model_variant == "tf1_mlp_core":
        return hidden_dim + target_dim + 2
    return hidden_dim + target_dim + 2 + hidden_dim + target_dim + 1


def _make_pc_model(config: FMPCTF1Config) -> PCNetwork:
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


def _make_psi_network(config: FMPCTF1Config) -> MLPNetwork:
    hidden_dim = int(sum(config.layer_dims[1:-1]))
    layer_dims = [_psi_input_dim(config), *config.psi_hidden_dims, hidden_dim]
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


def _lambda_id_for_epoch(config: FMPCTF1Config, epoch_index: int) -> float:
    if epoch_index < config.warmup_epochs:
        return 0.0
    if config.hybrid_ramp_epochs <= 0:
        return float(config.identity_loss_weight)
    progress = (epoch_index - config.warmup_epochs + 1) / float(config.hybrid_ramp_epochs)
    return float(config.identity_loss_weight) * float(np.clip(progress, 0.0, 1.0))


def _stage_for_epoch(config: FMPCTF1Config, epoch_index: int) -> str:
    return "warmup" if epoch_index < config.warmup_epochs else "hybrid"


def _collect_psi_supervision(
    context: FMPCTF1Context,
    psi_network: MLPNetwork,
    config: FMPCTF1Config,
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
        features = teacher_free_state_features(context, z_t)
        feature_tangents: FMPCTF1StateFeatureTangents | None = None
        if config.use_teacher_free_features and config.feature_aware_tangents:
            feature_tangents = teacher_free_feature_tangents(
                context,
                z_t,
                epsilon=config.tangent_epsilon,
            )
        inputs = build_tf1_input(
            z_t,
            context.targets,
            t=t_float,
            r=r_k,
            use_teacher_free_features=config.use_teacher_free_features,
            features=features if config.use_teacher_free_features else None,
        )
        input_tangent = build_tf1_input_tangent(
            features.g_t,
            target_dim=context.target_dim,
            use_teacher_free_features=config.use_teacher_free_features,
            feature_aware_tangents=config.feature_aware_tangents,
            feature_tangents=feature_tangents,
        )
        jvp_result = forward_tf1_mlp_with_jvp(psi_network, inputs, input_tangent)
        u_boot = bootstrap_average_velocity_target(
            context,
            z_t,
            t=t_float,
            r=r_k,
            integrator=config.bootstrap_integrator,
            substeps=config.bootstrap_substeps,
        )
        u_identity = build_tf1_identity_target(features.g_t, r_k, jvp_result.jvp)
        input_blocks.append(inputs)
        boot_targets.append(u_boot)
        identity_targets.append(u_identity)
    return (
        np.concatenate(input_blocks, axis=0).astype(np.float64, copy=False),
        np.concatenate(boot_targets, axis=0).astype(np.float64, copy=False),
        np.concatenate(identity_targets, axis=0).astype(np.float64, copy=False),
    )


def _learned_velocity_fn(
    context: FMPCTF1Context,
    psi_network: MLPNetwork,
    config: FMPCTF1Config,
):
    def _velocity(z_t: np.ndarray, t_k: float, r_k: float) -> np.ndarray:
        features = teacher_free_state_features(context, z_t)
        feature_tangents: FMPCTF1StateFeatureTangents | None = None
        if config.use_teacher_free_features and config.feature_aware_tangents:
            feature_tangents = teacher_free_feature_tangents(
                context,
                z_t,
                epsilon=config.tangent_epsilon,
            )
        inputs = build_tf1_input(
            z_t,
            context.targets,
            t=t_k,
            r=r_k,
            use_teacher_free_features=config.use_teacher_free_features,
            features=features if config.use_teacher_free_features else None,
        )
        input_tangent = build_tf1_input_tangent(
            features.g_t,
            target_dim=context.target_dim,
            use_teacher_free_features=config.use_teacher_free_features,
            feature_aware_tangents=config.feature_aware_tangents,
            feature_tangents=feature_tangents,
        )
        return forward_tf1_mlp_with_jvp(psi_network, inputs, input_tangent).output

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


def _train_one_batch(
    model: PCNetwork,
    psi_network: MLPNetwork,
    config: FMPCTF1Config,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    *,
    lambda_id: float,
    stage: str,
) -> tuple[float, float, float, float]:
    context = build_tf1_context(model, x_batch, y_batch)
    source_rollout = rollout_hidden_transport(
        context,
        context.z0,
        transport_steps=config.transport_steps,
        mode="local_field_only",
    )
    theta_rollout = source_rollout if stage == "warmup" else None

    psi_inputs, boot_targets, identity_targets = _collect_psi_supervision(
        context,
        psi_network,
        config,
        source_rollout.z_knots,
        source_rollout.knot_times,
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

    if stage == "hybrid":
        theta_rollout = rollout_hidden_transport(
            context,
            context.z0,
            transport_steps=config.transport_steps,
            mode="learned",
            velocity_fn=_learned_velocity_fn(context, psi_network, config),
        )
    if theta_rollout is None:
        raise RuntimeError("theta rollout must be available before the parameter update.")
    transported_energy = _theta_update_from_transported_state(
        model,
        context,
        theta_rollout.z_knots[-1],
    )
    return total_loss, boot_loss, identity_loss, transported_energy


def _evaluate_transport_split(
    model: PCNetwork,
    psi_network: MLPNetwork,
    config: FMPCTF1Config,
    x_split: np.ndarray,
    y_split: np.ndarray,
) -> _SplitTransportMetrics:
    context = build_tf1_context(model, x_split, y_split)
    learned = rollout_hidden_transport(
        context,
        context.z0,
        transport_steps=config.transport_steps,
        mode="learned",
        velocity_fn=_learned_velocity_fn(context, psi_network, config),
    )
    identity = rollout_hidden_transport(
        context,
        context.z0,
        transport_steps=config.transport_steps,
        mode="identity",
    )
    local_field = rollout_hidden_transport(
        context,
        context.z0,
        transport_steps=config.transport_steps,
        mode="local_field_only",
    )
    return _SplitTransportMetrics(
        transported_final_energy=learned.final_energy,
        identity_final_energy=identity.final_energy,
        local_field_only_final_energy=local_field.final_energy,
        rollout_knots=[float(value) for value in learned.knot_times.tolist()],
    )


def _evaluate_slow_pc_accuracy(
    model: PCNetwork,
    x_split: np.ndarray,
    y_split: np.ndarray,
) -> tuple[float, float]:
    predictions = model.predict(x_split)
    loss = float(np.mean((predictions - y_split) ** 2))
    accuracy = classification_accuracy(predictions, y_split)
    return loss, accuracy


def _config_payload(config: FMPCTF1Config) -> dict[str, Any]:
    return {
        "phase": "Phase TF1",
        "stage": "teacher_free_fmpc_v1",
        "preset_name": config.preset_name,
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
            "model_variant": config.model_variant,
            "use_teacher_free_features": bool(config.use_teacher_free_features),
            "feature_aware_tangents": bool(config.feature_aware_tangents),
            "tangent_epsilon": float(config.tangent_epsilon),
            "transport_steps": int(config.transport_steps),
            "bootstrap_integrator": config.bootstrap_integrator,
            "bootstrap_substeps": int(config.bootstrap_substeps),
            "identity_loss_weight": float(config.identity_loss_weight),
            "warmup_epochs": int(config.warmup_epochs),
            "hybrid_ramp_epochs": int(config.hybrid_ramp_epochs),
            "selection_metric": config.selection_metric,
            "checkpoint_selector": config.checkpoint_selector,
            "selection_metric_source": "val_metric",
            "report_metric_source": "test_metric",
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


def _epoch_passes_validation_gate(row: dict[str, Any]) -> bool:
    return bool(
        float(row["val_transported_final_energy"]) < float(row["val_identity_final_energy"])
        and float(row["val_transported_final_energy"]) <= float(row["val_local_field_only_final_energy"])
        and float(row["val_accuracy"]) > float(row["val_baseline_accuracy"])
    )


def _snapshot_for_epoch(epoch_snapshots: list[FMPCTF1EpochSnapshot], epoch: int) -> FMPCTF1EpochSnapshot:
    for snapshot in epoch_snapshots:
        if int(snapshot.epoch) == int(epoch):
            return snapshot
    raise ValueError(f"No snapshot recorded for epoch {epoch}.")


def _select_tf1_checkpoint_epoch(
    epoch_metrics: list[dict[str, Any]],
    checkpoint_selector: TF1CheckpointSelector,
    *,
    selection_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not epoch_metrics:
        raise ValueError("epoch_metrics must contain at least one row.")
    diagnostics = selection_diagnostics or build_tf1_epoch_selection_diagnostics(epoch_metrics)
    gate_rows = [row for row in epoch_metrics if _epoch_passes_validation_gate(row)]
    energy_row = min(epoch_metrics, key=lambda row: float(row["val_transported_final_energy"]))
    val_accuracy_row = max(epoch_metrics, key=lambda row: float(row["val_accuracy"]))
    gate_best_row = None if not gate_rows else max(gate_rows, key=lambda row: float(row["val_accuracy"]))

    fallback_used = False
    selection_reason = ""
    if checkpoint_selector == "energy_only":
        selected_row = energy_row
        selection_reason = "selected lowest validation transported energy"
    elif checkpoint_selector == "val_accuracy_only":
        selected_row = val_accuracy_row
        selection_reason = "selected highest validation accuracy"
    elif checkpoint_selector == "gate_constrained_accuracy_then_energy":
        if gate_best_row is not None:
            selected_row = gate_best_row
            selection_reason = "selected highest validation accuracy among gate-passing epochs"
        else:
            selected_row = energy_row
            fallback_used = True
            selection_reason = "no gate-passing epoch; fell back to lowest validation transported energy"
    elif checkpoint_selector == "gate_constrained_accuracy_then_val_accuracy":
        if gate_best_row is not None:
            selected_row = gate_best_row
            selection_reason = "selected highest validation accuracy among gate-passing epochs"
        else:
            selected_row = val_accuracy_row
            fallback_used = True
            selection_reason = "no gate-passing epoch; fell back to highest validation accuracy"
    else:
        raise ValueError(f"Unsupported checkpoint_selector '{checkpoint_selector}'.")

    return {
        "selected_epoch": int(selected_row["epoch"]),
        "selected_epoch_passes_gate": bool(_epoch_passes_validation_gate(selected_row)),
        "gate_passing_epoch_count": int(diagnostics.get("gate_passing_epoch_count", len(gate_rows))),
        "selector_fallback_used": bool(fallback_used),
        "selected_epoch_selection_reason": str(selection_reason),
        "selected_epoch_val_accuracy": float(selected_row["val_accuracy"]),
        "selected_epoch_val_transported_final_energy": float(selected_row["val_transported_final_energy"]),
    }


def run_fmpc_tf1_experiment(config: FMPCTF1Config) -> FMPCTF1RunResult:
    """Run the minimal teacher-free FMPC v1 experiment on digits."""

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
    epoch_snapshots: list[FMPCTF1EpochSnapshot] = []

    train_start = perf_counter()
    for epoch_index in range(config.epochs):
        lambda_id = _lambda_id_for_epoch(config, epoch_index)
        stage = _stage_for_epoch(config, epoch_index)
        batch_losses: list[float] = []
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
            train_loss, boot_loss, identity_loss, transported_energy = _train_one_batch(
                model,
                psi_network,
                config,
                x_batch,
                y_batch,
                lambda_id=lambda_id,
                stage=stage,
            )
            batch_losses.append(train_loss)
            batch_boot_losses.append(boot_loss)
            batch_identity_losses.append(identity_loss)
            batch_transport_energies.append(transported_energy)

        val_transport = _evaluate_transport_split(
            model,
            psi_network,
            config,
            split.x_val,
            split.y_val,
        )
        _, val_accuracy = _evaluate_slow_pc_accuracy(model, split.x_val, split.y_val)
        val_baseline_accuracy = majority_class_baseline_accuracy(split.y_val)
        val_energy_delta_vs_identity = (
            val_transport.transported_final_energy - val_transport.identity_final_energy
        )
        val_energy_delta_vs_local_field_only = (
            val_transport.transported_final_energy - val_transport.local_field_only_final_energy
        )

        row = asdict(
            FMPCTF1EpochMetrics(
                epoch=epoch_index + 1,
                lambda_id=float(lambda_id),
                stage=stage,
                train_loss=float(np.mean(batch_losses)),
                train_boot_loss=float(np.mean(batch_boot_losses)),
                train_identity_loss=float(np.mean(batch_identity_losses)),
                train_transported_final_energy=float(np.mean(batch_transport_energies)),
                val_transported_final_energy=val_transport.transported_final_energy,
                val_identity_final_energy=val_transport.identity_final_energy,
                val_local_field_only_final_energy=val_transport.local_field_only_final_energy,
                val_energy_delta_vs_identity=float(val_energy_delta_vs_identity),
                val_energy_delta_vs_local_field_only=float(val_energy_delta_vs_local_field_only),
                val_accuracy=val_accuracy,
                val_baseline_accuracy=val_baseline_accuracy,
            )
        )
        epoch_rows.append(row)
        epoch_snapshots.append(
            FMPCTF1EpochSnapshot(
                epoch=epoch_index + 1,
                model_snapshot=_snapshot_pc_parameters(model),
                psi_snapshot=_snapshot_mlp_parameters(psi_network),
            )
        )

    train_wall_time_seconds = float(perf_counter() - train_start)

    selection_diagnostics = build_tf1_epoch_selection_diagnostics(epoch_rows)
    checkpoint_selection = _select_tf1_checkpoint_epoch(
        epoch_rows,
        config.checkpoint_selector,
        selection_diagnostics=selection_diagnostics,
    )
    selected_epoch = int(checkpoint_selection["selected_epoch"])
    selected_snapshot = _snapshot_for_epoch(epoch_snapshots, selected_epoch)

    _restore_pc_parameters(model, selected_snapshot.model_snapshot)
    _restore_mlp_parameters(psi_network, selected_snapshot.psi_snapshot)

    evaluation_start = perf_counter()
    val_transport = _evaluate_transport_split(model, psi_network, config, split.x_val, split.y_val)
    test_transport = _evaluate_transport_split(model, psi_network, config, split.x_test, split.y_test)
    val_loss, val_accuracy = _evaluate_slow_pc_accuracy(model, split.x_val, split.y_val)
    test_loss, test_accuracy = _evaluate_slow_pc_accuracy(model, split.x_test, split.y_test)
    evaluation_wall_time_seconds = float(perf_counter() - evaluation_start)
    val_baseline_accuracy = majority_class_baseline_accuracy(split.y_val)
    test_baseline_accuracy = majority_class_baseline_accuracy(split.y_test)
    val_energy_delta_vs_identity = val_transport.transported_final_energy - val_transport.identity_final_energy
    val_energy_delta_vs_local_field_only = (
        val_transport.transported_final_energy - val_transport.local_field_only_final_energy
    )
    test_energy_delta_vs_identity = (
        test_transport.transported_final_energy - test_transport.identity_final_energy
    )
    test_energy_delta_vs_local_field_only = (
        test_transport.transported_final_energy - test_transport.local_field_only_final_energy
    )

    summary = {
        "phase": "Phase TF1",
        "stage": "teacher_free_fmpc_v1",
        "preset_name": config.preset_name,
        "teacher_free": True,
        "uses_teacher_artifacts": False,
        "transport_scope": "train_only",
        "energy_substrate": "baseline_pc_energy",
        "local_flow_definition": "exact_negative_hidden_state_gradient",
        "direct_anchor_source": "self_bootstrap_local_field",
        "dataset_name": config.dataset_name,
        "model_variant": config.model_variant,
        "use_teacher_free_features": bool(config.use_teacher_free_features),
        "feature_aware_tangents": bool(config.feature_aware_tangents),
        "tangent_epsilon": float(config.tangent_epsilon),
        "bootstrap_integrator": config.bootstrap_integrator,
        "bootstrap_substeps": int(config.bootstrap_substeps),
        "warmup_epochs": int(config.warmup_epochs),
        "hybrid_ramp_epochs": int(config.hybrid_ramp_epochs),
        "transport_steps": int(config.transport_steps),
        "rollout_knots": val_transport.rollout_knots,
        "identity_loss_weight": float(config.identity_loss_weight),
        "selection_metric": config.selection_metric,
        "checkpoint_selector": config.checkpoint_selector,
        "selection_metric_source": "val_metric",
        "report_metric_source": "test_metric",
        "selection_metric_value": float(val_transport.transported_final_energy),
        "selection_metric_higher_is_better": False,
        "best_epoch": int(selected_epoch),
        "selected_epoch_passes_gate": bool(checkpoint_selection["selected_epoch_passes_gate"]),
        "gate_passing_epoch_count": int(checkpoint_selection["gate_passing_epoch_count"]),
        "selector_fallback_used": bool(checkpoint_selection["selector_fallback_used"]),
        "selected_epoch_selection_reason": str(checkpoint_selection["selected_epoch_selection_reason"]),
        "val_transported_final_energy": float(val_transport.transported_final_energy),
        "test_transported_final_energy": float(test_transport.transported_final_energy),
        "val_energy_delta_vs_identity": float(val_energy_delta_vs_identity),
        "test_energy_delta_vs_identity": float(test_energy_delta_vs_identity),
        "val_energy_delta_vs_local_field_only": float(val_energy_delta_vs_local_field_only),
        "test_energy_delta_vs_local_field_only": float(test_energy_delta_vs_local_field_only),
        "val_accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy),
        "val_loss": float(val_loss),
        "test_loss": float(test_loss),
        "val_baseline_accuracy": float(val_baseline_accuracy),
        "test_baseline_accuracy": float(test_baseline_accuracy),
        "selection_diagnostics_artifact": "selection_diagnostics.json",
        "identity_baseline": {
            "val_transported_final_energy": float(val_transport.identity_final_energy),
            "test_transported_final_energy": float(test_transport.identity_final_energy),
            "transport_steps": int(config.transport_steps),
            "rollout_knots": val_transport.rollout_knots,
            "energy_metric": "baseline_pc_energy",
        },
        "local_field_only_baseline": {
            "val_transported_final_energy": float(val_transport.local_field_only_final_energy),
            "test_transported_final_energy": float(test_transport.local_field_only_final_energy),
            "transport_steps": int(config.transport_steps),
            "rollout_knots": val_transport.rollout_knots,
            "energy_metric": "baseline_pc_energy",
        },
        "timing": {
            "train_wall_time_seconds": train_wall_time_seconds,
            "final_evaluation_wall_time_seconds": evaluation_wall_time_seconds,
        },
        "validation_gate": {
            "validation_only_gating": True,
            "test_is_report_only": True,
            "passes_identity_comparison": bool(
                val_transport.transported_final_energy < val_transport.identity_final_energy
            ),
            "passes_local_field_only_comparison": bool(
                val_transport.transported_final_energy <= val_transport.local_field_only_final_energy
            ),
            "passes_majority_baseline_accuracy": bool(val_accuracy > val_baseline_accuracy),
        },
    }

    _write_epoch_metrics(run_dir / "epoch_metrics.csv", epoch_rows)
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "selection_diagnostics.json", selection_diagnostics)
    return FMPCTF1RunResult(
        run_dir=run_dir,
        config=_config_payload(config),
        epoch_metrics=epoch_rows,
        summary=summary,
        model=model,
        psi_network=psi_network,
        epoch_snapshots=epoch_snapshots,
        selection_diagnostics=selection_diagnostics,
    )
