from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Literal

import numpy as np

from ..activations import get_activation
from ..datasets import load_digits_split
from ..energy import compute_cache, total_energy
from ..transport_core_v1.fmpc_tf1 import (
    TF1CheckpointSelector,
    _select_tf1_checkpoint_epoch,
    build_tf1_epoch_selection_diagnostics,
    build_tf1_identity_target,
)
from ..transport_core_v1.fmpc_tf1_flow import (
    FMPCTF1Context,
    FMPCTF1StateFeatureTangents,
    bootstrap_average_velocity_target,
    build_tf1_context,
    hidden_energy_from_state,
    hidden_local_flow,
    hidden_states_from_state,
    rollout_hidden_transport,
    teacher_free_feature_tangents,
    teacher_free_state_features,
)
from ..transport_core_v1.fmpc_tf1_jvp import (
    forward_tf1_mlp_with_jvp,
    resolve_tf1_identity_tangent_mode,
)
from ..interval_velocity.fmpc_meanflow_jvp import MeanFlowMLPJVPResult
from ..inference import build_clamped_mask, compute_state_gradients, initialize_states
from ..layers import init_mlp_layers
from ..metrics import classification_accuracy, majority_class_baseline_accuracy
from ..minibatch import iter_minibatches
from ..mlp_baseline import MLPNetwork, init_mlp_baseline_layers
from ..models import PCNetwork
from ..training import apply_parameter_updates, parameter_gradients
from ..utils import ensure_finite_array, set_seed

TF2FamilyLineage = Literal["tf1_mlp_aug"]
TF2PresetName = Literal[
    "tf2_canonical",
    "tf2_corrective_transport_default",
    "tf2_corrective_transport_terminal_angleclip_default",
]
TF2SupervisionPolicy = Literal["local_only", "mixed"]
TF2ThetaUpdateBudget = Literal["matched", "unmatched"]
TF2ThetaUpdateCadence = Literal["terminal_only", "every_2_micro_steps", "every_micro_step"]
TF2InterleavingStart = Literal["epoch_0", "after_warmup"]
TF2PsiFamily = Literal["baseline_plain", "residualized_local_field"]
TF2TimeEncodingVariant = Literal["raw", "poly_rt2"]
TF2TerminalLocalFieldDirectionIntervention = Literal[
    "none",
    "local_field_direction_angle_clip_keep_live_norm",
    "local_field_direction_smooth_unified_cone_projection_keep_live_norm",
    "local_field_direction_hard_replace_keep_live_norm",
    "local_field_direction_angle_clip_keep_live_norm_rowspace_only",
    "local_field_direction_hard_replace_keep_live_norm_rowspace_only",
    "local_field_direction_angle_clip_keep_live_norm_orthogonal_only",
    "local_field_direction_angle_clip_keep_live_norm_split_threshold",
]
TF2TransportedOutputAlignmentSchedule = Literal["none", "final_micro_step_only", "every_micro_step"]
OutputLayout = Literal["single_dir", "run_id_subdir"]


@dataclass
class FMPCTF2Config:
    """Configuration for the teacher-free TF2 iFMPC bridge experiment."""

    experiment_name: str = "fmpc_tf2"
    dataset_name: str = "digits"
    run_seed: int = 0
    data_seed: int = 0
    model_init_seed: int = 0
    psi_init_seed: int = 0
    batch_order_seed: int = 0
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: OutputLayout = "single_dir"
    preset_name: TF2PresetName | None = "tf2_canonical"
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    layer_dims: tuple[int, ...] = (64, 64, 10)
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
    family_lineage: TF2FamilyLineage = "tf1_mlp_aug"
    use_teacher_free_features: bool = True
    feature_aware_tangents: bool = False
    tangent_epsilon: float = 1e-3
    micro_steps: int = 4
    incremental_weight_updates: bool = True
    supervision_policy: TF2SupervisionPolicy = "mixed"
    theta_update_budget: TF2ThetaUpdateBudget = "matched"
    theta_update_cadence: TF2ThetaUpdateCadence | None = None
    onpolicy_mix_ratio: float | None = None
    interleaving_start: TF2InterleavingStart = "epoch_0"
    psi_family: TF2PsiFamily = "baseline_plain"
    psi_hidden_dims: tuple[int, ...] = (128,)
    time_encoding_variant: TF2TimeEncodingVariant = "raw"
    terminal_local_field_direction_intervention: TF2TerminalLocalFieldDirectionIntervention = "none"
    terminal_local_field_angle_clip_degrees: float = 30.0
    terminal_local_field_intervention_step_offsets: tuple[int, ...] = (-1,)
    terminal_local_field_rowspace_angle_clip_degrees: float = 30.0
    terminal_local_field_orthogonal_angle_clip_degrees: float = 30.0
    transported_output_alignment_weight: float = 0.0
    transported_output_alignment_schedule: TF2TransportedOutputAlignmentSchedule = "none"
    psi_weight_scale: float = 0.05
    psi_eta_w: float = 0.01
    psi_eta_b: float | None = 0.01
    bootstrap_integrator: Literal["euler", "rk2"] = "rk2"
    bootstrap_substeps: int = 4
    identity_loss_weight: float = 0.2
    warmup_epochs: int = 5
    hybrid_ramp_epochs: int = 10
    selection_metric: Literal["val_transported_final_energy"] = "val_transported_final_energy"
    checkpoint_selector: TF1CheckpointSelector = "gate_constrained_accuracy_then_val_accuracy"

    def __post_init__(self) -> None:
        if self.dataset_name != "digits":
            raise ValueError("Phase Incremental Bridge currently supports digits only.")
        if len(self.layer_dims) < 3:
            raise ValueError("TF2 expects at least one hidden layer and one output layer.")
        if self.family_lineage != "tf1_mlp_aug":
            raise ValueError("TF2A currently supports only the tf1_mlp_aug lineage.")
        if not self.use_teacher_free_features:
            raise ValueError("TF2A requires use_teacher_free_features=True.")
        if self.preset_name not in {
            None,
            "tf2_canonical",
            "tf2_corrective_transport_default",
            "tf2_corrective_transport_terminal_angleclip_default",
        }:
            raise ValueError(
                "preset_name must be None, 'tf2_canonical', 'tf2_corrective_transport_default', "
                "or 'tf2_corrective_transport_terminal_angleclip_default'."
            )
        if self.micro_steps <= 0:
            raise ValueError("micro_steps must be positive.")
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
        if self.supervision_policy not in {"local_only", "mixed"}:
            raise ValueError("Unsupported supervision_policy.")
        if self.theta_update_budget not in {"matched", "unmatched"}:
            raise ValueError("Unsupported theta_update_budget.")
        if self.theta_update_cadence is not None and self.theta_update_cadence not in {
            "terminal_only",
            "every_2_micro_steps",
            "every_micro_step",
        }:
            raise ValueError("Unsupported theta_update_cadence.")
        if self.onpolicy_mix_ratio is not None and not (0.0 <= float(self.onpolicy_mix_ratio) <= 0.5):
            raise ValueError("onpolicy_mix_ratio must lie in [0.0, 0.5].")
        if self.interleaving_start not in {"epoch_0", "after_warmup"}:
            raise ValueError("Unsupported interleaving_start.")
        if self.psi_family not in {"baseline_plain", "residualized_local_field"}:
            raise ValueError("Unsupported psi_family.")
        if self.time_encoding_variant not in {"raw", "poly_rt2"}:
            raise ValueError("Unsupported time_encoding_variant.")
        if self.terminal_local_field_direction_intervention not in {
            "none",
            "local_field_direction_angle_clip_keep_live_norm",
            "local_field_direction_smooth_unified_cone_projection_keep_live_norm",
            "local_field_direction_hard_replace_keep_live_norm",
            "local_field_direction_angle_clip_keep_live_norm_rowspace_only",
            "local_field_direction_hard_replace_keep_live_norm_rowspace_only",
            "local_field_direction_angle_clip_keep_live_norm_orthogonal_only",
            "local_field_direction_angle_clip_keep_live_norm_split_threshold",
        }:
            raise ValueError("Unsupported terminal_local_field_direction_intervention.")
        if not (0.0 < float(self.terminal_local_field_angle_clip_degrees) <= 180.0):
            raise ValueError("terminal_local_field_angle_clip_degrees must lie in (0.0, 180.0].")
        if len(self.terminal_local_field_intervention_step_offsets) == 0:
            raise ValueError("terminal_local_field_intervention_step_offsets must contain at least one step.")
        resolved_intervention_indices: set[int] = set()
        for raw_offset in self.terminal_local_field_intervention_step_offsets:
            step_offset = int(raw_offset)
            resolved_index = int(self.micro_steps) + step_offset if step_offset < 0 else step_offset
            if not (0 <= resolved_index < int(self.micro_steps)):
                raise ValueError(
                    "terminal_local_field_intervention_step_offsets must resolve to valid micro-step indices."
                )
            resolved_intervention_indices.add(resolved_index)
        if not (0.0 < float(self.terminal_local_field_rowspace_angle_clip_degrees) <= 180.0):
            raise ValueError("terminal_local_field_rowspace_angle_clip_degrees must lie in (0.0, 180.0].")
        if not (0.0 < float(self.terminal_local_field_orthogonal_angle_clip_degrees) <= 180.0):
            raise ValueError("terminal_local_field_orthogonal_angle_clip_degrees must lie in (0.0, 180.0].")
        if float(self.transported_output_alignment_weight) < 0.0:
            raise ValueError("transported_output_alignment_weight must be non-negative.")
        if self.transported_output_alignment_schedule not in {
            "none",
            "final_micro_step_only",
            "every_micro_step",
        }:
            raise ValueError("Unsupported transported_output_alignment_schedule.")
        if (
            self.terminal_local_field_direction_intervention != "none"
            and self.supervision_policy != "local_only"
        ):
            raise ValueError(
                "terminal_local_field_direction_intervention currently requires supervision_policy='local_only'."
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
class FMPCTF2EpochMetrics:
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
class FMPCTF2EpochSnapshot:
    epoch: int
    model_snapshot: list[tuple[np.ndarray, np.ndarray]]
    psi_snapshot: list[tuple[np.ndarray, np.ndarray]]


@dataclass
class FMPCTF2RunResult:
    run_dir: Path
    config: dict[str, Any]
    epoch_metrics: list[dict[str, Any]]
    summary: dict[str, Any]
    model: PCNetwork | None = None
    psi_network: MLPNetwork | None = None
    epoch_snapshots: list[FMPCTF2EpochSnapshot] | None = None
    selection_diagnostics: dict[str, Any] | None = None


@dataclass(frozen=True)
class TF2MicroStepPlan:
    """Frozen-snapshot TF2 micro-step plan."""

    psi_inputs: np.ndarray
    boot_targets: np.ndarray
    identity_targets: np.ndarray
    z_on_next: np.ndarray
    z_lf_next: np.ndarray
    source_counts: dict[str, int]


@dataclass(frozen=True)
class _SplitTransportMetrics:
    transported_final_energy: float
    identity_final_energy: float
    local_field_only_final_energy: float
    rollout_knots: list[float]


TF2SampleCollector = Callable[[dict[str, Any]], None]


def build_tf2_canonical_config(**overrides: Any) -> FMPCTF2Config:
    """Return the hypothesis-driven canonical TF2 iFMPC candidate.

    The current canonical default keeps `feature_aware_tangents=False`. The more
    complete feature-aware identity approximation remains available by explicit
    override, but it is not the default until it shows a clear matched empirical
    gain.
    """

    payload: dict[str, Any] = {
        "preset_name": "tf2_canonical",
        "layer_dims": (64, 64, 10),
        "family_lineage": "tf1_mlp_aug",
        "use_teacher_free_features": True,
        "feature_aware_tangents": False,
        "micro_steps": 4,
        "incremental_weight_updates": True,
        "supervision_policy": "mixed",
        "theta_update_budget": "matched",
        "identity_loss_weight": 0.2,
        "warmup_epochs": 5,
        "hybrid_ramp_epochs": 10,
        "bootstrap_substeps": 4,
        "checkpoint_selector": "gate_constrained_accuracy_then_val_accuracy",
        "epochs": 60,
        "eval_steps": 15,
    }
    payload.update(overrides)
    return FMPCTF2Config(**payload)


def build_tf2_corrective_transport_default_config(**overrides: Any) -> FMPCTF2Config:
    """Return the current empirical TF2 bridge winner.

    This preset is evidence-driven and provisional. It reflects the current TF2
    suite winner without changing the underlying TF2 algorithm family.
    """

    payload: dict[str, Any] = {
        "preset_name": "tf2_corrective_transport_default",
        "layer_dims": (64, 64, 10),
        "family_lineage": "tf1_mlp_aug",
        "use_teacher_free_features": True,
        "feature_aware_tangents": False,
        "micro_steps": 4,
        "incremental_weight_updates": False,
        "supervision_policy": "local_only",
        "theta_update_budget": "matched",
        "identity_loss_weight": 0.2,
        "warmup_epochs": 5,
        "hybrid_ramp_epochs": 10,
        "bootstrap_substeps": 4,
        "checkpoint_selector": "gate_constrained_accuracy_then_val_accuracy",
        "epochs": 60,
        "eval_steps": 15,
    }
    payload.update(overrides)
    return FMPCTF2Config(**payload)


def build_tf2_corrective_transport_terminal_angleclip_default_config(**overrides: Any) -> FMPCTF2Config:
    """Return the adopted TF2 package with terminal local-field angle clipping.

    This preset keeps the historical corrective transport working reference intact
    while exposing the confirmed package:

    - `psi_family = "residualized_local_field"`
    - `time_encoding_variant = "poly_rt2"`
    - terminal local-field direction angle clip at 30 degrees on the final
      training micro-step only
    """

    payload: dict[str, Any] = {
        "preset_name": "tf2_corrective_transport_terminal_angleclip_default",
        "layer_dims": (64, 64, 10),
        "family_lineage": "tf1_mlp_aug",
        "use_teacher_free_features": True,
        "feature_aware_tangents": False,
        "micro_steps": 4,
        "incremental_weight_updates": False,
        "supervision_policy": "local_only",
        "theta_update_budget": "matched",
        "identity_loss_weight": 0.2,
        "warmup_epochs": 5,
        "hybrid_ramp_epochs": 10,
        "bootstrap_substeps": 4,
        "checkpoint_selector": "gate_constrained_accuracy_then_val_accuracy",
        "epochs": 60,
        "eval_steps": 15,
        "psi_family": "residualized_local_field",
        "time_encoding_variant": "poly_rt2",
        "terminal_local_field_direction_intervention": "local_field_direction_angle_clip_keep_live_norm",
        "terminal_local_field_angle_clip_degrees": 30.0,
    }
    payload.update(overrides)
    return FMPCTF2Config(**payload)


def build_tf2_preset_config(
    preset_name: TF2PresetName,
    **overrides: Any,
) -> FMPCTF2Config:
    """Build a TF2 config from one of the named TF2 presets."""

    if preset_name == "tf2_canonical":
        return build_tf2_canonical_config(**overrides)
    if preset_name == "tf2_corrective_transport_default":
        return build_tf2_corrective_transport_default_config(**overrides)
    if preset_name == "tf2_corrective_transport_terminal_angleclip_default":
        return build_tf2_corrective_transport_terminal_angleclip_default_config(**overrides)
    raise ValueError(f"Unsupported TF2 preset '{preset_name}'.")


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


def _resolved_eta_b(config: FMPCTF2Config) -> float:
    return float(config.eta_b if config.eta_b is not None else config.eta_w)


def _theta_update_count_for_cadence(
    micro_steps: int,
    cadence: TF2ThetaUpdateCadence,
) -> int:
    if cadence == "terminal_only":
        return 1
    return int(sum(_theta_update_due_for_step(cadence, step_index) for step_index in range(micro_steps)))


def _theta_micro_learning_rates(
    config: FMPCTF2Config,
    cadence: TF2ThetaUpdateCadence | None = None,
) -> tuple[float, float]:
    base_eta_b = _resolved_eta_b(config)
    resolved_cadence = _resolved_theta_update_cadence(config) if cadence is None else cadence
    if config.theta_update_budget == "matched":
        update_count = _theta_update_count_for_cadence(int(config.micro_steps), resolved_cadence)
        scale = float(update_count) if update_count > 0 else 1.0
        return float(config.eta_w / scale), float(base_eta_b / scale)
    return float(config.eta_w), float(base_eta_b)


def _resolved_theta_update_cadence(config: FMPCTF2Config) -> TF2ThetaUpdateCadence:
    if config.theta_update_cadence is not None:
        return config.theta_update_cadence
    return "every_micro_step" if config.incremental_weight_updates else "terminal_only"


def _resolved_onpolicy_mix_ratio(config: FMPCTF2Config) -> float:
    if config.onpolicy_mix_ratio is not None:
        return float(config.onpolicy_mix_ratio)
    return 0.0 if config.supervision_policy == "local_only" else 0.5


def _identity_tangent_mode(config: FMPCTF2Config) -> str:
    return resolve_tf1_identity_tangent_mode(
        use_teacher_free_features=config.use_teacher_free_features,
        feature_aware_tangents=config.feature_aware_tangents,
    )


def _active_theta_update_cadence(
    config: FMPCTF2Config,
    epoch_index: int | None,
) -> TF2ThetaUpdateCadence:
    cadence = _resolved_theta_update_cadence(config)
    if config.interleaving_start == "after_warmup" and epoch_index is not None and epoch_index < config.warmup_epochs:
        return "terminal_only"
    return cadence


def _active_onpolicy_mix_ratio(
    config: FMPCTF2Config,
    epoch_index: int | None,
) -> float:
    mix_ratio = _resolved_onpolicy_mix_ratio(config)
    if config.interleaving_start == "after_warmup" and epoch_index is not None and epoch_index < config.warmup_epochs:
        return 0.0
    return mix_ratio


def _theta_update_due_for_step(
    cadence: TF2ThetaUpdateCadence,
    step_index: int,
) -> bool:
    if cadence == "terminal_only":
        return False
    if cadence == "every_micro_step":
        return True
    if cadence == "every_2_micro_steps":
        return (step_index + 1) % 2 == 0
    raise ValueError(f"Unsupported theta update cadence '{cadence}'.")


def _onpolicy_example_count(batch_size: int, onpolicy_mix_ratio: float) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    ratio = float(onpolicy_mix_ratio)
    if ratio <= 0.0:
        return 0
    if ratio >= 0.5:
        return int(batch_size)
    raw_count = ratio * float(batch_size) / max(1e-12, 1.0 - ratio)
    count = int(np.round(raw_count))
    return max(1, min(int(batch_size), count))


def _take_evenly_spaced_rows(array: np.ndarray, count: int) -> np.ndarray:
    batch = np.asarray(array, dtype=np.float64)
    if count <= 0:
        return batch[:0].copy()
    if count >= batch.shape[0]:
        return batch.copy()
    indices = np.linspace(0, batch.shape[0] - 1, num=count, dtype=int)
    return batch[indices].copy()


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
        ensure_finite_array(pre_activation, f"tf2_pre_activation[{layer_index}]")
        ensure_finite_array(current, f"tf2_activation[{layer_index}]")
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
        ensure_finite_array(layer.weight, f"tf2_weight[{layer_index + 1}]")
        ensure_finite_array(layer.bias, f"tf2_bias[{layer_index + 1}]")
        if next_delta is not None:
            delta = next_delta


def _hidden_dim(config: FMPCTF2Config) -> int:
    return int(sum(config.layer_dims[1:-1]))


def _target_dim(config: FMPCTF2Config) -> int:
    return int(config.layer_dims[-1])


def _time_encoding_dim(variant: TF2TimeEncodingVariant) -> int:
    if variant == "raw":
        return 2
    if variant == "poly_rt2":
        return 5
    raise ValueError(f"Unsupported time_encoding_variant '{variant}'.")


def _teacher_free_feature_dim(config: FMPCTF2Config) -> int:
    if not config.use_teacher_free_features:
        return 0
    return _hidden_dim(config) + _target_dim(config) + 1


def _build_time_encoding(
    batch_size: int,
    *,
    t: float,
    r: float,
    variant: TF2TimeEncodingVariant,
) -> np.ndarray:
    t_block = np.full((batch_size, 1), float(t), dtype=np.float64)
    r_block = np.full((batch_size, 1), float(r), dtype=np.float64)
    if variant == "raw":
        return np.concatenate([t_block, r_block], axis=1).astype(np.float64, copy=False)
    if variant == "poly_rt2":
        return np.concatenate(
            [
                t_block,
                r_block,
                t_block * r_block,
                t_block * t_block,
                r_block * r_block,
            ],
            axis=1,
        ).astype(np.float64, copy=False)
    raise ValueError(f"Unsupported time_encoding_variant '{variant}'.")


def _build_time_encoding_tangent(
    batch_size: int,
    *,
    d_t: float,
    d_r: float,
    t: float,
    r: float,
    variant: TF2TimeEncodingVariant,
) -> np.ndarray:
    t_tangent = np.full((batch_size, 1), float(d_t), dtype=np.float64)
    r_tangent = np.full((batch_size, 1), float(d_r), dtype=np.float64)
    if variant == "raw":
        return np.concatenate([t_tangent, r_tangent], axis=1).astype(np.float64, copy=False)
    if variant == "poly_rt2":
        tr_tangent = np.full((batch_size, 1), float(r * d_t + t * d_r), dtype=np.float64)
        t2_tangent = np.full((batch_size, 1), float(2.0 * t * d_t), dtype=np.float64)
        r2_tangent = np.full((batch_size, 1), float(2.0 * r * d_r), dtype=np.float64)
        return np.concatenate(
            [t_tangent, r_tangent, tr_tangent, t2_tangent, r2_tangent],
            axis=1,
        ).astype(np.float64, copy=False)
    raise ValueError(f"Unsupported time_encoding_variant '{variant}'.")


def _build_psi_input(
    config: FMPCTF2Config,
    z_t: np.ndarray,
    target_onehot: np.ndarray,
    *,
    t: float,
    r: float,
    features: Any | None = None,
) -> np.ndarray:
    z_array = np.asarray(z_t, dtype=np.float64)
    target_array = np.asarray(target_onehot, dtype=np.float64)
    if z_array.ndim != 2 or target_array.ndim != 2:
        raise ValueError("z_t and target_onehot must be shaped (batch, features).")
    if z_array.shape[0] != target_array.shape[0]:
        raise ValueError("z_t and target_onehot must share the same batch size.")
    batch_size = int(z_array.shape[0])
    time_block = _build_time_encoding(
        batch_size,
        t=float(t),
        r=float(r),
        variant=config.time_encoding_variant,
    )
    blocks = [z_array, target_array, time_block]
    if config.use_teacher_free_features:
        if features is None:
            raise ValueError("features must be provided when use_teacher_free_features=True.")
        blocks.extend([features.g_t, features.e_out_t, features.F_t])
    return np.concatenate(blocks, axis=1).astype(np.float64, copy=False)


def _build_psi_input_tangent(
    config: FMPCTF2Config,
    g_t: np.ndarray,
    *,
    target_dim: int,
    t: float,
    r: float,
    feature_tangents: FMPCTF1StateFeatureTangents | None = None,
    d_t: float = 1.0,
    d_r: float = -1.0,
) -> np.ndarray:
    g_array = np.asarray(g_t, dtype=np.float64)
    if g_array.ndim != 2:
        raise ValueError("g_t must be shaped (batch, hidden_dim).")
    batch_size = int(g_array.shape[0])
    target_tangent = np.zeros((batch_size, int(target_dim)), dtype=np.float64)
    time_tangent = _build_time_encoding_tangent(
        batch_size,
        d_t=float(d_t),
        d_r=float(d_r),
        t=float(t),
        r=float(r),
        variant=config.time_encoding_variant,
    )
    blocks = [g_array, target_tangent, time_tangent]
    if config.use_teacher_free_features:
        if config.feature_aware_tangents:
            if feature_tangents is None:
                raise ValueError(
                    "feature_tangents must be provided when feature_aware_tangents=True."
                )
            feature_tangent_block = np.concatenate(
                [
                    feature_tangents.Dg_g_t,
                    feature_tangents.Dg_e_out_t,
                    feature_tangents.Dg_F_t,
                ],
                axis=1,
            ).astype(np.float64, copy=False)
        else:
            feature_dim = _teacher_free_feature_dim(config)
            feature_tangent_block = np.zeros((batch_size, feature_dim), dtype=np.float64)
        blocks.append(feature_tangent_block)
    return np.concatenate(blocks, axis=1).astype(np.float64, copy=False)


def _extract_detached_local_flow_anchor(
    inputs: np.ndarray,
    config: FMPCTF2Config,
) -> np.ndarray:
    if not config.use_teacher_free_features:
        raise ValueError("residualized_local_field requires use_teacher_free_features=True.")
    input_array = np.asarray(inputs, dtype=np.float64)
    if input_array.ndim != 2:
        raise ValueError("inputs must be shaped (batch, features).")
    feature_offset = _hidden_dim(config) + _target_dim(config) + _time_encoding_dim(config.time_encoding_variant)
    anchor = input_array[:, feature_offset : feature_offset + _hidden_dim(config)]
    if anchor.shape[1] != _hidden_dim(config):
        raise ValueError("Failed to extract detached local-flow anchor from psi inputs.")
    return anchor.astype(np.float64, copy=False)


def _action_from_step(z_k: np.ndarray, z_next: np.ndarray, dt: float) -> np.ndarray:
    return (np.asarray(z_next, dtype=np.float64) - np.asarray(z_k, dtype=np.float64)) / float(dt)


def _vector_norms(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(x, dtype=np.float64), axis=1)


def _safe_direction(x: np.ndarray) -> np.ndarray:
    array = np.asarray(x, dtype=np.float64)
    norms = _vector_norms(array)[:, None]
    return np.divide(array, np.maximum(norms, 1e-12), out=np.zeros_like(array), where=norms > 1e-12)


def _clip_direction_to_anchor_cone(
    raw_direction: np.ndarray,
    anchor_direction: np.ndarray,
    *,
    max_angle_degrees: float,
) -> np.ndarray:
    raw = np.asarray(raw_direction, dtype=np.float64)
    anchor = np.asarray(anchor_direction, dtype=np.float64)
    if raw.shape != anchor.shape:
        raise ValueError("raw_direction and anchor_direction must share the same shape.")
    clipped = np.zeros_like(raw)
    cos_threshold = float(np.cos(np.deg2rad(float(max_angle_degrees))))
    sin_threshold = float(np.sin(np.deg2rad(float(max_angle_degrees))))
    for row_index in range(int(raw.shape[0])):
        raw_row = raw[row_index]
        anchor_row = anchor[row_index]
        raw_norm = float(np.linalg.norm(raw_row))
        anchor_norm = float(np.linalg.norm(anchor_row))
        if raw_norm <= 1e-12 or anchor_norm <= 1e-12:
            clipped[row_index] = raw_row
            continue
        raw_unit = raw_row / raw_norm
        anchor_unit = anchor_row / anchor_norm
        cosine = float(np.clip(np.dot(raw_unit, anchor_unit), -1.0, 1.0))
        if cosine >= cos_threshold:
            clipped[row_index] = raw_unit
            continue
        tangent = raw_unit - (cosine * anchor_unit)
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1e-12:
            clipped[row_index] = anchor_unit
            continue
        tangent_unit = tangent / tangent_norm
        clipped[row_index] = (cos_threshold * anchor_unit) + (sin_threshold * tangent_unit)
    return _safe_direction(clipped)


def _smooth_project_direction_to_anchor_cone(
    raw_direction: np.ndarray,
    anchor_direction: np.ndarray,
    *,
    max_angle_degrees: float,
) -> np.ndarray:
    raw = np.asarray(raw_direction, dtype=np.float64)
    anchor = np.asarray(anchor_direction, dtype=np.float64)
    if raw.shape != anchor.shape:
        raise ValueError("raw_direction and anchor_direction must share the same shape.")
    projected = np.zeros_like(raw)
    theta_max = float(np.deg2rad(float(max_angle_degrees)))
    residual_span = max(float(np.pi - theta_max), 1e-12)
    for row_index in range(int(raw.shape[0])):
        raw_row = raw[row_index]
        anchor_row = anchor[row_index]
        raw_norm = float(np.linalg.norm(raw_row))
        anchor_norm = float(np.linalg.norm(anchor_row))
        if raw_norm <= 1e-12 or anchor_norm <= 1e-12:
            projected[row_index] = raw_row
            continue
        raw_unit = raw_row / raw_norm
        anchor_unit = anchor_row / anchor_norm
        cosine = float(np.clip(np.dot(raw_unit, anchor_unit), -1.0, 1.0))
        theta = float(np.arccos(cosine))
        if theta <= theta_max:
            projected[row_index] = raw_unit
            continue
        tangent = raw_unit - (cosine * anchor_unit)
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1e-12:
            projected[row_index] = anchor_unit
            continue
        tangent_unit = tangent / tangent_norm
        excess_fraction = float(np.clip((theta - theta_max) / residual_span, 0.0, 1.0))
        smooth_step = float(excess_fraction * excess_fraction * (3.0 - (2.0 * excess_fraction)))
        target_theta = float(theta_max * (1.0 - smooth_step))
        projected[row_index] = (np.cos(target_theta) * anchor_unit) + (np.sin(target_theta) * tangent_unit)
    return _safe_direction(projected)


def _free_hidden_state_indices_from_context(context: FMPCTF1Context) -> list[int]:
    if len(context.clamped_mask) <= 2:
        return []
    return [
        layer_index
        for layer_index in range(1, len(context.clamped_mask) - 1)
        if not context.clamped_mask[layer_index]
    ]


def _final_hidden_block_slice(context: FMPCTF1Context) -> slice:
    hidden_indices = _free_hidden_state_indices_from_context(context)
    if not hidden_indices:
        raise ValueError("TF2 terminal interventions require at least one free hidden layer.")
    offset = 0
    final_hidden_index = hidden_indices[-1]
    for layer_index in hidden_indices:
        width = int(np.asarray(context.states_template[layer_index], dtype=np.float64).shape[1])
        if layer_index == final_hidden_index:
            return slice(offset, offset + width)
        offset += width
    raise ValueError("Failed to resolve final hidden block slice.")


def _rowspace_basis_from_output_weight(weight: np.ndarray) -> np.ndarray:
    weight_array = np.asarray(weight, dtype=np.float64)
    _, singular_values, vh = np.linalg.svd(weight_array, full_matrices=False)
    rank = int(np.sum(singular_values > 1e-12))
    if rank == 0:
        return np.zeros((weight_array.shape[1], 0), dtype=np.float64)
    return vh[:rank].T.copy()


def _project_onto_rowspace(vectors: np.ndarray, basis: np.ndarray) -> np.ndarray:
    vectors_array = np.asarray(vectors, dtype=np.float64)
    if basis.shape[1] == 0:
        return np.zeros_like(vectors_array)
    return (vectors_array @ basis) @ basis.T


def _project_onto_orthogonal_complement(vectors: np.ndarray, basis: np.ndarray) -> np.ndarray:
    vectors_array = np.asarray(vectors, dtype=np.float64)
    return vectors_array - _project_onto_rowspace(vectors_array, basis)


def _apply_terminal_rowspace_only_direction_intervention(
    raw_action: np.ndarray,
    local_field_action: np.ndarray,
    output_weight: np.ndarray,
    context: FMPCTF1Context,
    config: FMPCTF2Config,
    *,
    hard_replace: bool,
) -> np.ndarray:
    final_hidden_slice = _final_hidden_block_slice(context)
    basis = _rowspace_basis_from_output_weight(output_weight)
    stabilized_action = np.asarray(raw_action, dtype=np.float64).copy()
    raw_final_hidden = stabilized_action[:, final_hidden_slice]
    anchor_final_hidden = np.asarray(local_field_action, dtype=np.float64)[:, final_hidden_slice]
    raw_row = _project_onto_rowspace(raw_final_hidden, basis)
    raw_orth = _project_onto_orthogonal_complement(raw_final_hidden, basis)
    anchor_row = _project_onto_rowspace(anchor_final_hidden, basis)
    row_norm = _vector_norms(raw_row)[:, None]
    if hard_replace:
        stabilized_row = _safe_direction(anchor_row) * row_norm
    else:
        stabilized_row = _clip_direction_to_anchor_cone(
            _safe_direction(raw_row),
            _safe_direction(anchor_row),
            max_angle_degrees=float(config.terminal_local_field_angle_clip_degrees),
        ) * row_norm
    stabilized_action[:, final_hidden_slice] = stabilized_row + raw_orth
    return stabilized_action


def _apply_terminal_orthogonal_only_direction_intervention(
    raw_action: np.ndarray,
    local_field_action: np.ndarray,
    output_weight: np.ndarray,
    context: FMPCTF1Context,
    config: FMPCTF2Config,
) -> np.ndarray:
    final_hidden_slice = _final_hidden_block_slice(context)
    basis = _rowspace_basis_from_output_weight(output_weight)
    stabilized_action = np.asarray(raw_action, dtype=np.float64).copy()
    raw_final_hidden = stabilized_action[:, final_hidden_slice]
    anchor_final_hidden = np.asarray(local_field_action, dtype=np.float64)[:, final_hidden_slice]
    raw_row = _project_onto_rowspace(raw_final_hidden, basis)
    raw_orth = _project_onto_orthogonal_complement(raw_final_hidden, basis)
    anchor_orth = _project_onto_orthogonal_complement(anchor_final_hidden, basis)
    orth_norm = _vector_norms(raw_orth)[:, None]
    stabilized_orth = _clip_direction_to_anchor_cone(
        _safe_direction(raw_orth),
        _safe_direction(anchor_orth),
        max_angle_degrees=float(config.terminal_local_field_angle_clip_degrees),
    ) * orth_norm
    stabilized_action[:, final_hidden_slice] = raw_row + stabilized_orth
    return stabilized_action


def _apply_terminal_split_threshold_direction_intervention(
    raw_action: np.ndarray,
    local_field_action: np.ndarray,
    output_weight: np.ndarray,
    context: FMPCTF1Context,
    config: FMPCTF2Config,
) -> np.ndarray:
    final_hidden_slice = _final_hidden_block_slice(context)
    basis = _rowspace_basis_from_output_weight(output_weight)
    stabilized_action = np.asarray(raw_action, dtype=np.float64).copy()
    raw_final_hidden = stabilized_action[:, final_hidden_slice]
    anchor_final_hidden = np.asarray(local_field_action, dtype=np.float64)[:, final_hidden_slice]
    raw_row = _project_onto_rowspace(raw_final_hidden, basis)
    raw_orth = _project_onto_orthogonal_complement(raw_final_hidden, basis)
    anchor_row = _project_onto_rowspace(anchor_final_hidden, basis)
    anchor_orth = _project_onto_orthogonal_complement(anchor_final_hidden, basis)
    row_norm = _vector_norms(raw_row)[:, None]
    orth_norm = _vector_norms(raw_orth)[:, None]
    stabilized_row = _clip_direction_to_anchor_cone(
        _safe_direction(raw_row),
        _safe_direction(anchor_row),
        max_angle_degrees=float(config.terminal_local_field_rowspace_angle_clip_degrees),
    ) * row_norm
    stabilized_orth = _clip_direction_to_anchor_cone(
        _safe_direction(raw_orth),
        _safe_direction(anchor_orth),
        max_angle_degrees=float(config.terminal_local_field_orthogonal_angle_clip_degrees),
    ) * orth_norm
    stabilized_action[:, final_hidden_slice] = stabilized_row + stabilized_orth
    return stabilized_action


def _apply_terminal_local_field_direction_intervention(
    z_on_k: np.ndarray,
    dt: float,
    plan: TF2MicroStepPlan,
    config: FMPCTF2Config,
    *,
    context: FMPCTF1Context,
    output_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    intervention = config.terminal_local_field_direction_intervention
    raw_action = _action_from_step(z_on_k, plan.z_on_next, dt)
    if intervention == "none":
        return raw_action, plan.z_on_next.copy()
    local_field_action = _extract_detached_local_flow_anchor(plan.psi_inputs, config)
    local_field_direction = _safe_direction(local_field_action)
    live_norm = _vector_norms(raw_action)[:, None]
    if intervention == "local_field_direction_hard_replace_keep_live_norm":
        stabilized_action = local_field_direction * live_norm
    elif intervention == "local_field_direction_angle_clip_keep_live_norm":
        stabilized_action = _clip_direction_to_anchor_cone(
            _safe_direction(raw_action),
            local_field_direction,
            max_angle_degrees=float(config.terminal_local_field_angle_clip_degrees),
        ) * live_norm
    elif intervention == "local_field_direction_smooth_unified_cone_projection_keep_live_norm":
        # Keep the same full-space local-field cone family as the adopted hard
        # clip, but use a smooth interior projection for out-of-cone actions.
        stabilized_action = _smooth_project_direction_to_anchor_cone(
            _safe_direction(raw_action),
            local_field_direction,
            max_angle_degrees=float(config.terminal_local_field_angle_clip_degrees),
        ) * live_norm
    elif intervention == "local_field_direction_hard_replace_keep_live_norm_rowspace_only":
        stabilized_action = _apply_terminal_rowspace_only_direction_intervention(
            raw_action,
            local_field_action,
            output_weight,
            context,
            config,
            hard_replace=True,
        )
    elif intervention == "local_field_direction_angle_clip_keep_live_norm_rowspace_only":
        stabilized_action = _apply_terminal_rowspace_only_direction_intervention(
            raw_action,
            local_field_action,
            output_weight,
            context,
            config,
            hard_replace=False,
        )
    elif intervention == "local_field_direction_angle_clip_keep_live_norm_orthogonal_only":
        stabilized_action = _apply_terminal_orthogonal_only_direction_intervention(
            raw_action,
            local_field_action,
            output_weight,
            context,
            config,
        )
    elif intervention == "local_field_direction_angle_clip_keep_live_norm_split_threshold":
        stabilized_action = _apply_terminal_split_threshold_direction_intervention(
            raw_action,
            local_field_action,
            output_weight,
            context,
            config,
        )
    else:
        raise ValueError(f"Unsupported terminal_local_field_direction_intervention '{intervention}'.")
    stabilized_next = np.asarray(z_on_k, dtype=np.float64) + float(dt) * stabilized_action
    ensure_finite_array(stabilized_action, "tf2_terminal_stabilized_action")
    ensure_finite_array(stabilized_next, "tf2_terminal_stabilized_z_on_next")
    return stabilized_action, stabilized_next


def _psi_predict(
    psi_network: MLPNetwork,
    inputs: np.ndarray,
    config: FMPCTF2Config,
) -> np.ndarray:
    base_output = psi_network.predict(inputs)
    if config.psi_family == "baseline_plain":
        return base_output
    if config.psi_family == "residualized_local_field":
        return _extract_detached_local_flow_anchor(inputs, config) + base_output
    raise ValueError(f"Unsupported psi_family '{config.psi_family}'.")


def _psi_forward_with_jvp(
    psi_network: MLPNetwork,
    inputs: np.ndarray,
    input_tangent: np.ndarray,
    config: FMPCTF2Config,
) -> MeanFlowMLPJVPResult:
    base_result = forward_tf1_mlp_with_jvp(psi_network, inputs, input_tangent)
    if config.psi_family == "baseline_plain":
        return base_result
    if config.psi_family == "residualized_local_field":
        anchor = _extract_detached_local_flow_anchor(inputs, config)
        return MeanFlowMLPJVPResult(output=anchor + base_result.output, jvp=base_result.jvp)
    raise ValueError(f"Unsupported psi_family '{config.psi_family}'.")


def _psi_input_dim(config: FMPCTF2Config) -> int:
    return _hidden_dim(config) + _target_dim(config) + _time_encoding_dim(config.time_encoding_variant) + _teacher_free_feature_dim(config)


def _make_pc_model(config: FMPCTF2Config) -> PCNetwork:
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


def _make_psi_network(config: FMPCTF2Config) -> MLPNetwork:
    hidden_dim = _hidden_dim(config)
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


def _lambda_id_for_epoch(config: FMPCTF2Config, epoch_index: int) -> float:
    if epoch_index < config.warmup_epochs:
        return 0.0
    if config.hybrid_ramp_epochs <= 0:
        return float(config.identity_loss_weight)
    progress = (epoch_index - config.warmup_epochs + 1) / float(config.hybrid_ramp_epochs)
    return float(config.identity_loss_weight) * float(np.clip(progress, 0.0, 1.0))


def _stage_for_epoch(config: FMPCTF2Config, epoch_index: int) -> str:
    return "warmup" if epoch_index < config.warmup_epochs else "hybrid"


def _rms(array: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(array, dtype=np.float64) ** 2)))


def _mean_l2_norm(array: np.ndarray) -> float:
    batch = np.asarray(array, dtype=np.float64)
    return float(np.mean(np.linalg.norm(batch, axis=1)))


def _forward_init_stability_metrics(
    model: PCNetwork,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
) -> dict[str, Any]:
    states = initialize_states(model.layers, x_batch, y=y_batch, init=model.state_init, mode="train")
    clamped_mask = build_clamped_mask(len(model.layers) + 1, mode="train")
    cache = compute_cache(states, model.layers)
    gradients = compute_state_gradients(states, cache, model.layers, clamped_mask)
    hidden_states = [np.asarray(state, dtype=np.float64) for state in states[1:-1]]
    hidden_gradients = [
        np.asarray(gradient, dtype=np.float64)
        for gradient in gradients[1:-1]
        if gradient is not None
    ]
    if hidden_gradients:
        gradient_block = np.concatenate(hidden_gradients, axis=1)
        gradient_rms = _rms(gradient_block)
        gradient_mean_l2 = _mean_l2_norm(gradient_block)
    else:
        gradient_rms = 0.0
        gradient_mean_l2 = 0.0
    hidden_layer_stats = [
        {
            "layer_index": int(layer_index),
            "feature_dim": int(hidden_state.shape[1]),
            "rms": _rms(hidden_state),
            "mean_l2_norm": _mean_l2_norm(hidden_state),
        }
        for layer_index, hidden_state in enumerate(hidden_states, start=1)
    ]
    return {
        "hidden_layer_stats": hidden_layer_stats,
        "initial_target_clamped_energy": float(total_energy(cache, model.layers, int(x_batch.shape[0]))),
        "initial_hidden_gradient_rms": float(gradient_rms),
        "initial_hidden_gradient_mean_l2_norm": float(gradient_mean_l2),
    }


def _feature_tangents_for_state(
    context: FMPCTF1Context,
    z_t: np.ndarray,
    config: FMPCTF2Config,
) -> FMPCTF1StateFeatureTangents | None:
    if not config.feature_aware_tangents:
        return None
    return teacher_free_feature_tangents(
        context,
        z_t,
        epsilon=config.tangent_epsilon,
    )


def _single_source_supervision(
    context: FMPCTF1Context,
    psi_network: MLPNetwork,
    config: FMPCTF2Config,
    z_t: np.ndarray,
    *,
    t_k: float,
    r_k: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = teacher_free_state_features(context, z_t)
    feature_tangents = _feature_tangents_for_state(context, z_t, config)
    inputs = _build_psi_input(
        config,
        z_t,
        context.targets,
        t=t_k,
        r=r_k,
        features=features,
    )
    input_tangent = _build_psi_input_tangent(
        config,
        features.g_t,
        target_dim=context.target_dim,
        t=t_k,
        r=r_k,
        feature_tangents=feature_tangents,
    )
    jvp_result = _psi_forward_with_jvp(psi_network, inputs, input_tangent, config)
    u_boot = bootstrap_average_velocity_target(
        context,
        z_t,
        t=t_k,
        r=r_k,
        integrator=config.bootstrap_integrator,
        substeps=config.bootstrap_substeps,
    )
    u_identity = build_tf1_identity_target(features.g_t, r_k, jvp_result.jvp)
    return inputs, u_boot, u_identity, jvp_result.output


def _plan_tf2_micro_step(
    context: FMPCTF1Context,
    psi_network: MLPNetwork,
    config: FMPCTF2Config,
    z_on_k: np.ndarray,
    z_lf_k: np.ndarray,
    *,
    t_k: float,
    dt: float,
    r_k: float,
    onpolicy_mix_ratio: float | None = None,
) -> TF2MicroStepPlan:
    lf_inputs, lf_boot, lf_identity, _ = _single_source_supervision(
        context,
        psi_network,
        config,
        z_lf_k,
        t_k=t_k,
        r_k=r_k,
    )
    on_inputs, on_boot, on_identity, on_velocity = _single_source_supervision(
        context,
        psi_network,
        config,
        z_on_k,
        t_k=t_k,
        r_k=r_k,
    )
    z_on_next = np.asarray(z_on_k, dtype=np.float64) + float(dt) * np.asarray(on_velocity, dtype=np.float64)
    z_lf_next = np.asarray(z_lf_k, dtype=np.float64) + float(dt) * hidden_local_flow(context, z_lf_k)
    ensure_finite_array(z_on_next, "tf2_z_on_next")
    ensure_finite_array(z_lf_next, "tf2_z_lf_next")

    effective_mix_ratio = (
        _resolved_onpolicy_mix_ratio(config)
        if onpolicy_mix_ratio is None
        else float(onpolicy_mix_ratio)
    )
    if effective_mix_ratio <= 0.0:
        psi_inputs = lf_inputs
        boot_targets = lf_boot
        identity_targets = lf_identity
        source_counts = {"local_field_only": int(lf_inputs.shape[0]), "learned_on_policy": 0}
    else:
        on_count = _onpolicy_example_count(int(lf_inputs.shape[0]), effective_mix_ratio)
        on_inputs_selected = _take_evenly_spaced_rows(on_inputs, on_count)
        on_boot_selected = _take_evenly_spaced_rows(on_boot, on_count)
        on_identity_selected = _take_evenly_spaced_rows(on_identity, on_count)
        psi_inputs = np.concatenate([lf_inputs, on_inputs_selected], axis=0).astype(np.float64, copy=False)
        boot_targets = np.concatenate([lf_boot, on_boot_selected], axis=0).astype(np.float64, copy=False)
        identity_targets = np.concatenate([lf_identity, on_identity_selected], axis=0).astype(np.float64, copy=False)
        source_counts = {
            "local_field_only": int(lf_inputs.shape[0]),
            "learned_on_policy": int(on_count),
        }
    return TF2MicroStepPlan(
        psi_inputs=psi_inputs,
        boot_targets=boot_targets,
        identity_targets=identity_targets,
        z_on_next=z_on_next,
        z_lf_next=z_lf_next,
        source_counts=source_counts,
    )


def _theta_update_from_transported_state(
    model: PCNetwork,
    context: FMPCTF1Context,
    transported_z: np.ndarray,
    *,
    eta_w: float,
    eta_b: float,
    output_alignment_scale: float = 0.0,
) -> float:
    states = hidden_states_from_state(context, transported_z)
    cache = compute_cache(states, model.layers)
    pre_update_energy = total_energy(cache, model.layers, context.batch_size)
    weight_gradients, bias_gradients = parameter_gradients(states, cache, model.layers)
    if float(output_alignment_scale) > 0.0:
        # The last layer gradient is the transported readout term; scaling it is the
        # smallest output-side alignment aid that leaves the TF2 transport family unchanged.
        weight_gradients[-1] = np.asarray(weight_gradients[-1], dtype=np.float64) * (1.0 + float(output_alignment_scale))
        bias_gradients[-1] = np.asarray(bias_gradients[-1], dtype=np.float64) * (1.0 + float(output_alignment_scale))
    apply_parameter_updates(
        model.layers,
        weight_gradients,
        bias_gradients,
        eta_w=eta_w,
        eta_b=eta_b,
    )
    return float(pre_update_energy)


def _output_alignment_scale_for_step(
    config: FMPCTF2Config,
    *,
    is_terminal_step: bool,
) -> float:
    weight = float(config.transported_output_alignment_weight)
    if weight <= 0.0:
        return 0.0
    schedule = config.transported_output_alignment_schedule
    if schedule == "none":
        return 0.0
    if schedule == "every_micro_step":
        return weight
    if schedule == "final_micro_step_only":
        return weight if is_terminal_step else 0.0
    raise ValueError(f"Unsupported transported_output_alignment_schedule '{schedule}'.")


def _resolved_terminal_local_field_intervention_step_indices(config: FMPCTF2Config) -> tuple[int, ...]:
    resolved_indices = {
        int(config.micro_steps) + int(step_offset) if int(step_offset) < 0 else int(step_offset)
        for step_offset in config.terminal_local_field_intervention_step_offsets
    }
    return tuple(sorted(resolved_indices))


def _run_tf2_micro_step(
    model: PCNetwork,
    psi_network: MLPNetwork,
    config: FMPCTF2Config,
    context: FMPCTF1Context,
    z_on_k: np.ndarray,
    z_lf_k: np.ndarray,
    *,
    t_k: float,
    dt: float,
    r_k: float,
    lambda_id: float,
    apply_theta_update: bool,
    theta_eta_w: float,
    theta_eta_b: float,
    is_terminal_step: bool = False,
    apply_direction_intervention: bool = False,
    onpolicy_mix_ratio: float | None = None,
    event_log: list[str] | None = None,
    sample_collector: TF2SampleCollector | None = None,
    sample_metadata: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float, TF2MicroStepPlan]:
    if event_log is not None:
        event_log.append("plan")
    plan = _plan_tf2_micro_step(
        context,
        psi_network,
        config,
        z_on_k,
        z_lf_k,
        t_k=t_k,
        dt=dt,
        r_k=r_k,
        onpolicy_mix_ratio=onpolicy_mix_ratio,
    )
    z_on_next = plan.z_on_next.copy()
    z_lf_next = plan.z_lf_next.copy()
    effective_prediction = _psi_predict(psi_network, plan.psi_inputs, config)
    if apply_direction_intervention and config.terminal_local_field_direction_intervention != "none":
        effective_prediction, z_on_next = _apply_terminal_local_field_direction_intervention(
            z_on_k,
            dt,
            plan,
            config,
            context=context,
            output_weight=model.layers[-1].weight,
        )
    if sample_collector is not None:
        payload: dict[str, Any] = {
            "psi_inputs": plan.psi_inputs.copy(),
            "boot_targets": plan.boot_targets.copy(),
            "identity_targets": plan.identity_targets.copy(),
            "lambda_id": float(lambda_id),
            "identity_tangent_mode": _identity_tangent_mode(config),
            "source_counts": dict(plan.source_counts),
            "t_k": float(t_k),
            "r_k": float(r_k),
            "dt": float(dt),
        }
        if sample_metadata is not None:
            payload.update(sample_metadata)
        sample_collector(payload)
    if event_log is not None:
        event_log.append("advance")
    if apply_theta_update:
        output_alignment_scale = _output_alignment_scale_for_step(
            config,
            is_terminal_step=is_terminal_step,
        )
        theta_energy = _theta_update_from_transported_state(
            model,
            context,
            z_on_next,
            eta_w=theta_eta_w,
            eta_b=theta_eta_b,
            output_alignment_scale=output_alignment_scale,
        )
        if event_log is not None:
            event_log.append("theta_update")
    else:
        theta_energy = hidden_energy_from_state(context, z_on_next)
    boot_loss = float(np.mean((effective_prediction - plan.boot_targets) ** 2))
    identity_loss = float(np.mean((effective_prediction - plan.identity_targets) ** 2))
    if lambda_id > 0.0:
        combined_target = (plan.boot_targets + (lambda_id * plan.identity_targets)) / (1.0 + lambda_id)
        loss_scale = 1.0 + lambda_id
    else:
        combined_target = plan.boot_targets
        loss_scale = 1.0
    total_loss = boot_loss + (lambda_id * identity_loss)
    _weighted_mse_step(psi_network, plan.psi_inputs, combined_target, loss_scale=loss_scale)
    if event_log is not None:
        event_log.append("psi_update")
    return z_on_next, z_lf_next, total_loss, boot_loss, identity_loss, float(theta_energy), plan


def _train_one_batch_tf2(
    model: PCNetwork,
    psi_network: MLPNetwork,
    config: FMPCTF2Config,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    *,
    lambda_id: float,
    epoch_index: int | None = None,
    sample_collector: TF2SampleCollector | None = None,
    sample_metadata: dict[str, Any] | None = None,
) -> tuple[float, float, float, float]:
    context = build_tf1_context(model, x_batch, y_batch)
    knots = np.linspace(0.0, 1.0, int(config.micro_steps) + 1, dtype=np.float64)
    z_on = context.z0.copy()
    z_lf = context.z0.copy()
    active_cadence = _active_theta_update_cadence(config, epoch_index)
    intervention_step_indices = (
        _resolved_terminal_local_field_intervention_step_indices(config)
        if config.terminal_local_field_direction_intervention != "none"
        else ()
    )
    micro_eta_w, micro_eta_b = _theta_micro_learning_rates(config, active_cadence)
    active_mix_ratio = _active_onpolicy_mix_ratio(config, epoch_index)
    total_losses: list[float] = []
    boot_losses: list[float] = []
    identity_losses: list[float] = []

    for step_index in range(config.micro_steps):
        t_k = float(knots[step_index])
        r_k = 1.0 - t_k
        dt = float(knots[step_index + 1] - knots[step_index])
        z_on, z_lf, total_loss, boot_loss, identity_loss, _, _ = _run_tf2_micro_step(
            model,
            psi_network,
            config,
            context,
            z_on,
            z_lf,
            t_k=t_k,
            dt=dt,
            r_k=r_k,
            lambda_id=lambda_id,
            is_terminal_step=bool(step_index == (int(config.micro_steps) - 1)),
            apply_direction_intervention=bool(step_index in intervention_step_indices),
            apply_theta_update=_theta_update_due_for_step(active_cadence, step_index),
            theta_eta_w=micro_eta_w,
            theta_eta_b=micro_eta_b,
            onpolicy_mix_ratio=active_mix_ratio,
            sample_collector=sample_collector,
            sample_metadata=(
                {
                    **({} if sample_metadata is None else sample_metadata),
                    "epoch_index": None if epoch_index is None else int(epoch_index),
                    "step_index": int(step_index),
                }
            ),
        )
        total_losses.append(total_loss)
        boot_losses.append(boot_loss)
        identity_losses.append(identity_loss)

    transported_final_energy = hidden_energy_from_state(context, z_on)
    if active_cadence == "terminal_only":
        _theta_update_from_transported_state(
            model,
            context,
            z_on,
            eta_w=float(config.eta_w),
            eta_b=_resolved_eta_b(config),
        )

    return (
        float(np.mean(total_losses)),
        float(np.mean(boot_losses)),
        float(np.mean(identity_losses)),
        float(transported_final_energy),
    )


def _learned_velocity_fn(
    context: FMPCTF1Context,
    psi_network: MLPNetwork,
    config: FMPCTF2Config,
):
    def _velocity(z_t: np.ndarray, t_k: float, r_k: float) -> np.ndarray:
        features = teacher_free_state_features(context, z_t)
        feature_tangents = _feature_tangents_for_state(context, z_t, config)
        inputs = _build_psi_input(
            config,
            z_t,
            context.targets,
            t=t_k,
            r=r_k,
            features=features,
        )
        input_tangent = _build_psi_input_tangent(
            config,
            features.g_t,
            target_dim=context.target_dim,
            t=t_k,
            r=r_k,
            feature_tangents=feature_tangents,
        )
        return _psi_forward_with_jvp(psi_network, inputs, input_tangent, config).output

    return _velocity


def _evaluate_transport_split(
    model: PCNetwork,
    psi_network: MLPNetwork,
    config: FMPCTF2Config,
    x_split: np.ndarray,
    y_split: np.ndarray,
) -> _SplitTransportMetrics:
    context = build_tf1_context(model, x_split, y_split)
    learned = rollout_hidden_transport(
        context,
        context.z0,
        transport_steps=config.micro_steps,
        mode="learned",
        velocity_fn=_learned_velocity_fn(context, psi_network, config),
    )
    identity = rollout_hidden_transport(
        context,
        context.z0,
        transport_steps=config.micro_steps,
        mode="identity",
    )
    local_field = rollout_hidden_transport(
        context,
        context.z0,
        transport_steps=config.micro_steps,
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


def _config_payload(config: FMPCTF2Config) -> dict[str, Any]:
    resolved_theta_update_cadence = _resolved_theta_update_cadence(config)
    theta_micro_lr, theta_micro_bias_lr = _theta_micro_learning_rates(
        config,
        resolved_theta_update_cadence,
    )
    return {
        "phase": "Phase Incremental Bridge",
        "stage": "ifmpc_bridge_stage",
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
            "eta_b": float(_resolved_eta_b(config)),
            "eval_steps": int(config.eval_steps),
            "state_init": config.state_init,
        },
        "transport": {
            "family_lineage": config.family_lineage,
            "use_teacher_free_features": bool(config.use_teacher_free_features),
            "feature_aware_tangents": bool(config.feature_aware_tangents),
            "identity_tangent_mode": _identity_tangent_mode(config),
            "tangent_epsilon": float(config.tangent_epsilon),
            "micro_steps": int(config.micro_steps),
            "incremental_weight_updates": bool(config.incremental_weight_updates),
            "supervision_policy": config.supervision_policy,
            "theta_update_budget": config.theta_update_budget,
            "theta_update_cadence": resolved_theta_update_cadence,
            "onpolicy_mix_ratio": float(_resolved_onpolicy_mix_ratio(config)),
            "interleaving_start": config.interleaving_start,
            "theta_micro_lr": float(theta_micro_lr),
            "theta_micro_bias_lr": float(theta_micro_bias_lr),
            "bootstrap_integrator": config.bootstrap_integrator,
            "bootstrap_substeps": int(config.bootstrap_substeps),
            "psi_family": config.psi_family,
            "time_encoding_variant": config.time_encoding_variant,
            "terminal_local_field_direction_intervention": config.terminal_local_field_direction_intervention,
            "terminal_local_field_angle_clip_degrees": float(config.terminal_local_field_angle_clip_degrees),
            "terminal_local_field_intervention_step_offsets": [
                int(value) for value in config.terminal_local_field_intervention_step_offsets
            ],
            "terminal_local_field_rowspace_angle_clip_degrees": float(
                config.terminal_local_field_rowspace_angle_clip_degrees
            ),
            "terminal_local_field_orthogonal_angle_clip_degrees": float(
                config.terminal_local_field_orthogonal_angle_clip_degrees
            ),
            "transported_output_alignment_weight": float(config.transported_output_alignment_weight),
            "transported_output_alignment_schedule": config.transported_output_alignment_schedule,
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


def run_fmpc_tf2_experiment(config: FMPCTF2Config) -> FMPCTF2RunResult:
    """Run the teacher-free TF2 iFMPC bridge experiment on digits."""

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

    forward_init_stability_metrics = _forward_init_stability_metrics(
        model,
        split.x_train[: config.batch_size],
        split.y_train[: config.batch_size],
    )

    epoch_rows: list[dict[str, Any]] = []
    epoch_snapshots: list[FMPCTF2EpochSnapshot] = []
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
            train_loss, boot_loss, identity_loss, transported_energy = _train_one_batch_tf2(
                model,
                psi_network,
                config,
                x_batch,
                y_batch,
                lambda_id=lambda_id,
                epoch_index=epoch_index,
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
            FMPCTF2EpochMetrics(
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
            FMPCTF2EpochSnapshot(
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
    selected_snapshot = next(snapshot for snapshot in epoch_snapshots if snapshot.epoch == selected_epoch)
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
    resolved_theta_update_cadence = _resolved_theta_update_cadence(config)
    theta_micro_lr, theta_micro_bias_lr = _theta_micro_learning_rates(
        config,
        resolved_theta_update_cadence,
    )
    resolved_onpolicy_mix_ratio = _resolved_onpolicy_mix_ratio(config)

    summary = {
        "phase": "Phase Incremental Bridge",
        "stage": "ifmpc_bridge_stage",
        "teacher_free": True,
        "uses_teacher_artifacts": False,
        "jpc_runtime_dependency": False,
        "preset_name": config.preset_name,
        "family_lineage": config.family_lineage,
        "use_teacher_free_features": bool(config.use_teacher_free_features),
        "feature_aware_tangents": bool(config.feature_aware_tangents),
        "identity_tangent_mode": _identity_tangent_mode(config),
        "tangent_epsilon": float(config.tangent_epsilon),
        "incremental_weight_updates": bool(config.incremental_weight_updates),
        "supervision_policy": config.supervision_policy,
        "micro_steps": int(config.micro_steps),
        "theta_update_budget": config.theta_update_budget,
        "theta_update_cadence": resolved_theta_update_cadence,
        "onpolicy_mix_ratio": float(resolved_onpolicy_mix_ratio),
        "interleaving_start": config.interleaving_start,
        "psi_family": config.psi_family,
        "time_encoding_variant": config.time_encoding_variant,
        "terminal_local_field_direction_intervention": config.terminal_local_field_direction_intervention,
        "terminal_local_field_angle_clip_degrees": float(config.terminal_local_field_angle_clip_degrees),
        "terminal_local_field_intervention_step_offsets": [
            int(value) for value in config.terminal_local_field_intervention_step_offsets
        ],
        "terminal_local_field_rowspace_angle_clip_degrees": float(
            config.terminal_local_field_rowspace_angle_clip_degrees
        ),
        "terminal_local_field_orthogonal_angle_clip_degrees": float(
            config.terminal_local_field_orthogonal_angle_clip_degrees
        ),
        "transported_output_alignment_weight": float(config.transported_output_alignment_weight),
        "transported_output_alignment_schedule": config.transported_output_alignment_schedule,
        "theta_micro_lr": float(theta_micro_lr),
        "theta_micro_bias_lr": float(theta_micro_bias_lr),
        "bootstrap_integrator": config.bootstrap_integrator,
        "bootstrap_substeps": int(config.bootstrap_substeps),
        "identity_loss_weight": float(config.identity_loss_weight),
        "warmup_epochs": int(config.warmup_epochs),
        "hybrid_ramp_epochs": int(config.hybrid_ramp_epochs),
        "selection_metric": config.selection_metric,
        "checkpoint_selector": config.checkpoint_selector,
        "selection_metric_source": "val_metric",
        "report_metric_source": "test_metric",
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
        "rollout_knots": val_transport.rollout_knots,
        "selection_diagnostics_artifact": "selection_diagnostics.json",
        "forward_init_stability_metrics": forward_init_stability_metrics,
        "identity_baseline": {
            "val_transported_final_energy": float(val_transport.identity_final_energy),
            "test_transported_final_energy": float(test_transport.identity_final_energy),
            "micro_steps": int(config.micro_steps),
            "rollout_knots": val_transport.rollout_knots,
            "energy_metric": "baseline_pc_energy",
        },
        "local_field_only_baseline": {
            "val_transported_final_energy": float(val_transport.local_field_only_final_energy),
            "test_transported_final_energy": float(test_transport.local_field_only_final_energy),
            "micro_steps": int(config.micro_steps),
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
    return FMPCTF2RunResult(
        run_dir=run_dir,
        config=_config_payload(config),
        epoch_metrics=epoch_rows,
        summary=summary,
        model=model,
        psi_network=psi_network,
        epoch_snapshots=epoch_snapshots,
        selection_diagnostics=selection_diagnostics,
    )
