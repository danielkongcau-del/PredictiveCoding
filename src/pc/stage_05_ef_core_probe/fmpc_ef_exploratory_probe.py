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
    FMPCTF1StateFeatures,
    FMPCTF1StateFeatureTangents,
    bootstrap_average_velocity_target,
    build_tf1_context,
    hidden_energy_from_state,
    hidden_local_flow,
    hidden_states_from_state,
    rollout_hidden_transport,
    teacher_free_feature_tangents,
    teacher_free_state_features,
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
EndpointSemigroupTargetMode = Literal["single_sided_detached_split_endpoint"]

RESIDUAL_MEANFLOW_TRANSPORT_FAMILY = "residual_meanflow_core"
TWO_BRANCH_RESIDUAL_MEANFLOW_TRANSPORT_FAMILY = "two_branch_residual_meanflow_core"
RESIDUAL_IDENTITY_MODE = "residual_corrected_meanflow"
STAGE05_V1_CANDIDATE_NAME = "stage05_v1_corrected_residual_meanflow_core"
STAGE05_V2_CANDIDATE_NAME = "stage05_v2_two_branch_corrected_residual_meanflow_core"
STAGE05_V3A_CANDIDATE_NAME = "stage05_v3a_explicit_transport_drift_contract"
STAGE05_V3B_CANDIDATE_NAME = "stage05_v3b_trajectory_curriculum_contract"
STAGE05_V3B_REFINED_CANDIDATE_NAME = "stage05_v3b_stronger_traj_curr_weight"
STAGE05_V3C_CANDIDATE_NAME = "stage05_v3c_endpoint_semigroup_consistency_contract"
U_PSI_INPUT_CONTRACT = "concat([z_t, target_onehot, t, r])"
M_TRAJ_INPUT_CONTRACT = "concat([z_t, target_onehot, t, r])"
M_STATE_INPUT_CONTRACT = "concat([g_t, e_out_t, F_t])"
BOOTSTRAP_TARGET_CONTRACT = "m_boot = ((Phi_LF_r(z_t; c) - z_t) / r) - g_t"
RESIDUAL_IDENTITY_TARGET_CONTRACT = "m_id = r * D_T g_t + r * D_T m_psi"
TWO_BRANCH_RESIDUAL_IDENTITY_TARGET_CONTRACT = (
    "m_id = r * D_T g_t + r * D_T m_traj + r * D_T m_state"
)
EXPLICIT_TRANSPORT_DRIFT_TARGET_CONTRACT = (
    "gbar_boot = avg local flow over the same bootstrap interval; "
    "d_boot = gbar_boot - g_t; q_boot = u_boot - gbar_boot"
)
TRAJECTORY_CURRICULUM_TARGET_CONTRACT = (
    "u_curr_target = alpha * u_boot(z_t, alpha * r, t; c) + "
    "(1 - alpha) * u_hat(z_s_boot, r_s, s; c) [detached target side]"
)
TRAJECTORY_CURRICULUM_SCHEDULE_IDENTITY = "warmup_sigmoid_to_alpha_floor"
SEMIGROUP_SPLIT_IDENTITY = "s = t + alpha * r; r_s = (1 - alpha) * r"
SEMIGROUP_TARGET_MODE = "single_sided_detached_split_endpoint"
SEMIGROUP_TARGET_CONTRACT = (
    "z_hat_split_target = stopgrad(z_hat_split); "
    "L_sg = || z_hat_direct - z_hat_split_target ||^2"
)
SEMIGROUP_UPDATE_PROXY_CONTRACT = (
    "m_sg_target = ((z_hat_split_target - z_t) / r) - g_t with per-sample r^2 weighting"
)


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
    use_two_branch_residual_core: bool = False
    feature_aware_state_branch_tangents: bool = False
    use_explicit_transport_drift_decomposition: bool = False
    use_trajectory_curriculum_contract: bool = False
    lambda_drift: float = 1.0
    lambda_traj_curr: float = 0.1
    alpha_floor: float = 0.5
    alpha_warmup_epochs: int = 3
    alpha_ramp_epochs: int = 3
    trajectory_curriculum_schedule: str = TRAJECTORY_CURRICULUM_SCHEDULE_IDENTITY
    psi_hidden_dims: tuple[int, ...] = (128,)
    psi_weight_scale: float = 0.01
    psi_eta_w: float = 0.01
    psi_eta_b: float | None = 0.01
    bootstrap_integrator: Literal["euler", "rk2"] = "rk2"
    bootstrap_substeps: int = 4
    candidate_name_override: str | None = None
    use_endpoint_semigroup_consistency_probe: bool = False
    lambda_sg: float = 0.05
    semigroup_target_mode: EndpointSemigroupTargetMode = SEMIGROUP_TARGET_MODE
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
        if self.lambda_drift < 0.0:
            raise ValueError("lambda_drift must be non-negative.")
        if self.lambda_traj_curr < 0.0:
            raise ValueError("lambda_traj_curr must be non-negative.")
        if self.lambda_sg < 0.0:
            raise ValueError("lambda_sg must be non-negative.")
        if self.tangent_epsilon <= 0.0:
            raise ValueError("tangent_epsilon must be positive.")
        if self.alpha_warmup_epochs < 0:
            raise ValueError("alpha_warmup_epochs must be non-negative.")
        if self.alpha_ramp_epochs < 0:
            raise ValueError("alpha_ramp_epochs must be non-negative.")
        if self.trajectory_curriculum_schedule != TRAJECTORY_CURRICULUM_SCHEDULE_IDENTITY:
            raise ValueError(
                "Stage 05 currently supports trajectory_curriculum_schedule="
                f"'{TRAJECTORY_CURRICULUM_SCHEDULE_IDENTITY}' only."
            )
        if self.semigroup_target_mode != SEMIGROUP_TARGET_MODE:
            raise ValueError(
                "Stage 05 currently supports semigroup_target_mode="
                f"'{SEMIGROUP_TARGET_MODE}' only."
            )
        if self.identity_mode != RESIDUAL_IDENTITY_MODE:
            raise ValueError(
                f"Stage 05 currently supports identity_mode='{RESIDUAL_IDENTITY_MODE}' only."
            )
        if self.selection_metric != "val_configured_transported_final_energy":
            raise ValueError("Only val_configured_transported_final_energy is supported.")
        if self.use_explicit_transport_drift_decomposition and not self.use_two_branch_residual_core:
            raise ValueError(
                "use_explicit_transport_drift_decomposition requires use_two_branch_residual_core=True."
            )
        if self.use_trajectory_curriculum_contract and not self.use_explicit_transport_drift_decomposition:
            raise ValueError(
                "use_trajectory_curriculum_contract requires "
                "use_explicit_transport_drift_decomposition=True."
            )
        if self.use_trajectory_curriculum_contract and not (0.0 < self.alpha_floor < 1.0):
            raise ValueError(
                "alpha_floor must satisfy 0 < alpha_floor < 1 when "
                "use_trajectory_curriculum_contract=True."
            )
        if (
            self.use_endpoint_semigroup_consistency_probe
            and not self.use_trajectory_curriculum_contract
        ):
            raise ValueError(
                "use_endpoint_semigroup_consistency_probe requires "
                "use_trajectory_curriculum_contract=True."
            )

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


@dataclass(frozen=True)
class CorrectedResidualIdentityTarget:
    """Corrected residual identity target and its explicit constituent terms."""

    target: np.ndarray
    anchor_derivative: np.ndarray
    residual_jvp: np.ndarray
    anchor_term: np.ndarray
    residual_term: np.ndarray
    trajectory_jvp: np.ndarray
    trajectory_term: np.ndarray
    state_jvp: np.ndarray
    state_term: np.ndarray

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
        object.__setattr__(
            self, "trajectory_jvp", _as_batch_first("trajectory_jvp", self.trajectory_jvp)
        )
        object.__setattr__(
            self, "trajectory_term", _as_batch_first("trajectory_term", self.trajectory_term)
        )
        object.__setattr__(self, "state_jvp", _as_batch_first("state_jvp", self.state_jvp))
        object.__setattr__(self, "state_term", _as_batch_first("state_term", self.state_term))
        if self.target.shape != self.anchor_derivative.shape:
            raise ValueError("target and anchor_derivative must share the same shape.")
        if self.target.shape != self.residual_jvp.shape:
            raise ValueError("target and residual_jvp must share the same shape.")
        if self.target.shape != self.anchor_term.shape:
            raise ValueError("target and anchor_term must share the same shape.")
        if self.target.shape != self.residual_term.shape:
            raise ValueError("target and residual_term must share the same shape.")
        if self.target.shape != self.trajectory_jvp.shape:
            raise ValueError("target and trajectory_jvp must share the same shape.")
        if self.target.shape != self.trajectory_term.shape:
            raise ValueError("target and trajectory_term must share the same shape.")
        if self.target.shape != self.state_jvp.shape:
            raise ValueError("target and state_jvp must share the same shape.")
        if self.target.shape != self.state_term.shape:
            raise ValueError("target and state_term must share the same shape.")


@dataclass(frozen=True)
class ExplicitTransportDriftBootstrapTargets:
    """Bootstrap target decomposition for the Stage 05 v3-A working-hypothesis candidate."""

    u_boot: np.ndarray
    g_t: np.ndarray
    gbar_boot: np.ndarray
    transport_target: np.ndarray
    drift_target: np.ndarray
    residual_target: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "u_boot", _as_batch_first("u_boot", self.u_boot))
        object.__setattr__(self, "g_t", _as_batch_first("g_t", self.g_t))
        object.__setattr__(self, "gbar_boot", _as_batch_first("gbar_boot", self.gbar_boot))
        object.__setattr__(
            self,
            "transport_target",
            _as_batch_first("transport_target", self.transport_target),
        )
        object.__setattr__(self, "drift_target", _as_batch_first("drift_target", self.drift_target))
        object.__setattr__(
            self,
            "residual_target",
            _as_batch_first("residual_target", self.residual_target),
        )
        reference_shape = self.u_boot.shape
        for name in (
            "g_t",
            "gbar_boot",
            "transport_target",
            "drift_target",
            "residual_target",
        ):
            if getattr(self, name).shape != reference_shape:
                raise ValueError(f"{name} must share the same shape as u_boot.")


@dataclass(frozen=True)
class TrajectoryCurriculumTargets:
    """Detached aggregate trajectory target for the first Stage 05 v3-B candidate."""

    alpha: float
    split_time: float
    continuation_remaining_horizon: float
    short_horizon_bootstrap_velocity: np.ndarray
    bootstrap_intermediate_state: np.ndarray
    continuation_velocity: np.ndarray
    current_velocity_target: np.ndarray
    residual_target: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "short_horizon_bootstrap_velocity",
            _as_batch_first(
                "short_horizon_bootstrap_velocity",
                self.short_horizon_bootstrap_velocity,
            ),
        )
        object.__setattr__(
            self,
            "bootstrap_intermediate_state",
            _as_batch_first("bootstrap_intermediate_state", self.bootstrap_intermediate_state),
        )
        object.__setattr__(
            self,
            "continuation_velocity",
            _as_batch_first("continuation_velocity", self.continuation_velocity),
        )
        object.__setattr__(
            self,
            "current_velocity_target",
            _as_batch_first("current_velocity_target", self.current_velocity_target),
        )
        object.__setattr__(
            self,
            "residual_target",
            _as_batch_first("residual_target", self.residual_target),
        )
        reference_shape = self.short_horizon_bootstrap_velocity.shape
        for name in (
            "bootstrap_intermediate_state",
            "continuation_velocity",
            "current_velocity_target",
            "residual_target",
        ):
            if getattr(self, name).shape != reference_shape:
                raise ValueError(f"{name} must share the same shape as short_horizon_bootstrap_velocity.")


@dataclass(frozen=True)
class EndpointSemigroupTargets:
    """Detached endpoint-semigroup targets for the first diagnostic-only v3-C probe."""

    alpha: float
    split_time: float
    continuation_remaining_horizon: float
    direct_velocity: np.ndarray
    short_horizon_velocity: np.ndarray
    continuation_velocity: np.ndarray
    direct_endpoint: np.ndarray
    midpoint_state: np.ndarray
    split_endpoint: np.ndarray
    split_endpoint_target: np.ndarray
    semigroup_residual: np.ndarray
    velocity_target: np.ndarray
    residual_target: np.ndarray
    loss_weights: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "direct_velocity", _as_batch_first("direct_velocity", self.direct_velocity))
        object.__setattr__(
            self,
            "short_horizon_velocity",
            _as_batch_first("short_horizon_velocity", self.short_horizon_velocity),
        )
        object.__setattr__(
            self,
            "continuation_velocity",
            _as_batch_first("continuation_velocity", self.continuation_velocity),
        )
        object.__setattr__(self, "direct_endpoint", _as_batch_first("direct_endpoint", self.direct_endpoint))
        object.__setattr__(self, "midpoint_state", _as_batch_first("midpoint_state", self.midpoint_state))
        object.__setattr__(self, "split_endpoint", _as_batch_first("split_endpoint", self.split_endpoint))
        object.__setattr__(
            self,
            "split_endpoint_target",
            _as_batch_first("split_endpoint_target", self.split_endpoint_target),
        )
        object.__setattr__(
            self,
            "semigroup_residual",
            _as_batch_first("semigroup_residual", self.semigroup_residual),
        )
        object.__setattr__(self, "velocity_target", _as_batch_first("velocity_target", self.velocity_target))
        object.__setattr__(self, "residual_target", _as_batch_first("residual_target", self.residual_target))
        object.__setattr__(self, "loss_weights", _as_batch_first("loss_weights", self.loss_weights))
        reference_shape = self.direct_velocity.shape
        for name in (
            "short_horizon_velocity",
            "continuation_velocity",
            "direct_endpoint",
            "midpoint_state",
            "split_endpoint",
            "split_endpoint_target",
            "semigroup_residual",
            "velocity_target",
            "residual_target",
        ):
            if getattr(self, name).shape != reference_shape:
                raise ValueError(f"{name} must share the same shape as direct_velocity.")
        if self.loss_weights.shape != (reference_shape[0], 1):
            raise ValueError("loss_weights must be shaped (batch, 1).")


@dataclass(frozen=True)
class ResidualSupervisionBatch:
    """Stage 05 supervision tensors, with optional v2 state-branch inputs."""

    trajectory_inputs: np.ndarray
    state_inputs: np.ndarray | None
    boot_targets: np.ndarray
    identity_targets: np.ndarray
    gbar_boot: np.ndarray | None = None
    transport_targets: np.ndarray | None = None
    drift_targets: np.ndarray | None = None
    trajectory_curriculum_targets: np.ndarray | None = None
    semigroup_targets: np.ndarray | None = None
    semigroup_loss_weights: np.ndarray | None = None
    explicit_transport_drift_decomposition_enabled: bool = False
    trajectory_curriculum_enabled: bool = False
    endpoint_semigroup_consistency_enabled: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "trajectory_inputs", _as_batch_first("trajectory_inputs", self.trajectory_inputs)
        )
        if self.state_inputs is not None:
            object.__setattr__(self, "state_inputs", _as_batch_first("state_inputs", self.state_inputs))
        object.__setattr__(self, "boot_targets", _as_batch_first("boot_targets", self.boot_targets))
        object.__setattr__(
            self, "identity_targets", _as_batch_first("identity_targets", self.identity_targets)
        )
        if self.gbar_boot is not None:
            object.__setattr__(self, "gbar_boot", _as_batch_first("gbar_boot", self.gbar_boot))
        if self.transport_targets is not None:
            object.__setattr__(
                self,
                "transport_targets",
                _as_batch_first("transport_targets", self.transport_targets),
            )
        if self.drift_targets is not None:
            object.__setattr__(
                self,
                "drift_targets",
                _as_batch_first("drift_targets", self.drift_targets),
            )
        if self.trajectory_curriculum_targets is not None:
            object.__setattr__(
                self,
                "trajectory_curriculum_targets",
                _as_batch_first("trajectory_curriculum_targets", self.trajectory_curriculum_targets),
            )
        if self.semigroup_targets is not None:
            object.__setattr__(
                self,
                "semigroup_targets",
                _as_batch_first("semigroup_targets", self.semigroup_targets),
            )
        if self.semigroup_loss_weights is not None:
            object.__setattr__(
                self,
                "semigroup_loss_weights",
                _as_batch_first("semigroup_loss_weights", self.semigroup_loss_weights),
            )
        if self.trajectory_inputs.shape[0] != self.boot_targets.shape[0]:
            raise ValueError("trajectory_inputs and boot_targets must share the same batch size.")
        if self.trajectory_inputs.shape[0] != self.identity_targets.shape[0]:
            raise ValueError("trajectory_inputs and identity_targets must share the same batch size.")
        if self.state_inputs is not None and self.state_inputs.shape[0] != self.trajectory_inputs.shape[0]:
            raise ValueError("state_inputs and trajectory_inputs must share the same batch size.")
        if self.gbar_boot is not None and self.gbar_boot.shape != self.boot_targets.shape:
            raise ValueError("gbar_boot must share the same shape as boot_targets.")
        if self.transport_targets is not None and self.transport_targets.shape != self.boot_targets.shape:
            raise ValueError("transport_targets must share the same shape as boot_targets.")
        if self.drift_targets is not None and self.drift_targets.shape != self.boot_targets.shape:
            raise ValueError("drift_targets must share the same shape as boot_targets.")
        if (
            self.trajectory_curriculum_targets is not None
            and self.trajectory_curriculum_targets.shape != self.boot_targets.shape
        ):
            raise ValueError(
                "trajectory_curriculum_targets must share the same shape as boot_targets."
            )
        if self.semigroup_targets is not None and self.semigroup_targets.shape != self.boot_targets.shape:
            raise ValueError("semigroup_targets must share the same shape as boot_targets.")
        if self.semigroup_loss_weights is not None and self.semigroup_loss_weights.shape != (
            self.boot_targets.shape[0],
            1,
        ):
            raise ValueError("semigroup_loss_weights must be shaped (batch, 1).")


@dataclass(frozen=True)
class FMPCEFExploratoryProbeEpochMetrics:
    epoch: int
    stage: str
    lambda_id: float
    alpha: float
    lambda_traj_curr: float
    lambda_sg: float
    train_total_loss: float
    train_boot_loss: float
    train_transport_loss: float
    train_drift_loss: float
    train_identity_loss: float
    train_traj_curr_loss: float
    train_semigroup_loss: float
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
    psi_snapshot: dict[str, list[tuple[np.ndarray, np.ndarray]] | None]


@dataclass
class FMPCEFExploratoryProbeRunResult:
    run_dir: Path
    config: dict[str, Any]
    epoch_metrics: list[dict[str, Any]]
    summary: dict[str, Any]
    model: PCNetwork | None = None
    psi_network: Stage05ResidualCoreNetworks | None = None


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
        "lambda_drift": 1.0,
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


def build_stage05_v3b_stronger_traj_curr_weight_config(
    **overrides: Any,
) -> FMPCEFExploratoryProbeConfig:
    """Return the promoted refined v3-B reference as an explicit probe-side preset."""

    payload: dict[str, Any] = {
        "use_two_branch_residual_core": True,
        "feature_aware_state_branch_tangents": True,
        "use_explicit_transport_drift_decomposition": True,
        "use_trajectory_curriculum_contract": True,
        "lambda_drift": 1.0,
        "lambda_traj_curr": 0.2,
        "alpha_floor": 0.5,
        "alpha_warmup_epochs": 3,
        "alpha_ramp_epochs": 3,
        "candidate_name_override": STAGE05_V3B_REFINED_CANDIDATE_NAME,
    }
    payload.update(overrides)
    return build_fmpc_ef_exploratory_probe_config(**payload)


def build_stage05_v3c_endpoint_semigroup_config(
    **overrides: Any,
) -> FMPCEFExploratoryProbeConfig:
    """Return the first diagnostic-only v3-C candidate on top of refined v3-B."""

    payload: dict[str, Any] = {
        "use_endpoint_semigroup_consistency_probe": True,
        "lambda_sg": 0.05,
        "candidate_name_override": STAGE05_V3C_CANDIDATE_NAME,
    }
    payload.update(overrides)
    return build_stage05_v3b_stronger_traj_curr_weight_config(**payload)


def _transport_family_name(config: FMPCEFExploratoryProbeConfig) -> str:
    if config.use_two_branch_residual_core:
        return TWO_BRANCH_RESIDUAL_MEANFLOW_TRANSPORT_FAMILY
    return RESIDUAL_MEANFLOW_TRANSPORT_FAMILY


def _candidate_name(config: FMPCEFExploratoryProbeConfig) -> str:
    if config.candidate_name_override is not None:
        return str(config.candidate_name_override)
    if config.use_endpoint_semigroup_consistency_probe:
        return STAGE05_V3C_CANDIDATE_NAME
    if config.use_trajectory_curriculum_contract:
        return STAGE05_V3B_CANDIDATE_NAME
    if config.use_explicit_transport_drift_decomposition:
        return STAGE05_V3A_CANDIDATE_NAME
    if config.use_two_branch_residual_core:
        return STAGE05_V2_CANDIDATE_NAME
    return STAGE05_V1_CANDIDATE_NAME


def _residual_branch_structure(config: FMPCEFExploratoryProbeConfig) -> str:
    if config.use_two_branch_residual_core:
        return "two_branch"
    return "single_branch"


def _psi_family_name(config: FMPCEFExploratoryProbeConfig) -> str:
    if config.use_two_branch_residual_core:
        return "two_branch_residual_local_flow_mlp"
    return "residual_local_flow_mlp"


def _velocity_parameterization(config: FMPCEFExploratoryProbeConfig) -> str:
    if config.use_explicit_transport_drift_decomposition:
        return (
            "u_psi = g_theta + q_psi(z_t, target_onehot, t, r) + "
            "d_psi(g_t, e_out_t, F_t)"
        )
    if config.use_two_branch_residual_core:
        return (
            "u_psi = g_theta + m_traj(z_t, target_onehot, t, r) + "
            "m_state(g_t, e_out_t, F_t)"
        )
    return "u_psi = g_theta + residual_mlp(z_t, target_onehot, t, r)"


def _u_psi_input_contract(config: FMPCEFExploratoryProbeConfig) -> str:
    if config.use_two_branch_residual_core:
        return (
            "u_psi = g_t + "
            + ("q_psi" if config.use_explicit_transport_drift_decomposition else "m_traj")
            + "(concat([z_t, target_onehot, t, r])) + "
            + ("d_psi" if config.use_explicit_transport_drift_decomposition else "m_state")
            + "(concat([g_t, e_out_t, F_t]))"
        )
    return U_PSI_INPUT_CONTRACT


def _residual_identity_target_contract(config: FMPCEFExploratoryProbeConfig) -> str:
    if config.use_two_branch_residual_core:
        return TWO_BRANCH_RESIDUAL_IDENTITY_TARGET_CONTRACT
    return RESIDUAL_IDENTITY_TARGET_CONTRACT


def _explicit_transport_drift_target_contract(
    config: FMPCEFExploratoryProbeConfig,
) -> str | None:
    if not config.use_explicit_transport_drift_decomposition:
        return None
    return EXPLICIT_TRANSPORT_DRIFT_TARGET_CONTRACT


def _pairwise_v2_placeholder(config: FMPCEFExploratoryProbeConfig) -> dict[str, Any] | None:
    if not config.use_explicit_transport_drift_decomposition:
        return None
    if config.use_endpoint_semigroup_consistency_probe:
        return {
            "status": "pending_real_fixed_budget_v2_vs_promoted_v3b_vs_v3c_comparison",
            "reference_candidate_name": STAGE05_V2_CANDIDATE_NAME,
        }
    if config.use_trajectory_curriculum_contract:
        return {
            "status": "pending_real_fixed_budget_v2_vs_v3a_vs_v3b_comparison",
            "reference_candidate_name": STAGE05_V2_CANDIDATE_NAME,
        }
    return {
        "status": "pending_formal_v2_vs_v3a_comparison",
        "reference_candidate_name": STAGE05_V2_CANDIDATE_NAME,
    }


def _pairwise_v3a_placeholder(config: FMPCEFExploratoryProbeConfig) -> dict[str, Any] | None:
    if config.use_endpoint_semigroup_consistency_probe:
        return None
    if not config.use_trajectory_curriculum_contract:
        return None
    return {
        "status": "pending_real_fixed_budget_v2_vs_v3a_vs_v3b_comparison",
        "reference_candidate_name": STAGE05_V3A_CANDIDATE_NAME,
    }


def _pairwise_promoted_v3b_placeholder(
    config: FMPCEFExploratoryProbeConfig,
) -> dict[str, Any] | None:
    if not config.use_endpoint_semigroup_consistency_probe:
        return None
    return {
        "status": "pending_real_fixed_budget_v2_vs_promoted_v3b_vs_v3c_comparison",
        "reference_candidate_name": STAGE05_V3B_REFINED_CANDIDATE_NAME,
    }


def _gap_closure_decision_placeholder(config: FMPCEFExploratoryProbeConfig) -> str | None:
    if config.use_endpoint_semigroup_consistency_probe:
        return "pending_real_fixed_budget_v2_vs_promoted_v3b_vs_v3c_comparison"
    if config.use_trajectory_curriculum_contract:
        return "pending_real_fixed_budget_v2_vs_v3a_vs_v3b_comparison"
    if not config.use_explicit_transport_drift_decomposition:
        return None
    return "pending_formal_v2_vs_v3a_comparison"


def _recommended_next_move(config: FMPCEFExploratoryProbeConfig) -> str:
    if config.use_endpoint_semigroup_consistency_probe:
        return "run_fixed_budget_v2_vs_promoted_v3b_vs_v3c_comparison"
    if config.use_trajectory_curriculum_contract:
        return "run_fixed_budget_v2_vs_v3a_vs_v3b_comparison"
    if config.use_explicit_transport_drift_decomposition:
        return "run_fixed_budget_v2_vs_v3a_comparison"
    return "no_change"


def _trajectory_curriculum_target_contract(
    config: FMPCEFExploratoryProbeConfig,
) -> str | None:
    if not config.use_trajectory_curriculum_contract:
        return None
    return TRAJECTORY_CURRICULUM_TARGET_CONTRACT


def _semigroup_target_contract(config: FMPCEFExploratoryProbeConfig) -> str | None:
    if not config.use_endpoint_semigroup_consistency_probe:
        return None
    return SEMIGROUP_TARGET_CONTRACT


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


def build_explicit_transport_drift_bootstrap_targets(
    context: FMPCTF1Context,
    z_t: np.ndarray,
    *,
    t: float,
    r: float,
    integrator: Literal["euler", "rk2"] = "rk2",
    substeps: int = 4,
) -> ExplicitTransportDriftBootstrapTargets:
    """Return the v3-A bootstrap decomposition on the same artifact-independent interval.

    This helper exists only to test the Stage 05 v3-A working hypothesis. It does
    not claim that the explicit transport-drift split is already a confirmed fix.
    """

    validate_tf1_time_pair(t, r)
    if integrator not in {"euler", "rk2"}:
        raise ValueError(f"Unsupported bootstrap integrator '{integrator}'.")
    if substeps <= 0:
        raise ValueError("substeps must be positive.")

    z_array = _as_batch_first("z_t", z_t)
    current = z_array.copy()
    step_size = float(r) / float(substeps)
    accumulated_local_flow = np.zeros_like(z_array)

    for _ in range(int(substeps)):
        if integrator == "euler":
            effective_flow = hidden_local_flow(context, current)
            current = current + step_size * effective_flow
        else:
            k1 = hidden_local_flow(context, current)
            mid = current + 0.5 * step_size * k1
            effective_flow = hidden_local_flow(context, mid)
            current = current + step_size * effective_flow
        accumulated_local_flow += effective_flow
        ensure_finite_array(current, "stage05_v3a_bootstrap_z")

    u_boot = (current - z_array) / float(r)
    g_t = hidden_local_flow(context, z_array)
    gbar_boot = accumulated_local_flow / float(substeps)
    transport_target = u_boot - gbar_boot
    drift_target = gbar_boot - g_t
    return ExplicitTransportDriftBootstrapTargets(
        u_boot=u_boot,
        g_t=g_t,
        gbar_boot=gbar_boot,
        transport_target=transport_target,
        drift_target=drift_target,
        residual_target=transport_target + drift_target,
    )


def build_trajectory_curriculum_targets(
    context: FMPCTF1Context,
    psi_network: Stage05ResidualCoreNetworks,
    config: FMPCEFExploratoryProbeConfig,
    z_t: np.ndarray,
    *,
    t: float,
    r: float,
    alpha: float,
) -> TrajectoryCurriculumTargets:
    """Return the first detached v3-B trajectory curriculum target.

    This helper keeps the current Stage 05 remaining-horizon semantics intact. The
    continuation target is evaluated on the target side only and treated as a fixed
    array during the update step.
    """

    validate_tf1_time_pair(t, r)
    if not config.use_trajectory_curriculum_contract:
        raise ValueError("Trajectory curriculum targets require use_trajectory_curriculum_contract=True.")
    alpha_value = float(alpha)
    if not (0.0 < alpha_value < 1.0):
        raise ValueError("Trajectory curriculum targets require 0 < alpha < 1.")

    z_array = _as_batch_first("z_t", z_t)
    short_r = alpha_value * float(r)
    split_time = float(t) + short_r
    continuation_r = (1.0 - alpha_value) * float(r)
    validate_tf1_time_pair(t, short_r)
    validate_tf1_time_pair(split_time, continuation_r)

    short_boot = bootstrap_average_velocity_target(
        context,
        z_array,
        t=t,
        r=short_r,
        integrator=config.bootstrap_integrator,
        substeps=config.bootstrap_substeps,
    )
    z_s_boot = z_array + short_r * short_boot
    continuation_velocity = _predict_total_velocity_at_state(
        context,
        psi_network,
        config,
        z_s_boot,
        t=split_time,
        r=continuation_r,
    )
    current_velocity_target = (
        alpha_value * short_boot
        + (1.0 - alpha_value) * continuation_velocity
    )
    g_t = hidden_local_flow(context, z_array)
    residual_target = current_velocity_target - g_t
    return TrajectoryCurriculumTargets(
        alpha=alpha_value,
        split_time=split_time,
        continuation_remaining_horizon=continuation_r,
        short_horizon_bootstrap_velocity=short_boot,
        bootstrap_intermediate_state=z_s_boot,
        continuation_velocity=continuation_velocity,
        current_velocity_target=current_velocity_target,
        residual_target=residual_target,
    )


def build_endpoint_semigroup_targets(
    context: FMPCTF1Context,
    psi_network: Stage05ResidualCoreNetworks,
    config: FMPCEFExploratoryProbeConfig,
    z_t: np.ndarray,
    *,
    t: float,
    r: float,
    alpha: float,
) -> EndpointSemigroupTargets:
    """Return the first diagnostic-only v3-C endpoint-semigroup probe targets.

    The split endpoint is computed on the target side only and treated as detached.
    The update is then applied through an exactly equivalent residual-target proxy
    with per-sample `r^2` weighting.
    """

    validate_tf1_time_pair(t, r)
    if not config.use_endpoint_semigroup_consistency_probe:
        raise ValueError(
            "Endpoint semigroup targets require use_endpoint_semigroup_consistency_probe=True."
        )
    alpha_value = float(alpha)
    if not (0.0 < alpha_value < 1.0):
        raise ValueError("Endpoint semigroup targets require 0 < alpha < 1.")

    z_array = _as_batch_first("z_t", z_t)
    short_r = alpha_value * float(r)
    split_time = float(t) + short_r
    continuation_r = (1.0 - alpha_value) * float(r)
    validate_tf1_time_pair(t, short_r)
    validate_tf1_time_pair(split_time, continuation_r)

    direct_velocity = _predict_total_velocity_at_state(
        context,
        psi_network,
        config,
        z_array,
        t=t,
        r=r,
    )
    short_horizon_velocity = _predict_total_velocity_at_state(
        context,
        psi_network,
        config,
        z_array,
        t=t,
        r=short_r,
    )
    midpoint_state = z_array + short_r * short_horizon_velocity
    continuation_velocity = _predict_total_velocity_at_state(
        context,
        psi_network,
        config,
        midpoint_state,
        t=split_time,
        r=continuation_r,
    )
    direct_endpoint = z_array + float(r) * direct_velocity
    split_endpoint = (
        z_array
        + short_r * short_horizon_velocity
        + continuation_r * continuation_velocity
    )
    split_endpoint_target = split_endpoint.copy()
    semigroup_residual = direct_endpoint - split_endpoint_target
    velocity_target = (split_endpoint_target - z_array) / float(r)
    g_t = hidden_local_flow(context, z_array)
    residual_target = velocity_target - g_t
    loss_weights = np.full((z_array.shape[0], 1), float(r) ** 2, dtype=np.float64)
    return EndpointSemigroupTargets(
        alpha=alpha_value,
        split_time=split_time,
        continuation_remaining_horizon=continuation_r,
        direct_velocity=direct_velocity,
        short_horizon_velocity=short_horizon_velocity,
        continuation_velocity=continuation_velocity,
        direct_endpoint=direct_endpoint,
        midpoint_state=midpoint_state,
        split_endpoint=split_endpoint,
        split_endpoint_target=split_endpoint_target,
        semigroup_residual=semigroup_residual,
        velocity_target=velocity_target,
        residual_target=residual_target,
        loss_weights=loss_weights,
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


def _snapshot_residual_core_parameters(
    residual_core: Stage05ResidualCoreNetworks,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]] | None]:
    return {
        "trajectory": _snapshot_mlp_parameters(residual_core.trajectory_network),
        "state": (
            _snapshot_mlp_parameters(residual_core.state_network)
            if residual_core.state_network is not None
            else None
        ),
    }


def _restore_residual_core_parameters(
    residual_core: Stage05ResidualCoreNetworks,
    snapshot: dict[str, list[tuple[np.ndarray, np.ndarray]] | None],
) -> None:
    trajectory_snapshot = snapshot.get("trajectory")
    if trajectory_snapshot is None:
        raise ValueError("Residual-core trajectory snapshot is required.")
    _restore_mlp_parameters(residual_core.trajectory_network, trajectory_snapshot)
    if residual_core.state_network is None:
        return
    state_snapshot = snapshot.get("state")
    if state_snapshot is None:
        raise ValueError("Residual-core state snapshot is required for two-branch mode.")
    _restore_mlp_parameters(residual_core.state_network, state_snapshot)


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


def alpha_for_epoch(config: FMPCEFExploratoryProbeConfig, epoch_index: int) -> float:
    if not config.use_trajectory_curriculum_contract:
        return 1.0
    if epoch_index < config.alpha_warmup_epochs:
        return 1.0
    if config.alpha_ramp_epochs <= 0:
        return float(config.alpha_floor)
    progress = (epoch_index - config.alpha_warmup_epochs + 1) / float(config.alpha_ramp_epochs)
    mix = _sigmoid_unit_interval(progress)
    return 1.0 - ((1.0 - float(config.alpha_floor)) * mix)


def lambda_traj_curr_for_epoch(config: FMPCEFExploratoryProbeConfig, epoch_index: int) -> float:
    if not config.use_trajectory_curriculum_contract:
        return 0.0
    if config.lambda_traj_curr <= 0.0:
        return 0.0
    if epoch_index < config.alpha_warmup_epochs:
        return 0.0
    if config.alpha_ramp_epochs <= 0:
        return float(config.lambda_traj_curr)
    progress = (epoch_index - config.alpha_warmup_epochs + 1) / float(config.alpha_ramp_epochs)
    return float(config.lambda_traj_curr) * _sigmoid_unit_interval(progress)


def lambda_sg_for_epoch(config: FMPCEFExploratoryProbeConfig, epoch_index: int) -> float:
    if not config.use_endpoint_semigroup_consistency_probe:
        return 0.0
    if config.lambda_sg <= 0.0:
        return 0.0
    alpha = alpha_for_epoch(config, epoch_index)
    if alpha >= 1.0 - 1e-12:
        return 0.0
    return float(config.lambda_sg)


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
    psi_network: Stage05ResidualCoreNetworks,
    z_t: np.ndarray,
    target_onehot: np.ndarray,
    *,
    t: float,
    r: float,
    tangent_epsilon: float,
    feature_aware_state_branch_tangents: bool = False,
) -> CorrectedResidualIdentityTarget:
    """Return the Stage 05 corrected residual identity target.

    The v1 target is `m_id = r * D_T g_t + r * D_T m_psi`.
    The v2 target is `m_id = r * D_T g_t + r * D_T m_traj + r * D_T m_state`.
    """

    validate_tf1_time_pair(t, r)
    z_array = _as_batch_first("z_t", z_t)
    target_array = _as_batch_first("target_onehot", target_onehot)
    if z_array.shape[0] != target_array.shape[0]:
        raise ValueError("z_t and target_onehot must share the same batch size.")
    g_t = hidden_local_flow(context, z_array)
    trajectory_input = build_exploratory_probe_input(z_array, target_array, t=t, r=r)
    trajectory_input_tangent = build_residual_input_tangent(g_t, target_dim=target_array.shape[1])
    trajectory_jvp = forward_tf1_mlp_with_jvp(
        psi_network.trajectory_network,
        trajectory_input,
        trajectory_input_tangent,
    ).jvp
    anchor_derivative = approximate_anchor_directional_derivative(
        context,
        z_array,
        epsilon=tangent_epsilon,
    )
    state_jvp = np.zeros_like(trajectory_jvp)
    if psi_network.state_network is not None:
        features = teacher_free_state_features(context, z_array)
        feature_tangents = None
        if feature_aware_state_branch_tangents:
            feature_tangents = teacher_free_feature_tangents(
                context,
                z_array,
                epsilon=tangent_epsilon,
            )
        state_input = build_state_branch_input(features)
        state_input_tangent = build_state_branch_input_tangent(
            features,
            feature_aware_state_branch_tangents=feature_aware_state_branch_tangents,
            feature_tangents=feature_tangents,
        )
        state_jvp = forward_tf1_mlp_with_jvp(
            psi_network.state_network,
            state_input,
            state_input_tangent,
        ).jvp
    residual_jvp = trajectory_jvp + state_jvp
    anchor_term = float(r) * anchor_derivative
    trajectory_term = float(r) * trajectory_jvp
    state_term = float(r) * state_jvp
    residual_term = trajectory_term + state_term
    return CorrectedResidualIdentityTarget(
        target=anchor_term + residual_term,
        anchor_derivative=anchor_derivative,
        residual_jvp=residual_jvp,
        anchor_term=anchor_term,
        residual_term=residual_term,
        trajectory_jvp=trajectory_jvp,
        trajectory_term=trajectory_term,
        state_jvp=state_jvp,
        state_term=state_term,
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
            "candidate_name": _candidate_name(config),
            "teacher_free": True,
            "uses_teacher_artifacts": False,
            "transport_family": _transport_family_name(config),
            "residual_identity_mode": config.identity_mode,
            "energy_substrate": "baseline_pc_energy",
            "local_flow_definition": "exact_negative_hidden_state_gradient",
            "direct_anchor_source": "self_bootstrap_local_field",
            "psi_family": _psi_family_name(config),
            "velocity_parameterization": _velocity_parameterization(config),
            "u_psi_input_contract": _u_psi_input_contract(config),
            "residual_branch_structure": _residual_branch_structure(config),
            "m_traj_input_contract": M_TRAJ_INPUT_CONTRACT,
            "m_state_input_contract": (
                M_STATE_INPUT_CONTRACT if config.use_two_branch_residual_core else None
            ),
            "bootstrap_target_contract": BOOTSTRAP_TARGET_CONTRACT,
            "explicit_transport_drift_target_contract": _explicit_transport_drift_target_contract(
                config
            ),
            "trajectory_curriculum_target_contract": _trajectory_curriculum_target_contract(config),
            "semigroup_target_contract": _semigroup_target_contract(config),
            "residual_identity_target_contract": _residual_identity_target_contract(config),
            "identity_loss_weight": float(config.identity_loss_weight),
            "lambda_drift": float(config.lambda_drift),
            "lambda_traj_curr": float(config.lambda_traj_curr),
            "lambda_sg": float(config.lambda_sg),
            "tangent_epsilon": float(config.tangent_epsilon),
            "lambda_id_warmup_epochs": int(config.lambda_id_warmup_epochs),
            "lambda_id_ramp_epochs": int(config.lambda_id_ramp_epochs),
            "alpha_floor": (
                float(config.alpha_floor) if config.use_trajectory_curriculum_contract else None
            ),
            "alpha_warmup_epochs": int(config.alpha_warmup_epochs),
            "alpha_ramp_epochs": int(config.alpha_ramp_epochs),
            "trajectory_curriculum_enabled": bool(config.use_trajectory_curriculum_contract),
            "trajectory_curriculum_schedule_identity": (
                config.trajectory_curriculum_schedule
                if config.use_trajectory_curriculum_contract
                else None
            ),
            "endpoint_semigroup_consistency_enabled": bool(
                config.use_endpoint_semigroup_consistency_probe
            ),
            "semigroup_split_identity": (
                SEMIGROUP_SPLIT_IDENTITY
                if config.use_endpoint_semigroup_consistency_probe
                else None
            ),
            "semigroup_target_mode": (
                config.semigroup_target_mode if config.use_endpoint_semigroup_consistency_probe else None
            ),
            "semigroup_target_is_single_sided_detached": bool(
                config.use_endpoint_semigroup_consistency_probe
            ),
            "semigroup_update_proxy_contract": (
                SEMIGROUP_UPDATE_PROXY_CONTRACT
                if config.use_endpoint_semigroup_consistency_probe
                else None
            ),
            "no_teacher_dependency_in_target_construction": True,
            "use_teacher_free_features": bool(config.use_two_branch_residual_core),
            "uses_current_state_features": bool(config.use_two_branch_residual_core),
            "use_two_branch_residual_core": bool(config.use_two_branch_residual_core),
            "explicit_transport_drift_decomposition_enabled": bool(
                config.use_explicit_transport_drift_decomposition
            ),
            "pairwise_deltas_vs_stage05_v3a_reference": _pairwise_v3a_placeholder(config),
            "pairwise_deltas_vs_promoted_refined_v3b_reference": _pairwise_promoted_v3b_placeholder(
                config
            ),
            "feature_aware_state_branch_tangents": bool(
                config.feature_aware_state_branch_tangents
            ),
            "transport_scope": "train_only",
            "transport_steps": int(config.transport_steps),
            "bootstrap_integrator": config.bootstrap_integrator,
            "bootstrap_substeps": int(config.bootstrap_substeps),
            "selection_metric": config.selection_metric,
            "selection_metric_source": "val_metric",
            "report_metric_source": "test_metric",
            "acceptance_contract": "mechanism_first",
            "pairwise_deltas_vs_stage05_v2_reference": _pairwise_v2_placeholder(config),
            "gap_closure_decision": _gap_closure_decision_placeholder(config),
            "recommended_next_move": _recommended_next_move(config),
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
    psi_network: Stage05ResidualCoreNetworks,
    config: FMPCEFExploratoryProbeConfig,
    *,
    z_knots: list[np.ndarray],
    knot_times: np.ndarray,
    trajectory_alpha: float | None = None,
) -> ResidualSupervisionBatch:
    trajectory_input_blocks: list[np.ndarray] = []
    state_input_blocks: list[np.ndarray] = []
    boot_targets: list[np.ndarray] = []
    identity_targets: list[np.ndarray] = []
    gbar_boot_blocks: list[np.ndarray] = []
    transport_targets: list[np.ndarray] = []
    drift_targets: list[np.ndarray] = []
    trajectory_curriculum_targets: list[np.ndarray] = []
    semigroup_targets: list[np.ndarray] = []
    semigroup_loss_weights: list[np.ndarray] = []
    trajectory_curriculum_active = bool(
        config.use_trajectory_curriculum_contract
        and trajectory_alpha is not None
        and float(trajectory_alpha) < 1.0 - 1e-12
    )
    semigroup_active = bool(
        config.use_endpoint_semigroup_consistency_probe and trajectory_curriculum_active
    )
    for knot_index, t_k in enumerate(knot_times[:-1]):
        z_t = z_knots[knot_index]
        t_float = float(t_k)
        r_k = 1.0 - t_float
        trajectory_input, state_input, _ = _residual_core_inputs_for_state(
            context,
            config,
            z_t,
            context.targets,
            t=t_float,
            r=r_k,
        )
        if config.use_explicit_transport_drift_decomposition:
            decomposed_bootstrap = build_explicit_transport_drift_bootstrap_targets(
                context,
                z_t,
                t=t_float,
                r=r_k,
                integrator=config.bootstrap_integrator,
                substeps=config.bootstrap_substeps,
            )
            u_boot = decomposed_bootstrap.u_boot
            g_t = decomposed_bootstrap.g_t
            gbar_boot_blocks.append(decomposed_bootstrap.gbar_boot)
            transport_targets.append(decomposed_bootstrap.transport_target)
            drift_targets.append(decomposed_bootstrap.drift_target)
            boot_targets.append(decomposed_bootstrap.residual_target)
        else:
            u_boot = bootstrap_average_velocity_target(
                context,
                z_t,
                t=t_float,
                r=r_k,
                integrator=config.bootstrap_integrator,
                substeps=config.bootstrap_substeps,
            )
            g_t = hidden_local_flow(context, z_t)
            boot_targets.append(u_boot - g_t)
        corrected_identity = build_corrected_residual_identity_target(
            context,
            psi_network,
            z_t,
            context.targets,
            t=t_float,
            r=r_k,
            tangent_epsilon=config.tangent_epsilon,
            feature_aware_state_branch_tangents=config.feature_aware_state_branch_tangents,
        )
        trajectory_input_blocks.append(trajectory_input)
        if state_input is not None:
            state_input_blocks.append(state_input)
        identity_targets.append(corrected_identity.target)
        if trajectory_curriculum_active:
            trajectory_targets = build_trajectory_curriculum_targets(
                context,
                psi_network,
                config,
                z_t,
                t=t_float,
                r=r_k,
                alpha=float(trajectory_alpha),
            )
            trajectory_curriculum_targets.append(trajectory_targets.residual_target)
        if semigroup_active:
            semigroup_probe = build_endpoint_semigroup_targets(
                context,
                psi_network,
                config,
                z_t,
                t=t_float,
                r=r_k,
                alpha=float(trajectory_alpha),
            )
            semigroup_targets.append(semigroup_probe.residual_target)
            semigroup_loss_weights.append(semigroup_probe.loss_weights)
    return ResidualSupervisionBatch(
        trajectory_inputs=np.concatenate(trajectory_input_blocks, axis=0).astype(
            np.float64,
            copy=False,
        ),
        state_inputs=(
            np.concatenate(state_input_blocks, axis=0).astype(np.float64, copy=False)
            if state_input_blocks
            else None
        ),
        boot_targets=np.concatenate(boot_targets, axis=0).astype(np.float64, copy=False),
        identity_targets=np.concatenate(identity_targets, axis=0).astype(np.float64, copy=False),
        gbar_boot=(
            np.concatenate(gbar_boot_blocks, axis=0).astype(np.float64, copy=False)
            if gbar_boot_blocks
            else None
        ),
        transport_targets=(
            np.concatenate(transport_targets, axis=0).astype(np.float64, copy=False)
            if transport_targets
            else None
        ),
        drift_targets=(
            np.concatenate(drift_targets, axis=0).astype(np.float64, copy=False)
            if drift_targets
            else None
        ),
        trajectory_curriculum_targets=(
            np.concatenate(trajectory_curriculum_targets, axis=0).astype(np.float64, copy=False)
            if trajectory_curriculum_targets
            else None
        ),
        semigroup_targets=(
            np.concatenate(semigroup_targets, axis=0).astype(np.float64, copy=False)
            if semigroup_targets
            else None
        ),
        semigroup_loss_weights=(
            np.concatenate(semigroup_loss_weights, axis=0).astype(np.float64, copy=False)
            if semigroup_loss_weights
            else None
        ),
        explicit_transport_drift_decomposition_enabled=bool(
            config.use_explicit_transport_drift_decomposition
        ),
        trajectory_curriculum_enabled=trajectory_curriculum_active,
        endpoint_semigroup_consistency_enabled=semigroup_active,
    )


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
        velocity_fn=_learned_velocity_fn(context, psi_network, config),
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
    psi_network: Stage05ResidualCoreNetworks,
    config: FMPCEFExploratoryProbeConfig,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    *,
    lambda_id: float,
    lambda_traj_curr: float,
    lambda_sg: float,
    trajectory_alpha: float,
    stage: ProbeStage,
) -> tuple[float, float, float, float, float, float, float, float]:
    context = build_tf1_context(model, x_batch, y_batch)
    source_rollout = rollout_hidden_transport(
        context,
        context.z0,
        transport_steps=config.transport_steps,
        mode="local_field_only",
    )
    supervision = _collect_residual_supervision(
        context,
        psi_network,
        config,
        z_knots=source_rollout.z_knots,
        knot_times=source_rollout.knot_times,
        trajectory_alpha=trajectory_alpha,
    )
    predictions = _predict_residual_from_inputs(
        psi_network,
        supervision.trajectory_inputs,
        state_inputs=supervision.state_inputs,
    )
    psi_predictions = predictions.total_residual
    transport_loss = float(np.mean((psi_predictions - supervision.boot_targets) ** 2))
    drift_loss = 0.0
    boot_loss = transport_loss
    identity_loss = float(np.mean((psi_predictions - supervision.identity_targets) ** 2))
    traj_curr_loss = 0.0
    semigroup_loss = 0.0
    if config.use_explicit_transport_drift_decomposition:
        if supervision.state_inputs is None:
            raise ValueError("Stage 05 v3-A requires state_inputs for the drift branch.")
        if supervision.transport_targets is None or supervision.drift_targets is None:
            raise ValueError(
                "Stage 05 v3-A supervision requires explicit transport and drift targets."
            )
        transport_loss = float(
            np.mean((predictions.trajectory_residual - supervision.transport_targets) ** 2)
        )
        drift_loss = float(np.mean((predictions.state_residual - supervision.drift_targets) ** 2))
        boot_loss = transport_loss + (float(config.lambda_drift) * drift_loss)
        total_loss = boot_loss + (lambda_id * identity_loss)
        _weighted_explicit_transport_drift_step(
            psi_network,
            supervision.trajectory_inputs,
            supervision.state_inputs,
            supervision.transport_targets,
            supervision.drift_targets,
            supervision.identity_targets,
            lambda_drift=float(config.lambda_drift),
            lambda_id=float(lambda_id),
        )
        if (
            config.use_trajectory_curriculum_contract
            and lambda_traj_curr > 0.0
            and supervision.trajectory_curriculum_targets is not None
            and supervision.state_inputs is not None
        ):
            traj_curr_loss = float(
                np.mean(
                    (psi_predictions - supervision.trajectory_curriculum_targets) ** 2
                )
            )
            total_loss += float(lambda_traj_curr) * traj_curr_loss
            _weighted_two_branch_mse_step(
                psi_network,
                supervision.trajectory_inputs,
                supervision.state_inputs,
                supervision.trajectory_curriculum_targets,
                loss_scale=float(lambda_traj_curr),
            )
        if (
            config.use_endpoint_semigroup_consistency_probe
            and lambda_sg > 0.0
            and supervision.semigroup_targets is not None
            and supervision.semigroup_loss_weights is not None
            and supervision.state_inputs is not None
        ):
            semigroup_loss = float(
                np.mean((psi_predictions - supervision.semigroup_targets) ** 2)
            )
            total_loss += float(lambda_sg) * semigroup_loss
            _weighted_two_branch_mse_step(
                psi_network,
                supervision.trajectory_inputs,
                supervision.state_inputs,
                supervision.semigroup_targets,
                loss_scale=float(lambda_sg),
                sample_weights=supervision.semigroup_loss_weights,
            )
    else:
        if lambda_id > 0.0:
            combined_target = (
                supervision.boot_targets + (lambda_id * supervision.identity_targets)
            ) / (1.0 + lambda_id)
            loss_scale = 1.0 + lambda_id
        else:
            combined_target = supervision.boot_targets
            loss_scale = 1.0
        total_loss = boot_loss + (lambda_id * identity_loss)
        if supervision.state_inputs is None:
            _weighted_mse_step(
                psi_network.trajectory_network,
                supervision.trajectory_inputs,
                combined_target,
                loss_scale=loss_scale,
            )
        else:
            _weighted_two_branch_mse_step(
                psi_network,
                supervision.trajectory_inputs,
                supervision.state_inputs,
                combined_target,
                loss_scale=loss_scale,
            )

    if stage == "warmup":
        theta_rollout = source_rollout
    else:
        theta_rollout = rollout_hidden_transport(
            context,
            context.z0,
            transport_steps=config.transport_steps,
            mode="learned",
            velocity_fn=_learned_velocity_fn(context, psi_network, config),
        )

    transported_energy = _theta_update_from_transported_state(
        model,
        context,
        theta_rollout.z_knots[-1],
    )
    return (
        total_loss,
        boot_loss,
        transport_loss,
        drift_loss,
        identity_loss,
        traj_curr_loss,
        semigroup_loss,
        transported_energy,
    )


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
        alpha = alpha_for_epoch(config, epoch_index)
        lambda_traj_curr = lambda_traj_curr_for_epoch(config, epoch_index)
        lambda_sg = lambda_sg_for_epoch(config, epoch_index)
        batch_total_losses: list[float] = []
        batch_boot_losses: list[float] = []
        batch_transport_losses: list[float] = []
        batch_drift_losses: list[float] = []
        batch_identity_losses: list[float] = []
        batch_traj_curr_losses: list[float] = []
        batch_semigroup_losses: list[float] = []
        batch_transport_energies: list[float] = []
        batch_seed = config.batch_order_seed + epoch_index
        for x_batch, y_batch in iter_minibatches(
            split.x_train,
            split.y_train,
            config.batch_size,
            shuffle=config.shuffle_batches,
            seed=batch_seed,
        ):
            (
                total_loss,
                boot_loss,
                transport_loss,
                drift_loss,
                identity_loss,
                traj_curr_loss,
                semigroup_loss,
                transported_energy,
            ) = _train_one_batch(
                model,
                psi_network,
                config,
                x_batch,
                y_batch,
                lambda_id=lambda_id,
                lambda_traj_curr=lambda_traj_curr,
                lambda_sg=lambda_sg,
                trajectory_alpha=alpha,
                stage=stage,
            )
            batch_total_losses.append(total_loss)
            batch_boot_losses.append(boot_loss)
            batch_transport_losses.append(transport_loss)
            batch_drift_losses.append(drift_loss)
            batch_identity_losses.append(identity_loss)
            batch_traj_curr_losses.append(traj_curr_loss)
            batch_semigroup_losses.append(semigroup_loss)
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
                alpha=float(alpha),
                lambda_traj_curr=float(lambda_traj_curr),
                lambda_sg=float(lambda_sg),
                train_total_loss=float(np.mean(batch_total_losses)),
                train_boot_loss=float(np.mean(batch_boot_losses)),
                train_transport_loss=float(np.mean(batch_transport_losses)),
                train_drift_loss=float(np.mean(batch_drift_losses)),
                train_identity_loss=float(np.mean(batch_identity_losses)),
                train_traj_curr_loss=float(np.mean(batch_traj_curr_losses)),
                train_semigroup_loss=float(np.mean(batch_semigroup_losses)),
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
                psi_snapshot=_snapshot_residual_core_parameters(psi_network),
            )
        )

    train_wall_time_seconds = float(perf_counter() - train_start)

    best_row = min(epoch_rows, key=lambda row: float(row["val_configured_transported_final_energy"]))
    selected_epoch = int(best_row["epoch"])
    selected_snapshot = _snapshot_for_epoch(epoch_snapshots, selected_epoch)
    _restore_pc_parameters(model, selected_snapshot.model_snapshot)
    _restore_residual_core_parameters(psi_network, selected_snapshot.psi_snapshot)

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
        "candidate_name": _candidate_name(config),
        "teacher_free": True,
        "uses_teacher_artifacts": False,
        "transport_family": _transport_family_name(config),
        "residual_identity_mode": config.identity_mode,
        "energy_substrate": "baseline_pc_energy",
        "local_flow_definition": "exact_negative_hidden_state_gradient",
        "direct_anchor_source": "self_bootstrap_local_field",
        "psi_family": _psi_family_name(config),
        "transport_scope": "train_only",
        "transport_steps": int(config.transport_steps),
        "u_psi_input_contract": _u_psi_input_contract(config),
        "residual_branch_structure": _residual_branch_structure(config),
        "m_traj_input_contract": M_TRAJ_INPUT_CONTRACT,
        "m_state_input_contract": (
            M_STATE_INPUT_CONTRACT if config.use_two_branch_residual_core else None
        ),
        "bootstrap_target_contract": BOOTSTRAP_TARGET_CONTRACT,
        "explicit_transport_drift_target_contract": _explicit_transport_drift_target_contract(
            config
        ),
        "trajectory_curriculum_target_contract": _trajectory_curriculum_target_contract(config),
        "semigroup_target_contract": _semigroup_target_contract(config),
        "residual_identity_target_contract": _residual_identity_target_contract(config),
        "identity_loss_weight": float(config.identity_loss_weight),
        "lambda_drift": float(config.lambda_drift),
        "lambda_traj_curr": float(config.lambda_traj_curr),
        "lambda_sg": float(config.lambda_sg),
        "tangent_epsilon": float(config.tangent_epsilon),
        "lambda_id_warmup_epochs": int(config.lambda_id_warmup_epochs),
        "lambda_id_ramp_epochs": int(config.lambda_id_ramp_epochs),
        "alpha_floor": (
            float(config.alpha_floor) if config.use_trajectory_curriculum_contract else None
        ),
        "alpha_warmup_epochs": int(config.alpha_warmup_epochs),
        "alpha_ramp_epochs": int(config.alpha_ramp_epochs),
        "trajectory_curriculum_enabled": bool(config.use_trajectory_curriculum_contract),
        "trajectory_curriculum_schedule_identity": (
            config.trajectory_curriculum_schedule
            if config.use_trajectory_curriculum_contract
            else None
        ),
        "endpoint_semigroup_consistency_enabled": bool(
            config.use_endpoint_semigroup_consistency_probe
        ),
        "semigroup_split_identity": (
            SEMIGROUP_SPLIT_IDENTITY if config.use_endpoint_semigroup_consistency_probe else None
        ),
        "semigroup_target_mode": (
            config.semigroup_target_mode if config.use_endpoint_semigroup_consistency_probe else None
        ),
        "semigroup_target_is_single_sided_detached": bool(
            config.use_endpoint_semigroup_consistency_probe
        ),
        "semigroup_update_proxy_contract": (
            SEMIGROUP_UPDATE_PROXY_CONTRACT
            if config.use_endpoint_semigroup_consistency_probe
            else None
        ),
        "use_two_branch_residual_core": bool(config.use_two_branch_residual_core),
        "explicit_transport_drift_decomposition_enabled": bool(
            config.use_explicit_transport_drift_decomposition
        ),
        "pairwise_deltas_vs_stage05_v3a_reference": _pairwise_v3a_placeholder(config),
        "pairwise_deltas_vs_promoted_refined_v3b_reference": _pairwise_promoted_v3b_placeholder(
            config
        ),
        "uses_current_state_features": bool(config.use_two_branch_residual_core),
        "feature_aware_state_branch_tangents": bool(
            config.feature_aware_state_branch_tangents
        ),
        "target_construction_artifact_independent": True,
        "selection_metric_source": "val_metric",
        "report_metric_source": "test_metric",
        "selection_metric_name": config.selection_metric,
        "selection_metric_value": float(val_configured.transported_final_energy),
        "selection_metric_higher_is_better": False,
        "selected_epoch": int(selected_epoch),
        "selected_epoch_stage": str(best_row["stage"]),
        "selected_epoch_lambda_id": float(best_row["lambda_id"]),
        "selected_epoch_alpha": float(best_row["alpha"]),
        "selected_epoch_lambda_traj_curr": float(best_row["lambda_traj_curr"]),
        "selected_epoch_lambda_sg": float(best_row["lambda_sg"]),
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
        "pairwise_deltas_vs_stage05_v2_reference": _pairwise_v2_placeholder(config),
        "pairwise_deltas_vs_stage05_v3a_reference": _pairwise_v3a_placeholder(config),
        "pairwise_deltas_vs_promoted_refined_v3b_reference": _pairwise_promoted_v3b_placeholder(
            config
        ),
        "gap_closure_decision": _gap_closure_decision_placeholder(config),
        "recommended_next_move": _recommended_next_move(config),
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
