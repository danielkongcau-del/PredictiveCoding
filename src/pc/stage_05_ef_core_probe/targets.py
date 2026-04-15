
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..stage_03_transport_core_v1.fmpc_tf1_flow import (
    FMPCTF1Context,
    bootstrap_average_velocity_target,
    hidden_local_flow,
    teacher_free_feature_tangents,
    teacher_free_state_features,
    validate_tf1_time_pair,
)
from ..stage_03_transport_core_v1.fmpc_tf1_jvp import forward_tf1_mlp_with_jvp
from ..utils import ensure_finite_array
from .common import _as_batch_first
from .configs import FMPCEFExploratoryProbeConfig
from .residual_core import (
    Stage05ResidualCoreNetworks,
    _predict_total_velocity_at_state,
    build_exploratory_probe_input,
    build_residual_input_tangent,
    build_state_branch_input,
    build_state_branch_input_tangent,
)

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
class FusedTrajectorySemigroupTargets:
    """Exact detached-target fusion of the current trajectory and semigroup residual targets."""

    trajectory_residual_target: np.ndarray
    semigroup_residual_target: np.ndarray
    fused_residual_target: np.ndarray
    fusion_weights: np.ndarray
    fusion_rho: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "trajectory_residual_target",
            _as_batch_first("trajectory_residual_target", self.trajectory_residual_target),
        )
        object.__setattr__(
            self,
            "semigroup_residual_target",
            _as_batch_first("semigroup_residual_target", self.semigroup_residual_target),
        )
        object.__setattr__(
            self,
            "fused_residual_target",
            _as_batch_first("fused_residual_target", self.fused_residual_target),
        )
        object.__setattr__(self, "fusion_weights", _as_batch_first("fusion_weights", self.fusion_weights))
        object.__setattr__(self, "fusion_rho", _as_batch_first("fusion_rho", self.fusion_rho))
        reference_shape = self.trajectory_residual_target.shape
        for name in ("semigroup_residual_target", "fused_residual_target"):
            if getattr(self, name).shape != reference_shape:
                raise ValueError(
                    f"{name} must share the same shape as trajectory_residual_target."
                )
        if self.fusion_weights.shape != (reference_shape[0], 1):
            raise ValueError("fusion_weights must be shaped (batch, 1).")
        if self.fusion_rho.shape != (reference_shape[0], 1):
            raise ValueError("fusion_rho must be shaped (batch, 1).")

@dataclass(frozen=True)
class MidpointReconstructedTrajectoryTargets:
    """Non-equivalent internal-knot trajectory targets for Stage 05 v3-C refinements."""

    bootstrap_midpoint_state: np.ndarray
    bootstrap_continuation_velocity: np.ndarray
    semigroup_implied_midpoint_state: np.ndarray
    reconstructed_midpoint_state: np.ndarray
    reconstructed_short_velocity: np.ndarray
    reevaluated_continuation_velocity: np.ndarray
    endpoint_implied_continuation_velocity: np.ndarray
    blended_continuation_velocity: np.ndarray
    unified_velocity_target: np.ndarray
    unified_residual_target: np.ndarray
    main_loss_weights: np.ndarray
    reconstruction_weight: np.ndarray
    midpoint_reconstruction_shift_norm: np.ndarray
    continuation_reevaluation_shift_norm: np.ndarray
    continuation_target_blend_shift_norm: np.ndarray
    semigroup_endpoint_closure_residual_norm: np.ndarray
    predictor_semigroup_defect_norm: np.ndarray
    corrected_semigroup_defect_norm: np.ndarray
    first_projection_short_leg_correction_norm: np.ndarray
    first_projection_continuation_correction_norm: np.ndarray
    second_projection_short_leg_correction_norm: np.ndarray
    second_projection_continuation_correction_norm: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "bootstrap_midpoint_state",
            "bootstrap_continuation_velocity",
            "semigroup_implied_midpoint_state",
            "reconstructed_midpoint_state",
            "reconstructed_short_velocity",
            "reevaluated_continuation_velocity",
            "endpoint_implied_continuation_velocity",
            "blended_continuation_velocity",
            "unified_velocity_target",
            "unified_residual_target",
        ):
            object.__setattr__(self, name, _as_batch_first(name, getattr(self, name)))
        for name in (
            "main_loss_weights",
            "reconstruction_weight",
            "midpoint_reconstruction_shift_norm",
            "continuation_reevaluation_shift_norm",
            "continuation_target_blend_shift_norm",
            "semigroup_endpoint_closure_residual_norm",
            "predictor_semigroup_defect_norm",
            "corrected_semigroup_defect_norm",
            "first_projection_short_leg_correction_norm",
            "first_projection_continuation_correction_norm",
            "second_projection_short_leg_correction_norm",
            "second_projection_continuation_correction_norm",
        ):
            object.__setattr__(self, name, _as_batch_first(name, getattr(self, name)))
        reference_shape = self.bootstrap_midpoint_state.shape
        for name in (
            "bootstrap_continuation_velocity",
            "semigroup_implied_midpoint_state",
            "reconstructed_midpoint_state",
            "reconstructed_short_velocity",
            "reevaluated_continuation_velocity",
            "endpoint_implied_continuation_velocity",
            "blended_continuation_velocity",
            "unified_velocity_target",
            "unified_residual_target",
        ):
            if getattr(self, name).shape != reference_shape:
                raise ValueError(
                    f"{name} must share the same shape as bootstrap_midpoint_state."
                )
        for name in (
            "main_loss_weights",
            "reconstruction_weight",
            "midpoint_reconstruction_shift_norm",
            "continuation_reevaluation_shift_norm",
            "continuation_target_blend_shift_norm",
            "semigroup_endpoint_closure_residual_norm",
            "predictor_semigroup_defect_norm",
            "corrected_semigroup_defect_norm",
            "first_projection_short_leg_correction_norm",
            "first_projection_continuation_correction_norm",
            "second_projection_short_leg_correction_norm",
            "second_projection_continuation_correction_norm",
        ):
            if getattr(self, name).shape != (reference_shape[0], 1):
                raise ValueError(f"{name} must be shaped (batch, 1).")

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
    midpoint_reconstructed_main_targets: np.ndarray | None = None
    midpoint_reconstructed_main_loss_weights: np.ndarray | None = None
    midpoint_reconstruction_shift_norms: np.ndarray | None = None
    continuation_reevaluation_shift_norms: np.ndarray | None = None
    continuation_target_blend_shift_norms: np.ndarray | None = None
    semigroup_endpoint_closure_residual_norms: np.ndarray | None = None
    predictor_semigroup_defect_norms: np.ndarray | None = None
    corrected_semigroup_defect_norms: np.ndarray | None = None
    first_projection_short_leg_correction_norms: np.ndarray | None = None
    first_projection_continuation_correction_norms: np.ndarray | None = None
    second_projection_short_leg_correction_norms: np.ndarray | None = None
    second_projection_continuation_correction_norms: np.ndarray | None = None
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
        if self.midpoint_reconstructed_main_targets is not None:
            object.__setattr__(
                self,
                "midpoint_reconstructed_main_targets",
                _as_batch_first(
                    "midpoint_reconstructed_main_targets",
                    self.midpoint_reconstructed_main_targets,
                ),
            )
        if self.midpoint_reconstructed_main_loss_weights is not None:
            object.__setattr__(
                self,
                "midpoint_reconstructed_main_loss_weights",
                _as_batch_first(
                    "midpoint_reconstructed_main_loss_weights",
                    self.midpoint_reconstructed_main_loss_weights,
                ),
            )
        if self.midpoint_reconstruction_shift_norms is not None:
            object.__setattr__(
                self,
                "midpoint_reconstruction_shift_norms",
                _as_batch_first(
                    "midpoint_reconstruction_shift_norms",
                    self.midpoint_reconstruction_shift_norms,
                ),
            )
        if self.continuation_reevaluation_shift_norms is not None:
            object.__setattr__(
                self,
                "continuation_reevaluation_shift_norms",
                _as_batch_first(
                    "continuation_reevaluation_shift_norms",
                    self.continuation_reevaluation_shift_norms,
                ),
            )
        if self.continuation_target_blend_shift_norms is not None:
            object.__setattr__(
                self,
                "continuation_target_blend_shift_norms",
                _as_batch_first(
                    "continuation_target_blend_shift_norms",
                    self.continuation_target_blend_shift_norms,
                ),
            )
        if self.semigroup_endpoint_closure_residual_norms is not None:
            object.__setattr__(
                self,
                "semigroup_endpoint_closure_residual_norms",
                _as_batch_first(
                    "semigroup_endpoint_closure_residual_norms",
                    self.semigroup_endpoint_closure_residual_norms,
                ),
            )
        for name in (
            "predictor_semigroup_defect_norms",
            "corrected_semigroup_defect_norms",
            "first_projection_short_leg_correction_norms",
            "first_projection_continuation_correction_norms",
            "second_projection_short_leg_correction_norms",
            "second_projection_continuation_correction_norms",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _as_batch_first(name, value))
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
        if (
            self.midpoint_reconstructed_main_targets is not None
            and self.midpoint_reconstructed_main_targets.shape != self.boot_targets.shape
        ):
            raise ValueError(
                "midpoint_reconstructed_main_targets must share the same shape as boot_targets."
            )
        for name in (
            "midpoint_reconstructed_main_loss_weights",
            "midpoint_reconstruction_shift_norms",
            "continuation_reevaluation_shift_norms",
            "continuation_target_blend_shift_norms",
            "semigroup_endpoint_closure_residual_norms",
            "predictor_semigroup_defect_norms",
            "corrected_semigroup_defect_norms",
            "first_projection_short_leg_correction_norms",
            "first_projection_continuation_correction_norms",
            "second_projection_short_leg_correction_norms",
            "second_projection_continuation_correction_norms",
        ):
            value = getattr(self, name)
            if value is not None and value.shape != (self.boot_targets.shape[0], 1):
                raise ValueError(f"{name} must be shaped (batch, 1).")

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

def build_fused_trajectory_semigroup_targets(
    trajectory_residual_target: np.ndarray,
    semigroup_residual_target: np.ndarray,
    semigroup_loss_weights: np.ndarray,
    *,
    lambda_traj_curr: float,
    lambda_sg: float,
) -> FusedTrajectorySemigroupTargets:
    """Return the exact detached-target fusion used by the consolidated v3-C candidate.

    Under the current Stage 05 detached-target construction, the stacked trajectory and
    semigroup objectives can be collapsed into one fused residual target with per-sample
    weight `W = lambda_tc + lambda_sg * r^2`.
    """

    if lambda_traj_curr < 0.0:
        raise ValueError("lambda_traj_curr must be non-negative.")
    if lambda_sg < 0.0:
        raise ValueError("lambda_sg must be non-negative.")
    trajectory_target = _as_batch_first("trajectory_residual_target", trajectory_residual_target)
    semigroup_target = _as_batch_first("semigroup_residual_target", semigroup_residual_target)
    loss_weights = _as_batch_first("semigroup_loss_weights", semigroup_loss_weights)
    if trajectory_target.shape != semigroup_target.shape:
        raise ValueError(
            "trajectory_residual_target and semigroup_residual_target must share the same shape."
        )
    if loss_weights.shape != (trajectory_target.shape[0], 1):
        raise ValueError("semigroup_loss_weights must be shaped (batch, 1).")

    fused_weights = float(lambda_traj_curr) + (float(lambda_sg) * loss_weights)
    if np.any(fused_weights <= 0.0):
        raise ValueError("Fused trajectory-semigroup weights must remain strictly positive.")
    fused_rho = (float(lambda_sg) * loss_weights) / fused_weights
    fused_target = (
        (float(lambda_traj_curr) * trajectory_target)
        + (float(lambda_sg) * loss_weights * semigroup_target)
    ) / fused_weights
    return FusedTrajectorySemigroupTargets(
        trajectory_residual_target=trajectory_target,
        semigroup_residual_target=semigroup_target,
        fused_residual_target=fused_target,
        fusion_weights=fused_weights,
        fusion_rho=fused_rho,
    )

def build_midpoint_reconstructed_trajectory_targets(
    context: FMPCTF1Context,
    psi_network: Stage05ResidualCoreNetworks,
    config: FMPCEFExploratoryProbeConfig,
    z_t: np.ndarray,
    *,
    t: float,
    r: float,
    alpha: float,
    lambda_traj_curr: float,
    lambda_sg: float,
) -> MidpointReconstructedTrajectoryTargets:
    """Return the active internal-knot reconstruction target family for the current v3-C line."""

    validate_tf1_time_pair(t, r)
    if not config.use_midpoint_reconstructed_trajectory_contract:
        raise ValueError(
            "Midpoint-reconstructed targets require "
            "use_midpoint_reconstructed_trajectory_contract=True."
        )
    alpha_value = float(alpha)
    if not (0.0 < alpha_value <= 1.0):
        raise ValueError("Midpoint-reconstructed targets require 0 < alpha <= 1.")
    if lambda_traj_curr < 0.0:
        raise ValueError("lambda_traj_curr must be non-negative.")
    if lambda_sg < 0.0:
        raise ValueError("lambda_sg must be non-negative.")

    z_array = _as_batch_first("z_t", z_t)
    short_r = alpha_value * float(r)
    if short_r <= 1e-12:
        raise ValueError("Midpoint reconstruction requires an active short horizon.")
    split_time = float(t) + short_r
    continuation_r = (1.0 - alpha_value) * float(r)
    continuation_active = continuation_r > 1e-12
    validate_tf1_time_pair(t, short_r)
    if continuation_active:
        validate_tf1_time_pair(split_time, continuation_r)

    u_boot_short = bootstrap_average_velocity_target(
        context,
        z_array,
        t=t,
        r=short_r,
        integrator=config.bootstrap_integrator,
        substeps=config.bootstrap_substeps,
    )
    z_boot_mid = z_array + short_r * u_boot_short
    if continuation_active:
        u_cont_boot = _predict_total_velocity_at_state(
            context,
            psi_network,
            config,
            z_boot_mid,
            t=split_time,
            r=continuation_r,
        )
    else:
        u_cont_boot = np.zeros_like(z_array)

    short_horizon_velocity = _predict_total_velocity_at_state(
        context,
        psi_network,
        config,
        z_array,
        t=t,
        r=short_r,
    )
    z_hat_mid = z_array + short_r * short_horizon_velocity
    if continuation_active:
        split_continuation_velocity = _predict_total_velocity_at_state(
            context,
            psi_network,
            config,
            z_hat_mid,
            t=split_time,
            r=continuation_r,
        )
    else:
        split_continuation_velocity = np.zeros_like(z_array)
    z_sg_star = (
        z_array
        + short_r * short_horizon_velocity
        + continuation_r * split_continuation_velocity
    ).copy()
    u_sg_star = (z_sg_star - z_array) / float(r)

    loss_weights = np.full((z_array.shape[0], 1), float(r) ** 2, dtype=np.float64)
    w_main = float(lambda_traj_curr) + (float(lambda_sg) * loss_weights)
    if np.any(w_main <= 0.0):
        raise ValueError("Midpoint reconstruction requires strictly positive main weights.")
    kappa = (float(lambda_sg) * loss_weights) / w_main
    zero_scalar = np.zeros((z_array.shape[0], 1), dtype=np.float64)

    if config.use_endpoint_line_midpoint_trajectory_contract:
        z_guidance_mid_star = z_array + (short_r * u_sg_star)
    else:
        z_guidance_mid_star = z_sg_star - continuation_r * u_cont_boot
    z_mid_star_0 = ((1.0 - kappa) * z_boot_mid) + (kappa * z_guidance_mid_star)
    u_short_star_0 = (z_mid_star_0 - z_array) / short_r

    if continuation_active:
        u_cont_traj_star_0 = _predict_total_velocity_at_state(
            context,
            psi_network,
            config,
            z_mid_star_0,
            t=split_time,
            r=continuation_r,
        )
    else:
        u_cont_traj_star_0 = np.zeros_like(z_array)

    predictor_semigroup_defect_norm = zero_scalar.copy()
    corrected_semigroup_defect_norm = zero_scalar.copy()
    first_projection_short_leg_correction_norm = zero_scalar.copy()
    first_projection_continuation_correction_norm = zero_scalar.copy()
    second_projection_short_leg_correction_norm = zero_scalar.copy()
    second_projection_continuation_correction_norm = zero_scalar.copy()
    u_main_star: np.ndarray | None = None

    if config.use_coupled_defect_projection_trajectory_contract:
        rho_denom = float(lambda_traj_curr) + (
            float(lambda_sg)
            * loss_weights
            * ((alpha_value ** 2) + ((1.0 - alpha_value) ** 2))
        )
        if np.any(rho_denom <= 0.0):
            raise ValueError("Coupled defect projection requires strictly positive rho denominator.")
        rho = (float(lambda_sg) * loss_weights) / rho_denom
        d_sg_0 = u_sg_star - (
            (alpha_value * u_short_star_0) + ((1.0 - alpha_value) * u_cont_traj_star_0)
        )
        u_short_star_half = u_short_star_0 + (rho * alpha_value * d_sg_0)
        u_cont_star_half = u_cont_traj_star_0 + (rho * (1.0 - alpha_value) * d_sg_0)
        z_mid_star_half = z_array + (short_r * u_short_star_half)
        if continuation_active:
            u_cont_traj_star_1 = _predict_total_velocity_at_state(
                context,
                psi_network,
                config,
                z_mid_star_half,
                t=split_time,
                r=continuation_r,
            )
        else:
            u_cont_traj_star_1 = np.zeros_like(z_array)
        d_sg_1 = u_sg_star - (
            (alpha_value * u_short_star_half) + ((1.0 - alpha_value) * u_cont_traj_star_1)
        )
        u_short_star = u_short_star_half + (rho * alpha_value * d_sg_1)
        u_cont_star = u_cont_traj_star_1 + (rho * (1.0 - alpha_value) * d_sg_1)
        z_mid_star = z_array + (short_r * u_short_star)
        u_cont_sg_star = u_cont_star.copy()
        u_cont_traj_star = u_cont_traj_star_1
        predictor_semigroup_defect_norm = np.linalg.norm(d_sg_0, axis=1, keepdims=True)
        corrected_semigroup_defect_norm = np.linalg.norm(d_sg_1, axis=1, keepdims=True)
        first_projection_short_leg_correction_norm = np.linalg.norm(
            u_short_star_half - u_short_star_0,
            axis=1,
            keepdims=True,
        )
        first_projection_continuation_correction_norm = np.linalg.norm(
            u_cont_star_half - u_cont_traj_star_0,
            axis=1,
            keepdims=True,
        )
        second_projection_short_leg_correction_norm = np.linalg.norm(
            u_short_star - u_short_star_half,
            axis=1,
            keepdims=True,
        )
        second_projection_continuation_correction_norm = np.linalg.norm(
            u_cont_star - u_cont_traj_star_1,
            axis=1,
            keepdims=True,
        )
    else:
        z_mid_star = z_mid_star_0
        u_short_star = u_short_star_0
        u_cont_traj_star = u_cont_traj_star_0
        if config.use_scaled_continuation_blend_trajectory_contract and continuation_active:
            kappa_eff = np.minimum(
                1.0,
                float(config.continuation_blend_scale) * kappa,
            )
            u_cont_sg_star = (z_sg_star - z_mid_star) / continuation_r
            u_cont_star = ((1.0 - kappa_eff) * u_cont_traj_star) + (kappa_eff * u_cont_sg_star)
        elif (
            config.use_precision_weighted_continuation_corrector_trajectory_contract
            and continuation_active
        ):
            eta_cont_denom = float(lambda_traj_curr) + (
                float(lambda_sg) * loss_weights * ((1.0 - alpha_value) ** 2)
            )
            if np.any(eta_cont_denom <= 0.0):
                raise ValueError(
                    "Precision-weighted continuation correction requires "
                    "strictly positive eta_cont denominator."
                )
            eta_cont = (
                float(lambda_sg) * loss_weights * ((1.0 - alpha_value) ** 2)
            ) / eta_cont_denom
            u_cont_sg_star = (z_sg_star - z_mid_star) / continuation_r
            u_cont_star = ((1.0 - eta_cont) * u_cont_traj_star) + (eta_cont * u_cont_sg_star)
            u_traj_mid_star = (alpha_value * u_short_star) + (
                (1.0 - alpha_value) * u_cont_traj_star
            )
            u_main_star = ((1.0 - eta_cont) * u_traj_mid_star) + (eta_cont * u_sg_star)
        elif config.use_endpoint_line_continuation_blend_trajectory_contract and continuation_active:
            u_cont_sg_star = (z_sg_star - z_mid_star) / continuation_r
            u_cont_star = ((1.0 - kappa) * u_cont_traj_star) + (kappa * u_cont_sg_star)
        else:
            u_cont_sg_star = u_cont_traj_star.copy()
            u_cont_star = u_cont_traj_star.copy()

    if u_main_star is None:
        u_main_star = (alpha_value * u_short_star) + ((1.0 - alpha_value) * u_cont_star)
    g_t = hidden_local_flow(context, z_array)
    m_main_star = u_main_star - g_t

    midpoint_shift_norm = np.linalg.norm(z_mid_star - z_boot_mid, axis=1, keepdims=True)
    continuation_shift_norm = np.linalg.norm(
        u_cont_traj_star - u_cont_boot,
        axis=1,
        keepdims=True,
    )
    continuation_target_blend_shift_norm = np.linalg.norm(
        u_cont_star - u_cont_traj_star,
        axis=1,
        keepdims=True,
    )
    semigroup_endpoint_closure_residual_norm = np.linalg.norm(
        z_sg_star - (z_array + float(r) * u_main_star),
        axis=1,
        keepdims=True,
    )
    return MidpointReconstructedTrajectoryTargets(
        bootstrap_midpoint_state=z_boot_mid,
        bootstrap_continuation_velocity=u_cont_boot,
        semigroup_implied_midpoint_state=z_guidance_mid_star,
        reconstructed_midpoint_state=z_mid_star,
        reconstructed_short_velocity=u_short_star,
        reevaluated_continuation_velocity=u_cont_traj_star,
        endpoint_implied_continuation_velocity=u_cont_sg_star,
        blended_continuation_velocity=u_cont_star,
        unified_velocity_target=u_main_star,
        unified_residual_target=m_main_star,
        main_loss_weights=w_main,
        reconstruction_weight=kappa,
        midpoint_reconstruction_shift_norm=midpoint_shift_norm,
        continuation_reevaluation_shift_norm=continuation_shift_norm,
        continuation_target_blend_shift_norm=continuation_target_blend_shift_norm,
        semigroup_endpoint_closure_residual_norm=semigroup_endpoint_closure_residual_norm,
        predictor_semigroup_defect_norm=predictor_semigroup_defect_norm,
        corrected_semigroup_defect_norm=corrected_semigroup_defect_norm,
        first_projection_short_leg_correction_norm=first_projection_short_leg_correction_norm,
        first_projection_continuation_correction_norm=first_projection_continuation_correction_norm,
        second_projection_short_leg_correction_norm=second_projection_short_leg_correction_norm,
        second_projection_continuation_correction_norm=second_projection_continuation_correction_norm,
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
