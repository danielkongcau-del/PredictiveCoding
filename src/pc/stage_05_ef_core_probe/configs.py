
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .contracts import *

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
    use_fused_trajectory_semigroup_contract: bool = False
    use_midpoint_reconstructed_trajectory_contract: bool = False
    use_endpoint_line_midpoint_trajectory_contract: bool = False
    use_endpoint_line_continuation_blend_trajectory_contract: bool = False
    use_scaled_continuation_blend_trajectory_contract: bool = False
    use_coupled_defect_projection_trajectory_contract: bool = False
    use_precision_weighted_continuation_corrector_trajectory_contract: bool = False
    continuation_blend_scale: float = 1.0
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
        if self.continuation_blend_scale <= 0.0:
            raise ValueError("continuation_blend_scale must be positive.")
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
        if (
            self.use_fused_trajectory_semigroup_contract
            and not self.use_endpoint_semigroup_consistency_probe
        ):
            raise ValueError(
                "use_fused_trajectory_semigroup_contract requires "
                "use_endpoint_semigroup_consistency_probe=True."
            )
        if (
            self.use_midpoint_reconstructed_trajectory_contract
            and not self.use_endpoint_semigroup_consistency_probe
        ):
            raise ValueError(
                "use_midpoint_reconstructed_trajectory_contract requires "
                "use_endpoint_semigroup_consistency_probe=True."
            )
        if (
            self.use_midpoint_reconstructed_trajectory_contract
            and self.use_fused_trajectory_semigroup_contract
        ):
            raise ValueError(
                "The midpoint-reconstructed and exact-fusion v3-C contracts are mutually exclusive."
            )
        if (
            self.use_endpoint_line_midpoint_trajectory_contract
            and not self.use_midpoint_reconstructed_trajectory_contract
        ):
            raise ValueError(
                "use_endpoint_line_midpoint_trajectory_contract requires "
                "use_midpoint_reconstructed_trajectory_contract=True."
            )
        if (
            self.use_endpoint_line_continuation_blend_trajectory_contract
            and not self.use_endpoint_line_midpoint_trajectory_contract
        ):
            raise ValueError(
                "use_endpoint_line_continuation_blend_trajectory_contract requires "
                "use_endpoint_line_midpoint_trajectory_contract=True."
            )
        if (
            self.use_scaled_continuation_blend_trajectory_contract
            and not self.use_endpoint_line_midpoint_trajectory_contract
        ):
            raise ValueError(
                "use_scaled_continuation_blend_trajectory_contract requires "
                "use_endpoint_line_midpoint_trajectory_contract=True."
            )
        if (
            self.use_scaled_continuation_blend_trajectory_contract
            and self.use_endpoint_line_continuation_blend_trajectory_contract
        ):
            raise ValueError(
                "The scaled-continuation-blend and endpoint-line-continuation-blend "
                "contracts are mutually exclusive."
            )
        if (
            self.use_coupled_defect_projection_trajectory_contract
            and not self.use_endpoint_line_midpoint_trajectory_contract
        ):
            raise ValueError(
                "use_coupled_defect_projection_trajectory_contract requires "
                "use_endpoint_line_midpoint_trajectory_contract=True."
            )
        if (
            self.use_coupled_defect_projection_trajectory_contract
            and self.use_endpoint_line_continuation_blend_trajectory_contract
        ):
            raise ValueError(
                "The coupled-defect-projection and endpoint-line-continuation-blend "
                "contracts are mutually exclusive."
            )
        if (
            self.use_precision_weighted_continuation_corrector_trajectory_contract
            and not self.use_endpoint_line_midpoint_trajectory_contract
        ):
            raise ValueError(
                "use_precision_weighted_continuation_corrector_trajectory_contract requires "
                "use_endpoint_line_midpoint_trajectory_contract=True."
            )
        if (
            self.use_precision_weighted_continuation_corrector_trajectory_contract
            and self.use_endpoint_line_continuation_blend_trajectory_contract
        ):
            raise ValueError(
                "The precision-weighted-continuation-corrector and "
                "endpoint-line-continuation-blend contracts are mutually exclusive."
            )
        if (
            self.use_precision_weighted_continuation_corrector_trajectory_contract
            and self.use_coupled_defect_projection_trajectory_contract
        ):
            raise ValueError(
                "The precision-weighted-continuation-corrector and coupled-defect-projection "
                "contracts are mutually exclusive."
            )
        if (
            self.use_scaled_continuation_blend_trajectory_contract
            and self.use_coupled_defect_projection_trajectory_contract
        ):
            raise ValueError(
                "The scaled-continuation-blend and coupled-defect-projection contracts are "
                "mutually exclusive."
            )
        if (
            self.use_scaled_continuation_blend_trajectory_contract
            and self.use_precision_weighted_continuation_corrector_trajectory_contract
        ):
            raise ValueError(
                "The scaled-continuation-blend and precision-weighted-continuation-corrector "
                "contracts are mutually exclusive."
            )

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_{self.run_seed}"

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

def build_stage05_v3c_stronger_semigroup_weight_config(
    **overrides: Any,
) -> FMPCEFExploratoryProbeConfig:
    """Return a single-axis v3-C refinement that only strengthens the semigroup probe weight."""

    payload: dict[str, Any] = {
        "lambda_sg": 0.10,
        "candidate_name_override": STAGE05_V3C_STRONGER_SEMIGROUP_CANDIDATE_NAME,
    }
    payload.update(overrides)
    return build_stage05_v3c_endpoint_semigroup_config(**payload)

def build_stage05_v3c_fused_trajectory_semigroup_contract_config(
    **overrides: Any,
) -> FMPCEFExploratoryProbeConfig:
    """Return the exact-fusion v3-C contract-consolidation candidate on top of active refined v3-C."""

    payload: dict[str, Any] = {
        "use_fused_trajectory_semigroup_contract": True,
        "candidate_name_override": STAGE05_V3C_FUSED_CANDIDATE_NAME,
    }
    payload.update(overrides)
    return build_stage05_v3c_stronger_semigroup_weight_config(**payload)

def build_stage05_v3c_midpoint_reconstructed_trajectory_contract_config(
    **overrides: Any,
) -> FMPCEFExploratoryProbeConfig:
    """Return the first non-equivalent internal-knot reconstruction v3-C candidate."""

    payload: dict[str, Any] = {
        "use_midpoint_reconstructed_trajectory_contract": True,
        "candidate_name_override": STAGE05_V3C_MIDPOINT_RECONSTRUCTED_CANDIDATE_NAME,
    }
    payload.update(overrides)
    return build_stage05_v3c_stronger_semigroup_weight_config(**payload)

def build_stage05_v3c_endpoint_line_midpoint_trajectory_contract_config(
    **overrides: Any,
) -> FMPCEFExploratoryProbeConfig:
    """Return the endpoint-line midpoint reconstruction refinement on top of active refined v3-C."""

    payload: dict[str, Any] = {
        "use_midpoint_reconstructed_trajectory_contract": True,
        "use_endpoint_line_midpoint_trajectory_contract": True,
        "candidate_name_override": STAGE05_V3C_ENDPOINT_LINE_MIDPOINT_CANDIDATE_NAME,
    }
    payload.update(overrides)
    return build_stage05_v3c_stronger_semigroup_weight_config(**payload)

def build_stage05_v3c_endpoint_line_continuation_blend_trajectory_contract_config(
    **overrides: Any,
) -> FMPCEFExploratoryProbeConfig:
    """Return the endpoint-line continuation-blend refinement on top of active refined v3-C."""

    payload: dict[str, Any] = {
        "use_midpoint_reconstructed_trajectory_contract": True,
        "use_endpoint_line_midpoint_trajectory_contract": True,
        "use_endpoint_line_continuation_blend_trajectory_contract": True,
        "candidate_name_override": STAGE05_V3C_ENDPOINT_LINE_CONTINUATION_BLEND_CANDIDATE_NAME,
    }
    payload.update(overrides)
    return build_stage05_v3c_stronger_semigroup_weight_config(**payload)

def build_stage05_v3c_scaled_continuation_blend_trajectory_contract_config(
    **overrides: Any,
) -> FMPCEFExploratoryProbeConfig:
    """Return the single-axis scaled continuation-blend diagnostic on top of active refined v3-C."""

    payload: dict[str, Any] = {
        "use_midpoint_reconstructed_trajectory_contract": True,
        "use_endpoint_line_midpoint_trajectory_contract": True,
        "use_scaled_continuation_blend_trajectory_contract": True,
        "continuation_blend_scale": 1.5,
        "candidate_name_override": STAGE05_V3C_SCALED_CONTINUATION_BLEND_CANDIDATE_NAME,
    }
    payload.update(overrides)
    return build_stage05_v3c_stronger_semigroup_weight_config(**payload)

def build_stage05_v3c_coupled_defect_projection_trajectory_contract_config(
    **overrides: Any,
) -> FMPCEFExploratoryProbeConfig:
    """Return the coupled local defect-projection refinement on top of active refined v3-C."""

    payload: dict[str, Any] = {
        "use_midpoint_reconstructed_trajectory_contract": True,
        "use_endpoint_line_midpoint_trajectory_contract": True,
        "use_coupled_defect_projection_trajectory_contract": True,
        "candidate_name_override": STAGE05_V3C_COUPLED_DEFECT_PROJECTION_CANDIDATE_NAME,
    }
    payload.update(overrides)
    return build_stage05_v3c_stronger_semigroup_weight_config(**payload)

def build_stage05_v3c_precision_weighted_continuation_corrector_trajectory_contract_config(
    **overrides: Any,
) -> FMPCEFExploratoryProbeConfig:
    """Return the asymmetric continuation-corrector refinement on top of active refined v3-C."""

    payload: dict[str, Any] = {
        "use_midpoint_reconstructed_trajectory_contract": True,
        "use_endpoint_line_midpoint_trajectory_contract": True,
        "use_precision_weighted_continuation_corrector_trajectory_contract": True,
        "candidate_name_override": (
            STAGE05_V3C_PRECISION_WEIGHTED_CONTINUATION_CORRECTOR_CANDIDATE_NAME
        ),
    }
    payload.update(overrides)
    return build_stage05_v3c_stronger_semigroup_weight_config(**payload)

def _transport_family_name(config: FMPCEFExploratoryProbeConfig) -> str:
    if config.use_two_branch_residual_core:
        return TWO_BRANCH_RESIDUAL_MEANFLOW_TRANSPORT_FAMILY
    return RESIDUAL_MEANFLOW_TRANSPORT_FAMILY

def _candidate_name(config: FMPCEFExploratoryProbeConfig) -> str:
    if config.candidate_name_override is not None:
        return str(config.candidate_name_override)
    if config.use_scaled_continuation_blend_trajectory_contract:
        return STAGE05_V3C_SCALED_CONTINUATION_BLEND_CANDIDATE_NAME
    if config.use_precision_weighted_continuation_corrector_trajectory_contract:
        return STAGE05_V3C_PRECISION_WEIGHTED_CONTINUATION_CORRECTOR_CANDIDATE_NAME
    if config.use_coupled_defect_projection_trajectory_contract:
        return STAGE05_V3C_COUPLED_DEFECT_PROJECTION_CANDIDATE_NAME
    if config.use_endpoint_line_continuation_blend_trajectory_contract:
        return STAGE05_V3C_ENDPOINT_LINE_CONTINUATION_BLEND_CANDIDATE_NAME
    if config.use_endpoint_line_midpoint_trajectory_contract:
        return STAGE05_V3C_ENDPOINT_LINE_MIDPOINT_CANDIDATE_NAME
    if config.use_midpoint_reconstructed_trajectory_contract:
        return STAGE05_V3C_MIDPOINT_RECONSTRUCTED_CANDIDATE_NAME
    if config.use_fused_trajectory_semigroup_contract:
        return STAGE05_V3C_FUSED_CANDIDATE_NAME
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
    if config.use_scaled_continuation_blend_trajectory_contract:
        return {
            "status": "pending_real_fixed_budget_stage05_v3c_continuation_strength_diagnostic",
            "reference_candidate_name": STAGE05_V2_CANDIDATE_NAME,
        }
    if config.use_precision_weighted_continuation_corrector_trajectory_contract:
        return {
            "status": (
                "pending_real_fixed_budget_v2_vs_active_v3c_vs_"
                "precision_weighted_continuation_corrector_contract_comparison"
            ),
            "reference_candidate_name": STAGE05_V2_CANDIDATE_NAME,
        }
    if config.use_coupled_defect_projection_trajectory_contract:
        return {
            "status": (
                "pending_real_fixed_budget_v2_vs_active_v3c_vs_"
                "coupled_defect_projection_contract_comparison"
            ),
            "reference_candidate_name": STAGE05_V2_CANDIDATE_NAME,
        }
    if config.use_endpoint_line_continuation_blend_trajectory_contract:
        return {
            "status": (
                "pending_real_fixed_budget_v2_vs_active_v3c_vs_"
                "endpoint_line_continuation_blend_contract_comparison"
            ),
            "reference_candidate_name": STAGE05_V2_CANDIDATE_NAME,
        }
    if config.use_endpoint_line_midpoint_trajectory_contract:
        return {
            "status": "pending_real_fixed_budget_v2_vs_active_v3c_vs_endpoint_line_midpoint_contract_comparison",
            "reference_candidate_name": STAGE05_V2_CANDIDATE_NAME,
        }
    if config.use_midpoint_reconstructed_trajectory_contract:
        return {
            "status": "pending_real_fixed_budget_v2_vs_active_v3c_vs_midpoint_reconstructed_contract_comparison",
            "reference_candidate_name": STAGE05_V2_CANDIDATE_NAME,
        }
    if config.use_fused_trajectory_semigroup_contract:
        return {
            "status": "pending_real_fixed_budget_v2_vs_active_v3c_vs_fused_contract_comparison",
            "reference_candidate_name": STAGE05_V2_CANDIDATE_NAME,
        }
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
    if (
        config.use_fused_trajectory_semigroup_contract
        or config.use_midpoint_reconstructed_trajectory_contract
    ):
        return None
    if not config.use_endpoint_semigroup_consistency_probe:
        return None
    return {
        "status": "pending_real_fixed_budget_v2_vs_promoted_v3b_vs_v3c_comparison",
        "reference_candidate_name": STAGE05_V3B_REFINED_CANDIDATE_NAME,
    }

def _pairwise_active_refined_v3c_placeholder(
    config: FMPCEFExploratoryProbeConfig,
) -> dict[str, Any] | None:
    if config.use_scaled_continuation_blend_trajectory_contract:
        return {
            "status": "pending_real_fixed_budget_stage05_v3c_continuation_strength_diagnostic",
            "reference_candidate_name": STAGE05_V3C_STRONGER_SEMIGROUP_CANDIDATE_NAME,
        }
    if config.use_precision_weighted_continuation_corrector_trajectory_contract:
        return {
            "status": (
                "pending_real_fixed_budget_v2_vs_active_v3c_vs_"
                "precision_weighted_continuation_corrector_contract_comparison"
            ),
            "reference_candidate_name": STAGE05_V3C_STRONGER_SEMIGROUP_CANDIDATE_NAME,
        }
    if config.use_coupled_defect_projection_trajectory_contract:
        return {
            "status": (
                "pending_real_fixed_budget_v2_vs_active_v3c_vs_"
                "coupled_defect_projection_contract_comparison"
            ),
            "reference_candidate_name": STAGE05_V3C_STRONGER_SEMIGROUP_CANDIDATE_NAME,
        }
    if config.use_endpoint_line_continuation_blend_trajectory_contract:
        return {
            "status": (
                "pending_real_fixed_budget_v2_vs_active_v3c_vs_"
                "endpoint_line_continuation_blend_contract_comparison"
            ),
            "reference_candidate_name": STAGE05_V3C_STRONGER_SEMIGROUP_CANDIDATE_NAME,
        }
    if config.use_endpoint_line_midpoint_trajectory_contract:
        return {
            "status": "pending_real_fixed_budget_v2_vs_active_v3c_vs_endpoint_line_midpoint_contract_comparison",
            "reference_candidate_name": STAGE05_V3C_STRONGER_SEMIGROUP_CANDIDATE_NAME,
        }
    if config.use_midpoint_reconstructed_trajectory_contract:
        return {
            "status": "pending_real_fixed_budget_v2_vs_active_v3c_vs_midpoint_reconstructed_contract_comparison",
            "reference_candidate_name": STAGE05_V3C_STRONGER_SEMIGROUP_CANDIDATE_NAME,
        }
    if not config.use_fused_trajectory_semigroup_contract:
        return None
    return {
        "status": "pending_real_fixed_budget_v2_vs_active_v3c_vs_fused_contract_comparison",
        "reference_candidate_name": STAGE05_V3C_STRONGER_SEMIGROUP_CANDIDATE_NAME,
    }

def _gap_closure_decision_placeholder(config: FMPCEFExploratoryProbeConfig) -> str | None:
    if config.use_scaled_continuation_blend_trajectory_contract:
        return "pending_real_fixed_budget_stage05_v3c_continuation_strength_diagnostic"
    if config.use_precision_weighted_continuation_corrector_trajectory_contract:
        return (
            "pending_real_fixed_budget_v2_vs_active_v3c_vs_"
            "precision_weighted_continuation_corrector_contract_comparison"
        )
    if config.use_coupled_defect_projection_trajectory_contract:
        return (
            "pending_real_fixed_budget_v2_vs_active_v3c_vs_"
            "coupled_defect_projection_contract_comparison"
        )
    if config.use_endpoint_line_continuation_blend_trajectory_contract:
        return (
            "pending_real_fixed_budget_v2_vs_active_v3c_vs_"
            "endpoint_line_continuation_blend_contract_comparison"
        )
    if config.use_endpoint_line_midpoint_trajectory_contract:
        return "pending_real_fixed_budget_v2_vs_active_v3c_vs_endpoint_line_midpoint_contract_comparison"
    if config.use_midpoint_reconstructed_trajectory_contract:
        return "pending_real_fixed_budget_v2_vs_active_v3c_vs_midpoint_reconstructed_contract_comparison"
    if config.use_fused_trajectory_semigroup_contract:
        return "pending_real_fixed_budget_v2_vs_active_v3c_vs_fused_contract_comparison"
    if config.use_endpoint_semigroup_consistency_probe:
        return "pending_real_fixed_budget_v2_vs_promoted_v3b_vs_v3c_comparison"
    if config.use_trajectory_curriculum_contract:
        return "pending_real_fixed_budget_v2_vs_v3a_vs_v3b_comparison"
    if not config.use_explicit_transport_drift_decomposition:
        return None
    return "pending_formal_v2_vs_v3a_comparison"

def _recommended_next_move(config: FMPCEFExploratoryProbeConfig) -> str:
    if config.use_scaled_continuation_blend_trajectory_contract:
        return "run_real_fixed_budget_stage05_v3c_continuation_strength_diagnostic"
    if config.use_precision_weighted_continuation_corrector_trajectory_contract:
        return (
            "run_fixed_budget_v2_vs_active_v3c_vs_"
            "precision_weighted_continuation_corrector_contract_comparison"
        )
    if config.use_coupled_defect_projection_trajectory_contract:
        return "run_fixed_budget_v2_vs_active_v3c_vs_coupled_defect_projection_contract_comparison"
    if config.use_endpoint_line_continuation_blend_trajectory_contract:
        return "run_fixed_budget_v2_vs_active_v3c_vs_endpoint_line_continuation_blend_contract_comparison"
    if config.use_endpoint_line_midpoint_trajectory_contract:
        return "run_fixed_budget_v2_vs_active_v3c_vs_endpoint_line_midpoint_contract_comparison"
    if config.use_midpoint_reconstructed_trajectory_contract:
        return "run_fixed_budget_v2_vs_active_v3c_vs_midpoint_reconstructed_contract_comparison"
    if config.use_fused_trajectory_semigroup_contract:
        return "run_fixed_budget_v2_vs_active_v3c_vs_fused_contract_comparison"
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

def _main_trajectory_contract_identity(config: FMPCEFExploratoryProbeConfig) -> str | None:
    if config.use_scaled_continuation_blend_trajectory_contract:
        return MAIN_TRAJECTORY_CONTRACT_IDENTITY_V3C_SCALED_CONTINUATION_BLEND
    if config.use_precision_weighted_continuation_corrector_trajectory_contract:
        return MAIN_TRAJECTORY_CONTRACT_IDENTITY_V3C_PRECISION_WEIGHTED_CONTINUATION_CORRECTOR
    if config.use_coupled_defect_projection_trajectory_contract:
        return MAIN_TRAJECTORY_CONTRACT_IDENTITY_V3C_COUPLED_DEFECT_PROJECTION
    if config.use_endpoint_line_continuation_blend_trajectory_contract:
        return MAIN_TRAJECTORY_CONTRACT_IDENTITY_V3C_ENDPOINT_LINE_CONTINUATION_BLEND
    if config.use_endpoint_line_midpoint_trajectory_contract:
        return MAIN_TRAJECTORY_CONTRACT_IDENTITY_V3C_ENDPOINT_LINE_MIDPOINT
    if config.use_midpoint_reconstructed_trajectory_contract:
        return MAIN_TRAJECTORY_CONTRACT_IDENTITY_V3C_MIDPOINT_RECONSTRUCTED
    if config.use_fused_trajectory_semigroup_contract:
        return MAIN_TRAJECTORY_CONTRACT_IDENTITY_V3C_FUSED
    if config.use_endpoint_semigroup_consistency_probe:
        return MAIN_TRAJECTORY_CONTRACT_IDENTITY_V3C_STACKED
    if config.use_trajectory_curriculum_contract:
        return MAIN_TRAJECTORY_CONTRACT_IDENTITY_V3B
    return None

def _main_trajectory_contract_target_contract(
    config: FMPCEFExploratoryProbeConfig,
) -> str | None:
    if config.use_scaled_continuation_blend_trajectory_contract:
        return SCALED_CONTINUATION_BLEND_TRAJECTORY_CONTRACT
    if config.use_precision_weighted_continuation_corrector_trajectory_contract:
        return PRECISION_WEIGHTED_CONTINUATION_CORRECTOR_TRAJECTORY_CONTRACT
    if config.use_coupled_defect_projection_trajectory_contract:
        return COUPLED_DEFECT_PROJECTION_TRAJECTORY_CONTRACT
    if config.use_endpoint_line_continuation_blend_trajectory_contract:
        return ENDPOINT_LINE_CONTINUATION_BLEND_TRAJECTORY_CONTRACT
    if config.use_endpoint_line_midpoint_trajectory_contract:
        return ENDPOINT_LINE_MIDPOINT_TRAJECTORY_CONTRACT
    if config.use_midpoint_reconstructed_trajectory_contract:
        return MIDPOINT_RECONSTRUCTED_TRAJECTORY_CONTRACT
    if config.use_fused_trajectory_semigroup_contract:
        return FUSED_TRAJECTORY_SEMIGROUP_CONTRACT
    if config.use_trajectory_curriculum_contract:
        return TRAJECTORY_CURRICULUM_TARGET_CONTRACT
    return None

def _semigroup_consistency_absorbed_into_main_trajectory_contract(
    config: FMPCEFExploratoryProbeConfig,
) -> bool:
    return bool(
        config.use_fused_trajectory_semigroup_contract
        or config.use_scaled_continuation_blend_trajectory_contract
        or config.use_midpoint_reconstructed_trajectory_contract
        or config.use_coupled_defect_projection_trajectory_contract
        or config.use_endpoint_line_continuation_blend_trajectory_contract
    )

def _semigroup_consistency_is_auxiliary_only(
    config: FMPCEFExploratoryProbeConfig,
) -> bool | None:
    if not config.use_endpoint_semigroup_consistency_probe:
        return None
    return not bool(
        config.use_fused_trajectory_semigroup_contract
        or config.use_scaled_continuation_blend_trajectory_contract
        or config.use_midpoint_reconstructed_trajectory_contract
        or config.use_coupled_defect_projection_trajectory_contract
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
