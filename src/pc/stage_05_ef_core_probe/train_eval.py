
from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np

from ..datasets import load_digits_split
from ..energy import compute_cache, total_energy
from ..metrics import classification_accuracy
from ..minibatch import iter_minibatches
from ..mlp_baseline import MLPNetwork
from ..models import PCNetwork
from ..stage_03_transport_core_v1.fmpc_tf1_flow import (
    FMPCTF1Context,
    build_tf1_context,
    hidden_energy_from_state,
    hidden_local_flow,
    hidden_states_from_state,
    rollout_hidden_transport,
)
from ..training import apply_parameter_updates, parameter_gradients
from ..utils import set_seed
from .configs import *
from .configs import (
    _candidate_name,
    _explicit_transport_drift_target_contract,
    _gap_closure_decision_placeholder,
    _main_trajectory_contract_identity,
    _main_trajectory_contract_target_contract,
    _pairwise_active_refined_v3c_placeholder,
    _pairwise_promoted_v3b_placeholder,
    _pairwise_v2_placeholder,
    _pairwise_v3a_placeholder,
    _psi_family_name,
    _recommended_next_move,
    _residual_branch_structure,
    _residual_identity_target_contract,
    _semigroup_consistency_absorbed_into_main_trajectory_contract,
    _semigroup_consistency_is_auxiliary_only,
    _semigroup_target_contract,
    _stage_for_epoch,
    _trajectory_curriculum_target_contract,
    _transport_family_name,
    _u_psi_input_contract,
    _velocity_parameterization,
)
from .contracts import *
from .io_utils import *
from .io_utils import (
    _prepare_run_dir,
    _resolve_run_dir,
    _restore_pc_parameters,
    _restore_residual_core_parameters,
    _sigma2_payload,
    _snapshot_pc_parameters,
    _snapshot_residual_core_parameters,
    _write_epoch_metrics,
    _write_json,
)
from .residual_core import *
from .residual_core import (
    _learned_velocity_fn,
    _make_pc_model,
    _make_psi_network,
    _predict_residual_from_inputs,
    _residual_core_inputs_for_state,
    _weighted_explicit_transport_drift_step,
    _weighted_mse_step,
    _weighted_two_branch_mse_step,
)
from .targets import *

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
    train_main_traj_contract_loss: float
    train_mean_midpoint_reconstruction_shift_norm: float
    train_mean_continuation_reevaluation_shift_norm: float
    train_mean_continuation_target_blend_shift_norm: float
    train_mean_semigroup_endpoint_closure_residual_norm: float
    train_mean_predictor_semigroup_defect_norm: float
    train_mean_corrected_semigroup_defect_norm: float
    train_mean_first_projection_short_leg_correction_norm: float
    train_mean_first_projection_continuation_correction_norm: float
    train_mean_second_projection_short_leg_correction_norm: float
    train_mean_second_projection_continuation_correction_norm: float
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
            "main_trajectory_contract_identity": _main_trajectory_contract_identity(config),
            "main_trajectory_contract_target_contract": _main_trajectory_contract_target_contract(
                config
            ),
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
            "contract_fusion_enabled": bool(config.use_fused_trajectory_semigroup_contract),
            "target_reconstruction_enabled": bool(
                config.use_midpoint_reconstructed_trajectory_contract
            ),
            "midpoint_reconstruction_enabled": bool(
                config.use_midpoint_reconstructed_trajectory_contract
            ),
            "endpoint_line_midpoint_reconstruction_enabled": bool(
                config.use_endpoint_line_midpoint_trajectory_contract
            ),
            "continuation_target_refinement_enabled": bool(
                config.use_scaled_continuation_blend_trajectory_contract
                or config.use_endpoint_line_continuation_blend_trajectory_contract
                or config.use_coupled_defect_projection_trajectory_contract
                or config.use_precision_weighted_continuation_corrector_trajectory_contract
            ),
            "continuation_target_blending_enabled": bool(
                config.use_scaled_continuation_blend_trajectory_contract
                or
                config.use_endpoint_line_continuation_blend_trajectory_contract
                or config.use_precision_weighted_continuation_corrector_trajectory_contract
            ),
            "endpoint_implied_continuation_target_enabled": bool(
                config.use_scaled_continuation_blend_trajectory_contract
                or config.use_endpoint_line_continuation_blend_trajectory_contract
                or config.use_precision_weighted_continuation_corrector_trajectory_contract
            ),
            "continuation_target_blend_identity": (
                CONTINUATION_BLEND_SCALE_IDENTITY
                if config.use_scaled_continuation_blend_trajectory_contract
                else CONTINUATION_MAP_COEFFICIENT_IDENTITY
                if config.use_precision_weighted_continuation_corrector_trajectory_contract
                else CONTINUATION_TARGET_BLEND_IDENTITY
                if config.use_endpoint_line_continuation_blend_trajectory_contract
                else None
            ),
            "scaled_continuation_blend_enabled": bool(
                config.use_scaled_continuation_blend_trajectory_contract
            ),
            "continuation_blend_scale_identity": (
                CONTINUATION_BLEND_SCALE_IDENTITY
                if config.use_scaled_continuation_blend_trajectory_contract
                else None
            ),
            "continuation_blend_scale_value": (
                float(config.continuation_blend_scale)
                if config.use_scaled_continuation_blend_trajectory_contract
                else None
            ),
            "base_continuation_coefficient_identity": (
                BASE_CONTINUATION_COEFFICIENT_IDENTITY
                if (
                    config.use_scaled_continuation_blend_trajectory_contract
                    or config.use_endpoint_line_continuation_blend_trajectory_contract
                )
                else CONTINUATION_MAP_COEFFICIENT_IDENTITY
                if config.use_precision_weighted_continuation_corrector_trajectory_contract
                else None
            ),
            "effective_continuation_blend_formula": (
                EFFECTIVE_SCALED_CONTINUATION_BLEND_FORMULA
                if config.use_scaled_continuation_blend_trajectory_contract
                else "kappa_eff = kappa"
                if config.use_endpoint_line_continuation_blend_trajectory_contract
                else None
            ),
            "continuation_coefficient_identity": (
                CONTINUATION_BLEND_SCALE_IDENTITY
                if config.use_scaled_continuation_blend_trajectory_contract
                else CONTINUATION_MAP_COEFFICIENT_IDENTITY
                if config.use_precision_weighted_continuation_corrector_trajectory_contract
                else CONTINUATION_TARGET_BLEND_IDENTITY
                if config.use_endpoint_line_continuation_blend_trajectory_contract
                else None
            ),
            "precision_weighted_continuation_corrector_enabled": bool(
                config.use_precision_weighted_continuation_corrector_trajectory_contract
            ),
            "continuation_map_closed_form_coefficient_enabled": bool(
                config.use_precision_weighted_continuation_corrector_trajectory_contract
            ),
            "coupled_defect_projection_enabled": bool(
                config.use_coupled_defect_projection_trajectory_contract
            ),
            "shared_semigroup_defect_coupling_enabled": bool(
                config.use_coupled_defect_projection_trajectory_contract
            ),
            "predictor_corrector_refinement_enabled": bool(
                config.use_coupled_defect_projection_trajectory_contract
            ),
            "second_pass_continuation_reevaluation_enabled": bool(
                config.use_coupled_defect_projection_trajectory_contract
            ),
            "defect_projection_coefficient_identity": (
                DEFECT_PROJECTION_COEFFICIENT_IDENTITY
                if config.use_coupled_defect_projection_trajectory_contract
                else None
            ),
            "continuation_reevaluated_at_reconstructed_midpoint": bool(
                config.use_midpoint_reconstructed_trajectory_contract
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
                if (
                    config.use_endpoint_semigroup_consistency_probe
                    and not config.use_fused_trajectory_semigroup_contract
                    and not config.use_midpoint_reconstructed_trajectory_contract
                )
                else None
            ),
            "semigroup_consistency_absorbed_into_main_trajectory_contract": (
                _semigroup_consistency_absorbed_into_main_trajectory_contract(config)
            ),
            "semigroup_consistency_is_auxiliary_only": _semigroup_consistency_is_auxiliary_only(
                config
            ),
            "exact_detached_target_barycentric_fusion_enabled": bool(
                config.use_fused_trajectory_semigroup_contract
            ),
            "target_construction_is_artifact_independent": True,
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
            "pairwise_deltas_vs_active_refined_v3c_reference": _pairwise_active_refined_v3c_placeholder(
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
    lambda_traj_curr: float = 0.0,
    lambda_sg: float = 0.0,
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
    midpoint_reconstructed_main_targets: list[np.ndarray] = []
    midpoint_reconstructed_main_loss_weights: list[np.ndarray] = []
    midpoint_reconstruction_shift_norms: list[np.ndarray] = []
    continuation_reevaluation_shift_norms: list[np.ndarray] = []
    continuation_target_blend_shift_norms: list[np.ndarray] = []
    semigroup_endpoint_closure_residual_norms: list[np.ndarray] = []
    predictor_semigroup_defect_norms: list[np.ndarray] = []
    corrected_semigroup_defect_norms: list[np.ndarray] = []
    first_projection_short_leg_correction_norms: list[np.ndarray] = []
    first_projection_continuation_correction_norms: list[np.ndarray] = []
    second_projection_short_leg_correction_norms: list[np.ndarray] = []
    second_projection_continuation_correction_norms: list[np.ndarray] = []
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
        if semigroup_active and config.use_midpoint_reconstructed_trajectory_contract:
            midpoint_targets = build_midpoint_reconstructed_trajectory_targets(
                context,
                psi_network,
                config,
                z_t,
                t=t_float,
                r=r_k,
                alpha=float(trajectory_alpha),
                lambda_traj_curr=float(lambda_traj_curr),
                lambda_sg=float(lambda_sg),
            )
            midpoint_reconstructed_main_targets.append(midpoint_targets.unified_residual_target)
            midpoint_reconstructed_main_loss_weights.append(midpoint_targets.main_loss_weights)
            midpoint_reconstruction_shift_norms.append(
                midpoint_targets.midpoint_reconstruction_shift_norm
            )
            continuation_reevaluation_shift_norms.append(
                midpoint_targets.continuation_reevaluation_shift_norm
            )
            continuation_target_blend_shift_norms.append(
                midpoint_targets.continuation_target_blend_shift_norm
            )
            semigroup_endpoint_closure_residual_norms.append(
                midpoint_targets.semigroup_endpoint_closure_residual_norm
            )
            predictor_semigroup_defect_norms.append(
                midpoint_targets.predictor_semigroup_defect_norm
            )
            corrected_semigroup_defect_norms.append(
                midpoint_targets.corrected_semigroup_defect_norm
            )
            first_projection_short_leg_correction_norms.append(
                midpoint_targets.first_projection_short_leg_correction_norm
            )
            first_projection_continuation_correction_norms.append(
                midpoint_targets.first_projection_continuation_correction_norm
            )
            second_projection_short_leg_correction_norms.append(
                midpoint_targets.second_projection_short_leg_correction_norm
            )
            second_projection_continuation_correction_norms.append(
                midpoint_targets.second_projection_continuation_correction_norm
            )
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
        midpoint_reconstructed_main_targets=(
            np.concatenate(midpoint_reconstructed_main_targets, axis=0).astype(np.float64, copy=False)
            if midpoint_reconstructed_main_targets
            else None
        ),
        midpoint_reconstructed_main_loss_weights=(
            np.concatenate(midpoint_reconstructed_main_loss_weights, axis=0).astype(
                np.float64, copy=False
            )
            if midpoint_reconstructed_main_loss_weights
            else None
        ),
        midpoint_reconstruction_shift_norms=(
            np.concatenate(midpoint_reconstruction_shift_norms, axis=0).astype(np.float64, copy=False)
            if midpoint_reconstruction_shift_norms
            else None
        ),
        continuation_reevaluation_shift_norms=(
            np.concatenate(continuation_reevaluation_shift_norms, axis=0).astype(
                np.float64, copy=False
            )
            if continuation_reevaluation_shift_norms
            else None
        ),
        continuation_target_blend_shift_norms=(
            np.concatenate(continuation_target_blend_shift_norms, axis=0).astype(
                np.float64, copy=False
            )
            if continuation_target_blend_shift_norms
            else None
        ),
        semigroup_endpoint_closure_residual_norms=(
            np.concatenate(semigroup_endpoint_closure_residual_norms, axis=0).astype(
                np.float64, copy=False
            )
            if semigroup_endpoint_closure_residual_norms
            else None
        ),
        predictor_semigroup_defect_norms=(
            np.concatenate(predictor_semigroup_defect_norms, axis=0).astype(
                np.float64, copy=False
            )
            if predictor_semigroup_defect_norms
            else None
        ),
        corrected_semigroup_defect_norms=(
            np.concatenate(corrected_semigroup_defect_norms, axis=0).astype(
                np.float64, copy=False
            )
            if corrected_semigroup_defect_norms
            else None
        ),
        first_projection_short_leg_correction_norms=(
            np.concatenate(first_projection_short_leg_correction_norms, axis=0).astype(
                np.float64, copy=False
            )
            if first_projection_short_leg_correction_norms
            else None
        ),
        first_projection_continuation_correction_norms=(
            np.concatenate(first_projection_continuation_correction_norms, axis=0).astype(
                np.float64, copy=False
            )
            if first_projection_continuation_correction_norms
            else None
        ),
        second_projection_short_leg_correction_norms=(
            np.concatenate(second_projection_short_leg_correction_norms, axis=0).astype(
                np.float64, copy=False
            )
            if second_projection_short_leg_correction_norms
            else None
        ),
        second_projection_continuation_correction_norms=(
            np.concatenate(second_projection_continuation_correction_norms, axis=0).astype(
                np.float64, copy=False
            )
            if second_projection_continuation_correction_norms
            else None
        ),
        explicit_transport_drift_decomposition_enabled=bool(
            config.use_explicit_transport_drift_decomposition
        ),
        trajectory_curriculum_enabled=trajectory_curriculum_active,
        endpoint_semigroup_consistency_enabled=semigroup_active,
    )

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
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]:
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
        lambda_traj_curr=lambda_traj_curr,
        lambda_sg=lambda_sg,
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
    main_traj_contract_loss = 0.0
    midpoint_reconstruction_shift_norm = 0.0
    continuation_reevaluation_shift_norm = 0.0
    continuation_target_blend_shift_norm = 0.0
    semigroup_endpoint_closure_residual_norm = 0.0
    predictor_semigroup_defect_norm = 0.0
    corrected_semigroup_defect_norm = 0.0
    first_projection_short_leg_correction_norm = 0.0
    first_projection_continuation_correction_norm = 0.0
    second_projection_short_leg_correction_norm = 0.0
    second_projection_continuation_correction_norm = 0.0
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
            config.use_midpoint_reconstructed_trajectory_contract
            and lambda_traj_curr > 0.0
            and lambda_sg > 0.0
            and supervision.midpoint_reconstructed_main_targets is not None
            and supervision.midpoint_reconstructed_main_loss_weights is not None
            and supervision.state_inputs is not None
        ):
            if supervision.trajectory_curriculum_targets is not None:
                traj_curr_loss = float(
                    np.mean((psi_predictions - supervision.trajectory_curriculum_targets) ** 2)
                )
            if supervision.semigroup_targets is not None:
                semigroup_loss = float(
                    np.mean((psi_predictions - supervision.semigroup_targets) ** 2)
                )
            if supervision.midpoint_reconstruction_shift_norms is not None:
                midpoint_reconstruction_shift_norm = float(
                    np.mean(supervision.midpoint_reconstruction_shift_norms)
                )
            if supervision.continuation_reevaluation_shift_norms is not None:
                continuation_reevaluation_shift_norm = float(
                    np.mean(supervision.continuation_reevaluation_shift_norms)
                )
            if supervision.continuation_target_blend_shift_norms is not None:
                continuation_target_blend_shift_norm = float(
                    np.mean(supervision.continuation_target_blend_shift_norms)
                )
            if supervision.semigroup_endpoint_closure_residual_norms is not None:
                semigroup_endpoint_closure_residual_norm = float(
                    np.mean(supervision.semigroup_endpoint_closure_residual_norms)
                )
            if supervision.predictor_semigroup_defect_norms is not None:
                predictor_semigroup_defect_norm = float(
                    np.mean(supervision.predictor_semigroup_defect_norms)
                )
            if supervision.corrected_semigroup_defect_norms is not None:
                corrected_semigroup_defect_norm = float(
                    np.mean(supervision.corrected_semigroup_defect_norms)
                )
            if supervision.first_projection_short_leg_correction_norms is not None:
                first_projection_short_leg_correction_norm = float(
                    np.mean(supervision.first_projection_short_leg_correction_norms)
                )
            if supervision.first_projection_continuation_correction_norms is not None:
                first_projection_continuation_correction_norm = float(
                    np.mean(supervision.first_projection_continuation_correction_norms)
                )
            if supervision.second_projection_short_leg_correction_norms is not None:
                second_projection_short_leg_correction_norm = float(
                    np.mean(supervision.second_projection_short_leg_correction_norms)
                )
            if supervision.second_projection_continuation_correction_norms is not None:
                second_projection_continuation_correction_norm = float(
                    np.mean(supervision.second_projection_continuation_correction_norms)
                )
            main_traj_contract_loss = float(
                np.mean(
                    supervision.midpoint_reconstructed_main_loss_weights
                    * ((psi_predictions - supervision.midpoint_reconstructed_main_targets) ** 2)
                )
            )
            total_loss += main_traj_contract_loss
            _weighted_two_branch_mse_step(
                psi_network,
                supervision.trajectory_inputs,
                supervision.state_inputs,
                supervision.midpoint_reconstructed_main_targets,
                loss_scale=1.0,
                sample_weights=supervision.midpoint_reconstructed_main_loss_weights,
            )
        elif (
            config.use_fused_trajectory_semigroup_contract
            and lambda_traj_curr > 0.0
            and lambda_sg > 0.0
            and supervision.trajectory_curriculum_targets is not None
            and supervision.semigroup_targets is not None
            and supervision.semigroup_loss_weights is not None
            and supervision.state_inputs is not None
        ):
            traj_curr_loss = float(
                np.mean((psi_predictions - supervision.trajectory_curriculum_targets) ** 2)
            )
            semigroup_loss = float(
                np.mean((psi_predictions - supervision.semigroup_targets) ** 2)
            )
            fused_targets = build_fused_trajectory_semigroup_targets(
                supervision.trajectory_curriculum_targets,
                supervision.semigroup_targets,
                supervision.semigroup_loss_weights,
                lambda_traj_curr=float(lambda_traj_curr),
                lambda_sg=float(lambda_sg),
            )
            main_traj_contract_loss = float(
                np.mean(
                    fused_targets.fusion_weights
                    * ((psi_predictions - fused_targets.fused_residual_target) ** 2)
                )
            )
            total_loss += main_traj_contract_loss
            _weighted_two_branch_mse_step(
                psi_network,
                supervision.trajectory_inputs,
                supervision.state_inputs,
                fused_targets.fused_residual_target,
                loss_scale=1.0,
                sample_weights=fused_targets.fusion_weights,
            )
        else:
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
        main_traj_contract_loss,
        midpoint_reconstruction_shift_norm,
        continuation_reevaluation_shift_norm,
        continuation_target_blend_shift_norm,
        semigroup_endpoint_closure_residual_norm,
        predictor_semigroup_defect_norm,
        corrected_semigroup_defect_norm,
        first_projection_short_leg_correction_norm,
        first_projection_continuation_correction_norm,
        second_projection_short_leg_correction_norm,
        second_projection_continuation_correction_norm,
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
        batch_main_traj_contract_losses: list[float] = []
        batch_midpoint_reconstruction_shift_norms: list[float] = []
        batch_continuation_reevaluation_shift_norms: list[float] = []
        batch_continuation_target_blend_shift_norms: list[float] = []
        batch_semigroup_endpoint_closure_residual_norms: list[float] = []
        batch_predictor_semigroup_defect_norms: list[float] = []
        batch_corrected_semigroup_defect_norms: list[float] = []
        batch_first_projection_short_leg_correction_norms: list[float] = []
        batch_first_projection_continuation_correction_norms: list[float] = []
        batch_second_projection_short_leg_correction_norms: list[float] = []
        batch_second_projection_continuation_correction_norms: list[float] = []
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
                main_traj_contract_loss,
                midpoint_reconstruction_shift_norm,
                continuation_reevaluation_shift_norm,
                continuation_target_blend_shift_norm,
                semigroup_endpoint_closure_residual_norm,
                predictor_semigroup_defect_norm,
                corrected_semigroup_defect_norm,
                first_projection_short_leg_correction_norm,
                first_projection_continuation_correction_norm,
                second_projection_short_leg_correction_norm,
                second_projection_continuation_correction_norm,
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
            batch_main_traj_contract_losses.append(main_traj_contract_loss)
            batch_midpoint_reconstruction_shift_norms.append(midpoint_reconstruction_shift_norm)
            batch_continuation_reevaluation_shift_norms.append(
                continuation_reevaluation_shift_norm
            )
            batch_continuation_target_blend_shift_norms.append(
                continuation_target_blend_shift_norm
            )
            batch_semigroup_endpoint_closure_residual_norms.append(
                semigroup_endpoint_closure_residual_norm
            )
            batch_predictor_semigroup_defect_norms.append(predictor_semigroup_defect_norm)
            batch_corrected_semigroup_defect_norms.append(corrected_semigroup_defect_norm)
            batch_first_projection_short_leg_correction_norms.append(
                first_projection_short_leg_correction_norm
            )
            batch_first_projection_continuation_correction_norms.append(
                first_projection_continuation_correction_norm
            )
            batch_second_projection_short_leg_correction_norms.append(
                second_projection_short_leg_correction_norm
            )
            batch_second_projection_continuation_correction_norms.append(
                second_projection_continuation_correction_norm
            )
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
                train_main_traj_contract_loss=float(np.mean(batch_main_traj_contract_losses)),
                train_mean_midpoint_reconstruction_shift_norm=float(
                    np.mean(batch_midpoint_reconstruction_shift_norms)
                ),
                train_mean_continuation_reevaluation_shift_norm=float(
                    np.mean(batch_continuation_reevaluation_shift_norms)
                ),
                train_mean_continuation_target_blend_shift_norm=float(
                    np.mean(batch_continuation_target_blend_shift_norms)
                ),
                train_mean_semigroup_endpoint_closure_residual_norm=float(
                    np.mean(batch_semigroup_endpoint_closure_residual_norms)
                ),
                train_mean_predictor_semigroup_defect_norm=float(
                    np.mean(batch_predictor_semigroup_defect_norms)
                ),
                train_mean_corrected_semigroup_defect_norm=float(
                    np.mean(batch_corrected_semigroup_defect_norms)
                ),
                train_mean_first_projection_short_leg_correction_norm=float(
                    np.mean(batch_first_projection_short_leg_correction_norms)
                ),
                train_mean_first_projection_continuation_correction_norm=float(
                    np.mean(batch_first_projection_continuation_correction_norms)
                ),
                train_mean_second_projection_short_leg_correction_norm=float(
                    np.mean(batch_second_projection_short_leg_correction_norms)
                ),
                train_mean_second_projection_continuation_correction_norm=float(
                    np.mean(batch_second_projection_continuation_correction_norms)
                ),
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
        "main_trajectory_contract_identity": _main_trajectory_contract_identity(config),
        "main_trajectory_contract_target_contract": _main_trajectory_contract_target_contract(
            config
        ),
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
        "contract_fusion_enabled": bool(config.use_fused_trajectory_semigroup_contract),
        "target_reconstruction_enabled": bool(
            config.use_midpoint_reconstructed_trajectory_contract
        ),
        "midpoint_reconstruction_enabled": bool(
            config.use_midpoint_reconstructed_trajectory_contract
        ),
        "endpoint_line_midpoint_reconstruction_enabled": bool(
            config.use_endpoint_line_midpoint_trajectory_contract
        ),
        "continuation_target_refinement_enabled": bool(
            config.use_scaled_continuation_blend_trajectory_contract
            or config.use_endpoint_line_continuation_blend_trajectory_contract
            or config.use_coupled_defect_projection_trajectory_contract
            or config.use_precision_weighted_continuation_corrector_trajectory_contract
        ),
        "continuation_target_blending_enabled": bool(
            config.use_scaled_continuation_blend_trajectory_contract
            or
            config.use_endpoint_line_continuation_blend_trajectory_contract
            or config.use_precision_weighted_continuation_corrector_trajectory_contract
        ),
        "endpoint_implied_continuation_target_enabled": bool(
            config.use_scaled_continuation_blend_trajectory_contract
            or config.use_endpoint_line_continuation_blend_trajectory_contract
            or config.use_precision_weighted_continuation_corrector_trajectory_contract
        ),
        "continuation_target_blend_identity": (
            CONTINUATION_BLEND_SCALE_IDENTITY
            if config.use_scaled_continuation_blend_trajectory_contract
            else CONTINUATION_MAP_COEFFICIENT_IDENTITY
            if config.use_precision_weighted_continuation_corrector_trajectory_contract
            else CONTINUATION_TARGET_BLEND_IDENTITY
            if config.use_endpoint_line_continuation_blend_trajectory_contract
            else None
        ),
        "scaled_continuation_blend_enabled": bool(
            config.use_scaled_continuation_blend_trajectory_contract
        ),
        "continuation_blend_scale_identity": (
            CONTINUATION_BLEND_SCALE_IDENTITY
            if config.use_scaled_continuation_blend_trajectory_contract
            else None
        ),
        "continuation_blend_scale_value": (
            float(config.continuation_blend_scale)
            if config.use_scaled_continuation_blend_trajectory_contract
            else None
        ),
        "base_continuation_coefficient_identity": (
            BASE_CONTINUATION_COEFFICIENT_IDENTITY
            if (
                config.use_scaled_continuation_blend_trajectory_contract
                or config.use_endpoint_line_continuation_blend_trajectory_contract
            )
            else CONTINUATION_MAP_COEFFICIENT_IDENTITY
            if config.use_precision_weighted_continuation_corrector_trajectory_contract
            else None
        ),
        "effective_continuation_blend_formula": (
            EFFECTIVE_SCALED_CONTINUATION_BLEND_FORMULA
            if config.use_scaled_continuation_blend_trajectory_contract
            else "kappa_eff = kappa"
            if config.use_endpoint_line_continuation_blend_trajectory_contract
            else None
        ),
        "precision_weighted_continuation_corrector_enabled": bool(
            config.use_precision_weighted_continuation_corrector_trajectory_contract
        ),
        "continuation_map_closed_form_coefficient_enabled": bool(
            config.use_precision_weighted_continuation_corrector_trajectory_contract
        ),
        "continuation_coefficient_identity": (
            CONTINUATION_BLEND_SCALE_IDENTITY
            if config.use_scaled_continuation_blend_trajectory_contract
            else CONTINUATION_MAP_COEFFICIENT_IDENTITY
            if config.use_precision_weighted_continuation_corrector_trajectory_contract
            else CONTINUATION_TARGET_BLEND_IDENTITY
            if config.use_endpoint_line_continuation_blend_trajectory_contract
            else None
        ),
        "coupled_defect_projection_enabled": bool(
            config.use_coupled_defect_projection_trajectory_contract
        ),
        "shared_semigroup_defect_coupling_enabled": bool(
            config.use_coupled_defect_projection_trajectory_contract
        ),
        "predictor_corrector_refinement_enabled": bool(
            config.use_coupled_defect_projection_trajectory_contract
        ),
        "second_pass_continuation_reevaluation_enabled": bool(
            config.use_coupled_defect_projection_trajectory_contract
        ),
        "defect_projection_coefficient_identity": (
            DEFECT_PROJECTION_COEFFICIENT_IDENTITY
            if config.use_coupled_defect_projection_trajectory_contract
            else None
        ),
        "continuation_reevaluated_at_reconstructed_midpoint": bool(
            config.use_midpoint_reconstructed_trajectory_contract
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
            if (
                config.use_endpoint_semigroup_consistency_probe
                and not config.use_fused_trajectory_semigroup_contract
                and not config.use_midpoint_reconstructed_trajectory_contract
            )
            else None
        ),
        "semigroup_consistency_absorbed_into_main_trajectory_contract": (
            _semigroup_consistency_absorbed_into_main_trajectory_contract(config)
        ),
        "semigroup_consistency_is_auxiliary_only": _semigroup_consistency_is_auxiliary_only(
            config
        ),
        "exact_detached_target_barycentric_fusion_enabled": bool(
            config.use_fused_trajectory_semigroup_contract
        ),
        "use_two_branch_residual_core": bool(config.use_two_branch_residual_core),
        "explicit_transport_drift_decomposition_enabled": bool(
            config.use_explicit_transport_drift_decomposition
        ),
        "pairwise_deltas_vs_stage05_v3a_reference": _pairwise_v3a_placeholder(config),
        "pairwise_deltas_vs_promoted_refined_v3b_reference": _pairwise_promoted_v3b_placeholder(
            config
        ),
        "pairwise_deltas_vs_active_refined_v3c_reference": _pairwise_active_refined_v3c_placeholder(
            config
        ),
        "uses_current_state_features": bool(config.use_two_branch_residual_core),
        "feature_aware_state_branch_tangents": bool(
            config.feature_aware_state_branch_tangents
        ),
        "target_construction_artifact_independent": True,
        "target_construction_is_artifact_independent": True,
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
        "selected_epoch_mean_midpoint_reconstruction_shift_norm": float(
            best_row["train_mean_midpoint_reconstruction_shift_norm"]
        ),
        "selected_epoch_mean_continuation_reevaluation_shift_norm": float(
            best_row["train_mean_continuation_reevaluation_shift_norm"]
        ),
        "selected_epoch_mean_continuation_target_blend_shift_norm": float(
            best_row["train_mean_continuation_target_blend_shift_norm"]
        ),
        "selected_epoch_mean_semigroup_endpoint_closure_residual_norm": float(
            best_row["train_mean_semigroup_endpoint_closure_residual_norm"]
        ),
        "selected_epoch_mean_predictor_semigroup_defect_norm": float(
            best_row["train_mean_predictor_semigroup_defect_norm"]
        ),
        "selected_epoch_mean_corrected_semigroup_defect_norm": float(
            best_row["train_mean_corrected_semigroup_defect_norm"]
        ),
        "selected_epoch_mean_first_projection_short_leg_correction_norm": float(
            best_row["train_mean_first_projection_short_leg_correction_norm"]
        ),
        "selected_epoch_mean_first_projection_continuation_correction_norm": float(
            best_row["train_mean_first_projection_continuation_correction_norm"]
        ),
        "selected_epoch_mean_second_projection_short_leg_correction_norm": float(
            best_row["train_mean_second_projection_short_leg_correction_norm"]
        ),
        "selected_epoch_mean_second_projection_continuation_correction_norm": float(
            best_row["train_mean_second_projection_continuation_correction_norm"]
        ),
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
