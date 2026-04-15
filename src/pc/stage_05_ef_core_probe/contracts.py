
from __future__ import annotations

from typing import Literal

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
STAGE05_V3C_STRONGER_SEMIGROUP_CANDIDATE_NAME = "stage05_v3c_stronger_semigroup_weight"
STAGE05_V3C_FUSED_CANDIDATE_NAME = "stage05_v3c_fused_trajectory_semigroup_contract"
STAGE05_V3C_MIDPOINT_RECONSTRUCTED_CANDIDATE_NAME = (
    "stage05_v3c_midpoint_reconstructed_trajectory_contract"
)
STAGE05_V3C_ENDPOINT_LINE_MIDPOINT_CANDIDATE_NAME = (
    "stage05_v3c_endpoint_line_midpoint_trajectory_contract"
)
STAGE05_V3C_ENDPOINT_LINE_CONTINUATION_BLEND_CANDIDATE_NAME = (
    "stage05_v3c_endpoint_line_continuation_blend_trajectory_contract"
)
STAGE05_V3C_SCALED_CONTINUATION_BLEND_CANDIDATE_NAME = (
    "stage05_v3c_scaled_continuation_blend_trajectory_contract"
)
STAGE05_V3C_COUPLED_DEFECT_PROJECTION_CANDIDATE_NAME = (
    "stage05_v3c_coupled_defect_projection_trajectory_contract"
)
STAGE05_V3C_PRECISION_WEIGHTED_CONTINUATION_CORRECTOR_CANDIDATE_NAME = (
    "stage05_v3c_precision_weighted_continuation_corrector_trajectory_contract"
)
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
FUSED_TRAJECTORY_SEMIGROUP_CONTRACT = (
    "W = lambda_tc + lambda_sg * r^2; "
    "m_fuse_star = (lambda_tc * m_traj_star + lambda_sg * r^2 * m_sg_star) / W; "
    "L_main_traj = W * ||m_hat - m_fuse_star||^2"
)
MIDPOINT_RECONSTRUCTED_TRAJECTORY_CONTRACT = (
    "u_B = u_boot(z_t, alpha * r, t; c); "
    "z_B = z_t + alpha * r * u_B; "
    "u_C_boot = stopgrad(u_hat(z_B, r_s, s; c)); "
    "z_sg_mid_star = z_sg_star - (1 - alpha) * r * u_C_boot; "
    "z_mid_star = (1 - kappa) * z_B + kappa * z_sg_mid_star; "
    "u_C_star = stopgrad(u_hat(z_mid_star, r_s, s; c)); "
    "u_main_star = alpha * ((z_mid_star - z_t) / (alpha * r)) + (1 - alpha) * u_C_star; "
    "L_main_traj = (lambda_tc + lambda_sg * r^2) * ||m_hat - (u_main_star - g_t)||^2"
)
ENDPOINT_LINE_MIDPOINT_TRAJECTORY_CONTRACT = (
    "u_B = u_boot(z_t, alpha * r, t; c); "
    "z_B = z_t + alpha * r * u_B; "
    "z_line_mid_star = z_t + alpha * (z_sg_star - z_t); "
    "z_mid_star = (1 - kappa) * z_B + kappa * z_line_mid_star; "
    "u_C_star = stopgrad(u_hat(z_mid_star, r_s, s; c)); "
    "u_main_star = alpha * ((z_mid_star - z_t) / (alpha * r)) + (1 - alpha) * u_C_star; "
    "L_main_traj = (lambda_tc + lambda_sg * r^2) * ||m_hat - (u_main_star - g_t)||^2"
)
ENDPOINT_LINE_CONTINUATION_BLEND_TRAJECTORY_CONTRACT = (
    "u_B = u_boot(z_t, alpha * r, t; c); "
    "z_B = z_t + alpha * r * u_B; "
    "z_line_mid_star = z_t + alpha * (z_sg_star - z_t); "
    "z_mid_star = (1 - kappa) * z_B + kappa * z_line_mid_star; "
    "u_C_traj_star = stopgrad(u_hat(z_mid_star, r_s, s; c)); "
    "u_C_sg_star = (z_sg_star - z_mid_star) / ((1 - alpha) * r); "
    "u_C_star = (1 - kappa) * u_C_traj_star + kappa * u_C_sg_star; "
    "u_main_star = alpha * ((z_mid_star - z_t) / (alpha * r)) + (1 - alpha) * u_C_star; "
    "L_main_traj = (lambda_tc + lambda_sg * r^2) * ||m_hat - (u_main_star - g_t)||^2"
)
SCALED_CONTINUATION_BLEND_TRAJECTORY_CONTRACT = (
    "u_B = u_boot(z_t, alpha * r, t; c); "
    "z_B = z_t + alpha * r * u_B; "
    "z_line_mid_star = z_t + alpha * (z_sg_star - z_t); "
    "z_mid_star = (1 - kappa) * z_B + kappa * z_line_mid_star; "
    "u_C_traj_star = stopgrad(u_hat(z_mid_star, r_s, s; c)); "
    "u_C_sg_star = (z_sg_star - z_mid_star) / ((1 - alpha) * r); "
    "kappa_eff = min(1.0, gamma_cont * kappa); "
    "u_C_star = (1 - kappa_eff) * u_C_traj_star + kappa_eff * u_C_sg_star; "
    "u_main_star = alpha * ((z_mid_star - z_t) / (alpha * r)) + (1 - alpha) * u_C_star; "
    "L_main_traj = (lambda_tc + lambda_sg * r^2) * ||m_hat - (u_main_star - g_t)||^2"
)
COUPLED_DEFECT_PROJECTION_TRAJECTORY_CONTRACT = (
    "u_B = u_boot(z_t, alpha * r, t; c); "
    "z_B = z_t + alpha * r * u_B; "
    "z_line_mid_star = z_t + alpha * (z_sg_star - z_t); "
    "z_mid_star_0 = (1 - kappa) * z_B + kappa * z_line_mid_star; "
    "u_C_traj_star_0 = stopgrad(u_hat(z_mid_star_0, r_s, s; c)); "
    "d_sg_0 = u_sg_star - (alpha * u_short_star_0 + (1 - alpha) * u_C_traj_star_0); "
    "rho = (lambda_sg * r^2) / (lambda_tc + lambda_sg * r^2 * (alpha^2 + (1 - alpha)^2)); "
    "u_short_star_half = u_short_star_0 + rho * alpha * d_sg_0; "
    "z_mid_star_half = z_t + alpha * r * u_short_star_half; "
    "u_C_traj_star_1 = stopgrad(u_hat(z_mid_star_half, r_s, s; c)); "
    "d_sg_1 = u_sg_star - (alpha * u_short_star_half + (1 - alpha) * u_C_traj_star_1); "
    "u_short_star = u_short_star_half + rho * alpha * d_sg_1; "
    "u_C_star = u_C_traj_star_1 + rho * (1 - alpha) * d_sg_1; "
    "u_main_star = alpha * u_short_star + (1 - alpha) * u_C_star; "
    "L_main_traj = (lambda_tc + lambda_sg * r^2) * ||m_hat - (u_main_star - g_t)||^2"
)
PRECISION_WEIGHTED_CONTINUATION_CORRECTOR_TRAJECTORY_CONTRACT = (
    "u_B = u_boot(z_t, alpha * r, t; c); "
    "z_B = z_t + alpha * r * u_B; "
    "z_line_mid_star = z_t + alpha * (z_sg_star - z_t); "
    "z_mid_star = (1 - kappa) * z_B + kappa * z_line_mid_star; "
    "u_C_traj_star = stopgrad(u_hat(z_mid_star, r_s, s; c)); "
    "eta_cont = (lambda_sg * r^2 * (1 - alpha)^2) / "
    "(lambda_tc + lambda_sg * r^2 * (1 - alpha)^2); "
    "u_main_star = (1 - eta_cont) * "
    "(alpha * ((z_mid_star - z_t) / (alpha * r)) + (1 - alpha) * u_C_traj_star) "
    "+ eta_cont * u_sg_star; "
    "L_main_traj = (lambda_tc + lambda_sg * r^2) * ||m_hat - (u_main_star - g_t)||^2"
)
MAIN_TRAJECTORY_CONTRACT_IDENTITY_V3B = "trajectory_curriculum_detached_continuation"
MAIN_TRAJECTORY_CONTRACT_IDENTITY_V3C_STACKED = (
    "stacked_trajectory_curriculum_plus_auxiliary_semigroup_probe"
)
MAIN_TRAJECTORY_CONTRACT_IDENTITY_V3C_FUSED = (
    "exact_detached_target_barycentric_fusion"
)
MAIN_TRAJECTORY_CONTRACT_IDENTITY_V3C_MIDPOINT_RECONSTRUCTED = (
    "midpoint_reconstructed_semigroup_internalized_trajectory_contract"
)
MAIN_TRAJECTORY_CONTRACT_IDENTITY_V3C_ENDPOINT_LINE_MIDPOINT = (
    "endpoint_line_midpoint_reconstructed_semigroup_internalized_trajectory_contract"
)
MAIN_TRAJECTORY_CONTRACT_IDENTITY_V3C_ENDPOINT_LINE_CONTINUATION_BLEND = (
    "endpoint_line_midpoint_with_continuation_target_blend_semigroup_internalized_trajectory_contract"
)
MAIN_TRAJECTORY_CONTRACT_IDENTITY_V3C_SCALED_CONTINUATION_BLEND = (
    "endpoint_line_midpoint_with_scaled_continuation_blend_contract"
)
MAIN_TRAJECTORY_CONTRACT_IDENTITY_V3C_COUPLED_DEFECT_PROJECTION = (
    "endpoint_line_midpoint_with_coupled_local_defect_projection_predictor_corrector_contract"
)
MAIN_TRAJECTORY_CONTRACT_IDENTITY_V3C_PRECISION_WEIGHTED_CONTINUATION_CORRECTOR = (
    "endpoint_line_midpoint_with_precision_weighted_continuation_corrector_contract"
)
CONTINUATION_TARGET_BLEND_IDENTITY = "kappa_closed_form_blend"
CONTINUATION_BLEND_SCALE_IDENTITY = "fixed_gamma_cont_scaled_kappa"
CONTINUATION_MAP_COEFFICIENT_IDENTITY = "local_continuation_map_closed_form_eta_cont"
DEFECT_PROJECTION_COEFFICIENT_IDENTITY = "two_segment_quadratic_closed_form_rho"
BASE_CONTINUATION_COEFFICIENT_IDENTITY = "kappa_closed_form_blend"
EFFECTIVE_SCALED_CONTINUATION_BLEND_FORMULA = "kappa_eff = min(1.0, gamma_cont * kappa)"
