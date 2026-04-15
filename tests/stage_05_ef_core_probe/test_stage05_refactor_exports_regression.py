from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.stage_05_ef_core_probe import fmpc_ef_exploratory_probe as stage05_probe


def test_stage05_facade_reexports_stage06_dependency_surface() -> None:
    required_symbols = [
        "FMPCEFExploratoryProbeConfig",
        "Stage05ResidualCoreNetworks",
        "EndpointSemigroupTargets",
        "TrajectoryCurriculumTargets",
        "_as_batch_first",
        "_evaluate_mechanism_metrics",
        "_evaluate_slow_pc_metrics",
        "_forward_mlp",
        "_learned_velocity_fn",
        "_make_pc_model",
        "_make_psi_network",
        "_predict_residual_from_inputs",
        "_prepare_run_dir",
        "_resolve_run_dir",
        "_restore_pc_parameters",
        "_restore_residual_core_parameters",
        "_snapshot_pc_parameters",
        "_snapshot_residual_core_parameters",
        "_theta_update_from_transported_state",
        "alpha_for_epoch",
        "build_endpoint_semigroup_targets",
        "build_stage05_v3c_stronger_semigroup_weight_config",
        "build_trajectory_curriculum_targets",
        "ensure_finite_array",
        "run_fmpc_ef_exploratory_probe",
    ]

    missing = [name for name in required_symbols if not hasattr(stage05_probe, name)]
    assert missing == []


def test_stage05_v3c_stronger_semigroup_builder_keeps_expected_fields() -> None:
    config = stage05_probe.build_stage05_v3c_stronger_semigroup_weight_config()

    assert config.candidate_name_override == "stage05_v3c_stronger_semigroup_weight"
    assert config.use_two_branch_residual_core is True
    assert config.feature_aware_state_branch_tangents is True
    assert config.use_explicit_transport_drift_decomposition is True
    assert config.use_trajectory_curriculum_contract is True
    assert config.use_endpoint_semigroup_consistency_probe is True
    assert config.use_fused_trajectory_semigroup_contract is False
    assert config.use_midpoint_reconstructed_trajectory_contract is False
    assert config.lambda_drift == 1.0
    assert config.lambda_traj_curr == 0.2
    assert config.lambda_sg == 0.10
    assert config.alpha_floor == 0.5
    assert config.alpha_warmup_epochs == 3
    assert config.alpha_ramp_epochs == 3
