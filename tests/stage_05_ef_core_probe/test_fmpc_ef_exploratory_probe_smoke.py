from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.datasets import load_digits_split
from pc.stage_03_transport_core_v1.fmpc_tf1_flow import (
    bootstrap_average_velocity_target,
    build_tf1_context,
    teacher_free_feature_tangents,
    teacher_free_state_features,
)
from pc.stage_05_ef_core_probe.fmpc_ef_exploratory_probe import (
    _make_pc_model,
    _make_psi_network,
    alpha_for_epoch,
    build_corrected_residual_identity_target,
    build_endpoint_semigroup_targets,
    build_explicit_transport_drift_bootstrap_targets,
    build_fused_trajectory_semigroup_targets,
    build_trajectory_curriculum_targets,
    build_stage05_v3b_stronger_traj_curr_weight_config,
    build_stage05_v3c_endpoint_semigroup_config,
    build_stage05_v3c_fused_trajectory_semigroup_contract_config,
    build_stage05_v3c_stronger_semigroup_weight_config,
    build_state_branch_input_tangent,
    build_fmpc_ef_exploratory_probe_config,
    lambda_id_for_epoch,
    lambda_sg_for_epoch,
    lambda_traj_curr_for_epoch,
)


def load_run():
    module = runpy.run_path(str(ROOT / "experiments" / "stage_05_ef_core_probe" / "fmpc_ef_exploratory_probe.py"))
    return module["run"]


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_exploratory_probe_writes_expected_teacher_free_artifacts(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="exploratory_probe_smoke",
        epochs=6,
        warmup_epochs=2,
        batch_size=128,
        eval_steps=10,
        transport_steps=2,
        layer_dims=(64, 16, 10),
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "epoch_metrics.csv").exists()
    assert (run_dir / "summary.json").exists()

    config = _read_json(run_dir / "config.json")
    summary = _read_json(run_dir / "summary.json")
    epoch_rows = _read_csv(run_dir / "epoch_metrics.csv")

    assert summary["phase"] == "post_incremental_bridge_exploratory"
    assert summary["stage"] == "ef_core_probe"
    assert summary["teacher_free"] is True
    assert summary["uses_teacher_artifacts"] is False
    assert summary["transport_family"] == "residual_meanflow_core"
    assert summary["residual_identity_mode"] == "residual_corrected_meanflow"
    assert summary["energy_substrate"] == "baseline_pc_energy"
    assert summary["local_flow_definition"] == "exact_negative_hidden_state_gradient"
    assert summary["direct_anchor_source"] == "self_bootstrap_local_field"
    assert summary["transport_scope"] == "train_only"
    assert summary["transport_steps"] == 2
    assert summary["u_psi_input_contract"] == "concat([z_t, target_onehot, t, r])"
    assert summary["residual_branch_structure"] == "single_branch"
    assert summary["m_traj_input_contract"] == "concat([z_t, target_onehot, t, r])"
    assert summary["m_state_input_contract"] is None
    assert summary["bootstrap_target_contract"] == "m_boot = ((Phi_LF_r(z_t; c) - z_t) / r) - g_t"
    assert summary["residual_identity_target_contract"] == "m_id = r * D_T g_t + r * D_T m_psi"
    assert summary["selection_metric_source"] == "val_metric"
    assert summary["report_metric_source"] == "test_metric"
    assert summary["acceptance_contract"] == "mechanism_first"
    assert summary["task_accuracy_is_gate"] is False
    assert summary["no_teacher_dependency_in_target_construction"] is True
    assert summary["use_two_branch_residual_core"] is False
    assert summary["uses_current_state_features"] is False
    assert summary["deterministic_artifacts"] is True
    assert config["transport"]["transport_family"] == "residual_meanflow_core"
    assert config["transport"]["residual_identity_mode"] == "residual_corrected_meanflow"
    assert config["transport"]["u_psi_input_contract"] == "concat([z_t, target_onehot, t, r])"
    assert config["transport"]["residual_branch_structure"] == "single_branch"
    assert config["transport"]["m_traj_input_contract"] == "concat([z_t, target_onehot, t, r])"
    assert config["transport"]["m_state_input_contract"] is None
    assert config["transport"]["bootstrap_target_contract"] == "m_boot = ((Phi_LF_r(z_t; c) - z_t) / r) - g_t"
    assert config["transport"]["residual_identity_target_contract"] == "m_id = r * D_T g_t + r * D_T m_psi"
    assert config["transport"]["no_teacher_dependency_in_target_construction"] is True
    assert config["transport"]["use_teacher_free_features"] is False
    assert config["transport"]["use_two_branch_residual_core"] is False
    assert config["transport"]["feature_aware_state_branch_tangents"] is False
    assert len(epoch_rows) == 6
    assert "lambda_id" in epoch_rows[0]
    assert "train_total_loss" in epoch_rows[0]
    assert "train_identity_loss" in epoch_rows[0]
    assert "val_one_step_energy_delta_vs_identity" in epoch_rows[0]
    assert "val_configured_fixed_point_residual_delta_vs_identity" in epoch_rows[0]


def test_two_branch_probe_writes_expected_v2_artifacts(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="exploratory_probe_v2_smoke",
        epochs=4,
        warmup_epochs=2,
        batch_size=128,
        eval_steps=8,
        transport_steps=2,
        layer_dims=(64, 16, 10),
        use_two_branch_residual_core=True,
        feature_aware_state_branch_tangents=True,
    )

    config = _read_json(result.run_dir / "config.json")
    summary = _read_json(result.run_dir / "summary.json")

    assert summary["transport_family"] == "two_branch_residual_meanflow_core"
    assert summary["residual_branch_structure"] == "two_branch"
    assert summary["candidate_name"] == "stage05_v2_two_branch_corrected_residual_meanflow_core"
    assert summary["m_traj_input_contract"] == "concat([z_t, target_onehot, t, r])"
    assert summary["m_state_input_contract"] == "concat([g_t, e_out_t, F_t])"
    assert summary["residual_identity_target_contract"] == (
        "m_id = r * D_T g_t + r * D_T m_traj + r * D_T m_state"
    )
    assert summary["use_two_branch_residual_core"] is True
    assert summary["explicit_transport_drift_decomposition_enabled"] is False
    assert summary["feature_aware_state_branch_tangents"] is True
    assert summary["uses_current_state_features"] is True

    assert config["transport"]["transport_family"] == "two_branch_residual_meanflow_core"
    assert config["transport"]["residual_branch_structure"] == "two_branch"
    assert config["transport"]["candidate_name"] == "stage05_v2_two_branch_corrected_residual_meanflow_core"
    assert config["transport"]["m_traj_input_contract"] == "concat([z_t, target_onehot, t, r])"
    assert config["transport"]["m_state_input_contract"] == "concat([g_t, e_out_t, F_t])"
    assert config["transport"]["residual_identity_target_contract"] == (
        "m_id = r * D_T g_t + r * D_T m_traj + r * D_T m_state"
    )
    assert config["transport"]["use_two_branch_residual_core"] is True
    assert config["transport"]["explicit_transport_drift_decomposition_enabled"] is False
    assert config["transport"]["feature_aware_state_branch_tangents"] is True
    assert config["transport"]["use_teacher_free_features"] is True


def test_explicit_transport_drift_bootstrap_targets_have_expected_shapes() -> None:
    config = build_fmpc_ef_exploratory_probe_config(
        run_seed=0,
        data_seed=0,
        model_init_seed=0,
        psi_init_seed=0,
        batch_order_seed=0,
        use_two_branch_residual_core=True,
        feature_aware_state_branch_tangents=True,
        use_explicit_transport_drift_decomposition=True,
    )
    split = load_digits_split(
        split_seed=config.data_seed,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
    )
    x_batch = split.x_train[:8]
    y_batch = split.y_train[:8]
    model = _make_pc_model(config)
    context = build_tf1_context(model, x_batch, y_batch)

    targets = build_explicit_transport_drift_bootstrap_targets(
        context,
        context.z0,
        t=0.25,
        r=0.75,
        integrator=config.bootstrap_integrator,
        substeps=config.bootstrap_substeps,
    )
    reference_u_boot = bootstrap_average_velocity_target(
        context,
        context.z0,
        t=0.25,
        r=0.75,
        integrator=config.bootstrap_integrator,
        substeps=config.bootstrap_substeps,
    )

    assert targets.gbar_boot.shape == context.z0.shape
    assert targets.transport_target.shape == context.z0.shape
    assert targets.drift_target.shape == context.z0.shape
    np.testing.assert_allclose(targets.u_boot, reference_u_boot, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        targets.residual_target,
        targets.transport_target + targets.drift_target,
        atol=1e-12,
        rtol=1e-12,
    )


def test_v3a_probe_writes_expected_artifacts(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="exploratory_probe_v3a_smoke",
        epochs=4,
        warmup_epochs=2,
        batch_size=128,
        eval_steps=8,
        transport_steps=2,
        layer_dims=(64, 16, 10),
        use_two_branch_residual_core=True,
        feature_aware_state_branch_tangents=True,
        use_explicit_transport_drift_decomposition=True,
        lambda_drift=1.0,
    )

    config = _read_json(result.run_dir / "config.json")
    summary = _read_json(result.run_dir / "summary.json")
    epoch_rows = _read_csv(result.run_dir / "epoch_metrics.csv")

    assert summary["candidate_name"] == "stage05_v3a_explicit_transport_drift_contract"
    assert summary["transport_family"] == "two_branch_residual_meanflow_core"
    assert summary["explicit_transport_drift_decomposition_enabled"] is True
    assert summary["explicit_transport_drift_target_contract"] == (
        "gbar_boot = avg local flow over the same bootstrap interval; "
        "d_boot = gbar_boot - g_t; q_boot = u_boot - gbar_boot"
    )
    assert summary["target_construction_artifact_independent"] is True
    assert summary["pairwise_deltas_vs_stage05_v2_reference"]["status"] == (
        "pending_formal_v2_vs_v3a_comparison"
    )
    assert summary["recommended_next_move"] == "run_fixed_budget_v2_vs_v3a_comparison"

    assert config["transport"]["candidate_name"] == "stage05_v3a_explicit_transport_drift_contract"
    assert config["transport"]["explicit_transport_drift_decomposition_enabled"] is True
    assert config["transport"]["lambda_drift"] == pytest.approx(1.0)
    assert config["transport"]["recommended_next_move"] == "run_fixed_budget_v2_vs_v3a_comparison"

    assert "train_transport_loss" in epoch_rows[0]
    assert "train_drift_loss" in epoch_rows[0]


def test_trajectory_curriculum_targets_have_expected_shapes() -> None:
    config = build_fmpc_ef_exploratory_probe_config(
        run_seed=0,
        data_seed=0,
        model_init_seed=0,
        psi_init_seed=0,
        batch_order_seed=0,
        use_two_branch_residual_core=True,
        feature_aware_state_branch_tangents=True,
        use_explicit_transport_drift_decomposition=True,
        use_trajectory_curriculum_contract=True,
        alpha_floor=0.5,
        alpha_warmup_epochs=1,
        alpha_ramp_epochs=2,
    )
    split = load_digits_split(
        split_seed=config.data_seed,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
    )
    x_batch = split.x_train[:8]
    y_batch = split.y_train[:8]
    model = _make_pc_model(config)
    psi_network = _make_psi_network(config)
    context = build_tf1_context(model, x_batch, y_batch)

    targets = build_trajectory_curriculum_targets(
        context,
        psi_network,
        config,
        context.z0,
        t=0.25,
        r=0.75,
        alpha=0.5,
    )

    assert targets.short_horizon_bootstrap_velocity.shape == context.z0.shape
    assert targets.bootstrap_intermediate_state.shape == context.z0.shape
    assert targets.continuation_velocity.shape == context.z0.shape
    assert targets.current_velocity_target.shape == context.z0.shape
    assert targets.residual_target.shape == context.z0.shape
    assert targets.alpha == pytest.approx(0.5)
    assert targets.split_time == pytest.approx(0.625)
    assert targets.continuation_remaining_horizon == pytest.approx(0.375)


def test_v3b_probe_writes_expected_artifacts(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="exploratory_probe_v3b_smoke",
        epochs=4,
        warmup_epochs=2,
        batch_size=128,
        eval_steps=8,
        transport_steps=2,
        layer_dims=(64, 16, 10),
        use_two_branch_residual_core=True,
        feature_aware_state_branch_tangents=True,
        use_explicit_transport_drift_decomposition=True,
        use_trajectory_curriculum_contract=True,
        lambda_drift=1.0,
        lambda_traj_curr=0.1,
        alpha_floor=0.5,
        alpha_warmup_epochs=1,
        alpha_ramp_epochs=2,
    )

    config = _read_json(result.run_dir / "config.json")
    summary = _read_json(result.run_dir / "summary.json")
    epoch_rows = _read_csv(result.run_dir / "epoch_metrics.csv")

    assert summary["candidate_name"] == "stage05_v3b_trajectory_curriculum_contract"
    assert summary["explicit_transport_drift_decomposition_enabled"] is True
    assert summary["trajectory_curriculum_enabled"] is True
    assert summary["trajectory_curriculum_schedule_identity"] == "warmup_sigmoid_to_alpha_floor"
    assert summary["trajectory_curriculum_target_contract"] == (
        "u_curr_target = alpha * u_boot(z_t, alpha * r, t; c) + "
        "(1 - alpha) * u_hat(z_s_boot, r_s, s; c) [detached target side]"
    )
    assert summary["alpha_floor"] == pytest.approx(0.5)
    assert summary["pairwise_deltas_vs_stage05_v2_reference"]["status"] == (
        "pending_real_fixed_budget_v2_vs_v3a_vs_v3b_comparison"
    )
    assert summary["pairwise_deltas_vs_stage05_v3a_reference"]["status"] == (
        "pending_real_fixed_budget_v2_vs_v3a_vs_v3b_comparison"
    )
    assert summary["recommended_next_move"] == "run_fixed_budget_v2_vs_v3a_vs_v3b_comparison"

    assert config["transport"]["candidate_name"] == "stage05_v3b_trajectory_curriculum_contract"
    assert config["transport"]["trajectory_curriculum_enabled"] is True
    assert config["transport"]["trajectory_curriculum_schedule_identity"] == (
        "warmup_sigmoid_to_alpha_floor"
    )
    assert config["transport"]["alpha_floor"] == pytest.approx(0.5)
    assert config["transport"]["lambda_traj_curr"] == pytest.approx(0.1)

    assert "alpha" in epoch_rows[0]
    assert "lambda_traj_curr" in epoch_rows[0]
    assert "train_traj_curr_loss" in epoch_rows[0]


def test_promoted_v3b_builder_exposes_explicit_candidate_identity() -> None:
    config = build_stage05_v3b_stronger_traj_curr_weight_config()

    assert config.candidate_name_override == "stage05_v3b_stronger_traj_curr_weight"


def test_refined_v3c_builder_exposes_explicit_candidate_identity_and_weight() -> None:
    config = build_stage05_v3c_stronger_semigroup_weight_config()

    assert config.candidate_name_override == "stage05_v3c_stronger_semigroup_weight"
    assert config.lambda_sg == pytest.approx(0.10)


def test_fused_v3c_builder_exposes_explicit_candidate_identity_and_fusion_flag() -> None:
    config = build_stage05_v3c_fused_trajectory_semigroup_contract_config()

    assert config.candidate_name_override == "stage05_v3c_fused_trajectory_semigroup_contract"
    assert config.use_fused_trajectory_semigroup_contract is True
    assert config.lambda_sg == pytest.approx(0.10)


def test_v3c_endpoint_semigroup_targets_have_expected_shapes() -> None:
    config = build_stage05_v3c_endpoint_semigroup_config(
        run_seed=0,
        data_seed=0,
        model_init_seed=0,
        psi_init_seed=0,
        batch_order_seed=0,
    )
    split = load_digits_split(
        split_seed=config.data_seed,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
    )
    x_batch = split.x_train[:8]
    y_batch = split.y_train[:8]
    model = _make_pc_model(config)
    psi_network = _make_psi_network(config)
    context = build_tf1_context(model, x_batch, y_batch)

    targets = build_endpoint_semigroup_targets(
        context,
        psi_network,
        config,
        context.z0,
        t=0.25,
        r=0.75,
        alpha=0.5,
    )

    assert targets.direct_velocity.shape == context.z0.shape
    assert targets.short_horizon_velocity.shape == context.z0.shape
    assert targets.continuation_velocity.shape == context.z0.shape
    assert targets.direct_endpoint.shape == context.z0.shape
    assert targets.midpoint_state.shape == context.z0.shape
    assert targets.split_endpoint.shape == context.z0.shape
    assert targets.split_endpoint_target.shape == context.z0.shape
    assert targets.semigroup_residual.shape == context.z0.shape
    assert targets.velocity_target.shape == context.z0.shape
    assert targets.residual_target.shape == context.z0.shape
    assert targets.loss_weights.shape == (context.z0.shape[0], 1)
    np.testing.assert_allclose(targets.split_endpoint_target, targets.split_endpoint, atol=1e-12, rtol=1e-12)


def test_fused_v3c_targets_have_expected_shapes() -> None:
    config = build_stage05_v3c_fused_trajectory_semigroup_contract_config(
        run_seed=0,
        data_seed=0,
        model_init_seed=0,
        psi_init_seed=0,
        batch_order_seed=0,
    )
    split = load_digits_split(
        split_seed=config.data_seed,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
    )
    x_batch = split.x_train[:8]
    y_batch = split.y_train[:8]
    model = _make_pc_model(config)
    psi_network = _make_psi_network(config)
    context = build_tf1_context(model, x_batch, y_batch)

    trajectory_targets = build_trajectory_curriculum_targets(
        context,
        psi_network,
        config,
        context.z0,
        t=0.25,
        r=0.75,
        alpha=0.5,
    )
    semigroup_targets = build_endpoint_semigroup_targets(
        context,
        psi_network,
        config,
        context.z0,
        t=0.25,
        r=0.75,
        alpha=0.5,
    )
    fused_targets = build_fused_trajectory_semigroup_targets(
        trajectory_targets.residual_target,
        semigroup_targets.residual_target,
        semigroup_targets.loss_weights,
        lambda_traj_curr=config.lambda_traj_curr,
        lambda_sg=config.lambda_sg,
    )

    assert fused_targets.trajectory_residual_target.shape == context.z0.shape
    assert fused_targets.semigroup_residual_target.shape == context.z0.shape
    assert fused_targets.fused_residual_target.shape == context.z0.shape
    assert fused_targets.fusion_weights.shape == (context.z0.shape[0], 1)
    assert fused_targets.fusion_rho.shape == (context.z0.shape[0], 1)
    assert np.all(fused_targets.fusion_weights > 0.0)
    assert np.all(fused_targets.fusion_rho > 0.0)
    assert np.all(fused_targets.fusion_rho < 1.0)


def test_v3c_probe_writes_expected_artifacts(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="exploratory_probe_v3c_smoke",
        epochs=4,
        warmup_epochs=2,
        batch_size=128,
        eval_steps=8,
        transport_steps=2,
        layer_dims=(64, 16, 10),
        **{
            key: value
            for key, value in build_stage05_v3c_endpoint_semigroup_config(
                output_root=tmp_path,
                run_id="unused",
                epochs=4,
                batch_size=128,
                eval_steps=8,
                transport_steps=2,
                layer_dims=(64, 16, 10),
            ).__dict__.items()
            if key
                not in {
                    "output_root",
                    "run_id",
                    "output_layout",
                    "warmup_epochs",
                    "epochs",
                    "batch_size",
                    "eval_steps",
                    "transport_steps",
                    "layer_dims",
            }
        },
    )

    config = _read_json(result.run_dir / "config.json")
    summary = _read_json(result.run_dir / "summary.json")
    epoch_rows = _read_csv(result.run_dir / "epoch_metrics.csv")

    assert summary["candidate_name"] == "stage05_v3c_endpoint_semigroup_consistency_contract"
    assert summary["endpoint_semigroup_consistency_enabled"] is True
    assert summary["semigroup_split_identity"] == "s = t + alpha * r; r_s = (1 - alpha) * r"
    assert summary["semigroup_target_mode"] == "single_sided_detached_split_endpoint"
    assert summary["semigroup_target_is_single_sided_detached"] is True
    assert summary["semigroup_target_contract"] == (
        "z_hat_split_target = stopgrad(z_hat_split); "
        "L_sg = || z_hat_direct - z_hat_split_target ||^2"
    )
    assert summary["pairwise_deltas_vs_promoted_refined_v3b_reference"]["status"] == (
        "pending_real_fixed_budget_v2_vs_promoted_v3b_vs_v3c_comparison"
    )
    assert summary["recommended_next_move"] == "run_fixed_budget_v2_vs_promoted_v3b_vs_v3c_comparison"

    assert config["transport"]["candidate_name"] == "stage05_v3c_endpoint_semigroup_consistency_contract"
    assert config["transport"]["endpoint_semigroup_consistency_enabled"] is True
    assert config["transport"]["semigroup_target_mode"] == "single_sided_detached_split_endpoint"
    assert config["transport"]["semigroup_target_is_single_sided_detached"] is True
    assert config["transport"]["lambda_sg"] == pytest.approx(0.05)

    assert "lambda_sg" in epoch_rows[0]
    assert "train_semigroup_loss" in epoch_rows[0]
    assert any(float(row["train_semigroup_loss"]) > 0.0 for row in epoch_rows)


def test_fused_v3c_probe_writes_expected_artifacts(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="exploratory_probe_v3c_fused_smoke",
        epochs=4,
        warmup_epochs=2,
        batch_size=128,
        eval_steps=8,
        transport_steps=2,
        layer_dims=(64, 16, 10),
        **{
            key: value
            for key, value in build_stage05_v3c_fused_trajectory_semigroup_contract_config(
                output_root=tmp_path,
                run_id="unused",
                epochs=4,
                batch_size=128,
                eval_steps=8,
                transport_steps=2,
                layer_dims=(64, 16, 10),
            ).__dict__.items()
            if key
            not in {
                "output_root",
                "run_id",
                "output_layout",
                "warmup_epochs",
                "epochs",
                "batch_size",
                "eval_steps",
                "transport_steps",
                "layer_dims",
            }
        },
    )

    config = _read_json(result.run_dir / "config.json")
    summary = _read_json(result.run_dir / "summary.json")
    epoch_rows = _read_csv(result.run_dir / "epoch_metrics.csv")

    assert summary["candidate_name"] == "stage05_v3c_fused_trajectory_semigroup_contract"
    assert summary["endpoint_semigroup_consistency_enabled"] is True
    assert summary["contract_fusion_enabled"] is True
    assert summary["semigroup_consistency_absorbed_into_main_trajectory_contract"] is True
    assert summary["semigroup_consistency_is_auxiliary_only"] is False
    assert summary["exact_detached_target_barycentric_fusion_enabled"] is True
    assert summary["main_trajectory_contract_identity"] == "exact_detached_target_barycentric_fusion"
    assert summary["pairwise_deltas_vs_active_refined_v3c_reference"]["status"] == (
        "pending_real_fixed_budget_v2_vs_active_v3c_vs_fused_contract_comparison"
    )
    assert summary["recommended_next_move"] == (
        "run_fixed_budget_v2_vs_active_v3c_vs_fused_contract_comparison"
    )

    assert config["transport"]["candidate_name"] == "stage05_v3c_fused_trajectory_semigroup_contract"
    assert config["transport"]["contract_fusion_enabled"] is True
    assert config["transport"]["semigroup_consistency_absorbed_into_main_trajectory_contract"] is True
    assert config["transport"]["semigroup_consistency_is_auxiliary_only"] is False
    assert config["transport"]["exact_detached_target_barycentric_fusion_enabled"] is True
    assert config["transport"]["main_trajectory_contract_identity"] == (
        "exact_detached_target_barycentric_fusion"
    )

    assert "train_main_traj_contract_loss" in epoch_rows[0]
    assert any(float(row["train_main_traj_contract_loss"]) > 0.0 for row in epoch_rows)


def test_corrected_residual_identity_target_includes_anchor_derivative_term() -> None:
    config = build_fmpc_ef_exploratory_probe_config(
        run_seed=0,
        data_seed=0,
        model_init_seed=0,
        psi_init_seed=0,
        batch_order_seed=0,
        tangent_epsilon=1e-3,
    )
    split = load_digits_split(
        split_seed=config.data_seed,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
    )
    x_batch = split.x_train[:8]
    y_batch = split.y_train[:8]
    model = _make_pc_model(config)
    psi_network = _make_psi_network(config)
    context = build_tf1_context(model, x_batch, y_batch)

    corrected = build_corrected_residual_identity_target(
        context,
        psi_network,
        context.z0,
        context.targets,
        t=0.25,
        r=0.75,
        tangent_epsilon=config.tangent_epsilon,
    )

    np.testing.assert_allclose(
        corrected.target,
        corrected.anchor_term + corrected.residual_term,
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        corrected.target - corrected.residual_term,
        corrected.anchor_term,
        atol=1e-12,
        rtol=1e-12,
    )
    assert np.linalg.norm(corrected.anchor_term) > 1e-12
    assert not np.allclose(corrected.target, corrected.residual_term)


def test_two_branch_identity_target_is_anchor_plus_traj_plus_state() -> None:
    config = build_fmpc_ef_exploratory_probe_config(
        run_seed=0,
        data_seed=0,
        model_init_seed=0,
        psi_init_seed=0,
        batch_order_seed=0,
        tangent_epsilon=1e-3,
        use_two_branch_residual_core=True,
        feature_aware_state_branch_tangents=True,
    )
    split = load_digits_split(
        split_seed=config.data_seed,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
    )
    x_batch = split.x_train[:8]
    y_batch = split.y_train[:8]
    model = _make_pc_model(config)
    psi_network = _make_psi_network(config)
    context = build_tf1_context(model, x_batch, y_batch)

    corrected = build_corrected_residual_identity_target(
        context,
        psi_network,
        context.z0,
        context.targets,
        t=0.25,
        r=0.75,
        tangent_epsilon=config.tangent_epsilon,
        feature_aware_state_branch_tangents=True,
    )

    np.testing.assert_allclose(
        corrected.target,
        corrected.anchor_term + corrected.trajectory_term + corrected.state_term,
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        corrected.residual_term,
        corrected.trajectory_term + corrected.state_term,
        atol=1e-12,
        rtol=1e-12,
    )
    assert np.linalg.norm(corrected.trajectory_term) > 1e-12
    assert np.linalg.norm(corrected.state_term) > 1e-12


def test_two_branch_state_tangent_is_not_zero_when_feature_aware_mode_is_enabled() -> None:
    config = build_fmpc_ef_exploratory_probe_config(
        run_seed=0,
        data_seed=0,
        model_init_seed=0,
        psi_init_seed=0,
        batch_order_seed=0,
        tangent_epsilon=1e-3,
        use_two_branch_residual_core=True,
        feature_aware_state_branch_tangents=True,
    )
    split = load_digits_split(
        split_seed=config.data_seed,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
    )
    x_batch = split.x_train[:8]
    y_batch = split.y_train[:8]
    model = _make_pc_model(config)
    context = build_tf1_context(model, x_batch, y_batch)
    features = teacher_free_state_features(context, context.z0)
    tangents = teacher_free_feature_tangents(
        context,
        context.z0,
        epsilon=config.tangent_epsilon,
    )

    state_tangent = build_state_branch_input_tangent(
        features,
        feature_aware_state_branch_tangents=True,
        feature_tangents=tangents,
    )

    assert np.linalg.norm(state_tangent) > 1e-12


def test_lambda_id_schedule_has_zero_warmup_and_positive_ramp() -> None:
    config = build_fmpc_ef_exploratory_probe_config(
        identity_loss_weight=0.2,
        lambda_id_warmup_epochs=2,
        lambda_id_ramp_epochs=3,
    )

    lambda_0 = lambda_id_for_epoch(config, 0)
    lambda_1 = lambda_id_for_epoch(config, 1)
    lambda_2 = lambda_id_for_epoch(config, 2)
    lambda_3 = lambda_id_for_epoch(config, 3)
    lambda_4 = lambda_id_for_epoch(config, 4)
    lambda_5 = lambda_id_for_epoch(config, 5)

    assert lambda_0 == pytest.approx(0.0)
    assert lambda_1 == pytest.approx(0.0)
    assert 0.0 < lambda_2 < lambda_3 < lambda_4 <= 0.2
    assert lambda_5 == pytest.approx(0.2)


def test_v3b_alpha_and_lambda_traj_schedule_are_explicit() -> None:
    config = build_fmpc_ef_exploratory_probe_config(
        use_two_branch_residual_core=True,
        use_explicit_transport_drift_decomposition=True,
        use_trajectory_curriculum_contract=True,
        alpha_floor=0.5,
        alpha_warmup_epochs=1,
        alpha_ramp_epochs=2,
        lambda_traj_curr=0.2,
    )

    alpha_0 = alpha_for_epoch(config, 0)
    alpha_1 = alpha_for_epoch(config, 1)
    alpha_2 = alpha_for_epoch(config, 2)
    alpha_3 = alpha_for_epoch(config, 3)
    lambda_0 = lambda_traj_curr_for_epoch(config, 0)
    lambda_1 = lambda_traj_curr_for_epoch(config, 1)
    lambda_2 = lambda_traj_curr_for_epoch(config, 2)
    lambda_3 = lambda_traj_curr_for_epoch(config, 3)

    assert alpha_0 == pytest.approx(1.0)
    assert 1.0 > alpha_1 > 0.5
    assert alpha_2 == pytest.approx(0.5)
    assert alpha_3 == pytest.approx(0.5)
    assert lambda_0 == pytest.approx(0.0)
    assert 0.0 < lambda_1 < lambda_2 <= 0.2
    assert lambda_3 == pytest.approx(0.2)


def test_v3c_lambda_sg_schedule_stays_zero_until_split_is_active() -> None:
    config = build_stage05_v3c_endpoint_semigroup_config(
        alpha_floor=0.5,
        alpha_warmup_epochs=1,
        alpha_ramp_epochs=2,
        lambda_sg=0.05,
    )

    assert lambda_sg_for_epoch(config, 0) == pytest.approx(0.0)
    assert lambda_sg_for_epoch(config, 1) == pytest.approx(0.05)
    assert lambda_sg_for_epoch(config, 2) == pytest.approx(0.05)


def test_exploratory_probe_is_deterministic_under_fixed_seeds(tmp_path: Path) -> None:
    run = load_run()
    result_a = run(
        output_root=tmp_path / "a",
        run_id="deterministic_a",
        epochs=4,
        warmup_epochs=2,
        batch_size=128,
        eval_steps=8,
        transport_steps=2,
        run_seed=7,
        data_seed=7,
        model_init_seed=7,
        psi_init_seed=7,
        batch_order_seed=7,
    )
    result_b = run(
        output_root=tmp_path / "b",
        run_id="deterministic_b",
        epochs=4,
        warmup_epochs=2,
        batch_size=128,
        eval_steps=8,
        transport_steps=2,
        run_seed=7,
        data_seed=7,
        model_init_seed=7,
        psi_init_seed=7,
        batch_order_seed=7,
    )

    summary_a = _read_json(result_a.run_dir / "summary.json")
    summary_b = _read_json(result_b.run_dir / "summary.json")

    assert summary_a["selected_epoch"] == summary_b["selected_epoch"]
    assert summary_a["val_accuracy"] == pytest.approx(summary_b["val_accuracy"])
    assert summary_a["test_accuracy"] == pytest.approx(summary_b["test_accuracy"])
    assert summary_a["val_output_mse"] == pytest.approx(summary_b["val_output_mse"])
    assert summary_a["test_output_mse"] == pytest.approx(summary_b["test_output_mse"])
    for step_name in ("one_step", "configured_steps"):
        metrics_a = summary_a["mechanism_metrics"][step_name]
        metrics_b = summary_b["mechanism_metrics"][step_name]
        assert metrics_a["transport_steps"] == metrics_b["transport_steps"]
        assert metrics_a["transported_final_energy"] == pytest.approx(
            metrics_b["transported_final_energy"]
        )
        assert metrics_a["energy_delta_vs_identity"] == pytest.approx(
            metrics_b["energy_delta_vs_identity"]
        )
        assert metrics_a["transported_final_fixed_point_residual_rms"] == pytest.approx(
            metrics_b["transported_final_fixed_point_residual_rms"]
        )
    assert summary_a["selected_epoch_lambda_id"] == pytest.approx(summary_b["selected_epoch_lambda_id"])
    assert summary_a["transport_family"] == summary_b["transport_family"]
    assert summary_a["residual_identity_mode"] == summary_b["residual_identity_mode"]


def test_exploratory_probe_smoke_improves_mechanism_metrics_vs_identity(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="exploratory_probe_mechanism_smoke",
        epochs=8,
        warmup_epochs=3,
        batch_size=128,
        eval_steps=12,
        transport_steps=2,
        layer_dims=(64, 16, 10),
    )

    summary = _read_json(result.run_dir / "summary.json")
    acceptance = summary["mechanism_acceptance"]
    mechanism = summary["mechanism_metrics"]

    assert acceptance["one_step_energy_decrease_vs_identity"] is True
    assert acceptance["configured_steps_fixed_point_residual_decrease_vs_identity"] is True
    assert mechanism["one_step"]["transported_final_energy"] < mechanism["one_step"]["identity_final_energy"]
    assert (
        mechanism["configured_steps"]["transported_final_fixed_point_residual_rms"]
        < mechanism["configured_steps"]["identity_final_fixed_point_residual_rms"]
    )
