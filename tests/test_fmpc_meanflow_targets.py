from __future__ import annotations

import numpy as np

from pc.fmpc_interval_normalization import FMPCIntervalNormalizationStats
from pc.fmpc_meanflow_student import (
    MeanFlowMLPStudent,
    MeanFlowMLPStudentConfig,
    MeanFlowTwoBranchStudentConfig,
    build_meanflow_full_identity_target_raw,
    build_meanflow_identity_target,
    meanflow_identity_active_mask,
    reconstruct_meanflow_velocity,
)


def _normalization() -> FMPCIntervalNormalizationStats:
    return FMPCIntervalNormalizationStats(
        z_state_mean=np.zeros((2,), dtype=np.float64),
        z_state_std=np.ones((2,), dtype=np.float64),
        u_mean=np.asarray([0.5, -0.5], dtype=np.float64),
        u_std=np.asarray([2.0, 4.0], dtype=np.float64),
        eps=1e-8,
    )


def test_build_meanflow_identity_target_for_direct_u_matches_expected_raw_formula() -> None:
    normalization = _normalization()
    g_s = np.asarray([[1.0, -2.0], [0.5, 0.25]], dtype=np.float64)
    delta_tau = np.asarray([[0.5], [0.25]], dtype=np.float64)
    predicted_target_normalized = np.zeros_like(g_s)
    jvp_normalized = np.asarray([[0.25, -0.5], [0.1, 0.2]], dtype=np.float64)

    target = build_meanflow_identity_target(
        normalization,
        g_s=g_s,
        dg_s=None,
        delta_tau=delta_tau,
        predicted_target_normalized=predicted_target_normalized,
        jvp_normalized=jvp_normalized,
        target_mode="u_star",
    )

    raw_jvp = jvp_normalized * normalization.u_std
    expected_raw = g_s + delta_tau * raw_jvp
    expected_normalized = normalization.transform_u(expected_raw)
    np.testing.assert_allclose(target, expected_normalized, atol=1e-12, rtol=1e-12)


def test_build_meanflow_identity_target_for_residual_matches_expected_raw_formula() -> None:
    normalization = _normalization()
    g_s = np.asarray([[1.0, -2.0], [0.5, 0.25]], dtype=np.float64)
    dg_s = np.asarray([[0.2, -0.1], [0.4, 0.3]], dtype=np.float64)
    delta_tau = np.asarray([[0.5], [0.25]], dtype=np.float64)
    predicted_target_normalized = np.zeros_like(g_s)
    jvp_normalized = np.asarray([[0.25, -0.5], [0.1, 0.2]], dtype=np.float64)

    target = build_meanflow_identity_target(
        normalization,
        g_s=g_s,
        dg_s=dg_s,
        delta_tau=delta_tau,
        predicted_target_normalized=predicted_target_normalized,
        jvp_normalized=jvp_normalized,
        target_mode="u_residual_local_field",
    )

    raw_jvp = jvp_normalized * normalization.u_std
    expected_raw = delta_tau * (dg_s + raw_jvp)
    expected_normalized = normalization.transform_u(expected_raw)
    np.testing.assert_allclose(target, expected_normalized, atol=1e-12, rtol=1e-12)


def test_build_meanflow_full_identity_target_raw_matches_expected_formula() -> None:
    g_s = np.asarray([[1.0, -2.0], [0.5, 0.25]], dtype=np.float64)
    delta_tau = np.asarray([[0.5], [0.25]], dtype=np.float64)
    combined_jvp = np.asarray([[0.2, -0.4], [0.3, 0.6]], dtype=np.float64)

    target = build_meanflow_full_identity_target_raw(
        g_s=g_s,
        delta_tau=delta_tau,
        combined_jvp_raw=combined_jvp,
    )

    expected = g_s + delta_tau * combined_jvp
    np.testing.assert_allclose(target, expected, atol=1e-12, rtol=1e-12)


def test_reconstruct_meanflow_velocity_rebuilds_direct_and_residual_predictions() -> None:
    g_s = np.asarray([[1.0, -2.0], [0.5, 0.25]], dtype=np.float64)
    predicted_target = np.asarray([[0.3, -0.4], [0.2, 0.1]], dtype=np.float64)

    direct = reconstruct_meanflow_velocity(g_s, predicted_target, target_mode="u_star")
    residual = reconstruct_meanflow_velocity(g_s, predicted_target, target_mode="u_residual_local_field")

    np.testing.assert_allclose(direct, predicted_target, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(residual, g_s + predicted_target, atol=1e-12, rtol=1e-12)


def test_residual_meanflow_identity_requires_dg_s() -> None:
    normalization = _normalization()
    g_s = np.asarray([[1.0, -2.0]], dtype=np.float64)
    delta_tau = np.asarray([[0.5]], dtype=np.float64)
    predicted_target_normalized = np.zeros_like(g_s)
    jvp_normalized = np.zeros_like(g_s)

    try:
        build_meanflow_identity_target(
            normalization,
            g_s=g_s,
            dg_s=None,
            delta_tau=delta_tau,
            predicted_target_normalized=predicted_target_normalized,
            jvp_normalized=jvp_normalized,
            target_mode="u_residual_local_field",
        )
    except ValueError as exc:
        assert "dg_s" in str(exc)
    else:
        raise AssertionError("Residual MeanFlow identity should require dg_s.")


def test_meanflow_identity_active_mask_can_restrict_identity_to_acceptance_segments() -> None:
    source_step_indices = np.asarray([0, 15, 0, 10, 4], dtype=np.int64)
    target_step_indices = np.asarray([15, 30, 30, 20, 9], dtype=np.int64)

    all_mask = meanflow_identity_active_mask(
        source_step_indices,
        target_step_indices,
        teacher_steps=30,
        identity_scope_mode="all_intervals",
    )
    knot_mask = meanflow_identity_active_mask(
        source_step_indices,
        target_step_indices,
        teacher_steps=30,
        identity_scope_mode="acceptance_schedule_segments_only",
    )

    np.testing.assert_array_equal(all_mask, np.asarray([True, True, True, True, True]))
    np.testing.assert_array_equal(knot_mask, np.asarray([True, True, False, True, False]))


def test_meanflow_linear_residual_initialization_is_deterministic() -> None:
    normalization = FMPCIntervalNormalizationStats(
        z_state_mean=np.zeros((2,), dtype=np.float64),
        z_state_std=np.ones((2,), dtype=np.float64),
        u_mean=np.zeros((2,), dtype=np.float64),
        u_std=np.ones((2,), dtype=np.float64),
        teacher_feature_mean=np.zeros((5,), dtype=np.float64),
        teacher_feature_std=np.ones((5,), dtype=np.float64),
        teacher_feature_names=("g_s", "e_out_s", "F_s"),
        eps=1e-8,
    )
    config = MeanFlowMLPStudentConfig(
        hidden_dims=(),
        family_name="meanflow_linear_residual",
        target_mode="u_residual_local_field",
        identity_loss_weight=0.1,
    )
    model_a = MeanFlowMLPStudent.initialize(
        z_dim=2,
        target_dim=3,
        normalization=normalization,
        config=config,
        seed=11,
        teacher_model=object(),
        teacher_feature_bundle=object(),
    )
    model_b = MeanFlowMLPStudent.initialize(
        z_dim=2,
        target_dim=3,
        normalization=normalization,
        config=config,
        seed=11,
        teacher_model=object(),
        teacher_feature_bundle=object(),
    )

    assert len(model_a.network.layers) == 1
    np.testing.assert_allclose(model_a.network.layers[0].weight, model_b.network.layers[0].weight)
    np.testing.assert_allclose(model_a.network.layers[0].bias, model_b.network.layers[0].bias)


def test_meanflow_twobranch_config_requires_g_s_in_local_branch_features() -> None:
    try:
        MeanFlowTwoBranchStudentConfig(
            correction_hidden_dims=(8,),
            local_branch_feature_names=("e_out_s", "F_s"),
            identity_loss_weight=0.1,
        )
    except ValueError as exc:
        assert "g_s" in str(exc)
    else:
        raise AssertionError("Two-branch config should require g_s in local_branch_feature_names.")


def test_meanflow_twobranch_warmstart_config_requires_positive_stage_a_epochs() -> None:
    try:
        MeanFlowTwoBranchStudentConfig(
            correction_hidden_dims=(8,),
            family_name="meanflow_twobranch_residual_warmstart",
            identity_loss_weight=0.1,
            correction_only_warmup_epochs=0,
            local_branch_warm_start=True,
        )
    except ValueError as exc:
        assert "correction_only_warmup_epochs" in str(exc)
    else:
        raise AssertionError("Warm-start two-branch config should require a positive Stage A length.")
