from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pc.fmpc_interval_normalization import FMPCIntervalNormalizationStats
from pc.fmpc_meanflow_student import (
    MeanFlowTwoBranchResidualStudent,
    MeanFlowTwoBranchStudentConfig,
)


def _normalization() -> FMPCIntervalNormalizationStats:
    return FMPCIntervalNormalizationStats(
        z_state_mean=np.zeros((2,), dtype=np.float64),
        z_state_std=np.ones((2,), dtype=np.float64),
        u_mean=np.zeros((2,), dtype=np.float64),
        u_std=np.ones((2,), dtype=np.float64),
        teacher_feature_mean=np.zeros((6,), dtype=np.float64),
        teacher_feature_std=np.ones((6,), dtype=np.float64),
        teacher_feature_names=("g_s", "e_out_s", "F_s"),
        eps=1e-8,
    )


def test_meanflow_twobranch_initialization_starts_near_local_branch_solution() -> None:
    model = MeanFlowTwoBranchResidualStudent.initialize(
        z_dim=2,
        target_dim=3,
        normalization=_normalization(),
        config=MeanFlowTwoBranchStudentConfig(
            correction_hidden_dims=(8,),
            identity_loss_weight=0.1,
        ),
        seed=5,
        teacher_model=object(),
        teacher_feature_bundle=object(),
    )

    local_inputs = np.asarray(
        [
            [1.0, -2.0, 0.3, -0.1, 0.2, 0.5],
            [0.25, 0.75, -0.2, 0.4, -0.3, 0.1],
        ],
        dtype=np.float64,
    )
    u_local = model.local_network.predict(local_inputs)
    u_corr = model.correction_network.predict(np.zeros((2, 13), dtype=np.float64))

    np.testing.assert_allclose(u_local, local_inputs[:, :2], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(u_corr, np.zeros((2, 2), dtype=np.float64), atol=1e-12, rtol=1e-12)


def test_meanflow_twobranch_forward_contract_keeps_feature_aware_local_tangent_active() -> None:
    model = MeanFlowTwoBranchResidualStudent.initialize(
        z_dim=2,
        target_dim=3,
        normalization=_normalization(),
        config=MeanFlowTwoBranchStudentConfig(
            correction_hidden_dims=(8,),
            identity_loss_weight=0.1,
        ),
        seed=7,
        teacher_model=object(),
        teacher_feature_bundle=object(),
    )

    correction_feature_matrix = np.zeros((2, 6), dtype=np.float64)
    correction_feature_tangent = np.zeros((2, 6), dtype=np.float64)
    local_feature_matrix = np.asarray(
        [
            [1.0, -2.0, 0.1, 0.2, -0.1, 0.5],
            [0.25, 0.75, -0.2, 0.4, -0.3, 0.1],
        ],
        dtype=np.float64,
    )
    local_feature_tangent = np.asarray(
        [
            [0.4, -0.5, 9.0, 9.0, 9.0, 9.0],
            [0.1, 0.2, 8.0, 8.0, 8.0, 8.0],
        ],
        dtype=np.float64,
    )
    forward = model._forward_branches(
        z_s=np.zeros((2, 2), dtype=np.float64),
        target_onehot=np.zeros((2, 3), dtype=np.float64),
        tau_s=np.zeros((2, 1), dtype=np.float64),
        tau_t=np.ones((2, 1), dtype=np.float64),
        correction_feature_matrix=correction_feature_matrix,
        correction_feature_tangent_matrix=correction_feature_tangent,
        local_feature_matrix=local_feature_matrix,
        local_feature_tangent_matrix=local_feature_tangent,
        g_s=local_feature_matrix[:, :2],
    )

    np.testing.assert_allclose(forward["u_local"], local_feature_matrix[:, :2], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(forward["u_corr"], np.zeros((2, 2), dtype=np.float64), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(forward["u_hat"], forward["u_local"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        forward["u_local_jvp"],
        local_feature_tangent[:, :2],
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(forward["u_hat_jvp"], forward["u_local_jvp"], atol=1e-12, rtol=1e-12)


def test_meanflow_twobranch_warmstart_applies_local_branch_parameters() -> None:
    warm_weight = np.asarray([[2.0, 0.0, 0.5, 0.0, 0.0, -1.0], [0.0, -3.0, 0.0, 0.25, 1.5, 0.0]], dtype=np.float64)
    warm_bias = np.asarray([0.2, -0.4], dtype=np.float64)
    model = MeanFlowTwoBranchResidualStudent.initialize(
        z_dim=2,
        target_dim=3,
        normalization=_normalization(),
        config=MeanFlowTwoBranchStudentConfig(
            correction_hidden_dims=(8,),
            family_name="meanflow_twobranch_residual_warmstart",
            identity_loss_weight=0.1,
            correction_only_warmup_epochs=3,
            local_branch_warm_start=True,
        ),
        seed=9,
        teacher_model=object(),
        teacher_feature_bundle=object(),
        local_branch_warm_start=(warm_weight, warm_bias),
    )

    local_inputs = np.asarray([[1.0, -2.0, 0.1, 0.2, -0.3, 0.4]], dtype=np.float64)
    expected = local_inputs @ warm_weight.T + warm_bias
    np.testing.assert_allclose(model.local_network.predict(local_inputs), expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        model.correction_network.predict(np.zeros((1, 13), dtype=np.float64)),
        np.zeros((1, 2), dtype=np.float64),
        atol=1e-12,
        rtol=1e-12,
    )


def test_meanflow_twobranch_train_batch_can_freeze_local_branch_in_stage_a() -> None:
    model = MeanFlowTwoBranchResidualStudent.initialize(
        z_dim=2,
        target_dim=3,
        normalization=_normalization(),
        config=MeanFlowTwoBranchStudentConfig(
            correction_hidden_dims=(8,),
            family_name="meanflow_twobranch_residual_warmstart",
            identity_loss_weight=0.1,
            correction_only_warmup_epochs=2,
            local_branch_warm_start=True,
        ),
        seed=11,
        teacher_model=object(),
        teacher_feature_bundle=SimpleNamespace(teacher_steps=30),
    )
    local_before = model.local_network.layers[0].weight.copy()
    correction_before = [(layer.weight.copy(), layer.bias.copy()) for layer in model.correction_network.layers]

    batch = SimpleNamespace(
        z_s=np.zeros((2, 2), dtype=np.float64),
        target_onehot=np.zeros((2, 3), dtype=np.float64),
        tau_s=np.zeros((2, 1), dtype=np.float64),
        tau_t=np.ones((2, 1), dtype=np.float64),
        delta_tau=np.ones((2, 1), dtype=np.float64),
        u_star=np.asarray([[1.0, -1.0], [0.5, 0.25]], dtype=np.float64),
        sample_row_indices=np.asarray([0, 1], dtype=np.int64),
        source_step_indices=np.asarray([0, 10], dtype=np.int64),
        target_step_indices=np.asarray([15, 20], dtype=np.int64),
    )
    state_features = SimpleNamespace(g_s=np.asarray([[1.0, -2.0], [0.25, 0.75]], dtype=np.float64))
    zero_tangent = np.zeros((2, 6), dtype=np.float64)
    model._trajectory_batch_features = lambda split_name, batch_obj: (  # type: ignore[method-assign]
        np.zeros((2, 6), dtype=np.float64),
        zero_tangent,
        np.asarray([[1.0, -2.0, 0.0, 0.0, 0.0, 0.0], [0.25, 0.75, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        zero_tangent,
        state_features,
        SimpleNamespace(Dg_g_s=np.zeros((2, 2), dtype=np.float64)),
    )
    model._forward_branches = lambda **kwargs: {  # type: ignore[method-assign]
        "correction_inputs": np.zeros((2, 13), dtype=np.float64),
        "local_inputs": kwargs["local_feature_matrix"],
        "u_local": kwargs["local_feature_matrix"][:, :2],
        "u_corr": np.zeros((2, 2), dtype=np.float64),
        "u_hat": kwargs["local_feature_matrix"][:, :2],
        "u_local_jvp": np.zeros((2, 2), dtype=np.float64),
        "u_corr_jvp": np.zeros((2, 2), dtype=np.float64),
        "u_hat_jvp": np.zeros((2, 2), dtype=np.float64),
    }

    diagnostics = model.train_batch(
        batch,
        split_name="train",
        teacher_loss_weight=1.0,
        identity_loss_weight=0.1,
        train_local_branch=False,
        train_correction_branch=True,
    )

    np.testing.assert_allclose(model.local_network.layers[0].weight, local_before, atol=1e-12, rtol=1e-12)
    assert diagnostics["train_local_branch"] is False
    assert diagnostics["train_correction_branch"] is True
    assert diagnostics["local_teacher_update_loss"] is None
    assert diagnostics["correction_teacher_update_loss"] is not None
    assert any(
        (not np.allclose(layer.weight, before_weight)) or (not np.allclose(layer.bias, before_bias))
        for layer, (before_weight, before_bias) in zip(model.correction_network.layers, correction_before, strict=True)
    )
