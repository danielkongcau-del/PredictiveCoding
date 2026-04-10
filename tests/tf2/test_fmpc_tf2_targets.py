from __future__ import annotations

import numpy as np

from pc.datasets import load_digits_split
from pc.tf1.fmpc_tf1_flow import FMPCTF1StateFeatureTangents, build_tf1_context
from pc.tf1.fmpc_tf1_jvp import build_tf1_input_tangent, resolve_tf1_identity_tangent_mode
from pc.tf2.fmpc_tf2 import _make_pc_model, _make_psi_network, _plan_tf2_micro_step, build_tf2_canonical_config


def test_tf2_supervision_policy_local_only_uses_only_local_field_source() -> None:
    split = load_digits_split(split_seed=0)
    config = build_tf2_canonical_config(
        layer_dims=(64, 16, 10),
        batch_size=4,
        micro_steps=2,
        supervision_policy="local_only",
        model_init_seed=0,
        psi_init_seed=0,
    )
    model = _make_pc_model(config)
    psi = _make_psi_network(config)
    context = build_tf1_context(model, split.x_train[:4], split.y_train[:4])

    plan = _plan_tf2_micro_step(
        context,
        psi,
        config,
        context.z0,
        context.z0,
        t_k=0.0,
        dt=0.5,
        r_k=1.0,
    )

    assert plan.source_counts["local_field_only"] == 4
    assert plan.source_counts["learned_on_policy"] == 0
    assert plan.psi_inputs.shape[0] == 4
    assert plan.boot_targets.shape == context.z0.shape
    assert plan.identity_targets.shape == context.z0.shape
    assert plan.z_on_next.shape == context.z0.shape
    assert plan.z_lf_next.shape == context.z0.shape


def test_tf2_supervision_policy_mixed_includes_local_and_on_policy_sources() -> None:
    split = load_digits_split(split_seed=1)
    config = build_tf2_canonical_config(
        layer_dims=(64, 16, 10),
        batch_size=4,
        micro_steps=2,
        supervision_policy="mixed",
        model_init_seed=1,
        psi_init_seed=1,
    )
    model = _make_pc_model(config)
    psi = _make_psi_network(config)
    context = build_tf1_context(model, split.x_train[:4], split.y_train[:4])

    plan = _plan_tf2_micro_step(
        context,
        psi,
        config,
        context.z0,
        context.z0,
        t_k=0.0,
        dt=0.5,
        r_k=1.0,
    )

    assert plan.source_counts["local_field_only"] == 4
    assert plan.source_counts["learned_on_policy"] == 4
    assert plan.psi_inputs.shape[0] == 8
    assert plan.boot_targets.shape[0] == 8
    assert plan.identity_targets.shape[0] == 8


def test_tf2_onpolicy_mix_ratio_025_subsamples_onpolicy_source_deterministically() -> None:
    split = load_digits_split(split_seed=2)
    config = build_tf2_canonical_config(
        layer_dims=(64, 16, 10),
        batch_size=4,
        micro_steps=2,
        supervision_policy="mixed",
        onpolicy_mix_ratio=0.25,
        model_init_seed=2,
        psi_init_seed=2,
    )
    model = _make_pc_model(config)
    psi = _make_psi_network(config)
    context = build_tf1_context(model, split.x_train[:4], split.y_train[:4])

    plan = _plan_tf2_micro_step(
        context,
        psi,
        config,
        context.z0,
        context.z0,
        t_k=0.0,
        dt=0.5,
        r_k=1.0,
    )

    assert plan.source_counts["local_field_only"] == 4
    assert plan.source_counts["learned_on_policy"] == 1
    assert plan.psi_inputs.shape[0] == 5
    assert plan.boot_targets.shape[0] == 5
    assert plan.identity_targets.shape[0] == 5


def test_tf2_identity_tangent_mode_and_feature_block_semantics_are_explicit() -> None:
    g_t = np.array([[1.0, -2.0]], dtype=np.float64)
    feature_tangents = FMPCTF1StateFeatureTangents(
        Dg_g_t=np.array([[0.5, 0.25]], dtype=np.float64),
        Dg_e_out_t=np.array([[0.75, -0.5, 0.25]], dtype=np.float64),
        Dg_F_t=np.array([[1.25]], dtype=np.float64),
        Dg_y_hat_t=np.zeros((1, 3), dtype=np.float64),
    )

    truncated_tangent = build_tf1_input_tangent(
        g_t,
        target_dim=3,
        use_teacher_free_features=True,
        feature_aware_tangents=False,
        feature_tangents=feature_tangents,
    )
    feature_aware_tangent = build_tf1_input_tangent(
        g_t,
        target_dim=3,
        use_teacher_free_features=True,
        feature_aware_tangents=True,
        feature_tangents=feature_tangents,
    )

    expected_feature_block = np.concatenate(
        [
            feature_tangents.Dg_g_t,
            feature_tangents.Dg_e_out_t,
            feature_tangents.Dg_F_t,
        ],
        axis=1,
    )
    feature_offset = g_t.shape[1] + 3 + 2
    assert np.allclose(truncated_tangent[:, feature_offset:], 0.0)
    assert np.allclose(feature_aware_tangent[:, feature_offset:], expected_feature_block)
    assert resolve_tf1_identity_tangent_mode(
        use_teacher_free_features=True,
        feature_aware_tangents=False,
    ) == "feature_frozen_truncated_identity_approx"
    assert resolve_tf1_identity_tangent_mode(
        use_teacher_free_features=True,
        feature_aware_tangents=True,
    ) == "feature_aware_total_derivative_approx"
