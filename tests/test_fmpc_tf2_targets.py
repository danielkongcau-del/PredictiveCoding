from __future__ import annotations

from pc.datasets import load_digits_split
from pc.fmpc_tf1_flow import build_tf1_context
from pc.fmpc_tf2 import _make_pc_model, _make_psi_network, _plan_tf2_micro_step, build_tf2_canonical_config


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
