from __future__ import annotations

import numpy as np

from pc.datasets import load_digits_split
from pc.fmpc_tf1_flow import build_tf1_context
from pc.fmpc_tf2 import (
    _make_pc_model,
    _make_psi_network,
    _run_tf2_micro_step,
    _theta_micro_learning_rates,
    _train_one_batch_tf2,
    build_tf2_canonical_config,
)


def _weights_snapshot(model):
    return [layer.weight.copy() for layer in model.layers]


def _any_weight_changed(before, model) -> bool:
    return any(not np.allclose(old, layer.weight) for old, layer in zip(before, model.layers, strict=True))


def _psi_weights_snapshot(network):
    return [layer.weight.copy() for layer in network.layers]


def _any_psi_weight_changed(before, network) -> bool:
    return any(not np.allclose(old, layer.weight) for old, layer in zip(before, network.layers, strict=True))


def test_tf2_micro_step_order_respects_frozen_snapshot_sequence() -> None:
    split = load_digits_split(split_seed=0)
    config = build_tf2_canonical_config(
        layer_dims=(64, 16, 10),
        batch_size=4,
        micro_steps=2,
        incremental_weight_updates=True,
        model_init_seed=0,
        psi_init_seed=0,
    )
    model = _make_pc_model(config)
    psi = _make_psi_network(config)
    context = build_tf1_context(model, split.x_train[:4], split.y_train[:4])
    theta_before = _weights_snapshot(model)
    psi_before = _psi_weights_snapshot(psi)
    micro_eta_w, micro_eta_b = _theta_micro_learning_rates(config)
    event_log: list[str] = []

    _run_tf2_micro_step(
        model,
        psi,
        config,
        context,
        context.z0,
        context.z0,
        t_k=0.0,
        dt=0.5,
        r_k=1.0,
        lambda_id=0.0,
        apply_theta_update=True,
        theta_eta_w=micro_eta_w,
        theta_eta_b=micro_eta_b,
        event_log=event_log,
    )

    assert event_log == ["plan", "advance", "theta_update", "psi_update"]
    assert _any_weight_changed(theta_before, model)
    assert _any_psi_weight_changed(psi_before, psi)


def test_tf2_matched_budget_scales_micro_lr_and_terminal_theta_update_still_happens() -> None:
    split = load_digits_split(split_seed=1)
    config = build_tf2_canonical_config(
        layer_dims=(64, 16, 10),
        batch_size=8,
        micro_steps=4,
        incremental_weight_updates=False,
        theta_update_budget="matched",
        model_init_seed=1,
        psi_init_seed=1,
    )
    model = _make_pc_model(config)
    psi = _make_psi_network(config)
    theta_before = _weights_snapshot(model)
    micro_eta_w, micro_eta_b = _theta_micro_learning_rates(config)

    assert micro_eta_w == config.eta_w / config.micro_steps
    assert micro_eta_b == (config.eta_b if config.eta_b is not None else config.eta_w) / config.micro_steps

    _train_one_batch_tf2(
        model,
        psi,
        config,
        split.x_train[:8],
        split.y_train[:8],
        lambda_id=0.0,
    )

    assert _any_weight_changed(theta_before, model)
