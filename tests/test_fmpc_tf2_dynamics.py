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


def test_tf2_matched_budget_scales_by_applied_theta_updates_and_terminal_theta_update_still_happens() -> None:
    split = load_digits_split(split_seed=1)
    terminal_config = build_tf2_canonical_config(
        layer_dims=(64, 16, 10),
        batch_size=8,
        micro_steps=4,
        incremental_weight_updates=False,
        theta_update_budget="matched",
        model_init_seed=1,
        psi_init_seed=1,
    )
    model = _make_pc_model(terminal_config)
    psi = _make_psi_network(terminal_config)
    theta_before = _weights_snapshot(model)
    micro_eta_w, micro_eta_b = _theta_micro_learning_rates(terminal_config, "terminal_only")

    assert micro_eta_w == terminal_config.eta_w
    assert micro_eta_b == (terminal_config.eta_b if terminal_config.eta_b is not None else terminal_config.eta_w)

    _train_one_batch_tf2(
        model,
        psi,
        terminal_config,
        split.x_train[:8],
        split.y_train[:8],
        lambda_id=0.0,
    )

    assert _any_weight_changed(theta_before, model)

    every_two_config = build_tf2_canonical_config(
        layer_dims=(64, 16, 10),
        batch_size=8,
        micro_steps=4,
        incremental_weight_updates=True,
        theta_update_budget="matched",
        theta_update_cadence="every_2_micro_steps",
        model_init_seed=1,
        psi_init_seed=1,
    )
    every_two_eta_w, every_two_eta_b = _theta_micro_learning_rates(
        every_two_config,
        "every_2_micro_steps",
    )
    assert every_two_eta_w == every_two_config.eta_w / 2.0
    assert every_two_eta_b == (
        (every_two_config.eta_b if every_two_config.eta_b is not None else every_two_config.eta_w)
        / 2.0
    )

    every_step_config = build_tf2_canonical_config(
        layer_dims=(64, 16, 10),
        batch_size=8,
        micro_steps=4,
        incremental_weight_updates=True,
        theta_update_budget="matched",
        theta_update_cadence="every_micro_step",
        model_init_seed=1,
        psi_init_seed=1,
    )
    every_step_eta_w, every_step_eta_b = _theta_micro_learning_rates(
        every_step_config,
        "every_micro_step",
    )
    assert every_step_eta_w == every_step_config.eta_w / every_step_config.micro_steps
    assert every_step_eta_b == (
        (every_step_config.eta_b if every_step_config.eta_b is not None else every_step_config.eta_w)
        / every_step_config.micro_steps
    )
