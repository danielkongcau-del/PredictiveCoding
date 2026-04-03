from __future__ import annotations

import numpy as np

from pc.inference import (
    build_clamped_mask,
    initialize_states,
    run_inference,
    run_teacher_inference_export,
)
from pc.layers import PCLayerParams
from pc.models import PCNetwork
from pc.state_io import flatten_hidden_states


def make_teacher_layers() -> list[PCLayerParams]:
    return [
        PCLayerParams(
            weight=np.array([[0.25, -0.10], [0.15, 0.20]], dtype=np.float64),
            bias=np.array([0.05, -0.02], dtype=np.float64),
            sigma2=1.0,
            activation_name="tanh",
        ),
        PCLayerParams(
            weight=np.array([[0.30, -0.25], [0.10, 0.40]], dtype=np.float64),
            bias=np.array([0.01, -0.03], dtype=np.float64),
            sigma2=1.0,
            activation_name="identity",
        ),
    ]


def test_teacher_export_z0_matches_initialized_hidden_states_before_updates() -> None:
    layers = make_teacher_layers()
    x = np.array([[1.0, -0.5], [0.25, 0.75]], dtype=np.float64)
    y = np.array([[0.10, 0.20], [-0.20, 0.30]], dtype=np.float64)
    clamped_mask = build_clamped_mask(len(layers) + 1, mode="train")
    initial_states = initialize_states(layers, x, y=y, init="forward", mode="train")

    teacher = run_teacher_inference_export(
        layers,
        x,
        y=y,
        init="forward",
        mode="train",
        eta_x=0.15,
        steps=5,
        method="euler",
        record_trajectory=False,
    )

    expected_z0 = flatten_hidden_states(initial_states, clamped_mask)
    np.testing.assert_allclose(teacher.z0, expected_z0)
    np.testing.assert_allclose(teacher.initial_states[1], initial_states[1])
    assert teacher.z_trajectory is None


def test_teacher_export_z_star_matches_hidden_states_after_configured_inference() -> None:
    layers = make_teacher_layers()
    x = np.array([[1.0, -0.5], [0.25, 0.75]], dtype=np.float64)
    y = np.array([[0.10, 0.20], [-0.20, 0.30]], dtype=np.float64)
    clamped_mask = build_clamped_mask(len(layers) + 1, mode="train")
    initial_states = initialize_states(layers, x, y=y, init="forward", mode="train")
    inference_result = run_inference(
        initial_states,
        layers,
        clamped_mask,
        eta_x=0.15,
        steps=5,
        method="rk2",
        record_trace=True,
        record_state_trajectory=False,
    )

    teacher = run_teacher_inference_export(
        layers,
        x,
        y=y,
        init="forward",
        mode="train",
        eta_x=0.15,
        steps=5,
        method="rk2",
        record_trajectory=False,
    )

    expected_z_star = flatten_hidden_states(inference_result.states, clamped_mask)
    np.testing.assert_allclose(teacher.z_star, expected_z_star)
    np.testing.assert_allclose(teacher.final_states[1], inference_result.states[1])
    np.testing.assert_allclose(teacher.energy_trace, inference_result.energy_trace)


def test_teacher_export_optional_trajectory_has_expected_length_and_endpoints() -> None:
    layers = make_teacher_layers()
    x = np.array([[1.0, -0.5], [0.25, 0.75]], dtype=np.float64)
    y = np.array([[0.10, 0.20], [-0.20, 0.30]], dtype=np.float64)

    teacher = run_teacher_inference_export(
        layers,
        x,
        y=y,
        init="forward",
        mode="train",
        eta_x=0.15,
        steps=4,
        method="euler",
        record_trajectory=True,
    )

    assert teacher.z_trajectory is not None
    assert len(teacher.z_trajectory) == 5
    np.testing.assert_allclose(teacher.z_trajectory[0], teacher.z0)
    np.testing.assert_allclose(teacher.z_trajectory[-1], teacher.z_star)
    for z_t in teacher.z_trajectory:
        assert z_t.shape == teacher.z0.shape
        assert z_t.dtype == np.float64


def test_train_batch_teacher_export_is_backward_compatible_and_opt_in() -> None:
    model = PCNetwork(
        layers=make_teacher_layers(),
        eta_x=0.15,
        eta_w=0.01,
        eta_b=0.01,
        train_steps=4,
        eval_steps=4,
        inference_method="euler",
        state_init="forward",
    )
    x = np.array([[1.0, -0.5], [0.25, 0.75]], dtype=np.float64)
    y = np.array([[0.10, 0.20], [-0.20, 0.30]], dtype=np.float64)

    baseline_result = model.train_batch(x, y, compute_post_update_energy=False)
    assert baseline_result.teacher_export is None

    model_with_export = PCNetwork(
        layers=make_teacher_layers(),
        eta_x=0.15,
        eta_w=0.01,
        eta_b=0.01,
        train_steps=4,
        eval_steps=4,
        inference_method="euler",
        state_init="forward",
    )
    export_result = model_with_export.train_batch(
        x,
        y,
        compute_post_update_energy=False,
        return_teacher_export=True,
        record_teacher_trajectory=True,
    )
    assert export_result.teacher_export is not None
    assert export_result.teacher_export.steps == 4
    assert export_result.teacher_export.inference_method == "euler"
    assert export_result.teacher_export.z_trajectory is not None
    assert len(export_result.teacher_export.z_trajectory) == 5
