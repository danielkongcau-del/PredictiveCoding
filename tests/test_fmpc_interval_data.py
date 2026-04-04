from __future__ import annotations

from pathlib import Path

import numpy as np

from pc.fmpc_interval_data import (
    acceptance_schedule_focus_pairs,
    build_fmpc_interval_inputs,
    iter_all_interval_blocks,
    iter_mixed_interval_batches,
    iter_weighted_interval_blocks,
    load_fmpc_interval_dataset,
    sample_balanced_interval_batch,
    teacher_step_aligned_rollout_schedules,
)
from pc.fmpc_protocol import FMPCPreparationConfig, run_fmpc_v0_preparation
from pc.real_pc import RealPCConfig


def _prepare_interval_teacher_artifact(tmp_path: Path, run_id: str) -> Path:
    result = run_fmpc_v0_preparation(
        FMPCPreparationConfig(
            dataset_name="digits",
            output_root=tmp_path,
            run_id=run_id,
            teacher_pc_config=RealPCConfig(
                dataset_name="digits",
                layer_dims=(64, 16, 10),
                epochs=1,
                batch_size=256,
                train_steps=3,
                eval_steps=3,
            ),
            teacher_export_steps=3,
            teacher_export_batch_size=256,
            export_trajectory=True,
        )
    )
    return result.run_dir


def test_fmpc_interval_dataset_loads_trajectory_contract(tmp_path: Path) -> None:
    teacher_dir = _prepare_interval_teacher_artifact(tmp_path, "interval_contract")
    dataset = load_fmpc_interval_dataset(teacher_dir, expected_dataset_name="digits")

    assert dataset.dataset_name == "digits"
    assert dataset.teacher_steps == 3
    assert dataset.interval_input_definition == "concat([z_s, target_onehot, tau_s, tau_t])"
    assert dataset.interval_target_definition == "u_star = (z_t - z_s) / (tau_t - tau_s)"
    assert dataset.metadata["trajectory_includes_endpoints"] is True
    assert dataset.metadata["trajectory_axis_semantics"] == "(batch, step, z_dim)"
    assert dataset.metadata["tau_definition"] == "tau_k = k / teacher_steps"
    assert dataset.train.z_trajectory.shape[1] == dataset.teacher_steps + 1
    np.testing.assert_allclose(dataset.train.z_trajectory[:, 0, :], dataset.train.z0, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(dataset.train.z_trajectory[:, -1, :], dataset.train.z_star, atol=1e-12, rtol=1e-12)


def test_build_fmpc_interval_inputs_shape_contract() -> None:
    z_s = np.zeros((4, 3), dtype=np.float64)
    target_onehot = np.eye(2, dtype=np.float64)[[0, 1, 0, 1]]
    tau_s = np.asarray([0.0, 0.25, 0.5, 0.75], dtype=np.float64)
    tau_t = np.asarray([0.25, 0.5, 0.75, 1.0], dtype=np.float64)

    inputs = build_fmpc_interval_inputs(z_s, target_onehot, tau_s, tau_t)

    assert inputs.shape == (4, 3 + 2 + 2)
    assert inputs.dtype == np.float64
    np.testing.assert_allclose(inputs[:, -2], tau_s, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(inputs[:, -1], tau_t, atol=0.0, rtol=0.0)


def test_interval_pair_sampler_returns_balanced_batch_contract(tmp_path: Path) -> None:
    teacher_dir = _prepare_interval_teacher_artifact(tmp_path, "interval_sampler")
    dataset = load_fmpc_interval_dataset(teacher_dir, expected_dataset_name="digits")

    batch = sample_balanced_interval_batch(dataset.train, batch_size=32, seed=123)

    assert batch.sample_row_indices.shape == (32,)
    assert batch.source_step_indices.shape == (32,)
    assert batch.target_step_indices.shape == (32,)
    assert batch.span_lengths.shape == (32,)
    assert batch.target_onehot.shape == (32, dataset.target_dim)
    assert batch.z_s.shape == (32, dataset.z_dim)
    assert batch.z_t.shape == (32, dataset.z_dim)
    assert batch.tau_s.shape == (32, 1)
    assert batch.tau_t.shape == (32, 1)
    assert batch.delta_tau.shape == (32, 1)
    assert batch.u_star.shape == (32, dataset.z_dim)
    assert batch.student_inputs.shape == (32, dataset.z_dim + dataset.target_dim + 2)
    assert batch.z_s.dtype == np.float64
    assert np.all(batch.span_lengths >= 1)
    assert np.all(batch.span_lengths <= dataset.teacher_steps)
    assert np.all(batch.target_step_indices > batch.source_step_indices)
    np.testing.assert_allclose(batch.delta_tau, batch.tau_t - batch.tau_s, atol=1e-12, rtol=1e-12)


def test_iter_all_interval_blocks_is_span_balanced(tmp_path: Path) -> None:
    teacher_dir = _prepare_interval_teacher_artifact(tmp_path, "interval_span_balance")
    dataset = load_fmpc_interval_dataset(teacher_dir, expected_dataset_name="digits")

    total_weight_by_span: dict[int, float] = {}
    for block in iter_all_interval_blocks(dataset.train):
        total_weight_by_span.setdefault(int(block["span_length"]), 0.0)
        total_weight_by_span[int(block["span_length"])] += float(block["pair_weight"])

    expected_spans = set(range(1, dataset.teacher_steps + 1))
    assert set(total_weight_by_span.keys()) == expected_spans
    per_span_weights = np.asarray([total_weight_by_span[span] for span in sorted(expected_spans)], dtype=np.float64)
    np.testing.assert_allclose(per_span_weights, per_span_weights[0], atol=1e-12, rtol=1e-12)


def test_teacher_step_aligned_rollout_schedules_use_teacher_knots() -> None:
    schedules = teacher_step_aligned_rollout_schedules(6)
    assert schedules["1-step"] == (0, 6)
    assert schedules["2-step"] == (0, 3, 6)
    assert schedules["3-step"] == (0, 2, 4, 6)


def test_acceptance_schedule_focus_pairs_match_2step_and_3step_segments() -> None:
    focus_pairs = acceptance_schedule_focus_pairs(30, schedule_names=("2-step", "3-step"))
    assert focus_pairs == ((0, 15), (15, 30), (0, 10), (10, 20), (20, 30))


def test_weighted_interval_blocks_add_explicit_knot_focus_mass(tmp_path: Path) -> None:
    teacher_dir = _prepare_interval_teacher_artifact(tmp_path, "interval_weighted_blocks")
    dataset = load_fmpc_interval_dataset(teacher_dir, expected_dataset_name="digits")

    base_weight = 0.0
    focused_weight = 0.0
    focus_pairs = set(acceptance_schedule_focus_pairs(dataset.teacher_steps))
    for block in iter_weighted_interval_blocks(
        dataset.train,
        knot_focused_schedule_names=("2-step", "3-step"),
        knot_focus_mixture=0.5,
    ):
        pair = (int(block["source_index"]), int(block["target_index"]))
        if pair in focus_pairs:
            focused_weight += float(block["pair_weight"])
        base_weight += float(block["pair_weight"])

    np.testing.assert_allclose(base_weight, 1.0, atol=1e-12, rtol=1e-12)
    assert focused_weight > 0.5


def test_mixed_interval_batches_can_focus_only_on_acceptance_segments(tmp_path: Path) -> None:
    teacher_dir = _prepare_interval_teacher_artifact(tmp_path, "interval_mixed_batches")
    dataset = load_fmpc_interval_dataset(teacher_dir, expected_dataset_name="digits")
    focus_pairs = set(acceptance_schedule_focus_pairs(dataset.teacher_steps))

    batches = list(
        iter_mixed_interval_batches(
            dataset.train,
            batch_size=32,
            num_batches=2,
            seed=123,
            knot_focused_schedule_names=("2-step", "3-step"),
            knot_focus_probability=1.0,
        )
    )
    assert len(batches) == 2
    for batch in batches:
        observed_pairs = set(
            zip(
                batch.source_step_indices.tolist(),
                batch.target_step_indices.tolist(),
                strict=True,
            )
        )
        assert observed_pairs.issubset(focus_pairs)
