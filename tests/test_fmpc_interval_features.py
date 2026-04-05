from __future__ import annotations

from pathlib import Path

import numpy as np

from pc.fmpc_interval_data import load_fmpc_interval_dataset
from pc.fmpc_interval_features import (
    compute_interval_teacher_state_features,
    prepare_interval_teacher_feature_context,
    precompute_interval_teacher_trajectory_features,
)
from pc.fmpc_protocol import FMPCPreparationConfig, load_prepared_teacher_runtime, run_fmpc_v0_preparation
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


def test_current_state_teacher_features_have_expected_shapes_and_dtype(tmp_path: Path) -> None:
    teacher_dir = _prepare_interval_teacher_artifact(tmp_path, "interval_teacher_features_contract")
    interval_dataset = load_fmpc_interval_dataset(teacher_dir, expected_dataset_name="digits")
    teacher_model, teacher_split = load_prepared_teacher_runtime(teacher_dir)
    context = prepare_interval_teacher_feature_context(
        interval_dataset.train,
        teacher_split,
        teacher_export_batch_size=int(interval_dataset.metadata["teacher_export_batch_size"]),
    )

    features = compute_interval_teacher_state_features(
        teacher_model,
        context.x,
        context.y,
        interval_dataset.train.z0,
        teacher_steps=interval_dataset.teacher_steps,
        batch_size=context.teacher_export_batch_size,
    )

    assert features.y_hat_s.shape == context.y.shape
    assert features.e_out_s.shape == context.y.shape
    assert features.g_s.shape == interval_dataset.train.z0.shape
    assert features.F_s.shape == (interval_dataset.train.num_samples, 1)
    assert features.y_hat_s.dtype == np.float64
    assert features.e_out_s.dtype == np.float64
    assert features.g_s.dtype == np.float64
    assert features.F_s.dtype == np.float64
    np.testing.assert_allclose(features.e_out_s, context.y - features.y_hat_s, atol=1e-12, rtol=1e-12)


def test_precomputed_teacher_field_matches_saved_one_step_trajectory_delta(tmp_path: Path) -> None:
    teacher_dir = _prepare_interval_teacher_artifact(tmp_path, "interval_teacher_features_trajectory")
    interval_dataset = load_fmpc_interval_dataset(teacher_dir, expected_dataset_name="digits")
    teacher_model, teacher_split = load_prepared_teacher_runtime(teacher_dir)
    context = prepare_interval_teacher_feature_context(
        interval_dataset.train,
        teacher_split,
        teacher_export_batch_size=int(interval_dataset.metadata["teacher_export_batch_size"]),
    )
    trajectory_features = precompute_interval_teacher_trajectory_features(
        teacher_model,
        interval_dataset.train,
        context,
    )

    step0_features = trajectory_features.step_features(0, selected_feature_names=("g_s",))
    expected = interval_dataset.teacher_steps * (
        interval_dataset.train.z_trajectory[:, 1, :] - interval_dataset.train.z_trajectory[:, 0, :]
    )
    np.testing.assert_allclose(step0_features.g_s, expected, atol=1e-12, rtol=1e-12)
    assert np.all(step0_features.F_s >= 0.0)


def test_precomputed_teacher_features_can_be_gathered_for_mixed_source_steps(tmp_path: Path) -> None:
    teacher_dir = _prepare_interval_teacher_artifact(tmp_path, "interval_teacher_features_gather")
    interval_dataset = load_fmpc_interval_dataset(teacher_dir, expected_dataset_name="digits")
    teacher_model, teacher_split = load_prepared_teacher_runtime(teacher_dir)
    context = prepare_interval_teacher_feature_context(
        interval_dataset.train,
        teacher_split,
        teacher_export_batch_size=int(interval_dataset.metadata["teacher_export_batch_size"]),
    )
    trajectory_features = precompute_interval_teacher_trajectory_features(
        teacher_model,
        interval_dataset.train,
        context,
    )

    sample_rows = np.asarray([0, 3, 5], dtype=np.int64)
    source_steps = np.asarray([0, 1, 2], dtype=np.int64)
    gathered = trajectory_features.gather_batch_features(sample_rows, source_steps)

    assert gathered.g_s.shape == (3, interval_dataset.z_dim)
    assert gathered.e_out_s.shape == (3, interval_dataset.target_dim)
    assert gathered.F_s.shape == (3, 1)
    np.testing.assert_allclose(
        gathered.g_s[0],
        trajectory_features.g_trajectory[sample_rows[0], source_steps[0], :],
        atol=1e-12,
        rtol=1e-12,
    )
