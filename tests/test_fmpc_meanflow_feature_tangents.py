from __future__ import annotations

from pathlib import Path

import numpy as np

from pc.fmpc_interval_data import load_fmpc_interval_dataset
from pc.fmpc_interval_features import (
    compute_interval_teacher_state_feature_tangents,
    compute_interval_teacher_state_features,
    precompute_interval_teacher_trajectory_feature_tangents,
    prepare_interval_teacher_feature_context,
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


def test_teacher_feature_directional_derivatives_have_expected_contract(tmp_path: Path) -> None:
    teacher_dir = _prepare_interval_teacher_artifact(tmp_path, "meanflow_feature_tangent_contract")
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
    tangents = compute_interval_teacher_state_feature_tangents(
        teacher_model,
        context.x,
        context.y,
        interval_dataset.train.z0,
        teacher_steps=interval_dataset.teacher_steps,
        batch_size=context.teacher_export_batch_size,
        fd_epsilon=1e-3,
        state_features=features,
    )

    assert tangents.Dg_y_hat_s.shape == features.y_hat_s.shape
    assert tangents.Dg_e_out_s.shape == features.e_out_s.shape
    assert tangents.Dg_g_s.shape == features.g_s.shape
    assert tangents.Dg_F_s.shape == features.F_s.shape
    assert tangents.Dg_g_s.dtype == np.float64
    assert tangents.Dg_F_s.dtype == np.float64
    np.testing.assert_allclose(
        tangents.Dg_e_out_s,
        -tangents.Dg_y_hat_s,
        atol=1e-9,
        rtol=1e-7,
    )


def test_precomputed_feature_tangents_can_be_gathered_per_sample_and_source_step(tmp_path: Path) -> None:
    teacher_dir = _prepare_interval_teacher_artifact(tmp_path, "meanflow_feature_tangent_gather")
    interval_dataset = load_fmpc_interval_dataset(teacher_dir, expected_dataset_name="digits")
    teacher_model, teacher_split = load_prepared_teacher_runtime(teacher_dir)
    context = prepare_interval_teacher_feature_context(
        interval_dataset.train,
        teacher_split,
        teacher_export_batch_size=int(interval_dataset.metadata["teacher_export_batch_size"]),
    )
    tangent_trajectory = precompute_interval_teacher_trajectory_feature_tangents(
        teacher_model,
        interval_dataset.train,
        context,
        fd_epsilon=1e-3,
    )

    sample_rows = np.asarray([0, 4, 7], dtype=np.int64)
    source_steps = np.asarray([0, 1, 3], dtype=np.int64)
    gathered = tangent_trajectory.gather_batch_feature_tangents(sample_rows, source_steps)
    tangent_matrix = gathered.feature_tangent_matrix(("g_s", "e_out_s", "F_s"))

    assert gathered.Dg_g_s.shape == (3, interval_dataset.z_dim)
    assert gathered.Dg_e_out_s.shape == (3, interval_dataset.target_dim)
    assert gathered.Dg_F_s.shape == (3, 1)
    assert tangent_matrix.shape == (3, interval_dataset.z_dim + interval_dataset.target_dim + 1)
