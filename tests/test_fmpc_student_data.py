from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pc.fmpc_protocol import FMPCPreparationConfig, run_fmpc_v0_preparation
from pc.fmpc_student_data import build_fmpc_student_inputs, load_fmpc_student_dataset
from pc.real_pc import RealPCConfig


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_synthetic_teacher_targets(run_dir: Path) -> Path:
    teacher_targets_dir = run_dir / "teacher_targets"
    teacher_targets_dir.mkdir(parents=True, exist_ok=True)
    teacher_model_dir = run_dir / "teacher_model"
    teacher_model_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(teacher_model_dir / "checkpoint.npz", stub=np.array([1.0], dtype=np.float64))

    def make_split(split_name: str, offset: float) -> dict[str, object]:
        sample_indices = np.arange(3, dtype=np.int64) + int(offset * 10)
        targets = np.eye(10, dtype=np.float64)[[0, 1, 2]]
        z0 = np.array(
            [
                [0.0 + offset, 1.0 + offset, 2.0 + offset, 3.0 + offset],
                [4.0 + offset, 5.0 + offset, 6.0 + offset, 7.0 + offset],
                [8.0 + offset, 9.0 + offset, 10.0 + offset, 11.0 + offset],
            ],
            dtype=np.float64,
        )
        z_star = z0 + 0.5
        split_path = teacher_targets_dir / f"{split_name}.npz"
        np.savez_compressed(
            split_path,
            sample_indices=sample_indices,
            targets=targets,
            z0=z0,
            z_star=z_star,
        )
        return {
            "split_name": split_name,
            "relative_path": split_path.name,
            "path": split_path.name,
            "num_samples": int(sample_indices.shape[0]),
            "sample_indices_shape": [3],
            "targets_shape": [3, 10],
            "z0_shape": [3, 4],
            "z_star_shape": [3, 4],
            "z_trajectory_shape": None,
            "dtype": "float64",
            "teacher_mode": "train",
            "teacher_target_semantics": "target-clamped supervised teacher",
            "teacher_backend": "pc_euler",
            "teacher_steps": 5,
        }

    manifest = {
        "schema_version": "phase5_portable_v1",
        "phase": "Phase 4",
        "protocol_name": "fmpc_v0_preparation",
        "dataset_name": "digits",
        "teacher_backend": "pc_euler",
        "teacher_steps": 5,
        "teacher_export_batch_size": 16,
        "teacher_mode": "train",
        "teacher_target_semantics": "target-clamped supervised teacher",
        "export_trajectory": False,
        "teacher_checkpoint": {
            "format": "pc_teacher_checkpoint_npz",
            "dtype": "float64",
            "relative_path": "../teacher_model/checkpoint.npz",
            "layer_dims": [64, 4, 10],
            "train_steps": 5,
            "eval_steps": 5,
            "inference_backend": "pc_euler",
            "inference_method": "euler",
            "state_init": "forward",
        },
        "splits": {
            "train": make_split("train", 0.0),
            "val": make_split("val", 1.0),
            "test": make_split("test", 2.0),
        },
    }
    _write_json(teacher_targets_dir / "manifest.json", manifest)
    return teacher_targets_dir


def test_build_fmpc_student_inputs_concatenates_batch_first_float64() -> None:
    z0 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    targets = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    student_inputs = build_fmpc_student_inputs(z0, targets)

    assert student_inputs.shape == (2, 4)
    assert student_inputs.dtype == np.float64
    np.testing.assert_allclose(student_inputs[:, :2], z0)
    np.testing.assert_allclose(student_inputs[:, 2:], targets)


def test_load_fmpc_student_dataset_builds_expected_contract(tmp_path: Path) -> None:
    teacher_targets_dir = _write_synthetic_teacher_targets(tmp_path / "synthetic_teacher")

    dataset = load_fmpc_student_dataset(teacher_targets_dir)

    assert dataset.dataset_name == "digits"
    assert dataset.schema_version == "phase5_portable_v1"
    assert dataset.teacher_mode == "train"
    assert dataset.teacher_target_semantics == "target-clamped supervised teacher"
    assert dataset.teacher_checkpoint_path is not None
    assert dataset.teacher_checkpoint_path.name == "checkpoint.npz"
    assert dataset.student_input_definition == "concat([z0, target_onehot])"
    assert dataset.student_target_definition == "delta_z = z_star - z0"
    assert dataset.z_dim == 4
    assert dataset.target_dim == 10

    for split in (dataset.train, dataset.val, dataset.test):
        assert split.sample_indices.shape == (3,)
        assert split.sample_indices.dtype == np.int64
        assert split.target_onehot.shape == (3, 10)
        assert split.z0.shape == (3, 4)
        assert split.z_star.shape == (3, 4)
        assert split.delta_z.shape == (3, 4)
        assert split.student_inputs.shape == (3, 14)
        assert split.target_onehot.dtype == np.float64
        assert split.z0.dtype == np.float64
        assert split.z_star.dtype == np.float64
        assert split.delta_z.dtype == np.float64
        assert split.student_inputs.dtype == np.float64
        np.testing.assert_allclose(split.delta_z, split.z_star - split.z0)
        assert split.metadata["delta_z_rms"] > 0.0
        np.testing.assert_allclose(
            split.student_inputs,
            np.concatenate([split.z0, split.target_onehot], axis=1),
        )


def test_load_fmpc_student_dataset_rejects_dataset_name_mismatch(tmp_path: Path) -> None:
    teacher_targets_dir = _write_synthetic_teacher_targets(tmp_path / "synthetic_teacher")

    with pytest.raises(ValueError, match="Expected dataset_name"):
        load_fmpc_student_dataset(teacher_targets_dir, expected_dataset_name="fashion_mnist")


def test_load_fmpc_student_dataset_rejects_shape_drift(tmp_path: Path) -> None:
    teacher_targets_dir = _write_synthetic_teacher_targets(tmp_path / "synthetic_teacher")
    test_npz = teacher_targets_dir / "test.npz"
    with np.load(test_npz) as payload:
        np.savez_compressed(
            test_npz,
            sample_indices=payload["sample_indices"],
            targets=payload["targets"][:, :9],
            z0=payload["z0"],
            z_star=payload["z_star"],
        )

    with pytest.raises(ValueError, match="targets feature dimension"):
        load_fmpc_student_dataset(teacher_targets_dir)


def test_load_fmpc_student_dataset_supports_legacy_absolute_manifest_paths(tmp_path: Path) -> None:
    teacher_targets_dir = _write_synthetic_teacher_targets(tmp_path / "synthetic_teacher")
    manifest_path = teacher_targets_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    for split_name in ("train", "val", "test"):
        relative_name = manifest["splits"][split_name].pop("relative_path")
        manifest["splits"][split_name]["path"] = str((teacher_targets_dir / relative_name).resolve())

    manifest.pop("schema_version")
    manifest.pop("teacher_checkpoint")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    dataset = load_fmpc_student_dataset(teacher_targets_dir)

    assert dataset.schema_version == "phase4_legacy_absolute_v0"
    assert dataset.teacher_checkpoint_path is None
    assert dataset.train.student_inputs.shape[0] == 3


def test_load_fmpc_student_dataset_reads_phase4_preparation_artifact(tmp_path: Path) -> None:
    prep_result = run_fmpc_v0_preparation(
        FMPCPreparationConfig(
            dataset_name="digits",
            output_root=tmp_path,
            run_id="student_data_smoke",
            teacher_pc_config=RealPCConfig(
                dataset_name="digits",
                layer_dims=(64, 16, 10),
                epochs=1,
                batch_size=256,
                train_steps=2,
                eval_steps=2,
            ),
            teacher_export_steps=2,
            teacher_export_batch_size=256,
            export_trajectory=False,
        )
    )

    dataset = load_fmpc_student_dataset(prep_result.run_dir)

    assert dataset.dataset_name == "digits"
    assert dataset.metadata["teacher_backend"] == "pc_euler"
    assert dataset.metadata["teacher_steps"] == 2
    assert dataset.train.student_inputs.shape[1] == dataset.z_dim + dataset.target_dim
    assert dataset.train.metadata["teacher_steps"] == 2
