from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pc.fmpc_protocol import (
    FMPCPreparationConfig,
    FMPC_PREPARATION_SCHEMA_VERSION,
    load_prepared_teacher_runtime,
    run_fmpc_v0_preparation,
)
from pc.minibatch import iter_minibatches
from pc.layers import init_mlp_layers
from pc.models import PCNetwork
from pc.real_pc import RealPCConfig, RealPCRunResult
from pc.toy_data import SupervisedDataSplit


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_fmpc_teacher_only_smoke_writes_expected_artifacts(tmp_path: Path) -> None:
    config = FMPCPreparationConfig(
        dataset_name="digits",
        output_root=tmp_path,
        run_id="teacher_only_smoke",
        teacher_pc_config=RealPCConfig(
            dataset_name="digits",
            layer_dims=(64, 32, 10),
            epochs=1,
            batch_size=256,
            train_steps=2,
            eval_steps=2,
        ),
        teacher_export_backend="pc_euler",
        teacher_export_steps=2,
        teacher_export_batch_size=256,
        export_trajectory=False,
    )

    result = run_fmpc_v0_preparation(config)

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "teacher_model" / "config.json").exists()
    assert (run_dir / "teacher_model" / "epoch_metrics.csv").exists()
    assert (run_dir / "teacher_model" / "summary.json").exists()
    assert (run_dir / "teacher_model" / "checkpoint.npz").exists()
    assert (run_dir / "teacher_targets" / "manifest.json").exists()
    assert (run_dir / "teacher_targets" / "train.npz").exists()
    assert (run_dir / "teacher_targets" / "val.npz").exists()
    assert (run_dir / "teacher_targets" / "test.npz").exists()

    summary = _read_json(run_dir / "summary.json")
    manifest = _read_json(run_dir / "teacher_targets" / "manifest.json")

    assert summary["phase"] == "Phase 4"
    assert summary["protocol_name"] == "fmpc_v0_preparation"
    assert summary["mode"] == "teacher_only_preparation"
    assert summary["dataset_name"] == "digits"
    assert summary["student_placeholder"]["implemented"] is False
    assert summary["student_placeholder"]["backend"] == "fmpc"
    assert summary["teacher"]["export_backend"] == "pc_euler"
    assert summary["teacher"]["export_steps"] == 2

    assert manifest["schema_version"] == FMPC_PREPARATION_SCHEMA_VERSION
    assert manifest["teacher_mode"] == "train"
    assert manifest["teacher_target_semantics"] == "target-clamped supervised teacher"
    assert manifest["teacher_backend"] == "pc_euler"
    assert manifest["teacher_steps"] == 2
    assert manifest["export_trajectory"] is False
    assert manifest["teacher_checkpoint"]["relative_path"] == "../teacher_model/checkpoint.npz"
    assert not Path(manifest["teacher_checkpoint"]["relative_path"]).is_absolute()
    assert manifest["splits"]["train"]["relative_path"] == "train.npz"
    assert not Path(manifest["splits"]["train"]["relative_path"]).is_absolute()
    assert manifest["splits"]["train"]["delta_z_rms"] > 0.0

    train_npz = np.load(run_dir / "teacher_targets" / "train.npz")
    assert train_npz["sample_indices"].ndim == 1
    assert train_npz["targets"].ndim == 2
    assert train_npz["z0"].ndim == 2
    assert train_npz["z_star"].ndim == 2
    assert train_npz["targets"].shape[1] == 10
    assert train_npz["z0"].shape == train_npz["z_star"].shape
    assert train_npz["z0"].dtype == np.float64


def test_fmpc_preparation_checkpoint_roundtrip_reproduces_saved_teacher_targets(tmp_path: Path) -> None:
    result = run_fmpc_v0_preparation(
        FMPCPreparationConfig(
            dataset_name="digits",
            output_root=tmp_path,
            run_id="teacher_checkpoint_roundtrip",
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

    loaded_model, loaded_split = load_prepared_teacher_runtime(result.run_dir)
    with np.load(result.run_dir / "teacher_targets" / "train.npz") as saved:
        saved_z0 = np.asarray(saved["z0"], dtype=np.float64)
        saved_z_star = np.asarray(saved["z_star"], dtype=np.float64)

    z0_batches: list[np.ndarray] = []
    z_star_batches: list[np.ndarray] = []
    for x_batch, y_batch in iter_minibatches(
        loaded_split.x_train,
        loaded_split.y_train,
        256,
        shuffle=False,
    ):
        teacher_export = loaded_model.export_teacher_targets(
            np.asarray(x_batch, dtype=np.float64),
            np.asarray(y_batch, dtype=np.float64),
            record_trace=True,
            record_trajectory=False,
        )
        z0_batches.append(teacher_export.z0)
        z_star_batches.append(teacher_export.z_star)

    replay_z0 = np.concatenate(z0_batches, axis=0)
    replay_z_star = np.concatenate(z_star_batches, axis=0)

    np.testing.assert_allclose(replay_z0, saved_z0, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(replay_z_star, saved_z_star, atol=1e-12, rtol=1e-12)


def test_fmpc_preparation_supports_fashion_mnist_dataset_name_without_network(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured_dataset_names: list[str] = []

    x_train = np.zeros((4, 784), dtype=np.float64)
    y_train = np.eye(10, dtype=np.float64)[[0, 1, 2, 3]]
    split = SupervisedDataSplit(
        x_train=x_train,
        y_train=y_train,
        x_val=x_train.copy(),
        y_val=y_train.copy(),
        x_test=x_train.copy(),
        y_test=y_train.copy(),
        metadata={
            "dataset_name": "fashion_mnist",
            "train_indices": [0, 1, 2, 3],
            "val_indices": [4, 5, 6, 7],
            "test_indices": [8, 9, 10, 11],
        },
    )

    def fake_run_real_pc_experiment(config, *, return_runtime_objects=False):
        captured_dataset_names.append(config.dataset_name)
        run_dir = tmp_path / "fashion_teacher_model"
        run_dir.mkdir(parents=True, exist_ok=True)
        return RealPCRunResult(
            run_dir=run_dir,
            config={},
            epoch_metrics=[],
            summary={
                "inference_backend": "pc_euler",
                "eval_steps": 1,
            },
            model=PCNetwork(
                layers=init_mlp_layers([784, 8, 10], seed=0, dtype=np.float64),
                eta_x=0.1,
                eta_w=0.01,
                train_steps=1,
                eval_steps=1,
                inference_backend="pc_euler",
                state_init="forward",
            ),
            split=split,
        )

    monkeypatch.setattr("pc.fmpc_protocol.run_real_pc_experiment", fake_run_real_pc_experiment)

    result = run_fmpc_v0_preparation(
        FMPCPreparationConfig(
            dataset_name="fashion_mnist",
            output_root=tmp_path,
            run_id="fashion_stub",
            teacher_pc_config=RealPCConfig(dataset_name="fashion_mnist", layer_dims=(784, 8, 10)),
            teacher_export_steps=1,
            teacher_export_batch_size=8,
        )
    )

    assert captured_dataset_names == ["fashion_mnist"]
    assert result.summary["dataset_name"] == "fashion_mnist"
    assert (result.run_dir / "teacher_targets" / "train.npz").exists()


def test_fmpc_preparation_defaults_teacher_export_steps_to_train_steps(
    monkeypatch,
    tmp_path: Path,
) -> None:
    x_train = np.zeros((4, 64), dtype=np.float64)
    y_train = np.eye(10, dtype=np.float64)[[0, 1, 2, 3]]
    split = SupervisedDataSplit(
        x_train=x_train,
        y_train=y_train,
        x_val=x_train.copy(),
        y_val=y_train.copy(),
        x_test=x_train.copy(),
        y_test=y_train.copy(),
        metadata={
            "dataset_name": "digits",
            "train_indices": [0, 1, 2, 3],
            "val_indices": [4, 5, 6, 7],
            "test_indices": [8, 9, 10, 11],
        },
    )

    def fake_run_real_pc_experiment(config, *, return_runtime_objects=False):
        run_dir = tmp_path / "teacher_model_train_steps_default"
        run_dir.mkdir(parents=True, exist_ok=True)
        return RealPCRunResult(
            run_dir=run_dir,
            config={},
            epoch_metrics=[],
            summary={
                "inference_backend": "pc_euler",
                "eval_steps": 7,
            },
            model=PCNetwork(
                layers=init_mlp_layers([64, 8, 10], seed=0, dtype=np.float64),
                eta_x=0.1,
                eta_w=0.01,
                train_steps=3,
                eval_steps=7,
                inference_backend="pc_euler",
                state_init="forward",
            ),
            split=split,
        )

    monkeypatch.setattr("pc.fmpc_protocol.run_real_pc_experiment", fake_run_real_pc_experiment)

    result = run_fmpc_v0_preparation(
        FMPCPreparationConfig(
            dataset_name="digits",
            output_root=tmp_path,
            run_id="train_steps_default",
            teacher_pc_config=RealPCConfig(dataset_name="digits", layer_dims=(64, 8, 10)),
            teacher_export_steps=None,
            teacher_export_batch_size=8,
        )
    )

    manifest = _read_json(result.run_dir / "teacher_targets" / "manifest.json")
    summary = _read_json(result.run_dir / "summary.json")

    assert manifest["teacher_steps"] == 3
    assert manifest["splits"]["train"]["teacher_steps"] == 3
    assert summary["teacher"]["export_steps"] == 3


def test_fmpc_preparation_trajectory_artifact_roundtrip_is_exact(tmp_path: Path) -> None:
    result = run_fmpc_v0_preparation(
        FMPCPreparationConfig(
            dataset_name="digits",
            output_root=tmp_path,
            run_id="teacher_trajectory_roundtrip",
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

    manifest = _read_json(result.run_dir / "teacher_targets" / "manifest.json")
    assert manifest["export_trajectory"] is True
    assert manifest["trajectory_includes_endpoints"] is True
    assert manifest["trajectory_axis_semantics"] == "(batch, step, z_dim)"
    assert manifest["tau_definition"] == "tau_k = k / teacher_steps"
    assert manifest["splits"]["train"]["z_trajectory_shape"][1] == 4

    loaded_model, loaded_split = load_prepared_teacher_runtime(result.run_dir)
    with np.load(result.run_dir / "teacher_targets" / "train.npz") as saved:
        saved_z0 = np.asarray(saved["z0"], dtype=np.float64)
        saved_z_star = np.asarray(saved["z_star"], dtype=np.float64)
        saved_trajectory = np.asarray(saved["z_trajectory"], dtype=np.float64)

    z0_batches: list[np.ndarray] = []
    z_star_batches: list[np.ndarray] = []
    trajectory_batches: list[np.ndarray] = []
    for x_batch, y_batch in iter_minibatches(
        loaded_split.x_train,
        loaded_split.y_train,
        256,
        shuffle=False,
    ):
        teacher_export = loaded_model.export_teacher_targets(
            np.asarray(x_batch, dtype=np.float64),
            np.asarray(y_batch, dtype=np.float64),
            record_trace=True,
            record_trajectory=True,
        )
        z0_batches.append(teacher_export.z0)
        z_star_batches.append(teacher_export.z_star)
        trajectory_batches.append(
            np.stack(teacher_export.z_trajectory, axis=0).transpose(1, 0, 2)
        )

    replay_z0 = np.concatenate(z0_batches, axis=0)
    replay_z_star = np.concatenate(z_star_batches, axis=0)
    replay_trajectory = np.concatenate(trajectory_batches, axis=0)

    np.testing.assert_allclose(replay_z0, saved_z0, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(replay_z_star, saved_z_star, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(replay_trajectory, saved_trajectory, atol=1e-12, rtol=1e-12)
