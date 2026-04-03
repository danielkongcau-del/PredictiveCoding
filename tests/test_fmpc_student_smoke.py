from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path

import pytest

from pc.fmpc_protocol import FMPCPreparationConfig, run_fmpc_v0_preparation
from pc.fmpc_student import FMPCStudentConfig, run_fmpc_student_experiment
from pc.real_pc import RealPCConfig


ROOT = Path(__file__).resolve().parents[1]


def load_run():
    module = runpy.run_path(str(ROOT / "experiments" / "fmpc_v0_student.py"))
    return module["run"]


def _read_epoch_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_summary(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _summary_without_timing(summary: dict[str, object]) -> dict[str, object]:
    sanitized = json.loads(json.dumps(summary))

    def _replace_timing(node):
        if isinstance(node, dict):
            for key, value in list(node.items()):
                if key.endswith("_wall_time_seconds") or key.endswith("_speedup_vs_teacher"):
                    node[key] = "__timing__"
                else:
                    _replace_timing(value)
        elif isinstance(node, list):
            for item in node:
                _replace_timing(item)

    _replace_timing(sanitized)
    return sanitized


def _epoch_rows_without_timing(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    sanitized = json.loads(json.dumps(rows))
    for row in sanitized:
        for key in list(row.keys()):
            if key.endswith("_wall_time_seconds") or key.endswith("_speedup_vs_teacher"):
                row[key] = "__timing__"
    return sanitized


def _prepare_teacher_artifact(tmp_path: Path, run_id: str) -> Path:
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
                train_steps=2,
                eval_steps=2,
            ),
            teacher_export_steps=2,
            teacher_export_batch_size=256,
            export_trajectory=False,
        )
    )
    return result.run_dir


def test_fmpc_student_smoke_run_writes_expected_artifacts(tmp_path: Path) -> None:
    teacher_dir = _prepare_teacher_artifact(tmp_path, "teacher_for_student_smoke")

    result = run_fmpc_student_experiment(
        FMPCStudentConfig(
            teacher_preparation_path=teacher_dir,
            output_root=tmp_path,
            run_id="student_smoke",
            hidden_dims=(16,),
            epochs=2,
            batch_size=256,
        )
    )

    run_dir = result.run_dir
    assert run_dir == tmp_path / "fmpc_v0_student"
    assert (run_dir / "config.json").exists()
    assert (run_dir / "epoch_metrics.csv").exists()
    assert (run_dir / "summary.json").exists()

    summary = _read_summary(run_dir / "summary.json")
    assert summary["phase"] == "Phase 5"
    assert summary["stage"] == "offline_fmpc_v0_student"
    assert summary["dataset_name"] == "digits"
    assert summary["selection_metric_source"] == "val_metric"
    assert summary["report_metric_source"] == "test_metric"
    assert summary["teacher_checkpoint_loaded"] is True
    assert Path(str(summary["teacher_artifact_dir"])).is_absolute() is False
    assert Path(str(summary["teacher_manifest_path"])).is_absolute() is False
    assert Path(str(summary["teacher_checkpoint_path"])).is_absolute() is False
    assert "\\" not in str(summary["teacher_artifact_dir"])
    assert "\\" not in str(summary["teacher_manifest_path"])
    assert "\\" not in str(summary["teacher_checkpoint_path"])
    assert summary["teacher_checkpoint_path"].endswith("checkpoint.npz")
    assert summary["student_input_definition"] == "concat([z0, target_onehot])"
    assert summary["student_target_definition"] == "delta_z = z_star - z0"
    assert summary["student_output_definition"] == "delta_z_hat"
    assert summary["transporter_output_definition"] == "z_hat = z0 + delta_z_hat"
    assert summary["best_epoch"] >= 1
    assert summary["best_epoch"] <= summary["epochs"]
    assert summary["val_metric"] == summary["best_val_metric"]
    assert summary["refinement_enabled"] is False
    assert "identity_baseline" in summary
    assert "comparison_to_identity" in summary
    assert "teacher_target_stats" in summary

    required_metric_keys = [
        "train_state_l2_gap",
        "val_state_l2_gap",
        "test_state_l2_gap",
        "train_state_rms_gap",
        "val_state_rms_gap",
        "test_state_rms_gap",
        "train_teacher_energy",
        "val_teacher_energy",
        "test_teacher_energy",
        "train_student_energy",
        "val_student_energy",
        "test_student_energy",
        "train_predicted_energy",
        "val_predicted_energy",
        "test_predicted_energy",
        "train_energy_gap_to_teacher",
        "val_energy_gap_to_teacher",
        "test_energy_gap_to_teacher",
        "train_update_direction_cosine",
        "val_update_direction_cosine",
        "test_update_direction_cosine",
        "train_transport_wall_time_seconds",
        "val_transport_wall_time_seconds",
        "test_transport_wall_time_seconds",
        "train_teacher_wall_time_seconds",
        "val_teacher_wall_time_seconds",
        "test_teacher_wall_time_seconds",
        "train_teacher_inference_wall_time_seconds",
        "val_teacher_inference_wall_time_seconds",
        "test_teacher_inference_wall_time_seconds",
        "train_speedup_vs_teacher",
        "val_speedup_vs_teacher",
        "test_speedup_vs_teacher",
    ]
    for key in required_metric_keys:
        assert key in summary
    assert "test_state_rms_gap" in summary["identity_baseline"]
    assert "test_energy_gap_to_teacher" in summary["identity_baseline"]
    assert "student_beats_identity_on_test_metric" in summary["comparison_to_identity"]

    epoch_rows = _read_epoch_rows(run_dir / "epoch_metrics.csv")
    assert len(epoch_rows) == 2
    assert "train_loss" in epoch_rows[0]
    assert "val_loss" in epoch_rows[0]
    assert "train_state_rms_gap" in epoch_rows[0]
    assert "val_state_rms_gap" in epoch_rows[0]
    assert "weight_norm_l1" in epoch_rows[0]
    assert "bias_norm_l1" in epoch_rows[0]


def test_fmpc_student_smoke_run_loads_teacher_checkpoint_by_default(tmp_path: Path, monkeypatch) -> None:
    teacher_dir = _prepare_teacher_artifact(tmp_path, "teacher_for_student_checkpoint")
    monkeypatch.setattr(
        "pc.fmpc_student._reconstruct_teacher_runtime",
        lambda dataset: (_ for _ in ()).throw(RuntimeError("legacy retrain fallback should not be used")),
    )

    result = run_fmpc_student_experiment(
        FMPCStudentConfig(
            teacher_preparation_path=teacher_dir,
            output_root=tmp_path,
            run_id="student_checkpoint_only",
            hidden_dims=(16,),
            epochs=1,
            batch_size=256,
        )
    )

    summary = _read_summary(result.run_dir / "summary.json")
    assert summary["allow_teacher_retrain"] is False
    assert summary["teacher_checkpoint_loaded"] is True
    assert Path(str(summary["teacher_checkpoint_path"])).is_absolute() is False
    assert "\\" not in str(summary["teacher_checkpoint_path"])
    assert summary["teacher_checkpoint_path"].endswith("checkpoint.npz")


def test_fmpc_student_requires_checkpoint_by_default(tmp_path: Path) -> None:
    teacher_dir = _prepare_teacher_artifact(tmp_path, "teacher_without_checkpoint")
    checkpoint_path = teacher_dir / "teacher_model" / "checkpoint.npz"
    checkpoint_path.unlink()

    with pytest.raises(FileNotFoundError, match="exact serialized teacher checkpoint"):
        run_fmpc_student_experiment(
            FMPCStudentConfig(
                teacher_preparation_path=teacher_dir,
                output_root=tmp_path,
                run_id="student_missing_checkpoint",
                hidden_dims=(16,),
                epochs=1,
                batch_size=256,
            )
        )


def test_fmpc_student_allows_explicit_teacher_retrain_fallback(tmp_path: Path) -> None:
    teacher_dir = _prepare_teacher_artifact(tmp_path, "teacher_with_opt_in_retrain")
    checkpoint_path = teacher_dir / "teacher_model" / "checkpoint.npz"
    checkpoint_path.unlink()

    result = run_fmpc_student_experiment(
        FMPCStudentConfig(
            teacher_preparation_path=teacher_dir,
            output_root=tmp_path,
            run_id="student_opt_in_retrain",
            hidden_dims=(16,),
            epochs=1,
            batch_size=256,
            allow_teacher_retrain=True,
        )
    )

    summary = _read_summary(result.run_dir / "summary.json")
    assert summary["allow_teacher_retrain"] is True
    assert summary["teacher_checkpoint_loaded"] is False


def test_fmpc_student_smoke_run_is_reproducible_except_timing(tmp_path: Path) -> None:
    teacher_dir = _prepare_teacher_artifact(tmp_path, "teacher_for_student_repro")
    run = load_run()

    first = run(
        teacher_preparation_path=teacher_dir,
        output_root=tmp_path / "first",
        run_id="student_repro",
        hidden_dims=(16,),
        epochs=2,
        batch_size=256,
    )
    second = run(
        teacher_preparation_path=teacher_dir,
        output_root=tmp_path / "second",
        run_id="student_repro",
        hidden_dims=(16,),
        epochs=2,
        batch_size=256,
    )

    first_summary = _read_summary(first.run_dir / "summary.json")
    second_summary = _read_summary(second.run_dir / "summary.json")
    assert _summary_without_timing(first_summary) == _summary_without_timing(second_summary)

    first_epoch_rows = _read_epoch_rows(first.run_dir / "epoch_metrics.csv")
    second_epoch_rows = _read_epoch_rows(second.run_dir / "epoch_metrics.csv")
    assert _epoch_rows_without_timing(first_epoch_rows) == _epoch_rows_without_timing(second_epoch_rows)
