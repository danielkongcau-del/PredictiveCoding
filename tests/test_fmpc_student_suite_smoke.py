from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path

from pc.fmpc_protocol import FMPCPreparationConfig, run_fmpc_v0_preparation
from pc.fmpc_student_suite import FMPCStudentSuiteConfig, run_fmpc_student_suite
from pc.real_pc import RealPCConfig


ROOT = Path(__file__).resolve().parents[1]


def load_run():
    module = runpy.run_path(str(ROOT / "experiments" / "fmpc_v0_student_suite.py"))
    return module["run"]


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


def _read_summary(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_candidates(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_fmpc_student_suite_smoke_run_writes_expected_artifacts(tmp_path: Path) -> None:
    teacher_dir = _prepare_teacher_artifact(tmp_path, "teacher_for_student_suite")

    result = run_fmpc_student_suite(
        FMPCStudentSuiteConfig(
            teacher_preparation_path=teacher_dir,
            output_root=tmp_path,
            run_id="student_suite_smoke",
            ridge_alphas=(1e-4, 1.0),
            mlp_hidden_dims_candidates=((16,),),
            mlp_epochs_candidates=(2,),
            mlp_eta_w_candidates=(0.01,),
            mlp_batch_size=256,
        )
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "candidates.csv").exists()
    assert (run_dir / "summary.json").exists()

    summary = _read_summary(run_dir / "summary.json")
    assert summary["phase"] == "Phase 5"
    assert summary["stage"] == "phase5a_student_signal_rescue"
    assert Path(str(summary["teacher_artifact_dir"])).is_absolute() is False
    assert Path(str(summary["teacher_manifest_path"])).is_absolute() is False
    assert Path(str(summary["teacher_checkpoint_path"])).is_absolute() is False
    assert "\\" not in str(summary["teacher_artifact_dir"])
    assert "\\" not in str(summary["teacher_manifest_path"])
    assert "\\" not in str(summary["teacher_checkpoint_path"])
    assert "identity_baseline" in summary
    assert "class_mean_delta" in summary
    assert "ridge" in summary
    assert "mlp_standardized" in summary
    assert "winner" in summary

    candidates = _read_candidates(run_dir / "candidates.csv")
    families = {row["family"] for row in candidates}
    assert families == {"identity", "class_mean_delta", "ridge", "mlp_standardized"}

    learned_winner_config = str(summary["winner"]["learned_winner_config_id_by_val"])
    learned_winner_rows = [row for row in candidates if row["config_id"] == learned_winner_config]
    assert len(learned_winner_rows) == 1
    assert learned_winner_rows[0]["is_learned_winner"] == "True"
    assert learned_winner_rows[0]["is_family_best"] == "True"


def test_fmpc_student_suite_run_function_is_available(tmp_path: Path) -> None:
    teacher_dir = _prepare_teacher_artifact(tmp_path, "teacher_for_student_suite_runpy")
    run = load_run()

    result = run(
        teacher_preparation_path=teacher_dir,
        output_root=tmp_path,
        run_id="student_suite_runpy",
        ridge_alphas=(1e-4,),
        mlp_hidden_dims_candidates=((16,),),
        mlp_epochs_candidates=(1,),
        mlp_eta_w_candidates=(0.01,),
        mlp_batch_size=256,
    )

    assert (result.run_dir / "summary.json").exists()
