from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path

from pc.fmpc_interval_student import FMPCIntervalSuiteConfig, run_fmpc_interval_suite
from pc.fmpc_protocol import FMPCPreparationConfig, run_fmpc_v0_preparation
from pc.real_pc import RealPCConfig


ROOT = Path(__file__).resolve().parents[1]


def load_run():
    module = runpy.run_path(str(ROOT / "experiments" / "fmpc_interval_suite.py"))
    return module["run"]


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


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_fmpc_interval_suite_smoke_run_writes_expected_artifacts(tmp_path: Path) -> None:
    teacher_dir = _prepare_interval_teacher_artifact(tmp_path, "teacher_for_interval_suite")

    result = run_fmpc_interval_suite(
        FMPCIntervalSuiteConfig(
            teacher_preparation_path=teacher_dir,
            output_root=tmp_path,
            run_id="interval_suite_smoke",
            interval_ridge_alphas=(1e-4, 1.0),
            mlp_hidden_dims_candidates=((16,),),
            mlp_epochs_candidates=(1,),
            mlp_eta_w_candidates=(0.01,),
            mlp_batch_size=256,
        )
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "candidates.csv").exists()
    assert (run_dir / "summary.json").exists()

    summary = _read_json(run_dir / "summary.json")
    assert summary["phase"] == "Phase 5"
    assert summary["stage"] == "phase5b_interval_conditioned_transporter"
    assert Path(str(summary["teacher_artifact_dir"])).is_absolute() is False
    assert Path(str(summary["teacher_manifest_path"])).is_absolute() is False
    assert Path(str(summary["teacher_checkpoint_path"])).is_absolute() is False
    assert "identity_baseline" in summary
    assert "phase5a_endpoint_ridge_baseline" in summary
    assert "interval_ridge" in summary
    assert "interval_ridge_aug" in summary
    assert "interval_ridge_residual" in summary
    assert "interval_mlp_standardized" in summary
    assert "rollout_aware_rescue" in summary
    assert "gradient_augmented_rescue" in summary
    assert summary["rollout_aware_rescue"]["active"] is True
    assert summary["gradient_augmented_rescue"]["active"] is True
    assert summary["gradient_augmented_rescue"]["teacher_feature_names"] == ["g_s", "e_out_s", "F_s"]
    assert "winner" in summary
    assert summary["teacher_steps"] == 3
    assert set(summary["schedule_knots"].keys()) == {"1-step", "2-step", "3-step"}
    assert set(summary["rollout_aware_auxiliary_schedules"].keys()) == {"2-step", "3-step"}

    candidates = _read_csv(run_dir / "candidates.csv")
    families = {row["family"] for row in candidates}
    assert families == {
        "identity",
        "phase5a_endpoint_ridge",
        "interval_ridge",
        "interval_ridge_aug",
        "interval_ridge_residual",
        "interval_mlp_standardized",
    }
    winner_rows = [row for row in candidates if row["is_overall_winner"] == "True"]
    assert len(winner_rows) == 1
    assert winner_rows[0]["family"] in {
        "interval_ridge",
        "interval_ridge_aug",
        "interval_ridge_residual",
        "interval_mlp_standardized",
    }
    assert winner_rows[0]["evaluated_on_test"] == "True"
    mlp_rows = [row for row in candidates if row["family"] == "interval_mlp_standardized"]
    assert mlp_rows
    assert all("rollout_aux_weight" in row for row in mlp_rows)
    assert all("train_primary_interval_loss" in row for row in mlp_rows)
    assert all("train_rollout_aux_velocity_mse" in row for row in mlp_rows)
    aug_rows = [row for row in candidates if row["family"] in {"interval_ridge_aug", "interval_ridge_residual"}]
    assert aug_rows
    assert all(row["feature_contract"] == "g_s,e_out_s,F_s" for row in aug_rows)
    assert all(row["knot_focused_schedule_names"] == "2-step,3-step" for row in aug_rows)


def test_fmpc_interval_suite_run_function_is_available(tmp_path: Path) -> None:
    teacher_dir = _prepare_interval_teacher_artifact(tmp_path, "teacher_for_interval_suite_runpy")
    run = load_run()

    result = run(
        teacher_preparation_path=teacher_dir,
        output_root=tmp_path,
        run_id="interval_suite_runpy",
        interval_ridge_alphas=(1e-4,),
        mlp_hidden_dims_candidates=((16,),),
        mlp_epochs_candidates=(1,),
        mlp_eta_w_candidates=(0.01,),
        mlp_batch_size=256,
    )

    assert (result.run_dir / "summary.json").exists()
