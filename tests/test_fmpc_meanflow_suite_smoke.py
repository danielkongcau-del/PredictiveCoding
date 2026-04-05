from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path

from pc.fmpc_meanflow_student import FMPCMeanFlowSuiteConfig, run_fmpc_meanflow_suite
from pc.fmpc_protocol import FMPCPreparationConfig, run_fmpc_v0_preparation
from pc.real_pc import RealPCConfig


ROOT = Path(__file__).resolve().parents[1]


def load_run():
    module = runpy.run_path(str(ROOT / "experiments" / "fmpc_meanflow_suite.py"))
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


def test_fmpc_meanflow_suite_smoke_run_writes_expected_artifacts(tmp_path: Path) -> None:
    teacher_dir = _prepare_interval_teacher_artifact(tmp_path, "teacher_for_meanflow_suite")

    result = run_fmpc_meanflow_suite(
        FMPCMeanFlowSuiteConfig(
            teacher_preparation_path=teacher_dir,
            output_root=tmp_path,
            run_id="meanflow_suite_smoke",
            hidden_dims_candidates=((16,),),
            epochs_candidates=(1,),
            eta_w_candidates=(0.01,),
            identity_loss_weight_candidates=(0.1,),
            rollout_aux_weight_candidates=(0.0,),
            knot_focus_probability_candidates=(0.0,),
        )
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "candidates.csv").exists()
    assert (run_dir / "summary.json").exists()

    config = _read_json(run_dir / "config.json")
    summary = _read_json(run_dir / "summary.json")
    assert summary["phase"] == "Phase 6"
    assert summary["stage"] == "phase6a3_warmstarted_two_branch_meanflow"
    assert Path(str(summary["teacher_artifact_dir"])).is_absolute() is False
    assert Path(str(summary["teacher_manifest_path"])).is_absolute() is False
    assert Path(str(summary["teacher_checkpoint_path"])).is_absolute() is False
    assert "identity_baseline" in summary
    assert "phase5a_endpoint_ridge_baseline" in summary
    assert "phase5b2_interval_ridge_residual_baseline" in summary
    assert "phase6a1_linear_residual_baseline" in summary
    assert "phase6a2_twobranch_residual_baseline" in summary
    assert "teacher_only_mlp_aug" in summary
    assert "meanflow_mlp_aug" in summary
    assert "meanflow_mlp_residual" in summary
    assert "meanflow_linear_residual" in summary
    assert "meanflow_twobranch_residual" in summary
    assert "meanflow_twobranch_residual_warmstart" in summary
    assert "winner" in summary
    assert summary["meanflow_contract"]["manual_numpy_jvp"] is True
    assert summary["meanflow_contract"]["identity_target_uses_stop_gradient"] is True
    assert summary["meanflow_contract"]["feature_aware_teacher_tangents"] is True
    assert summary["meanflow_contract"]["residual_identity_includes_dg_s"] is True
    assert summary["meanflow_contract"]["two_branch_identity_applies_to_full_u_hat"] is True
    assert "identity_scope_modes" in config["search_space"]

    candidates = _read_csv(run_dir / "candidates.csv")
    families = {row["family"] for row in candidates}
    assert families == {
        "identity",
        "phase5a_endpoint_ridge",
        "phase5b2_interval_ridge_residual",
        "phase6a1_linear_residual_baseline",
        "phase6a2_twobranch_residual_baseline",
        "teacher_only_mlp_aug",
        "meanflow_mlp_aug",
        "meanflow_mlp_residual",
        "meanflow_linear_residual",
        "meanflow_twobranch_residual",
        "meanflow_twobranch_residual_warmstart",
    }
    assert any(row["meanflow_identity_enabled"] == "True" for row in candidates if row["family"].startswith("meanflow_"))
    assert any(row["identity_scope_mode"] == "acceptance_schedule_segments_only" for row in candidates if row["family"].startswith("meanflow_"))
    assert any(row["feature_aware_teacher_tangents"] == "True" for row in candidates if row["family"].startswith("meanflow_"))
    assert any(row["residual_identity_includes_dg_s"] == "True" for row in candidates if row["family"] in {"meanflow_mlp_residual", "meanflow_linear_residual"})
    assert any(row["is_two_branch"] == "True" for row in candidates if row["family"] == "meanflow_twobranch_residual")
    assert any(row["is_two_branch"] == "True" for row in candidates if row["family"] == "meanflow_twobranch_residual_warmstart")
    assert any(row["local_branch_input_set"] == "g_s,e_out_s,F_s" for row in candidates if row["family"] == "meanflow_twobranch_residual")
    assert any(row["local_branch_warm_started"] == "True" for row in candidates if row["family"] == "meanflow_twobranch_residual_warmstart")
    assert any(row["warm_start_source"] == "phase6a1_linear_residual_baseline" for row in candidates if row["family"] == "meanflow_twobranch_residual_warmstart")
    assert any(row["correction_only_warmup_epochs"] == "10" for row in candidates if row["family"] == "meanflow_twobranch_residual_warmstart")
    assert any(row["correction_branch_family"] == "mlp" for row in candidates if row["family"] == "meanflow_twobranch_residual")
    winner_rows = [row for row in candidates if row["is_overall_winner"] == "True"]
    assert len(winner_rows) == 1
    assert winner_rows[0]["family"] in {
        "teacher_only_mlp_aug",
        "meanflow_mlp_aug",
        "meanflow_mlp_residual",
        "meanflow_linear_residual",
        "meanflow_twobranch_residual",
        "meanflow_twobranch_residual_warmstart",
    }


def test_fmpc_meanflow_suite_run_function_is_available(tmp_path: Path) -> None:
    teacher_dir = _prepare_interval_teacher_artifact(tmp_path, "teacher_for_meanflow_suite_runpy")
    run = load_run()

    result = run(
        teacher_preparation_path=teacher_dir,
        output_root=tmp_path,
        run_id="meanflow_suite_runpy",
        hidden_dims_candidates=((16,),),
        epochs_candidates=(1,),
        eta_w_candidates=(0.01,),
        identity_loss_weight_candidates=(0.1,),
        rollout_aux_weight_candidates=(0.0,),
        knot_focus_probability_candidates=(0.0,),
    )

    assert (result.run_dir / "summary.json").exists()
