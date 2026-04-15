from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_run():
    module = runpy.run_path(
        str(
            ROOT
            / "experiments"
            / "stage_06_low_budget_efficiency"
            / "stage06_v2_low_budget_comparison.py"
        )
    )
    return module["run"]


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_stage06_v2_low_budget_comparison_writes_expected_artifacts(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="stage06_v2_comparison_smoke",
        seeds=(0,),
        tier1_epochs=2,
        tier2_epochs=3,
        rescue_epochs=4,
        allow_rescue_tier3=False,
        batch_size=128,
        layer_dims=(64, 16, 10),
        transport_steps=2,
        eval_steps=5,
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()
    assert (run_dir / "comparison_report.json").exists()
    assert (run_dir / "comparison_report.md").exists()
    assert "stage06_v2_low_budget_comparison" in str(run_dir)
    assert "stage06_v1_low_budget_comparison" not in str(run_dir)

    rows = _read_csv(run_dir / "aggregate_runs.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")
    config = _read_json(run_dir / "config.json")

    assert len(rows) == 4
    method_names = {row["method_name"] for row in rows}
    assert method_names == {
        "stage06_v2_persistent_overlap_objective_curriculum_energydrop_default",
        "stage05_v3c_stronger_semigroup_weight",
    }

    assert summary["stage"] == "stage06_v2_low_budget_comparison"
    assert summary["candidate_name"] == (
        "stage06_v2_persistent_overlap_objective_curriculum_energydrop_default"
    )
    assert summary["matched_budget_control_name"] == "stage05_v3c_stronger_semigroup_weight"
    assert summary["candidate_objective_schedule_variant"] == "persistent_overlap"
    assert summary["candidate_hard_late_handoff_enabled"] is False
    assert summary["candidate_persistent_overlap_enabled"] is True
    assert summary["candidate_beta_obj_final_value"] == 0.75
    assert summary["candidate_late_phase_trajectory_weight"] == 0.25
    assert summary["candidate_late_phase_semigroup_weight"] == 0.75
    assert summary["stage05_two_branch_parameterization_preserved"] is True
    assert summary["stage05_target_builder_reuse_enabled"] is True
    assert summary["stage05_branchwise_supervision_preserved"] is False

    assert config["stage"] == "stage06_v2_low_budget_comparison"
    assert config["candidate_name"] == (
        "stage06_v2_persistent_overlap_objective_curriculum_energydrop_default"
    )
    assert config["matched_budget_control"] == "stage05_v3c_stronger_semigroup_weight"
    assert config["candidate_objective_schedule_variant"] == "persistent_overlap"
    assert config["candidate_hard_late_handoff_enabled"] is False
    assert config["candidate_persistent_overlap_enabled"] is True
    assert config["candidate_beta_obj_final_value"] == 0.75

    candidate_row = next(
        row
        for row in rows
        if row["method_name"]
        == "stage06_v2_persistent_overlap_objective_curriculum_energydrop_default"
    )
    candidate_summary = _read_json(Path(candidate_row["run_dir"]) / "summary.json")
    assert candidate_summary["stage"] == "stage06_v2_objective_curriculum"
    assert candidate_summary["candidate_name"] == (
        "stage06_v2_persistent_overlap_objective_curriculum_energydrop_default"
    )
    assert candidate_summary["objective_schedule_variant"] == "persistent_overlap"
    assert candidate_summary["hard_late_handoff_enabled"] is False
    assert candidate_summary["persistent_overlap_enabled"] is True
    assert candidate_summary["beta_obj_final_value"] == 0.75
    assert candidate_summary["late_phase_trajectory_weight"] == 0.25
    assert candidate_summary["late_phase_semigroup_weight"] == 0.75
    assert candidate_summary["stage05_branchwise_supervision_preserved"] is False
