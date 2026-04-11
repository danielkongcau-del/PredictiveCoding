from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_comparison_run():
    module = runpy.run_path(
        str(
            ROOT
            / "experiments"
            / "stage_05_ef_core_probe"
            / "frozen_bridge_vs_corrected_core_comparison.py"
        )
    )
    return module["run"]


def load_diagnostic_run():
    module = runpy.run_path(
        str(
            ROOT
            / "experiments"
            / "stage_05_ef_core_probe"
            / "stage05_v2_diagnostics.py"
        )
    )
    return module["run"]


def test_stage05_v2_diagnostics_writes_expected_artifacts(tmp_path: Path) -> None:
    source_result = load_comparison_run()(
        output_root=tmp_path / "source",
        run_id="stage05_v1_vs_v2_source_smoke",
        comparison_variant="stage05_v1_vs_v2",
        seeds=(0,),
        stage05_epochs=3,
        stage05_eval_steps=5,
        stage05_layer_dims=(64, 16, 10),
        stage05_transport_steps=2,
    )

    result = load_diagnostic_run()(
        output_root=tmp_path / "diagnostics",
        run_id="stage05_v2_diagnostics_smoke",
        source_comparison_dir=source_result.run_dir,
        seeds=(0,),
        near_final_window=2,
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "diagnostic_summary.json").exists()
    assert (run_dir / "diagnostic_report.md").exists()
    assert (run_dir / "epoch_diagnostics.csv").exists()
    assert (run_dir / "branch_diagnostics.csv").exists()

    config = _read_json(run_dir / "config.json")
    summary = _read_json(run_dir / "diagnostic_summary.json")
    epoch_rows = _read_csv(run_dir / "epoch_diagnostics.csv")
    branch_rows = _read_csv(run_dir / "branch_diagnostics.csv")

    assert config["stage"] == "stage05_v2_diagnostics"
    assert summary["stage"] == "stage05_v2_diagnostics"
    assert summary["source_artifacts"]["seeds"] == [0]
    assert "training_boundary" in summary
    assert "epoch_level" in summary
    assert "branch_contribution" in summary
    assert "rollout_diagnosis" in summary
    assert "selection_rule_diagnosis" in summary
    assert summary["selected_diagnosis_label"] in {
        "likely_undertrained",
        "state_branch_underutilized",
        "configured_step_rollout_accumulation_is_primary_gap",
        "selection_pressure_misaligned_with_report_accuracy",
    }
    assert summary["recommended_next_stage05_v3_target"] in {
        "longer_training_or_budget",
        "branch_strengthening",
        "rollout_aware_multi_step_strengthening",
        "selection_rule_refinement",
    }
    assert len(epoch_rows) == 3
    assert len(branch_rows) == 2
    assert "m_state_mean_l2" in branch_rows[0]
    assert "step_energy_delta_vs_identity" in branch_rows[0]
