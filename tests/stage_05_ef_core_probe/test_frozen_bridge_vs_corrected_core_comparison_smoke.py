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


def load_run():
    module = runpy.run_path(
        str(
            ROOT
            / "experiments"
            / "stage_05_ef_core_probe"
            / "frozen_bridge_vs_corrected_core_comparison.py"
        )
    )
    return module["run"]


def test_frozen_bridge_vs_corrected_core_comparison_writes_expected_artifacts(
    tmp_path: Path,
) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="frozen_bridge_vs_corrected_core_smoke",
        seeds=(0,),
        stage04_epochs=2,
        stage04_eval_steps=5,
        stage04_layer_dims=(64, 16, 10),
        stage05_epochs=3,
        stage05_eval_steps=5,
        stage05_layer_dims=(64, 16, 10),
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()
    assert (run_dir / "comparison_report.json").exists()
    assert (run_dir / "comparison_report.md").exists()

    rows = _read_csv(run_dir / "aggregate_runs.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")
    report = _read_json(run_dir / "comparison_report.json")

    assert len(rows) == 2
    method_names = {row["method_name"] for row in rows}
    assert method_names == {
        "stage_04_frozen_bridge",
        "stage_05_corrected_residual_core",
    }

    first_row = rows[0]
    assert "one_step_energy_delta_vs_identity" in first_row
    assert "configured_step_energy_delta_vs_identity" in first_row
    assert "configured_step_fixed_point_residual_delta_vs_identity" in first_row
    assert "configured_step_energy_delta_vs_local_field_only" in first_row
    assert "val_accuracy" in first_row
    assert "test_accuracy" in first_row
    assert "deterministic_artifact_checks_passed" in first_row

    assert "comparison_protocol" in summary
    assert "by_method" in summary
    assert "pairwise_stage05_vs_stage04" in summary
    assert "stage05_corrected_residual_core_justifies_v2_charter" in summary
    assert "stage05_v2_charter_decision_detail" in summary
    assert summary["comparison_report_json_path"] == "comparison_report.json"
    assert summary["comparison_report_md_path"] == "comparison_report.md"

    assert "decision" in report
    assert "supports" in report
    assert "does_not_support" in report
    assert "stage05_corrected_residual_core_justifies_v2_charter" in report["decision"]


def test_stage05_v1_vs_v2_comparison_writes_expected_artifacts(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="corrected_core_v1_vs_v2_smoke",
        comparison_variant="stage05_v1_vs_v2",
        seeds=(0,),
        stage05_epochs=3,
        stage05_eval_steps=5,
        stage05_layer_dims=(64, 16, 10),
        stage05_transport_steps=2,
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()
    assert (run_dir / "comparison_report.json").exists()
    assert (run_dir / "comparison_report.md").exists()

    rows = _read_csv(run_dir / "aggregate_runs.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")
    report = _read_json(run_dir / "comparison_report.json")

    assert len(rows) == 2
    method_names = {row["method_name"] for row in rows}
    assert method_names == {
        "stage_05_corrected_residual_core_v1",
        "stage_05_two_branch_corrected_residual_core_v2",
    }

    first_row = rows[0]
    assert "transport_family" in first_row
    assert "residual_branch_structure" in first_row
    assert "one_step_energy_delta_vs_identity" in first_row
    assert "configured_step_energy_delta_vs_identity" in first_row
    assert "configured_step_fixed_point_residual_delta_vs_identity" in first_row
    assert "deterministic_artifact_checks_passed" in first_row

    assert summary["stage"] == "corrected_residual_core_v1_vs_v2_comparison"
    assert "comparison_protocol" in summary
    assert "by_method" in summary
    assert "pairwise_stage05_v2_vs_v1" in summary
    assert "stage05_v2_improves_mechanism_magnitude_over_v1" in summary
    assert "stage05_v2_vs_v1_decision_detail" in summary
    assert summary["comparison_report_json_path"] == "comparison_report.json"
    assert summary["comparison_report_md_path"] == "comparison_report.md"

    assert "decision" in report
    assert "supports" in report
    assert "does_not_support" in report
    assert "stage05_v2_improves_mechanism_magnitude_over_v1" in report["decision"]
