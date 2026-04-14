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
            / "stage06_v1_low_budget_comparison.py"
        )
    )
    return module["run"]


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_stage06_low_budget_comparison_writes_expected_artifacts(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="stage06_comparison_smoke",
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

    rows = _read_csv(run_dir / "aggregate_runs.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")
    report = _read_json(run_dir / "comparison_report.json")

    assert len(rows) == 4
    method_names = {row["method_name"] for row in rows}
    assert method_names == {
        "stage06_v1_objective_curriculum_energydrop_default",
        "stage05_v3c_stronger_semigroup_weight",
    }

    assert summary["stage"] == "stage06_v1_low_budget_comparison"
    assert "comparison_protocol" in summary
    assert "tier_summaries" in summary
    assert "passes_tier1_viability" in summary
    assert "passes_tier2_main_gate" in summary
    assert "tier2_positive_trend_for_rescue" in summary
    assert "rescue_512_warranted" in summary
    assert "materially_beats_matched_budget_stage05_control" in summary
    assert "shows_better_cost_effectiveness_than_stage05_control" in summary
    assert "recommended_stage06_next_move" in summary
    assert "tier_2" in summary["tier_summaries"]
    assert "tier_3" in summary["tier_summaries"]
    assert "tier_2" in report["tier_summaries"]
    assert "tier_3" in report["tier_summaries"]
