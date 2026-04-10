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
    module = runpy.run_path(str(ROOT / "experiments" / "stage_04_incremental_bridge" / "fmpc_tf2_late_rollout_drift_control_suite.py"))
    return module["run"]


def test_fmpc_tf2_late_rollout_drift_control_suite_writes_expected_schema(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="tf2_late_rollout_drift_control_smoke",
        seeds=(0,),
        epochs=2,
        batch_size=64,
        eval_steps=5,
        layer_dims=(64, 16, 10),
        include_preterminal_only_reference=True,
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "knotwise_drift_metrics.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()

    aggregate_rows = _read_csv(run_dir / "aggregate_runs.csv")
    knot_rows = _read_csv(run_dir / "knotwise_drift_metrics.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")

    assert len(aggregate_rows) == 4
    assert len(knot_rows) >= 40
    assert summary["stage"] == "adopted_package_late_rollout_drift_control_diagnostic"
    assert "by_case" in summary
    assert "pairwise_vs_control" in summary
    assert "diagnosis" in summary
    assert "recommended_next_narrow_tf2_move" in summary

    aggregate_row = aggregate_rows[0]
    assert "intervention_step_offsets" in aggregate_row
    assert "val_terminal_rowspace_rms" in aggregate_row
    assert "val_terminal_rowspace_fraction" in aggregate_row
    assert "val_preterminal_rowspace_share" in aggregate_row
    assert "val_preterminal_output_share" in aggregate_row

    knot_row = knot_rows[0]
    assert "case_name" in knot_row
    assert "hidden_state_rms_gap_rowspace" in knot_row
    assert "hidden_state_rms_gap_orthogonal" in knot_row
    assert "hidden_state_gap_rowspace_fraction" in knot_row
    assert "output_state_rms_gap" in knot_row
