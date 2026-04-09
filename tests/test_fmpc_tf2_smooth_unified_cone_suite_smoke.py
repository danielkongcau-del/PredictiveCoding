from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_run():
    module = runpy.run_path(str(ROOT / "experiments" / "fmpc_tf2_smooth_unified_cone_suite.py"))
    return module["run"]


def test_fmpc_tf2_smooth_unified_cone_suite_writes_expected_schema(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="tf2_smooth_unified_cone_smoke",
        seeds=(0,),
        epochs=2,
        batch_size=64,
        eval_steps=5,
        layer_dims=(64, 16, 10),
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()

    rows = _read_csv(run_dir / "aggregate_runs.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")

    assert len(rows) == 2
    assert summary["stage"] == "adopted_package_smooth_unified_cone_diagnostic"
    assert "by_candidate" in summary
    assert "pairwise_vs_control" in summary
    assert "previous_hard_interior_reference" in summary
    assert "smooth_candidate_vs_previous_hard_interior_reference" in summary
    assert "diagnosis" in summary
    assert "decision" in summary
    assert "recommended_next_move" in summary

    first_row = rows[0]
    assert "terminal_local_field_direction_intervention" in first_row
    assert "terminal_local_field_angle_clip_degrees" in first_row
    assert "val_full_space_angle_stabilized_vs_lf_deg" in first_row
    assert "val_full_space_angle_above_30deg_rate_after_stabilization" in first_row
    assert "val_delta_h_rms_rowspace" in first_row
    assert "val_delta_h_rms_orthogonal" in first_row
