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
    module = runpy.run_path(str(ROOT / "experiments" / "fmpc_tf2_basis_drift_localization_suite.py"))
    return module["run"]


def test_fmpc_tf2_basis_drift_localization_suite_writes_expected_schema(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="tf2_basis_drift_localization_smoke",
        seeds=(0,),
        epochs=2,
        batch_size=64,
        eval_steps=5,
        layer_dims=(64, 16, 10),
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "knotwise_localization.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()

    aggregate_rows = _read_csv(run_dir / "aggregate_runs.csv")
    knot_rows = _read_csv(run_dir / "knotwise_localization.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")

    assert len(aggregate_rows) == 1
    assert len(knot_rows) >= 10
    assert summary["stage"] == "adopted_package_basis_drift_source_localization"
    assert "integrated_behavior" in summary
    assert "validation_knot_breakdown" in summary
    assert "test_knot_breakdown" in summary
    assert "diagnosis" in summary
    assert "recommended_next_narrow_tf2_move" in summary

    aggregate_row = aggregate_rows[0]
    assert "val_preterminal_rowspace_share" in aggregate_row
    assert "val_terminal_rowspace_share" in aggregate_row
    assert "test_preterminal_output_share" in aggregate_row
    assert "runtime_proxy_seconds" in aggregate_row

    knot_row = knot_rows[0]
    assert "hidden_state_rms_gap_to_slow_pc" in knot_row
    assert "output_state_rms_gap_to_slow_pc" in knot_row
    assert "hidden_state_rms_gap_rowspace" in knot_row
    assert "hidden_state_rms_gap_orthogonal" in knot_row
    assert "hidden_state_gap_rowspace_fraction" in knot_row
    assert "centroid_displacement_rms_rowspace" in knot_row
    assert "transported_fisher_separability_ratio" in knot_row
    assert "slow_pc_fisher_separability_ratio" in knot_row
