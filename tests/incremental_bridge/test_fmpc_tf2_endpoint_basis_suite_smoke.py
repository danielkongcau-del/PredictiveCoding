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
    module = runpy.run_path(str(ROOT / "experiments" / "incremental_bridge" / "fmpc_tf2_endpoint_basis_suite.py"))
    return module["run"]


def test_fmpc_tf2_endpoint_basis_suite_writes_expected_schema(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="tf2_endpoint_basis_smoke",
        seeds=(0,),
        epochs=2,
        batch_size=64,
        eval_steps=5,
        layer_dims=(64, 16, 10),
        reference_summary_path=tmp_path / "missing_reference.json",
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()

    rows = _read_csv(run_dir / "aggregate_runs.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")

    assert len(rows) == 1
    assert summary["stage"] == "adopted_package_endpoint_basis_separability"
    assert "integrated_behavior" in summary
    assert "endpoint_cases" in summary
    assert "delta_geometry" in summary
    assert "pairwise_endpoint_deltas" in summary
    assert "validation_knot_breakdown" in summary
    assert "diagnosis" in summary
    assert "recommended_next_narrow_tf2_move" in summary

    first_row = rows[0]
    assert "integrated_val_accuracy" in first_row
    assert "transported_val_fisher_separability_ratio" in first_row
    assert "slow_pc_val_fisher_separability_ratio" in first_row
    assert "val_delta_h_rowspace_fraction" in first_row
    assert "val_knot_peak_fisher_gap_region" in first_row
