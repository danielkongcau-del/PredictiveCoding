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
    module = runpy.run_path(str(ROOT / "experiments" / "fmpc_tf2_readout_refit_suite.py"))
    return module["run"]


def test_fmpc_tf2_readout_refit_suite_writes_expected_schema(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="tf2_readout_refit_smoke",
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

    assert len(rows) == 3
    assert "mean_std_val_accuracy_by_case" in summary
    assert "mean_std_test_accuracy_by_case" in summary
    assert "mean_std_report_output_mse_by_case" in summary
    assert "mean_std_supervised_output_mse_by_case" in summary
    assert "pairwise_transported_endpoint_readout_refit_vs_adopted_control" in summary
    assert "pairwise_slow_pc_endpoint_readout_refit_vs_adopted_control" in summary
    assert "pairwise_slow_pc_endpoint_readout_refit_vs_transported_endpoint_readout_refit" in summary
    assert "diagnosis" in summary
    assert "recommended_next_narrow_tf2_move" in summary

    first_row = rows[0]
    assert "case_name" in first_row
    assert "refit_basis" in first_row
    assert "readout_refit_selected_ridge" in first_row
