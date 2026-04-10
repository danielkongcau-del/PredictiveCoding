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
    module = runpy.run_path(str(ROOT / "experiments" / "tf2" / "fmpc_tf2_readout_alignment_suite.py"))
    return module["run"]


def test_fmpc_tf2_readout_alignment_suite_writes_expected_schema(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="tf2_readout_alignment_smoke",
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

    assert len(rows) == 3
    assert "mean_std_val_accuracy_by_candidate" in summary
    assert "mean_std_test_accuracy_by_candidate" in summary
    assert "mean_std_report_output_mse_by_candidate" in summary
    assert "mean_std_supervised_transported_output_mse_by_candidate" in summary
    assert "mean_std_internal_slow_pc_output_mse_by_candidate" in summary
    assert "mean_std_transport_minus_internal_slow_pc_output_mse_by_candidate" in summary
    assert "pairwise_vs_adopted_control" in summary
    assert "should_promote_readout_alignment_variant" in summary
    assert "adoption_decision" in summary

    first_row = rows[0]
    assert "candidate_name" in first_row
    assert "transported_output_alignment_weight" in first_row
    assert "transported_output_alignment_schedule" in first_row
