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
    module = runpy.run_path(str(ROOT / "experiments" / "stage_04_incremental_bridge" / "fmpc_tf2_successor_increment_source_suite.py"))
    return module["run"]


def test_fmpc_tf2_successor_increment_source_suite_writes_expected_schema(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="tf2_successor_increment_source_smoke",
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

    aggregate_rows = _read_csv(run_dir / "aggregate_runs.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")

    assert len(aggregate_rows) == 6
    assert summary["stage"] == "adopted_package_preterminal_successor_increment_source_localization"
    assert "increment_component_table" in summary
    assert "by_case" in summary
    assert "pairwise_vs_control" in summary
    assert "pairwise_vs_failed_anchor" in summary
    assert "recovery_fractions_vs_failed_anchor" in summary
    assert "control_relative_recovery" in summary
    assert "diagnosis" in summary
    assert "recommended_next_narrow_tf2_move" in summary

    first_row = aggregate_rows[0]
    assert "increment_mode" in first_row
    assert "gate_passing_epoch_count" in first_row
    assert "val_terminal_rowspace_rms" in first_row
    assert "val_terminal_rowspace_fraction" in first_row
