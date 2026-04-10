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
    module = runpy.run_path(str(ROOT / "experiments" / "stage_04_incremental_bridge" / "fmpc_tf2_preterminal_source_localization_suite.py"))
    return module["run"]


def test_fmpc_tf2_preterminal_source_localization_suite_writes_expected_schema(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="tf2_preterminal_source_localization_smoke",
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

    assert len(aggregate_rows) == 5
    assert summary["stage"] == "adopted_package_preterminal_update_source_localization"
    assert "by_case" in summary
    assert "pairwise_vs_failed_anchor" in summary
    assert "diagnosis" in summary
    assert "recommended_next_narrow_tf2_move" in summary

    first_row = aggregate_rows[0]
    assert "direction_source_mode" in first_row
    assert "norm_handling_mode" in first_row
    assert "handoff_mode" in first_row
    assert "gate_passing_epoch_count" in first_row
    assert "val_terminal_rowspace_rms" in first_row
    assert "val_terminal_rowspace_fraction" in first_row
