from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("rows must contain at least one item.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_run():
    module = runpy.run_path(str(ROOT / "experiments" / "incremental_bridge" / "fmpc_tf2_microstep_horizon_suite.py"))
    return module["run"]


def test_fmpc_tf2_microstep_horizon_suite_writes_expected_schema(tmp_path: Path) -> None:
    slow_runs = tmp_path / "refs" / "slow_runs.csv"
    _write_csv(
        slow_runs,
        [
            {
                "preset_name": "canonical_slow_pc_digits_baseline",
                "seed": 0,
                "val_accuracy": 0.86,
                "test_accuracy": 0.88,
            }
        ],
    )

    result = load_run()(
        output_root=tmp_path,
        run_id="tf2_microstep_horizon_smoke",
        protocols=("fixed_outer_training", "matched_inner_compute"),
        micro_steps_options=(4, 6),
        seeds=(0,),
        base_epochs=2,
        batch_size=64,
        eval_steps=5,
        layer_dims=(64, 16, 10),
        slow_pc_reference_runs_path=slow_runs,
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()

    rows = _read_csv(run_dir / "aggregate_runs.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")

    assert len(rows) == 4
    assert "mean_std_val_accuracy_by_protocol_and_micro_steps" in summary
    assert "mean_std_test_accuracy_by_protocol_and_micro_steps" in summary
    assert "mean_gate_passing_epoch_count_by_protocol_and_micro_steps" in summary
    assert "mean_val_transported_final_energy_by_protocol_and_micro_steps" in summary
    assert "mean_wall_clock_runtime_by_protocol_and_micro_steps" in summary
    assert "effective_microstep_budget_by_protocol_and_micro_steps" in summary
    assert "gap_to_canonical_slow_pc_by_protocol_and_micro_steps" in summary
    assert "pairwise_comparison_vs_micro_steps_4_by_protocol" in summary
    assert "does_micro_steps_greater_than_4_win_under_matched_inner_compute" in summary
    assert "recommended_micro_steps_default" in summary
    assert "transport_horizon_vs_compute_interpretation" in summary
    assert "next_single_narrow_research_move" in summary

    first_row = rows[0]
    assert "protocol" in first_row
    assert "micro_steps" in first_row
    assert "epochs_used" in first_row
    assert "effective_microstep_budget" in first_row
    assert "train_wall_time_seconds" in first_row
    assert "total_wall_time_seconds" in first_row
    assert "run_status" in first_row
    assert "nan_or_inf_failure" in first_row
