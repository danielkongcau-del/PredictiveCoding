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
    module = runpy.run_path(str(ROOT / "experiments" / "stage_04_incremental_bridge" / "fmpc_tf2_curriculum_suite.py"))
    return module["run"]


def test_fmpc_tf2_curriculum_suite_writes_expected_schema(tmp_path: Path) -> None:
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
        run_id="tf2_curriculum_smoke",
        seeds=(0,),
        epochs=2,
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

    assert len(rows) >= 8
    assert "tested_configuration_order" in summary
    assert "staged_search_trace" in summary
    assert "mean_std_val_accuracy_by_configuration" in summary
    assert "mean_std_test_accuracy_by_configuration" in summary
    assert "mean_gate_passing_epoch_count_by_configuration" in summary
    assert "mean_val_transported_final_energy_by_configuration" in summary
    assert "mean_wall_clock_runtime_by_configuration" in summary
    assert "gap_to_canonical_slow_pc_by_configuration" in summary
    assert "pairwise_comparison_against_current_corrective_default" in summary
    assert "best_configuration_by_mean_test_accuracy" in summary
    assert "does_better_bootstrap_identity_curriculum_improve_fixed_4_step_transport_quality" in summary
    assert "should_corrective_default_change" in summary
    assert "if_no_improvement_next_move_is_bootstrap_target_fidelity" in summary
    assert "bootstrap_identity_curriculum_interpretation" in summary
    assert "next_single_narrow_research_move" in summary

    first_row = rows[0]
    assert "candidate_key" in first_row
    assert "identity_loss_weight" in first_row
    assert "warmup_epochs" in first_row
    assert "hybrid_ramp_epochs" in first_row
    assert "train_wall_time_seconds" in first_row
    assert "total_wall_time_seconds" in first_row
    assert "run_status" in first_row
    assert "nan_or_inf_failure" in first_row
