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
    module = runpy.run_path(str(ROOT / "experiments" / "stage_04_incremental_bridge" / "fmpc_tf2_bootstrap_fidelity_suite.py"))
    return module["run"]


def test_fmpc_tf2_bootstrap_fidelity_suite_writes_expected_schema(tmp_path: Path) -> None:
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
        run_id="tf2_bootstrap_fidelity_smoke",
        offline_probe_seeds=(0,),
        end_to_end_seeds=(0,),
        epochs=2,
        batch_size=64,
        eval_steps=5,
        layer_dims=(64, 16, 10),
        offline_probe_epoch_indices=(0, 1),
        sample_batches_per_probe_epoch=1,
        bootstrap_integrators=("euler", "rk2"),
        bootstrap_substeps_options=(1, 2),
        reference_substeps=8,
        max_pruned_nondefault_candidates=1,
        slow_pc_reference_runs_path=slow_runs,
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "offline_target_fidelity.csv").exists()
    assert (run_dir / "end_to_end_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()

    offline_rows = _read_csv(run_dir / "offline_target_fidelity.csv")
    end_to_end_rows = _read_csv(run_dir / "end_to_end_runs.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")

    assert len(offline_rows) == 5
    assert len(end_to_end_rows) >= 2
    assert "offline_reference_definition" in summary
    assert "pruned_candidates_for_end_to_end" in summary
    assert len(summary["pruned_candidates_for_end_to_end"]) <= 2
    assert "offline_target_fidelity_by_candidate" in summary
    assert "mean_std_val_accuracy_by_candidate" in summary
    assert "mean_std_test_accuracy_by_candidate" in summary
    assert "mean_gate_passing_epoch_count_by_candidate" in summary
    assert "mean_val_transported_final_energy_by_candidate" in summary
    assert "gap_to_canonical_slow_pc_by_candidate" in summary
    assert "mean_wall_clock_runtime_by_candidate" in summary
    assert "pairwise_delta_vs_current_corrective_default" in summary
    assert "best_end_to_end_candidate" in summary
    assert "does_bootstrap_target_fidelity_materially_improve_end_to_end_behavior" in summary
    assert "should_corrective_default_change" in summary
    assert "bootstrap_target_fidelity_is_current_bottleneck" in summary
    assert "next_single_narrow_research_move" in summary

    first_offline = offline_rows[0]
    assert "candidate_key" in first_offline
    assert "bootstrap_integrator" in first_offline
    assert "bootstrap_substeps" in first_offline
    assert "mean_relative_mse_to_reference_average_velocity" in first_offline
    assert "mean_wall_time_seconds_per_eval" in first_offline

    first_e2e = end_to_end_rows[0]
    assert "candidate_key" in first_e2e
    assert "val_accuracy" in first_e2e
    assert "test_accuracy" in first_e2e
    assert "total_wall_time_seconds" in first_e2e
    assert "run_status" in first_e2e
