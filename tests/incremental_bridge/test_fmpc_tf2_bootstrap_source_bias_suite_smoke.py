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
    module = runpy.run_path(str(ROOT / "experiments" / "incremental_bridge" / "fmpc_tf2_bootstrap_source_bias_suite.py"))
    return module["run"]


def test_fmpc_tf2_bootstrap_source_bias_suite_writes_expected_schema(tmp_path: Path) -> None:
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
        run_id="tf2_bootstrap_source_bias_smoke",
        offline_probe_seeds=(0,),
        end_to_end_seeds=(0,),
        epochs=2,
        batch_size=64,
        eval_steps=5,
        layer_dims=(64, 16, 10),
        offline_probe_epoch_indices=(0, 1),
        sample_batches_per_probe_epoch=1,
        diagnostic_slow_pc_steps=(4,),
        max_pruned_diagnostic_challengers=1,
        slow_pc_reference_runs_path=slow_runs,
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "offline_source_bias.csv").exists()
    assert (run_dir / "end_to_end_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()

    offline_rows = _read_csv(run_dir / "offline_source_bias.csv")
    end_to_end_rows = _read_csv(run_dir / "end_to_end_runs.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")

    assert len(offline_rows) > 0
    assert len(end_to_end_rows) == 2
    assert "offline_source_bias_by_candidate" in summary
    assert "pruned_candidates_for_end_to_end" in summary
    assert len(summary["pruned_candidates_for_end_to_end"]) <= 2
    assert "mean_std_val_accuracy_by_candidate" in summary
    assert "mean_std_test_accuracy_by_candidate" in summary
    assert "mean_gate_passing_epoch_count_by_candidate" in summary
    assert "mean_val_transported_final_energy_by_candidate" in summary
    assert "gap_to_canonical_slow_pc_by_candidate" in summary
    assert "mean_wall_clock_runtime_by_candidate" in summary
    assert "pairwise_delta_vs_current_corrective_default" in summary
    assert "best_end_to_end_candidate" in summary
    assert "is_bootstrap_target_bottlenecked_by_terminal_source_bias" in summary
    assert "does_detached_slow_pc_source_materially_beat_local_field_source" in summary
    assert "current_mainline_safe_result" in summary
    assert "diagnostic_only_finding" in summary
    assert "next_teacher_free_surrogate_if_source_bias_wins" in summary
    assert "next_single_narrow_research_move" in summary

    first_offline = offline_rows[0]
    assert "candidate_key" in first_offline
    assert "source_family" in first_offline
    assert "mse_to_local_field_average_velocity" in first_offline
    assert "endpoint_hidden_energy" in first_offline
    assert "endpoint_output_mse" in first_offline
    assert "endpoint_accuracy" in first_offline

    first_e2e = end_to_end_rows[0]
    assert "candidate_key" in first_e2e
    assert "source_family" in first_e2e
    assert "diagnostic_only" in first_e2e
    assert "val_accuracy" in first_e2e
    assert "test_accuracy" in first_e2e
    assert "total_wall_time_seconds" in first_e2e
    assert "run_status" in first_e2e
