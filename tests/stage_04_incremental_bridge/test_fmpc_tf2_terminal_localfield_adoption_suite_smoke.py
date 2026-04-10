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
    module = runpy.run_path(str(ROOT / "experiments" / "stage_04_incremental_bridge" / "fmpc_tf2_terminal_localfield_adoption_suite.py"))
    return module["run"]


def test_fmpc_tf2_terminal_localfield_adoption_suite_writes_expected_schema(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        experiment_name="tf2_lfa_smoke",
        run_id="s",
        seeds=(0,),
        epochs=2,
        batch_size=64,
        eval_steps=5,
        layer_dims=(64, 16, 10),
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "end_to_end_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()

    run_rows = _read_csv(run_dir / "end_to_end_runs.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")

    assert len(run_rows) == 7

    first_run = run_rows[0]
    assert "stabilizer_mode" in first_run
    assert "candidate_key" in first_run
    assert "terminal_direction_intervention" in first_run
    assert "strong_mix_weight_toward_local_field" in first_run
    assert "angle_clip_degrees" in first_run
    assert "val_accuracy" in first_run
    assert "test_accuracy" in first_run
    assert "gate_passing_epoch_count" in first_run
    assert "selected_epoch_passes_gate" in first_run
    assert "selector_fallback_used" in first_run
    assert "run_status" in first_run

    assert "end_to_end_summary" in summary
    assert "pairwise_delta_vs_baseline_closed_loop_run" in summary
    assert "pairwise_delta_vs_terminal_live_raw_closed_loop" in summary
    assert "adoption_thresholds" in summary
    assert "candidate_adoption_assessment" in summary
    assert "best_mean_test_candidate" in summary
    assert "recommended_next_tf2_experimental_default" in summary
    assert "is_any_terminal_local_field_stabilizer_strong_enough_for_adoption" in summary
    assert "is_hard_replace_necessary" in summary
    assert "tradeoff_note" in summary
    assert "dominant_interpretation" in summary
    assert "next_single_narrow_move" in summary
