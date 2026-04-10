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
    module = runpy.run_path(str(ROOT / "experiments" / "stage_04_incremental_bridge" / "fmpc_tf2_terminal_localfield_trust_region_suite.py"))
    return module["run"]


def test_fmpc_tf2_terminal_localfield_trust_region_suite_writes_expected_schema(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        experiment_name="tf2_lftr_smoke",
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
    assert (run_dir / "terminal_localfield_trust_region_drift_epoch_metrics.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()

    run_rows = _read_csv(run_dir / "end_to_end_runs.csv")
    drift_rows = _read_csv(run_dir / "terminal_localfield_trust_region_drift_epoch_metrics.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")

    assert len(run_rows) == 7
    assert len(drift_rows) == 10

    first_run = run_rows[0]
    assert "handoff_mode" in first_run
    assert "candidate_key" in first_run
    assert "replay_prefix_steps" in first_run
    assert "direction_anchor_mode" in first_run
    assert "val_accuracy" in first_run
    assert "test_accuracy" in first_run
    assert "selected_epoch" in first_run
    assert "selected_epoch_passes_gate" in first_run
    assert "selector_fallback_used" in first_run
    assert "run_status" in first_run

    first_drift = drift_rows[0]
    assert "candidate_key" in first_drift
    assert "handoff_mode" in first_drift
    assert "replay_prefix_steps" in first_drift
    assert "direction_anchor_mode" in first_drift
    assert "relative_terminal_action_vector_delta_vs_baseline_replay" in first_drift
    assert "terminal_action_direction_cosine_similarity_vs_baseline_replay" in first_drift
    assert "terminal_action_direction_cosine_similarity_vs_terminal_local_field" in first_drift
    assert "terminal_action_norm_ratio_vs_live_raw" in first_drift
    assert "terminal_action_norm_delta_vs_live_raw" in first_drift
    assert "terminal_state_slot_displacement" in first_drift
    assert "terminal_next_state_displacement_vs_baseline_replay" in first_drift
    assert "terminal_next_state_displacement_vs_live_raw" in first_drift

    assert "end_to_end_summary" in summary
    assert "terminal_localfield_trust_region_drift_summary" in summary
    assert "pairwise_delta_vs_baseline_closed_loop_run" in summary
    assert "pairwise_delta_vs_challenger_terminal_live_raw_closed_loop" in summary
    assert "is_hard_replace_local_field_direction_rescue" in summary
    assert "is_strong_mix_local_field_direction_rescue" in summary
    assert "is_angle_clip_local_field_direction_rescue" in summary
    assert "is_any_local_field_trust_region_rescue" in summary
    assert "dominant_interpretation" in summary
    assert "next_single_narrow_move" in summary
