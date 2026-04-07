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
    module = runpy.run_path(str(ROOT / "experiments" / "fmpc_tf2_batch_frozen_target_state_suite.py"))
    return module["run"]


def test_fmpc_tf2_batch_frozen_target_state_suite_writes_expected_schema(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="tf2_batch_frozen_target_state_smoke",
        seeds=(0,),
        epochs=2,
        batch_size=64,
        eval_steps=5,
        layer_dims=(64, 16, 10),
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "end_to_end_runs.csv").exists()
    assert (run_dir / "within_batch_drift_epoch_metrics.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()

    run_rows = _read_csv(run_dir / "end_to_end_runs.csv")
    drift_rows = _read_csv(run_dir / "within_batch_drift_epoch_metrics.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")

    assert len(run_rows) == 4
    assert len(drift_rows) == 8

    first_run = run_rows[0]
    assert "mode" in first_run
    assert "candidate_key" in first_run
    assert "val_accuracy" in first_run
    assert "test_accuracy" in first_run
    assert "selected_epoch" in first_run
    assert "selected_epoch_passes_gate" in first_run
    assert "selector_fallback_used" in first_run
    assert "run_status" in first_run

    first_drift = drift_rows[0]
    assert "mode" in first_drift
    assert "candidate_key" in first_drift
    assert "epoch" in first_drift
    assert "mean_relative_bootstrap_target_delta" in first_drift
    assert "mean_relative_identity_target_delta" in first_drift
    assert "mean_bootstrap_target_cosine_similarity" in first_drift
    assert "mean_identity_target_cosine_similarity" in first_drift
    assert "mean_relative_psi_input_delta" in first_drift

    assert "end_to_end_summary" in summary
    assert "within_batch_drift_summary" in summary
    assert "pairwise_delta_vs_baseline_live_target_run" in summary
    assert "pairwise_delta_vs_same_family_live_target_run" in summary
    assert "is_within_batch_live_target_state_coupling_present" in summary
    assert "is_challenger_rescued_by_batch_frozen_target_state_cache" in summary
    assert "dominant_interpretation" in summary
    assert "next_single_narrow_move" in summary
