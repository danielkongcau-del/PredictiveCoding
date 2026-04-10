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
    module = runpy.run_path(str(ROOT / "experiments" / "tf2" / "fmpc_tf2_downstream_coupling_suite.py"))
    return module["run"]


def test_fmpc_tf2_downstream_coupling_suite_writes_expected_schema(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="tf2_downstream_coupling_smoke",
        seeds=(0,),
        epochs=2,
        batch_size=64,
        eval_steps=5,
        layer_dims=(64, 16, 10),
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "end_to_end_runs.csv").exists()
    assert (run_dir / "selector_epoch_metrics.csv").exists()
    assert (run_dir / "selector_reselection.csv").exists()
    assert (run_dir / "replay_vs_self_drift.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()

    run_rows = _read_csv(run_dir / "end_to_end_runs.csv")
    selector_epoch_rows = _read_csv(run_dir / "selector_epoch_metrics.csv")
    selector_rows = _read_csv(run_dir / "selector_reselection.csv")
    drift_rows = _read_csv(run_dir / "replay_vs_self_drift.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")

    assert len(run_rows) == 4
    assert len(selector_epoch_rows) == 8
    assert len(selector_rows) == 12
    assert len(drift_rows) == 4

    assert "live_theta_end_to_end" in summary
    assert "frozen_theta_end_to_end" in summary
    assert "live_vs_frozen_pairwise_gap" in summary
    assert "replay_vs_self_drift_summary" in summary
    assert "selector_reselection_summary_live_theta" in summary
    assert "is_rollout_state_distribution_drift_present" in summary
    assert "is_moving_target_theta_coupling_present" in summary
    assert "is_selector_sensitivity_the_main_limiter" in summary
    assert "oracle_reselection_still_fails_to_rescue_challenger" in summary
    assert "dominant_bottleneck" in summary
    assert "next_single_narrow_move" in summary

    first_run = run_rows[0]
    assert "theta_mode" in first_run
    assert "candidate_key" in first_run
    assert "val_accuracy" in first_run
    assert "test_accuracy" in first_run
    assert "run_status" in first_run

    first_selector_epoch = selector_epoch_rows[0]
    assert "theta_mode" in first_selector_epoch
    assert "candidate_key" in first_selector_epoch
    assert "epoch" in first_selector_epoch
    assert "train_loss" in first_selector_epoch
    assert "gate_pass" in first_selector_epoch

    first_selector = selector_rows[0]
    assert "selector_name" in first_selector
    assert "selected_epoch" in first_selector
    assert "test_accuracy" in first_selector

    first_drift = drift_rows[0]
    assert "state_mode" in first_drift
    assert "bootstrap_target_mse" in first_drift
    assert "identity_target_mse" in first_drift
    assert "identity_residual_error" in first_drift
    assert "hybrid_loss" in first_drift
    assert "rollout_final_hidden_energy" in first_drift
