from __future__ import annotations

import csv
import json
from pathlib import Path

from pc.pc_sensitivity import run_pc_sensitivity_study


def test_pc_sensitivity_writes_expected_artifacts(tmp_path: Path) -> None:
    result = run_pc_sensitivity_study(
        "toy_regression",
        output_root=tmp_path,
        run_id="toy_regression_sensitivity",
        plot_energy=False,
        plot_summary=False,
    )

    run_dir = tmp_path / "pc_sensitivity_toy_regression"
    assert result.run_dir == run_dir
    assert (run_dir / "base_pc_config.json").exists()
    assert (run_dir / "candidate_grid.json").exists()
    assert (run_dir / "trial_table.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()
    assert (run_dir / "mlp_reference" / "summary.json").exists()

    trial_directories = sorted(path.name for path in (run_dir / "trials").iterdir() if path.is_dir())
    assert trial_directories == [
        "default",
        "eta_w_double",
        "eta_w_half",
        "eta_x_double",
        "eta_x_half",
        "state_init_zeros",
        "train_steps_double",
        "train_steps_half",
    ]

    with (run_dir / "candidate_grid.json").open("r", encoding="utf-8") as handle:
        candidate_grid = json.load(handle)
    with (run_dir / "aggregate_summary.json").open("r", encoding="utf-8") as handle:
        aggregate_summary = json.load(handle)
    with (run_dir / "trial_table.csv").open("r", encoding="utf-8", newline="") as handle:
        trial_rows = list(csv.DictReader(handle))

    assert candidate_grid["trial_design"] == "one_at_a_time"
    assert candidate_grid["notes"]["state_init"].startswith("state_init affects the full predictive-coding")
    assert aggregate_summary["trial_count"] == 8
    assert aggregate_summary["successful_trial_count"] == 8
    assert len(trial_rows) == 8
    assert trial_rows[0]["trial_id"] == "default"
    assert trial_rows[0]["primary_metric_delta_vs_default"] == "0.0"
    assert trial_rows[0]["final_pre_update_energy_delta_vs_default"] == "0.0"
