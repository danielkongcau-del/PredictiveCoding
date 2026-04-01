from __future__ import annotations

import csv
import json
from pathlib import Path

from pc.pc_joint_search import run_pc_joint_search


def test_pc_joint_search_writes_expected_artifacts(tmp_path: Path) -> None:
    result = run_pc_joint_search(
        "toy_regression",
        output_root=tmp_path,
        run_id="joint_search_fixture",
        plot_energy=False,
        search_space_override={
            "eta_x": [0.1, 0.2],
            "eta_w": [0.1],
            "train_steps": [25],
            "epochs": [4],
        },
    )

    run_dir = tmp_path / "phase2f_joint_search" / "toy_regression"
    assert result.run_dir == run_dir
    assert (run_dir / "study_config.json").exists()
    assert (run_dir / "search_results.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()
    assert (run_dir / "best_config_summary.json").exists()
    assert (run_dir / "mlp_reference" / "summary.json").exists()

    trial_directories = sorted(path.name for path in (run_dir / "trials").iterdir() if path.is_dir())
    assert trial_directories == ["cfg_001", "cfg_002"]

    with (run_dir / "study_config.json").open("r", encoding="utf-8") as handle:
        study_config = json.load(handle)
    with (run_dir / "aggregate_summary.json").open("r", encoding="utf-8") as handle:
        aggregate_summary = json.load(handle)
    with (run_dir / "best_config_summary.json").open("r", encoding="utf-8") as handle:
        best_config_summary = json.load(handle)
    with (run_dir / "search_results.csv").open("r", encoding="utf-8", newline="") as handle:
        search_rows = list(csv.DictReader(handle))

    assert study_config["ranking_metric_source"] == "val_metric"
    assert study_config["total_search_space_size"] == 2
    assert aggregate_summary["trial_count"] == 2
    assert aggregate_summary["successful_trial_count"] == 2
    assert best_config_summary["best_config_id"] == aggregate_summary["best_config_id"]
    assert len(search_rows) == 2
    assert sorted(row["val_metric_rank"] for row in search_rows) == ["1", "2"]
    assert all(row["metric_name"] == "mse" for row in search_rows)
    assert all(row["train_metric"] != "" for row in search_rows)
    assert all(row["val_metric"] != "" for row in search_rows)
    assert all(row["test_metric"] != "" for row in search_rows)
    assert best_config_summary["selection_metric_source"] == "val_metric"
    assert best_config_summary["report_metric_source"] == "test_metric"
