from __future__ import annotations

import csv
import json
from pathlib import Path

from pc.phase2g_matched_search import run_phase2g_matched_search


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_phase2g_matched_search_writes_expected_artifacts(tmp_path: Path) -> None:
    result = run_phase2g_matched_search(
        "toy_regression",
        output_root=tmp_path,
        run_id="phase2g_fixture",
        plot_energy=False,
        pc_search_space_override={
            "eta_x": [0.1, 0.2],
            "eta_w": [0.1],
            "train_steps": [25],
            "epochs": [4],
        },
        mlp_search_space_override={
            "eta_w": [0.05, 0.1],
            "epochs": [4],
        },
    )

    run_dir = tmp_path / "phase2g_matched_search" / "toy_regression"
    assert result.run_dir == run_dir
    assert (run_dir / "study_config.json").exists()
    assert (run_dir / "pc_search_results.csv").exists()
    assert (run_dir / "mlp_search_results.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()
    assert (run_dir / "best_pc_config_summary.json").exists()
    assert (run_dir / "best_mlp_config_summary.json").exists()

    pc_trial_directories = sorted(path.name for path in (run_dir / "pc_trials").iterdir() if path.is_dir())
    mlp_trial_directories = sorted(path.name for path in (run_dir / "mlp_trials").iterdir() if path.is_dir())
    assert pc_trial_directories == ["cfg_001", "cfg_002"]
    assert mlp_trial_directories == ["cfg_001", "cfg_002"]

    study_config = _load_json(run_dir / "study_config.json")
    aggregate_summary = _load_json(run_dir / "aggregate_summary.json")
    best_pc_summary = _load_json(run_dir / "best_pc_config_summary.json")
    best_mlp_summary = _load_json(run_dir / "best_mlp_config_summary.json")
    pc_rows = _load_csv(run_dir / "pc_search_results.csv")
    mlp_rows = _load_csv(run_dir / "mlp_search_results.csv")

    assert study_config["search_target"] == "matched_pc_and_mlp"
    assert study_config["selected_by_metric_source"] == "val_metric"
    assert study_config["final_report_metric_source"] == "test_metric"
    assert study_config["test_metric_used_for_selection"] is False
    assert study_config["pc_search_space_size"] == 2
    assert study_config["mlp_search_space_size"] == 2

    assert aggregate_summary["pc_trial_count"] == 2
    assert aggregate_summary["mlp_trial_count"] == 2
    assert aggregate_summary["selected_by_metric_source"] == "val_metric"
    assert aggregate_summary["final_report_metric_source"] == "test_metric"
    assert aggregate_summary["best_pc_config_id"] == best_pc_summary["best_config_id"]
    assert aggregate_summary["best_mlp_config_id"] == best_mlp_summary["best_config_id"]

    assert len(pc_rows) == 2
    assert len(mlp_rows) == 2
    assert sorted(row["selection_metric_rank"] for row in pc_rows) == ["1", "2"]
    assert sorted(row["selection_metric_rank"] for row in mlp_rows) == ["1", "2"]
    assert all(row["selection_metric_source"] == "val_metric" for row in pc_rows)
    assert all(row["report_metric_source"] == "test_metric" for row in pc_rows)
    assert all(row["selection_metric_source"] == "val_metric" for row in mlp_rows)
    assert all(row["report_metric_source"] == "test_metric" for row in mlp_rows)
