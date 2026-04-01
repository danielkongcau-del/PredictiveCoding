from __future__ import annotations

import csv
import json
from pathlib import Path

from pc.phase2g1_boundary_check import run_phase2g1_boundary_check


def _write_phase2g_reference_artifacts(tmp_path: Path, benchmark_name: str) -> None:
    reference_dir = tmp_path / "phase2g_matched_search" / benchmark_name
    reference_dir.mkdir(parents=True, exist_ok=True)
    study_config = {
        "run_id": "phase2g_fixture",
        "benchmark_name": benchmark_name,
        "pc_search_space": {
            "eta_x": [0.05, 0.1],
            "eta_w": [0.2, 0.4],
            "train_steps": [25, 50],
            "epochs": [4, 6],
        },
        "mlp_search_space": {
            "eta_w": [0.1, 0.2],
            "epochs": [4, 6],
        },
    }
    aggregate_summary = {
        "run_id": "phase2g_fixture",
        "test_winner": "pc",
        "test_winner_reason": "predictive coding has lower mse on test",
    }
    best_pc_summary = {
        "run_id": "phase2g_fixture",
        "best_config_id": "cfg_pc",
        "selection_metric_value": 0.10,
        "val_metric": 0.10,
        "test_metric": 0.11,
        "best_config": {
            "eta_x": 0.05,
            "eta_w": 0.4,
            "eta_b": 0.4,
            "train_steps": 25,
            "eval_steps": 25,
            "epochs": 6,
            "state_init": "forward",
        },
    }
    best_mlp_summary = {
        "run_id": "phase2g_fixture",
        "best_config_id": "cfg_mlp",
        "selection_metric_value": 0.12,
        "val_metric": 0.12,
        "test_metric": 0.13,
        "best_config": {
            "eta_w": 0.2,
            "eta_b": 0.2,
            "epochs": 6,
        },
    }
    for name, payload in (
        ("study_config.json", study_config),
        ("aggregate_summary.json", aggregate_summary),
        ("best_pc_config_summary.json", best_pc_summary),
        ("best_mlp_config_summary.json", best_mlp_summary),
    ):
        with (reference_dir / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_phase2g1_boundary_check_writes_expected_artifacts(tmp_path: Path) -> None:
    _write_phase2g_reference_artifacts(tmp_path, "toy_regression")
    result = run_phase2g1_boundary_check(
        "toy_regression",
        output_root=tmp_path,
        previous_search_output_root=tmp_path,
        run_id="phase2g1_fixture",
        plot_energy=False,
        pc_boundary_extensions_override={
            "eta_x": [0.025],
            "eta_w": [0.6],
            "epochs": [8],
        },
        mlp_boundary_extensions_override={
            "eta_w": [0.3],
            "epochs": [8],
        },
    )

    run_dir = tmp_path / "phase2g1_boundary_check" / "toy_regression"
    assert result.run_dir == run_dir
    assert (run_dir / "study_config.json").exists()
    assert (run_dir / "pc_boundary_results.csv").exists()
    assert (run_dir / "mlp_boundary_results.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()
    assert (run_dir / "best_pc_config_summary.json").exists()
    assert (run_dir / "best_mlp_config_summary.json").exists()

    study_config = _load_json(run_dir / "study_config.json")
    aggregate_summary = _load_json(run_dir / "aggregate_summary.json")
    best_pc_summary = _load_json(run_dir / "best_pc_config_summary.json")
    best_mlp_summary = _load_json(run_dir / "best_mlp_config_summary.json")
    pc_rows = _load_csv(run_dir / "pc_boundary_results.csv")
    mlp_rows = _load_csv(run_dir / "mlp_boundary_results.csv")

    assert study_config["selected_by_metric_source"] == "val_metric"
    assert study_config["final_report_metric_source"] == "test_metric"
    assert study_config["previous_phase2g_run_id"] == "phase2g_fixture"
    assert study_config["pc_boundary_report"]["probed_dimensions"] == ["eta_x", "eta_w", "epochs"]
    assert study_config["pc_boundary_report"]["unprobed_boundary_dimensions"] == ["train_steps"]

    assert aggregate_summary["selection_split"] == "validation"
    assert aggregate_summary["final_report_split"] == "test"
    assert aggregate_summary["previous_phase2g_test_winner"] == "pc"
    assert aggregate_summary["boundary_check_best_pc_config_id"] == best_pc_summary["boundary_check_best_config_id"]
    assert aggregate_summary["boundary_check_best_mlp_config_id"] == best_mlp_summary["boundary_check_best_config_id"]
    assert aggregate_summary["pc_unprobed_boundary_dimensions"] == ["train_steps"]

    assert len(pc_rows) == 5
    assert len(mlp_rows) == 4
    assert sorted(row["selection_metric_rank"] for row in pc_rows) == ["1", "2", "3", "4", "5"]
    assert sorted(row["selection_metric_rank"] for row in mlp_rows) == ["1", "2", "3", "4"]
    assert all(row["selection_metric_source"] == "val_metric" for row in pc_rows)
    assert all(row["report_metric_source"] == "test_metric" for row in pc_rows)
    assert all(row["selection_metric_source"] == "val_metric" for row in mlp_rows)
    assert all(row["report_metric_source"] == "test_metric" for row in mlp_rows)
