from __future__ import annotations

import csv
import json
from pathlib import Path

from pc.pc_multiseed import run_pc_multiseed_study


def _write_phase2g1_best_config_summaries(tmp_path: Path, benchmark_name: str) -> None:
    best_config_dir = tmp_path / "phase2g1_boundary_check" / benchmark_name
    best_config_dir.mkdir(parents=True, exist_ok=True)
    pc_payload = {
        "run_id": "phase2g1_fixture",
        "boundary_check_best_config_id": "cfg_pc_refined",
        "boundary_check_val_metric": 0.01,
        "boundary_check_test_metric": 0.011,
        "boundary_check_best_config": {
            "eta_x": 0.2,
            "eta_w": 0.4,
            "eta_b": 0.4,
            "train_steps": 10,
            "eval_steps": 10,
            "epochs": 120,
            "state_init": "forward",
        },
    }
    mlp_payload = {
        "run_id": "phase2g1_fixture",
        "boundary_check_best_config_id": "cfg_mlp_refined",
        "boundary_check_val_metric": 0.02,
        "boundary_check_test_metric": 0.021,
        "boundary_check_best_config": {
            "eta_w": 0.2,
            "eta_b": 0.2,
            "epochs": 160,
        },
    }
    with (best_config_dir / "best_pc_config_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(pc_payload, handle, indent=2)
    with (best_config_dir / "best_mlp_config_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(mlp_payload, handle, indent=2)


def test_pc_multiseed_writes_expected_artifacts(tmp_path: Path) -> None:
    _write_phase2g1_best_config_summaries(tmp_path, "toy_regression")
    result = run_pc_multiseed_study(
        "toy_regression",
        output_root=tmp_path,
        run_id="toy_regression_multiseed",
        plot_energy=False,
        plot_summary=False,
        seed_values=[0, 1],
        joint_search_output_root=tmp_path,
    )

    run_dir = tmp_path / "pc_multiseed_phase2g1_toy_regression"
    assert result.run_dir == run_dir
    assert (run_dir / "study_config.json").exists()
    assert (run_dir / "seed_records.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()
    assert (run_dir / "seeds" / "seed_0000" / "default_pc" / "summary.json").exists()
    assert (run_dir / "seeds" / "seed_0000" / "tuned_pc" / "summary.json").exists()
    assert (run_dir / "seeds" / "seed_0000" / "mlp" / "summary.json").exists()
    assert (run_dir / "seeds" / "seed_0001" / "default_pc" / "summary.json").exists()
    assert (run_dir / "seeds" / "seed_0001" / "tuned_pc" / "summary.json").exists()
    assert (run_dir / "seeds" / "seed_0001" / "mlp" / "summary.json").exists()

    with (run_dir / "study_config.json").open("r", encoding="utf-8") as handle:
        study_config = json.load(handle)
    with (run_dir / "seed_records.csv").open("r", encoding="utf-8", newline="") as handle:
        seed_rows = list(csv.DictReader(handle))
    with (run_dir / "aggregate_summary.json").open("r", encoding="utf-8") as handle:
        aggregate_summary = json.load(handle)

    assert study_config["seed_values"] == [0, 1]
    assert study_config["seed_semantics"]["data_seed"] == "fixed"
    assert study_config["seed_semantics"]["run_seed"] == "varies_with_seed"
    assert study_config["seed_semantics"]["model_init_seed"] == "varies_with_seed"
    assert study_config["config_source"] == "phase2g1_boundary_check"
    assert study_config["selected_by_metric_source"] == "val_metric"
    assert study_config["final_report_metric_source"] == "test_metric"
    assert study_config["tuned_pc_source"] == "phase2g1_boundary_check"
    assert study_config["tuned_pc_preset_name"] == "cfg_pc_refined"
    assert study_config["tuned_pc_training"] == {
        "epochs": 120,
        "eta_x": 0.2,
        "eta_w": 0.4,
        "eta_b": 0.4,
        "train_steps": 10,
        "eval_steps": 10,
        "state_init": "forward",
    }
    assert study_config["tuned_pc_selection_config_id"] == "cfg_pc_refined"
    assert study_config["mlp_source"] == "phase2g1_boundary_check"
    assert study_config["mlp_preset_name"] == "cfg_mlp_refined"
    assert study_config["mlp_training"] == {
        "epochs": 160,
        "eta_w": 0.2,
        "eta_b": 0.2,
    }

    assert len(seed_rows) == 2
    assert seed_rows[0]["run_seed"] == "0"
    assert seed_rows[0]["model_init_seed"] == "0"
    assert seed_rows[0]["data_seed"] == "0"
    assert seed_rows[0]["default_pc_status"] == "ok"
    assert seed_rows[0]["tuned_pc_status"] == "ok"
    assert seed_rows[0]["mlp_status"] == "ok"
    assert seed_rows[0]["tuned_pc_beats_default_pc"] in {"True", "False"}
    assert seed_rows[0]["tuned_pc_beats_mlp"] in {"True", "False"}

    assert aggregate_summary["planned_seed_count"] == 2
    assert aggregate_summary["config_source"] == "phase2g1_boundary_check"
    assert aggregate_summary["selected_by_metric_source"] == "val_metric"
    assert aggregate_summary["final_report_metric_source"] == "test_metric"
    assert aggregate_summary["selection_split"] == "validation"
    assert aggregate_summary["final_report_split"] == "test"
    assert aggregate_summary["tuned_pc_source"] == "phase2g1_boundary_check"
    assert aggregate_summary["tuned_pc_preset_name"] == "cfg_pc_refined"
    assert aggregate_summary["selected_pc_config"] == {
        "epochs": 120,
        "eta_x": 0.2,
        "eta_w": 0.4,
        "eta_b": 0.4,
        "train_steps": 10,
        "eval_steps": 10,
        "state_init": "forward",
    }
    assert aggregate_summary["mlp_source"] == "phase2g1_boundary_check"
    assert aggregate_summary["mlp_preset_name"] == "cfg_mlp_refined"
    assert aggregate_summary["selected_mlp_config"] == {
        "epochs": 160,
        "eta_w": 0.2,
        "eta_b": 0.2,
    }
    assert aggregate_summary["headline_test_comparison_target"] == "selected_pc_vs_selected_mlp"
    assert aggregate_summary["headline_test_comparison_split"] == "test"
    assert aggregate_summary["headline_test_winner"] in {"pc", "mlp", "tie"}
    assert aggregate_summary["headline_test_pc_beats_mlp"] in {True, False}
    assert aggregate_summary["tie_rtol"] == 1.0e-12
    assert aggregate_summary["tie_atol"] == 1.0e-12
