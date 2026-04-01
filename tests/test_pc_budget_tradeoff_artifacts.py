from __future__ import annotations

import csv
import json
from pathlib import Path

from pc.pc_budget_tradeoff import PHASE2E_VARIANT_GROUPS, run_pc_budget_tradeoff_study


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


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_pc_budget_tradeoff_writes_expected_artifacts(tmp_path: Path) -> None:
    _write_phase2g1_best_config_summaries(tmp_path, "toy_regression")
    result = run_pc_budget_tradeoff_study(
        "toy_regression",
        output_root=tmp_path,
        run_id="toy_regression_budget_tradeoff",
        plot_energy=False,
        plot_summary=False,
        seed_values=[0, 1],
        joint_search_output_root=tmp_path,
    )

    run_dir = tmp_path / "pc_budget_tradeoff_phase2g1_toy_regression"
    assert result.run_dir == run_dir
    assert (run_dir / "study_config.json").exists()
    assert (run_dir / "seed_budget_records.csv").exists()
    assert (run_dir / "budget_summary.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()

    for seed in (0, 1):
        seed_dir = run_dir / "seeds" / f"seed_{seed:04d}"
        assert (seed_dir / "tuned_pc_1x" / "summary.json").exists()
        assert (seed_dir / "tuned_pc_2x" / "summary.json").exists()
        assert (seed_dir / "tuned_pc_4x" / "summary.json").exists()
        assert (seed_dir / "mlp" / "summary.json").exists()

    study_config = _load_json(run_dir / "study_config.json")
    seed_rows = _load_csv(run_dir / "seed_budget_records.csv")
    budget_rows = _load_csv(run_dir / "budget_summary.csv")

    assert study_config["variant_groups"] == PHASE2E_VARIANT_GROUPS
    assert study_config["config_source"] == "phase2g1_boundary_check"
    assert study_config["selected_by_metric_source"] == "val_metric"
    assert study_config["final_report_metric_source"] == "test_metric"
    assert study_config["tuned_pc_source"] == "phase2g1_boundary_check"
    assert study_config["tuned_pc_preset_name"] == "cfg_pc_refined"
    assert study_config["budget_levels"] == {
        "1x": {"budget_multiplier": "1x", "train_steps": 10, "eval_steps": 10},
        "2x": {"budget_multiplier": "2x", "train_steps": 20, "eval_steps": 20},
        "4x": {"budget_multiplier": "4x", "train_steps": 40, "eval_steps": 40},
    }
    assert study_config["budget_definition"] == "budget = inference step count, not wall-clock or FLOP matching"
    assert study_config["mlp_source"] == "phase2g1_boundary_check"
    assert study_config["mlp_preset_name"] == "cfg_mlp_refined"
    assert study_config["mlp_training"] == {
        "epochs": 160,
        "eta_w": 0.2,
        "eta_b": 0.2,
    }

    mlp_seed_rows = [row for row in seed_rows if row["variant"] == "mlp"]
    assert len(mlp_seed_rows) == 2
    assert {row["is_reference_variant"] for row in mlp_seed_rows} == {"True"}
    assert {row["budget_multiplier"] for row in mlp_seed_rows} == {"reference"}

    mlp_budget_row = next(row for row in budget_rows if row["variant"] == "mlp")
    assert mlp_budget_row["budget_multiplier"] == "reference"
    assert mlp_budget_row["train_steps"] == ""

    aggregate_summary = _load_json(run_dir / "aggregate_summary.json")
    assert aggregate_summary["config_source"] == "phase2g1_boundary_check"
    assert aggregate_summary["selection_split"] == "validation"
    assert aggregate_summary["final_report_split"] == "test"
    assert aggregate_summary["selected_pc_base_config"] == {
        "epochs": 120,
        "eta_x": 0.2,
        "eta_w": 0.4,
        "eta_b": 0.4,
        "train_steps": 10,
        "eval_steps": 10,
        "state_init": "forward",
    }
    assert aggregate_summary["selected_mlp_config"] == {
        "epochs": 160,
        "eta_w": 0.2,
        "eta_b": 0.2,
    }
    assert aggregate_summary["headline_test_comparison_target"] == "selected_pc_1x_vs_selected_mlp"
    assert aggregate_summary["headline_test_comparison_split"] == "test"
    assert aggregate_summary["headline_test_winner"] in {"pc", "mlp", "tie"}
