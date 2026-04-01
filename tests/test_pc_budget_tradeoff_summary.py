from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from pc.pc_budget_tradeoff import run_pc_budget_tradeoff_study


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _std(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=0))


def _metric_improvement_amount(old_value: float, new_value: float) -> float:
    return old_value - new_value


def test_pc_budget_tradeoff_summary_matches_budget_aggregates(tmp_path: Path) -> None:
    best_config_dir = tmp_path / "phase2g1_boundary_check" / "toy_sine_regression"
    best_config_dir.mkdir(parents=True, exist_ok=True)
    with (best_config_dir / "best_pc_config_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_id": "phase2g1_fixture",
                "boundary_check_best_config_id": "cfg_pc_refined",
                "boundary_check_val_metric": 0.02,
                "boundary_check_test_metric": 0.021,
                "boundary_check_best_config": {
                    "eta_x": 0.15,
                    "eta_w": 0.2,
                    "eta_b": 0.2,
                    "train_steps": 12,
                    "eval_steps": 12,
                    "epochs": 160,
                    "state_init": "forward",
                },
            },
            handle,
            indent=2,
        )
    with (best_config_dir / "best_mlp_config_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_id": "phase2g1_fixture",
                "boundary_check_best_config_id": "cfg_mlp_refined",
                "boundary_check_val_metric": 0.03,
                "boundary_check_test_metric": 0.031,
                "boundary_check_best_config": {
                    "eta_w": 0.2,
                    "eta_b": 0.2,
                    "epochs": 160,
                },
            },
            handle,
            indent=2,
        )
    run_pc_budget_tradeoff_study(
        "toy_sine_regression",
        output_root=tmp_path,
        run_id="toy_sine_budget_tradeoff_summary",
        plot_energy=False,
        plot_summary=False,
        seed_values=[3, 4],
        joint_search_output_root=tmp_path,
    )

    run_dir = tmp_path / "pc_budget_tradeoff_phase2g1_toy_sine_regression"
    seed_rows = _load_csv(run_dir / "seed_budget_records.csv")
    budget_rows = _load_csv(run_dir / "budget_summary.csv")
    aggregate_summary = _load_json(run_dir / "aggregate_summary.json")

    row_by_variant = {row["variant"]: row for row in budget_rows}
    means_by_variant = {
        variant: float(row["primary_metric_mean"])
        for variant, row in row_by_variant.items()
        if row["primary_metric_mean"] != ""
    }

    best_variant = min(
        ("tuned_pc_1x", "tuned_pc_2x", "tuned_pc_4x"),
        key=lambda variant: means_by_variant[variant],
    )
    assert aggregate_summary["best_budget_variant"] == best_variant
    assert aggregate_summary["best_budget_primary_metric_mean"] == means_by_variant[best_variant]

    gap_1x = float(row_by_variant["tuned_pc_1x"]["primary_metric_delta_tuned_pc_minus_mlp_mean"])
    gap_2x = float(row_by_variant["tuned_pc_2x"]["primary_metric_delta_tuned_pc_minus_mlp_mean"])
    gap_4x = float(row_by_variant["tuned_pc_4x"]["primary_metric_delta_tuned_pc_minus_mlp_mean"])
    assert aggregate_summary["gap_to_mlp_at_1x_mean"] == gap_1x
    assert aggregate_summary["gap_to_mlp_at_2x_mean"] == gap_2x
    assert aggregate_summary["gap_to_mlp_at_4x_mean"] == gap_4x
    assert aggregate_summary["gap_to_mlp_shrinks_from_1x_to_2x"] is (abs(gap_2x) < abs(gap_1x))
    assert aggregate_summary["gap_to_mlp_shrinks_from_2x_to_4x"] is (abs(gap_4x) < abs(gap_2x))
    assert aggregate_summary["selected_by_metric_source"] == "val_metric"
    assert aggregate_summary["final_report_metric_source"] == "test_metric"
    assert aggregate_summary["selection_split"] == "validation"
    assert aggregate_summary["final_report_split"] == "test"
    assert aggregate_summary["config_source"] == "phase2g1_boundary_check"
    assert aggregate_summary["tuned_pc_source"] == "phase2g1_boundary_check"
    assert aggregate_summary["tuned_pc_preset_name"] == "cfg_pc_refined"
    assert aggregate_summary["selected_pc_base_config"] == {
        "epochs": 160,
        "eta_x": 0.15,
        "eta_w": 0.2,
        "eta_b": 0.2,
        "train_steps": 12,
        "eval_steps": 12,
        "state_init": "forward",
    }
    assert aggregate_summary["mlp_source"] == "phase2g1_boundary_check"
    assert aggregate_summary["mlp_preset_name"] == "cfg_mlp_refined"
    assert aggregate_summary["selected_mlp_config"] == {
        "epochs": 160,
        "eta_w": 0.2,
        "eta_b": 0.2,
    }
    mlp_mean = float(row_by_variant["mlp"]["primary_metric_mean"])
    expected_winner = "pc" if means_by_variant["tuned_pc_1x"] < mlp_mean else "mlp"
    assert aggregate_summary["headline_test_comparison_target"] == "selected_pc_1x_vs_selected_mlp"
    assert aggregate_summary["headline_test_comparison_split"] == "test"
    assert aggregate_summary["headline_test_winner"] == expected_winner
    assert aggregate_summary["headline_test_pc_metric_mean"] == means_by_variant["tuned_pc_1x"]
    assert aggregate_summary["headline_test_mlp_metric_mean"] == mlp_mean
    assert aggregate_summary["headline_test_metric_difference_mlp_minus_pc"] == (
        mlp_mean - means_by_variant["tuned_pc_1x"]
    )
    assert aggregate_summary["headline_test_pc_beats_mlp"] is (expected_winner == "pc")

    improve_12 = _metric_improvement_amount(
        means_by_variant["tuned_pc_1x"],
        means_by_variant["tuned_pc_2x"],
    )
    improve_24 = _metric_improvement_amount(
        means_by_variant["tuned_pc_2x"],
        means_by_variant["tuned_pc_4x"],
    )
    expected_diminishing = improve_12 > 0.0 and (
        not (means_by_variant["tuned_pc_4x"] < means_by_variant["tuned_pc_2x"])
        or (improve_12 > improve_24)
    )
    assert aggregate_summary["evidence_of_diminishing_returns"] is expected_diminishing

    budget2x_seed_deltas = [
        float(row["primary_metric_delta_tuned_pc_minus_mlp"])
        for row in seed_rows
        if row["variant"] == "tuned_pc_2x" and row["primary_metric_delta_tuned_pc_minus_mlp"] != ""
    ]
    assert float(row_by_variant["tuned_pc_2x"]["primary_metric_delta_tuned_pc_minus_mlp_mean"]) == _mean(
        budget2x_seed_deltas
    )
    assert float(row_by_variant["tuned_pc_2x"]["primary_metric_delta_tuned_pc_minus_mlp_std"]) == _std(
        budget2x_seed_deltas
    )
