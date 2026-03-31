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
    run_pc_budget_tradeoff_study(
        "toy_sine_regression",
        output_root=tmp_path,
        run_id="toy_sine_budget_tradeoff_summary",
        plot_energy=False,
        plot_summary=False,
        seed_values=[3, 4],
    )

    run_dir = tmp_path / "pc_budget_tradeoff_toy_sine_regression"
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
