from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from pc.pc_diagnostics import run_pc_diagnostics_study


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_float(value: str) -> float | None:
    if value == "":
        return None
    return float(value)


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _std(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=0))


def _correlation(rows: list[dict[str, str]], variant: str, energy_field: str) -> tuple[float | None, int]:
    filtered = [
        row
        for row in rows
        if row["variant"] == variant
        and row[energy_field] != ""
        and row["primary_metric_mean"] != ""
    ]
    x = np.asarray([float(row[energy_field]) for row in filtered], dtype=np.float64)
    y = np.asarray([float(row["primary_metric_mean"]) for row in filtered], dtype=np.float64)
    if x.shape[0] < 2:
        return None, int(x.shape[0])
    if np.isclose(np.std(x, ddof=0), 0.0) or np.isclose(np.std(y, ddof=0), 0.0):
        return None, int(x.shape[0])
    return float(np.corrcoef(x, y)[0, 1]), int(x.shape[0])


def test_pc_diagnostics_summary_matches_epoch_and_seed_aggregates(tmp_path: Path) -> None:
    run_pc_diagnostics_study(
        "toy_regression",
        output_root=tmp_path,
        run_id="toy_regression_diagnostics_summary",
        plot_energy=False,
        plot_summary=False,
        seed_values=[0, 1],
    )

    run_dir = tmp_path / "pc_diagnostics_toy_regression"
    seed_records = _load_csv(run_dir / "seed_records.csv")
    epoch_records = _load_csv(run_dir / "epoch_records.csv")
    epoch_summary = _load_csv(run_dir / "epoch_summary.csv")
    diagnostic_summary = _load_json(run_dir / "diagnostic_summary.json")

    tuned_final_minus_best = [
        float(row["tuned_pc_final_minus_best_metric"])
        for row in seed_records
        if row["tuned_pc_final_minus_best_metric"] != ""
    ]
    budget_values = [
        float(row["tuned_pc_budget2x_primary_metric_value"])
        for row in seed_records
        if row["tuned_pc_budget2x_primary_metric_value"] != ""
    ]
    tuned_values = [
        float(row["tuned_pc_primary_metric_value"])
        for row in seed_records
        if row["tuned_pc_primary_metric_value"] != ""
    ]
    budget_deltas = [
        float(row["primary_metric_delta_budget2x_minus_tuned_pc"])
        for row in seed_records
        if row["primary_metric_delta_budget2x_minus_tuned_pc"] != ""
    ]

    assert diagnostic_summary["tuned_pc_final_minus_best_metric_mean"] == _mean(
        tuned_final_minus_best
    )
    assert diagnostic_summary["tuned_pc_final_minus_best_metric_std"] == _std(
        tuned_final_minus_best
    )
    assert diagnostic_summary["budget2x_delta_vs_tuned_mean"] == _mean(budget_deltas)
    assert diagnostic_summary["budget2x_delta_vs_tuned_std"] == _std(budget_deltas)

    budget_mean = _mean(budget_values)
    tuned_mean = _mean(tuned_values)
    assert diagnostic_summary["budget2x_mean_beats_tuned"] is (budget_mean < tuned_mean)

    corr, count = _correlation(epoch_summary, "default_pc", "pre_update_energy_mean")
    assert diagnostic_summary["default_pc_pre_update_energy_metric_correlation_sample_count"] == count
    assert diagnostic_summary["default_pc_pre_update_energy_metric_correlation"] == corr

    tuned_corr, tuned_count = _correlation(epoch_summary, "tuned_pc", "post_update_energy_mean")
    assert diagnostic_summary["tuned_pc_post_update_energy_metric_correlation_sample_count"] == tuned_count
    assert diagnostic_summary["tuned_pc_post_update_energy_metric_correlation"] == tuned_corr

    default_epoch_count = sum(1 for row in epoch_summary if row["variant"] == "default_pc")
    raw_default_epoch_rows = sum(1 for row in epoch_records if row["variant"] == "default_pc")
    assert diagnostic_summary["default_pc_pre_update_energy_metric_correlation_sample_count"] == default_epoch_count
    assert raw_default_epoch_rows > default_epoch_count
    assert "final_minus_best_metric" in diagnostic_summary["notes"]
