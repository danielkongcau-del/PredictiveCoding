from __future__ import annotations

import pytest

from pc.benchmark_specs import get_benchmark_spec
from pc.pc_multiseed import _build_aggregate_summary, _metric_value_is_better


def test_metric_value_is_better_respects_mse_direction() -> None:
    assert _metric_value_is_better("mse", 0.10, 0.20) is True
    assert _metric_value_is_better("mse", 0.20, 0.10) is False


def test_multiseed_aggregate_summary_counts_and_statistics() -> None:
    spec = get_benchmark_spec("toy_regression")
    seed_records = [
        {
            "run_seed": 0,
            "default_pc_status": "ok",
            "default_pc_primary_metric_value": 0.20,
            "tuned_pc_status": "ok",
            "tuned_pc_primary_metric_value": 0.15,
            "mlp_status": "ok",
            "mlp_primary_metric_value": 0.10,
            "primary_metric_delta_tuned_pc_minus_default_pc": -0.05,
            "primary_metric_delta_mlp_minus_tuned_pc": -0.05,
        },
        {
            "run_seed": 1,
            "default_pc_status": "ok",
            "default_pc_primary_metric_value": 0.18,
            "tuned_pc_status": "ok",
            "tuned_pc_primary_metric_value": 0.18,
            "mlp_status": "ok",
            "mlp_primary_metric_value": 0.12,
            "primary_metric_delta_tuned_pc_minus_default_pc": 0.0,
            "primary_metric_delta_mlp_minus_tuned_pc": -0.06,
        },
        {
            "run_seed": 2,
            "default_pc_status": "ok",
            "default_pc_primary_metric_value": 0.16,
            "tuned_pc_status": "ok",
            "tuned_pc_primary_metric_value": 0.22,
            "mlp_status": "failed",
            "mlp_primary_metric_value": None,
            "primary_metric_delta_tuned_pc_minus_default_pc": 0.06,
            "primary_metric_delta_mlp_minus_tuned_pc": None,
        },
    ]

    summary = _build_aggregate_summary(
        base_spec=spec,
        run_id="aggregate_fixture",
        seed_values=[0, 1, 2],
        seed_records=seed_records,
    )

    assert summary["planned_seed_count"] == 3
    assert summary["default_pc_success_count"] == 3
    assert summary["tuned_pc_success_count"] == 3
    assert summary["mlp_success_count"] == 2
    assert summary["paired_default_vs_tuned_count"] == 3
    assert summary["paired_tuned_vs_mlp_count"] == 2
    assert summary["default_pc_primary_metric_mean"] == pytest.approx(0.18)
    assert summary["tuned_pc_primary_metric_mean"] == pytest.approx(0.18333333333333335)
    assert summary["mlp_primary_metric_mean"] == pytest.approx(0.11)
    assert summary["primary_metric_delta_tuned_pc_minus_default_pc_mean"] == pytest.approx(
        0.003333333333333334
    )
    assert summary["primary_metric_delta_mlp_minus_tuned_pc_mean"] == pytest.approx(-0.055)
    assert summary["tuned_pc_better_than_default_pc_seed_count"] == 1
    assert summary["default_pc_better_than_tuned_pc_seed_count"] == 1
    assert summary["tuned_pc_vs_default_pc_tie_seed_count"] == 1
    assert summary["tuned_pc_better_than_mlp_seed_count"] == 0
    assert summary["mlp_better_than_tuned_pc_seed_count"] == 2
    assert summary["tuned_pc_vs_mlp_tie_seed_count"] == 0
    assert summary["tuned_pc_mean_beats_default_pc"] is False
    assert summary["tuned_pc_mean_beats_mlp"] is False
    assert summary["tie_rtol"] == 1.0e-12
    assert summary["tie_atol"] == 1.0e-12
