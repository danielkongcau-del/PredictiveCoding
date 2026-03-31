from __future__ import annotations

from pc.pc_sensitivity import _build_aggregate_summary, metric_value_is_better, select_sensitivity_winner


def test_select_sensitivity_winner_respects_metric_direction() -> None:
    winner, reason = select_sensitivity_winner(
        "mse",
        "best_pc",
        "best_pc_primary_metric_value",
        0.10,
        "default",
        "default_pc_primary_metric_value",
        0.20,
    )
    assert winner == "best_pc"
    assert reason == "lower_is_better: best_pc_primary_metric_value < default_pc_primary_metric_value"

    winner, reason = select_sensitivity_winner(
        "accuracy",
        "best_pc",
        "best_pc_primary_metric_value",
        0.90,
        "mlp_reference",
        "mlp_reference_primary_metric_value",
        0.85,
    )
    assert winner == "best_pc"
    assert reason == "higher_is_better: best_pc_primary_metric_value > mlp_reference_primary_metric_value"


def test_metric_value_is_better_uses_strict_directional_comparisons() -> None:
    assert metric_value_is_better("mse", 0.10, 0.20) is True
    assert metric_value_is_better("mse", 0.20, 0.10) is False
    assert metric_value_is_better("accuracy", 0.90, 0.80) is True
    assert metric_value_is_better("accuracy", 0.80, 0.90) is False


def test_aggregate_summary_counts_failed_trials_and_tracks_booleans() -> None:
    trial_rows = [
        {
            "trial_id": "default",
            "parameter_group": "default",
            "status": "ok",
            "failure_reason": "",
            "primary_metric_name": "mse",
            "primary_metric_value": 0.20,
            "primary_metric_delta_vs_default": 0.0,
            "baseline_metric_name": "baseline_mse",
            "baseline_metric_value": 0.30,
            "beats_task_baseline": True,
            "final_pre_update_energy": 0.25,
            "final_pre_update_energy_delta_vs_default": 0.0,
            "final_post_update_energy": 0.24,
            "best_epoch": 5,
            "trial_summary_path": "trials/default/summary.json",
        },
        {
            "trial_id": "eta_x_double",
            "parameter_group": "eta_x",
            "status": "ok",
            "failure_reason": "",
            "primary_metric_name": "mse",
            "primary_metric_value": 0.10,
            "primary_metric_delta_vs_default": -0.10,
            "baseline_metric_name": "baseline_mse",
            "baseline_metric_value": 0.30,
            "beats_task_baseline": True,
            "final_pre_update_energy": 0.15,
            "final_pre_update_energy_delta_vs_default": -0.10,
            "final_post_update_energy": 0.14,
            "best_epoch": 7,
            "trial_summary_path": "trials/eta_x_double/summary.json",
        },
        {
            "trial_id": "state_init_zeros",
            "parameter_group": "state_init",
            "status": "failed",
            "failure_reason": "FloatingPointError: trial diverged",
            "primary_metric_name": "mse",
            "primary_metric_value": None,
            "primary_metric_delta_vs_default": None,
            "baseline_metric_name": "baseline_mse",
            "baseline_metric_value": 0.30,
            "beats_task_baseline": None,
            "final_pre_update_energy": None,
            "final_pre_update_energy_delta_vs_default": None,
            "final_post_update_energy": None,
            "best_epoch": None,
            "trial_summary_path": "",
        },
    ]
    mlp_reference_summary = {
        "primary_metric_value": 0.12,
    }

    summary = _build_aggregate_summary(
        benchmark_name="toy_regression",
        run_id="aggregate_fixture",
        task_name="regression",
        metric_name="mse",
        baseline_metric_name="baseline_mse",
        trial_rows=trial_rows,
        mlp_reference_summary=mlp_reference_summary,
    )

    assert summary["failed_trial_count"] == 1
    assert summary["failed_trial_ids"] == ["state_init_zeros"]
    assert summary["best_pc_trial_id"] == "eta_x_double"
    assert summary["best_pc_beats_task_baseline"] is True
    assert summary["best_pc_beats_mlp_reference"] is True
    assert summary["best_pc_vs_mlp_reference_winner"] == "best_pc"
    assert summary["parameter_group_summaries"]["eta_x"]["primary_metric_delta_vs_default"] == -0.10
    assert summary["parameter_group_summaries"]["state_init"]["best_trial_id"] is None
