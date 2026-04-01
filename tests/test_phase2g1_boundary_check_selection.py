from __future__ import annotations

from pc.phase2g1_boundary_check import _build_aggregate_summary, _rank_rows


def test_phase2g1_boundary_check_selects_by_validation_metric_not_test_metric() -> None:
    pc_rows = [
        {
            "config_id": "cfg_001",
            "trial_label": "phase2g_best",
            "model_family": "predictive_coding",
            "changed_fields": [],
            "uses_extended_boundary_values": False,
            "eta_x": 0.05,
            "eta_w": 0.4,
            "eta_b": 0.4,
            "train_steps": 25,
            "eval_steps": 25,
            "epochs": 240,
            "state_init": "forward",
            "status": "ok",
            "failure_reason": "",
            "metric_name": "mse",
            "metric_higher_is_better": False,
            "train_metric": 0.04,
            "val_metric": 0.10,
            "test_metric": 0.30,
            "selection_metric_source": "val_metric",
            "selection_metric_value": 0.10,
            "report_metric_source": "test_metric",
            "report_metric_value": 0.30,
            "selection_metric_rank": None,
            "selection_metric_delta_vs_best": None,
            "train_baseline_metric": 0.4,
            "val_baseline_metric": 0.5,
            "test_baseline_metric": 0.5,
            "beats_val_baseline": True,
            "best_epoch": 40,
            "final_pre_update_energy": 0.1,
            "final_post_update_energy": 0.09,
            "final_loss": None,
            "summary_path": "pc_trials/cfg_001/summary.json",
        },
        {
            "config_id": "cfg_002",
            "trial_label": "probe_epochs_480",
            "model_family": "predictive_coding",
            "changed_fields": ["epochs"],
            "uses_extended_boundary_values": True,
            "eta_x": 0.05,
            "eta_w": 0.4,
            "eta_b": 0.4,
            "train_steps": 25,
            "eval_steps": 25,
            "epochs": 480,
            "state_init": "forward",
            "status": "ok",
            "failure_reason": "",
            "metric_name": "mse",
            "metric_higher_is_better": False,
            "train_metric": 0.03,
            "val_metric": 0.20,
            "test_metric": 0.05,
            "selection_metric_source": "val_metric",
            "selection_metric_value": 0.20,
            "report_metric_source": "test_metric",
            "report_metric_value": 0.05,
            "selection_metric_rank": None,
            "selection_metric_delta_vs_best": None,
            "train_baseline_metric": 0.4,
            "val_baseline_metric": 0.5,
            "test_baseline_metric": 0.5,
            "beats_val_baseline": True,
            "best_epoch": 60,
            "final_pre_update_energy": 0.08,
            "final_post_update_energy": 0.07,
            "final_loss": None,
            "summary_path": "pc_trials/cfg_002/summary.json",
        },
    ]
    mlp_rows = [
        {
            "config_id": "cfg_001",
            "trial_label": "phase2g_best",
            "model_family": "mlp",
            "changed_fields": [],
            "uses_extended_boundary_values": False,
            "eta_x": None,
            "eta_w": 0.2,
            "eta_b": 0.2,
            "train_steps": None,
            "eval_steps": None,
            "epochs": 320,
            "state_init": None,
            "status": "ok",
            "failure_reason": "",
            "metric_name": "mse",
            "metric_higher_is_better": False,
            "train_metric": 0.02,
            "val_metric": 0.09,
            "test_metric": 0.08,
            "selection_metric_source": "val_metric",
            "selection_metric_value": 0.09,
            "report_metric_source": "test_metric",
            "report_metric_value": 0.08,
            "selection_metric_rank": None,
            "selection_metric_delta_vs_best": None,
            "train_baseline_metric": 0.4,
            "val_baseline_metric": 0.5,
            "test_baseline_metric": 0.5,
            "beats_val_baseline": True,
            "best_epoch": 70,
            "final_pre_update_energy": None,
            "final_post_update_energy": None,
            "final_loss": 0.02,
            "summary_path": "mlp_trials/cfg_001/summary.json",
        }
    ]

    ranked_pc_rows = _rank_rows(pc_rows, "mse")
    ranked_mlp_rows = _rank_rows(mlp_rows, "mse")
    assert {row["config_id"]: row["selection_metric_rank"] for row in ranked_pc_rows} == {
        "cfg_001": 1,
        "cfg_002": 2,
    }

    summary = _build_aggregate_summary(
        benchmark_name="toy_regression",
        run_id="boundary_selection_fixture",
        task_name="regression",
        metric_name="mse",
        previous_reference_artifacts={
            "reference_root": "outputs/phase2g_matched_search/toy_regression",
            "aggregate_summary": {
                "run_id": "phase2g_fixture",
                "test_winner": "mlp",
                "test_winner_reason": "mlp had lower test mse",
            },
            "best_pc_summary": {
                "best_config_id": "cfg_pc",
                "best_config": {
                    "eta_x": 0.05,
                    "eta_w": 0.4,
                    "eta_b": 0.4,
                    "train_steps": 25,
                    "eval_steps": 25,
                    "epochs": 240,
                    "state_init": "forward",
                },
                "val_metric": 0.10,
                "test_metric": 0.11,
            },
            "best_mlp_summary": {
                "best_config_id": "cfg_mlp",
                "best_config": {"eta_w": 0.2, "eta_b": 0.2, "epochs": 320},
                "val_metric": 0.09,
                "test_metric": 0.08,
            },
        },
        pc_search_rows=ranked_pc_rows,
        mlp_search_rows=ranked_mlp_rows,
        pc_boundary_report={
            "boundary_dimensions": {"epochs": {"lower_edge_value": 60, "upper_edge_value": 240}},
            "probed_dimensions": ["epochs"],
            "unprobed_boundary_dimensions": [],
        },
        mlp_boundary_report={
            "boundary_dimensions": {},
            "probed_dimensions": [],
            "unprobed_boundary_dimensions": [],
        },
    )

    assert summary["boundary_check_best_pc_config_id"] == "cfg_001"
    assert summary["boundary_check_best_pc_val_metric"] == 0.10
    assert summary["boundary_check_best_pc_test_metric"] == 0.30
    assert summary["boundary_check_best_mlp_config_id"] == "cfg_001"
    assert summary["boundary_check_best_mlp_test_metric"] == 0.08
    assert summary["boundary_check_test_winner"] == "mlp"
    assert summary["prior_conclusion_survived_boundary_check"] is True
    assert summary["headline_conclusion_changed"] is False
