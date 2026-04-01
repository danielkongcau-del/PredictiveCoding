from __future__ import annotations

from pc.pc_joint_search import _build_aggregate_summary, rank_joint_search_rows


def test_joint_search_ranks_by_val_metric_not_train_or_test_metric() -> None:
    rows = [
        {
            "config_id": "cfg_001",
            "eta_x": 0.1,
            "eta_w": 0.1,
            "eta_b": 0.1,
            "train_steps": 25,
            "eval_steps": 25,
            "epochs": 60,
            "state_init": "forward",
            "status": "ok",
            "failure_reason": "",
            "metric_name": "mse",
            "metric_higher_is_better": False,
            "train_metric": 0.05,
            "val_metric": 0.20,
            "test_metric": 0.08,
            "val_metric_rank": None,
            "val_metric_delta_vs_best": None,
            "train_baseline_metric": 0.25,
            "val_baseline_metric": 0.30,
            "test_baseline_metric": 0.30,
            "beats_val_baseline": True,
            "best_epoch": 10,
            "final_pre_update_energy": 0.40,
            "final_post_update_energy": 0.38,
            "summary_path": "trials/cfg_001/summary.json",
        },
        {
            "config_id": "cfg_002",
            "eta_x": 0.2,
            "eta_w": 0.2,
            "eta_b": 0.2,
            "train_steps": 50,
            "eval_steps": 50,
            "epochs": 60,
            "state_init": "forward",
            "status": "ok",
            "failure_reason": "",
            "metric_name": "mse",
            "metric_higher_is_better": False,
            "train_metric": 0.10,
            "val_metric": 0.10,
            "test_metric": 0.11,
            "val_metric_rank": None,
            "val_metric_delta_vs_best": None,
            "train_baseline_metric": 0.25,
            "val_baseline_metric": 0.30,
            "test_baseline_metric": 0.30,
            "beats_val_baseline": True,
            "best_epoch": 15,
            "final_pre_update_energy": 0.20,
            "final_post_update_energy": 0.19,
            "summary_path": "trials/cfg_002/summary.json",
        },
    ]
    ranked_rows = rank_joint_search_rows(rows, "mse")
    rank_mapping = {row["config_id"]: row["val_metric_rank"] for row in ranked_rows}
    assert rank_mapping["cfg_002"] == 1
    assert rank_mapping["cfg_001"] == 2

    summary = _build_aggregate_summary(
        benchmark_name="toy_regression",
        run_id="selection_fixture",
        task_name="regression",
        metric_name="mse",
        search_rows=ranked_rows,
        mlp_summary={"train_metric": 0.08, "val_metric": 0.12, "test_metric": 0.12},
    )
    assert summary["best_config_id"] == "cfg_002"
    assert summary["best_train_metric"] == 0.10
    assert summary["best_val_metric"] == 0.10
    assert summary["best_test_metric"] == 0.11
    assert summary["best_pc_beats_mlp_reference"] is True
    assert summary["selection_reason"] == "selected lowest val_mse across 2 successful configurations"
