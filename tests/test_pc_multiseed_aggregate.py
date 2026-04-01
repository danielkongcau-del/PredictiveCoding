from __future__ import annotations

import json

import pytest

from pc.benchmark_specs import MLPTrainingSpec, PCTrainingSpec, get_benchmark_spec
from pc.pc_multiseed import (
    MLPSelection,
    PHASE2_MLP_SOURCE_PHASE2G1,
    PHASE2_TUNED_SOURCE_PHASE2G1,
    TunedPCSelection,
    _build_aggregate_summary,
    _metric_value_is_better,
    resolve_mlp_selection,
    resolve_tuned_pc_selection,
)


def test_metric_value_is_better_respects_mse_direction() -> None:
    assert _metric_value_is_better("mse", 0.10, 0.20) is True
    assert _metric_value_is_better("mse", 0.20, 0.10) is False


def test_phase2g1_selection_loading_reads_pc_and_mlp_configs(tmp_path) -> None:
    best_config_dir = tmp_path / "phase2g1_boundary_check" / "toy_regression"
    best_config_dir.mkdir(parents=True, exist_ok=True)
    with (best_config_dir / "best_pc_config_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
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
            },
            handle,
            indent=2,
        )
    with (best_config_dir / "best_mlp_config_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_id": "phase2g1_fixture",
                "boundary_check_best_config_id": "cfg_mlp_refined",
                "boundary_check_val_metric": 0.02,
                "boundary_check_test_metric": 0.021,
                "boundary_check_best_config": {
                    "eta_w": 0.2,
                    "eta_b": 0.2,
                    "epochs": 160,
                },
            },
            handle,
            indent=2,
        )

    tuned_selection = resolve_tuned_pc_selection(
        "toy_regression",
        tuned_source=PHASE2_TUNED_SOURCE_PHASE2G1,
        joint_search_output_root=tmp_path,
    )
    mlp_selection = resolve_mlp_selection(
        "toy_regression",
        mlp_source=PHASE2_MLP_SOURCE_PHASE2G1,
        joint_search_output_root=tmp_path,
    )

    assert tuned_selection.selection_config_id == "cfg_pc_refined"
    assert tuned_selection.epochs == 120
    assert tuned_selection.selection_val_metric == pytest.approx(0.01)
    assert tuned_selection.selection_test_metric == pytest.approx(0.011)
    assert tuned_selection.pc_training.train_steps == 10
    assert mlp_selection.selection_config_id == "cfg_mlp_refined"
    assert mlp_selection.epochs == 160
    assert mlp_selection.selection_val_metric == pytest.approx(0.02)
    assert mlp_selection.selection_test_metric == pytest.approx(0.021)
    assert mlp_selection.mlp_training.eta_w == pytest.approx(0.2)


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
        tuned_selection=TunedPCSelection(
            source="phase2g1_boundary_check",
            name="cfg_pc",
            pc_training=PCTrainingSpec(
                eta_x=0.2,
                eta_w=0.4,
                eta_b=0.4,
                train_steps=10,
                eval_steps=10,
                state_init="forward",
            ),
            epochs=120,
            selection_artifact_path="outputs/phase2g1_boundary_check/toy_regression/best_pc_config_summary.json",
            selection_config_id="cfg_pc",
            selection_run_id="phase2g1_fixture",
            selection_val_metric=0.01,
            selection_test_metric=0.011,
        ),
        mlp_selection=MLPSelection(
            source="phase2g1_boundary_check",
            name="cfg_mlp",
            mlp_training=MLPTrainingSpec(
                eta_w=0.2,
                eta_b=0.2,
            ),
            epochs=160,
            selection_artifact_path="outputs/phase2g1_boundary_check/toy_regression/best_mlp_config_summary.json",
            selection_config_id="cfg_mlp",
            selection_run_id="phase2g1_fixture",
            selection_val_metric=0.02,
            selection_test_metric=0.021,
        ),
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
    assert summary["selected_by_metric_source"] == "val_metric"
    assert summary["final_report_metric_source"] == "test_metric"
    assert summary["selection_split"] == "validation"
    assert summary["final_report_split"] == "test"
    assert summary["config_source"] == "phase2g1_boundary_check"
    assert summary["tuned_pc_source"] == "phase2g1_boundary_check"
    assert summary["tuned_pc_preset_name"] == "cfg_pc"
    assert summary["selected_pc_config"] == {
        "epochs": 120,
        "eta_x": 0.2,
        "eta_w": 0.4,
        "eta_b": 0.4,
        "train_steps": 10,
        "eval_steps": 10,
        "state_init": "forward",
    }
    assert summary["mlp_source"] == "phase2g1_boundary_check"
    assert summary["mlp_preset_name"] == "cfg_mlp"
    assert summary["selected_mlp_config"] == {
        "epochs": 160,
        "eta_w": 0.2,
        "eta_b": 0.2,
    }
    assert summary["headline_test_comparison_target"] == "selected_pc_vs_selected_mlp"
    assert summary["headline_test_comparison_split"] == "test"
    assert summary["headline_test_winner"] == "mlp"
    assert summary["headline_test_pc_metric_mean"] == pytest.approx(0.18333333333333335)
    assert summary["headline_test_mlp_metric_mean"] == pytest.approx(0.11)
    assert summary["headline_test_metric_difference_mlp_minus_pc"] == pytest.approx(-0.07333333333333335)
    assert summary["headline_test_pc_beats_mlp"] is False
    assert summary["tie_rtol"] == 1.0e-12
    assert summary["tie_atol"] == 1.0e-12
