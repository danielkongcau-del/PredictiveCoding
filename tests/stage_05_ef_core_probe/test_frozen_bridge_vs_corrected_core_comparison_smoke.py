from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_stage05_contextual_summary(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "by_method": {
            "stage_05_two_branch_corrected_residual_core_v2_budget_push": {
                "configured_transport_steps": 2,
                "one_step_energy_delta_vs_identity": {"mean": -0.008, "std": 0.0},
                "configured_step_energy_delta_vs_identity": {"mean": -0.0075, "std": 0.0},
                "configured_step_fixed_point_residual_delta_vs_identity": {
                    "mean": -3.0e-05,
                    "std": 0.0,
                },
                "one_step_energy_delta_vs_local_field_only": {"mean": -0.0011, "std": 0.0},
                "configured_step_energy_delta_vs_local_field_only": {
                    "mean": -0.00105,
                    "std": 0.0,
                },
                "configured_step_fixed_point_residual_delta_vs_local_field_only": {
                    "mean": -3.5e-06,
                    "std": 0.0,
                },
                "val_accuracy": {"mean": 0.9, "std": 0.0},
                "test_accuracy": {"mean": 0.9, "std": 0.0},
                "val_output_mse": {"mean": 0.04, "std": 0.0},
                "test_output_mse": {"mean": 0.04, "std": 0.0},
                "selected_epoch": {"mean": 6.0, "std": 0.0},
                "selection_hits_final_training_boundary_rate": 1.0,
                "runtime_proxy_seconds": {"mean": 1.0, "std": 0.0},
            }
        }
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_run():
    module = runpy.run_path(
        str(
            ROOT
            / "experiments"
            / "stage_05_ef_core_probe"
            / "frozen_bridge_vs_corrected_core_comparison.py"
        )
    )
    return module["run"]


def test_frozen_bridge_vs_corrected_core_comparison_writes_expected_artifacts(
    tmp_path: Path,
) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="frozen_bridge_vs_corrected_core_smoke",
        seeds=(0,),
        stage04_epochs=2,
        stage04_eval_steps=5,
        stage04_layer_dims=(64, 16, 10),
        stage05_epochs=3,
        stage05_eval_steps=5,
        stage05_layer_dims=(64, 16, 10),
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()
    assert (run_dir / "comparison_report.json").exists()
    assert (run_dir / "comparison_report.md").exists()

    rows = _read_csv(run_dir / "aggregate_runs.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")
    report = _read_json(run_dir / "comparison_report.json")

    assert len(rows) == 2
    method_names = {row["method_name"] for row in rows}
    assert method_names == {
        "stage_04_frozen_bridge",
        "stage_05_corrected_residual_core",
    }

    first_row = rows[0]
    assert "one_step_energy_delta_vs_identity" in first_row
    assert "configured_step_energy_delta_vs_identity" in first_row
    assert "configured_step_fixed_point_residual_delta_vs_identity" in first_row
    assert "configured_step_energy_delta_vs_local_field_only" in first_row
    assert "val_accuracy" in first_row
    assert "test_accuracy" in first_row
    assert "deterministic_artifact_checks_passed" in first_row

    assert "comparison_protocol" in summary
    assert "by_method" in summary
    assert "pairwise_stage05_vs_stage04" in summary
    assert "stage05_corrected_residual_core_justifies_v2_charter" in summary
    assert "stage05_v2_charter_decision_detail" in summary
    assert summary["comparison_report_json_path"] == "comparison_report.json"
    assert summary["comparison_report_md_path"] == "comparison_report.md"

    assert "decision" in report
    assert "supports" in report
    assert "does_not_support" in report
    assert "stage05_corrected_residual_core_justifies_v2_charter" in report["decision"]


def test_stage05_v1_vs_v2_comparison_writes_expected_artifacts(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="corrected_core_v1_vs_v2_smoke",
        comparison_variant="stage05_v1_vs_v2",
        seeds=(0,),
        stage05_epochs=3,
        stage05_eval_steps=5,
        stage05_layer_dims=(64, 16, 10),
        stage05_transport_steps=2,
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()
    assert (run_dir / "comparison_report.json").exists()
    assert (run_dir / "comparison_report.md").exists()

    rows = _read_csv(run_dir / "aggregate_runs.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")
    report = _read_json(run_dir / "comparison_report.json")

    assert len(rows) == 2
    method_names = {row["method_name"] for row in rows}
    assert method_names == {
        "stage_05_corrected_residual_core_v1",
        "stage_05_two_branch_corrected_residual_core_v2",
    }

    first_row = rows[0]
    assert "transport_family" in first_row
    assert "residual_branch_structure" in first_row
    assert "one_step_energy_delta_vs_identity" in first_row
    assert "configured_step_energy_delta_vs_identity" in first_row
    assert "configured_step_fixed_point_residual_delta_vs_identity" in first_row
    assert "deterministic_artifact_checks_passed" in first_row

    assert summary["stage"] == "corrected_residual_core_v1_vs_v2_comparison"
    assert "comparison_protocol" in summary
    assert "by_method" in summary
    assert "pairwise_stage05_v2_vs_v1" in summary
    assert "stage05_v2_improves_mechanism_magnitude_over_v1" in summary
    assert "stage05_v2_vs_v1_decision_detail" in summary
    assert summary["comparison_report_json_path"] == "comparison_report.json"
    assert summary["comparison_report_md_path"] == "comparison_report.md"

    assert "decision" in report
    assert "supports" in report
    assert "does_not_support" in report
    assert "stage05_v2_improves_mechanism_magnitude_over_v1" in report["decision"]


def test_stage05_v2_vs_v3a_comparison_writes_expected_artifacts(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="stage05_v2_vs_v3a_smoke",
        comparison_variant="stage05_v2_vs_v3a",
        seeds=(0,),
        stage05_epochs=3,
        stage05_eval_steps=5,
        stage05_layer_dims=(64, 16, 10),
        stage05_transport_steps=2,
        lambda_drift=1.0,
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()
    assert (run_dir / "comparison_report.json").exists()
    assert (run_dir / "comparison_report.md").exists()

    rows = _read_csv(run_dir / "aggregate_runs.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")
    report = _read_json(run_dir / "comparison_report.json")

    assert len(rows) == 2
    method_names = {row["method_name"] for row in rows}
    assert method_names == {
        "stage_05_two_branch_corrected_residual_core_v2",
        "stage05_v3a_explicit_transport_drift_contract",
    }

    assert summary["stage"] == "stage05_v2_vs_v3a_explicit_transport_drift_comparison"
    assert summary["comparison_scope"] == "smoke_only"
    assert "comparison_protocol" in summary
    assert "by_method" in summary
    assert "pairwise_stage05_v3a_vs_v2" in summary
    assert "pairwise_deltas_vs_stage05_v2_reference" in summary
    assert "contextual_3072_reference" in summary
    assert "stage05_v3a_shows_positive_gap_closure_signal_vs_v2" in summary
    assert "stage05_v3a_materially_improves_configured_step_mechanism" in summary
    assert "stage05_v3a_avoids_obvious_report_accuracy_regression" in summary
    assert "recommended_next_move" in summary
    assert summary["comparison_report_json_path"] == "comparison_report.json"
    assert summary["comparison_report_md_path"] == "comparison_report.md"

    assert "decision" in report
    assert "pairwise_deltas_vs_stage05_v2_reference" in report
    assert "contextual_3072_reference" in report
    assert "supports" in report
    assert "does_not_support" in report
    assert "stage05_v3a_shows_positive_gap_closure_signal_vs_v2" in report["decision"]
    assert "recommended_next_move" in report["decision"]


def test_stage05_v2_v3a_v3b_comparison_writes_expected_artifacts(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="stage05_v2_v3a_v3b_smoke",
        comparison_variant="stage05_v2_v3a_v3b",
        seeds=(0,),
        stage05_epochs=3,
        stage05_eval_steps=5,
        stage05_layer_dims=(64, 16, 10),
        stage05_transport_steps=2,
        lambda_drift=1.0,
        lambda_traj_curr=0.1,
        alpha_floor=0.5,
        alpha_warmup_epochs=1,
        alpha_ramp_epochs=1,
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()
    assert (run_dir / "comparison_report.json").exists()
    assert (run_dir / "comparison_report.md").exists()

    rows = _read_csv(run_dir / "aggregate_runs.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")
    report = _read_json(run_dir / "comparison_report.json")

    assert len(rows) == 3
    method_names = {row["method_name"] for row in rows}
    assert method_names == {
        "stage_05_two_branch_corrected_residual_core_v2",
        "stage05_v3a_explicit_transport_drift_contract",
        "stage05_v3b_trajectory_curriculum_contract",
    }

    assert summary["stage"] == "stage05_v2_v3a_v3b_trajectory_curriculum_comparison"
    assert summary["comparison_scope"] == "smoke_only"
    assert "comparison_protocol" in summary
    assert "by_method" in summary
    assert "pairwise_stage05_v3b_vs_v2" in summary
    assert "pairwise_stage05_v3b_vs_v3a" in summary
    assert "pairwise_deltas_vs_stage05_v2_reference" in summary
    assert "pairwise_deltas_vs_stage05_v3a_reference" in summary
    assert "stage05_v3b_keeps_one_step_mechanism_positive" in summary
    assert "stage05_v3b_materially_improves_configured_step_mechanism_vs_v3a" in summary
    assert "stage05_v3b_shows_positive_gap_closure_signal_vs_v3a" in summary
    assert "gap_closure_decision" in summary
    assert "recommended_next_move" in summary
    assert summary["comparison_report_json_path"] == "comparison_report.json"
    assert summary["comparison_report_md_path"] == "comparison_report.md"

    assert "decision" in report
    assert "pairwise_deltas_vs_stage05_v2_reference" in report
    assert "pairwise_deltas_vs_stage05_v3a_reference" in report
    assert "supports" in report
    assert "does_not_support" in report
    assert report["decision"]["recommended_next_move"] in {
        "run_fixed_budget_v2_vs_v3a_vs_v3b_comparison",
        "another_v3b_implementation_pass",
    }


def test_stage05_v2_v3a_v3b_fixed_budget_comparison_reuses_existing_reference_artifacts(
    tmp_path: Path,
) -> None:
    contextual_path = tmp_path / "contextual_3072_summary.json"
    _write_stage05_contextual_summary(contextual_path)

    reference_result = load_run()(
        output_root=tmp_path,
        run_id="stage05_v2_vs_v3a_reference",
        comparison_variant="stage05_v2_vs_v3a",
        comparison_scope="fixed_budget_comparison",
        seeds=(0,),
        stage05_epochs=3,
        stage05_eval_steps=5,
        stage05_layer_dims=(64, 16, 10),
        stage05_transport_steps=2,
        lambda_drift=1.0,
        reuse_stage05_v2_reference_artifacts=False,
        contextual_reference_summary_path=contextual_path,
        contextual_reference_stage05_epochs=6,
    )

    run_dir = load_run()(
        output_root=tmp_path,
        run_id="stage05_v2_v3a_v3b_fixed_budget",
        comparison_variant="stage05_v2_v3a_v3b",
        comparison_scope="fixed_budget_comparison",
        experiment_name="stage05_v2_v3a_v3b_fixed_budget_comparison",
        seeds=(0,),
        stage05_epochs=3,
        stage05_eval_steps=5,
        stage05_layer_dims=(64, 16, 10),
        stage05_transport_steps=2,
        lambda_drift=1.0,
        lambda_traj_curr=0.1,
        alpha_floor=0.5,
        alpha_warmup_epochs=1,
        alpha_ramp_epochs=1,
        reuse_stage05_v2_reference_artifacts=True,
        reference_artifact_root=(
            reference_result.run_dir
            / "runs"
            / "stage_05_two_branch_corrected_residual_core_v2"
        ),
        reuse_stage05_v3a_reference_artifacts=True,
        v3a_reference_artifact_root=(
            reference_result.run_dir
            / "runs"
            / "stage05_v3a_explicit_transport_drift_contract"
        ),
        contextual_reference_summary_path=contextual_path,
        contextual_reference_stage05_epochs=6,
    ).run_dir

    summary = _read_json(run_dir / "aggregate_summary.json")
    report = _read_json(run_dir / "comparison_report.json")

    assert summary["stage"] == "stage05_v2_v3a_v3b_fixed_budget_comparison"
    assert summary["comparison_scope"] == "fixed_budget_comparison"
    assert (
        summary["comparison_protocol"]["stage_05_v2_reference"]["reference_reused_from_existing_artifacts"]
        is True
    )
    assert (
        summary["comparison_protocol"]["stage_05_v3a_reference"]["reference_reused_from_existing_artifacts"]
        is True
    )
    assert "contextual_gap_closure_fractions_vs_3072_reference" in summary
    assert "stage05_v3b_materially_improves_configured_step_mechanism_vs_v3a" in summary
    assert "stage05_v3b_shows_positive_gap_closure_signal_vs_v3a" in summary
    assert summary["recommended_next_move"] in {
        "proceed_to_stage05_v3c_charter",
        "keep_v3b_and_refine_implementation",
        "retain_v3a_as_active_reference",
    }
    assert "contextual_gap_closure_fractions_vs_3072_reference" in report
    assert "supports" in report
    assert "does_not_support" in report


def test_stage05_v3b_refinement_diagnostic_reuses_existing_reference_artifacts(
    tmp_path: Path,
) -> None:
    contextual_path = tmp_path / "contextual_3072_summary.json"
    _write_stage05_contextual_summary(contextual_path)

    v3a_reference = load_run()(
        output_root=tmp_path,
        run_id="stage05_v3b_refinement_v3a_reference",
        comparison_variant="stage05_v2_vs_v3a",
        comparison_scope="fixed_budget_comparison",
        seeds=(0,),
        stage05_epochs=3,
        stage05_eval_steps=5,
        stage05_layer_dims=(64, 16, 10),
        stage05_transport_steps=2,
        lambda_drift=1.0,
        reuse_stage05_v2_reference_artifacts=False,
        contextual_reference_summary_path=contextual_path,
        contextual_reference_stage05_epochs=6,
    )

    v3b_control = load_run()(
        output_root=tmp_path,
        run_id="stage05_v3b_refinement_v3b_control",
        comparison_variant="stage05_v2_v3a_v3b",
        comparison_scope="fixed_budget_comparison",
        experiment_name="stage05_v3b_refinement_v3b_control_fixture",
        seeds=(0,),
        stage05_epochs=3,
        stage05_eval_steps=5,
        stage05_layer_dims=(64, 16, 10),
        stage05_transport_steps=2,
        lambda_drift=1.0,
        lambda_traj_curr=0.1,
        alpha_floor=0.5,
        alpha_warmup_epochs=1,
        alpha_ramp_epochs=1,
        reuse_stage05_v2_reference_artifacts=True,
        reference_artifact_root=(
            v3a_reference.run_dir
            / "runs"
            / "stage_05_two_branch_corrected_residual_core_v2"
        ),
        reuse_stage05_v3a_reference_artifacts=True,
        v3a_reference_artifact_root=(
            v3a_reference.run_dir
            / "runs"
            / "stage05_v3a_explicit_transport_drift_contract"
        ),
        contextual_reference_summary_path=contextual_path,
        contextual_reference_stage05_epochs=6,
    )

    result = load_run()(
        output_root=tmp_path,
        run_id="stage05_v3b_refinement_smoke",
        comparison_variant="stage05_v3b_refinement_diagnostic",
        comparison_scope="fixed_budget_comparison",
        experiment_name="stage05_v3b_refinement_diagnostic_smoke",
        seeds=(0,),
        stage05_epochs=3,
        stage05_eval_steps=5,
        stage05_layer_dims=(64, 16, 10),
        stage05_transport_steps=2,
        lambda_drift=1.0,
        reuse_stage05_v2_reference_artifacts=True,
        reference_artifact_root=(
            v3a_reference.run_dir
            / "runs"
            / "stage_05_two_branch_corrected_residual_core_v2"
        ),
        reuse_stage05_v3a_reference_artifacts=True,
        v3a_reference_artifact_root=(
            v3a_reference.run_dir
            / "runs"
            / "stage05_v3a_explicit_transport_drift_contract"
        ),
        reuse_stage05_v3b_control_artifacts=True,
        v3b_control_artifact_root=(
            v3b_control.run_dir
            / "runs"
            / "stage05_v3b_trajectory_curriculum_contract"
        ),
        control_lambda_traj_curr=0.1,
        control_alpha_floor=0.5,
        control_alpha_warmup_epochs=1,
        control_alpha_ramp_epochs=1,
        alpha_earlier_transition_alpha_floor=0.25,
        alpha_earlier_transition_alpha_warmup_epochs=0,
        alpha_earlier_transition_alpha_ramp_epochs=1,
        stronger_traj_curr_lambda_traj_curr=0.2,
        contextual_reference_summary_path=contextual_path,
        contextual_reference_stage05_epochs=6,
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()
    assert (run_dir / "comparison_report.json").exists()
    assert (run_dir / "comparison_report.md").exists()

    rows = _read_csv(run_dir / "aggregate_runs.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")
    report = _read_json(run_dir / "comparison_report.json")

    assert len(rows) == 5
    method_names = {row["method_name"] for row in rows}
    assert method_names == {
        "stage_05_two_branch_corrected_residual_core_v2",
        "stage05_v3a_explicit_transport_drift_contract",
        "stage05_v3b_trajectory_curriculum_contract",
        "stage05_v3b_alpha_earlier_transition",
        "stage05_v3b_stronger_traj_curr_weight",
    }

    first_row = rows[0]
    assert "candidate_name" in first_row
    assert "lambda_traj_curr" in first_row
    assert "alpha_floor" in first_row
    assert "alpha_warmup_epochs" in first_row
    assert "alpha_ramp_epochs" in first_row

    assert summary["stage"] == "stage05_v3b_refinement_diagnostic_smoke"
    assert summary["comparison_scope"] == "fixed_budget_comparison"
    assert "tested_variant_names" in summary
    assert "pairwise_deltas_vs_v3b_control" in summary
    assert "pairwise_deltas_vs_v3a_reference" in summary
    assert "configured_step_mechanism_ranking" in summary
    assert "best_variant_name" in summary
    assert "narrow_v3b_refinement_materially_beats_v3b_control" in summary
    assert "narrow_v3b_refinement_materially_beats_v3a_reference" in summary
    assert "recommended_next_move" in summary
    assert "decision_rationale" in summary

    assert "decision" in report
    assert "pairwise_deltas_vs_v3b_control" in report
    assert "pairwise_deltas_vs_v3a_reference" in report
    assert "supports" in report
    assert "does_not_support" in report
    assert report["decision"]["recommended_next_move"] in {
        "promote_refined_v3b_and_recompare",
        "retain_v3a_as_active_reference_and_stop_v3b",
        "retain_v3a_as_active_reference_but_keep_v3b_as_future_context",
    }


def test_stage05_v2_v3a_refined_v3b_recompare_uses_promoted_candidate_artifacts(
    tmp_path: Path,
) -> None:
    contextual_path = tmp_path / "contextual_3072_summary.json"
    _write_stage05_contextual_summary(contextual_path)

    v3a_reference = load_run()(
        output_root=tmp_path,
        run_id="stage05_refined_recompare_v3a_reference",
        comparison_variant="stage05_v2_vs_v3a",
        comparison_scope="fixed_budget_comparison",
        seeds=(0,),
        stage05_epochs=3,
        stage05_eval_steps=5,
        stage05_layer_dims=(64, 16, 10),
        stage05_transport_steps=2,
        lambda_drift=1.0,
        reuse_stage05_v2_reference_artifacts=False,
        contextual_reference_summary_path=contextual_path,
        contextual_reference_stage05_epochs=6,
    )

    refinement = load_run()(
        output_root=tmp_path,
        run_id="stage05_refined_recompare_refinement_fixture",
        comparison_variant="stage05_v3b_refinement_diagnostic",
        comparison_scope="fixed_budget_comparison",
        experiment_name="stage05_refined_recompare_refinement_fixture",
        seeds=(0,),
        stage05_epochs=3,
        stage05_eval_steps=5,
        stage05_layer_dims=(64, 16, 10),
        stage05_transport_steps=2,
        lambda_drift=1.0,
        reuse_stage05_v2_reference_artifacts=True,
        reference_artifact_root=(
            v3a_reference.run_dir
            / "runs"
            / "stage_05_two_branch_corrected_residual_core_v2"
        ),
        reuse_stage05_v3a_reference_artifacts=True,
        v3a_reference_artifact_root=(
            v3a_reference.run_dir
            / "runs"
            / "stage05_v3a_explicit_transport_drift_contract"
        ),
        contextual_reference_summary_path=contextual_path,
        contextual_reference_stage05_epochs=6,
    )

    result = load_run()(
        output_root=tmp_path,
        run_id="stage05_refined_recompare_smoke",
        comparison_variant="stage05_v2_v3a_refined_v3b_recompare",
        comparison_scope="fixed_budget_comparison",
        experiment_name="stage05_v2_v3a_refined_v3b_recompare_smoke",
        seeds=(0,),
        stage05_epochs=3,
        stage05_eval_steps=5,
        stage05_layer_dims=(64, 16, 10),
        stage05_transport_steps=2,
        lambda_drift=1.0,
        lambda_traj_curr=0.2,
        alpha_floor=0.5,
        alpha_warmup_epochs=3,
        alpha_ramp_epochs=3,
        reuse_stage05_v2_reference_artifacts=True,
        reference_artifact_root=(
            v3a_reference.run_dir
            / "runs"
            / "stage_05_two_branch_corrected_residual_core_v2"
        ),
        reuse_stage05_v3a_reference_artifacts=True,
        v3a_reference_artifact_root=(
            v3a_reference.run_dir
            / "runs"
            / "stage05_v3a_explicit_transport_drift_contract"
        ),
        reuse_stage05_v3b_candidate_artifacts=True,
        v3b_candidate_artifact_root=(
            refinement.run_dir
            / "runs"
            / "stage05_v3b_stronger_traj_curr_weight"
        ),
        v3b_candidate_method_name="stage05_v3b_stronger_traj_curr_weight",
        contextual_reference_summary_path=contextual_path,
        contextual_reference_stage05_epochs=6,
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()
    assert (run_dir / "comparison_report.json").exists()
    assert (run_dir / "comparison_report.md").exists()

    rows = _read_csv(run_dir / "aggregate_runs.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")
    report = _read_json(run_dir / "comparison_report.json")

    assert len(rows) == 3
    assert {row["method_name"] for row in rows} == {
        "stage_05_two_branch_corrected_residual_core_v2",
        "stage05_v3a_explicit_transport_drift_contract",
        "stage05_v3b_stronger_traj_curr_weight",
    }

    assert summary["stage"] == "stage05_v2_v3a_refined_v3b_recompare_smoke"
    assert summary["promoted_v3b_candidate_name"] == "stage05_v3b_stronger_traj_curr_weight"
    assert (
        summary["comparison_protocol"]["stage_05_v3b_candidate"]["method_name"]
        == "stage05_v3b_stronger_traj_curr_weight"
    )
    assert (
        summary["comparison_protocol"]["stage_05_v3b_candidate"]["reference_reused_from_existing_artifacts"]
        is True
    )
    assert "pairwise_promoted_refined_v3b_vs_v2" in summary
    assert "pairwise_promoted_refined_v3b_vs_v3a" in summary
    assert "promoted_refined_v3b_materially_beats_v3a" in summary
    assert "promoted_refined_v3b_avoids_obvious_report_accuracy_regression" in summary
    assert "promoted_refined_v3b_replaces_v3a_as_active_reference" in summary
    assert summary["recommended_next_move"] in {
        "promote_refined_v3b_as_active_reference",
        "retain_v3a_as_active_reference",
    }

    assert report["promoted_v3b_candidate_name"] == "stage05_v3b_stronger_traj_curr_weight"
    assert "pairwise_promoted_refined_v3b_vs_v2" in report
    assert "pairwise_promoted_refined_v3b_vs_v3a" in report
    assert report["decision"]["recommended_next_move"] in {
        "promote_refined_v3b_as_active_reference",
        "retain_v3a_as_active_reference",
    }


def test_stage05_v2_promoted_v3b_v3c_comparison_writes_expected_artifacts(
    tmp_path: Path,
) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="stage05_v2_promoted_v3b_v3c_smoke",
        comparison_variant="stage05_v2_promoted_v3b_v3c",
        comparison_scope="smoke_only",
        seeds=(0,),
        stage05_epochs=4,
        stage05_eval_steps=5,
        stage05_layer_dims=(64, 16, 10),
        stage05_transport_steps=2,
        lambda_sg=0.05,
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()
    assert (run_dir / "comparison_report.json").exists()
    assert (run_dir / "comparison_report.md").exists()

    rows = _read_csv(run_dir / "aggregate_runs.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")
    report = _read_json(run_dir / "comparison_report.json")

    assert len(rows) == 3
    assert {row["method_name"] for row in rows} == {
        "stage_05_two_branch_corrected_residual_core_v2",
        "stage05_v3b_stronger_traj_curr_weight",
        "stage05_v3c_endpoint_semigroup_consistency_contract",
    }

    assert summary["stage"] == "stage05_v2_promoted_v3b_v3c_comparison"
    assert summary["comparison_scope"] == "smoke_only"
    assert "comparison_protocol" in summary
    assert summary["comparison_protocol"]["stage_05_v3c_candidate"]["method_name"] == (
        "stage05_v3c_endpoint_semigroup_consistency_contract"
    )
    assert "pairwise_deltas_vs_stage05_v2_reference" in summary
    assert "pairwise_deltas_vs_promoted_refined_v3b_reference" in summary
    assert "stage05_v3c_materially_improves_configured_step_mechanism_vs_promoted_v3b" in summary
    assert "refined_v3c_formal_comparison_candidate_name" in summary
    assert "promoted_refined_v3c_materially_beats_promoted_v3b" in summary
    assert "promoted_refined_v3c_avoids_obvious_report_accuracy_regression" in summary
    assert "promoted_refined_v3c_replaces_promoted_v3b_as_active_reference" in summary
    assert summary["recommended_next_move"] == "run_fixed_budget_v2_vs_promoted_v3b_vs_v3c_comparison"
    assert (
        summary["comparison_protocol"]["stage_05_v3c_candidate"]["semigroup_target_mode"]
        == "single_sided_detached_split_endpoint"
    )
    assert (
        summary["comparison_protocol"]["stage_05_v3c_candidate"]["endpoint_semigroup_consistency_enabled"]
        is True
    )

    assert "pairwise_deltas_vs_promoted_refined_v3b_reference" in report
    assert report["decision"]["recommended_next_move"] == (
        "run_fixed_budget_v2_vs_promoted_v3b_vs_v3c_comparison"
    )


def test_stage05_v3c_refinement_diagnostic_writes_expected_artifacts(
    tmp_path: Path,
) -> None:
    v3c_fixture = load_run()(
        output_root=tmp_path,
        run_id="stage05_v3c_refinement_fixture",
        comparison_variant="stage05_v2_promoted_v3b_v3c",
        comparison_scope="smoke_only",
        experiment_name="stage05_v3c_refinement_fixture",
        seeds=(0,),
        stage05_epochs=4,
        stage05_eval_steps=5,
        stage05_layer_dims=(64, 16, 10),
        stage05_transport_steps=2,
        lambda_sg=0.05,
    )

    result = load_run()(
        output_root=tmp_path,
        run_id="stage05_v3c_refinement_smoke",
        comparison_variant="stage05_v3c_refinement_diagnostic",
        comparison_scope="smoke_only",
        experiment_name="stage05_v3c_refinement_diagnostic_smoke",
        seeds=(0,),
        stage05_epochs=4,
        stage05_eval_steps=5,
        stage05_layer_dims=(64, 16, 10),
        stage05_transport_steps=2,
        reuse_stage05_v2_reference_artifacts=True,
        reference_artifact_root=(
            v3c_fixture.run_dir
            / "runs"
            / "stage_05_two_branch_corrected_residual_core_v2"
        ),
        reuse_stage05_v3b_reference_artifacts=True,
        v3b_reference_artifact_root=(
            v3c_fixture.run_dir
            / "runs"
            / "stage05_v3b_stronger_traj_curr_weight"
        ),
        reuse_stage05_v3c_control_artifacts=True,
        v3c_control_artifact_root=(
            v3c_fixture.run_dir
            / "runs"
            / "stage05_v3c_endpoint_semigroup_consistency_contract"
        ),
        control_lambda_sg=0.05,
        stronger_semigroup_lambda_sg=0.10,
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()
    assert (run_dir / "comparison_report.json").exists()
    assert (run_dir / "comparison_report.md").exists()

    rows = _read_csv(run_dir / "aggregate_runs.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")
    report = _read_json(run_dir / "comparison_report.json")

    assert len(rows) == 4
    assert {row["method_name"] for row in rows} == {
        "stage_05_two_branch_corrected_residual_core_v2",
        "stage05_v3b_stronger_traj_curr_weight",
        "stage05_v3c_endpoint_semigroup_consistency_contract",
        "stage05_v3c_stronger_semigroup_weight",
    }

    assert summary["stage"] == "stage05_v3c_refinement_diagnostic_smoke"
    assert summary["comparison_scope"] == "smoke_only"
    assert summary["best_variant_name"] == "stage05_v3c_stronger_semigroup_weight"
    assert "v3c_variant_settings" in summary
    assert "pairwise_deltas_vs_v3c_control" in summary
    assert "pairwise_deltas_vs_promoted_refined_v3b_reference" in summary
    assert "pairwise_deltas_vs_stage05_v2_reference" in summary
    assert "narrow_v3c_refinement_materially_beats_v3c_control" in summary
    assert "narrow_v3c_refinement_materially_beats_promoted_v3b_reference" in summary
    assert summary["recommended_next_move"] == "run_real_fixed_budget_v3c_refinement_diagnostic"
    assert (
        summary["v3c_variant_settings"]["stage05_v3c_stronger_semigroup_weight"]["lambda_sg"]
        == 0.10
    )

    assert "pairwise_deltas_vs_v3c_control" in report
    assert "pairwise_deltas_vs_promoted_refined_v3b_reference" in report
    assert report["decision"]["recommended_next_move"] == (
        "run_real_fixed_budget_v3c_refinement_diagnostic"
    )


def test_stage05_v2_promoted_v3b_refined_v3c_recompare_reuses_existing_artifacts(
    tmp_path: Path,
) -> None:
    contextual_path = tmp_path / "contextual_3072_summary.json"
    _write_stage05_contextual_summary(contextual_path)

    v3c_fixture = load_run()(
        output_root=tmp_path,
        run_id="stage05_refined_v3c_recompare_fixture",
        comparison_variant="stage05_v2_promoted_v3b_v3c",
        comparison_scope="smoke_only",
        experiment_name="stage05_refined_v3c_recompare_fixture",
        seeds=(0,),
        stage05_epochs=4,
        stage05_eval_steps=5,
        stage05_layer_dims=(64, 16, 10),
        stage05_transport_steps=2,
        lambda_sg=0.05,
    )

    v3c_refinement = load_run()(
        output_root=tmp_path,
        run_id="stage05_refined_v3c_recompare_refinement",
        comparison_variant="stage05_v3c_refinement_diagnostic",
        comparison_scope="smoke_only",
        experiment_name="stage05_refined_v3c_recompare_refinement",
        seeds=(0,),
        stage05_epochs=4,
        stage05_eval_steps=5,
        stage05_layer_dims=(64, 16, 10),
        stage05_transport_steps=2,
        reuse_stage05_v2_reference_artifacts=True,
        reference_artifact_root=(
            v3c_fixture.run_dir
            / "runs"
            / "stage_05_two_branch_corrected_residual_core_v2"
        ),
        reuse_stage05_v3b_reference_artifacts=True,
        v3b_reference_artifact_root=(
            v3c_fixture.run_dir
            / "runs"
            / "stage05_v3b_stronger_traj_curr_weight"
        ),
        reuse_stage05_v3c_control_artifacts=True,
        v3c_control_artifact_root=(
            v3c_fixture.run_dir
            / "runs"
            / "stage05_v3c_endpoint_semigroup_consistency_contract"
        ),
        control_lambda_sg=0.05,
        stronger_semigroup_lambda_sg=0.10,
    )

    result = load_run()(
        output_root=tmp_path,
        run_id="stage05_refined_v3c_recompare_smoke",
        comparison_variant="stage05_v2_promoted_v3b_refined_v3c_recompare",
        comparison_scope="fixed_budget_comparison",
        experiment_name="stage05_v2_promoted_v3b_refined_v3c_recompare_smoke",
        seeds=(0,),
        stage05_epochs=4,
        stage05_eval_steps=5,
        stage05_layer_dims=(64, 16, 10),
        stage05_transport_steps=2,
        reuse_stage05_v2_reference_artifacts=True,
        reference_artifact_root=(
            v3c_fixture.run_dir
            / "runs"
            / "stage_05_two_branch_corrected_residual_core_v2"
        ),
        reuse_stage05_v3b_reference_artifacts=True,
        v3b_reference_artifact_root=(
            v3c_fixture.run_dir
            / "runs"
            / "stage05_v3b_stronger_traj_curr_weight"
        ),
        reuse_stage05_v3c_candidate_artifacts=True,
        v3c_candidate_method_name="stage05_v3c_stronger_semigroup_weight",
        v3c_candidate_artifact_root=(
            v3c_refinement.run_dir
            / "runs"
            / "stage05_v3c_stronger_semigroup_weight"
        ),
        lambda_sg=0.10,
        contextual_reference_summary_path=contextual_path,
        contextual_reference_stage05_epochs=6,
    )

    run_dir = result.run_dir
    summary = _read_json(run_dir / "aggregate_summary.json")
    report = _read_json(run_dir / "comparison_report.json")

    assert summary["stage"] == "stage05_v2_promoted_v3b_refined_v3c_recompare_smoke"
    assert summary["comparison_roles"]["active_reference_at_comparison_start"] == (
        "stage05_v3b_stronger_traj_curr_weight"
    )
    assert summary["comparison_roles"]["refined_v3c_formal_comparison_candidate"] == (
        "stage05_v3c_stronger_semigroup_weight"
    )
    assert summary["comparison_protocol"]["stage_05_v3c_candidate"]["method_name"] == (
        "stage05_v3c_stronger_semigroup_weight"
    )
    assert summary["comparison_protocol"]["stage_05_v3c_candidate"][
        "reference_reused_from_existing_artifacts"
    ] is True
    assert summary["refined_v3c_formal_comparison_candidate_name"] == (
        "stage05_v3c_stronger_semigroup_weight"
    )
    assert "promoted_refined_v3c_materially_beats_promoted_v3b" in summary
    assert "promoted_refined_v3c_avoids_obvious_report_accuracy_regression" in summary
    assert "promoted_refined_v3c_replaces_promoted_v3b_as_active_reference" in summary
    assert "refined_v3c" in summary["contextual_gap_closure_fractions_vs_3072_reference"]
    assert (
        "refined_v3c_minus_promoted_v3b"
        in summary["contextual_gap_closure_fractions_vs_3072_reference"]
    )
    assert summary["recommended_next_move"] in {
        "promote_refined_v3c_as_active_reference",
        "retain_promoted_v3b_as_active_reference",
    }

    assert report["comparison_protocol"]["stage_05_v3c_candidate"]["method_name"] == (
        "stage05_v3c_stronger_semigroup_weight"
    )
    assert report["decision"]["refined_v3c_formal_comparison_candidate_name"] == (
        "stage05_v3c_stronger_semigroup_weight"
    )
    assert report["decision"]["recommended_next_move"] in {
        "promote_refined_v3c_as_active_reference",
        "retain_promoted_v3b_as_active_reference",
    }


def test_stage05_v2_active_v3c_fused_contract_comparison_writes_expected_artifacts(
    tmp_path: Path,
) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="stage05_active_v3c_fused_smoke",
        comparison_variant="stage05_v2_active_v3c_fused_contract_comparison",
        comparison_scope="smoke_only",
        seeds=(0,),
        stage05_epochs=4,
        stage05_eval_steps=5,
        stage05_layer_dims=(64, 16, 10),
        stage05_transport_steps=2,
        active_v3c_lambda_sg=0.10,
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()
    assert (run_dir / "comparison_report.json").exists()
    assert (run_dir / "comparison_report.md").exists()

    rows = _read_csv(run_dir / "aggregate_runs.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")
    report = _read_json(run_dir / "comparison_report.json")

    assert len(rows) == 3
    assert {row["method_name"] for row in rows} == {
        "stage_05_two_branch_corrected_residual_core_v2",
        "stage05_v3c_stronger_semigroup_weight",
        "stage05_v3c_fused_trajectory_semigroup_contract",
    }

    assert summary["stage"] == "stage05_v2_active_v3c_fused_contract_comparison"
    assert summary["comparison_scope"] == "smoke_only"
    assert summary["comparison_roles"]["active_reference_at_comparison_start"] == (
        "stage05_v3c_stronger_semigroup_weight"
    )
    assert summary["comparison_roles"]["fused_candidate"] == (
        "stage05_v3c_fused_trajectory_semigroup_contract"
    )
    assert summary["comparison_protocol"]["stage_05_fused_candidate"]["contract_fusion_enabled"] is True
    assert summary["comparison_protocol"]["stage_05_fused_candidate"][
        "exact_detached_target_barycentric_fusion_enabled"
    ] is True
    assert "pairwise_deltas_vs_stage05_v2_reference" in summary
    assert "pairwise_deltas_vs_active_refined_v3c_reference" in summary
    assert "fused_contract_materially_beats_active_v3c_reference" in summary
    assert "fused_contract_avoids_obvious_report_accuracy_regression" in summary
    assert "fused_contract_replaces_active_v3c_reference" in summary
    assert summary["recommended_next_move"] == (
        "run_real_fixed_budget_v2_vs_active_v3c_vs_fused_contract_comparison"
    )

    assert "pairwise_deltas_vs_active_refined_v3c_reference" in report
    assert report["decision"]["fused_contract_candidate_name"] == (
        "stage05_v3c_fused_trajectory_semigroup_contract"
    )
    assert report["decision"]["recommended_next_move"] == (
        "run_real_fixed_budget_v2_vs_active_v3c_vs_fused_contract_comparison"
    )


def test_stage05_v2_active_v3c_midpoint_reconstructed_contract_comparison_writes_expected_artifacts(
    tmp_path: Path,
) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="stage05_active_v3c_midpoint_reconstructed_smoke",
        comparison_variant="stage05_v2_active_v3c_midpoint_reconstructed_contract_comparison",
        comparison_scope="smoke_only",
        seeds=(0,),
        stage05_epochs=4,
        stage05_eval_steps=5,
        stage05_layer_dims=(64, 16, 10),
        stage05_transport_steps=2,
        active_v3c_lambda_sg=0.10,
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()
    assert (run_dir / "comparison_report.json").exists()
    assert (run_dir / "comparison_report.md").exists()

    rows = _read_csv(run_dir / "aggregate_runs.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")
    report = _read_json(run_dir / "comparison_report.json")

    assert len(rows) == 3
    assert {row["method_name"] for row in rows} == {
        "stage_05_two_branch_corrected_residual_core_v2",
        "stage05_v3c_stronger_semigroup_weight",
        "stage05_v3c_midpoint_reconstructed_trajectory_contract",
    }

    assert summary["stage"] == "stage05_v2_active_v3c_midpoint_reconstructed_contract_comparison"
    assert summary["comparison_scope"] == "smoke_only"
    assert summary["comparison_roles"]["active_reference_at_comparison_start"] == (
        "stage05_v3c_stronger_semigroup_weight"
    )
    assert summary["comparison_roles"]["midpoint_reconstructed_candidate"] == (
        "stage05_v3c_midpoint_reconstructed_trajectory_contract"
    )
    assert summary["comparison_protocol"]["stage_05_midpoint_reconstructed_candidate"][
        "target_reconstruction_enabled"
    ] is True
    assert summary["comparison_protocol"]["stage_05_midpoint_reconstructed_candidate"][
        "midpoint_reconstruction_enabled"
    ] is True
    assert summary["comparison_protocol"]["stage_05_midpoint_reconstructed_candidate"][
        "continuation_reevaluated_at_reconstructed_midpoint"
    ] is True
    assert "pairwise_deltas_vs_stage05_v2_reference" in summary
    assert "pairwise_deltas_vs_active_refined_v3c_reference" in summary
    assert "midpoint_reconstructed_contract_materially_beats_active_v3c_reference" in summary
    assert "midpoint_reconstructed_contract_avoids_obvious_report_accuracy_regression" in summary
    assert "midpoint_reconstructed_contract_replaces_active_v3c_reference" in summary
    assert summary["recommended_next_move"] == (
        "run_real_fixed_budget_v2_vs_active_v3c_vs_midpoint_reconstructed_contract_comparison"
    )

    assert "pairwise_deltas_vs_active_refined_v3c_reference" in report
    assert report["decision"]["midpoint_reconstructed_contract_candidate_name"] == (
        "stage05_v3c_midpoint_reconstructed_trajectory_contract"
    )
    assert report["decision"]["recommended_next_move"] == (
        "run_real_fixed_budget_v2_vs_active_v3c_vs_midpoint_reconstructed_contract_comparison"
    )


def test_frozen_bridge_vs_stage05_v2_comparison_writes_expected_artifacts(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="frozen_bridge_vs_stage05_v2_smoke",
        comparison_variant="stage04_vs_stage05_v2",
        seeds=(0,),
        stage04_epochs=2,
        stage04_eval_steps=5,
        stage04_layer_dims=(64, 16, 10),
        stage05_epochs=3,
        stage05_eval_steps=5,
        stage05_layer_dims=(64, 16, 10),
        stage05_transport_steps=2,
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()
    assert (run_dir / "comparison_report.json").exists()
    assert (run_dir / "comparison_report.md").exists()

    rows = _read_csv(run_dir / "aggregate_runs.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")
    report = _read_json(run_dir / "comparison_report.json")

    assert len(rows) == 2
    method_names = {row["method_name"] for row in rows}
    assert method_names == {
        "stage_04_frozen_bridge",
        "stage_05_two_branch_corrected_residual_core_v2",
    }

    first_row = rows[0]
    assert "method_name" in first_row
    assert "stage_name" in first_row
    assert "configured_transport_steps" in first_row
    assert "one_step_energy_delta_vs_identity" in first_row
    assert "configured_step_energy_delta_vs_identity" in first_row
    assert "configured_step_fixed_point_residual_delta_vs_identity" in first_row
    assert "one_step_energy_delta_vs_local_field_only" in first_row
    assert "configured_step_energy_delta_vs_local_field_only" in first_row
    assert "configured_step_fixed_point_residual_delta_vs_local_field_only" in first_row
    assert "val_accuracy" in first_row
    assert "test_accuracy" in first_row
    assert "val_output_mse" in first_row
    assert "test_output_mse" in first_row
    assert "runtime_proxy_seconds" in first_row
    assert "deterministic_artifact_checks_passed" in first_row

    assert summary["stage"] == "frozen_bridge_vs_two_branch_corrected_residual_core_comparison"
    assert "comparison_protocol" in summary
    assert "by_method" in summary
    assert "pairwise_stage05_v2_vs_stage04" in summary
    assert "stage05_v2_justifies_continued_exploration" in summary
    assert "stage05_v2_replaces_frozen_bridge_on_main" in summary
    assert "stage05_v2_as_new_exploratory_reference" in summary
    assert "stage05_v2_vs_stage04_decision_rationale" in summary
    assert "one_step_mechanism_vs_stage04" in summary
    assert "configured_step_mechanism_vs_stage04" in summary
    assert "report_only_accuracy_vs_stage04" in summary

    assert "decision" in report
    assert "supports" in report
    assert "does_not_support" in report
    assert "stage05_v2_justifies_continued_exploration" in report["decision"]
    assert "stage05_v2_replaces_frozen_bridge_on_main" in report["decision"]
    assert "stage05_v2_as_new_exploratory_reference" in report["decision"]


def test_stage05_v2_longer_training_validation_writes_expected_artifacts(
    tmp_path: Path,
) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="stage05_v2_longer_training_smoke",
        comparison_variant="stage05_v2_longer_training_validation",
        seeds=(0,),
        current_stage05_epochs=3,
        longer_stage05_epochs=5,
        stage05_eval_steps=5,
        stage05_layer_dims=(64, 16, 10),
        stage05_transport_steps=2,
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()
    assert (run_dir / "comparison_report.json").exists()
    assert (run_dir / "comparison_report.md").exists()

    rows = _read_csv(run_dir / "aggregate_runs.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")
    report = _read_json(run_dir / "comparison_report.json")

    assert len(rows) == 2
    method_names = {row["method_name"] for row in rows}
    assert method_names == {
        "stage_05_two_branch_corrected_residual_core_v2_current_budget",
        "stage_05_two_branch_corrected_residual_core_v2_longer_training",
    }

    first_row = rows[0]
    assert "selected_epoch" in first_row
    assert "total_training_epochs" in first_row
    assert "selection_hits_final_training_boundary" in first_row
    assert "configured_step_energy_delta_vs_identity" in first_row
    assert "configured_step_fixed_point_residual_delta_vs_identity" in first_row
    assert "val_accuracy" in first_row
    assert "test_accuracy" in first_row
    assert "runtime_proxy_seconds" in first_row

    assert summary["stage"] == "stage05_v2_longer_training_validation"
    assert "comparison_protocol" in summary
    assert "by_method" in summary
    assert "pairwise_longer_budget_vs_current_budget" in summary
    assert (
        "stage05_v2_longer_training_materially_improves_configured_step_mechanism"
        in summary
    )
    assert (
        "stage05_v2_longer_training_materially_improves_report_only_accuracy"
        in summary
    )
    assert "recommended_next_move" in summary
    assert "decision_rationale" in summary

    assert "decision" in report
    assert "supports" in report
    assert "does_not_support" in report
    assert (
        "stage05_v2_longer_training_materially_improves_configured_step_mechanism"
        in report["decision"]
    )
    assert "recommended_next_move" in report["decision"]


def test_stage05_v2_budget_push_validation_writes_expected_artifacts(
    tmp_path: Path,
) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="stage05_v2_budget_push_smoke",
        comparison_variant="stage05_v2_budget_push_validation",
        experiment_name="stage05_v2_budget_push_validation_smoke_custom",
        seeds=(0,),
        reference_stage05_epochs=4,
        stronger_stage05_epochs=6,
        stage05_eval_steps=5,
        stage05_layer_dims=(64, 16, 10),
        stage05_transport_steps=2,
    )

    run_dir = result.run_dir
    assert run_dir.name == "stage05_v2_budget_push_validation_smoke_custom"
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()
    assert (run_dir / "comparison_report.json").exists()
    assert (run_dir / "comparison_report.md").exists()

    rows = _read_csv(run_dir / "aggregate_runs.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")
    report = _read_json(run_dir / "comparison_report.json")

    assert len(rows) == 2
    method_names = {row["method_name"] for row in rows}
    assert method_names == {
        "stage_05_two_branch_corrected_residual_core_v2_budget_reference",
        "stage_05_two_branch_corrected_residual_core_v2_budget_push",
    }

    first_row = rows[0]
    assert "selected_epoch" in first_row
    assert "total_training_epochs" in first_row
    assert "selection_hits_final_training_boundary" in first_row
    assert "configured_step_energy_delta_vs_identity" in first_row
    assert "configured_step_fixed_point_residual_delta_vs_identity" in first_row
    assert "val_accuracy" in first_row
    assert "test_accuracy" in first_row
    assert "runtime_proxy_seconds" in first_row

    assert summary["stage"] == "stage05_v2_budget_push_validation_smoke_custom"
    assert "comparison_protocol" in summary
    assert "by_method" in summary
    assert "pairwise_budget_push_vs_reference_budget" in summary
    assert "stage05_v2_budget_push_materially_improves_configured_step_mechanism" in summary
    assert "stage05_v2_budget_push_materially_improves_report_only_accuracy" in summary
    assert "configured_step_gain_fraction_vs_reference" in summary
    assert "report_accuracy_gain_vs_reference" in summary
    assert "budget_push_selection_hits_final_training_boundary_on_all_seeds" in summary
    assert "budget_line_still_looks_boundary_limited" in summary
    assert "budget_line_should_continue" in summary
    assert "budget_line_should_stop_and_open_v3" in summary
    assert "budget_line_interpretation" in summary
    assert "contextual_accuracy_note" in summary
    assert "recommended_next_move" in summary
    assert "decision_rationale" in summary

    assert "decision" in report
    assert "contextual_accuracy_note" in report
    assert "supports" in report
    assert "does_not_support" in report
    assert "stage05_v2_budget_push_materially_improves_configured_step_mechanism" in report["decision"]
    assert "configured_step_gain_fraction_vs_reference" in report["decision"]
    assert "budget_line_should_continue" in report["decision"]
    assert "budget_line_interpretation" in report["decision"]
    assert "recommended_next_move" in report["decision"]


def test_stage05_v2_efficiency_diagnostic_writes_expected_artifacts(
    tmp_path: Path,
) -> None:
    reference_fixture = load_run()(
        output_root=tmp_path,
        run_id="stage05_v2_efficiency_reference_fixture",
        comparison_variant="stage05_v2_budget_push_validation",
        experiment_name="stage05_v2_efficiency_reference_fixture",
        seeds=(0,),
        reference_stage05_epochs=4,
        stronger_stage05_epochs=6,
        stage05_eval_steps=5,
        stage05_layer_dims=(64, 16, 10),
        stage05_transport_steps=2,
    )

    result = load_run()(
        output_root=tmp_path,
        run_id="stage05_v2_efficiency_smoke",
        comparison_variant="stage05_v2_efficiency_diagnostic_at_1536",
        experiment_name="stage05_v2_efficiency_diagnostic_smoke_custom",
        seeds=(0,),
        reference_stage05_epochs=4,
        contextual_reference_stage05_epochs=6,
        stage05_eval_steps=5,
        stage05_layer_dims=(64, 16, 10),
        stage05_transport_steps=2,
        optimized_lambda_id_warmup_epochs=1,
        optimized_lambda_id_ramp_epochs=1,
        reference_artifact_root=(
            reference_fixture.run_dir
            / "runs"
            / "stage_05_two_branch_corrected_residual_core_v2_budget_reference"
        ),
        contextual_reference_summary_path=reference_fixture.run_dir / "aggregate_summary.json",
    )

    run_dir = result.run_dir
    assert run_dir.name == "stage05_v2_efficiency_diagnostic_smoke_custom"
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()
    assert (run_dir / "comparison_report.json").exists()
    assert (run_dir / "comparison_report.md").exists()

    rows = _read_csv(run_dir / "aggregate_runs.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")
    report = _read_json(run_dir / "comparison_report.json")

    assert len(rows) == 2
    method_names = {row["method_name"] for row in rows}
    assert method_names == {
        "stage_05_two_branch_corrected_residual_core_v2_efficiency_reference",
        "stage_05_two_branch_corrected_residual_core_v2_efficiency_candidate",
    }

    first_row = rows[0]
    assert "selected_epoch" in first_row
    assert "total_training_epochs" in first_row
    assert "selection_hits_final_training_boundary" in first_row
    assert "configured_step_energy_delta_vs_identity" in first_row
    assert "configured_step_fixed_point_residual_delta_vs_identity" in first_row
    assert "val_accuracy" in first_row
    assert "test_accuracy" in first_row
    assert "runtime_proxy_seconds" in first_row

    assert summary["stage"] == "stage05_v2_efficiency_diagnostic_smoke_custom"
    assert "comparison_protocol" in summary
    assert "by_method" in summary
    assert "contextual_3072_reference" in summary
    assert "pairwise_best_optimized_1536_vs_current_1536_default" in summary
    assert (
        "same_family_efficiency_change_materially_improves_configured_step_mechanism"
        in summary
    )
    assert "same_family_efficiency_change_materially_improves_report_only_accuracy" in summary
    assert "same_family_efficiency_change_materially_narrows_gap_to_3072_reference" in summary
    assert "configured_step_gap_closed_fraction_vs_3072_reference" in summary
    assert "report_accuracy_gap_closed_fraction_vs_3072_reference" in summary
    assert "recommended_next_move" in summary
    assert "decision_rationale" in summary

    assert "decision" in report
    assert "contextual_3072_reference" in report
    assert "supports" in report
    assert "does_not_support" in report
    assert (
        "same_family_efficiency_change_materially_improves_configured_step_mechanism"
        in report["decision"]
    )
    assert "same_family_efficiency_change_materially_narrows_gap_to_3072_reference" in report["decision"]
    assert "recommended_next_move" in report["decision"]
