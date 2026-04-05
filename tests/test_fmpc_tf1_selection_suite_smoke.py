from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path

from pc.fmpc_tf1 import FMPCTF1RunResult
from pc.fmpc_tf1_selection_suite import (
    FMPCTF1SelectionSuiteConfig,
    run_fmpc_tf1_selection_suite,
)


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _fake_tf1_run(config):
    run_dir = Path(config.output_root) / config.experiment_name / str(config.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "phase": "Phase TF1",
        "stage": "teacher_free_fmpc_v1",
        "preset_name": "baseline_comparable",
        "selection_metric": "val_transported_final_energy",
        "selection_metric_source": "val_metric",
        "report_metric_source": "test_metric",
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return FMPCTF1RunResult(
        run_dir=run_dir,
        config={
            "dataset": {
                "data_seed": 0,
                "train_fraction": 0.7,
                "val_fraction": 0.15,
                "test_fraction": 0.15,
            }
        },
        epoch_metrics=[],
        summary=summary,
    )


def _fake_selection_policy_report(result, config):
    variant_bonus = 0.05 if config.model_variant == "tf1_mlp_aug" else 0.0
    step_bonus = 0.02 * float(config.transport_steps)
    return {
        "phase": "Phase TF1",
        "stage": "teacher_free_fmpc_v1_selection_suite_run",
        "preset_name": "baseline_comparable",
        "run_id": config.run_id,
        "model_variant": config.model_variant,
        "warmup_epochs": int(config.warmup_epochs),
        "transport_steps": int(config.transport_steps),
        "identity_loss_weight": float(config.identity_loss_weight),
        "feature_aware_tangents": bool(config.feature_aware_tangents),
        "selection_rules": {
            "val_transported_final_energy": {
                "selected_epoch": 7,
                "val_accuracy": 0.40 + variant_bonus + step_bonus,
                "test_accuracy": 0.41 + variant_bonus + step_bonus,
                "val_transported_final_energy": 0.50 - variant_bonus - step_bonus,
                "val_energy_delta_vs_local_field_only": -0.01,
                "validation_gate_passed": True,
            },
            "val_accuracy": {
                "selected_epoch": 9,
                "val_accuracy": 0.45 + variant_bonus + step_bonus,
                "test_accuracy": 0.46 + variant_bonus + step_bonus,
                "val_transported_final_energy": 0.55 - variant_bonus - step_bonus,
                "val_energy_delta_vs_local_field_only": 0.00,
                "validation_gate_passed": False,
            },
            "val_energy_delta_vs_local_field_only": {
                "selected_epoch": 8,
                "val_accuracy": 0.43 + variant_bonus + step_bonus,
                "test_accuracy": 0.44 + variant_bonus + step_bonus,
                "val_transported_final_energy": 0.52 - variant_bonus - step_bonus,
                "val_energy_delta_vs_local_field_only": -0.02,
                "validation_gate_passed": True,
            },
        },
    }


def test_fmpc_tf1_selection_suite_writes_expected_artifacts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("pc.fmpc_tf1_selection_suite.run_fmpc_tf1_experiment", _fake_tf1_run)
    monkeypatch.setattr(
        "pc.fmpc_tf1_selection_suite._selection_policy_report_for_run",
        _fake_selection_policy_report,
    )

    result = run_fmpc_tf1_selection_suite(
        FMPCTF1SelectionSuiteConfig(output_root=tmp_path, run_id="selection_suite_smoke")
    )
    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()

    config = _read_json(run_dir / "config.json")
    summary = _read_json(run_dir / "aggregate_summary.json")
    rows = _read_csv(run_dir / "aggregate_runs.csv")
    per_run_report = _read_json(
        run_dir
        / rows[0]["selection_policy_summary_path"].replace("/", "\\")
    )

    assert config["stage"] == "teacher_free_fmpc_v1_selection_suite"
    assert config["search_space"]["warmup_epochs"] == 5
    assert config["search_space"]["feature_aware_tangents"] is False
    assert len(rows) == 4
    assert "average_test_accuracy_by_selector" in summary
    assert "average_val_accuracy_by_selector" in summary
    assert "winner_by_selector" in summary
    assert "selector_changes_recover_accuracy_left_on_table" in summary
    assert "selection_mismatch_is_substantial" in summary
    assert "val_energy_delta_vs_local_field_only_is_useful" in summary
    assert "mean_test_accuracy_gain_vs_val_energy_selection" in summary
    assert "fraction_runs_improved_test_accuracy_vs_val_energy_selection" in summary
    assert per_run_report["stage"] == "teacher_free_fmpc_v1_selection_suite_run"
    assert set(per_run_report["selection_rules"].keys()) == {
        "val_transported_final_energy",
        "val_accuracy",
        "val_energy_delta_vs_local_field_only",
    }


def test_fmpc_tf1_selection_suite_aggregate_is_deterministic(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("pc.fmpc_tf1_selection_suite.run_fmpc_tf1_experiment", _fake_tf1_run)
    monkeypatch.setattr(
        "pc.fmpc_tf1_selection_suite._selection_policy_report_for_run",
        _fake_selection_policy_report,
    )

    first = run_fmpc_tf1_selection_suite(
        FMPCTF1SelectionSuiteConfig(output_root=tmp_path / "first", run_id="deterministic")
    )
    second = run_fmpc_tf1_selection_suite(
        FMPCTF1SelectionSuiteConfig(output_root=tmp_path / "second", run_id="deterministic")
    )

    first_summary = _read_json(first.run_dir / "aggregate_summary.json")
    second_summary = _read_json(second.run_dir / "aggregate_summary.json")
    first_rows = _read_csv(first.run_dir / "aggregate_runs.csv")
    second_rows = _read_csv(second.run_dir / "aggregate_runs.csv")

    assert first_summary == second_summary
    assert first_rows == second_rows


def test_fmpc_tf1_selection_suite_entrypoint_is_available(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("pc.fmpc_tf1_selection_suite.run_fmpc_tf1_experiment", _fake_tf1_run)
    monkeypatch.setattr(
        "pc.fmpc_tf1_selection_suite._selection_policy_report_for_run",
        _fake_selection_policy_report,
    )
    module = runpy.run_path(str(ROOT / "experiments" / "fmpc_tf1_selection_suite.py"))
    run = module["run"]

    result = run(output_root=tmp_path, run_id="selection_suite_entrypoint")
    assert (result.run_dir / "aggregate_summary.json").exists()
