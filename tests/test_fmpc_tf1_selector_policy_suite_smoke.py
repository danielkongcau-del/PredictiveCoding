from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path
from pathlib import PurePosixPath

from pc.fmpc_tf1 import FMPCTF1RunResult
from pc.fmpc_tf1_selector_policy_suite import (
    FMPCTF1SelectorPolicySuiteConfig,
    run_fmpc_tf1_selector_policy_suite,
)


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve_posix_relative(base_dir: Path, relative_path: str) -> Path:
    return base_dir.joinpath(*PurePosixPath(relative_path).parts)


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
        config={},
        epoch_metrics=[],
        summary=summary,
    )


def _fake_selector_policy_report(result, config):
    gate_exists = float(config.identity_loss_weight) >= 0.2 or int(config.run_seed) % 2 == 0
    energy_test = 0.50 + (0.01 * int(config.run_seed))
    valacc_test = energy_test + 0.03
    cascade_energy_test = energy_test + (0.02 if gate_exists else 0.0)
    cascade_valacc_test = energy_test + (0.05 if gate_exists else 0.03)
    return {
        "phase": "Phase TF1",
        "stage": "teacher_free_fmpc_v1_selector_policy_run",
        "run_id": config.run_id,
        "preset_name": "baseline_comparable",
        "seed": int(config.run_seed),
        "model_variant": config.model_variant,
        "transport_steps": int(config.transport_steps),
        "warmup_epochs": int(config.warmup_epochs),
        "identity_loss_weight": float(config.identity_loss_weight),
        "feature_aware_tangents": bool(config.feature_aware_tangents),
        "whether_any_gate_passing_epoch_exists": gate_exists,
        "number_of_gate_passing_epochs": 4 if gate_exists else 0,
        "best_val_accuracy_among_gate_passing_epochs": 0.62 if gate_exists else None,
        "best_val_transported_energy_among_gate_passing_epochs": 0.36 if gate_exists else None,
        "selector_policies": {
            "energy_only": {
                "selected_epoch": 10,
                "val_accuracy": 0.55,
                "test_accuracy": energy_test,
                "val_transported_final_energy": 0.36,
                "validation_gate_passed": gate_exists,
            },
            "val_accuracy_only": {
                "selected_epoch": 12,
                "val_accuracy": 0.63,
                "test_accuracy": valacc_test,
                "val_transported_final_energy": 0.41,
                "validation_gate_passed": False,
            },
            "gate_constrained_accuracy_then_energy": {
                "selected_epoch": 11,
                "val_accuracy": 0.60 if gate_exists else 0.55,
                "test_accuracy": cascade_energy_test,
                "val_transported_final_energy": 0.37,
                "validation_gate_passed": gate_exists,
            },
            "gate_constrained_accuracy_then_val_accuracy": {
                "selected_epoch": 11 if gate_exists else 12,
                "val_accuracy": 0.62 if gate_exists else 0.63,
                "test_accuracy": cascade_valacc_test,
                "val_transported_final_energy": 0.38 if gate_exists else 0.41,
                "validation_gate_passed": gate_exists,
            },
        },
    }


def test_fmpc_tf1_selector_policy_suite_writes_expected_artifacts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("pc.fmpc_tf1_selector_policy_suite.run_fmpc_tf1_experiment", _fake_tf1_run)
    monkeypatch.setattr(
        "pc.fmpc_tf1_selector_policy_suite._selector_policy_report_for_run",
        _fake_selector_policy_report,
    )

    result = run_fmpc_tf1_selector_policy_suite(
        FMPCTF1SelectorPolicySuiteConfig(output_root=tmp_path, run_id="selector_policy_smoke")
    )
    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()

    config = _read_json(run_dir / "config.json")
    summary = _read_json(run_dir / "aggregate_summary.json")
    rows = _read_csv(run_dir / "aggregate_runs.csv")
    per_run = _read_json(_resolve_posix_relative(run_dir, rows[0]["selector_policy_summary_path"]))

    assert config["stage"] == "teacher_free_fmpc_v1_selector_policy_suite"
    assert len(rows) == 10
    assert "/" in rows[0]["selector_policy_summary_path"]
    assert "\\" not in rows[0]["selector_policy_summary_path"]
    assert "mean_val_accuracy_by_selector_policy_and_identity_weight" in summary
    assert "mean_test_accuracy_by_selector_policy_and_identity_weight" in summary
    assert "std_test_accuracy_by_selector_policy_and_identity_weight" in summary
    assert "selector_policy_winner_by_mean_test_accuracy" in summary
    assert "selector_policy_winner_by_mean_val_accuracy" in summary
    assert "selector_cascade_should_replace_energy_only_default" in summary
    assert "recommended_default_selector_policy" in summary
    assert "recommended_working_identity_loss_weight" in summary
    assert per_run["stage"] == "teacher_free_fmpc_v1_selector_policy_run"


def test_fmpc_tf1_selector_policy_suite_aggregate_is_deterministic(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("pc.fmpc_tf1_selector_policy_suite.run_fmpc_tf1_experiment", _fake_tf1_run)
    monkeypatch.setattr(
        "pc.fmpc_tf1_selector_policy_suite._selector_policy_report_for_run",
        _fake_selector_policy_report,
    )

    first = run_fmpc_tf1_selector_policy_suite(
        FMPCTF1SelectorPolicySuiteConfig(output_root=tmp_path / "first", run_id="deterministic")
    )
    second = run_fmpc_tf1_selector_policy_suite(
        FMPCTF1SelectorPolicySuiteConfig(output_root=tmp_path / "second", run_id="deterministic")
    )

    assert _read_json(first.run_dir / "aggregate_summary.json") == _read_json(
        second.run_dir / "aggregate_summary.json"
    )
    assert _read_csv(first.run_dir / "aggregate_runs.csv") == _read_csv(
        second.run_dir / "aggregate_runs.csv"
    )


def test_fmpc_tf1_selector_policy_suite_entrypoint_is_available(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("pc.fmpc_tf1_selector_policy_suite.run_fmpc_tf1_experiment", _fake_tf1_run)
    monkeypatch.setattr(
        "pc.fmpc_tf1_selector_policy_suite._selector_policy_report_for_run",
        _fake_selector_policy_report,
    )
    module = runpy.run_path(str(ROOT / "experiments" / "fmpc_tf1_selector_policy_suite.py"))
    run = module["run"]

    result = run(output_root=tmp_path, run_id="selector_policy_entrypoint")
    assert (result.run_dir / "aggregate_summary.json").exists()
