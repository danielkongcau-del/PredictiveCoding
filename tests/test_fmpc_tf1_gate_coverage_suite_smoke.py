from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path
from pathlib import PurePosixPath

from pc.fmpc_tf1 import FMPCTF1RunResult
from pc.fmpc_tf1_gate_coverage_suite import (
    FMPCTF1GateCoverageSuiteConfig,
    run_fmpc_tf1_gate_coverage_suite,
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


def _fake_gate_coverage_report(result, config):
    base_epochs = 4 if config.model_variant == "tf1_mlp_aug" else 0
    if bool(config.feature_aware_tangents):
        base_epochs += 2
    if float(config.identity_loss_weight) == 0.1:
        base_epochs += 1
    gate_exists = base_epochs > 0
    return {
        "phase": "Phase TF1",
        "stage": "teacher_free_fmpc_v1_gate_coverage_run",
        "run_id": config.run_id,
        "preset_name": "baseline_comparable",
        "model_variant": config.model_variant,
        "transport_steps": int(config.transport_steps),
        "identity_loss_weight": float(config.identity_loss_weight),
        "feature_aware_tangents": bool(config.feature_aware_tangents),
        "number_of_gate_passing_epochs": int(base_epochs),
        "first_gate_passing_epoch": 8 if gate_exists else None,
        "best_val_accuracy_among_gate_passing_epochs": 0.6 if gate_exists else None,
        "best_val_transported_energy_among_gate_passing_epochs": 0.35 if gate_exists else None,
        "best_overall_val_accuracy": 0.7,
        "gap_between_best_overall_val_accuracy_and_best_gate_passing_val_accuracy": 0.1 if gate_exists else None,
        "whether_any_gate_passing_epoch_exists": gate_exists,
        "selector_reports": {
            "val_transported_final_energy": {
                "selected_epoch": 10,
                "val_accuracy": 0.5,
                "test_accuracy": 0.51,
                "val_transported_final_energy": 0.40,
            },
            "val_accuracy": {
                "selected_epoch": 12,
                "val_accuracy": 0.7,
                "test_accuracy": 0.69,
                "val_transported_final_energy": 0.45,
            },
            "gate_constrained_val_accuracy": None
            if not gate_exists
            else {
                "selected_epoch": 11,
                "val_accuracy": 0.6,
                "test_accuracy": 0.58,
                "val_transported_final_energy": 0.39,
            },
        },
    }


def test_fmpc_tf1_gate_coverage_suite_writes_expected_artifacts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("pc.fmpc_tf1_gate_coverage_suite.run_fmpc_tf1_experiment", _fake_tf1_run)
    monkeypatch.setattr(
        "pc.fmpc_tf1_gate_coverage_suite._gate_coverage_report_for_run",
        _fake_gate_coverage_report,
    )

    result = run_fmpc_tf1_gate_coverage_suite(
        FMPCTF1GateCoverageSuiteConfig(output_root=tmp_path, run_id="gate_coverage_smoke")
    )
    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()

    config = _read_json(run_dir / "config.json")
    summary = _read_json(run_dir / "aggregate_summary.json")
    rows = _read_csv(run_dir / "aggregate_runs.csv")
    per_run = _read_json(_resolve_posix_relative(run_dir, rows[0]["gate_coverage_summary_path"]))

    assert config["stage"] == "teacher_free_fmpc_v1_gate_coverage_suite"
    assert len(rows) == 9
    assert "/" in rows[0]["gate_coverage_summary_path"]
    assert "\\" not in rows[0]["gate_coverage_summary_path"]
    assert "gate_feasible_configurations" in summary
    assert "tf1_mlp_aug_transport_steps_1_is_only_gate_feasible_family" in summary
    assert "feature_aware_tangents_help_gate_coverage" in summary
    assert "gate_epochs_by_identity_loss_weight" in summary
    assert "gate_constrained_accuracy_selection_meaningful" in summary
    assert per_run["stage"] == "teacher_free_fmpc_v1_gate_coverage_run"
    assert "selector_reports" in per_run


def test_fmpc_tf1_gate_coverage_suite_aggregate_is_deterministic(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("pc.fmpc_tf1_gate_coverage_suite.run_fmpc_tf1_experiment", _fake_tf1_run)
    monkeypatch.setattr(
        "pc.fmpc_tf1_gate_coverage_suite._gate_coverage_report_for_run",
        _fake_gate_coverage_report,
    )

    first = run_fmpc_tf1_gate_coverage_suite(
        FMPCTF1GateCoverageSuiteConfig(output_root=tmp_path / "first", run_id="deterministic")
    )
    second = run_fmpc_tf1_gate_coverage_suite(
        FMPCTF1GateCoverageSuiteConfig(output_root=tmp_path / "second", run_id="deterministic")
    )

    assert _read_json(first.run_dir / "aggregate_summary.json") == _read_json(
        second.run_dir / "aggregate_summary.json"
    )
    assert _read_csv(first.run_dir / "aggregate_runs.csv") == _read_csv(
        second.run_dir / "aggregate_runs.csv"
    )


def test_fmpc_tf1_gate_coverage_suite_entrypoint_is_available(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("pc.fmpc_tf1_gate_coverage_suite.run_fmpc_tf1_experiment", _fake_tf1_run)
    monkeypatch.setattr(
        "pc.fmpc_tf1_gate_coverage_suite._gate_coverage_report_for_run",
        _fake_gate_coverage_report,
    )
    module = runpy.run_path(str(ROOT / "experiments" / "fmpc_tf1_gate_coverage_suite.py"))
    run = module["run"]

    result = run(output_root=tmp_path, run_id="gate_coverage_entrypoint")
    assert (result.run_dir / "aggregate_summary.json").exists()
