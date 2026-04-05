from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path
from pathlib import PurePosixPath

from pc.fmpc_tf1 import FMPCTF1Config, FMPCTF1RunResult
from pc.fmpc_tf1_external_comparison_suite import (
    FMPCTF1ExternalComparisonSuiteConfig,
    run_fmpc_tf1_external_comparison_suite,
)
from pc.real_pc import RealPCRunResult


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve_posix_relative(base_dir: Path, relative_path: str) -> Path:
    return base_dir.joinpath(*PurePosixPath(relative_path).parts)


def _fake_tf1_run(config: FMPCTF1Config) -> FMPCTF1RunResult:
    run_dir = Path(config.output_root) / config.experiment_name / str(config.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    preset_bonus = {
        "baseline_comparable": 0.00,
        "baseline_working_default": 0.06,
    }[str(config.preset_name)]
    summary = {
        "phase": "Phase TF1",
        "stage": "teacher_free_fmpc_v1",
        "preset_name": str(config.preset_name),
        "checkpoint_selector": str(config.checkpoint_selector),
        "val_accuracy": 0.62 + preset_bonus + (0.01 * int(config.run_seed)),
        "test_accuracy": 0.60 + preset_bonus + (0.01 * int(config.run_seed)),
        "val_transported_final_energy": 0.43 - preset_bonus + (0.002 * int(config.run_seed)),
        "gate_passing_epoch_count": 1 if str(config.preset_name) == "baseline_comparable" else 5,
        "timing": {"train_wall_time_seconds": 1.5 + (0.1 * int(config.run_seed))},
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return FMPCTF1RunResult(run_dir=run_dir, config={}, epoch_metrics=[], summary=summary)


def _fake_digits_pc_run(config) -> RealPCRunResult:
    run_dir = Path(config.output_root) / config.experiment_name / str(config.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "phase": "Phase 3",
        "model_family": "pc",
        "val_metric": 0.81 + (0.005 * int(config.run_seed)),
        "test_metric": 0.80 + (0.005 * int(config.run_seed)),
        "timing": {"train_wall_time_seconds": 2.5 + (0.1 * int(config.run_seed))},
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return RealPCRunResult(run_dir=run_dir, config={}, epoch_metrics=[], summary=summary)


def test_fmpc_tf1_external_comparison_suite_writes_expected_artifacts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("pc.fmpc_tf1_external_comparison_suite.run_fmpc_tf1_experiment", _fake_tf1_run)
    monkeypatch.setattr("pc.fmpc_tf1_external_comparison_suite.run_digits_pc_experiment", _fake_digits_pc_run)

    result = run_fmpc_tf1_external_comparison_suite(
        FMPCTF1ExternalComparisonSuiteConfig(output_root=tmp_path, run_id="external_comparison_smoke")
    )
    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()

    config = _read_json(run_dir / "config.json")
    summary = _read_json(run_dir / "aggregate_summary.json")
    rows = _read_csv(run_dir / "aggregate_runs.csv")
    first_run_summary = _read_json(_resolve_posix_relative(run_dir, rows[0]["run_summary_path"]))

    assert config["stage"] == "teacher_free_fmpc_v1_external_comparison_suite"
    assert len(rows) == 15
    assert rows[0]["method_name"] in {
        "baseline_comparable",
        "baseline_working_default",
        "canonical_slow_pc_digits_baseline",
    }
    assert "mean_std_val_accuracy_by_method" in summary
    assert "mean_std_test_accuracy_by_method" in summary
    assert "mean_gate_passing_epoch_count_by_tf1_preset" in summary
    assert "baseline_working_default_over_baseline_comparable" in summary
    assert "baseline_working_default_vs_slow_pc_baseline_gap" in summary
    assert "baseline_working_default_strong_enough_to_remain_main_tf1_preset" in summary
    assert "recommended_next_research_focus" in summary
    assert first_run_summary["phase"] in {"Phase TF1", "Phase 3"}


def test_fmpc_tf1_external_comparison_suite_aggregate_is_deterministic(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("pc.fmpc_tf1_external_comparison_suite.run_fmpc_tf1_experiment", _fake_tf1_run)
    monkeypatch.setattr("pc.fmpc_tf1_external_comparison_suite.run_digits_pc_experiment", _fake_digits_pc_run)

    first = run_fmpc_tf1_external_comparison_suite(
        FMPCTF1ExternalComparisonSuiteConfig(output_root=tmp_path / "first", run_id="deterministic")
    )
    second = run_fmpc_tf1_external_comparison_suite(
        FMPCTF1ExternalComparisonSuiteConfig(output_root=tmp_path / "second", run_id="deterministic")
    )

    assert _read_json(first.run_dir / "aggregate_summary.json") == _read_json(second.run_dir / "aggregate_summary.json")
    assert _read_csv(first.run_dir / "aggregate_runs.csv") == _read_csv(second.run_dir / "aggregate_runs.csv")


def test_fmpc_tf1_external_comparison_suite_entrypoint_is_available(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("pc.fmpc_tf1_external_comparison_suite.run_fmpc_tf1_experiment", _fake_tf1_run)
    monkeypatch.setattr("pc.fmpc_tf1_external_comparison_suite.run_digits_pc_experiment", _fake_digits_pc_run)
    module = runpy.run_path(str(ROOT / "experiments" / "fmpc_tf1_external_comparison_suite.py"))
    run = module["run"]

    result = run(output_root=tmp_path, run_id="external_comparison_entrypoint")
    assert (result.run_dir / "aggregate_summary.json").exists()
