from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path
from pathlib import PurePosixPath

from pc.fmpc_tf1 import FMPCTF1Config, FMPCTF1RunResult
from pc.fmpc_tf1_default_adoption_suite import (
    FMPCTF1DefaultAdoptionSuiteConfig,
    run_fmpc_tf1_default_adoption_suite,
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


def _fake_tf1_run(config: FMPCTF1Config) -> FMPCTF1RunResult:
    run_dir = Path(config.output_root) / config.experiment_name / str(config.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    preset_name = str(config.preset_name)
    preset_bonus = {
        "mechanism_smoke": 0.00,
        "baseline_comparable": 0.03,
        "baseline_working_default": 0.05,
    }[preset_name]
    selector = config.checkpoint_selector
    gate_count = {"mechanism_smoke": 2, "baseline_comparable": 1, "baseline_working_default": 4}[preset_name]
    summary = {
        "phase": "Phase TF1",
        "stage": "teacher_free_fmpc_v1",
        "preset_name": preset_name,
        "checkpoint_selector": selector,
        "best_epoch": 10 + int(config.run_seed),
        "selected_epoch_passes_gate": preset_name != "baseline_comparable" or int(config.run_seed) % 2 == 0,
        "gate_passing_epoch_count": gate_count,
        "selector_fallback_used": preset_name == "baseline_working_default" and int(config.run_seed) == 1,
        "selected_epoch_selection_reason": "fake summary for smoke",
        "val_accuracy": 0.40 + preset_bonus + (0.01 * int(config.run_seed)),
        "test_accuracy": 0.42 + preset_bonus + (0.01 * int(config.run_seed)),
        "val_transported_final_energy": 0.50 - preset_bonus + (0.005 * int(config.run_seed)),
        "val_energy_delta_vs_identity": -0.03 - preset_bonus,
        "val_energy_delta_vs_local_field_only": -0.01 - preset_bonus,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return FMPCTF1RunResult(run_dir=run_dir, config={}, epoch_metrics=[], summary=summary)


def test_fmpc_tf1_default_adoption_suite_writes_expected_artifacts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("pc.fmpc_tf1_default_adoption_suite.run_fmpc_tf1_experiment", _fake_tf1_run)

    result = run_fmpc_tf1_default_adoption_suite(
        FMPCTF1DefaultAdoptionSuiteConfig(output_root=tmp_path, run_id="default_adoption_smoke")
    )
    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()

    config = _read_json(run_dir / "config.json")
    summary = _read_json(run_dir / "aggregate_summary.json")
    rows = _read_csv(run_dir / "aggregate_runs.csv")
    first_run_summary = _read_json(_resolve_posix_relative(run_dir, rows[0]["run_summary_path"]))

    assert config["stage"] == "teacher_free_fmpc_v1_default_adoption_suite"
    assert len(rows) == 15
    assert rows[0]["preset_name"] in {"mechanism_smoke", "baseline_comparable", "baseline_working_default"}
    assert rows[0]["checkpoint_selector"] in {
        "energy_only",
        "gate_constrained_accuracy_then_val_accuracy",
    }
    assert "mean_std_val_accuracy_by_preset" in summary
    assert "mean_std_test_accuracy_by_preset" in summary
    assert "mean_gate_passing_epoch_count_by_preset" in summary
    assert "gate_feasible_fraction_by_preset" in summary
    assert "mean_selected_epoch_by_preset" in summary
    assert "baseline_working_default_improves_over_baseline_comparable" in summary
    assert "mechanism_smoke_remains_smoke_preset_not_practical_default" in summary
    assert "recommended_main_tf1_preset_after_adoption_validation" in summary
    assert first_run_summary["stage"] == "teacher_free_fmpc_v1"


def test_fmpc_tf1_default_adoption_suite_aggregate_is_deterministic(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("pc.fmpc_tf1_default_adoption_suite.run_fmpc_tf1_experiment", _fake_tf1_run)

    first = run_fmpc_tf1_default_adoption_suite(
        FMPCTF1DefaultAdoptionSuiteConfig(output_root=tmp_path / "first", run_id="deterministic")
    )
    second = run_fmpc_tf1_default_adoption_suite(
        FMPCTF1DefaultAdoptionSuiteConfig(output_root=tmp_path / "second", run_id="deterministic")
    )

    assert _read_json(first.run_dir / "aggregate_summary.json") == _read_json(second.run_dir / "aggregate_summary.json")
    assert _read_csv(first.run_dir / "aggregate_runs.csv") == _read_csv(second.run_dir / "aggregate_runs.csv")


def test_fmpc_tf1_default_adoption_suite_entrypoint_is_available(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("pc.fmpc_tf1_default_adoption_suite.run_fmpc_tf1_experiment", _fake_tf1_run)
    module = runpy.run_path(str(ROOT / "experiments" / "fmpc_tf1_default_adoption_suite.py"))
    run = module["run"]

    result = run(output_root=tmp_path, run_id="default_adoption_entrypoint")
    assert (result.run_dir / "aggregate_summary.json").exists()
