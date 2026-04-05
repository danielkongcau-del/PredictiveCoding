from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path
from pathlib import PurePosixPath

from pc.fmpc_tf1 import FMPCTF1Config, FMPCTF1RunResult
from pc.fmpc_tf1_accuracy_tuning_suite import (
    FMPCTF1AccuracyTuningSuiteConfig,
    run_fmpc_tf1_accuracy_tuning_suite,
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
    accuracy_bonus = (0.02 * float(config.identity_loss_weight)) + (0.004 * float(config.bootstrap_substeps)) - (
        0.003 * float(config.hybrid_ramp_epochs)
    )
    summary = {
        "phase": "Phase TF1",
        "stage": "teacher_free_fmpc_v1",
        "preset_name": str(config.preset_name),
        "val_accuracy": 0.65 + accuracy_bonus + (0.005 * int(config.run_seed)),
        "test_accuracy": 0.64 + accuracy_bonus + (0.005 * int(config.run_seed)),
        "gate_passing_epoch_count": int(2 + (5 * float(config.identity_loss_weight)) + (config.bootstrap_substeps // 4)),
        "val_transported_final_energy": 0.40 - accuracy_bonus,
        "best_epoch": 40 + int(config.run_seed),
        "selected_epoch_passes_gate": True,
        "selector_fallback_used": False,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return FMPCTF1RunResult(run_dir=run_dir, config={}, epoch_metrics=[], summary=summary)


def _write_fake_external_summary(path: Path) -> None:
    payload = {
        "mean_std_test_accuracy_by_method": {
            "canonical_slow_pc_digits_baseline": {
                "mean": 0.90,
                "std": 0.01,
            }
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_fmpc_tf1_accuracy_tuning_suite_writes_expected_artifacts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("pc.fmpc_tf1_accuracy_tuning_suite.run_fmpc_tf1_experiment", _fake_tf1_run)
    external_summary_path = tmp_path / "outputs" / "fmpc_tf1_external_comparison_suite" / "aggregate_summary.json"
    _write_fake_external_summary(external_summary_path)

    result = run_fmpc_tf1_accuracy_tuning_suite(
        FMPCTF1AccuracyTuningSuiteConfig(
            output_root=tmp_path,
            run_id="accuracy_tuning_smoke",
            slow_pc_reference_summary_path=external_summary_path,
        )
    )
    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()

    config = _read_json(run_dir / "config.json")
    summary = _read_json(run_dir / "aggregate_summary.json")
    rows = _read_csv(run_dir / "aggregate_runs.csv")
    first_run_summary = _read_json(_resolve_posix_relative(run_dir, rows[0]["run_summary_path"]))

    assert config["stage"] == "teacher_free_fmpc_v1_accuracy_tuning_suite"
    assert len(rows) == 60
    assert rows[0]["identity_loss_weight"] in {"0.1", "0.2", "0.3"}
    assert rows[0]["hybrid_ramp_epochs"] in {"5", "10"}
    assert rows[0]["bootstrap_substeps"] in {"4", "8"}
    assert "mean_std_val_accuracy_by_configuration" in summary
    assert "mean_std_test_accuracy_by_configuration" in summary
    assert "mean_gate_passing_epoch_count_by_configuration" in summary
    assert "best_configuration_by_mean_test_accuracy" in summary
    assert "any_configuration_materially_narrows_gap_to_slow_pc_baseline" in summary
    assert "recommended_next_working_default" in summary
    assert first_run_summary["stage"] == "teacher_free_fmpc_v1"


def test_fmpc_tf1_accuracy_tuning_suite_aggregate_is_deterministic(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("pc.fmpc_tf1_accuracy_tuning_suite.run_fmpc_tf1_experiment", _fake_tf1_run)
    external_summary_path = tmp_path / "outputs" / "fmpc_tf1_external_comparison_suite" / "aggregate_summary.json"
    _write_fake_external_summary(external_summary_path)

    first = run_fmpc_tf1_accuracy_tuning_suite(
        FMPCTF1AccuracyTuningSuiteConfig(
            output_root=tmp_path / "first",
            run_id="deterministic",
            slow_pc_reference_summary_path=external_summary_path,
        )
    )
    second = run_fmpc_tf1_accuracy_tuning_suite(
        FMPCTF1AccuracyTuningSuiteConfig(
            output_root=tmp_path / "second",
            run_id="deterministic",
            slow_pc_reference_summary_path=external_summary_path,
        )
    )

    assert _read_json(first.run_dir / "aggregate_summary.json") == _read_json(second.run_dir / "aggregate_summary.json")
    assert _read_csv(first.run_dir / "aggregate_runs.csv") == _read_csv(second.run_dir / "aggregate_runs.csv")


def test_fmpc_tf1_accuracy_tuning_suite_entrypoint_is_available(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("pc.fmpc_tf1_accuracy_tuning_suite.run_fmpc_tf1_experiment", _fake_tf1_run)
    external_summary_path = tmp_path / "outputs" / "fmpc_tf1_external_comparison_suite" / "aggregate_summary.json"
    _write_fake_external_summary(external_summary_path)

    module = runpy.run_path(str(ROOT / "experiments" / "fmpc_tf1_accuracy_tuning_suite.py"))
    run = module["run"]
    result = run(output_root=tmp_path, run_id="accuracy_tuning_entrypoint")
    assert (result.run_dir / "aggregate_summary.json").exists()
