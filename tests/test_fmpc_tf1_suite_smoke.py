from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path

from pc.fmpc_tf1 import (
    FMPCTF1Config,
    FMPCTF1RunResult,
    build_tf1_baseline_comparable_config,
    build_tf1_mechanism_smoke_config,
)
from pc.fmpc_tf1_suite import FMPCTF1SuiteConfig, run_fmpc_tf1_suite


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _fake_tf1_run(config: FMPCTF1Config) -> FMPCTF1RunResult:
    run_dir = Path(config.output_root) / config.experiment_name / str(config.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    layer_bonus = 0.01 if config.layer_dims == (64, 64, 10) else 0.0
    variant_bonus = 0.015 if config.model_variant == "tf1_mlp_aug" else 0.0
    warmup_penalty = 0.001 * float(config.warmup_epochs)
    step_bonus = 0.02 * float(config.transport_steps)
    identity_bonus = 0.01 * float(config.identity_loss_weight)
    val_energy = 0.6 - layer_bonus - variant_bonus - step_bonus - identity_bonus + warmup_penalty
    val_accuracy = 0.30 + layer_bonus + variant_bonus + (0.03 * float(config.transport_steps))
    test_accuracy = val_accuracy + 0.01
    summary = {
        "phase": "Phase TF1",
        "stage": "teacher_free_fmpc_v1",
        "preset_name": config.preset_name,
        "teacher_free": True,
        "uses_teacher_artifacts": False,
        "transport_scope": "train_only",
        "model_variant": config.model_variant,
        "use_teacher_free_features": bool(config.use_teacher_free_features),
        "feature_aware_tangents": bool(config.feature_aware_tangents),
        "transport_steps": int(config.transport_steps),
        "warmup_epochs": int(config.warmup_epochs),
        "identity_loss_weight": float(config.identity_loss_weight),
        "selection_metric": "val_transported_final_energy",
        "selection_metric_source": "val_metric",
        "report_metric_source": "test_metric",
        "val_transported_final_energy": float(val_energy),
        "val_energy_delta_vs_identity": float(-0.03),
        "val_energy_delta_vs_local_field_only": float(-0.01),
        "test_transported_final_energy": float(val_energy + 0.02),
        "val_accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy),
        "val_baseline_accuracy": 0.1,
        "test_baseline_accuracy": 0.1,
        "identity_baseline": {
            "val_transported_final_energy": float(val_energy + 0.03),
            "test_transported_final_energy": float(val_energy + 0.05),
            "transport_steps": int(config.transport_steps),
            "rollout_knots": [0.0, 1.0] if config.transport_steps == 1 else [0.0, 0.5, 1.0],
            "energy_metric": "baseline_pc_energy",
        },
        "local_field_only_baseline": {
            "val_transported_final_energy": float(val_energy + 0.01),
            "test_transported_final_energy": float(val_energy + 0.03),
            "transport_steps": int(config.transport_steps),
            "rollout_knots": [0.0, 1.0] if config.transport_steps == 1 else [0.0, 0.5, 1.0],
            "energy_metric": "baseline_pc_energy",
        },
        "validation_gate": {
            "validation_only_gating": True,
            "test_is_report_only": True,
            "passes_identity_comparison": True,
            "passes_local_field_only_comparison": True,
            "passes_majority_baseline_accuracy": True,
        },
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return FMPCTF1RunResult(
        run_dir=run_dir,
        config={},
        epoch_metrics=[],
        summary=summary,
        model=None,
        psi_network=None,
    )


def test_tf1_preset_builders_are_explicitly_labeled() -> None:
    smoke = build_tf1_mechanism_smoke_config()
    baseline = build_tf1_baseline_comparable_config()

    assert smoke.preset_name == "mechanism_smoke"
    assert smoke.layer_dims == (64, 16, 10)
    assert smoke.warmup_epochs == 5
    assert baseline.preset_name == "baseline_comparable"
    assert baseline.layer_dims == (64, 64, 10)
    assert baseline.warmup_epochs == 5


def test_fmpc_tf1_suite_smoke_writes_expected_artifacts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("pc.fmpc_tf1_suite.run_fmpc_tf1_experiment", _fake_tf1_run)

    result = run_fmpc_tf1_suite(FMPCTF1SuiteConfig(output_root=tmp_path, run_id="tf1_suite_smoke"))
    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()

    config = _read_json(run_dir / "config.json")
    summary = _read_json(run_dir / "aggregate_summary.json")
    rows = _read_csv(run_dir / "aggregate_runs.csv")

    assert config["phase"] == "Phase TF1"
    assert config["stage"] == "teacher_free_fmpc_v1_calibration_suite"
    assert config["search_space"]["feature_aware_tangents"] is False
    assert config["search_space"]["test_report_only"] is True
    assert len(rows) == 32
    assert rows[0]["selection_metric_source"] == "val_metric"
    assert rows[0]["report_metric_source"] == "test_metric"
    assert rows[0]["preset_name"] in {"mechanism_smoke", "baseline_comparable"}
    assert rows[0]["model_variant"] in {"tf1_mlp_core", "tf1_mlp_aug"}
    assert "winner_by_val_transported_final_energy" in summary
    assert "winner_by_val_accuracy" in summary
    assert "winner_by_test_accuracy" in summary
    assert "winner_by_val_accuracy_among_gate_passing" in summary
    assert "winner_by_test_accuracy_among_gate_passing" in summary
    assert "val_energy_val_accuracy_pearson_correlation" in summary
    assert "val_energy_delta_vs_identity_val_accuracy_pearson_correlation" in summary
    assert "val_energy_delta_vs_local_field_only_val_accuracy_pearson_correlation" in summary
    assert "smaller_val_transported_energy_predictive_of_better_val_accuracy" in summary
    assert "by_preset" in summary
    assert set(summary["by_preset"].keys()) == {"mechanism_smoke", "baseline_comparable"}
    assert "winner_by_val_accuracy_among_gate_passing" in summary["by_preset"]["mechanism_smoke"]


def test_fmpc_tf1_suite_aggregate_manifest_is_deterministic(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("pc.fmpc_tf1_suite.run_fmpc_tf1_experiment", _fake_tf1_run)

    first = run_fmpc_tf1_suite(FMPCTF1SuiteConfig(output_root=tmp_path / "first", run_id="deterministic"))
    second = run_fmpc_tf1_suite(FMPCTF1SuiteConfig(output_root=tmp_path / "second", run_id="deterministic"))

    first_summary = _read_json(first.run_dir / "aggregate_summary.json")
    second_summary = _read_json(second.run_dir / "aggregate_summary.json")
    first_rows = _read_csv(first.run_dir / "aggregate_runs.csv")
    second_rows = _read_csv(second.run_dir / "aggregate_runs.csv")

    assert first_summary == second_summary
    assert first_rows == second_rows


def test_fmpc_tf1_suite_entrypoint_is_available(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("pc.fmpc_tf1_suite.run_fmpc_tf1_experiment", _fake_tf1_run)
    module = runpy.run_path(str(ROOT / "experiments" / "fmpc_tf1_suite.py"))
    run = module["run"]

    result = run(output_root=tmp_path, run_id="tf1_suite_entrypoint")
    assert (result.run_dir / "aggregate_summary.json").exists()
