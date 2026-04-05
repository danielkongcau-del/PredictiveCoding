from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_run():
    module = runpy.run_path(str(ROOT / "experiments" / "fmpc_tf1.py"))
    return module["run"]


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_fmpc_tf1_smoke_run_writes_expected_teacher_free_artifacts(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="fmpc_tf1_smoke",
        preset_name="mechanism_smoke",
        model_variant="tf1_mlp_core",
        use_teacher_free_features=False,
        feature_aware_tangents=False,
        transport_steps=2,
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "epoch_metrics.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "selection_diagnostics.json").exists()

    config = _read_json(run_dir / "config.json")
    summary = _read_json(run_dir / "summary.json")
    diagnostics = _read_json(run_dir / "selection_diagnostics.json")
    epoch_rows = _read_csv(run_dir / "epoch_metrics.csv")

    assert summary["phase"] == "Phase TF1"
    assert summary["stage"] == "teacher_free_fmpc_v1"
    assert summary["teacher_free"] is True
    assert summary["uses_teacher_artifacts"] is False
    assert summary["transport_scope"] == "train_only"
    assert summary["preset_name"] == "mechanism_smoke"
    assert summary["model_variant"] == "tf1_mlp_core"
    assert summary["use_teacher_free_features"] is False
    assert summary["feature_aware_tangents"] is False
    assert summary["transport_steps"] == 2
    assert summary["selection_metric"] == "val_transported_final_energy"
    assert summary["checkpoint_selector"] == "energy_only"
    assert summary["selection_metric_source"] == "val_metric"
    assert summary["report_metric_source"] == "test_metric"
    assert "selected_epoch_passes_gate" in summary
    assert "gate_passing_epoch_count" in summary
    assert "selector_fallback_used" in summary
    assert "selected_epoch_selection_reason" in summary
    assert summary["validation_gate"]["validation_only_gating"] is True
    assert summary["validation_gate"]["test_is_report_only"] is True
    assert "identity_baseline" in summary
    assert "local_field_only_baseline" in summary
    assert summary["selection_diagnostics_artifact"] == "selection_diagnostics.json"
    assert summary["rollout_knots"] == [0.0, 0.5, 1.0]
    assert "val_energy_delta_vs_identity" in summary
    assert "val_energy_delta_vs_local_field_only" in summary
    assert summary["val_transported_final_energy"] < summary["identity_baseline"]["val_transported_final_energy"]
    assert summary["val_transported_final_energy"] <= summary["local_field_only_baseline"]["val_transported_final_energy"]
    assert summary["val_accuracy"] > summary["val_baseline_accuracy"]
    assert diagnostics["selection_rules"]["val_transported_final_energy"]["selected_epoch"] >= 1
    assert diagnostics["selection_rules"]["val_accuracy"]["selected_epoch"] >= 1
    assert diagnostics["selection_rules"]["val_energy_delta_vs_local_field_only"]["selected_epoch"] >= 1
    assert "accuracy_gap_best_energy_vs_best_accuracy" in diagnostics
    assert "significant_validation_accuracy_left_on_table" in diagnostics

    assert config["preset_name"] == "mechanism_smoke"
    assert config["transport"]["model_variant"] == "tf1_mlp_core"
    assert config["transport"]["use_teacher_free_features"] is False
    assert config["transport"]["feature_aware_tangents"] is False
    assert config["transport"]["transport_steps"] == 2
    assert config["transport"]["selection_metric"] == "val_transported_final_energy"
    assert config["transport"]["checkpoint_selector"] == "energy_only"
    assert config["transport"]["selection_metric_source"] == "val_metric"
    assert config["transport"]["report_metric_source"] == "test_metric"
    assert len(epoch_rows) == config["run"]["epochs"]
    assert "val_transported_final_energy" in epoch_rows[0]
    assert "val_identity_final_energy" in epoch_rows[0]
    assert "val_local_field_only_final_energy" in epoch_rows[0]
    assert "val_energy_delta_vs_identity" in epoch_rows[0]
    assert "val_energy_delta_vs_local_field_only" in epoch_rows[0]
