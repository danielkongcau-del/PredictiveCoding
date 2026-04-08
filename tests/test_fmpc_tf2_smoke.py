from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path

from pc.fmpc_tf2 import (
    build_tf2_canonical_config,
    build_tf2_corrective_transport_default_config,
    build_tf2_corrective_transport_terminal_angleclip_default_config,
    build_tf2_preset_config,
)


ROOT = Path(__file__).resolve().parents[1]


def load_run():
    module = runpy.run_path(str(ROOT / "experiments" / "fmpc_tf2.py"))
    return module["run"]


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_fmpc_tf2_smoke_run_writes_expected_artifacts(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="fmpc_tf2_smoke",
        preset_name="tf2_canonical",
        layer_dims=(64, 16, 10),
        epochs=3,
        batch_size=64,
        eval_steps=5,
        micro_steps=2,
        incremental_weight_updates=True,
        supervision_policy="mixed",
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "epoch_metrics.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "selection_diagnostics.json").exists()

    config = _read_json(run_dir / "config.json")
    summary = _read_json(run_dir / "summary.json")
    epoch_rows = _read_csv(run_dir / "epoch_metrics.csv")

    assert summary["phase"] == "Phase TF2"
    assert summary["stage"] == "ifmpc_bridge_stage"
    assert summary["teacher_free"] is True
    assert summary["uses_teacher_artifacts"] is False
    assert summary["jpc_runtime_dependency"] is False
    assert summary["preset_name"] == "tf2_canonical"
    assert summary["family_lineage"] == "tf1_mlp_aug"
    assert summary["incremental_weight_updates"] is True
    assert summary["supervision_policy"] == "mixed"
    assert summary["micro_steps"] == 2
    assert summary["theta_update_budget"] == "matched"
    assert summary["identity_tangent_mode"] == "feature_frozen_truncated_identity_approx"
    assert summary["psi_family"] == "baseline_plain"
    assert summary["time_encoding_variant"] == "raw"
    assert summary["terminal_local_field_direction_intervention"] == "none"
    assert summary["terminal_local_field_angle_clip_degrees"] == 30.0
    assert summary["terminal_local_field_rowspace_angle_clip_degrees"] == 30.0
    assert summary["terminal_local_field_orthogonal_angle_clip_degrees"] == 30.0
    assert summary["transported_output_alignment_weight"] == 0.0
    assert summary["transported_output_alignment_schedule"] == "none"
    assert summary["checkpoint_selector"] == "gate_constrained_accuracy_then_val_accuracy"
    assert summary["selection_metric_source"] == "val_metric"
    assert summary["report_metric_source"] == "test_metric"
    assert summary["theta_micro_lr"] > 0.0
    assert summary["theta_micro_bias_lr"] > 0.0
    assert "forward_init_stability_metrics" in summary
    assert "hidden_layer_stats" in summary["forward_init_stability_metrics"]
    assert "initial_target_clamped_energy" in summary["forward_init_stability_metrics"]
    assert "initial_hidden_gradient_rms" in summary["forward_init_stability_metrics"]
    assert "selected_epoch_passes_gate" in summary
    assert "gate_passing_epoch_count" in summary
    assert "selector_fallback_used" in summary
    assert "selected_epoch_selection_reason" in summary
    assert summary["validation_gate"]["validation_only_gating"] is True
    assert summary["validation_gate"]["test_is_report_only"] is True
    assert len(epoch_rows) == config["run"]["epochs"]
    assert config["preset_name"] == "tf2_canonical"
    assert config["transport"]["identity_tangent_mode"] == "feature_frozen_truncated_identity_approx"
    assert config["transport"]["psi_family"] == "baseline_plain"
    assert config["transport"]["time_encoding_variant"] == "raw"
    assert config["transport"]["terminal_local_field_direction_intervention"] == "none"
    assert config["transport"]["terminal_local_field_angle_clip_degrees"] == 30.0
    assert config["transport"]["terminal_local_field_rowspace_angle_clip_degrees"] == 30.0
    assert config["transport"]["terminal_local_field_orthogonal_angle_clip_degrees"] == 30.0
    assert config["transport"]["transported_output_alignment_weight"] == 0.0
    assert config["transport"]["transported_output_alignment_schedule"] == "none"
    assert "val_transported_final_energy" in epoch_rows[0]
    assert "val_local_field_only_final_energy" in epoch_rows[0]


def test_fmpc_tf2_entrypoint_defaults_to_adopted_terminal_angleclip_preset(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="fmpc_tf2_default_entrypoint_smoke",
        epochs=2,
        batch_size=64,
        eval_steps=5,
        layer_dims=(64, 16, 10),
    )

    config = _read_json(result.run_dir / "config.json")
    summary = _read_json(result.run_dir / "summary.json")

    assert config["preset_name"] == "tf2_corrective_transport_terminal_angleclip_default"
    assert summary["preset_name"] == "tf2_corrective_transport_terminal_angleclip_default"
    assert summary["psi_family"] == "residualized_local_field"
    assert summary["time_encoding_variant"] == "poly_rt2"
    assert (
        summary["terminal_local_field_direction_intervention"]
        == "local_field_direction_angle_clip_keep_live_norm"
    )
    assert summary["terminal_local_field_angle_clip_degrees"] == 30.0
    assert summary["terminal_local_field_rowspace_angle_clip_degrees"] == 30.0
    assert summary["terminal_local_field_orthogonal_angle_clip_degrees"] == 30.0
    assert summary["transported_output_alignment_weight"] == 0.0
    assert summary["transported_output_alignment_schedule"] == "none"


def test_tf2_preset_builders_keep_canonical_and_expose_corrective_default() -> None:
    canonical = build_tf2_canonical_config()
    canonical_via_dispatch = build_tf2_preset_config("tf2_canonical")
    corrective = build_tf2_corrective_transport_default_config()
    corrective_via_dispatch = build_tf2_preset_config("tf2_corrective_transport_default")
    corrective_angleclip = build_tf2_corrective_transport_terminal_angleclip_default_config()
    corrective_angleclip_via_dispatch = build_tf2_preset_config(
        "tf2_corrective_transport_terminal_angleclip_default"
    )

    assert canonical.preset_name == "tf2_canonical"
    assert canonical_via_dispatch.preset_name == "tf2_canonical"
    assert canonical.incremental_weight_updates is True
    assert canonical.supervision_policy == "mixed"
    assert canonical.micro_steps == 4
    assert canonical.feature_aware_tangents is False
    assert canonical.theta_update_budget == "matched"

    assert corrective.preset_name == "tf2_corrective_transport_default"
    assert corrective_via_dispatch.preset_name == "tf2_corrective_transport_default"
    assert corrective.incremental_weight_updates is False
    assert corrective.supervision_policy == "local_only"
    assert corrective.micro_steps == 4
    assert corrective.feature_aware_tangents is False
    assert corrective.theta_update_budget == "matched"
    assert corrective.terminal_local_field_direction_intervention == "none"
    assert corrective.terminal_local_field_rowspace_angle_clip_degrees == 30.0
    assert corrective.terminal_local_field_orthogonal_angle_clip_degrees == 30.0

    assert corrective_angleclip.preset_name == "tf2_corrective_transport_terminal_angleclip_default"
    assert (
        corrective_angleclip_via_dispatch.preset_name
        == "tf2_corrective_transport_terminal_angleclip_default"
    )
    assert corrective_angleclip.incremental_weight_updates is False
    assert corrective_angleclip.supervision_policy == "local_only"
    assert corrective_angleclip.micro_steps == 4
    assert corrective_angleclip.feature_aware_tangents is False
    assert corrective_angleclip.theta_update_budget == "matched"
    assert corrective_angleclip.psi_family == "residualized_local_field"
    assert corrective_angleclip.time_encoding_variant == "poly_rt2"
    assert (
        corrective_angleclip.terminal_local_field_direction_intervention
        == "local_field_direction_angle_clip_keep_live_norm"
    )
    assert corrective_angleclip.terminal_local_field_angle_clip_degrees == 30.0
    assert corrective_angleclip.terminal_local_field_rowspace_angle_clip_degrees == 30.0
    assert corrective_angleclip.terminal_local_field_orthogonal_angle_clip_degrees == 30.0
    assert corrective_angleclip.transported_output_alignment_weight == 0.0
    assert corrective_angleclip.transported_output_alignment_schedule == "none"
