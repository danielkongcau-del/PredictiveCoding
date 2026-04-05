from __future__ import annotations

from pc.fmpc_tf1 import (
    build_tf1_baseline_comparable_config,
    build_tf1_mechanism_smoke_config,
    build_tf1_baseline_working_default_config,
    build_tf1_epoch_selection_diagnostics,
    build_tf1_preset_config,
    _select_tf1_checkpoint_epoch,
)


def _epoch_row(
    epoch: int,
    *,
    val_energy: float,
    val_identity: float,
    val_local: float,
    val_accuracy: float,
    val_baseline_accuracy: float = 0.1,
) -> dict[str, float | int]:
    return {
        "epoch": int(epoch),
        "val_transported_final_energy": float(val_energy),
        "val_identity_final_energy": float(val_identity),
        "val_local_field_only_final_energy": float(val_local),
        "val_energy_delta_vs_identity": float(val_energy - val_identity),
        "val_energy_delta_vs_local_field_only": float(val_energy - val_local),
        "val_accuracy": float(val_accuracy),
        "val_baseline_accuracy": float(val_baseline_accuracy),
    }


def test_tf1_old_presets_remain_unchanged() -> None:
    smoke = build_tf1_mechanism_smoke_config()
    baseline = build_tf1_baseline_comparable_config()

    assert smoke.preset_name == "mechanism_smoke"
    assert smoke.model_variant == "tf1_mlp_core"
    assert smoke.use_teacher_free_features is False
    assert smoke.transport_steps == 2
    assert smoke.checkpoint_selector == "energy_only"

    assert baseline.preset_name == "baseline_comparable"
    assert baseline.model_variant == "tf1_mlp_core"
    assert baseline.use_teacher_free_features is False
    assert baseline.transport_steps == 2
    assert baseline.checkpoint_selector == "energy_only"


def test_tf1_working_default_preset_builder_matches_current_contract() -> None:
    working = build_tf1_baseline_working_default_config()
    via_dispatch = build_tf1_preset_config("baseline_working_default")

    assert working == via_dispatch
    assert working.preset_name == "baseline_working_default"
    assert working.model_variant == "tf1_mlp_aug"
    assert working.use_teacher_free_features is True
    assert working.transport_steps == 1
    assert working.warmup_epochs == 5
    assert working.feature_aware_tangents is False
    assert working.identity_loss_weight == 0.2
    assert working.checkpoint_selector == "gate_constrained_accuracy_then_val_accuracy"


def test_tf1_checkpoint_selector_logic_covers_gate_and_fallback_cases() -> None:
    epoch_rows = [
        _epoch_row(1, val_energy=0.30, val_identity=0.32, val_local=0.31, val_accuracy=0.40),
        _epoch_row(2, val_energy=0.35, val_identity=0.37, val_local=0.36, val_accuracy=0.70),
        _epoch_row(3, val_energy=0.29, val_identity=0.28, val_local=0.27, val_accuracy=0.80),
    ]
    diagnostics = build_tf1_epoch_selection_diagnostics(epoch_rows)

    energy_only = _select_tf1_checkpoint_epoch(
        epoch_rows,
        "energy_only",
        selection_diagnostics=diagnostics,
    )
    val_accuracy_only = _select_tf1_checkpoint_epoch(
        epoch_rows,
        "val_accuracy_only",
        selection_diagnostics=diagnostics,
    )
    gate_then_energy = _select_tf1_checkpoint_epoch(
        epoch_rows,
        "gate_constrained_accuracy_then_energy",
        selection_diagnostics=diagnostics,
    )
    gate_then_valacc = _select_tf1_checkpoint_epoch(
        epoch_rows,
        "gate_constrained_accuracy_then_val_accuracy",
        selection_diagnostics=diagnostics,
    )

    assert energy_only["selected_epoch"] == 3
    assert energy_only["selected_epoch_passes_gate"] is False
    assert val_accuracy_only["selected_epoch"] == 3
    assert gate_then_energy["selected_epoch"] == 2
    assert gate_then_energy["selected_epoch_passes_gate"] is True
    assert gate_then_energy["selector_fallback_used"] is False
    assert gate_then_valacc["selected_epoch"] == 2
    assert gate_then_valacc["selected_epoch_passes_gate"] is True

    no_gate_rows = [
        _epoch_row(1, val_energy=0.30, val_identity=0.29, val_local=0.28, val_accuracy=0.40),
        _epoch_row(2, val_energy=0.31, val_identity=0.30, val_local=0.29, val_accuracy=0.75),
    ]
    no_gate_diagnostics = build_tf1_epoch_selection_diagnostics(no_gate_rows)

    no_gate_then_energy = _select_tf1_checkpoint_epoch(
        no_gate_rows,
        "gate_constrained_accuracy_then_energy",
        selection_diagnostics=no_gate_diagnostics,
    )
    no_gate_then_valacc = _select_tf1_checkpoint_epoch(
        no_gate_rows,
        "gate_constrained_accuracy_then_val_accuracy",
        selection_diagnostics=no_gate_diagnostics,
    )

    assert no_gate_then_energy["selected_epoch"] == 1
    assert no_gate_then_energy["selector_fallback_used"] is True
    assert no_gate_then_energy["selected_epoch_passes_gate"] is False
    assert no_gate_then_valacc["selected_epoch"] == 2
    assert no_gate_then_valacc["selector_fallback_used"] is True
    assert no_gate_then_valacc["selected_epoch_passes_gate"] is False
