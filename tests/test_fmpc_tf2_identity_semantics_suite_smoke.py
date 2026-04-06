from __future__ import annotations

import csv
import json
from pathlib import Path

from pc.fmpc_tf2_identity_semantics_suite import (
    FMPCTF2IdentitySemanticsSuiteConfig,
    run_fmpc_tf2_identity_semantics_suite,
)


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_fmpc_tf2_identity_semantics_suite_smoke_writes_expected_schema(tmp_path: Path) -> None:
    result = run_fmpc_tf2_identity_semantics_suite(
        FMPCTF2IdentitySemanticsSuiteConfig(
            output_root=tmp_path,
            run_id="tf2_identity_semantics_smoke",
            tf2_preset_names=("tf2_canonical", "tf2_corrective_transport_default"),
            feature_aware_tangent_options=(False, True),
            seeds=(0,),
            epochs=2,
            batch_size=64,
            eval_steps=5,
            layer_dims=(64, 16, 10),
        )
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()

    rows = _read_csv(run_dir / "aggregate_runs.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")
    assert len(rows) == 4
    assert "mean_std_val_accuracy_by_preset_and_identity_semantics" in summary
    assert "mean_std_test_accuracy_by_preset_and_identity_semantics" in summary
    assert "mean_val_transported_final_energy_by_preset_and_identity_semantics" in summary
    assert "mean_gate_passing_epoch_count_by_preset_and_identity_semantics" in summary
    assert "stability_by_preset_and_identity_semantics" in summary
    assert "pairwise_semantics_comparison_by_preset" in summary
    assert "feature_aware_tangents_should_become_canonical_tf2_default" in summary
    assert "recommended_canonical_tf2_identity_semantics" in summary
    assert "current_tf2_identity_evidence_interpretation" in summary
    assert "recommended_next_research_stage_after_identity_semantics_decision" in summary

    first_row = rows[0]
    assert "preset_name" in first_row
    assert "feature_aware_tangents" in first_row
    assert "identity_semantics" in first_row
    assert "identity_tangent_mode" in first_row
    assert "theta_update_cadence" in first_row
    assert "run_status" in first_row
    assert "nan_or_inf_failure" in first_row
