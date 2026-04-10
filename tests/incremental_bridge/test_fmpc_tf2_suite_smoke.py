from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from pc.incremental_bridge.fmpc_tf2_suite import FMPCTF2SuiteConfig, run_fmpc_tf2_suite


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_fmpc_tf2_suite_smoke_writes_expected_aggregate_schema(tmp_path: Path) -> None:
    tf1_ref = tmp_path / "refs" / "tf1.json"
    slow_ref = tmp_path / "refs" / "slow.json"
    _write_json(
        tf1_ref,
        {
            "mean_std_val_accuracy_by_preset": {"baseline_working_default": {"mean": 0.60}},
            "mean_std_test_accuracy_by_preset": {"baseline_working_default": {"mean": 0.61}},
        },
    )
    _write_json(
        slow_ref,
        {
            "mean_std_val_accuracy_by_method": {"canonical_slow_pc_digits_baseline": {"mean": 0.80}},
            "mean_std_test_accuracy_by_method": {"canonical_slow_pc_digits_baseline": {"mean": 0.81}},
        },
    )

    result = run_fmpc_tf2_suite(
        FMPCTF2SuiteConfig(
            output_root=tmp_path,
            run_id="tf2_suite_smoke",
            epochs=2,
            batch_size=64,
            eval_steps=5,
            layer_dims=(64, 16, 10),
            incremental_weight_updates_options=(False, True),
            supervision_policies=("local_only", "mixed"),
            micro_steps_options=(2,),
            seeds=(0,),
            sealed_tf1_summary_path=tf1_ref,
            slow_pc_summary_path=slow_ref,
        )
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()

    rows = _read_csv(run_dir / "aggregate_runs.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")
    assert len(rows) == 4
    assert "mean_std_val_accuracy_by_config" in summary
    assert "mean_std_test_accuracy_by_config" in summary
    assert "mean_gate_passing_epoch_count_by_config" in summary
    assert "pairwise_comparison_against_sealed_tf1_working_default" in summary
    assert "pairwise_gap_to_canonical_slow_pc_digits_baseline" in summary
    assert "mixed_policy_supervision_helps" in summary
    assert "incremental_theta_updates_help" in summary
    assert "matched_budget_tf2_narrows_slow_pc_gap_materially" in summary
    assert "recommended_next_stage" in summary


def test_fmpc_tf2_suite_requires_reference_artifacts(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_fmpc_tf2_suite(
            FMPCTF2SuiteConfig(
                output_root=tmp_path,
                run_id="tf2_suite_missing_refs",
                seeds=(0,),
                micro_steps_options=(2,),
                sealed_tf1_summary_path=tmp_path / "missing_tf1.json",
                slow_pc_summary_path=tmp_path / "missing_slow.json",
            )
        )
