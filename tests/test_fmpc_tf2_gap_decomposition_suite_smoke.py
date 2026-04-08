from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_run():
    module = runpy.run_path(str(ROOT / "experiments" / "fmpc_tf2_gap_decomposition_suite.py"))
    return module["run"]


def test_fmpc_tf2_gap_decomposition_suite_writes_expected_schema(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="tf2_gap_decomposition_smoke",
        seeds=(0,),
        tf2_epochs=2,
        tf2_batch_size=64,
        tf2_eval_steps=5,
        tf2_layer_dims=(64, 16, 10),
        slow_pc_epochs=2,
        slow_pc_batch_size=64,
        slow_pc_train_steps=5,
        slow_pc_eval_steps=5,
        slow_pc_layer_dims=(64, 16, 10),
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()

    rows = _read_csv(run_dir / "aggregate_runs.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")

    assert len(rows) == 3
    assert "mean_std_val_accuracy_by_method" in summary
    assert "mean_std_test_accuracy_by_method" in summary
    assert "mean_report_output_mse_by_method" in summary
    assert "mean_supervised_final_energy_by_method" in summary
    assert "mean_endpoint_hidden_state_rms_gap_to_internal_slow_pc_by_method" in summary
    assert "mean_endpoint_output_state_rms_gap_to_internal_slow_pc_by_method" in summary
    assert "pairwise_adopted_default_vs_canonical_slow_pc_digits_baseline" in summary
    assert "adopted_package_validation_knot_breakdown" in summary
    assert "remaining_gap_primary_diagnosis" in summary
    assert "recommended_next_narrow_tf2_move" in summary

    adopted = summary["by_method"]["tf2_corrective_transport_terminal_angleclip_default"]
    assert "mean_val_transport_minus_internal_slow_pc_output_mse" in adopted

    first_row = rows[0]
    assert "method_name" in first_row
    assert "selected_epoch" in first_row
    assert "val_endpoint_hidden_state_rms_gap_to_internal_slow_pc" in first_row
