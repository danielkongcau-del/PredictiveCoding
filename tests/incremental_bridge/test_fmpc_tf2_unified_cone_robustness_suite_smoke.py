from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_case(
    suite_root: Path,
    *,
    case_name: str,
    run_dir_name: str,
    seed: int,
    selected_epoch: int,
    val_accuracy: float,
    test_accuracy: float,
    gate_count: int,
    selected_epoch_passes_gate: bool,
    selector_fallback_used: bool,
    val_report_output_mse: float,
    val_supervised_transport_output_mse: float,
    val_delta_h_rms_rowspace: float,
    val_delta_h_rowspace_fraction: float,
    runtime_proxy_seconds: float,
    epoch_rows: list[dict[str, object]],
) -> None:
    run_dir = suite_root / "tf2_runs" / "tf2" / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    _write_csv(run_dir / "epoch_metrics.csv", epoch_rows)
    aggregate_row = {
        "case_name": case_name,
        "seed": seed,
        "run_id": run_dir_name,
        "run_summary_path": f"tf2_runs/tf2/{run_dir_name}/summary.json",
        "selected_epoch": selected_epoch,
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "gate_passing_epoch_count": gate_count,
        "selected_epoch_passes_gate": str(selected_epoch_passes_gate),
        "selector_fallback_used": str(selector_fallback_used),
        "val_report_output_mse": val_report_output_mse,
        "val_supervised_transport_output_mse": val_supervised_transport_output_mse,
        "val_delta_h_rms_rowspace": val_delta_h_rms_rowspace,
        "val_delta_h_rowspace_fraction": val_delta_h_rowspace_fraction,
        "runtime_proxy_seconds": runtime_proxy_seconds,
    }
    aggregate_path = suite_root / "aggregate_runs.csv"
    if aggregate_path.exists():
        rows = _read_csv(aggregate_path)
        rows.append({key: str(value) for key, value in aggregate_row.items()})
        _write_csv(aggregate_path, rows)
    else:
        _write_csv(aggregate_path, [{key: str(value) for key, value in aggregate_row.items()}])


def load_run():
    module = runpy.run_path(str(ROOT / "experiments" / "incremental_bridge" / "fmpc_tf2_unified_cone_robustness_suite.py"))
    return module["run"]


def test_fmpc_tf2_unified_cone_robustness_suite_writes_expected_schema(tmp_path: Path) -> None:
    hard_shape_root = tmp_path / "shape_source"
    smooth_root = tmp_path / "smooth_source"

    base_epochs = [
        {
            "epoch": 1,
            "stage": "warmup",
            "val_accuracy": 0.20,
            "val_baseline_accuracy": 0.10,
            "val_transported_final_energy": 0.70,
            "val_identity_final_energy": 0.60,
            "val_local_field_only_final_energy": 0.59,
        },
        {
            "epoch": 2,
            "stage": "hybrid",
            "val_accuracy": 0.75,
            "val_baseline_accuracy": 0.10,
            "val_transported_final_energy": 0.30,
            "val_identity_final_energy": 0.31,
            "val_local_field_only_final_energy": 0.30,
        },
        {
            "epoch": 3,
            "stage": "hybrid",
            "val_accuracy": 0.73,
            "val_baseline_accuracy": 0.10,
            "val_transported_final_energy": 0.31,
            "val_identity_final_energy": 0.33,
            "val_local_field_only_final_energy": 0.32,
        },
    ]
    hard20_epochs = [
        {
            "epoch": 1,
            "stage": "warmup",
            "val_accuracy": 0.22,
            "val_baseline_accuracy": 0.10,
            "val_transported_final_energy": 0.72,
            "val_identity_final_energy": 0.60,
            "val_local_field_only_final_energy": 0.59,
        },
        {
            "epoch": 2,
            "stage": "hybrid",
            "val_accuracy": 0.78,
            "val_baseline_accuracy": 0.10,
            "val_transported_final_energy": 0.32,
            "val_identity_final_energy": 0.31,
            "val_local_field_only_final_energy": 0.30,
        },
        {
            "epoch": 3,
            "stage": "hybrid",
            "val_accuracy": 0.76,
            "val_baseline_accuracy": 0.10,
            "val_transported_final_energy": 0.31,
            "val_identity_final_energy": 0.30,
            "val_local_field_only_final_energy": 0.30,
        },
    ]
    smooth_epochs = [
        {
            "epoch": 1,
            "stage": "warmup",
            "val_accuracy": 0.21,
            "val_baseline_accuracy": 0.10,
            "val_transported_final_energy": 0.71,
            "val_identity_final_energy": 0.60,
            "val_local_field_only_final_energy": 0.59,
        },
        {
            "epoch": 2,
            "stage": "hybrid",
            "val_accuracy": 0.77,
            "val_baseline_accuracy": 0.10,
            "val_transported_final_energy": 0.305,
            "val_identity_final_energy": 0.31,
            "val_local_field_only_final_energy": 0.30,
        },
        {
            "epoch": 3,
            "stage": "hybrid",
            "val_accuracy": 0.74,
            "val_baseline_accuracy": 0.10,
            "val_transported_final_energy": 0.315,
            "val_identity_final_energy": 0.325,
            "val_local_field_only_final_energy": 0.32,
        },
    ]

    _write_case(
        smooth_root,
        case_name="adopted_control",
        run_dir_name="control30_s0",
        seed=0,
        selected_epoch=2,
        val_accuracy=0.75,
        test_accuracy=0.74,
        gate_count=2,
        selected_epoch_passes_gate=True,
        selector_fallback_used=False,
        val_report_output_mse=0.06,
        val_supervised_transport_output_mse=0.05,
        val_delta_h_rms_rowspace=0.15,
        val_delta_h_rowspace_fraction=0.54,
        runtime_proxy_seconds=1.0,
        epoch_rows=base_epochs,
    )
    _write_case(
        hard_shape_root,
        case_name="unified_cone_interior_margin_20",
        run_dir_name="interior20_s0",
        seed=0,
        selected_epoch=2,
        val_accuracy=0.78,
        test_accuracy=0.77,
        gate_count=0,
        selected_epoch_passes_gate=False,
        selector_fallback_used=True,
        val_report_output_mse=0.059,
        val_supervised_transport_output_mse=0.049,
        val_delta_h_rms_rowspace=0.149,
        val_delta_h_rowspace_fraction=0.53,
        runtime_proxy_seconds=1.1,
        epoch_rows=hard20_epochs,
    )
    _write_case(
        smooth_root,
        case_name="smooth_unified_cone_projection_30",
        run_dir_name="smooth30_s0",
        seed=0,
        selected_epoch=2,
        val_accuracy=0.77,
        test_accuracy=0.76,
        gate_count=1,
        selected_epoch_passes_gate=True,
        selector_fallback_used=False,
        val_report_output_mse=0.058,
        val_supervised_transport_output_mse=0.048,
        val_delta_h_rms_rowspace=0.148,
        val_delta_h_rowspace_fraction=0.535,
        runtime_proxy_seconds=1.2,
        epoch_rows=smooth_epochs,
    )

    result = load_run()(
        output_root=tmp_path,
        run_id="tf2_unified_cone_robustness_smoke",
        hard_shape_source_root=hard_shape_root,
        smooth_source_root=smooth_root,
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "epoch_robustness_diagnostics.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()

    rows = _read_csv(run_dir / "aggregate_runs.csv")
    epoch_rows = _read_csv(run_dir / "epoch_robustness_diagnostics.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")

    assert len(rows) == 3
    assert len(epoch_rows) == 9
    assert summary["stage"] == "adopted_package_unified_cone_robustness_diagnostic"
    assert "by_candidate" in summary
    assert "pairwise_vs_control" in summary
    assert "diagnosis" in summary
    assert "decision" in summary
    assert "recommended_next_move" in summary
    assert "epoch_robustness_csv_path" in summary

    first_row = rows[0]
    assert "seed_gate_positive" in first_row
    assert "val_accuracy_margin_mean" in first_row
    assert "gate_energy_margin_mean" in first_row
    assert "val_accuracy_volatility" in first_row
    assert "gate_failure_energy_only_fraction" in first_row
    assert "per_epoch_gate_pass_sequence" in first_row

    first_epoch_row = epoch_rows[0]
    assert "gate_pass" in first_epoch_row
    assert "gate_failure_type" in first_epoch_row
    assert "val_accuracy_margin" in first_epoch_row
    assert "gate_energy_margin" in first_epoch_row
    assert "selector_score" in first_epoch_row
    assert "selected_epoch_flag" in first_epoch_row
