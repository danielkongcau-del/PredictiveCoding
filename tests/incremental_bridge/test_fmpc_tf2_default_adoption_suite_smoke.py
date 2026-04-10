from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("rows must contain at least one item.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_run():
    module = runpy.run_path(str(ROOT / "experiments" / "incremental_bridge" / "fmpc_tf2_default_adoption_suite.py"))
    return module["run"]


def test_fmpc_tf2_default_adoption_suite_writes_expected_schema(tmp_path: Path) -> None:
    tf1_summary = tmp_path / "refs" / "tf1_summary.json"
    tf1_runs = tmp_path / "refs" / "tf1_runs.csv"
    slow_summary = tmp_path / "refs" / "slow_summary.json"
    slow_runs = tmp_path / "refs" / "slow_runs.csv"
    jpc_probe_summary = tmp_path / "refs" / "jpc_probe_summary.json"

    _write_json(
        tf1_summary,
        {
            "mean_std_val_accuracy_by_preset": {"baseline_working_default": {"mean": 0.68}},
            "mean_std_test_accuracy_by_preset": {"baseline_working_default": {"mean": 0.67}},
        },
    )
    _write_csv(
        tf1_runs,
        [
            {
                "preset_name": "baseline_working_default",
                "seed": 0,
                "checkpoint_selector": "gate_constrained_accuracy_then_val_accuracy",
                "selected_epoch": 10,
                "selected_epoch_passes_gate": "True",
                "gate_passing_epoch_count": 3,
                "selector_fallback_used": "False",
                "val_accuracy": 0.68,
                "test_accuracy": 0.67,
                "val_transported_final_energy": 0.4,
                "run_summary_path": "runs/baseline_working_default_seed0/summary.json",
            }
        ],
    )
    _write_json(
        slow_summary,
        {
            "mean_std_val_accuracy_by_method": {"canonical_slow_pc_digits_baseline": {"mean": 0.88}},
            "mean_std_test_accuracy_by_method": {"canonical_slow_pc_digits_baseline": {"mean": 0.89}},
        },
    )
    _write_csv(
        slow_runs,
        [
            {
                "method_name": "canonical_slow_pc_digits_baseline",
                "seed": 0,
                "checkpoint_selector": "",
                "val_accuracy": 0.88,
                "test_accuracy": 0.89,
                "val_transported_final_energy": "",
                "run_summary_path": "runs/canonical_slow_pc_digits_baseline_seed0/summary.json",
            }
        ],
    )
    _write_json(
        jpc_probe_summary,
        {
            "recommended_tf2_emphasis": "incremental scheduling",
            "whether_mupc_like_scaling_appears_to_improve_forward_init_stability": False,
            "whether_many_step_inference_still_materially_outperforms_1_step_in_jpc_standard_pc": True,
        },
    )

    result = load_run()(
        output_root=tmp_path,
        run_id="tf2_default_adoption_smoke",
        epochs=2,
        batch_size=64,
        eval_steps=5,
        layer_dims=(64, 16, 10),
        seeds=(0,),
        tf1_reference_summary_path=tf1_summary,
        tf1_reference_runs_path=tf1_runs,
        slow_pc_reference_summary_path=slow_summary,
        slow_pc_reference_runs_path=slow_runs,
        jpc_probe_summary_path=jpc_probe_summary,
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()

    rows = _read_csv(run_dir / "aggregate_runs.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")
    assert len(rows) == 4
    assert "mean_std_val_accuracy_by_preset" in summary
    assert "mean_std_test_accuracy_by_preset" in summary
    assert "mean_gate_passing_epoch_count_by_preset" in summary
    assert "pairwise_tf2_corrective_transport_default_vs_tf2_canonical" in summary
    assert "pairwise_tf2_corrective_transport_default_vs_sealed_tf1_working_default" in summary
    assert "pairwise_tf2_corrective_transport_default_vs_canonical_slow_pc_digits_baseline" in summary
    assert "tf2_corrective_transport_default_should_become_main_tf2_preset" in summary
    assert summary["current_tf2_evidence_interpretation"] in {
        "corrective transport bridge",
        "full incremental iFMPC",
        "both",
        "neither",
    }
    assert summary["jpc_probe_reference"]["recommended_tf2_emphasis"] == "incremental scheduling"
    assert summary["recommended_main_tf2_preset_after_adoption_validation"]["preset_name"] in {
        "tf2_canonical",
        "tf2_corrective_transport_default",
    }
