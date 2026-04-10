from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


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
    module = runpy.run_path(str(ROOT / "experiments" / "incremental_bridge" / "fmpc_tf2b_interleaving_suite.py"))
    return module["run"]


def test_fmpc_tf2b_interleaving_suite_writes_expected_schema(tmp_path: Path) -> None:
    slow_runs = tmp_path / "refs" / "slow_runs.csv"
    _write_csv(
        slow_runs,
        [
            {
                "method_name": "canonical_slow_pc_digits_baseline",
                "seed": 0,
                "val_accuracy": 0.86,
                "test_accuracy": 0.88,
            }
        ],
    )

    result = load_run()(
        output_root=tmp_path,
        run_id="tf2b_interleaving_smoke",
        epochs=2,
        batch_size=64,
        eval_steps=5,
        layer_dims=(64, 16, 10),
        theta_update_cadences=("terminal_only", "every_2_micro_steps"),
        onpolicy_mix_ratios=(0.0, 0.25),
        interleaving_start_options=("epoch_0",),
        seeds=(0,),
        slow_pc_reference_runs_path=slow_runs,
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "aggregate_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()

    rows = _read_csv(run_dir / "aggregate_runs.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")

    assert len(rows) == 4
    assert "mean_std_val_accuracy_by_configuration" in summary
    assert "mean_std_test_accuracy_by_configuration" in summary
    assert "mean_gate_passing_epoch_count_by_configuration" in summary
    assert "pairwise_comparison_against_current_tf2_corrective_transport_default" in summary
    assert "whether_gentle_interleaving_helps" in summary
    assert "whether_low_ratio_onpolicy_supervision_helps" in summary
    assert "whether_delayed_interleaving_helps" in summary
    assert "whether_any_configuration_narrows_the_slow_pc_test_gap_below_current_tf2_default" in summary
    assert summary["recommended_next_tf2_step"] in {
        "keep corrective transport default",
        "adopt softened interleaving",
        "adopt softened on-policy supervision",
        "both",
        "neither",
    }
    first_row = rows[0]
    assert "theta_update_cadence" in first_row
    assert "onpolicy_mix_ratio" in first_row
    assert "interleaving_start" in first_row
    assert "selected_epoch_passes_gate" in first_row
    assert "selector_fallback_used" in first_row
