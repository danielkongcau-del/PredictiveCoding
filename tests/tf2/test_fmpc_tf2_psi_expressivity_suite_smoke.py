from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path

import numpy as np

from pc.tf2.fmpc_tf2 import _build_psi_input_tangent, _psi_input_dim, build_tf2_corrective_transport_default_config


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
    module = runpy.run_path(str(ROOT / "experiments" / "tf2" / "fmpc_tf2_psi_expressivity_suite.py"))
    return module["run"]


def test_poly_time_encoding_tangent_respects_chain_rule_without_unfreezing_feature_block() -> None:
    raw_config = build_tf2_corrective_transport_default_config(layer_dims=(64, 16, 10))
    poly_config = build_tf2_corrective_transport_default_config(
        layer_dims=(64, 16, 10),
        time_encoding_variant="poly_rt2",
    )
    g_t = np.ones((1, 16), dtype=np.float64)

    tangent = _build_psi_input_tangent(
        poly_config,
        g_t,
        target_dim=10,
        t=0.25,
        r=0.75,
    )

    assert _psi_input_dim(poly_config) == _psi_input_dim(raw_config) + 3
    time_offset = 16 + 10
    assert np.allclose(tangent[:, time_offset : time_offset + 5], [[1.0, -1.0, 0.5, 0.5, -1.5]])
    assert np.allclose(tangent[:, time_offset + 5 :], 0.0)


def test_fmpc_tf2_psi_expressivity_suite_writes_expected_schema(tmp_path: Path) -> None:
    slow_runs = tmp_path / "refs" / "slow_runs.csv"
    _write_csv(
        slow_runs,
        [
            {
                "preset_name": "canonical_slow_pc_digits_baseline",
                "seed": 0,
                "val_accuracy": 0.86,
                "test_accuracy": 0.88,
            }
        ],
    )

    result = load_run()(
        output_root=tmp_path,
        run_id="tf2_psi_expressivity_smoke",
        offline_train_seeds=(0,),
        offline_val_seeds=(1,),
        end_to_end_seeds=(0,),
        epochs=2,
        batch_size=64,
        eval_steps=5,
        layer_dims=(64, 16, 10),
        offline_probe_epoch_indices=(0, 1),
        sample_batches_per_probe_epoch=1,
        offline_fit_epochs=4,
        slow_pc_reference_runs_path=slow_runs,
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "offline_train_samples.npz").exists()
    assert (run_dir / "offline_val_samples.npz").exists()
    assert (run_dir / "offline_family_metrics.csv").exists()
    assert (run_dir / "offline_encoding_metrics.csv").exists()
    assert (run_dir / "end_to_end_runs.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()

    family_rows = _read_csv(run_dir / "offline_family_metrics.csv")
    encoding_rows = _read_csv(run_dir / "offline_encoding_metrics.csv")
    end_to_end_rows = _read_csv(run_dir / "end_to_end_runs.csv")
    summary = _read_json(run_dir / "aggregate_summary.json")

    assert len(family_rows) == 3
    assert len(encoding_rows) == 2
    assert len(end_to_end_rows) >= 1
    assert "offline_dataset_artifacts" in summary
    assert "family_stage_offline_metrics" in summary
    assert "encoding_stage_offline_metrics" in summary
    assert "stage1_winner" in summary
    assert "selected_end_to_end_challengers" in summary
    assert "capacity_bottleneck_signal" in summary
    assert "input_representation_bottleneck_signal" in summary
    assert "output_parameterization_bottleneck_signal" in summary
    assert "mean_std_val_accuracy_by_candidate" in summary
    assert "mean_std_test_accuracy_by_candidate" in summary
    assert "mean_gate_passing_epoch_count_by_candidate" in summary
    assert "mean_val_transported_final_energy_by_candidate" in summary
    assert "mean_wall_clock_runtime_by_candidate" in summary
    assert "gap_to_canonical_slow_pc_by_candidate" in summary
    assert "pairwise_delta_vs_current_corrective_default" in summary
    assert "decision_logic_outcome" in summary
    assert "is_psi_expressivity_a_real_limiter" in summary
    assert "best_material_end_to_end_candidate" in summary
    assert "next_single_narrow_research_move" in summary

    first_family = family_rows[0]
    assert "candidate_key" in first_family
    assert "psi_family" in first_family
    assert "time_encoding_variant" in first_family
    assert "parameter_count" in first_family
    assert "val_bootstrap_target_mse" in first_family
    assert "val_identity_residual_error" in first_family
    assert "val_hybrid_loss" in first_family
    assert "runtime_per_batch_seconds" in first_family

    first_e2e = end_to_end_rows[0]
    assert "candidate_key" in first_e2e
    assert "psi_family" in first_e2e
    assert "time_encoding_variant" in first_e2e
    assert "val_accuracy" in first_e2e
    assert "test_accuracy" in first_e2e
    assert "total_wall_time_seconds" in first_e2e
    assert "run_status" in first_e2e
