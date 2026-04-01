from __future__ import annotations

import csv
import json
from pathlib import Path

from pc.pc_multiseed import run_pc_multiseed_study


def _write_phase2g1_best_config_summaries(tmp_path: Path, benchmark_name: str) -> None:
    best_config_dir = tmp_path / "phase2g1_boundary_check" / benchmark_name
    best_config_dir.mkdir(parents=True, exist_ok=True)
    pc_payload = {
        "run_id": "phase2g1_fixture",
        "boundary_check_best_config_id": "cfg_pc_refined",
        "boundary_check_val_metric": 0.01,
        "boundary_check_best_config": {
            "eta_x": 0.2,
            "eta_w": 0.4,
            "eta_b": 0.4,
            "train_steps": 10,
            "eval_steps": 10,
            "epochs": 120,
            "state_init": "forward",
        },
    }
    mlp_payload = {
        "run_id": "phase2g1_fixture",
        "boundary_check_best_config_id": "cfg_mlp_refined",
        "boundary_check_val_metric": 0.02,
        "boundary_check_best_config": {
            "eta_w": 0.2,
            "eta_b": 0.2,
            "epochs": 160,
        },
    }
    with (best_config_dir / "best_pc_config_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(pc_payload, handle, indent=2)
    with (best_config_dir / "best_mlp_config_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(mlp_payload, handle, indent=2)


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_pc_multiseed_is_deterministic_with_same_explicit_seeds(tmp_path: Path) -> None:
    _write_phase2g1_best_config_summaries(tmp_path, "toy_regression")
    root_a = tmp_path / "run_a"
    root_b = tmp_path / "run_b"
    result_a = run_pc_multiseed_study(
        "toy_regression",
        output_root=root_a,
        run_id="deterministic_multiseed",
        plot_energy=False,
        plot_summary=False,
        seed_values=[0, 1],
        joint_search_output_root=tmp_path,
    )
    result_b = run_pc_multiseed_study(
        "toy_regression",
        output_root=root_b,
        run_id="deterministic_multiseed",
        plot_energy=False,
        plot_summary=False,
        seed_values=[0, 1],
        joint_search_output_root=tmp_path,
    )

    assert _load_csv_rows(result_a.run_dir / "seed_records.csv") == _load_csv_rows(
        result_b.run_dir / "seed_records.csv"
    )
    assert _load_json(result_a.run_dir / "aggregate_summary.json") == _load_json(
        result_b.run_dir / "aggregate_summary.json"
    )
