from __future__ import annotations

import csv
import json
from pathlib import Path

from pc.pc_budget_tradeoff import run_pc_budget_tradeoff_study


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_pc_budget_tradeoff_is_deterministic_with_same_explicit_seeds(tmp_path: Path) -> None:
    best_config_dir = tmp_path / "phase2g1_boundary_check" / "toy_regression"
    best_config_dir.mkdir(parents=True, exist_ok=True)
    with (best_config_dir / "best_pc_config_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
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
            },
            handle,
            indent=2,
        )
    with (best_config_dir / "best_mlp_config_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_id": "phase2g1_fixture",
                "boundary_check_best_config_id": "cfg_mlp_refined",
                "boundary_check_val_metric": 0.02,
                "boundary_check_best_config": {
                    "eta_w": 0.2,
                    "eta_b": 0.2,
                    "epochs": 160,
                },
            },
            handle,
            indent=2,
        )
    root_a = tmp_path / "run_a"
    root_b = tmp_path / "run_b"
    seed_values = [0, 1]

    result_a = run_pc_budget_tradeoff_study(
        "toy_regression",
        output_root=root_a,
        run_id="toy_regression_budget_det",
        plot_energy=False,
        plot_summary=False,
        seed_values=seed_values,
        joint_search_output_root=tmp_path,
    )
    result_b = run_pc_budget_tradeoff_study(
        "toy_regression",
        output_root=root_b,
        run_id="toy_regression_budget_det",
        plot_energy=False,
        plot_summary=False,
        seed_values=seed_values,
        joint_search_output_root=tmp_path,
    )

    assert _load_csv(result_a.run_dir / "seed_budget_records.csv") == _load_csv(
        result_b.run_dir / "seed_budget_records.csv"
    )
    assert _load_csv(result_a.run_dir / "budget_summary.csv") == _load_csv(
        result_b.run_dir / "budget_summary.csv"
    )
    assert _load_json(result_a.run_dir / "aggregate_summary.json") == _load_json(
        result_b.run_dir / "aggregate_summary.json"
    )
