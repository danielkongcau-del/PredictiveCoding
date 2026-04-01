from __future__ import annotations

import csv
import json
from pathlib import Path

from pc.phase2g1_boundary_check import run_phase2g1_boundary_check


def _write_phase2g_reference_artifacts(tmp_path: Path, benchmark_name: str) -> None:
    reference_dir = tmp_path / "phase2g_matched_search" / benchmark_name
    reference_dir.mkdir(parents=True, exist_ok=True)
    study_config = {
        "run_id": "phase2g_fixture",
        "benchmark_name": benchmark_name,
        "pc_search_space": {
            "eta_x": [0.05, 0.1],
            "eta_w": [0.2, 0.4],
            "train_steps": [25, 50],
            "epochs": [4, 6],
        },
        "mlp_search_space": {
            "eta_w": [0.1, 0.2],
            "epochs": [4, 6],
        },
    }
    aggregate_summary = {
        "run_id": "phase2g_fixture",
        "test_winner": "pc",
        "test_winner_reason": "predictive coding has lower mse on test",
    }
    best_pc_summary = {
        "run_id": "phase2g_fixture",
        "best_config_id": "cfg_pc",
        "selection_metric_value": 0.10,
        "val_metric": 0.10,
        "test_metric": 0.11,
        "best_config": {
            "eta_x": 0.05,
            "eta_w": 0.4,
            "eta_b": 0.4,
            "train_steps": 25,
            "eval_steps": 25,
            "epochs": 6,
            "state_init": "forward",
        },
    }
    best_mlp_summary = {
        "run_id": "phase2g_fixture",
        "best_config_id": "cfg_mlp",
        "selection_metric_value": 0.12,
        "val_metric": 0.12,
        "test_metric": 0.13,
        "best_config": {
            "eta_w": 0.2,
            "eta_b": 0.2,
            "epochs": 6,
        },
    }
    for name, payload in (
        ("study_config.json", study_config),
        ("aggregate_summary.json", aggregate_summary),
        ("best_pc_config_summary.json", best_pc_summary),
        ("best_mlp_config_summary.json", best_mlp_summary),
    ):
        with (reference_dir / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_phase2g1_boundary_check_is_deterministic(tmp_path: Path) -> None:
    _write_phase2g_reference_artifacts(tmp_path, "toy_regression")
    common_kwargs = {
        "benchmark_name": "toy_regression",
        "previous_search_output_root": tmp_path,
        "plot_energy": False,
        "pc_boundary_extensions_override": {
            "eta_x": [0.025],
            "eta_w": [0.6],
            "epochs": [8],
        },
        "mlp_boundary_extensions_override": {
            "eta_w": [0.3],
            "epochs": [8],
        },
    }
    run_a = run_phase2g1_boundary_check(
        output_root=tmp_path / "run_a",
        run_id="boundary_a",
        output_layout="run_id_subdir",
        **common_kwargs,
    )
    run_b = run_phase2g1_boundary_check(
        output_root=tmp_path / "run_b",
        run_id="boundary_b",
        output_layout="run_id_subdir",
        **common_kwargs,
    )

    summary_a = _load_json(run_a.run_dir / "aggregate_summary.json")
    summary_b = _load_json(run_b.run_dir / "aggregate_summary.json")
    summary_a["run_id"] = "normalized"
    summary_b["run_id"] = "normalized"
    assert summary_a == summary_b

    study_a = _load_json(run_a.run_dir / "study_config.json")
    study_b = _load_json(run_b.run_dir / "study_config.json")
    study_a["run_id"] = "normalized"
    study_b["run_id"] = "normalized"
    assert study_a == study_b

    assert _load_csv(run_a.run_dir / "pc_boundary_results.csv") == _load_csv(
        run_b.run_dir / "pc_boundary_results.csv"
    )
    assert _load_csv(run_a.run_dir / "mlp_boundary_results.csv") == _load_csv(
        run_b.run_dir / "mlp_boundary_results.csv"
    )
