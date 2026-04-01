from __future__ import annotations

import csv
import json
from pathlib import Path

from pc.phase2g_matched_search import run_phase2g_matched_search


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_phase2g_matched_search_is_deterministic_under_fixed_search_spaces(tmp_path: Path) -> None:
    pc_override = {
        "eta_x": [0.1],
        "eta_w": [0.1, 0.2],
        "train_steps": [25],
        "epochs": [4],
    }
    mlp_override = {
        "eta_w": [0.05, 0.1],
        "epochs": [4],
    }
    root_a = tmp_path / "run_a"
    root_b = tmp_path / "run_b"

    result_a = run_phase2g_matched_search(
        "toy_regression",
        output_root=root_a,
        run_id="phase2g_a",
        plot_energy=False,
        pc_search_space_override=pc_override,
        mlp_search_space_override=mlp_override,
    )
    result_b = run_phase2g_matched_search(
        "toy_regression",
        output_root=root_b,
        run_id="phase2g_b",
        plot_energy=False,
        pc_search_space_override=pc_override,
        mlp_search_space_override=mlp_override,
    )

    study_a = _load_json(result_a.run_dir / "study_config.json")
    study_b = _load_json(result_b.run_dir / "study_config.json")
    study_a["run_id"] = "normalized"
    study_b["run_id"] = "normalized"

    aggregate_a = _load_json(result_a.run_dir / "aggregate_summary.json")
    aggregate_b = _load_json(result_b.run_dir / "aggregate_summary.json")
    aggregate_a["run_id"] = "normalized"
    aggregate_b["run_id"] = "normalized"

    best_pc_a = _load_json(result_a.run_dir / "best_pc_config_summary.json")
    best_pc_b = _load_json(result_b.run_dir / "best_pc_config_summary.json")
    best_pc_a["run_id"] = "normalized"
    best_pc_b["run_id"] = "normalized"

    best_mlp_a = _load_json(result_a.run_dir / "best_mlp_config_summary.json")
    best_mlp_b = _load_json(result_b.run_dir / "best_mlp_config_summary.json")
    best_mlp_a["run_id"] = "normalized"
    best_mlp_b["run_id"] = "normalized"

    assert study_a == study_b
    assert aggregate_a == aggregate_b
    assert best_pc_a == best_pc_b
    assert best_mlp_a == best_mlp_b
    assert _load_rows(result_a.run_dir / "pc_search_results.csv") == _load_rows(
        result_b.run_dir / "pc_search_results.csv"
    )
    assert _load_rows(result_a.run_dir / "mlp_search_results.csv") == _load_rows(
        result_b.run_dir / "mlp_search_results.csv"
    )
