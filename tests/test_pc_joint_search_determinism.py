from __future__ import annotations

import csv
import json
from pathlib import Path

from pc.pc_joint_search import run_pc_joint_search


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_pc_joint_search_is_deterministic_under_fixed_search_space(tmp_path: Path) -> None:
    search_space_override = {
        "eta_x": [0.1],
        "eta_w": [0.1, 0.2],
        "train_steps": [25],
        "epochs": [4],
    }
    root_a = tmp_path / "run_a"
    root_b = tmp_path / "run_b"

    result_a = run_pc_joint_search(
        "toy_regression",
        output_root=root_a,
        run_id="joint_search_a",
        plot_energy=False,
        search_space_override=search_space_override,
    )
    result_b = run_pc_joint_search(
        "toy_regression",
        output_root=root_b,
        run_id="joint_search_b",
        plot_energy=False,
        search_space_override=search_space_override,
    )

    aggregate_a = _load_json(result_a.run_dir / "aggregate_summary.json")
    aggregate_b = _load_json(result_b.run_dir / "aggregate_summary.json")
    aggregate_a["run_id"] = "normalized"
    aggregate_b["run_id"] = "normalized"

    best_a = _load_json(result_a.run_dir / "best_config_summary.json")
    best_b = _load_json(result_b.run_dir / "best_config_summary.json")
    best_a["run_id"] = "normalized"
    best_b["run_id"] = "normalized"

    study_a = _load_json(result_a.run_dir / "study_config.json")
    study_b = _load_json(result_b.run_dir / "study_config.json")
    study_a["run_id"] = "normalized"
    study_b["run_id"] = "normalized"

    assert aggregate_a == aggregate_b
    assert best_a == best_b
    assert study_a == study_b
    assert _load_rows(result_a.run_dir / "search_results.csv") == _load_rows(
        result_b.run_dir / "search_results.csv"
    )
