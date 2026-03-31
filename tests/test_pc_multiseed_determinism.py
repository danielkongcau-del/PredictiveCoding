from __future__ import annotations

import csv
import json
from pathlib import Path

from pc.pc_multiseed import run_pc_multiseed_study


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_pc_multiseed_is_deterministic_with_same_explicit_seeds(tmp_path: Path) -> None:
    root_a = tmp_path / "run_a"
    root_b = tmp_path / "run_b"
    result_a = run_pc_multiseed_study(
        "toy_regression",
        output_root=root_a,
        run_id="deterministic_multiseed",
        plot_energy=False,
        plot_summary=False,
        seed_values=[0, 1],
    )
    result_b = run_pc_multiseed_study(
        "toy_regression",
        output_root=root_b,
        run_id="deterministic_multiseed",
        plot_energy=False,
        plot_summary=False,
        seed_values=[0, 1],
    )

    assert _load_csv_rows(result_a.run_dir / "seed_records.csv") == _load_csv_rows(
        result_b.run_dir / "seed_records.csv"
    )
    assert _load_json(result_a.run_dir / "aggregate_summary.json") == _load_json(
        result_b.run_dir / "aggregate_summary.json"
    )
