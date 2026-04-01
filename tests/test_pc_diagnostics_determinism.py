from __future__ import annotations

import csv
import json
from pathlib import Path

from pc.pc_diagnostics import run_pc_diagnostics_study


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_pc_diagnostics_is_deterministic_under_fixed_seed_set(tmp_path: Path) -> None:
    root_a = tmp_path / "run_a"
    root_b = tmp_path / "run_b"
    seed_values = [3, 4]

    result_a = run_pc_diagnostics_study(
        "toy_sine_regression",
        output_root=root_a,
        run_id="toy_sine_diagnostics_det",
        plot_energy=False,
        plot_summary=False,
        seed_values=seed_values,
    )
    result_b = run_pc_diagnostics_study(
        "toy_sine_regression",
        output_root=root_b,
        run_id="toy_sine_diagnostics_det",
        plot_energy=False,
        plot_summary=False,
        seed_values=seed_values,
    )

    assert _load_csv(result_a.run_dir / "seed_records.csv") == _load_csv(
        result_b.run_dir / "seed_records.csv"
    )
    assert _load_csv(result_a.run_dir / "epoch_summary.csv") == _load_csv(
        result_b.run_dir / "epoch_summary.csv"
    )
    assert _load_json(result_a.run_dir / "diagnostic_summary.json") == _load_json(
        result_b.run_dir / "diagnostic_summary.json"
    )
