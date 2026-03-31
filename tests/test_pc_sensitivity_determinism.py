from __future__ import annotations

import csv
import json
from pathlib import Path

from pc.pc_sensitivity import run_pc_sensitivity_study


def _load_trial_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_pc_sensitivity_is_deterministic_with_separate_output_roots(tmp_path: Path) -> None:
    root_a = tmp_path / "run_a"
    root_b = tmp_path / "run_b"
    result_a = run_pc_sensitivity_study(
        "toy_regression",
        output_root=root_a,
        run_id="sensitivity_a",
        plot_energy=False,
        plot_summary=False,
    )
    result_b = run_pc_sensitivity_study(
        "toy_regression",
        output_root=root_b,
        run_id="sensitivity_b",
        plot_energy=False,
        plot_summary=False,
    )

    aggregate_a = _load_json(result_a.run_dir / "aggregate_summary.json")
    aggregate_b = _load_json(result_b.run_dir / "aggregate_summary.json")
    aggregate_a["run_id"] = "normalized"
    aggregate_b["run_id"] = "normalized"

    mlp_summary_a = _load_json(result_a.run_dir / "mlp_reference" / "summary.json")
    mlp_summary_b = _load_json(result_b.run_dir / "mlp_reference" / "summary.json")
    mlp_summary_a["run_id"] = "normalized"
    mlp_summary_b["run_id"] = "normalized"

    assert aggregate_a == aggregate_b
    assert mlp_summary_a == mlp_summary_b
    assert _load_trial_rows(result_a.run_dir / "trial_table.csv") == _load_trial_rows(
        result_b.run_dir / "trial_table.csv"
    )
