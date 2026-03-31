from __future__ import annotations

import csv
import json
from pathlib import Path

from pc.pc_diagnostics import PHASE2D_VARIANT_GROUPS, run_pc_diagnostics_study


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_pc_diagnostics_writes_expected_artifacts(tmp_path: Path) -> None:
    result = run_pc_diagnostics_study(
        "toy_regression",
        output_root=tmp_path,
        run_id="toy_regression_diagnostics",
        plot_energy=False,
        plot_summary=False,
        seed_values=[0, 1],
    )

    run_dir = tmp_path / "pc_diagnostics_toy_regression"
    assert result.run_dir == run_dir
    assert (run_dir / "study_config.json").exists()
    assert (run_dir / "seed_records.csv").exists()
    assert (run_dir / "epoch_records.csv").exists()
    assert (run_dir / "epoch_summary.csv").exists()
    assert (run_dir / "diagnostic_summary.json").exists()

    for seed in (0, 1):
        seed_dir = run_dir / "seeds" / f"seed_{seed:04d}"
        assert (seed_dir / "default_pc" / "summary.json").exists()
        assert (seed_dir / "tuned_pc" / "summary.json").exists()
        assert (seed_dir / "tuned_pc_budget2x" / "summary.json").exists()
        assert (seed_dir / "mlp" / "summary.json").exists()

    study_config = _load_json(run_dir / "study_config.json")
    diagnostic_summary = _load_json(run_dir / "diagnostic_summary.json")
    epoch_records = _load_csv(run_dir / "epoch_records.csv")
    epoch_summary = _load_csv(run_dir / "epoch_summary.csv")

    assert study_config["variant_groups"] == PHASE2D_VARIANT_GROUPS
    assert study_config["budget_diagnostic_variant"] == "tuned_pc_budget2x"
    assert diagnostic_summary["variant_groups"] == PHASE2D_VARIANT_GROUPS
    assert diagnostic_summary["budget_diagnostic_variant"] == "tuned_pc_budget2x"
    assert "budget diagnostic branch" in diagnostic_summary["budget_diagnostic_note"]

    expected_pairs = {
        ("default_pc", "main_pc"),
        ("tuned_pc", "main_pc"),
        ("tuned_pc_budget2x", "budget_check"),
        ("mlp", "mlp_reference"),
    }
    assert {(row["variant"], row["variant_group"]) for row in epoch_records} == expected_pairs
    assert {(row["variant"], row["variant_group"]) for row in epoch_summary} == expected_pairs
