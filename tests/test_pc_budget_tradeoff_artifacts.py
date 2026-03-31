from __future__ import annotations

import csv
import json
from pathlib import Path

from pc.pc_budget_tradeoff import PHASE2E_VARIANT_GROUPS, run_pc_budget_tradeoff_study


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_pc_budget_tradeoff_writes_expected_artifacts(tmp_path: Path) -> None:
    result = run_pc_budget_tradeoff_study(
        "toy_regression",
        output_root=tmp_path,
        run_id="toy_regression_budget_tradeoff",
        plot_energy=False,
        plot_summary=False,
        seed_values=[0, 1],
    )

    run_dir = tmp_path / "pc_budget_tradeoff_toy_regression"
    assert result.run_dir == run_dir
    assert (run_dir / "study_config.json").exists()
    assert (run_dir / "seed_budget_records.csv").exists()
    assert (run_dir / "budget_summary.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()

    for seed in (0, 1):
        seed_dir = run_dir / "seeds" / f"seed_{seed:04d}"
        assert (seed_dir / "tuned_pc_1x" / "summary.json").exists()
        assert (seed_dir / "tuned_pc_2x" / "summary.json").exists()
        assert (seed_dir / "tuned_pc_4x" / "summary.json").exists()
        assert (seed_dir / "mlp" / "summary.json").exists()

    study_config = _load_json(run_dir / "study_config.json")
    seed_rows = _load_csv(run_dir / "seed_budget_records.csv")
    budget_rows = _load_csv(run_dir / "budget_summary.csv")

    assert study_config["variant_groups"] == PHASE2E_VARIANT_GROUPS
    assert study_config["budget_levels"] == {
        "1x": {"budget_multiplier": "1x", "train_steps": 25, "eval_steps": 25},
        "2x": {"budget_multiplier": "2x", "train_steps": 50, "eval_steps": 50},
        "4x": {"budget_multiplier": "4x", "train_steps": 100, "eval_steps": 100},
    }
    assert study_config["budget_definition"] == "budget = inference step count, not wall-clock or FLOP matching"

    mlp_seed_rows = [row for row in seed_rows if row["variant"] == "mlp"]
    assert len(mlp_seed_rows) == 2
    assert {row["is_reference_variant"] for row in mlp_seed_rows} == {"True"}
    assert {row["budget_multiplier"] for row in mlp_seed_rows} == {"reference"}

    mlp_budget_row = next(row for row in budget_rows if row["variant"] == "mlp")
    assert mlp_budget_row["budget_multiplier"] == "reference"
    assert mlp_budget_row["train_steps"] == ""
