from __future__ import annotations

import csv
import json
from pathlib import Path

from pc.pc_multiseed import run_pc_multiseed_study


def test_pc_multiseed_writes_expected_artifacts(tmp_path: Path) -> None:
    result = run_pc_multiseed_study(
        "toy_regression",
        output_root=tmp_path,
        run_id="toy_regression_multiseed",
        plot_energy=False,
        plot_summary=False,
        seed_values=[0, 1],
    )

    run_dir = tmp_path / "pc_multiseed_toy_regression"
    assert result.run_dir == run_dir
    assert (run_dir / "study_config.json").exists()
    assert (run_dir / "seed_records.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()
    assert (run_dir / "seeds" / "seed_0000" / "default_pc" / "summary.json").exists()
    assert (run_dir / "seeds" / "seed_0000" / "tuned_pc" / "summary.json").exists()
    assert (run_dir / "seeds" / "seed_0000" / "mlp" / "summary.json").exists()
    assert (run_dir / "seeds" / "seed_0001" / "default_pc" / "summary.json").exists()
    assert (run_dir / "seeds" / "seed_0001" / "tuned_pc" / "summary.json").exists()
    assert (run_dir / "seeds" / "seed_0001" / "mlp" / "summary.json").exists()

    with (run_dir / "study_config.json").open("r", encoding="utf-8") as handle:
        study_config = json.load(handle)
    with (run_dir / "seed_records.csv").open("r", encoding="utf-8", newline="") as handle:
        seed_rows = list(csv.DictReader(handle))
    with (run_dir / "aggregate_summary.json").open("r", encoding="utf-8") as handle:
        aggregate_summary = json.load(handle)

    assert study_config["seed_values"] == [0, 1]
    assert study_config["seed_semantics"]["data_seed"] == "fixed"
    assert study_config["seed_semantics"]["run_seed"] == "varies_with_seed"
    assert study_config["seed_semantics"]["model_init_seed"] == "varies_with_seed"
    assert study_config["tuned_pc_preset_name"] == "eta_w_double"
    assert study_config["tuned_pc_training"] == {
        "epochs": 60,
        "eta_x": 0.2,
        "eta_w": 0.1,
        "eta_b": 0.1,
        "train_steps": 25,
        "eval_steps": 25,
        "state_init": "forward",
    }

    assert len(seed_rows) == 2
    assert seed_rows[0]["run_seed"] == "0"
    assert seed_rows[0]["model_init_seed"] == "0"
    assert seed_rows[0]["data_seed"] == "0"
    assert seed_rows[0]["default_pc_status"] == "ok"
    assert seed_rows[0]["tuned_pc_status"] == "ok"
    assert seed_rows[0]["mlp_status"] == "ok"
    assert seed_rows[0]["tuned_pc_beats_default_pc"] in {"True", "False"}
    assert seed_rows[0]["tuned_pc_beats_mlp"] in {"True", "False"}

    assert aggregate_summary["planned_seed_count"] == 2
    assert aggregate_summary["tie_rtol"] == 1.0e-12
    assert aggregate_summary["tie_atol"] == 1.0e-12
