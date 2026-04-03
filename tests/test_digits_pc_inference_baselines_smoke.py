from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path

from pc.real_pc_inference_baselines import (
    RealPCInferenceBaselineCandidate,
    RealPCInferenceBaselineStudyConfig,
)

ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_digits_pc_inference_baseline_study_smoke(tmp_path: Path) -> None:
    module = runpy.run_path(str(ROOT / "experiments" / "digits_pc_inference_baselines.py"))
    run = module["run"]

    config = RealPCInferenceBaselineStudyConfig(
        output_root=tmp_path,
        run_id="smoke",
        epochs=1,
        batch_size=256,
        candidates=(
            RealPCInferenceBaselineCandidate("euler_steps_01", "euler", 1),
            RealPCInferenceBaselineCandidate("rk2_steps_01", "rk2", 1),
        ),
    )
    result = run(output_root=tmp_path, run_id="ignored", plot_curves=False, config=config)

    run_dir = result.run_dir
    assert (run_dir / "study_config.json").exists()
    assert (run_dir / "trial_table.csv").exists()
    assert (run_dir / "aggregate_summary.json").exists()

    study_config = _read_json(run_dir / "study_config.json")
    aggregate_summary = _read_json(run_dir / "aggregate_summary.json")
    trial_rows = _read_csv_rows(run_dir / "trial_table.csv")

    assert study_config["selected_by"] == "val_metric"
    assert aggregate_summary["selected_by"] == "val_metric"
    assert aggregate_summary["num_candidates"] == 2
    assert len(trial_rows) == 2
    assert aggregate_summary["selected_candidate_id"] in {
        row["candidate_id"] for row in trial_rows
    }
    assert "inference_backend" in aggregate_summary["selected_candidate"]
    assert {row["inference_backend"] for row in trial_rows} == {"pc_euler", "pc_rk2"}
    assert {row["inference_method"] for row in trial_rows} == {"euler", "rk2"}
