from __future__ import annotations

import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_run():
    module = runpy.run_path(str(ROOT / "experiments" / "stabilize_digits_pc.py"))
    return module["run"]


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_digits_pc_stabilization_smoke_runs_small_sweep_and_writes_summary(tmp_path: Path) -> None:
    small_candidates = [
        {
            "config_id": "cfg_a",
            "description": "small candidate a",
            "overrides": {
                "epochs": 2,
                "train_steps": 5,
                "eval_steps": 5,
                "batch_size": 128,
            },
        },
        {
            "config_id": "cfg_b",
            "description": "small candidate b",
            "overrides": {
                "layer_dims": (64, 16, 10),
                "epochs": 3,
                "train_steps": 8,
                "eval_steps": 8,
                "eta_x": 0.10,
                "eta_w": 0.02,
                "eta_b": 0.02,
                "batch_size": 128,
            },
        },
    ]

    summary = load_run()(output_root=tmp_path, candidate_specs=small_candidates, plot_curves=False)

    summary_path = tmp_path / "digits_pc_stabilization" / "summary.json"
    assert summary_path.exists()
    written = _read_json(summary_path)
    assert written == summary

    assert summary["phase"] == "Phase 3"
    assert summary["study_type"] == "digits_pc_stabilization_sweep"
    assert summary["dataset_name"] == "digits"
    assert summary["selected_by"] == "val_metric"
    assert summary["selection_metric_source"] == "val_metric"
    assert summary["report_metric_source"] == "test_metric"
    assert summary["selected_config_id"] in {"cfg_a", "cfg_b"}
    assert len(summary["candidates"]) == 2

    best_val_metric = max(float(candidate["val_metric"]) for candidate in summary["candidates"])
    assert float(summary["selected_candidate"]["val_metric"]) == best_val_metric

    for candidate in summary["candidates"]:
        assert Path(candidate["run_dir"]).exists()
        assert Path(candidate["summary_path"]).exists()
        assert candidate["baseline_metric_name"] == "baseline_accuracy"
