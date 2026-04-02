from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_run():
    module = runpy.run_path(str(ROOT / "experiments" / "digits_mlp.py"))
    return module["run"]


def _read_epoch_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_summary(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_digits_mlp_smoke_run_writes_expected_artifacts(tmp_path: Path) -> None:
    result = load_run()(output_root=tmp_path, run_id="digits_smoke", plot_curves=False)

    run_dir = result.run_dir
    assert run_dir == tmp_path / "digits_mlp"
    assert (run_dir / "config.json").exists()
    assert (run_dir / "epoch_metrics.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert not (run_dir / "plots").exists()

    summary = _read_summary(run_dir / "summary.json")
    assert summary["phase"] == "Phase 3"
    assert summary["model_family"] == "mlp"
    assert summary["dataset_name"] == "digits"
    assert summary["primary_metric_name"] == "accuracy"
    assert summary["primary_metric_higher_is_better"] is True
    assert summary["selection_metric_source"] == "val_metric"
    assert summary["report_metric_source"] == "test_metric"
    assert summary["batch_size"] == 64
    assert summary["epochs"] == 100
    assert summary["best_epoch"] >= 1
    assert summary["best_epoch"] <= summary["epochs"]
    assert summary["val_metric"] == summary["best_val_metric"]
    assert summary["test_metric"] > summary["test_baseline_metric"]
    assert summary["best_val_metric"] > summary["val_baseline_metric"]

    epoch_rows = _read_epoch_rows(run_dir / "epoch_metrics.csv")
    assert len(epoch_rows) == summary["epochs"]
    assert "train_loss" in epoch_rows[0]
    assert "train_accuracy" in epoch_rows[0]
    assert "val_loss" in epoch_rows[0]
    assert "val_accuracy" in epoch_rows[0]
    assert "weight_norm_l1" in epoch_rows[0]
    assert "bias_norm_l1" in epoch_rows[0]


def test_digits_mlp_smoke_run_is_reproducible_under_fixed_seeds(tmp_path: Path) -> None:
    run = load_run()

    first = run(output_root=tmp_path / "first", run_id="digits_repro", plot_curves=False)
    second = run(output_root=tmp_path / "second", run_id="digits_repro", plot_curves=False)

    first_summary = _read_summary(first.run_dir / "summary.json")
    second_summary = _read_summary(second.run_dir / "summary.json")
    assert first_summary == second_summary

    first_epoch_rows = _read_epoch_rows(first.run_dir / "epoch_metrics.csv")
    second_epoch_rows = _read_epoch_rows(second.run_dir / "epoch_metrics.csv")
    assert first_epoch_rows == second_epoch_rows
