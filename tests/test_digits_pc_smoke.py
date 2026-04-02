from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path

import numpy as np

from pc.real_pc import RealPCConfig, run_digits_pc_experiment
from pc.toy_data import SupervisedDataSplit
from pc.training import TrainBatchResult


ROOT = Path(__file__).resolve().parents[1]


def load_run():
    module = runpy.run_path(str(ROOT / "experiments" / "digits_pc.py"))
    return module["run"]


def _read_epoch_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_summary(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_digits_pc_smoke_run_writes_expected_artifacts(tmp_path: Path) -> None:
    result = load_run()(output_root=tmp_path, run_id="digits_pc_smoke", plot_curves=False)

    run_dir = result.run_dir
    assert run_dir == tmp_path / "digits_pc"
    assert (run_dir / "config.json").exists()
    assert (run_dir / "epoch_metrics.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert not (run_dir / "plots").exists()

    summary = _read_summary(run_dir / "summary.json")
    assert summary["phase"] == "Phase 3"
    assert summary["model_family"] == "pc"
    assert summary["dataset_name"] == "digits"
    assert summary["primary_metric_name"] == "accuracy"
    assert summary["primary_metric_higher_is_better"] is True
    assert summary["selection_metric_source"] == "val_metric"
    assert summary["report_metric_source"] == "test_metric"
    assert summary["batch_size"] == 64
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
    assert "train_mean_pre_update_energy" in epoch_rows[0]
    assert "train_mean_post_update_energy" in epoch_rows[0]
    assert "weight_norm_l1" in epoch_rows[0]
    assert "bias_norm_l1" in epoch_rows[0]


def test_digits_pc_smoke_run_is_reproducible_under_fixed_seeds(tmp_path: Path) -> None:
    run = load_run()

    first = run(output_root=tmp_path / "first", run_id="digits_pc_repro", plot_curves=False)
    second = run(output_root=tmp_path / "second", run_id="digits_pc_repro", plot_curves=False)

    first_summary = _read_summary(first.run_dir / "summary.json")
    second_summary = _read_summary(second.run_dir / "summary.json")
    assert first_summary == second_summary

    first_epoch_rows = _read_epoch_rows(first.run_dir / "epoch_metrics.csv")
    second_epoch_rows = _read_epoch_rows(second.run_dir / "epoch_metrics.csv")
    assert first_epoch_rows == second_epoch_rows


def test_digits_pc_restores_best_checkpoint_before_final_report(
    monkeypatch,
    tmp_path: Path,
) -> None:
    x_train = np.array(
        [[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [1.5, 1.5]],
        dtype=np.float64,
    )
    y_train = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    split = SupervisedDataSplit(
        x_train=x_train,
        y_train=y_train,
        x_val=x_train.copy(),
        y_val=y_train.copy(),
        x_test=x_train.copy(),
        y_test=y_train.copy(),
        metadata={"dataset_name": "tiny_digits_stub"},
    )

    epoch_state = {"epoch": 0}

    def fake_load_digits_split(**_: object) -> SupervisedDataSplit:
        return split

    def fake_train_batch(self, x, y, compute_post_update_energy=False) -> TrainBatchResult:
        _ = (x, y)
        epoch_state["epoch"] += 1
        marker = float(epoch_state["epoch"])
        for layer in self.layers:
            layer.weight.fill(marker)
            layer.bias.fill(marker)
        parameter_norms = {
            "weight_norms": [float(np.linalg.norm(layer.weight)) for layer in self.layers],
            "bias_norms": [float(np.linalg.norm(layer.bias)) for layer in self.layers],
        }
        return TrainBatchResult(
            train_steps=self.train_steps,
            energy_trace=[marker],
            pre_update_energy=marker,
            post_update_energy=(marker + 0.5) if compute_post_update_energy else None,
            parameter_norms=parameter_norms,
        )

    def fake_evaluate_model(model, x, y):
        _ = (x, y)
        marker = float(model.layers[0].weight[0, 0])
        lookup = {
            1.0: (0.1, 0.9),
            2.0: (0.2, 0.6),
            3.0: (0.3, 0.4),
        }
        return lookup[marker]

    monkeypatch.setattr("pc.real_pc.load_digits_split", fake_load_digits_split)
    monkeypatch.setattr("pc.real_pc.PCNetwork.train_batch", fake_train_batch)
    monkeypatch.setattr("pc.real_pc._evaluate_model", fake_evaluate_model)

    result = run_digits_pc_experiment(
        RealPCConfig(
            output_root=tmp_path,
            run_id="restore_logic",
            layer_dims=(2, 3, 2),
            epochs=3,
            batch_size=10,
        )
    )

    summary = result.summary
    epoch_rows = result.epoch_metrics

    assert summary["best_epoch"] == 1
    assert summary["best_val_metric"] == 0.9
    assert summary["val_metric"] == 0.9
    assert summary["test_metric"] == 0.9
    assert float(epoch_rows[-1]["val_accuracy"]) == 0.4
    assert summary["val_metric"] != float(epoch_rows[-1]["val_accuracy"])
