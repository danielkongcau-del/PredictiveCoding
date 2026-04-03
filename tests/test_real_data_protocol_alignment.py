from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from pc.real_mlp import RealMLPConfig, run_digits_mlp_experiment
from pc.real_pc import RealPCConfig, run_digits_pc_experiment


CORE_SUMMARY_FIELDS = {
    "phase",
    "math_version",
    "dataset_name",
    "task_name",
    "metric_name",
    "primary_metric_name",
    "metric_higher_is_better",
    "primary_metric_higher_is_better",
    "selection_metric_source",
    "report_metric_source",
    "train_metric",
    "val_metric",
    "test_metric",
    "train_baseline_metric",
    "val_baseline_metric",
    "test_baseline_metric",
    "baseline_metric_name",
    "best_epoch",
    "best_val_metric",
    "batch_size",
    "batches_per_epoch",
    "epochs",
    "run_seed",
    "data_seed",
    "model_init_seed",
    "batch_order_seed",
}


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_epoch_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _majority_fraction(class_counts: list[int]) -> float:
    counts = np.asarray(class_counts, dtype=np.float64)
    return float(np.max(counts) / np.sum(counts))


def _first_best_epoch(rows: list[dict[str, str]]) -> int:
    val_accuracies = [float(row["val_accuracy"]) for row in rows]
    best_value = max(val_accuracies)
    for index, value in enumerate(val_accuracies, start=1):
        if value == best_value:
            return index
    raise AssertionError("Expected at least one epoch row.")


@pytest.fixture(scope="module")
def aligned_real_data_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("real_data_protocol_alignment")
    common_seeds = {
        "run_seed": 7,
        "data_seed": 11,
        "model_init_seed": 13,
        "batch_order_seed": 17,
    }

    mlp_result = run_digits_mlp_experiment(
        RealMLPConfig(
            output_root=root / "mlp_root",
            run_id="protocol_alignment",
            epochs=5,
            batch_size=128,
            plot_curves=False,
            **common_seeds,
        )
    )
    pc_result = run_digits_pc_experiment(
        RealPCConfig(
            output_root=root / "pc_root",
            run_id="protocol_alignment",
            epochs=5,
            batch_size=128,
            train_steps=10,
            eval_steps=10,
            plot_curves=False,
            **common_seeds,
        )
    )

    return {
        "mlp_config": _read_json(mlp_result.run_dir / "config.json"),
        "mlp_summary": _read_json(mlp_result.run_dir / "summary.json"),
        "mlp_epochs": _read_epoch_rows(mlp_result.run_dir / "epoch_metrics.csv"),
        "pc_config": _read_json(pc_result.run_dir / "config.json"),
        "pc_summary": _read_json(pc_result.run_dir / "summary.json"),
        "pc_epochs": _read_epoch_rows(pc_result.run_dir / "epoch_metrics.csv"),
    }


def test_real_data_mlp_and_pc_share_protocol_contract(
    aligned_real_data_runs: dict[str, object],
) -> None:
    mlp_config = aligned_real_data_runs["mlp_config"]
    pc_config = aligned_real_data_runs["pc_config"]
    mlp_summary = aligned_real_data_runs["mlp_summary"]
    pc_summary = aligned_real_data_runs["pc_summary"]

    assert mlp_config["data"] == pc_config["data"]
    assert mlp_config["data"]["dataset_name"] == "digits"
    assert pc_config["data"]["dataset_name"] == "digits"
    assert mlp_summary["dataset_name"] == "digits"
    assert pc_summary["dataset_name"] == "digits"

    assert CORE_SUMMARY_FIELDS.issubset(mlp_summary.keys())
    assert CORE_SUMMARY_FIELDS.issubset(pc_summary.keys())

    for field in (
        "phase",
        "dataset_name",
        "task_name",
        "metric_name",
        "primary_metric_name",
        "metric_higher_is_better",
        "primary_metric_higher_is_better",
        "selection_metric_source",
        "report_metric_source",
        "baseline_metric_name",
        "run_seed",
        "data_seed",
        "model_init_seed",
        "batch_order_seed",
        "batch_size",
        "batches_per_epoch",
        "epochs",
    ):
        assert mlp_summary[field] == pc_summary[field]

    assert mlp_summary["phase"] == "Phase 3"
    assert pc_summary["phase"] == "Phase 3"
    assert mlp_summary["selection_metric_source"] == "val_metric"
    assert pc_summary["selection_metric_source"] == "val_metric"
    assert mlp_summary["report_metric_source"] == "test_metric"
    assert pc_summary["report_metric_source"] == "test_metric"
    assert mlp_summary["primary_metric_name"] == "accuracy"
    assert pc_summary["primary_metric_name"] == "accuracy"
    assert mlp_summary["primary_metric_higher_is_better"] is True
    assert pc_summary["primary_metric_higher_is_better"] is True
    assert pc_config["training"]["inference_backend"] == "pc_euler"
    assert pc_config["training"]["inference_method"] == "euler"
    assert pc_config["evaluation"]["teacher_reference_backend"] is None
    assert pc_config["evaluation"]["teacher_reference_eval_steps"] is None
    assert pc_config["evaluation"]["teacher_reference_metrics_enabled"] is False
    assert "predict-mode" in pc_config["evaluation"]["teacher_reference_disable_reason"]
    assert pc_summary["inference_backend"] == "pc_euler"
    assert pc_summary["inference_method"] == "euler"
    assert pc_summary["teacher_reference"]["enabled"] is False
    assert "predict-mode" in pc_summary["teacher_reference"]["reason"]

    for summary in (mlp_summary, pc_summary):
        assert summary["seeds"]["run_seed"] == summary["run_seed"]
        assert summary["seeds"]["data_seed"] == summary["data_seed"]
        assert summary["seeds"]["model_init_seed"] == summary["model_init_seed"]
        assert summary["seeds"]["batch_order_seed"] == summary["batch_order_seed"]

    for config in (mlp_config, pc_config):
        assert config["seeds"]["run_seed"] == config["run_seed"]
        assert config["seeds"]["data_seed"] == config["data_seed"]
        assert config["seeds"]["model_init_seed"] == config["model_init_seed"]
        assert config["seeds"]["batch_order_seed"] == config["batch_order_seed"]


def test_real_data_baseline_definition_and_best_checkpoint_rule_match(
    aligned_real_data_runs: dict[str, object],
) -> None:
    mlp_config = aligned_real_data_runs["mlp_config"]
    pc_config = aligned_real_data_runs["pc_config"]
    mlp_summary = aligned_real_data_runs["mlp_summary"]
    pc_summary = aligned_real_data_runs["pc_summary"]
    mlp_epochs = aligned_real_data_runs["mlp_epochs"]
    pc_epochs = aligned_real_data_runs["pc_epochs"]

    for config, summary, rows in (
        (mlp_config, mlp_summary, mlp_epochs),
        (pc_config, pc_summary, pc_epochs),
    ):
        assert np.isclose(summary["val_metric"], summary["best_val_metric"])
        assert summary["best_epoch"] == _first_best_epoch(rows)

        data = config["data"]
        assert np.isclose(
            summary["train_baseline_metric"],
            _majority_fraction(data["train_class_counts"]),
        )
        assert np.isclose(
            summary["val_baseline_metric"],
            _majority_fraction(data["val_class_counts"]),
        )
        assert np.isclose(
            summary["test_baseline_metric"],
            _majority_fraction(data["test_class_counts"]),
        )

    for field in (
        "train_baseline_metric",
        "val_baseline_metric",
        "test_baseline_metric",
    ):
        assert np.isclose(mlp_summary[field], pc_summary[field])
