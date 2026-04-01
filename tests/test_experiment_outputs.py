from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

from pc.experiment import ExperimentConfig, run_supervised_experiment
from pc.layers import init_mlp_layers
from pc.metrics import regression_mean_baseline_mse, regression_mse
from pc.models import PCNetwork
from pc.toy_data import make_linear_regression_split


def make_small_regression_run(
    tmp_path: Path,
    run_id: str | None,
    plot_energy: bool,
) -> tuple[Path, dict[str, object]]:
    split = make_linear_regression_split(num_points=8, val_num_points=33, test_num_points=35)
    x = split.x_train
    y = split.y_train
    model = PCNetwork(
        layers=init_mlp_layers([1, 3, 1], seed=19, weight_scale=0.12),
        eta_x=0.2,
        eta_w=0.05,
        eta_b=0.05,
        train_steps=12,
        eval_steps=12,
        state_init="forward",
    )
    config = ExperimentConfig(
        experiment_name="tiny_regression_outputs",
        seed=19,
        data_seed=23,
        model_init_seed=29,
        epochs=4,
        output_root=tmp_path,
        run_id=run_id,
        plot_energy=plot_energy,
        task={"name": "regression"},
        data={
            "dataset_name": "linear_regression",
            "train_num_points": 8,
            "val_num_points": 33,
            "test_num_points": 35,
            "train_size": 8,
            "val_size": 33,
            "test_size": 35,
            "evaluation_protocol": "dense_offset_val_test_grids",
            "data_seed": 23,
        },
        model={
            "layer_dims": [1, 3, 1],
            "hidden_activation": "tanh",
            "output_activation": "identity",
            "model_init_seed": 29,
        },
        training={"epochs": 4, "train_steps": 12, "eval_steps": 12, "run_seed": 19},
    )
    result = run_supervised_experiment(
        config=config,
        model=model,
        x=x,
        y=y,
        x_val=split.x_val,
        y_val=split.y_val,
        x_test=split.x_test,
        y_test=split.y_test,
        task_name="regression",
        primary_metric_name="mse",
        primary_metric_higher_is_better=False,
        primary_metric_fn=regression_mse,
        baseline_metric_name="baseline_mse",
        baseline_metric_fn=regression_mean_baseline_mse,
    )
    return result.run_dir, result.summary


def test_experiment_outputs_with_explicit_run_id_and_default_no_plot(tmp_path: Path) -> None:
    run_dir, summary = make_small_regression_run(tmp_path, run_id="explicit_run", plot_energy=False)

    assert run_dir == tmp_path / "tiny_regression_outputs"
    assert (run_dir / "config.json").exists()
    assert (run_dir / "epoch_metrics.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "energy_traces.npz").exists()
    assert (run_dir / "energy_traces_manifest.json").exists()
    assert not (run_dir / "plots").exists()

    with (run_dir / "config.json").open("r", encoding="utf-8") as handle:
        config_payload = json.load(handle)
    assert list(config_payload.keys()) == [
        "experiment_name",
        "run_id",
        "seed",
        "seeds",
        "task",
        "data",
        "model",
        "training",
        "logging",
    ]
    assert config_payload["run_id"] == "explicit_run"
    assert config_payload["seed"] == 19
    assert config_payload["seeds"] == {
        "run_seed": 19,
        "data_seed": 23,
        "model_init_seed": 29,
    }
    assert config_payload["data"]["train_size"] == 8
    assert config_payload["data"]["val_size"] == 33
    assert config_payload["data"]["test_size"] == 35
    assert config_payload["data"]["evaluation_protocol"] == "dense_offset_val_test_grids"

    with (run_dir / "epoch_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["epoch"] == "1"
    assert rows[0]["train_steps"] == "12"
    assert "train_mse" in rows[0]
    assert "val_mse" in rows[0]
    assert "train_baseline_mse" in rows[0]
    assert "val_baseline_mse" in rows[0]
    assert "mse" in rows[0]
    assert "baseline_mse" in rows[0]

    assert summary["phase"] == "Phase 1"
    assert summary["math_version"] == "phase0-baseline"
    assert summary["metric_name"] == "mse"
    assert summary["train_metric"] >= 0.0
    assert summary["val_metric"] >= 0.0
    assert summary["test_metric"] >= 0.0
    assert summary["train_baseline_metric"] >= 0.0
    assert summary["val_baseline_metric"] >= 0.0
    assert summary["test_baseline_metric"] >= 0.0
    assert summary["selection_metric_source"] == "val_metric"
    assert summary["report_metric_source"] == "test_metric"
    assert summary["primary_metric_higher_is_better"] is False


def test_timestamp_run_id_is_used_only_when_none_and_plots_are_optional(tmp_path: Path) -> None:
    run_dir, _ = make_small_regression_run(tmp_path, run_id=None, plot_energy=True)
    with (run_dir / "config.json").open("r", encoding="utf-8") as handle:
        config_payload = json.load(handle)
    assert re.fullmatch(r"\d{8}_\d{6}_seed_19", config_payload["run_id"]) is not None
    assert config_payload["logging"]["output_layout"] == "single_dir"
    assert (run_dir / "plots").exists()
    png_paths = sorted(path.name for path in (run_dir / "plots").glob("*.png"))
    assert png_paths

    traces = np.load(run_dir / "energy_traces.npz")
    assert sorted(traces.files)


def test_run_id_subdir_layout_is_available_without_changing_runner_logic(tmp_path: Path) -> None:
    split = make_linear_regression_split(num_points=8, val_num_points=33, test_num_points=35)
    x = split.x_train
    y = split.y_train
    model = PCNetwork(
        layers=init_mlp_layers([1, 3, 1], seed=31, weight_scale=0.12),
        eta_x=0.2,
        eta_w=0.05,
        eta_b=0.05,
        train_steps=12,
        eval_steps=12,
        state_init="forward",
    )
    config = ExperimentConfig(
        experiment_name="tiny_regression_outputs",
        seed=31,
        data_seed=37,
        model_init_seed=41,
        epochs=2,
        output_root=tmp_path,
        run_id="nested_run",
        output_layout="run_id_subdir",
        plot_energy=False,
        task={"name": "regression"},
        data={
            "dataset_name": "linear_regression",
            "train_num_points": 8,
            "val_num_points": 33,
            "test_num_points": 35,
            "train_size": 8,
            "val_size": 33,
            "test_size": 35,
            "evaluation_protocol": "dense_offset_val_test_grids",
            "data_seed": 37,
        },
        model={
            "layer_dims": [1, 3, 1],
            "hidden_activation": "tanh",
            "output_activation": "identity",
            "model_init_seed": 41,
        },
        training={"epochs": 2, "train_steps": 12, "eval_steps": 12, "run_seed": 31},
    )
    result = run_supervised_experiment(
        config=config,
        model=model,
        x=x,
        y=y,
        x_val=split.x_val,
        y_val=split.y_val,
        x_test=split.x_test,
        y_test=split.y_test,
        task_name="regression",
        primary_metric_name="mse",
        primary_metric_higher_is_better=False,
        primary_metric_fn=regression_mse,
        baseline_metric_name="baseline_mse",
        baseline_metric_fn=regression_mean_baseline_mse,
    )

    assert result.run_dir == tmp_path / "tiny_regression_outputs" / "nested_run"
    with (result.run_dir / "config.json").open("r", encoding="utf-8") as handle:
        config_payload = json.load(handle)
    assert config_payload["logging"]["output_layout"] == "run_id_subdir"
