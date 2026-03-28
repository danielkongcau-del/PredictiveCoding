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
from pc.toy_data import make_linear_regression_data


def make_small_regression_run(
    tmp_path: Path,
    run_id: str | None,
    plot_energy: bool,
) -> tuple[Path, dict[str, object]]:
    x, y = make_linear_regression_data(num_points=8)
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
        epochs=4,
        output_root=tmp_path,
        run_id=run_id,
        plot_energy=plot_energy,
        task={"name": "regression"},
        data={"dataset_name": "linear_regression", "num_points": 8},
        model={"layer_dims": [1, 3, 1], "hidden_activation": "tanh", "output_activation": "identity"},
        training={"epochs": 4, "train_steps": 12, "eval_steps": 12},
    )
    result = run_supervised_experiment(
        config=config,
        model=model,
        x=x,
        y=y,
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

    assert run_dir.name == "explicit_run"
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
        "task",
        "data",
        "model",
        "training",
        "logging",
    ]
    assert config_payload["run_id"] == "explicit_run"

    with (run_dir / "epoch_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["epoch"] == "1"
    assert rows[0]["train_steps"] == "12"
    assert "mse" in rows[0]
    assert "baseline_mse" in rows[0]

    assert summary["phase"] == "Phase 1"
    assert summary["math_version"] == "phase0-baseline"
    assert summary["primary_metric_higher_is_better"] is False


def test_timestamp_run_id_is_used_only_when_none_and_plots_are_optional(tmp_path: Path) -> None:
    run_dir, _ = make_small_regression_run(tmp_path, run_id=None, plot_energy=True)
    assert re.fullmatch(r"\d{8}_\d{6}_seed_19", run_dir.name) is not None
    assert (run_dir / "plots").exists()
    png_paths = sorted(path.name for path in (run_dir / "plots").glob("*.png"))
    assert png_paths

    traces = np.load(run_dir / "energy_traces.npz")
    assert sorted(traces.files)
