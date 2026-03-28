from __future__ import annotations

import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_run(script_name: str):
    module = runpy.run_path(str(ROOT / "experiments" / script_name))
    return module["run"]


def test_benchmark_scripts_write_outputs_and_beat_baselines(tmp_path: Path) -> None:
    regression_result = load_run("toy_regression.py")(
        output_root=tmp_path,
        run_id="regression_smoke",
        plot_energy=False,
    )
    sine_result = load_run("toy_sine_regression.py")(
        output_root=tmp_path,
        run_id="sine_smoke",
        plot_energy=False,
    )
    classification_result = load_run("toy_blobs_classification.py")(
        output_root=tmp_path,
        run_id="classification_smoke",
        plot_energy=False,
    )

    for result in (regression_result, sine_result, classification_result):
        assert (result.run_dir / "config.json").exists()
        assert (result.run_dir / "epoch_metrics.csv").exists()
        assert (result.run_dir / "summary.json").exists()

    assert regression_result.summary["primary_metric_value"] < regression_result.summary["baseline_metric_value"]
    assert sine_result.summary["primary_metric_value"] < sine_result.summary["baseline_metric_value"]

    best_accuracy = max(row["accuracy"] for row in classification_result.epoch_metrics)
    assert best_accuracy > classification_result.summary["baseline_metric_value"]


def test_benchmark_configs_record_explicit_seed_roles(tmp_path: Path) -> None:
    runs = {
        "toy_regression.py": {"run_seed": 0, "data_seed": 0, "model_init_seed": 0},
        "toy_sine_regression.py": {"run_seed": 3, "data_seed": 3, "model_init_seed": 3},
        "toy_blobs_classification.py": {"run_seed": 11, "data_seed": 11, "model_init_seed": 11},
    }

    for script_name, expected in runs.items():
        result = load_run(script_name)(
            output_root=tmp_path,
            run_id=script_name.replace(".py", "_seeds"),
            plot_energy=False,
        )
        with (result.run_dir / "config.json").open("r", encoding="utf-8") as handle:
            config_payload = json.load(handle)

        assert config_payload["seed"] == expected["run_seed"]
        assert config_payload["seeds"] == expected
        assert config_payload["data"]["data_seed"] == expected["data_seed"]
        assert config_payload["model"]["model_init_seed"] == expected["model_init_seed"]
