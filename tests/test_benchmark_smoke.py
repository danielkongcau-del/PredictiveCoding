from __future__ import annotations

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
