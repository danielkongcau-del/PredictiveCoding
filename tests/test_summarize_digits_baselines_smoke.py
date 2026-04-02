from __future__ import annotations

import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_run():
    module = runpy.run_path(str(ROOT / "experiments" / "summarize_digits_baselines.py"))
    return module["run"]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def test_digits_baseline_summary_script_reads_existing_summaries_and_writes_artifact(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "outputs"

    common_data = {
        "dataset_name": "digits",
        "dataset_loader": "sklearn.datasets.load_digits",
        "train_size": 1257,
        "val_size": 270,
        "test_size": 270,
    }
    mlp_summary = {
        "phase": "Phase 3",
        "model_family": "mlp",
        "dataset_name": "digits",
        "selection_metric_source": "val_metric",
        "report_metric_source": "test_metric",
        "baseline_metric_name": "baseline_accuracy",
        "best_epoch": 99,
        "val_metric": 0.91,
        "test_metric": 0.94,
        "test_baseline_metric": 0.10,
        "batch_size": 64,
        "batches_per_epoch": 20,
        "epochs": 100,
    }
    mlp_config = {
        "model": {
            "layer_dims": [64, 64, 10],
            "hidden_activation": "tanh",
            "output_activation": "identity",
            "weight_scale": 0.05,
        },
        "training": {
            "eta_w": 0.05,
            "eta_b": 0.05,
        },
        "data": common_data,
    }
    pc_summary = {
        "phase": "Phase 3",
        "model_family": "pc",
        "dataset_name": "digits",
        "selection_metric_source": "val_metric",
        "report_metric_source": "test_metric",
        "baseline_metric_name": "baseline_accuracy",
        "best_epoch": 29,
        "val_metric": 0.58,
        "test_metric": 0.67,
        "test_baseline_metric": 0.10,
        "batch_size": 64,
        "batches_per_epoch": 20,
        "epochs": 30,
        "eta_x": 0.15,
        "eta_w": 0.01,
        "eta_b": 0.01,
        "train_steps": 15,
        "eval_steps": 15,
        "state_init": "forward",
    }
    pc_config = {
        "model": {
            "layer_dims": [64, 32, 10],
            "hidden_activation": "tanh",
            "output_activation": "identity",
            "weight_scale": 0.05,
        },
        "training": {
            "eta_x": 0.15,
            "eta_w": 0.01,
            "eta_b": 0.01,
            "train_steps": 15,
            "eval_steps": 15,
            "state_init": "forward",
        },
        "data": common_data,
    }

    _write_json(output_root / "digits_mlp" / "summary.json", mlp_summary)
    _write_json(output_root / "digits_mlp" / "config.json", mlp_config)
    _write_json(output_root / "digits_pc" / "summary.json", pc_summary)
    _write_json(output_root / "digits_pc" / "config.json", pc_config)

    result = load_run()(output_root=output_root)

    summary_path = output_root / "digits_baselines" / "summary.json"
    assert summary_path.exists()
    with summary_path.open("r", encoding="utf-8") as handle:
        written = json.load(handle)

    assert written == result
    assert written["phase"] == "Phase 3"
    assert written["summary_type"] == "first_pass_real_data_side_by_side"
    assert written["dataset_name"] == "digits"
    assert written["selection_metric_source"] == "val_metric"
    assert written["report_metric_source"] == "test_metric"
    assert written["models"]["mlp"]["best_epoch"] == 99
    assert written["models"]["pc"]["best_epoch"] == 29
    assert written["models"]["mlp"]["hyperparameters"]["eta_w"] == 0.05
    assert written["models"]["pc"]["hyperparameters"]["eta_x"] == 0.15
    assert written["models"]["pc"]["hyperparameters"]["train_steps"] == 15
