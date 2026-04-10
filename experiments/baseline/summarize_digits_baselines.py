from __future__ import annotations

import json
import shutil
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, indent=2)


def _load_baseline_artifacts(output_root: Path, experiment_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = output_root / experiment_name
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing {summary_path}. Run experiments/{experiment_name}.py first."
        )
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing {config_path}. The baseline artifact set is incomplete."
        )
    return _read_json(summary_path), _read_json(config_path)


def _hyperparameter_summary(
    model_family: str,
    summary: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    hyperparameters: dict[str, Any] = {
        "layer_dims": model_config.get("layer_dims"),
        "hidden_activation": model_config.get("hidden_activation"),
        "output_activation": model_config.get("output_activation"),
        "batch_size": summary["batch_size"],
        "epochs": summary["epochs"],
    }
    if model_family == "mlp":
        hyperparameters.update(
            {
                "eta_w": training_config.get("eta_w"),
                "eta_b": training_config.get("eta_b"),
                "weight_scale": model_config.get("weight_scale"),
            }
        )
    elif model_family == "pc":
        hyperparameters.update(
            {
                "eta_x": summary.get("eta_x"),
                "eta_w": summary.get("eta_w"),
                "eta_b": summary.get("eta_b"),
                "train_steps": summary.get("train_steps"),
                "eval_steps": summary.get("eval_steps"),
                "state_init": summary.get("state_init"),
                "weight_scale": model_config.get("weight_scale"),
            }
        )
    return hyperparameters


def _model_entry(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    model_family = str(summary["model_family"])
    return {
        "model_family": model_family,
        "best_epoch": summary["best_epoch"],
        "val_metric": summary["val_metric"],
        "test_metric": summary["test_metric"],
        "baseline_metric_name": summary["baseline_metric_name"],
        "test_baseline_metric": summary["test_baseline_metric"],
        "batch_size": summary["batch_size"],
        "batches_per_epoch": summary["batches_per_epoch"],
        "epochs": summary["epochs"],
        "hyperparameters": _hyperparameter_summary(model_family, summary, config),
    }


def run(
    output_root: str | Path = "outputs",
    write_summary: bool = True,
) -> dict[str, Any]:
    """Summarize standalone digits MLP and PC baseline artifacts side by side."""
    output_root = Path(output_root)
    mlp_summary, mlp_config = _load_baseline_artifacts(output_root, "digits_mlp")
    pc_summary, pc_config = _load_baseline_artifacts(output_root, "digits_pc")

    if mlp_summary.get("dataset_name") != "digits" or pc_summary.get("dataset_name") != "digits":
        raise ValueError("Both summaries must come from the digits dataset.")

    result = {
        "phase": "Phase 3",
        "summary_type": "first_pass_real_data_side_by_side",
        "dataset_name": "digits",
        "selection_metric_source": mlp_summary["selection_metric_source"],
        "report_metric_source": mlp_summary["report_metric_source"],
        "source_artifacts": {
            "mlp_summary_path": str((output_root / "digits_mlp" / "summary.json").as_posix()),
            "pc_summary_path": str((output_root / "digits_pc" / "summary.json").as_posix()),
        },
        "models": {
            "mlp": _model_entry(mlp_summary, mlp_config),
            "pc": _model_entry(pc_summary, pc_config),
        },
        "first_pass_test_metric_difference_pc_minus_mlp": float(
            pc_summary["test_metric"] - mlp_summary["test_metric"]
        ),
        "notes": [
            "This is a standalone side-by-side summary of existing digits baselines.",
            "It is not the formal Phase 3 comparison pipeline.",
            "No matched tuning or fair-comparison claim is implied by this artifact.",
        ],
    }

    if write_summary:
        run_dir = output_root / "digits_baselines"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(run_dir / "summary.json", result)

    return result


def main() -> None:
    """Print a short standalone summary for the current digits baselines."""
    result = run()
    mlp = result["models"]["mlp"]
    pc = result["models"]["pc"]
    print("Digits baseline side-by-side summary completed.")
    print("Source artifacts: outputs/digits_mlp/summary.json, outputs/digits_pc/summary.json")
    print("Output summary: outputs/digits_baselines/summary.json")
    print("This is a first-pass standalone summary, not a matched comparison pipeline.")
    print(f"MLP: best_epoch={mlp['best_epoch']}, val_accuracy={mlp['val_metric']:.6f}, test_accuracy={mlp['test_metric']:.6f}")
    print(f"PC:  best_epoch={pc['best_epoch']}, val_accuracy={pc['val_metric']:.6f}, test_accuracy={pc['test_metric']:.6f}")


if __name__ == "__main__":
    main()
