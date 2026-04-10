"""Small, deterministic stabilization sweep for the standalone digits PC baseline.

This script is intentionally narrow:
- it reuses the existing digits PC training shell
- it selects the best candidate by validation accuracy only
- it is not a formal comparison pipeline
- it is not matched tuning or broad HPO
"""

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

from pc.real_pc import RealPCConfig, RealPCRunResult, run_digits_pc_experiment


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, indent=2)


def _default_candidates() -> list[dict[str, Any]]:
    return [
        {
            "config_id": "cfg_01_baseline",
            "description": "Current canonical digits_pc defaults.",
            "overrides": {},
        },
        {
            "config_id": "cfg_02_wide_hidden",
            "description": "Wider hidden layer only.",
            "overrides": {
                "layer_dims": (64, 64, 10),
            },
        },
        {
            "config_id": "cfg_03_more_epochs",
            "description": "Longer training horizon only.",
            "overrides": {
                "epochs": 60,
            },
        },
        {
            "config_id": "cfg_04_more_steps",
            "description": "More inference/train steps only.",
            "overrides": {
                "train_steps": 30,
                "eval_steps": 30,
            },
        },
        {
            "config_id": "cfg_05_higher_eta_w",
            "description": "Slightly larger parameter learning rate only.",
            "overrides": {
                "eta_w": 0.02,
                "eta_b": 0.02,
            },
        },
        {
            "config_id": "cfg_06_stable_combo",
            "description": "Small combined stability-oriented extension around the baseline.",
            "overrides": {
                "layer_dims": (64, 64, 10),
                "epochs": 60,
                "train_steps": 30,
                "eval_steps": 30,
                "eta_x": 0.10,
                "eta_w": 0.02,
                "eta_b": 0.02,
            },
        },
    ]


def _candidate_hyperparameters(config: RealPCConfig) -> dict[str, Any]:
    return {
        "layer_dims": list(config.layer_dims),
        "hidden_activation": config.hidden_activation,
        "output_activation": config.output_activation,
        "weight_scale": config.weight_scale,
        "sigma2": config.sigma2,
        "eta_x": config.eta_x,
        "eta_w": config.eta_w,
        "eta_b": config.eta_w if config.eta_b is None else config.eta_b,
        "train_steps": config.train_steps,
        "eval_steps": config.train_steps if config.eval_steps is None else config.eval_steps,
        "state_init": config.state_init,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
    }


def _candidate_record(
    candidate_spec: dict[str, Any],
    config: RealPCConfig,
    result: RealPCRunResult,
) -> dict[str, Any]:
    summary = result.summary
    return {
        "config_id": candidate_spec["config_id"],
        "description": candidate_spec["description"],
        "run_dir": str(result.run_dir.as_posix()),
        "summary_path": str((result.run_dir / "summary.json").as_posix()),
        "hyperparameters": _candidate_hyperparameters(config),
        "best_epoch": summary["best_epoch"],
        "val_metric": summary["val_metric"],
        "test_metric": summary["test_metric"],
        "baseline_metric_name": summary["baseline_metric_name"],
        "test_baseline_metric": summary["test_baseline_metric"],
    }


def _select_best_candidate(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("records must contain at least one candidate.")
    best_record = records[0]
    for record in records[1:]:
        if float(record["val_metric"]) > float(best_record["val_metric"]):
            best_record = record
    return best_record


def run(
    output_root: str | Path = "outputs",
    candidate_specs: list[dict[str, Any]] | None = None,
    plot_curves: bool = False,
) -> dict[str, Any]:
    """Run a small deterministic stabilization sweep for digits PC."""
    root = Path(output_root) / "digits_pc_stabilization"
    if root.exists():
        shutil.rmtree(root)
    candidates_dir = root / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    using_default_candidates = candidate_specs is None
    candidate_specs = _default_candidates() if candidate_specs is None else candidate_specs
    if using_default_candidates and not 4 <= len(candidate_specs) <= 8:
        raise ValueError("Default stabilization sweep must keep 4 to 8 candidates.")

    base_config = RealPCConfig()
    records: list[dict[str, Any]] = []

    for candidate_spec in candidate_specs:
        config_id = str(candidate_spec["config_id"])
        overrides = dict(candidate_spec.get("overrides", {}))
        config = RealPCConfig(
            **{
                **base_config.__dict__,
                **overrides,
                "experiment_name": config_id,
                "output_root": candidates_dir,
                "run_id": config_id,
                "plot_curves": plot_curves,
            }
        )
        result = run_digits_pc_experiment(config)
        records.append(_candidate_record(candidate_spec, config, result))

    selected = _select_best_candidate(records)
    canonical_reference = records[0]

    sweep_summary = {
        "phase": "Phase 3",
        "study_type": "digits_pc_stabilization_sweep",
        "dataset_name": "digits",
        "selected_by": "val_metric",
        "selection_metric_source": "val_metric",
        "report_metric_source": "test_metric",
        "config_source": "manual_small_stabilization_sweep",
        "canonical_reference_config_id": canonical_reference["config_id"],
        "selected_config_id": selected["config_id"],
        "selected_candidate": selected,
        "candidates": records,
        "selected_val_metric_minus_reference": float(selected["val_metric"]) - float(canonical_reference["val_metric"]),
        "selected_test_metric_minus_reference": float(selected["test_metric"]) - float(canonical_reference["test_metric"]),
        "notes": [
            "This is a narrow stabilization sweep for the standalone digits PC baseline.",
            "It is not a formal comparison pipeline.",
            "It is not matched tuning and not broad HPO.",
        ],
    }

    _write_json(root / "summary.json", sweep_summary)
    return sweep_summary


def main() -> None:
    """Run the default stabilization sweep and print a short summary."""
    summary = run()
    selected = summary["selected_candidate"]
    print("Digits PC stabilization sweep completed.")
    print("This is a narrow stabilization sweep, not a formal comparison or matched tuning run.")
    print("Output summary: outputs/digits_pc_stabilization/summary.json")
    print(f"Selected by: {summary['selected_by']}")
    print(f"Selected config: {summary['selected_config_id']}")
    print(f"Validation accuracy: {selected['val_metric']:.6f}")
    print(f"Test accuracy: {selected['test_metric']:.6f}")


if __name__ == "__main__":
    main()
