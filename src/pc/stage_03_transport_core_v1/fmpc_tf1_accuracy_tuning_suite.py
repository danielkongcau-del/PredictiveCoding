from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .fmpc_tf1 import build_tf1_baseline_working_default_config, run_fmpc_tf1_experiment


@dataclass
class FMPCTF1AccuracyTuningSuiteConfig:
    """Very narrow accuracy-improvement study on top of the TF1 working default."""

    experiment_name: str = "fmpc_tf1_accuracy_tuning_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    identity_loss_weight_candidates: tuple[float, ...] = (0.1, 0.2, 0.3)
    hybrid_ramp_epochs_candidates: tuple[int, ...] = (5, 10)
    bootstrap_substeps_candidates: tuple[int, ...] = (4, 8)
    slow_pc_reference_summary_path: str | Path = "outputs/fmpc_tf1_external_comparison_suite/aggregate_summary.json"
    material_gap_narrowing_threshold: float = 0.02

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"


@dataclass
class FMPCTF1AccuracyTuningSuiteRunResult:
    run_dir: Path
    config: dict[str, Any]
    aggregate_rows: list[dict[str, Any]]
    summary: dict[str, Any]


def _resolve_run_dir(output_root: str | Path, experiment_name: str, run_id: str, output_layout: str) -> Path:
    if output_layout == "single_dir":
        return Path(output_root) / experiment_name
    if output_layout == "run_id_subdir":
        return Path(output_root) / experiment_name / run_id
    raise ValueError(f"Unsupported output_layout '{output_layout}'.")


def _prepare_run_dir(run_dir: Path) -> Path:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("rows must contain at least one entry.")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _relative_posix(base_dir: Path, target: Path) -> str:
    return target.relative_to(base_dir).as_posix()


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Mean requires at least one value.")
    return float(sum(values) / float(len(values)))


def _std(values: list[float]) -> float:
    if not values:
        raise ValueError("Std requires at least one value.")
    mean_value = _mean(values)
    variance = sum((float(value) - mean_value) ** 2 for value in values) / float(len(values))
    return float(variance**0.5)


def _config_key(identity_loss_weight: float, hybrid_ramp_epochs: int, bootstrap_substeps: int) -> str:
    id_label = str(identity_loss_weight).replace(".", "p")
    return f"id{id_label}_r{hybrid_ramp_epochs}_b{bootstrap_substeps}"


def _suite_config_payload(config: FMPCTF1AccuracyTuningSuiteConfig) -> dict[str, Any]:
    return {
        "phase": "FMPC Stage 03 Transport Core v1",
        "stage": "teacher_free_fmpc_v1_accuracy_tuning_suite",
        "base_preset": "baseline_working_default",
        "fixed_family": {
            "model_variant": "tf1_mlp_aug",
            "transport_steps": 1,
            "warmup_epochs": 5,
            "feature_aware_tangents": False,
            "checkpoint_selector": "gate_constrained_accuracy_then_val_accuracy",
        },
        "search_space": {
            "identity_loss_weight_candidates": [float(v) for v in config.identity_loss_weight_candidates],
            "hybrid_ramp_epochs_candidates": [int(v) for v in config.hybrid_ramp_epochs_candidates],
            "bootstrap_substeps_candidates": [int(v) for v in config.bootstrap_substeps_candidates],
            "seeds": [int(v) for v in config.seeds],
        },
        "slow_pc_reference_summary_path": str(Path(config.slow_pc_reference_summary_path).as_posix()),
        "material_gap_narrowing_threshold": float(config.material_gap_narrowing_threshold),
    }


def _load_slow_pc_reference_mean_test_accuracy(path: str | Path) -> float:
    summary_path = Path(path)
    if not summary_path.exists():
        raise FileNotFoundError(
            "TF1 accuracy-tuning suite requires the external-comparison summary to exist at "
            f"'{summary_path.as_posix()}'."
        )
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return float(payload["mean_std_test_accuracy_by_method"]["canonical_slow_pc_digits_baseline"]["mean"])


def _run_id_for_candidate(identity_loss_weight: float, hybrid_ramp_epochs: int, bootstrap_substeps: int, seed: int) -> str:
    return f"{_config_key(identity_loss_weight, hybrid_ramp_epochs, bootstrap_substeps)}_seed{seed}"


def _aggregate_row(
    run_index: int,
    run_dir: Path,
    seed: int,
    identity_loss_weight: float,
    hybrid_ramp_epochs: int,
    bootstrap_substeps: int,
    result: Any,
) -> dict[str, Any]:
    summary = result.summary
    return {
        "run_index": int(run_index),
        "config_key": _config_key(identity_loss_weight, hybrid_ramp_epochs, bootstrap_substeps),
        "seed": int(seed),
        "identity_loss_weight": float(identity_loss_weight),
        "hybrid_ramp_epochs": int(hybrid_ramp_epochs),
        "bootstrap_substeps": int(bootstrap_substeps),
        "run_id": str(result.run_dir.name),
        "run_summary_path": _relative_posix(run_dir, result.run_dir / "summary.json"),
        "val_accuracy": float(summary["val_accuracy"]),
        "test_accuracy": float(summary["test_accuracy"]),
        "gate_passing_epoch_count": int(summary["gate_passing_epoch_count"]),
        "val_transported_final_energy": float(summary["val_transported_final_energy"]),
        "selected_epoch": int(summary["best_epoch"]),
        "selected_epoch_passes_gate": bool(summary["selected_epoch_passes_gate"]),
        "selector_fallback_used": bool(summary["selector_fallback_used"]),
    }


def _rows_for_config(rows: list[dict[str, Any]], config_key: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["config_key"]) == config_key]


def _config_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Configuration summary requires at least one row.")
    val_accuracies = [float(row["val_accuracy"]) for row in rows]
    test_accuracies = [float(row["test_accuracy"]) for row in rows]
    gate_counts = [float(row["gate_passing_epoch_count"]) for row in rows]
    return {
        "identity_loss_weight": float(rows[0]["identity_loss_weight"]),
        "hybrid_ramp_epochs": int(rows[0]["hybrid_ramp_epochs"]),
        "bootstrap_substeps": int(rows[0]["bootstrap_substeps"]),
        "mean_val_accuracy": _mean(val_accuracies),
        "std_val_accuracy": _std(val_accuracies),
        "mean_test_accuracy": _mean(test_accuracies),
        "std_test_accuracy": _std(test_accuracies),
        "mean_gate_passing_epoch_count": _mean(gate_counts),
    }


def run_fmpc_tf1_accuracy_tuning_suite(
    config: FMPCTF1AccuracyTuningSuiteConfig,
) -> FMPCTF1AccuracyTuningSuiteRunResult:
    """Run a tiny accuracy-improvement study around the TF1 working default."""

    slow_pc_mean_test_accuracy = _load_slow_pc_reference_mean_test_accuracy(config.slow_pc_reference_summary_path)

    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, config.resolved_run_id(), config.output_layout)
    )
    _write_json(run_dir / "config.json", _suite_config_payload(config))

    rows: list[dict[str, Any]] = []
    runs_root = run_dir / "runs"
    run_index = 0
    config_keys: list[str] = []
    for identity_loss_weight in config.identity_loss_weight_candidates:
        for hybrid_ramp_epochs in config.hybrid_ramp_epochs_candidates:
            for bootstrap_substeps in config.bootstrap_substeps_candidates:
                config_key = _config_key(identity_loss_weight, hybrid_ramp_epochs, bootstrap_substeps)
                config_keys.append(config_key)
                for seed in config.seeds:
                    run_index += 1
                    child_config = build_tf1_baseline_working_default_config(
                        identity_loss_weight=identity_loss_weight,
                        hybrid_ramp_epochs=hybrid_ramp_epochs,
                        bootstrap_substeps=bootstrap_substeps,
                        run_seed=seed,
                        data_seed=seed,
                        model_init_seed=seed,
                        psi_init_seed=seed,
                        batch_order_seed=seed,
                        output_root=runs_root,
                        output_layout="run_id_subdir",
                        run_id=_run_id_for_candidate(identity_loss_weight, hybrid_ramp_epochs, bootstrap_substeps, seed),
                    )
                    result = run_fmpc_tf1_experiment(child_config)
                    rows.append(
                        _aggregate_row(
                            run_index,
                            run_dir,
                            seed,
                            identity_loss_weight,
                            hybrid_ramp_epochs,
                            bootstrap_substeps,
                            result,
                        )
                    )

    csv_rows = [
        {
            **row,
            "selected_epoch_passes_gate": str(bool(row["selected_epoch_passes_gate"])),
            "selector_fallback_used": str(bool(row["selector_fallback_used"])),
        }
        for row in rows
    ]
    _write_csv(run_dir / "aggregate_runs.csv", csv_rows)

    by_configuration = {config_key: _config_summary(_rows_for_config(rows, config_key)) for config_key in config_keys}
    best_configuration_key = max(
        by_configuration,
        key=lambda key: (
            float(by_configuration[key]["mean_test_accuracy"]),
            float(by_configuration[key]["mean_val_accuracy"]),
            float(by_configuration[key]["mean_gate_passing_epoch_count"]),
        ),
    )
    current_default_key = _config_key(0.2, 10, 4)
    current_default_summary = by_configuration[current_default_key]
    best_summary = by_configuration[best_configuration_key]

    current_gap_to_slow_pc = float(slow_pc_mean_test_accuracy) - float(current_default_summary["mean_test_accuracy"])
    best_gap_to_slow_pc = float(slow_pc_mean_test_accuracy) - float(best_summary["mean_test_accuracy"])
    gap_reduction = float(current_gap_to_slow_pc - best_gap_to_slow_pc)
    materially_narrows_gap = bool(gap_reduction >= float(config.material_gap_narrowing_threshold))

    recommended_next_working_default = {
        "config_key": best_configuration_key,
        "identity_loss_weight": float(best_summary["identity_loss_weight"]),
        "hybrid_ramp_epochs": int(best_summary["hybrid_ramp_epochs"]),
        "bootstrap_substeps": int(best_summary["bootstrap_substeps"]),
        "changed_from_current_default": bool(best_configuration_key != current_default_key),
    }

    summary = {
        "phase": "FMPC Stage 03 Transport Core v1",
        "stage": "teacher_free_fmpc_v1_accuracy_tuning_suite",
        "num_runs": int(len(rows)),
        "slow_pc_reference_mean_test_accuracy": float(slow_pc_mean_test_accuracy),
        "current_working_default_configuration": current_default_key,
        "mean_std_val_accuracy_by_configuration": {
            config_key: {
                "mean": float(config_summary["mean_val_accuracy"]),
                "std": float(config_summary["std_val_accuracy"]),
            }
            for config_key, config_summary in by_configuration.items()
        },
        "mean_std_test_accuracy_by_configuration": {
            config_key: {
                "mean": float(config_summary["mean_test_accuracy"]),
                "std": float(config_summary["std_test_accuracy"]),
            }
            for config_key, config_summary in by_configuration.items()
        },
        "mean_gate_passing_epoch_count_by_configuration": {
            config_key: float(config_summary["mean_gate_passing_epoch_count"])
            for config_key, config_summary in by_configuration.items()
        },
        "by_configuration": by_configuration,
        "best_configuration_by_mean_test_accuracy": {
            "config_key": best_configuration_key,
            "identity_loss_weight": float(best_summary["identity_loss_weight"]),
            "hybrid_ramp_epochs": int(best_summary["hybrid_ramp_epochs"]),
            "bootstrap_substeps": int(best_summary["bootstrap_substeps"]),
            "mean_test_accuracy": float(best_summary["mean_test_accuracy"]),
            "mean_val_accuracy": float(best_summary["mean_val_accuracy"]),
            "mean_gate_passing_epoch_count": float(best_summary["mean_gate_passing_epoch_count"]),
        },
        "current_working_default_gap_to_slow_pc_baseline": float(current_gap_to_slow_pc),
        "best_configuration_gap_to_slow_pc_baseline": float(best_gap_to_slow_pc),
        "gap_reduction_vs_current_working_default": float(gap_reduction),
        "material_gap_narrowing_threshold": float(config.material_gap_narrowing_threshold),
        "any_configuration_materially_narrows_gap_to_slow_pc_baseline": materially_narrows_gap,
        "recommended_next_working_default": recommended_next_working_default,
        "aggregate_csv_path": "aggregate_runs.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)

    return FMPCTF1AccuracyTuningSuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config),
        aggregate_rows=rows,
        summary=summary,
    )
