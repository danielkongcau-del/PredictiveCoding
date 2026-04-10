from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .fmpc_tf1 import (
    FMPCTF1Config,
    FMPCTF1RunResult,
    build_tf1_preset_config,
    run_fmpc_tf1_experiment,
)


@dataclass
class FMPCTF1SuiteConfig:
    """Small calibration suite configuration for teacher-free FMPC v1."""

    experiment_name: str = "fmpc_tf1_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    model_variants: tuple[str, ...] = ("tf1_mlp_core", "tf1_mlp_aug")
    warmup_epochs_candidates: tuple[int, ...] = (5, 10)
    transport_steps_candidates: tuple[int, ...] = (1, 2)
    layer_dims_candidates: tuple[tuple[int, ...], ...] = ((64, 16, 10), (64, 64, 10))
    identity_loss_weight_candidates: tuple[float, ...] = (0.0, 0.1)

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"


@dataclass
class FMPCTF1SuiteRunResult:
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


def _preset_name_for_layer_dims(layer_dims: tuple[int, ...]) -> str:
    if layer_dims == (64, 16, 10):
        return "mechanism_smoke"
    if layer_dims == (64, 64, 10):
        return "baseline_comparable"
    raise ValueError(f"No canonical TF1 preset is defined for layer_dims={layer_dims}.")


def _run_id_from_row_index(
    row_index: int,
    *,
    model_variant: str,
    warmup_epochs: int,
    transport_steps: int,
    layer_dims: tuple[int, ...],
    identity_loss_weight: float,
) -> str:
    dims_label = "-".join(str(value) for value in layer_dims)
    id_label = str(identity_loss_weight).replace(".", "p")
    return (
        f"run_{row_index:03d}_{model_variant}_w{warmup_epochs}_s{transport_steps}"
        f"_d{dims_label}_id{id_label}"
    )


def _validation_gate_passed(summary: dict[str, Any]) -> bool:
    gate = summary["validation_gate"]
    return bool(
        gate["passes_identity_comparison"]
        and gate["passes_local_field_only_comparison"]
        and gate["passes_majority_baseline_accuracy"]
    )


def _to_float_or_none(value: float | None) -> float | None:
    return None if value is None else float(value)


def _pearson_correlation(
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
) -> float | None:
    x_values = np.asarray([float(row[x_key]) for row in rows], dtype=np.float64)
    y_values = np.asarray([float(row[y_key]) for row in rows], dtype=np.float64)
    if x_values.size < 2:
        return None
    if np.allclose(np.std(x_values), 0.0) or np.allclose(np.std(y_values), 0.0):
        return None
    return float(np.corrcoef(x_values, y_values)[0, 1])


def _winner_by(rows: list[dict[str, Any]], key: str, *, higher_is_better: bool) -> dict[str, Any]:
    if higher_is_better:
        return max(rows, key=lambda row: float(row[key]))
    return min(rows, key=lambda row: float(row[key]))


def _winner_by_or_none(
    rows: list[dict[str, Any]],
    key: str,
    *,
    higher_is_better: bool,
) -> dict[str, Any] | None:
    if not rows:
        return None
    return _winner_by(rows, key, higher_is_better=higher_is_better)


def _rows_for_preset(rows: list[dict[str, Any]], preset_name: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["preset_name"]) == preset_name]


def _preset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gate_passing_rows = [row for row in rows if bool(row["validation_gate_passed"])]
    correlation = _pearson_correlation(rows, "val_transported_final_energy", "val_accuracy")
    delta_identity_correlation = _pearson_correlation(
        rows,
        "val_energy_delta_vs_identity",
        "val_accuracy",
    )
    delta_local_correlation = _pearson_correlation(
        rows,
        "val_energy_delta_vs_local_field_only",
        "val_accuracy",
    )
    val_energy_winner = _winner_by(rows, "val_transported_final_energy", higher_is_better=False)
    val_accuracy_winner = _winner_by(rows, "val_accuracy", higher_is_better=True)
    test_accuracy_winner = _winner_by(rows, "test_accuracy", higher_is_better=True)
    return {
        "num_runs": int(len(rows)),
        "num_gate_passing_runs": int(len(gate_passing_rows)),
        "val_energy_val_accuracy_pearson_correlation": _to_float_or_none(correlation),
        "val_energy_delta_vs_identity_val_accuracy_pearson_correlation": _to_float_or_none(
            delta_identity_correlation
        ),
        "val_energy_delta_vs_local_field_only_val_accuracy_pearson_correlation": _to_float_or_none(
            delta_local_correlation
        ),
        "smaller_val_transported_energy_predictive_of_better_val_accuracy": bool(
            correlation is not None and correlation < 0.0
        ),
        "winner_by_val_transported_final_energy": val_energy_winner,
        "winner_by_val_accuracy": val_accuracy_winner,
        "winner_by_test_accuracy": test_accuracy_winner,
        "winner_by_val_accuracy_among_gate_passing": _winner_by_or_none(
            gate_passing_rows,
            "val_accuracy",
            higher_is_better=True,
        ),
        "winner_by_test_accuracy_among_gate_passing": _winner_by_or_none(
            gate_passing_rows,
            "test_accuracy",
            higher_is_better=True,
        ),
        "best_val_energy_matches_best_val_accuracy": bool(
            val_energy_winner["run_id"] == val_accuracy_winner["run_id"]
        ),
        "best_val_energy_matches_best_test_accuracy": bool(
            val_energy_winner["run_id"] == test_accuracy_winner["run_id"]
        ),
    }


def _suite_config_payload(config: FMPCTF1SuiteConfig) -> dict[str, Any]:
    return {
        "phase": "FMPC Stage 03 Transport Core v1",
        "stage": "teacher_free_fmpc_v1_calibration_suite",
        "search_space": {
            "model_variants": list(config.model_variants),
            "warmup_epochs_candidates": [int(value) for value in config.warmup_epochs_candidates],
            "transport_steps_candidates": [int(value) for value in config.transport_steps_candidates],
            "layer_dims_candidates": [list(candidate) for candidate in config.layer_dims_candidates],
            "identity_loss_weight_candidates": [
                float(value) for value in config.identity_loss_weight_candidates
            ],
            "feature_aware_tangents": False,
            "validation_only_gating": True,
            "test_report_only": True,
        },
    }


def run_fmpc_tf1_suite(config: FMPCTF1SuiteConfig) -> FMPCTF1SuiteRunResult:
    """Run the small TF1 calibration suite and aggregate energy/accuracy relations."""

    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, config.resolved_run_id(), config.output_layout)
    )
    _write_json(run_dir / "config.json", _suite_config_payload(config))

    rows: list[dict[str, Any]] = []
    row_index = 0
    runs_root = run_dir / "runs"
    for model_variant in config.model_variants:
        for warmup_epochs in config.warmup_epochs_candidates:
            for transport_steps in config.transport_steps_candidates:
                for layer_dims in config.layer_dims_candidates:
                    for identity_loss_weight in config.identity_loss_weight_candidates:
                        row_index += 1
                        preset_name = _preset_name_for_layer_dims(layer_dims)
                        use_teacher_free_features = model_variant == "tf1_mlp_aug"
                        child_config = build_tf1_preset_config(
                            preset_name,  # type: ignore[arg-type]
                            output_root=runs_root,
                            run_id=_run_id_from_row_index(
                                row_index,
                                model_variant=model_variant,
                                warmup_epochs=warmup_epochs,
                                transport_steps=transport_steps,
                                layer_dims=layer_dims,
                                identity_loss_weight=identity_loss_weight,
                            ),
                            output_layout="run_id_subdir",
                            model_variant=model_variant,  # type: ignore[arg-type]
                            use_teacher_free_features=use_teacher_free_features,
                            feature_aware_tangents=False,
                            warmup_epochs=warmup_epochs,
                            transport_steps=transport_steps,
                            layer_dims=layer_dims,
                            identity_loss_weight=identity_loss_weight,
                        )
                        result = run_fmpc_tf1_experiment(child_config)
                        summary = result.summary
                        row = {
                            "run_index": int(row_index),
                            "run_id": child_config.run_id,
                            "preset_name": preset_name,
                            "run_summary_path": _relative_posix(run_dir, result.run_dir / "summary.json"),
                            "model_variant": model_variant,
                            "use_teacher_free_features": bool(use_teacher_free_features),
                            "feature_aware_tangents": False,
                            "warmup_epochs": int(warmup_epochs),
                            "transport_steps": int(transport_steps),
                            "layer_dims": list(layer_dims),
                            "identity_loss_weight": float(identity_loss_weight),
                            "selection_metric": summary["selection_metric"],
                            "selection_metric_source": summary["selection_metric_source"],
                            "report_metric_source": summary["report_metric_source"],
                            "val_transported_final_energy": float(summary["val_transported_final_energy"]),
                            "val_identity_final_energy": float(summary["identity_baseline"]["val_transported_final_energy"]),
                            "val_local_field_only_final_energy": float(
                                summary["local_field_only_baseline"]["val_transported_final_energy"]
                            ),
                            "val_energy_delta_vs_identity": float(summary["val_energy_delta_vs_identity"]),
                            "val_energy_delta_vs_local_field_only": float(
                                summary["val_energy_delta_vs_local_field_only"]
                            ),
                            "val_accuracy": float(summary["val_accuracy"]),
                            "test_accuracy": float(summary["test_accuracy"]),
                            "validation_gate_passed": _validation_gate_passed(summary),
                        }
                        rows.append(row)

    aggregate_rows = [
        {
            **row,
            "layer_dims": "x".join(str(value) for value in row["layer_dims"]),
            "validation_gate_passed": str(bool(row["validation_gate_passed"])),
            "use_teacher_free_features": str(bool(row["use_teacher_free_features"])),
            "feature_aware_tangents": str(bool(row["feature_aware_tangents"])),
        }
        for row in rows
    ]
    _write_csv(run_dir / "aggregate_runs.csv", aggregate_rows)

    gate_passing_rows = [row for row in rows if bool(row["validation_gate_passed"])]
    overall_summary = _preset_summary(rows)
    by_preset = {
        "mechanism_smoke": _preset_summary(_rows_for_preset(rows, "mechanism_smoke")),
        "baseline_comparable": _preset_summary(_rows_for_preset(rows, "baseline_comparable")),
    }
    summary = {
        "phase": "FMPC Stage 03 Transport Core v1",
        "stage": "teacher_free_fmpc_v1_calibration_suite",
        "num_runs": int(len(rows)),
        "validation_only_gating": True,
        "test_report_only": True,
        "num_gate_passing_runs": int(len(gate_passing_rows)),
        "val_energy_val_accuracy_pearson_correlation": overall_summary[
            "val_energy_val_accuracy_pearson_correlation"
        ],
        "val_energy_delta_vs_identity_val_accuracy_pearson_correlation": overall_summary[
            "val_energy_delta_vs_identity_val_accuracy_pearson_correlation"
        ],
        "val_energy_delta_vs_local_field_only_val_accuracy_pearson_correlation": overall_summary[
            "val_energy_delta_vs_local_field_only_val_accuracy_pearson_correlation"
        ],
        "smaller_val_transported_energy_predictive_of_better_val_accuracy": overall_summary[
            "smaller_val_transported_energy_predictive_of_better_val_accuracy"
        ],
        "winner_by_val_transported_final_energy": overall_summary["winner_by_val_transported_final_energy"],
        "winner_by_val_accuracy": overall_summary["winner_by_val_accuracy"],
        "winner_by_test_accuracy": overall_summary["winner_by_test_accuracy"],
        "winner_by_val_accuracy_among_gate_passing": overall_summary[
            "winner_by_val_accuracy_among_gate_passing"
        ],
        "winner_by_test_accuracy_among_gate_passing": overall_summary[
            "winner_by_test_accuracy_among_gate_passing"
        ],
        "best_val_energy_matches_best_val_accuracy": overall_summary[
            "best_val_energy_matches_best_val_accuracy"
        ],
        "best_val_energy_matches_best_test_accuracy": overall_summary[
            "best_val_energy_matches_best_test_accuracy"
        ],
        "by_preset": by_preset,
        "aggregate_csv_path": "aggregate_runs.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF1SuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config),
        aggregate_rows=rows,
        summary=summary,
    )
