from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .fmpc_tf1 import FMPCTF1RunResult, TF1PresetName, build_tf1_preset_config, run_fmpc_tf1_experiment


@dataclass
class FMPCTF1DefaultAdoptionSuiteConfig:
    """Validate the practical effect of adopting the TF1 working-default preset."""

    experiment_name: str = "fmpc_tf1_default_adoption_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    preset_names: tuple[TF1PresetName, ...] = (
        "mechanism_smoke",
        "baseline_comparable",
        "baseline_working_default",
    )
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"


@dataclass
class FMPCTF1DefaultAdoptionSuiteRunResult:
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
    return float(variance ** 0.5)


def _suite_config_payload(config: FMPCTF1DefaultAdoptionSuiteConfig) -> dict[str, Any]:
    return {
        "phase": "Phase Transport Core v1",
        "stage": "teacher_free_fmpc_v1_default_adoption_suite",
        "presets": [str(name) for name in config.preset_names],
        "seeds": [int(seed) for seed in config.seeds],
        "validation_only_gating": True,
        "test_report_only": True,
        "uses_main_tf1_path_only": True,
    }


def _candidate_run_id(preset_name: TF1PresetName, seed: int) -> str:
    return f"{preset_name}_seed{seed}"


def _aggregate_row_from_result(
    run_index: int,
    result: FMPCTF1RunResult,
    preset_name: TF1PresetName,
    seed: int,
    run_dir: Path,
) -> dict[str, Any]:
    summary = result.summary
    return {
        "run_index": int(run_index),
        "preset_name": str(preset_name),
        "seed": int(seed),
        "run_id": str(result.run_dir.name),
        "run_summary_path": _relative_posix(run_dir, result.run_dir / "summary.json"),
        "checkpoint_selector": str(summary["checkpoint_selector"]),
        "selected_epoch": int(summary["best_epoch"]),
        "selected_epoch_passes_gate": bool(summary["selected_epoch_passes_gate"]),
        "gate_passing_epoch_count": int(summary["gate_passing_epoch_count"]),
        "selector_fallback_used": bool(summary["selector_fallback_used"]),
        "val_accuracy": float(summary["val_accuracy"]),
        "test_accuracy": float(summary["test_accuracy"]),
        "val_transported_final_energy": float(summary["val_transported_final_energy"]),
        "val_energy_delta_vs_identity": float(summary["val_energy_delta_vs_identity"]),
        "val_energy_delta_vs_local_field_only": float(summary["val_energy_delta_vs_local_field_only"]),
    }


def _rows_for_preset(rows: list[dict[str, Any]], preset_name: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["preset_name"]) == preset_name]


def _preset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Preset summary requires at least one row.")
    val_accuracies = [float(row["val_accuracy"]) for row in rows]
    test_accuracies = [float(row["test_accuracy"]) for row in rows]
    gate_counts = [float(row["gate_passing_epoch_count"]) for row in rows]
    selected_epochs = [float(row["selected_epoch"]) for row in rows]
    gate_feasible_fraction = float(
        sum(float(row["gate_passing_epoch_count"]) > 0.0 for row in rows) / float(len(rows))
    )
    selector_fallback_fraction = float(
        sum(bool(row["selector_fallback_used"]) for row in rows) / float(len(rows))
    )
    selectors = sorted({str(row["checkpoint_selector"]) for row in rows})
    return {
        "num_runs": int(len(rows)),
        "checkpoint_selectors": selectors,
        "mean_val_accuracy": _mean(val_accuracies),
        "std_val_accuracy": _std(val_accuracies),
        "mean_test_accuracy": _mean(test_accuracies),
        "std_test_accuracy": _std(test_accuracies),
        "mean_gate_passing_epoch_count": _mean(gate_counts),
        "gate_feasible_fraction": gate_feasible_fraction,
        "mean_selected_epoch": _mean(selected_epochs),
        "selector_fallback_fraction": selector_fallback_fraction,
    }


def _recommended_preset(by_preset: dict[str, dict[str, Any]]) -> dict[str, Any]:
    best_name: str | None = None
    best_summary: dict[str, Any] | None = None
    for preset_name, summary in by_preset.items():
        if best_summary is None:
            best_name = preset_name
            best_summary = summary
            continue
        candidate = (
            float(summary["mean_test_accuracy"]),
            float(summary["mean_val_accuracy"]),
            float(summary["gate_feasible_fraction"]),
            -float(summary["selector_fallback_fraction"]),
        )
        incumbent = (
            float(best_summary["mean_test_accuracy"]),
            float(best_summary["mean_val_accuracy"]),
            float(best_summary["gate_feasible_fraction"]),
            -float(best_summary["selector_fallback_fraction"]),
        )
        if candidate > incumbent:
            best_name = preset_name
            best_summary = summary
    if best_name is None or best_summary is None:
        raise ValueError("No preset summary available.")
    return {
        "preset_name": str(best_name),
        "mean_test_accuracy": float(best_summary["mean_test_accuracy"]),
        "mean_val_accuracy": float(best_summary["mean_val_accuracy"]),
        "gate_feasible_fraction": float(best_summary["gate_feasible_fraction"]),
        "checkpoint_selectors": list(best_summary["checkpoint_selectors"]),
    }


def run_fmpc_tf1_default_adoption_suite(
    config: FMPCTF1DefaultAdoptionSuiteConfig,
) -> FMPCTF1DefaultAdoptionSuiteRunResult:
    """Run a narrow adoption validation pass across the three canonical TF1 presets."""

    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, config.resolved_run_id(), config.output_layout)
    )
    _write_json(run_dir / "config.json", _suite_config_payload(config))

    rows: list[dict[str, Any]] = []
    runs_root = run_dir / "runs"
    run_index = 0
    for preset_name in config.preset_names:
        for seed in config.seeds:
            run_index += 1
            child_config = build_tf1_preset_config(
                preset_name,
                run_seed=seed,
                data_seed=seed,
                model_init_seed=seed,
                psi_init_seed=seed,
                batch_order_seed=seed,
                output_root=runs_root,
                output_layout="run_id_subdir",
                run_id=_candidate_run_id(preset_name, seed),
            )
            result = run_fmpc_tf1_experiment(child_config)
            rows.append(_aggregate_row_from_result(run_index, result, preset_name, seed, run_dir))

    csv_rows = [
        {
            **row,
            "selected_epoch_passes_gate": str(bool(row["selected_epoch_passes_gate"])),
            "selector_fallback_used": str(bool(row["selector_fallback_used"])),
        }
        for row in rows
    ]
    _write_csv(run_dir / "aggregate_runs.csv", csv_rows)

    by_preset = {preset_name: _preset_summary(_rows_for_preset(rows, preset_name)) for preset_name in config.preset_names}
    mechanism_summary = by_preset["mechanism_smoke"]
    comparable_summary = by_preset["baseline_comparable"]
    working_summary = by_preset["baseline_working_default"]

    working_improves_over_comparable = bool(
        float(working_summary["mean_test_accuracy"]) > float(comparable_summary["mean_test_accuracy"])
        and float(working_summary["mean_val_accuracy"]) >= float(comparable_summary["mean_val_accuracy"])
        and float(working_summary["gate_feasible_fraction"]) >= float(comparable_summary["gate_feasible_fraction"])
    )
    mechanism_remains_smoke_only = bool(
        float(mechanism_summary["mean_test_accuracy"]) < float(working_summary["mean_test_accuracy"])
        and float(mechanism_summary["mean_test_accuracy"]) < float(comparable_summary["mean_test_accuracy"])
    )

    summary = {
        "phase": "Phase Transport Core v1",
        "stage": "teacher_free_fmpc_v1_default_adoption_suite",
        "num_runs": int(len(rows)),
        "mean_std_val_accuracy_by_preset": {
            preset_name: {
                "mean": float(preset_summary["mean_val_accuracy"]),
                "std": float(preset_summary["std_val_accuracy"]),
            }
            for preset_name, preset_summary in by_preset.items()
        },
        "mean_std_test_accuracy_by_preset": {
            preset_name: {
                "mean": float(preset_summary["mean_test_accuracy"]),
                "std": float(preset_summary["std_test_accuracy"]),
            }
            for preset_name, preset_summary in by_preset.items()
        },
        "mean_gate_passing_epoch_count_by_preset": {
            preset_name: float(preset_summary["mean_gate_passing_epoch_count"])
            for preset_name, preset_summary in by_preset.items()
        },
        "gate_feasible_fraction_by_preset": {
            preset_name: float(preset_summary["gate_feasible_fraction"])
            for preset_name, preset_summary in by_preset.items()
        },
        "mean_selected_epoch_by_preset": {
            preset_name: float(preset_summary["mean_selected_epoch"])
            for preset_name, preset_summary in by_preset.items()
        },
        "by_preset": by_preset,
        "baseline_working_default_improves_over_baseline_comparable": working_improves_over_comparable,
        "mechanism_smoke_remains_smoke_preset_not_practical_default": mechanism_remains_smoke_only,
        "recommended_main_tf1_preset_after_adoption_validation": _recommended_preset(by_preset),
        "aggregate_csv_path": "aggregate_runs.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF1DefaultAdoptionSuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config),
        aggregate_rows=rows,
        summary=summary,
    )
