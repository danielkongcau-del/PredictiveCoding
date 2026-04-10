from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .fmpc_tf1 import TF1PresetName, build_tf1_preset_config, run_fmpc_tf1_experiment
from ..real_pc import RealPCConfig, run_digits_pc_experiment


@dataclass
class FMPCTF1ExternalComparisonSuiteConfig:
    """Compare the adopted TF1 presets against the canonical slow-PC digits baseline."""

    experiment_name: str = "fmpc_tf1_external_comparison_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    tf1_preset_names: tuple[TF1PresetName, ...] = (
        "baseline_comparable",
        "baseline_working_default",
    )
    slow_pc_method_name: str = "canonical_slow_pc_digits_baseline"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"


@dataclass
class FMPCTF1ExternalComparisonSuiteRunResult:
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


def _suite_config_payload(config: FMPCTF1ExternalComparisonSuiteConfig) -> dict[str, Any]:
    return {
        "phase": "Phase TF1",
        "stage": "teacher_free_fmpc_v1_external_comparison_suite",
        "methods": {
            "tf1_presets": [str(name) for name in config.tf1_preset_names],
            "slow_pc_reference": str(config.slow_pc_method_name),
        },
        "seeds": [int(seed) for seed in config.seeds],
        "validation_only_gating": True,
        "test_report_only": True,
        "no_selector_policy_override": True,
        "uses_main_tf1_path_only_for_tf1": True,
        "uses_canonical_digits_pc_path_for_reference": True,
    }


def _tf1_run_id(preset_name: TF1PresetName, seed: int) -> str:
    return f"{preset_name}_seed{seed}"


def _slow_pc_run_id(seed: int) -> str:
    return f"canonical_slow_pc_digits_baseline_seed{seed}"


def _method_rows(rows: list[dict[str, Any]], method_name: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["method_name"]) == method_name]


def _tf1_row(run_index: int, run_dir: Path, preset_name: TF1PresetName, seed: int, result: Any) -> dict[str, Any]:
    summary = result.summary
    timing = summary.get("timing", {})
    return {
        "run_index": int(run_index),
        "method_name": str(preset_name),
        "family_or_preset_name": str(preset_name),
        "seed": int(seed),
        "run_id": str(result.run_dir.name),
        "run_summary_path": _relative_posix(run_dir, result.run_dir / "summary.json"),
        "checkpoint_selector": str(summary["checkpoint_selector"]),
        "val_accuracy": float(summary["val_accuracy"]),
        "test_accuracy": float(summary["test_accuracy"]),
        "val_transported_final_energy": float(summary["val_transported_final_energy"]),
        "gate_passing_epoch_count": int(summary["gate_passing_epoch_count"]),
        "runtime_proxy_seconds": float(timing.get("train_wall_time_seconds", 0.0)),
        "uses_transport_only_training": True,
        "uses_slow_iterative_pc_inference": False,
    }


def _slow_pc_row(run_index: int, run_dir: Path, seed: int, result: Any, method_name: str) -> dict[str, Any]:
    summary = result.summary
    timing = summary.get("timing", {})
    return {
        "run_index": int(run_index),
        "method_name": str(method_name),
        "family_or_preset_name": str(method_name),
        "seed": int(seed),
        "run_id": str(result.run_dir.name),
        "run_summary_path": _relative_posix(run_dir, result.run_dir / "summary.json"),
        "checkpoint_selector": "",
        "val_accuracy": float(summary["val_metric"]),
        "test_accuracy": float(summary["test_metric"]),
        "val_transported_final_energy": None,
        "gate_passing_epoch_count": None,
        "runtime_proxy_seconds": float(timing.get("train_wall_time_seconds", 0.0)),
        "uses_transport_only_training": False,
        "uses_slow_iterative_pc_inference": True,
    }


def _method_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Method summary requires at least one row.")
    val_accuracies = [float(row["val_accuracy"]) for row in rows]
    test_accuracies = [float(row["test_accuracy"]) for row in rows]
    runtime_values = [float(row["runtime_proxy_seconds"]) for row in rows]
    gate_values = [
        float(row["gate_passing_epoch_count"])
        for row in rows
        if row["gate_passing_epoch_count"] not in (None, "")
    ]
    return {
        "num_runs": int(len(rows)),
        "checkpoint_selectors": sorted({str(row["checkpoint_selector"]) for row in rows if str(row["checkpoint_selector"])}),
        "mean_val_accuracy": _mean(val_accuracies),
        "std_val_accuracy": _std(val_accuracies),
        "mean_test_accuracy": _mean(test_accuracies),
        "std_test_accuracy": _std(test_accuracies),
        "mean_runtime_proxy_seconds": _mean(runtime_values),
        "mean_gate_passing_epoch_count": None if not gate_values else _mean(gate_values),
    }


def _pairwise_difference(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean_val_accuracy_delta": float(candidate["mean_val_accuracy"]) - float(reference["mean_val_accuracy"]),
        "mean_test_accuracy_delta": float(candidate["mean_test_accuracy"]) - float(reference["mean_test_accuracy"]),
    }


def _recommended_focus(
    working_improves_over_comparable: bool,
    test_gap_to_slow_pc: float,
) -> str:
    if not working_improves_over_comparable:
        return "revisit transport objective"
    if test_gap_to_slow_pc > 0.10:
        return "improve accuracy further before broader rollout"
    return "continue TF1 mainline as-is"


def run_fmpc_tf1_external_comparison_suite(
    config: FMPCTF1ExternalComparisonSuiteConfig,
) -> FMPCTF1ExternalComparisonSuiteRunResult:
    """Run a narrow external-comparison validation pass for the adopted TF1 preset."""

    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, config.resolved_run_id(), config.output_layout)
    )
    _write_json(run_dir / "config.json", _suite_config_payload(config))

    rows: list[dict[str, Any]] = []
    runs_root = run_dir / "runs"
    run_index = 0

    for preset_name in config.tf1_preset_names:
        for seed in config.seeds:
            run_index += 1
            tf1_config = build_tf1_preset_config(
                preset_name,
                run_seed=seed,
                data_seed=seed,
                model_init_seed=seed,
                psi_init_seed=seed,
                batch_order_seed=seed,
                output_root=runs_root,
                output_layout="run_id_subdir",
                run_id=_tf1_run_id(preset_name, seed),
            )
            tf1_result = run_fmpc_tf1_experiment(tf1_config)
            rows.append(_tf1_row(run_index, run_dir, preset_name, seed, tf1_result))

    for seed in config.seeds:
        run_index += 1
        pc_config = RealPCConfig(
            output_root=runs_root,
            output_layout="run_id_subdir",
            run_id=_slow_pc_run_id(seed),
            plot_curves=False,
            run_seed=seed,
            data_seed=seed,
            model_init_seed=seed,
            batch_order_seed=seed,
        )
        pc_result = run_digits_pc_experiment(pc_config)
        rows.append(_slow_pc_row(run_index, run_dir, seed, pc_result, config.slow_pc_method_name))

    csv_rows = [
        {
            **row,
            "uses_transport_only_training": str(bool(row["uses_transport_only_training"])),
            "uses_slow_iterative_pc_inference": str(bool(row["uses_slow_iterative_pc_inference"])),
        }
        for row in rows
    ]
    _write_csv(run_dir / "aggregate_runs.csv", csv_rows)

    method_names = [*config.tf1_preset_names, config.slow_pc_method_name]
    by_method = {method_name: _method_summary(_method_rows(rows, method_name)) for method_name in method_names}
    comparable_summary = by_method["baseline_comparable"]
    working_summary = by_method["baseline_working_default"]
    slow_pc_summary = by_method[config.slow_pc_method_name]

    working_vs_comparable = _pairwise_difference(comparable_summary, working_summary)
    working_vs_slow_pc = {
        "mean_val_accuracy_gap": float(working_summary["mean_val_accuracy"]) - float(slow_pc_summary["mean_val_accuracy"]),
        "mean_test_accuracy_gap": float(working_summary["mean_test_accuracy"]) - float(slow_pc_summary["mean_test_accuracy"]),
    }
    working_improves_over_comparable = bool(
        float(working_summary["mean_test_accuracy"]) > float(comparable_summary["mean_test_accuracy"])
        and float(working_summary["mean_val_accuracy"]) >= float(comparable_summary["mean_val_accuracy"])
    )
    working_remains_main_tf1 = bool(working_improves_over_comparable)
    recommended_focus = _recommended_focus(
        working_improves_over_comparable=working_improves_over_comparable,
        test_gap_to_slow_pc=float(slow_pc_summary["mean_test_accuracy"]) - float(working_summary["mean_test_accuracy"]),
    )

    summary = {
        "phase": "Phase TF1",
        "stage": "teacher_free_fmpc_v1_external_comparison_suite",
        "num_runs": int(len(rows)),
        "mean_std_val_accuracy_by_method": {
            method_name: {
                "mean": float(method_summary["mean_val_accuracy"]),
                "std": float(method_summary["std_val_accuracy"]),
            }
            for method_name, method_summary in by_method.items()
        },
        "mean_std_test_accuracy_by_method": {
            method_name: {
                "mean": float(method_summary["mean_test_accuracy"]),
                "std": float(method_summary["std_test_accuracy"]),
            }
            for method_name, method_summary in by_method.items()
        },
        "mean_gate_passing_epoch_count_by_tf1_preset": {
            preset_name: by_method[preset_name]["mean_gate_passing_epoch_count"] for preset_name in config.tf1_preset_names
        },
        "by_method": by_method,
        "baseline_working_default_over_baseline_comparable": working_vs_comparable,
        "baseline_working_default_vs_slow_pc_baseline_gap": working_vs_slow_pc,
        "baseline_working_default_strong_enough_to_remain_main_tf1_preset": working_remains_main_tf1,
        "recommended_next_research_focus": recommended_focus,
        "aggregate_csv_path": "aggregate_runs.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)

    return FMPCTF1ExternalComparisonSuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config),
        aggregate_rows=rows,
        summary=summary,
    )
