from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .fmpc_tf1 import (
    FMPCTF1Config,
    FMPCTF1RunResult,
    build_tf1_baseline_comparable_config,
    build_tf1_epoch_selection_diagnostics,
    run_fmpc_tf1_experiment,
)
from .fmpc_tf1_gate_coverage_suite import _evaluate_epoch_selector, _gate_passing_epoch_rows


@dataclass
class FMPCTF1SelectorPolicySuiteConfig:
    """Narrow selector-cascade study for the current TF1 gate-feasible family."""

    experiment_name: str = "fmpc_tf1_selector_policy_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    identity_loss_weight_candidates: tuple[float, ...] = (0.1, 0.2)

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"


@dataclass
class FMPCTF1SelectorPolicySuiteRunResult:
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


def _suite_config_payload(config: FMPCTF1SelectorPolicySuiteConfig) -> dict[str, Any]:
    return {
        "phase": "Phase Transport Core v1",
        "stage": "teacher_free_fmpc_v1_selector_policy_suite",
        "search_space": {
            "preset_name": "baseline_comparable",
            "model_variant": "tf1_mlp_aug",
            "transport_steps": 1,
            "warmup_epochs": 5,
            "feature_aware_tangents": False,
            "identity_loss_weight_candidates": [float(value) for value in config.identity_loss_weight_candidates],
            "seeds": [int(value) for value in config.seeds],
            "selector_policies": [
                "energy_only",
                "val_accuracy_only",
                "gate_constrained_accuracy_then_energy",
                "gate_constrained_accuracy_then_val_accuracy",
            ],
            "validation_only_gating": True,
            "test_report_only": True,
        },
    }


def _run_id_for_candidate(seed: int, identity_loss_weight: float) -> str:
    id_label = str(identity_loss_weight).replace(".", "p")
    return f"tf1_mlp_aug_baseline_comparable_w5_s1_id{id_label}_fa0_seed{seed}"


def _selector_policy_report_for_run(result: FMPCTF1RunResult, config: FMPCTF1Config) -> dict[str, Any]:
    if not result.epoch_metrics:
        raise ValueError("Selector policy suite requires epoch metrics.")

    selection_diagnostics = result.selection_diagnostics or build_tf1_epoch_selection_diagnostics(
        result.epoch_metrics
    )
    gate_rows = _gate_passing_epoch_rows(result.epoch_metrics)
    gate_exists = bool(gate_rows)
    best_gate_val_accuracy = (
        float(max(float(row["val_accuracy"]) for row in gate_rows))
        if gate_rows
        else None
    )
    best_gate_energy = (
        float(min(float(row["val_transported_final_energy"]) for row in gate_rows))
        if gate_rows
        else None
    )

    energy_epoch = int(selection_diagnostics["selection_rules"]["val_transported_final_energy"]["selected_epoch"])
    val_accuracy_epoch = int(selection_diagnostics["selection_rules"]["val_accuracy"]["selected_epoch"])
    gate_accuracy_epoch = (
        None if not gate_rows else int(max(gate_rows, key=lambda row: float(row["val_accuracy"]))["epoch"])
    )

    policy_epochs = {
        "energy_only": energy_epoch,
        "val_accuracy_only": val_accuracy_epoch,
        "gate_constrained_accuracy_then_energy": (
            gate_accuracy_epoch if gate_accuracy_epoch is not None else energy_epoch
        ),
        "gate_constrained_accuracy_then_val_accuracy": (
            gate_accuracy_epoch if gate_accuracy_epoch is not None else val_accuracy_epoch
        ),
    }

    selector_reports = {
        policy_name: _evaluate_epoch_selector(result, config, selected_epoch)
        for policy_name, selected_epoch in policy_epochs.items()
    }

    return {
        "phase": "Phase Transport Core v1",
        "stage": "teacher_free_fmpc_v1_selector_policy_run",
        "run_id": config.run_id,
        "preset_name": "baseline_comparable",
        "seed": int(config.run_seed),
        "model_variant": config.model_variant,
        "transport_steps": int(config.transport_steps),
        "warmup_epochs": int(config.warmup_epochs),
        "identity_loss_weight": float(config.identity_loss_weight),
        "feature_aware_tangents": bool(config.feature_aware_tangents),
        "whether_any_gate_passing_epoch_exists": gate_exists,
        "number_of_gate_passing_epochs": int(len(gate_rows)),
        "best_val_accuracy_among_gate_passing_epochs": best_gate_val_accuracy,
        "best_val_transported_energy_among_gate_passing_epochs": best_gate_energy,
        "selector_policies": selector_reports,
    }


def _aggregate_row_from_report(
    run_index: int,
    result: FMPCTF1RunResult,
    config: FMPCTF1Config,
    report: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_index": int(run_index),
        "run_id": str(config.run_id),
        "preset_name": "baseline_comparable",
        "seed": int(config.run_seed),
        "run_summary_path": _relative_posix(run_dir, result.run_dir / "summary.json"),
        "selector_policy_summary_path": _relative_posix(
            run_dir,
            result.run_dir / "selector_policy_summary.json",
        ),
        "model_variant": config.model_variant,
        "transport_steps": int(config.transport_steps),
        "warmup_epochs": int(config.warmup_epochs),
        "identity_loss_weight": float(config.identity_loss_weight),
        "feature_aware_tangents": bool(config.feature_aware_tangents),
        "whether_any_gate_passing_epoch_exists": bool(report["whether_any_gate_passing_epoch_exists"]),
        "number_of_gate_passing_epochs": int(report["number_of_gate_passing_epochs"]),
        "best_val_accuracy_among_gate_passing_epochs": report["best_val_accuracy_among_gate_passing_epochs"],
    }
    for policy_name, policy_payload in report["selector_policies"].items():
        row[f"selected_epoch_by_{policy_name}"] = int(policy_payload["selected_epoch"])
        row[f"val_accuracy_by_{policy_name}"] = float(policy_payload["val_accuracy"])
        row[f"test_accuracy_by_{policy_name}"] = float(policy_payload["test_accuracy"])
        row[f"val_transported_final_energy_by_{policy_name}"] = float(
            policy_payload["val_transported_final_energy"]
        )
        row[f"selected_epoch_passes_energy_gate_by_{policy_name}"] = bool(policy_payload["validation_gate_passed"])
    return row


def _winner_by_metric(
    metric_by_weight_and_policy: dict[str, dict[str, float]],
    secondary_metric_by_weight_and_policy: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    best_payload: dict[str, Any] | None = None
    for weight_key, policy_payload in metric_by_weight_and_policy.items():
        for policy_name, metric_value in policy_payload.items():
            candidate = {
                "identity_loss_weight": float(weight_key),
                "selector_policy": policy_name,
                "metric_value": float(metric_value),
            }
            if secondary_metric_by_weight_and_policy is not None:
                candidate["secondary_metric_value"] = secondary_metric_by_weight_and_policy[weight_key][policy_name]
            if best_payload is None or float(candidate["metric_value"]) > float(best_payload["metric_value"]):
                best_payload = candidate
    if best_payload is None:
        raise ValueError("No valid selector-policy winner found.")
    return best_payload


def run_fmpc_tf1_selector_policy_suite(
    config: FMPCTF1SelectorPolicySuiteConfig,
) -> FMPCTF1SelectorPolicySuiteRunResult:
    """Run the narrow TF1 selector-cascade study."""

    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, config.resolved_run_id(), config.output_layout)
    )
    _write_json(run_dir / "config.json", _suite_config_payload(config))

    rows: list[dict[str, Any]] = []
    runs_root = run_dir / "runs"
    run_index = 0
    for identity_loss_weight in config.identity_loss_weight_candidates:
        for seed in config.seeds:
            run_index += 1
            child_config = build_tf1_baseline_comparable_config(
                model_variant="tf1_mlp_aug",
                use_teacher_free_features=True,
                feature_aware_tangents=False,
                transport_steps=1,
                warmup_epochs=5,
                identity_loss_weight=identity_loss_weight,
                run_seed=seed,
                data_seed=seed,
                model_init_seed=seed,
                psi_init_seed=seed,
                batch_order_seed=seed,
                output_root=runs_root,
                output_layout="run_id_subdir",
                run_id=_run_id_for_candidate(seed, identity_loss_weight),
            )
            result = run_fmpc_tf1_experiment(child_config)
            report = _selector_policy_report_for_run(result, child_config)
            _write_json(result.run_dir / "selector_policy_summary.json", report)
            rows.append(_aggregate_row_from_report(run_index, result, child_config, report, run_dir))

    csv_rows = [
        {
            **row,
            "whether_any_gate_passing_epoch_exists": str(bool(row["whether_any_gate_passing_epoch_exists"])),
            "feature_aware_tangents": str(bool(row["feature_aware_tangents"])),
            **{
                key: str(bool(value)) if key.startswith("selected_epoch_passes_energy_gate_by_") else value
                for key, value in row.items()
                if key.startswith("selected_epoch_passes_energy_gate_by_")
            },
        }
        for row in rows
    ]
    _write_csv(run_dir / "aggregate_runs.csv", csv_rows)

    policies = (
        "energy_only",
        "val_accuracy_only",
        "gate_constrained_accuracy_then_energy",
        "gate_constrained_accuracy_then_val_accuracy",
    )
    gate_passing_fraction_by_identity_weight: dict[str, float] = {}
    mean_val_accuracy_by_selector_policy_and_identity_weight: dict[str, dict[str, float]] = {}
    mean_test_accuracy_by_selector_policy_and_identity_weight: dict[str, dict[str, float]] = {}
    std_test_accuracy_by_selector_policy_and_identity_weight: dict[str, dict[str, float]] = {}

    for identity_loss_weight in config.identity_loss_weight_candidates:
        weight_key = str(identity_loss_weight)
        weight_rows = [row for row in rows if float(row["identity_loss_weight"]) == float(identity_loss_weight)]
        gate_passing_fraction_by_identity_weight[weight_key] = float(
            sum(bool(row["whether_any_gate_passing_epoch_exists"]) for row in weight_rows) / float(len(weight_rows))
        )
        mean_val_accuracy_by_selector_policy_and_identity_weight[weight_key] = {}
        mean_test_accuracy_by_selector_policy_and_identity_weight[weight_key] = {}
        std_test_accuracy_by_selector_policy_and_identity_weight[weight_key] = {}
        for policy_name in policies:
            val_values = [float(row[f"val_accuracy_by_{policy_name}"]) for row in weight_rows]
            test_values = [float(row[f"test_accuracy_by_{policy_name}"]) for row in weight_rows]
            mean_val_accuracy_by_selector_policy_and_identity_weight[weight_key][policy_name] = _mean(val_values)
            mean_test_accuracy_by_selector_policy_and_identity_weight[weight_key][policy_name] = _mean(test_values)
            std_test_accuracy_by_selector_policy_and_identity_weight[weight_key][policy_name] = _std(test_values)

    selector_policy_winner_by_mean_test_accuracy = _winner_by_metric(
        mean_test_accuracy_by_selector_policy_and_identity_weight,
        mean_val_accuracy_by_selector_policy_and_identity_weight,
    )
    selector_policy_winner_by_mean_val_accuracy = _winner_by_metric(
        mean_val_accuracy_by_selector_policy_and_identity_weight,
        mean_test_accuracy_by_selector_policy_and_identity_weight,
    )

    best_energy_only = max(
        (
            {
                "identity_loss_weight": float(weight_key),
                "metric_value": float(policy_payload["energy_only"]),
            }
            for weight_key, policy_payload in mean_test_accuracy_by_selector_policy_and_identity_weight.items()
        ),
        key=lambda payload: float(payload["metric_value"]),
    )
    should_replace_energy_only = bool(
        selector_policy_winner_by_mean_test_accuracy["selector_policy"] != "energy_only"
        and float(selector_policy_winner_by_mean_test_accuracy["metric_value"]) > float(best_energy_only["metric_value"])
    )

    summary = {
        "phase": "Phase Transport Core v1",
        "stage": "teacher_free_fmpc_v1_selector_policy_suite",
        "num_runs": int(len(rows)),
        "gate_passing_fraction_by_identity_weight": gate_passing_fraction_by_identity_weight,
        "mean_val_accuracy_by_selector_policy_and_identity_weight": (
            mean_val_accuracy_by_selector_policy_and_identity_weight
        ),
        "mean_test_accuracy_by_selector_policy_and_identity_weight": (
            mean_test_accuracy_by_selector_policy_and_identity_weight
        ),
        "std_test_accuracy_by_selector_policy_and_identity_weight": (
            std_test_accuracy_by_selector_policy_and_identity_weight
        ),
        "selector_policy_winner_by_mean_test_accuracy": selector_policy_winner_by_mean_test_accuracy,
        "selector_policy_winner_by_mean_val_accuracy": selector_policy_winner_by_mean_val_accuracy,
        "selector_cascade_should_replace_energy_only_default": should_replace_energy_only,
        "recommended_default_selector_policy": str(
            selector_policy_winner_by_mean_test_accuracy["selector_policy"]
        ),
        "recommended_working_identity_loss_weight": float(
            selector_policy_winner_by_mean_test_accuracy["identity_loss_weight"]
        ),
        "aggregate_csv_path": "aggregate_runs.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF1SelectorPolicySuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config),
        aggregate_rows=rows,
        summary=summary,
    )
