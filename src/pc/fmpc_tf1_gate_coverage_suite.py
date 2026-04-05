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
    _evaluate_slow_pc_accuracy,
    _evaluate_transport_split,
    _restore_mlp_parameters,
    _restore_pc_parameters,
    _snapshot_mlp_parameters,
    _snapshot_pc_parameters,
)
from .metrics import majority_class_baseline_accuracy


@dataclass
class FMPCTF1GateCoverageSuiteConfig:
    """Narrow gate-coverage rescue study for baseline-comparable TF1 runs."""

    experiment_name: str = "fmpc_tf1_gate_coverage_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    warmup_epochs: int = 5
    identity_loss_weight_candidates: tuple[float, ...] = (0.05, 0.1, 0.2)
    feature_aware_tangents_candidates: tuple[bool, ...] = (False, True)

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"


@dataclass
class FMPCTF1GateCoverageSuiteRunResult:
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


def _gate_coverage_suite_config_payload(config: FMPCTF1GateCoverageSuiteConfig) -> dict[str, Any]:
    return {
        "phase": "Phase TF1",
        "stage": "teacher_free_fmpc_v1_gate_coverage_suite",
        "search_space": {
            "preset_name": "baseline_comparable",
            "warmup_epochs": int(config.warmup_epochs),
            "identity_loss_weight_candidates": [float(value) for value in config.identity_loss_weight_candidates],
            "feature_aware_tangents_candidates": [bool(value) for value in config.feature_aware_tangents_candidates],
            "model_variants": ["tf1_mlp_core", "tf1_mlp_aug"],
            "tf1_mlp_core_transport_steps": 2,
            "tf1_mlp_aug_transport_steps": 1,
            "validation_only_gating": True,
            "test_report_only": True,
        },
    }


def _run_id_for_candidate(
    *,
    model_variant: str,
    transport_steps: int,
    identity_loss_weight: float,
    feature_aware_tangents: bool,
) -> str:
    id_label = str(identity_loss_weight).replace(".", "p")
    tangent_label = "fa1" if feature_aware_tangents else "fa0"
    return (
        f"{model_variant}_baseline_comparable_w5_s{transport_steps}"
        f"_id{id_label}_{tangent_label}"
    )


def _snapshot_for_epoch(result: FMPCTF1RunResult, epoch: int):
    if not result.epoch_snapshots:
        raise ValueError("Gate coverage suite requires retained epoch snapshots.")
    for snapshot in result.epoch_snapshots:
        if snapshot.epoch == epoch:
            return snapshot
    raise ValueError(f"No epoch snapshot found for epoch {epoch}.")


def _evaluate_epoch_selector(
    result: FMPCTF1RunResult,
    config: FMPCTF1Config,
    epoch: int,
) -> dict[str, Any]:
    if result.model is None or result.psi_network is None:
        raise ValueError("Gate coverage suite requires returned model and psi_network.")

    from .datasets import load_digits_split

    split_cfg = result.config["dataset"]
    digits_split = load_digits_split(
        split_seed=int(split_cfg["data_seed"]),
        train_fraction=float(split_cfg["train_fraction"]),
        val_fraction=float(split_cfg["val_fraction"]),
        test_fraction=float(split_cfg["test_fraction"]),
    )
    model_snapshot = _snapshot_pc_parameters(result.model)
    psi_snapshot = _snapshot_mlp_parameters(result.psi_network)
    try:
        epoch_snapshot = _snapshot_for_epoch(result, epoch)
        _restore_pc_parameters(result.model, epoch_snapshot.model_snapshot)
        _restore_mlp_parameters(result.psi_network, epoch_snapshot.psi_snapshot)
        val_transport = _evaluate_transport_split(
            result.model,
            result.psi_network,
            config,
            digits_split.x_val,
            digits_split.y_val,
        )
        _, val_accuracy = _evaluate_slow_pc_accuracy(result.model, digits_split.x_val, digits_split.y_val)
        _, test_accuracy = _evaluate_slow_pc_accuracy(result.model, digits_split.x_test, digits_split.y_test)
        val_baseline_accuracy = majority_class_baseline_accuracy(digits_split.y_val)
        gate_passed = bool(
            (val_transport.transported_final_energy < val_transport.identity_final_energy)
            and (val_transport.transported_final_energy <= val_transport.local_field_only_final_energy)
            and (val_accuracy > val_baseline_accuracy)
        )
        return {
            "selected_epoch": int(epoch),
            "val_accuracy": float(val_accuracy),
            "test_accuracy": float(test_accuracy),
            "val_transported_final_energy": float(val_transport.transported_final_energy),
            "val_energy_delta_vs_local_field_only": float(
                val_transport.transported_final_energy - val_transport.local_field_only_final_energy
            ),
            "validation_gate_passed": gate_passed,
        }
    finally:
        _restore_pc_parameters(result.model, model_snapshot)
        _restore_mlp_parameters(result.psi_network, psi_snapshot)


def _gate_passing_epoch_rows(epoch_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in epoch_metrics
        if (
            float(row["val_transported_final_energy"]) < float(row["val_identity_final_energy"])
            and float(row["val_transported_final_energy"]) <= float(row["val_local_field_only_final_energy"])
            and float(row["val_accuracy"]) > float(row["val_baseline_accuracy"])
        )
    ]


def _gate_coverage_report_for_run(
    result: FMPCTF1RunResult,
    config: FMPCTF1Config,
) -> dict[str, Any]:
    if not result.epoch_metrics:
        raise ValueError("Gate coverage suite requires epoch metrics.")

    selection_diagnostics = result.selection_diagnostics or build_tf1_epoch_selection_diagnostics(
        result.epoch_metrics
    )
    gate_rows = _gate_passing_epoch_rows(result.epoch_metrics)
    best_overall_val_accuracy = float(max(float(row["val_accuracy"]) for row in result.epoch_metrics))
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
    gate_constrained_report: dict[str, Any] | None = None
    if gate_rows:
        gate_best_row = max(gate_rows, key=lambda row: float(row["val_accuracy"]))
        gate_constrained_report = _evaluate_epoch_selector(result, config, int(gate_best_row["epoch"]))

    energy_epoch = int(selection_diagnostics["selection_rules"]["val_transported_final_energy"]["selected_epoch"])
    val_accuracy_epoch = int(selection_diagnostics["selection_rules"]["val_accuracy"]["selected_epoch"])
    energy_selector_report = _evaluate_epoch_selector(result, config, energy_epoch)
    val_accuracy_selector_report = _evaluate_epoch_selector(result, config, val_accuracy_epoch)

    return {
        "phase": "Phase TF1",
        "stage": "teacher_free_fmpc_v1_gate_coverage_run",
        "run_id": config.run_id,
        "preset_name": "baseline_comparable",
        "model_variant": config.model_variant,
        "transport_steps": int(config.transport_steps),
        "identity_loss_weight": float(config.identity_loss_weight),
        "feature_aware_tangents": bool(config.feature_aware_tangents),
        "number_of_gate_passing_epochs": int(len(gate_rows)),
        "first_gate_passing_epoch": None if not gate_rows else int(gate_rows[0]["epoch"]),
        "best_val_accuracy_among_gate_passing_epochs": best_gate_val_accuracy,
        "best_val_transported_energy_among_gate_passing_epochs": best_gate_energy,
        "best_overall_val_accuracy": best_overall_val_accuracy,
        "gap_between_best_overall_val_accuracy_and_best_gate_passing_val_accuracy": (
            None if best_gate_val_accuracy is None else float(best_overall_val_accuracy - best_gate_val_accuracy)
        ),
        "whether_any_gate_passing_epoch_exists": bool(gate_rows),
        "selector_reports": {
            "val_transported_final_energy": energy_selector_report,
            "val_accuracy": val_accuracy_selector_report,
            "gate_constrained_val_accuracy": gate_constrained_report,
        },
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
        "run_summary_path": _relative_posix(run_dir, result.run_dir / "summary.json"),
        "gate_coverage_summary_path": _relative_posix(run_dir, result.run_dir / "gate_coverage_summary.json"),
        "model_variant": config.model_variant,
        "transport_steps": int(config.transport_steps),
        "identity_loss_weight": float(config.identity_loss_weight),
        "feature_aware_tangents": bool(config.feature_aware_tangents),
        "number_of_gate_passing_epochs": int(report["number_of_gate_passing_epochs"]),
        "first_gate_passing_epoch": report["first_gate_passing_epoch"],
        "best_val_accuracy_among_gate_passing_epochs": report["best_val_accuracy_among_gate_passing_epochs"],
        "best_val_transported_energy_among_gate_passing_epochs": report[
            "best_val_transported_energy_among_gate_passing_epochs"
        ],
        "best_overall_val_accuracy": float(report["best_overall_val_accuracy"]),
        "gap_between_best_overall_val_accuracy_and_best_gate_passing_val_accuracy": report[
            "gap_between_best_overall_val_accuracy_and_best_gate_passing_val_accuracy"
        ],
        "whether_any_gate_passing_epoch_exists": bool(report["whether_any_gate_passing_epoch_exists"]),
    }
    for selector_name, selector_payload in report["selector_reports"].items():
        row[f"selected_epoch_by_{selector_name}"] = (
            None if selector_payload is None else int(selector_payload["selected_epoch"])
        )
        row[f"val_accuracy_by_{selector_name}"] = (
            None if selector_payload is None else float(selector_payload["val_accuracy"])
        )
        row[f"test_accuracy_by_{selector_name}"] = (
            None if selector_payload is None else float(selector_payload["test_accuracy"])
        )
        row[f"val_transported_final_energy_by_{selector_name}"] = (
            None if selector_payload is None else float(selector_payload["val_transported_final_energy"])
        )
    return row


def _run_candidates(config: FMPCTF1GateCoverageSuiteConfig) -> list[FMPCTF1Config]:
    candidates: list[FMPCTF1Config] = []
    for identity_loss_weight in config.identity_loss_weight_candidates:
        candidates.append(
            build_tf1_baseline_comparable_config(
                model_variant="tf1_mlp_core",
                use_teacher_free_features=False,
                feature_aware_tangents=False,
                warmup_epochs=config.warmup_epochs,
                transport_steps=2,
                identity_loss_weight=identity_loss_weight,
                run_id=_run_id_for_candidate(
                    model_variant="tf1_mlp_core",
                    transport_steps=2,
                    identity_loss_weight=identity_loss_weight,
                    feature_aware_tangents=False,
                ),
            )
        )
        for feature_aware_tangents in config.feature_aware_tangents_candidates:
            candidates.append(
                build_tf1_baseline_comparable_config(
                    model_variant="tf1_mlp_aug",
                    use_teacher_free_features=True,
                    feature_aware_tangents=feature_aware_tangents,
                    warmup_epochs=config.warmup_epochs,
                    transport_steps=1,
                    identity_loss_weight=identity_loss_weight,
                    run_id=_run_id_for_candidate(
                        model_variant="tf1_mlp_aug",
                        transport_steps=1,
                        identity_loss_weight=identity_loss_weight,
                        feature_aware_tangents=feature_aware_tangents,
                    ),
                )
            )
    return candidates


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / float(len(values)))


def run_fmpc_tf1_gate_coverage_suite(
    config: FMPCTF1GateCoverageSuiteConfig,
) -> FMPCTF1GateCoverageSuiteRunResult:
    """Run the narrow TF1 gate-coverage rescue study."""

    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, config.resolved_run_id(), config.output_layout)
    )
    _write_json(run_dir / "config.json", _gate_coverage_suite_config_payload(config))

    rows: list[dict[str, Any]] = []
    runs_root = run_dir / "runs"
    for run_index, child_config in enumerate(_run_candidates(config), start=1):
        child_config.output_root = runs_root
        child_config.output_layout = "run_id_subdir"
        result = run_fmpc_tf1_experiment(child_config)
        report = _gate_coverage_report_for_run(result, child_config)
        _write_json(result.run_dir / "gate_coverage_summary.json", report)
        rows.append(_aggregate_row_from_report(run_index, result, child_config, report, run_dir))

    csv_rows = [
        {
            **row,
            "feature_aware_tangents": str(bool(row["feature_aware_tangents"])),
            "whether_any_gate_passing_epoch_exists": str(bool(row["whether_any_gate_passing_epoch_exists"])),
        }
        for row in rows
    ]
    _write_csv(run_dir / "aggregate_runs.csv", csv_rows)

    gate_feasible_rows = [row for row in rows if bool(row["whether_any_gate_passing_epoch_exists"])]
    gate_feasible_configurations = [
        {
            "run_id": row["run_id"],
            "model_variant": row["model_variant"],
            "transport_steps": int(row["transport_steps"]),
            "identity_loss_weight": float(row["identity_loss_weight"]),
            "feature_aware_tangents": bool(row["feature_aware_tangents"]),
            "number_of_gate_passing_epochs": int(row["number_of_gate_passing_epochs"]),
            "best_val_accuracy_among_gate_passing_epochs": row["best_val_accuracy_among_gate_passing_epochs"],
        }
        for row in gate_feasible_rows
    ]
    only_aug_step1_gate_feasible = bool(gate_feasible_rows) and all(
        str(row["model_variant"]) == "tf1_mlp_aug" and int(row["transport_steps"]) == 1
        for row in gate_feasible_rows
    )
    aug_false_rows = [
        row
        for row in rows
        if str(row["model_variant"]) == "tf1_mlp_aug" and not bool(row["feature_aware_tangents"])
    ]
    aug_true_rows = [
        row
        for row in rows
        if str(row["model_variant"]) == "tf1_mlp_aug" and bool(row["feature_aware_tangents"])
    ]
    avg_gate_epochs_aug_false = _mean([float(row["number_of_gate_passing_epochs"]) for row in aug_false_rows])
    avg_gate_epochs_aug_true = _mean([float(row["number_of_gate_passing_epochs"]) for row in aug_true_rows])
    feature_aware_tangents_help_gate_coverage = bool(
        avg_gate_epochs_aug_true is not None
        and avg_gate_epochs_aug_false is not None
        and avg_gate_epochs_aug_true > avg_gate_epochs_aug_false
    )
    gate_epochs_by_identity_loss_weight: dict[str, dict[str, Any]] = {}
    for identity_loss_weight in config.identity_loss_weight_candidates:
        weight_rows = [
            row for row in rows if float(row["identity_loss_weight"]) == float(identity_loss_weight)
        ]
        gate_epochs_by_identity_loss_weight[str(identity_loss_weight)] = {
            "num_runs": int(len(weight_rows)),
            "num_gate_feasible_runs": int(
                sum(bool(row["whether_any_gate_passing_epoch_exists"]) for row in weight_rows)
            ),
            "average_gate_passing_epochs": _mean(
                [float(row["number_of_gate_passing_epochs"]) for row in weight_rows]
            ),
        }
    gate_feasible_with_gate_selector = [
        row
        for row in gate_feasible_rows
        if row["test_accuracy_by_gate_constrained_val_accuracy"] is not None
    ]
    gate_constrained_mean_gain = _mean(
        [
            float(row["test_accuracy_by_gate_constrained_val_accuracy"])
            - float(row["test_accuracy_by_val_transported_final_energy"])
            for row in gate_feasible_with_gate_selector
        ]
    )
    gate_constrained_accuracy_selection_meaningful = bool(
        gate_constrained_mean_gain is not None and gate_constrained_mean_gain > 0.01
    )
    summary = {
        "phase": "Phase TF1",
        "stage": "teacher_free_fmpc_v1_gate_coverage_suite",
        "num_runs": int(len(rows)),
        "num_gate_feasible_runs": int(len(gate_feasible_rows)),
        "gate_feasible_configurations": gate_feasible_configurations,
        "tf1_mlp_aug_transport_steps_1_is_only_gate_feasible_family": only_aug_step1_gate_feasible,
        "feature_aware_tangents_help_gate_coverage": feature_aware_tangents_help_gate_coverage,
        "average_gate_passing_epochs_tf1_mlp_aug_feature_aware_false": avg_gate_epochs_aug_false,
        "average_gate_passing_epochs_tf1_mlp_aug_feature_aware_true": avg_gate_epochs_aug_true,
        "gate_epochs_by_identity_loss_weight": gate_epochs_by_identity_loss_weight,
        "gate_constrained_accuracy_selection_meaningful": gate_constrained_accuracy_selection_meaningful,
        "mean_test_accuracy_gain_gate_constrained_vs_energy_on_gate_feasible_runs": gate_constrained_mean_gain,
        "aggregate_csv_path": "aggregate_runs.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF1GateCoverageSuiteRunResult(
        run_dir=run_dir,
        config=_gate_coverage_suite_config_payload(config),
        aggregate_rows=rows,
        summary=summary,
    )
