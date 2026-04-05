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
class FMPCTF1SelectionSuiteConfig:
    """Narrow checkpoint-selection diagnostics for baseline-comparable TF1 runs."""

    experiment_name: str = "fmpc_tf1_selection_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    model_variants: tuple[str, ...] = ("tf1_mlp_core", "tf1_mlp_aug")
    warmup_epochs: int = 5
    transport_steps_candidates: tuple[int, ...] = (1, 2)
    identity_loss_weight: float = 0.1
    feature_aware_tangents: bool = False

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"


@dataclass
class FMPCTF1SelectionSuiteRunResult:
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


def _run_id_from_variant(model_variant: str, transport_steps: int) -> str:
    return f"{model_variant}_baseline_comparable_w5_s{transport_steps}_id0p1"


def _selection_suite_config_payload(config: FMPCTF1SelectionSuiteConfig) -> dict[str, Any]:
    return {
        "phase": "Phase TF1",
        "stage": "teacher_free_fmpc_v1_selection_suite",
        "selection_rules": [
            "val_transported_final_energy",
            "val_accuracy",
            "val_energy_delta_vs_local_field_only",
        ],
        "search_space": {
            "preset_name": "baseline_comparable",
            "model_variants": list(config.model_variants),
            "warmup_epochs": int(config.warmup_epochs),
            "transport_steps_candidates": [int(value) for value in config.transport_steps_candidates],
            "identity_loss_weight": float(config.identity_loss_weight),
            "feature_aware_tangents": bool(config.feature_aware_tangents),
            "validation_only_gating": True,
            "test_report_only": True,
        },
    }


def _snapshot_for_epoch(result: FMPCTF1RunResult, epoch: int):
    if not result.epoch_snapshots:
        raise ValueError("TF1 selection suite requires retained epoch snapshots.")
    for snapshot in result.epoch_snapshots:
        if snapshot.epoch == epoch:
            return snapshot
    raise ValueError(f"No epoch snapshot found for epoch {epoch}.")


def _selection_policy_report_for_run(
    result: FMPCTF1RunResult,
    config: FMPCTF1Config,
) -> dict[str, Any]:
    if result.model is None or result.psi_network is None:
        raise ValueError("TF1 selection suite requires returned model and psi_network.")

    split = result.config["dataset"]
    from .datasets import load_digits_split

    digits_split = load_digits_split(
        split_seed=int(split["data_seed"]),
        train_fraction=float(split["train_fraction"]),
        val_fraction=float(split["val_fraction"]),
        test_fraction=float(split["test_fraction"]),
    )
    selection_diagnostics = result.selection_diagnostics or build_tf1_epoch_selection_diagnostics(
        result.epoch_metrics
    )
    model_snapshot = _snapshot_pc_parameters(result.model)
    psi_snapshot = _snapshot_mlp_parameters(result.psi_network)
    reports: dict[str, Any] = {}
    try:
        for rule_name, rule_payload in selection_diagnostics["selection_rules"].items():
            selected_epoch = int(rule_payload["selected_epoch"])
            epoch_snapshot = _snapshot_for_epoch(result, selected_epoch)
            _restore_pc_parameters(result.model, epoch_snapshot.model_snapshot)
            _restore_mlp_parameters(result.psi_network, epoch_snapshot.psi_snapshot)
            val_transport = _evaluate_transport_split(
                result.model,
                result.psi_network,
                config,
                digits_split.x_val,
                digits_split.y_val,
            )
            test_transport = _evaluate_transport_split(
                result.model,
                result.psi_network,
                config,
                digits_split.x_test,
                digits_split.y_test,
            )
            _, val_accuracy = _evaluate_slow_pc_accuracy(result.model, digits_split.x_val, digits_split.y_val)
            _, test_accuracy = _evaluate_slow_pc_accuracy(result.model, digits_split.x_test, digits_split.y_test)
            val_baseline_accuracy = majority_class_baseline_accuracy(digits_split.y_val)
            val_delta_local = (
                val_transport.transported_final_energy - val_transport.local_field_only_final_energy
            )
            reports[rule_name] = {
                "selected_epoch": int(selected_epoch),
                "val_accuracy": float(val_accuracy),
                "test_accuracy": float(test_accuracy),
                "val_transported_final_energy": float(val_transport.transported_final_energy),
                "val_energy_delta_vs_local_field_only": float(val_delta_local),
                "validation_gate_passed": bool(
                    (val_transport.transported_final_energy < val_transport.identity_final_energy)
                    and (val_transport.transported_final_energy <= val_transport.local_field_only_final_energy)
                    and (val_accuracy > val_baseline_accuracy)
                ),
            }
    finally:
        _restore_pc_parameters(result.model, model_snapshot)
        _restore_mlp_parameters(result.psi_network, psi_snapshot)

    return {
        "phase": "Phase TF1",
        "stage": "teacher_free_fmpc_v1_selection_suite_run",
        "preset_name": "baseline_comparable",
        "run_id": config.run_id,
        "model_variant": config.model_variant,
        "warmup_epochs": int(config.warmup_epochs),
        "transport_steps": int(config.transport_steps),
        "identity_loss_weight": float(config.identity_loss_weight),
        "feature_aware_tangents": bool(config.feature_aware_tangents),
        "selection_rules": reports,
    }


def _aggregate_row_from_selection_report(
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
        "selection_policy_summary_path": _relative_posix(
            run_dir,
            result.run_dir / "selection_policy_summary.json",
        ),
        "model_variant": config.model_variant,
        "warmup_epochs": int(config.warmup_epochs),
        "transport_steps": int(config.transport_steps),
        "identity_loss_weight": float(config.identity_loss_weight),
        "feature_aware_tangents": bool(config.feature_aware_tangents),
    }
    for rule_name, rule_payload in report["selection_rules"].items():
        suffix = f"by_{rule_name}"
        row[f"selected_epoch_{suffix}"] = int(rule_payload["selected_epoch"])
        row[f"val_accuracy_{suffix}"] = float(rule_payload["val_accuracy"])
        row[f"test_accuracy_{suffix}"] = float(rule_payload["test_accuracy"])
        row[f"val_transported_final_energy_{suffix}"] = float(
            rule_payload["val_transported_final_energy"]
        )
        row[f"val_energy_delta_vs_local_field_only_{suffix}"] = float(
            rule_payload["val_energy_delta_vs_local_field_only"]
        )
        row[f"validation_gate_passed_{suffix}"] = bool(rule_payload["validation_gate_passed"])
    return row


def _rule_best_row(rows: list[dict[str, Any]], rule_name: str, metric_name: str, *, higher_is_better: bool) -> dict[str, Any]:
    key = f"{metric_name}_by_{rule_name}"
    if higher_is_better:
        return max(rows, key=lambda row: float(row[key]))
    return min(rows, key=lambda row: float(row[key]))


def run_fmpc_tf1_selection_suite(config: FMPCTF1SelectionSuiteConfig) -> FMPCTF1SelectionSuiteRunResult:
    """Run the narrow TF1 checkpoint-selection alignment study."""

    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, config.resolved_run_id(), config.output_layout)
    )
    _write_json(run_dir / "config.json", _selection_suite_config_payload(config))

    rows: list[dict[str, Any]] = []
    runs_root = run_dir / "runs"
    run_index = 0
    for model_variant in config.model_variants:
        for transport_steps in config.transport_steps_candidates:
            run_index += 1
            use_teacher_free_features = model_variant == "tf1_mlp_aug"
            child_config = build_tf1_baseline_comparable_config(
                output_root=runs_root,
                run_id=_run_id_from_variant(model_variant, transport_steps),
                output_layout="run_id_subdir",
                model_variant=model_variant,  # type: ignore[arg-type]
                use_teacher_free_features=use_teacher_free_features,
                feature_aware_tangents=config.feature_aware_tangents,
                warmup_epochs=config.warmup_epochs,
                transport_steps=transport_steps,
                identity_loss_weight=config.identity_loss_weight,
            )
            result = run_fmpc_tf1_experiment(child_config)
            selection_report = _selection_policy_report_for_run(result, child_config)
            _write_json(result.run_dir / "selection_policy_summary.json", selection_report)
            rows.append(
                _aggregate_row_from_selection_report(
                    run_index,
                    result,
                    child_config,
                    selection_report,
                    run_dir,
                )
            )

    csv_rows = [
        {
            **row,
            "feature_aware_tangents": str(bool(row["feature_aware_tangents"])),
            "validation_gate_passed_by_val_transported_final_energy": str(
                bool(row["validation_gate_passed_by_val_transported_final_energy"])
            ),
            "validation_gate_passed_by_val_accuracy": str(bool(row["validation_gate_passed_by_val_accuracy"])),
            "validation_gate_passed_by_val_energy_delta_vs_local_field_only": str(
                bool(row["validation_gate_passed_by_val_energy_delta_vs_local_field_only"])
            ),
        }
        for row in rows
    ]
    _write_csv(run_dir / "aggregate_runs.csv", csv_rows)

    mean_test_accuracy = {
        rule: float(
            sum(float(row[f"test_accuracy_by_{rule}"]) for row in rows) / float(len(rows))
        )
        for rule in (
            "val_transported_final_energy",
            "val_accuracy",
            "val_energy_delta_vs_local_field_only",
        )
    }
    mean_val_accuracy = {
        rule: float(
            sum(float(row[f"val_accuracy_by_{rule}"]) for row in rows) / float(len(rows))
        )
        for rule in (
            "val_transported_final_energy",
            "val_accuracy",
            "val_energy_delta_vs_local_field_only",
        )
    }
    mean_test_accuracy_gain_vs_val_energy = {
        "val_accuracy": float(
            sum(
                float(row["test_accuracy_by_val_accuracy"])
                - float(row["test_accuracy_by_val_transported_final_energy"])
                for row in rows
            )
            / float(len(rows))
        ),
        "val_energy_delta_vs_local_field_only": float(
            sum(
                float(row["test_accuracy_by_val_energy_delta_vs_local_field_only"])
                - float(row["test_accuracy_by_val_transported_final_energy"])
                for row in rows
            )
            / float(len(rows))
        ),
    }
    fraction_runs_improved_test_accuracy = {
        "val_accuracy": float(
            sum(
                float(row["test_accuracy_by_val_accuracy"])
                > float(row["test_accuracy_by_val_transported_final_energy"])
                for row in rows
            )
            / float(len(rows))
        ),
        "val_energy_delta_vs_local_field_only": float(
            sum(
                float(row["test_accuracy_by_val_energy_delta_vs_local_field_only"])
                > float(row["test_accuracy_by_val_transported_final_energy"])
                for row in rows
            )
            / float(len(rows))
        ),
    }
    winner_by_selector = {
        "val_transported_final_energy": _rule_best_row(
            rows,
            "val_transported_final_energy",
            "test_accuracy",
            higher_is_better=True,
        ),
        "val_accuracy": _rule_best_row(rows, "val_accuracy", "test_accuracy", higher_is_better=True),
        "val_energy_delta_vs_local_field_only": _rule_best_row(
            rows,
            "val_energy_delta_vs_local_field_only",
            "test_accuracy",
            higher_is_better=True,
        ),
    }
    selector_changes_recover_accuracy_left_on_table = bool(
        mean_test_accuracy["val_accuracy"] > mean_test_accuracy["val_transported_final_energy"]
        and mean_val_accuracy["val_accuracy"] > mean_val_accuracy["val_transported_final_energy"]
    )
    selection_mismatch_is_substantial = bool(
        (mean_test_accuracy["val_accuracy"] - mean_test_accuracy["val_transported_final_energy"]) > 0.02
        or (mean_val_accuracy["val_accuracy"] - mean_val_accuracy["val_transported_final_energy"]) > 0.05
    )
    val_energy_delta_vs_local_field_only_is_useful = bool(
        mean_test_accuracy["val_energy_delta_vs_local_field_only"]
        > mean_test_accuracy["val_transported_final_energy"]
    )
    summary = {
        "phase": "Phase TF1",
        "stage": "teacher_free_fmpc_v1_selection_suite",
        "num_runs": int(len(rows)),
        "validation_only_gating": True,
        "test_report_only": True,
        "selection_rules": [
            "val_transported_final_energy",
            "val_accuracy",
            "val_energy_delta_vs_local_field_only",
        ],
        "average_test_accuracy_by_selector": mean_test_accuracy,
        "average_val_accuracy_by_selector": mean_val_accuracy,
        "winner_by_selector": winner_by_selector,
        "selector_changes_recover_accuracy_left_on_table": selector_changes_recover_accuracy_left_on_table,
        "selection_mismatch_is_substantial": selection_mismatch_is_substantial,
        "val_energy_delta_vs_local_field_only_is_useful": val_energy_delta_vs_local_field_only_is_useful,
        "winner_by_test_accuracy_under_selection_rule": winner_by_selector,
        "mean_test_accuracy_by_selection_rule": mean_test_accuracy,
        "mean_val_accuracy_by_selection_rule": mean_val_accuracy,
        "mean_test_accuracy_gain_vs_val_energy_selection": mean_test_accuracy_gain_vs_val_energy,
        "fraction_runs_improved_test_accuracy_vs_val_energy_selection": (
            fraction_runs_improved_test_accuracy
        ),
        "aggregate_csv_path": "aggregate_runs.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF1SelectionSuiteRunResult(
        run_dir=run_dir,
        config=_selection_suite_config_payload(config),
        aggregate_rows=rows,
        summary=summary,
    )
