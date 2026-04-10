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
class FMPCTF1MultiSeedSuiteConfig:
    """Very narrow multiseed confirmation study for the gate-feasible TF1 family."""

    experiment_name: str = "fmpc_tf1_multiseed_suite"
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
class FMPCTF1MultiSeedSuiteRunResult:
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


def _multiseed_suite_config_payload(config: FMPCTF1MultiSeedSuiteConfig) -> dict[str, Any]:
    return {
        "phase": "FMPC Stage 03 Transport Core v1",
        "stage": "teacher_free_fmpc_v1_multiseed_confirmation_suite",
        "search_space": {
            "preset_name": "baseline_comparable",
            "model_variant": "tf1_mlp_aug",
            "transport_steps": 1,
            "warmup_epochs": 5,
            "feature_aware_tangents": False,
            "identity_loss_weight_candidates": [float(value) for value in config.identity_loss_weight_candidates],
            "checkpoint_selectors": [
                "val_transported_final_energy",
                "gate_constrained_val_accuracy",
            ],
            "seeds": [int(value) for value in config.seeds],
            "validation_only_gating": True,
            "test_report_only": True,
        },
    }


def _run_id_for_candidate(seed: int, identity_loss_weight: float) -> str:
    id_label = str(identity_loss_weight).replace(".", "p")
    return f"tf1_mlp_aug_baseline_comparable_w5_s1_id{id_label}_fa0_seed{seed}"


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / float(len(values)))


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    mean_value = float(sum(values) / float(len(values)))
    variance = sum((float(value) - mean_value) ** 2 for value in values) / float(len(values))
    return float(variance ** 0.5)


def _selector_report_for_run(result: FMPCTF1RunResult, config: FMPCTF1Config) -> dict[str, Any]:
    if not result.epoch_metrics:
        raise ValueError("Multiseed TF1 suite requires epoch metrics.")

    selection_diagnostics = result.selection_diagnostics or build_tf1_epoch_selection_diagnostics(
        result.epoch_metrics
    )
    gate_rows = _gate_passing_epoch_rows(result.epoch_metrics)
    best_gate_val_accuracy = (
        float(max(float(row["val_accuracy"]) for row in gate_rows))
        if gate_rows
        else None
    )
    energy_epoch = int(selection_diagnostics["selection_rules"]["val_transported_final_energy"]["selected_epoch"])
    energy_selector_report = _evaluate_epoch_selector(result, config, energy_epoch)

    gate_constrained_report: dict[str, Any] | None = None
    if gate_rows:
        gate_best_row = max(gate_rows, key=lambda row: float(row["val_accuracy"]))
        gate_constrained_report = _evaluate_epoch_selector(result, config, int(gate_best_row["epoch"]))

    return {
        "phase": "FMPC Stage 03 Transport Core v1",
        "stage": "teacher_free_fmpc_v1_multiseed_run",
        "run_id": config.run_id,
        "preset_name": "baseline_comparable",
        "seed": int(config.run_seed),
        "model_variant": config.model_variant,
        "transport_steps": int(config.transport_steps),
        "warmup_epochs": int(config.warmup_epochs),
        "identity_loss_weight": float(config.identity_loss_weight),
        "feature_aware_tangents": bool(config.feature_aware_tangents),
        "whether_any_gate_passing_epoch_exists": bool(gate_rows),
        "number_of_gate_passing_epochs": int(len(gate_rows)),
        "best_val_accuracy_among_gate_passing_epochs": best_gate_val_accuracy,
        "selector_reports": {
            "val_transported_final_energy": energy_selector_report,
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
        "seed": int(config.run_seed),
        "run_summary_path": _relative_posix(run_dir, result.run_dir / "summary.json"),
        "multiseed_selection_summary_path": _relative_posix(
            run_dir,
            result.run_dir / "multiseed_selection_summary.json",
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


def _summary_stats_by_weight(
    rows: list[dict[str, Any]],
    identity_loss_weight_candidates: tuple[float, ...],
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, dict[str, float | None]],
    dict[str, dict[str, float | None]],
    dict[str, dict[str, float | None]],
]:
    gate_feasible_fraction: dict[str, float] = {}
    mean_gate_epochs: dict[str, float] = {}
    mean_val_accuracy: dict[str, dict[str, float | None]] = {}
    mean_test_accuracy: dict[str, dict[str, float | None]] = {}
    std_test_accuracy: dict[str, dict[str, float | None]] = {}
    selectors = ("val_transported_final_energy", "gate_constrained_val_accuracy")
    for identity_loss_weight in identity_loss_weight_candidates:
        key = str(identity_loss_weight)
        weight_rows = [row for row in rows if float(row["identity_loss_weight"]) == float(identity_loss_weight)]
        gate_feasible_fraction[key] = float(
            sum(bool(row["whether_any_gate_passing_epoch_exists"]) for row in weight_rows) / float(len(weight_rows))
        )
        mean_gate_epochs[key] = float(
            sum(float(row["number_of_gate_passing_epochs"]) for row in weight_rows) / float(len(weight_rows))
        )
        mean_val_accuracy[key] = {}
        mean_test_accuracy[key] = {}
        std_test_accuracy[key] = {}
        for selector in selectors:
            val_values = [
                float(row[f"val_accuracy_by_{selector}"])
                for row in weight_rows
                if row[f"val_accuracy_by_{selector}"] is not None
            ]
            test_values = [
                float(row[f"test_accuracy_by_{selector}"])
                for row in weight_rows
                if row[f"test_accuracy_by_{selector}"] is not None
            ]
            mean_val_accuracy[key][selector] = _mean(val_values)
            mean_test_accuracy[key][selector] = _mean(test_values)
            std_test_accuracy[key][selector] = _std(test_values)
    return (
        gate_feasible_fraction,
        mean_gate_epochs,
        mean_val_accuracy,
        mean_test_accuracy,
        std_test_accuracy,
    )


def _winner_by_metric(
    metric_by_weight_and_selector: dict[str, dict[str, float | None]],
    secondary_metric_by_weight_and_selector: dict[str, dict[str, float | None]] | None = None,
) -> dict[str, Any]:
    best_payload: dict[str, Any] | None = None
    for weight_key, selector_payload in metric_by_weight_and_selector.items():
        for selector_name, metric_value in selector_payload.items():
            if metric_value is None:
                continue
            candidate = {
                "identity_loss_weight": float(weight_key),
                "checkpoint_selector": selector_name,
                "metric_value": float(metric_value),
            }
            if secondary_metric_by_weight_and_selector is not None:
                secondary_value = secondary_metric_by_weight_and_selector[weight_key][selector_name]
                candidate["secondary_metric_value"] = secondary_value
            if best_payload is None or float(candidate["metric_value"]) > float(best_payload["metric_value"]):
                best_payload = candidate
    if best_payload is None:
        raise ValueError("No valid winner found for multiseed summary.")
    return best_payload


def run_fmpc_tf1_multiseed_suite(
    config: FMPCTF1MultiSeedSuiteConfig,
) -> FMPCTF1MultiSeedSuiteRunResult:
    """Run a very narrow multiseed confirmation study for the gate-feasible TF1 family."""

    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, config.resolved_run_id(), config.output_layout)
    )
    _write_json(run_dir / "config.json", _multiseed_suite_config_payload(config))

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
            report = _selector_report_for_run(result, child_config)
            _write_json(result.run_dir / "multiseed_selection_summary.json", report)
            rows.append(_aggregate_row_from_report(run_index, result, child_config, report, run_dir))

    csv_rows = [
        {
            **row,
            "whether_any_gate_passing_epoch_exists": str(bool(row["whether_any_gate_passing_epoch_exists"])),
            "feature_aware_tangents": str(bool(row["feature_aware_tangents"])),
        }
        for row in rows
    ]
    _write_csv(run_dir / "aggregate_runs.csv", csv_rows)

    (
        gate_feasible_fraction_by_identity_weight,
        mean_number_of_gate_passing_epochs_by_identity_weight,
        mean_val_accuracy_by_identity_weight_and_selector,
        mean_test_accuracy_by_identity_weight_and_selector,
        std_test_accuracy_by_identity_weight_and_selector,
    ) = _summary_stats_by_weight(rows, config.identity_loss_weight_candidates)

    winner_by_mean_test_accuracy = _winner_by_metric(
        mean_test_accuracy_by_identity_weight_and_selector,
        mean_val_accuracy_by_identity_weight_and_selector,
    )
    winner_by_mean_val_accuracy = _winner_by_metric(
        mean_val_accuracy_by_identity_weight_and_selector,
        mean_test_accuracy_by_identity_weight_and_selector,
    )

    best_test_by_weight: dict[str, dict[str, Any]] = {}
    for weight_key, selector_payload in mean_test_accuracy_by_identity_weight_and_selector.items():
        best_selector = max(
            (
                {
                    "checkpoint_selector": selector_name,
                    "mean_test_accuracy": metric_value,
                    "std_test_accuracy": std_test_accuracy_by_identity_weight_and_selector[weight_key][selector_name],
                }
                for selector_name, metric_value in selector_payload.items()
                if metric_value is not None
            ),
            key=lambda payload: float(payload["mean_test_accuracy"]),
        )
        best_test_by_weight[weight_key] = best_selector

    best_weight_01 = best_test_by_weight["0.1"]
    best_weight_02 = best_test_by_weight["0.2"]
    mean_gap_02_vs_01 = float(best_weight_02["mean_test_accuracy"] - best_weight_01["mean_test_accuracy"])
    variance_threshold = float(
        max(
            float(best_weight_01["std_test_accuracy"] or 0.0),
            float(best_weight_02["std_test_accuracy"] or 0.0),
        )
    )
    identity_0p2_beats_0p1_once_variance_considered = bool(
        mean_gap_02_vs_01 > 0.0 and mean_gap_02_vs_01 > variance_threshold
    )

    recommended_next_canonical_tf1_config = {
        "preset_name": "baseline_comparable",
        "model_variant": "tf1_mlp_aug",
        "transport_steps": 1,
        "warmup_epochs": 5,
        "feature_aware_tangents": False,
        "identity_loss_weight": float(winner_by_mean_test_accuracy["identity_loss_weight"]),
        "checkpoint_selector": str(winner_by_mean_test_accuracy["checkpoint_selector"]),
        "reason": "highest mean test accuracy across multiseed gate-feasible confirmation study",
    }

    gate_constrained_values = [
        value
        for selector_payload in mean_test_accuracy_by_identity_weight_and_selector.values()
        for selector_name, value in selector_payload.items()
        if selector_name == "gate_constrained_val_accuracy" and value is not None
    ]
    energy_values = [
        value
        for selector_payload in mean_test_accuracy_by_identity_weight_and_selector.values()
        for selector_name, value in selector_payload.items()
        if selector_name == "val_transported_final_energy" and value is not None
    ]
    gate_constrained_default_recommended = bool(
        gate_constrained_values
        and energy_values
        and max(float(value) for value in gate_constrained_values) > max(float(value) for value in energy_values)
    )

    summary = {
        "phase": "FMPC Stage 03 Transport Core v1",
        "stage": "teacher_free_fmpc_v1_multiseed_confirmation_suite",
        "num_runs": int(len(rows)),
        "gate_feasible_fraction_by_identity_weight": gate_feasible_fraction_by_identity_weight,
        "mean_number_of_gate_passing_epochs_by_identity_weight": mean_number_of_gate_passing_epochs_by_identity_weight,
        "mean_val_accuracy_by_identity_weight_and_selector": mean_val_accuracy_by_identity_weight_and_selector,
        "mean_test_accuracy_by_identity_weight_and_selector": mean_test_accuracy_by_identity_weight_and_selector,
        "std_test_accuracy_by_identity_weight_and_selector": std_test_accuracy_by_identity_weight_and_selector,
        "winner_by_mean_test_accuracy": winner_by_mean_test_accuracy,
        "winner_by_mean_val_accuracy": winner_by_mean_val_accuracy,
        "best_test_accuracy_selector_by_identity_weight": best_test_by_weight,
        "identity_loss_weight_0p2_beats_0p1_once_multiseed_variance_is_considered": (
            identity_0p2_beats_0p1_once_variance_considered
        ),
        "identity_loss_weight_0p2_vs_0p1_mean_test_accuracy_gap": mean_gap_02_vs_01,
        "identity_loss_weight_0p2_vs_0p1_variance_threshold": variance_threshold,
        "recommended_next_canonical_tf1_config": recommended_next_canonical_tf1_config,
        "gate_constrained_val_accuracy_should_be_default_within_gate_feasible_family": (
            gate_constrained_default_recommended
        ),
        "aggregate_csv_path": "aggregate_runs.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF1MultiSeedSuiteRunResult(
        run_dir=run_dir,
        config=_multiseed_suite_config_payload(config),
        aggregate_rows=rows,
        summary=summary,
    )
