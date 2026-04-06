from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .fmpc_tf2 import TF2PresetName, build_tf2_preset_config, run_fmpc_tf2_experiment


IdentitySemanticsOption = bool


@dataclass
class FMPCTF2IdentitySemanticsSuiteConfig:
    """Configuration for the narrow TF2 identity-semantics comparison suite."""

    experiment_name: str = "fmpc_tf2_identity_semantics_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    tf2_preset_names: tuple[TF2PresetName, ...] = (
        "tf2_canonical",
        "tf2_corrective_transport_default",
    )
    feature_aware_tangent_options: tuple[IdentitySemanticsOption, ...] = (False, True)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 60
    batch_size: int = 128
    eval_steps: int = 15
    layer_dims: tuple[int, ...] = (64, 64, 10)

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"


@dataclass
class FMPCTF2IdentitySemanticsSuiteRunResult:
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


def _semantics_label(feature_aware_tangents: bool) -> str:
    return "feature_aware_total_derivative_approx" if feature_aware_tangents else "truncated_identity_approx"


def _candidate_run_id(preset_name: TF2PresetName, feature_aware_tangents: bool, seed: int) -> str:
    semantics_tag = "fat1" if feature_aware_tangents else "fat0"
    return f"{preset_name}_{semantics_tag}_seed{seed}"


def _suite_config_payload(config: FMPCTF2IdentitySemanticsSuiteConfig) -> dict[str, Any]:
    return {
        "phase": "Phase TF2",
        "stage": "ifmpc_bridge_identity_semantics_suite",
        "tf2_preset_names": [str(name) for name in config.tf2_preset_names],
        "feature_aware_tangent_options": [bool(flag) for flag in config.feature_aware_tangent_options],
        "seeds": [int(seed) for seed in config.seeds],
        "validation_only_checkpoint_selection": True,
        "test_report_only": True,
        "goal": "decide whether canonical TF2 should keep truncated identity semantics or switch to feature-aware tangents",
    }


def _failure_row(
    *,
    run_index: int,
    preset_name: TF2PresetName,
    feature_aware_tangents: bool,
    seed: int,
    error: Exception,
) -> dict[str, Any]:
    message = str(error)
    lowered = message.lower()
    return {
        "run_index": int(run_index),
        "preset_name": str(preset_name),
        "feature_aware_tangents": bool(feature_aware_tangents),
        "identity_semantics": _semantics_label(feature_aware_tangents),
        "identity_tangent_mode": "",
        "seed": int(seed),
        "theta_update_cadence": "",
        "micro_steps": None,
        "incremental_weight_updates": None,
        "supervision_policy": "",
        "theta_update_budget": "",
        "checkpoint_selector": "",
        "val_accuracy": None,
        "test_accuracy": None,
        "gate_passing_epoch_count": None,
        "val_transported_final_energy": None,
        "selected_epoch": None,
        "selected_epoch_passes_gate": None,
        "selector_fallback_used": None,
        "run_status": "failed",
        "nan_or_inf_failure": bool("nan" in lowered or "inf" in lowered),
        "failure_type": type(error).__name__,
        "failure_message": message,
        "run_summary_path": "",
    }


def _success_row(
    *,
    run_index: int,
    preset_name: TF2PresetName,
    feature_aware_tangents: bool,
    seed: int,
    result: Any,
    run_dir: Path,
) -> dict[str, Any]:
    summary = result.summary
    return {
        "run_index": int(run_index),
        "preset_name": str(preset_name),
        "feature_aware_tangents": bool(feature_aware_tangents),
        "identity_semantics": _semantics_label(feature_aware_tangents),
        "identity_tangent_mode": str(summary["identity_tangent_mode"]),
        "seed": int(seed),
        "theta_update_cadence": str(summary["theta_update_cadence"]),
        "micro_steps": int(summary["micro_steps"]),
        "incremental_weight_updates": bool(summary["incremental_weight_updates"]),
        "supervision_policy": str(summary["supervision_policy"]),
        "theta_update_budget": str(summary["theta_update_budget"]),
        "checkpoint_selector": str(summary["checkpoint_selector"]),
        "val_accuracy": float(summary["val_accuracy"]),
        "test_accuracy": float(summary["test_accuracy"]),
        "gate_passing_epoch_count": int(summary["gate_passing_epoch_count"]),
        "val_transported_final_energy": float(summary["val_transported_final_energy"]),
        "selected_epoch": int(summary["best_epoch"]),
        "selected_epoch_passes_gate": bool(summary["selected_epoch_passes_gate"]),
        "selector_fallback_used": bool(summary["selector_fallback_used"]),
        "run_status": "completed",
        "nan_or_inf_failure": False,
        "failure_type": "",
        "failure_message": "",
        "run_summary_path": _relative_posix(run_dir, result.run_dir / "summary.json"),
    }


def _group_key(row: dict[str, Any]) -> tuple[str, bool]:
    return str(row["preset_name"]), bool(row["feature_aware_tangents"])


def _successful_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["run_status"]) == "completed"]


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = _successful_rows(rows)
    completed = len(successes)
    failures = len(rows) - completed
    nan_failures = sum(1 for row in rows if bool(row["nan_or_inf_failure"]))
    exemplar = rows[0]
    payload: dict[str, Any] = {
        "preset_name": str(exemplar["preset_name"]),
        "feature_aware_tangents": bool(exemplar["feature_aware_tangents"]),
        "identity_semantics": str(exemplar["identity_semantics"]),
        "identity_tangent_mode": str(successes[0]["identity_tangent_mode"]) if successes else "",
        "theta_update_cadence": str(successes[0]["theta_update_cadence"]) if successes else "",
        "micro_steps": int(successes[0]["micro_steps"]) if successes else None,
        "incremental_weight_updates": bool(successes[0]["incremental_weight_updates"]) if successes else None,
        "supervision_policy": str(successes[0]["supervision_policy"]) if successes else "",
        "theta_update_budget": str(successes[0]["theta_update_budget"]) if successes else "",
        "checkpoint_selector": str(successes[0]["checkpoint_selector"]) if successes else "",
        "num_runs": int(len(rows)),
        "num_completed_runs": int(completed),
        "num_failures": int(failures),
        "nan_or_inf_failure_count": int(nan_failures),
        "successful_fraction": float(completed / float(len(rows))),
    }
    if not successes:
        payload.update(
            {
                "mean_val_accuracy": None,
                "std_val_accuracy": None,
                "mean_test_accuracy": None,
                "std_test_accuracy": None,
                "mean_val_transported_final_energy": None,
                "mean_gate_passing_epoch_count": None,
            }
        )
        return payload

    val_accuracies = [float(row["val_accuracy"]) for row in successes]
    test_accuracies = [float(row["test_accuracy"]) for row in successes]
    energies = [float(row["val_transported_final_energy"]) for row in successes]
    gate_counts = [float(row["gate_passing_epoch_count"]) for row in successes]
    payload.update(
        {
            "mean_val_accuracy": _mean(val_accuracies),
            "std_val_accuracy": _std(val_accuracies),
            "mean_test_accuracy": _mean(test_accuracies),
            "std_test_accuracy": _std(test_accuracies),
            "mean_val_transported_final_energy": _mean(energies),
            "mean_gate_passing_epoch_count": _mean(gate_counts),
        }
    )
    return payload


def _pairwise_delta(
    *,
    truncated_summary: dict[str, Any],
    feature_aware_summary: dict[str, Any],
) -> dict[str, Any]:
    if truncated_summary["mean_test_accuracy"] is None or feature_aware_summary["mean_test_accuracy"] is None:
        return {
            "mean_val_accuracy_delta": None,
            "mean_test_accuracy_delta": None,
            "mean_val_transported_final_energy_delta": None,
            "mean_gate_passing_epoch_count_delta": None,
        }
    return {
        "mean_val_accuracy_delta": float(feature_aware_summary["mean_val_accuracy"])
        - float(truncated_summary["mean_val_accuracy"]),
        "mean_test_accuracy_delta": float(feature_aware_summary["mean_test_accuracy"])
        - float(truncated_summary["mean_test_accuracy"]),
        "mean_val_transported_final_energy_delta": float(feature_aware_summary["mean_val_transported_final_energy"])
        - float(truncated_summary["mean_val_transported_final_energy"]),
        "mean_gate_passing_epoch_count_delta": float(feature_aware_summary["mean_gate_passing_epoch_count"])
        - float(truncated_summary["mean_gate_passing_epoch_count"]),
    }


def _should_promote_feature_aware(canonical_pairwise: dict[str, Any], feature_aware_summary: dict[str, Any]) -> bool:
    if int(feature_aware_summary["num_failures"]) > 0 or int(feature_aware_summary["nan_or_inf_failure_count"]) > 0:
        return False
    mean_test_delta = canonical_pairwise["mean_test_accuracy_delta"]
    mean_val_delta = canonical_pairwise["mean_val_accuracy_delta"]
    gate_delta = canonical_pairwise["mean_gate_passing_epoch_count_delta"]
    if mean_test_delta is None or mean_val_delta is None or gate_delta is None:
        return False
    return (
        float(mean_test_delta) >= 0.01
        and float(mean_val_delta) >= 0.005
        and float(gate_delta) >= 0.0
    )


def _recommended_next_stage(promote_feature_aware: bool, corrective_pairwise: dict[str, Any]) -> str:
    if promote_feature_aware:
        return "promote feature-aware identity semantics inside TF2 before any EF expansion"
    corrective_test_delta = corrective_pairwise["mean_test_accuracy_delta"]
    if corrective_test_delta is not None and float(corrective_test_delta) > 0.0:
        return "keep truncated canonical default, but revisit feature-aware tangents later inside the corrective family"
    return "keep truncated identity default and continue corrective-transport TF2 bridge work"


def run_fmpc_tf2_identity_semantics_suite(
    config: FMPCTF2IdentitySemanticsSuiteConfig,
) -> FMPCTF2IdentitySemanticsSuiteRunResult:
    """Run a narrow matched comparison of truncated vs feature-aware TF2 identity semantics."""

    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, config.resolved_run_id(), config.output_layout)
    )
    _write_json(run_dir / "config.json", _suite_config_payload(config))

    rows: list[dict[str, Any]] = []
    runs_root = run_dir / "runs"
    run_index = 0
    for preset_name in config.tf2_preset_names:
        for feature_aware_tangents in config.feature_aware_tangent_options:
            for seed in config.seeds:
                run_index += 1
                tf2_config = build_tf2_preset_config(
                    preset_name,
                    output_root=runs_root,
                    output_layout="run_id_subdir",
                    run_id=_candidate_run_id(preset_name, feature_aware_tangents, seed),
                    run_seed=seed,
                    data_seed=seed,
                    model_init_seed=seed,
                    psi_init_seed=seed,
                    batch_order_seed=seed,
                    epochs=config.epochs,
                    batch_size=config.batch_size,
                    eval_steps=config.eval_steps,
                    layer_dims=config.layer_dims,
                    feature_aware_tangents=feature_aware_tangents,
                )
                try:
                    result = run_fmpc_tf2_experiment(tf2_config)
                except Exception as error:  # pragma: no cover - stability path
                    rows.append(
                        _failure_row(
                            run_index=run_index,
                            preset_name=preset_name,
                            feature_aware_tangents=feature_aware_tangents,
                            seed=seed,
                            error=error,
                        )
                    )
                else:
                    rows.append(
                        _success_row(
                            run_index=run_index,
                            preset_name=preset_name,
                            feature_aware_tangents=feature_aware_tangents,
                            seed=seed,
                            result=result,
                            run_dir=run_dir,
                        )
                    )

    csv_rows = [
        {
            **row,
            "feature_aware_tangents": str(bool(row["feature_aware_tangents"])),
            "incremental_weight_updates": ""
            if row["incremental_weight_updates"] is None
            else str(bool(row["incremental_weight_updates"])),
            "selected_epoch_passes_gate": ""
            if row["selected_epoch_passes_gate"] is None
            else str(bool(row["selected_epoch_passes_gate"])),
            "selector_fallback_used": ""
            if row["selector_fallback_used"] is None
            else str(bool(row["selector_fallback_used"])),
            "nan_or_inf_failure": str(bool(row["nan_or_inf_failure"])),
        }
        for row in rows
    ]
    _write_csv(run_dir / "aggregate_runs.csv", csv_rows)

    grouped_rows: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for row in rows:
        grouped_rows.setdefault(_group_key(row), []).append(row)

    grouped_summaries = {
        key: _group_summary(group_rows) for key, group_rows in grouped_rows.items()
    }

    mean_std_val_accuracy_by_preset_and_identity_semantics: dict[str, dict[str, Any]] = {}
    mean_std_test_accuracy_by_preset_and_identity_semantics: dict[str, dict[str, Any]] = {}
    mean_val_transported_final_energy_by_preset_and_identity_semantics: dict[str, dict[str, Any]] = {}
    mean_gate_passing_epoch_count_by_preset_and_identity_semantics: dict[str, dict[str, Any]] = {}
    stability_by_preset_and_identity_semantics: dict[str, dict[str, Any]] = {}
    pairwise_semantics_comparison_by_preset: dict[str, dict[str, Any]] = {}

    for preset_name in config.tf2_preset_names:
        truncated_summary = grouped_summaries[(str(preset_name), False)]
        feature_aware_summary = grouped_summaries[(str(preset_name), True)]
        pairwise = _pairwise_delta(
            truncated_summary=truncated_summary,
            feature_aware_summary=feature_aware_summary,
        )
        pairwise_semantics_comparison_by_preset[str(preset_name)] = {
            **pairwise,
            "theta_update_cadence": feature_aware_summary["theta_update_cadence"] or truncated_summary["theta_update_cadence"],
        }
        mean_std_val_accuracy_by_preset_and_identity_semantics[str(preset_name)] = {
            "truncated": {
                "mean": truncated_summary["mean_val_accuracy"],
                "std": truncated_summary["std_val_accuracy"],
            },
            "feature_aware": {
                "mean": feature_aware_summary["mean_val_accuracy"],
                "std": feature_aware_summary["std_val_accuracy"],
            },
        }
        mean_std_test_accuracy_by_preset_and_identity_semantics[str(preset_name)] = {
            "truncated": {
                "mean": truncated_summary["mean_test_accuracy"],
                "std": truncated_summary["std_test_accuracy"],
            },
            "feature_aware": {
                "mean": feature_aware_summary["mean_test_accuracy"],
                "std": feature_aware_summary["std_test_accuracy"],
            },
        }
        mean_val_transported_final_energy_by_preset_and_identity_semantics[str(preset_name)] = {
            "truncated": truncated_summary["mean_val_transported_final_energy"],
            "feature_aware": feature_aware_summary["mean_val_transported_final_energy"],
        }
        mean_gate_passing_epoch_count_by_preset_and_identity_semantics[str(preset_name)] = {
            "truncated": truncated_summary["mean_gate_passing_epoch_count"],
            "feature_aware": feature_aware_summary["mean_gate_passing_epoch_count"],
        }
        stability_by_preset_and_identity_semantics[str(preset_name)] = {
            "truncated": {
                "num_runs": truncated_summary["num_runs"],
                "num_completed_runs": truncated_summary["num_completed_runs"],
                "num_failures": truncated_summary["num_failures"],
                "nan_or_inf_failure_count": truncated_summary["nan_or_inf_failure_count"],
            },
            "feature_aware": {
                "num_runs": feature_aware_summary["num_runs"],
                "num_completed_runs": feature_aware_summary["num_completed_runs"],
                "num_failures": feature_aware_summary["num_failures"],
                "nan_or_inf_failure_count": feature_aware_summary["nan_or_inf_failure_count"],
            },
        }

    canonical_truncated = grouped_summaries[("tf2_canonical", False)]
    canonical_feature_aware = grouped_summaries[("tf2_canonical", True)]
    corrective_pairwise = pairwise_semantics_comparison_by_preset["tf2_corrective_transport_default"]
    promote_feature_aware = _should_promote_feature_aware(
        pairwise_semantics_comparison_by_preset["tf2_canonical"],
        canonical_feature_aware,
    )
    recommended_semantics = (
        "feature_aware_total_derivative_approx"
        if promote_feature_aware
        else "feature_frozen_truncated_identity_approx"
    )

    summary = {
        "phase": "Phase TF2",
        "stage": "ifmpc_bridge_identity_semantics_suite",
        "num_runs": int(len(rows)),
        "mean_std_val_accuracy_by_preset_and_identity_semantics": mean_std_val_accuracy_by_preset_and_identity_semantics,
        "mean_std_test_accuracy_by_preset_and_identity_semantics": mean_std_test_accuracy_by_preset_and_identity_semantics,
        "mean_val_transported_final_energy_by_preset_and_identity_semantics": mean_val_transported_final_energy_by_preset_and_identity_semantics,
        "mean_gate_passing_epoch_count_by_preset_and_identity_semantics": mean_gate_passing_epoch_count_by_preset_and_identity_semantics,
        "stability_by_preset_and_identity_semantics": stability_by_preset_and_identity_semantics,
        "pairwise_semantics_comparison_by_preset": pairwise_semantics_comparison_by_preset,
        "feature_aware_tangents_should_become_canonical_tf2_default": bool(promote_feature_aware),
        "recommended_canonical_tf2_identity_semantics": recommended_semantics,
        "whether_feature_aware_tangents_help_current_empirical_default": bool(
            corrective_pairwise["mean_test_accuracy_delta"] is not None
            and float(corrective_pairwise["mean_test_accuracy_delta"]) > 0.0
            and int(grouped_summaries[("tf2_corrective_transport_default", True)]["num_failures"]) == 0
        ),
        "theta_update_cadence_sensitivity_context": {
            "tf2_canonical": canonical_truncated["theta_update_cadence"],
            "tf2_corrective_transport_default": grouped_summaries[("tf2_corrective_transport_default", False)][
                "theta_update_cadence"
            ],
        },
        "current_tf2_identity_evidence_interpretation": (
            "feature-aware tangents materially improve the canonical TF2 path"
            if promote_feature_aware
            else "truncated identity approximation remains preferred empirically for the canonical TF2 path"
        ),
        "recommended_next_research_stage_after_identity_semantics_decision": _recommended_next_stage(
            promote_feature_aware,
            corrective_pairwise,
        ),
        "decision_rationale": {
            "canonical_mean_val_accuracy_delta_feature_aware_minus_truncated": pairwise_semantics_comparison_by_preset[
                "tf2_canonical"
            ]["mean_val_accuracy_delta"],
            "canonical_mean_test_accuracy_delta_feature_aware_minus_truncated": pairwise_semantics_comparison_by_preset[
                "tf2_canonical"
            ]["mean_test_accuracy_delta"],
            "canonical_mean_val_transported_final_energy_delta_feature_aware_minus_truncated": pairwise_semantics_comparison_by_preset[
                "tf2_canonical"
            ]["mean_val_transported_final_energy_delta"],
            "canonical_mean_gate_passing_epoch_count_delta_feature_aware_minus_truncated": pairwise_semantics_comparison_by_preset[
                "tf2_canonical"
            ]["mean_gate_passing_epoch_count_delta"],
            "canonical_feature_aware_num_failures": int(canonical_feature_aware["num_failures"]),
            "canonical_truncated_num_failures": int(canonical_truncated["num_failures"]),
        },
        "aggregate_csv_path": "aggregate_runs.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF2IdentitySemanticsSuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config),
        aggregate_rows=rows,
        summary=summary,
    )
