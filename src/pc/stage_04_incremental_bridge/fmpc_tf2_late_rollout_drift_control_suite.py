from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..datasets import load_digits_split
from ..stage_03_transport_core_v1.fmpc_tf1_flow import build_tf1_context
from .fmpc_tf2 import (
    build_tf2_corrective_transport_terminal_angleclip_default_config,
    run_fmpc_tf2_experiment,
)
from .fmpc_tf2_basis_drift_localization_suite import (
    _contribution_from_series,
    _rms_gap,
    _slow_pc_penultimate_by_knot,
)
from .fmpc_tf2_endpoint_basis_suite import (
    _delta_geometry,
    _mean,
    _mean_list,
    _relative_posix,
    _rowspace_basis,
    _runtime_proxy_seconds,
    _std,
    _transport_penultimate_by_knot,
)


@dataclass(frozen=True)
class _CaseSpec:
    case_name: str
    description: str
    intervention_step_offsets: tuple[int, ...]


@dataclass
class FMPCTF2LateRolloutDriftControlSuiteConfig:
    """Run a narrow adopted-package late-rollout drift-control diagnostic."""

    experiment_name: str = "fmpc_tf2_late_rollout_drift_control_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 60
    batch_size: int = 128
    eval_steps: int = 15
    layer_dims: tuple[int, ...] = (64, 64, 10)
    include_preterminal_only_reference: bool = True
    material_val_gain_threshold: float = 0.005
    max_gate_count_drop_for_adoption: float = 1.0

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"

    def case_specs(self) -> tuple[_CaseSpec, ...]:
        specs: list[_CaseSpec] = [
            _CaseSpec(
                case_name="adopted_control_terminal_only",
                description="Current adopted default: full-vector hard 30 degree cone on the terminal micro-step only.",
                intervention_step_offsets=(-1,),
            ),
            _CaseSpec(
                case_name="same_geometry_penultimate_plus_terminal",
                description="Apply the same full-vector hard 30 degree cone on the penultimate and terminal micro-steps.",
                intervention_step_offsets=(-2, -1),
            ),
            _CaseSpec(
                case_name="same_geometry_last_two_preterminal_plus_terminal",
                description="Apply the same full-vector hard 30 degree cone on the last two preterminal micro-steps plus terminal.",
                intervention_step_offsets=(-3, -2, -1),
            ),
        ]
        if self.include_preterminal_only_reference:
            specs.append(
                _CaseSpec(
                    case_name="same_geometry_penultimate_only_reference",
                    description="Diagnostic-only control: apply the same full-vector hard 30 degree cone on the penultimate micro-step only.",
                    intervention_step_offsets=(-2,),
                )
            )
        return tuple(specs)


@dataclass
class FMPCTF2LateRolloutDriftControlSuiteRunResult:
    run_dir: Path
    config: dict[str, Any]
    aggregate_rows: list[dict[str, Any]]
    knot_rows: list[dict[str, Any]]
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


def _suite_config_payload(config: FMPCTF2LateRolloutDriftControlSuiteConfig) -> dict[str, Any]:
    return {
        "phase": "FMPC Stage 04 Incremental Bridge",
        "stage": "adopted_package_late_rollout_drift_control_diagnostic",
        "base_preset_name": "tf2_corrective_transport_terminal_angleclip_default",
        "transport_family_fixed": True,
        "same_geometry_control_family": "local_field_direction_angle_clip_keep_live_norm",
        "case_specs": [
            {
                "case_name": spec.case_name,
                "description": spec.description,
                "intervention_step_offsets": [int(value) for value in spec.intervention_step_offsets],
            }
            for spec in config.case_specs()
        ],
        "seeds": [int(seed) for seed in config.seeds],
        "thresholds": {
            "material_val_gain_threshold": float(config.material_val_gain_threshold),
            "max_gate_count_drop_for_adoption": float(config.max_gate_count_drop_for_adoption),
        },
    }


def _run_id(case_name: str, seed: int) -> str:
    return f"{case_name}_s{seed}"


def _mean_bool(rows: list[dict[str, Any]], field_name: str) -> float:
    return _mean([1.0 if bool(row[field_name]) else 0.0 for row in rows])


def _rows_for_case(rows: list[dict[str, Any]], case_name: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["case_name"]) == case_name]


def _pairwise_delta(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]) -> dict[str, float]:
    left_by_seed = {int(row["seed"]): row for row in left_rows}
    right_by_seed = {int(row["seed"]): row for row in right_rows}
    shared_seeds = sorted(set(left_by_seed) & set(right_by_seed))
    if not shared_seeds:
        raise ValueError("Pairwise comparison requires at least one shared seed.")

    def mean_delta(metric_name: str) -> float:
        return _mean(
            [float(left_by_seed[seed][metric_name]) - float(right_by_seed[seed][metric_name]) for seed in shared_seeds]
        )

    return {
        "mean_val_accuracy_delta": mean_delta("val_accuracy"),
        "mean_test_accuracy_delta": mean_delta("test_accuracy"),
        "mean_gate_passing_epoch_count_delta": mean_delta("gate_passing_epoch_count"),
        "selected_epoch_passes_gate_rate_delta": mean_delta("selected_epoch_passes_gate_flag"),
        "selector_fallback_used_rate_delta": mean_delta("selector_fallback_used_flag"),
        "mean_val_transported_final_energy_delta": mean_delta("val_transported_final_energy"),
        "mean_val_report_output_mse_delta": mean_delta("val_report_output_mse"),
        "mean_val_terminal_rowspace_rms_delta": mean_delta("val_terminal_rowspace_rms"),
        "mean_val_terminal_rowspace_fraction_delta": mean_delta("val_terminal_rowspace_fraction"),
        "mean_val_preterminal_rowspace_share_delta": mean_delta("val_preterminal_rowspace_share"),
        "mean_val_preterminal_output_share_delta": mean_delta("val_preterminal_output_share"),
        "mean_runtime_proxy_seconds_delta": mean_delta("runtime_proxy_seconds"),
    }


def _case_summary(rows: list[dict[str, Any]], knot_rows: list[dict[str, Any]]) -> dict[str, Any]:
    val_case_rows = [row for row in knot_rows if str(row["split"]) == "validation"]
    test_case_rows = [row for row in knot_rows if str(row["split"]) == "test"]
    knot_indices = sorted({int(row["knot_index"]) for row in val_case_rows})
    return {
        "num_runs": len(rows),
        "mean_val_accuracy": _mean([float(row["val_accuracy"]) for row in rows]),
        "std_val_accuracy": _std([float(row["val_accuracy"]) for row in rows]),
        "mean_test_accuracy": _mean([float(row["test_accuracy"]) for row in rows]),
        "std_test_accuracy": _std([float(row["test_accuracy"]) for row in rows]),
        "mean_gate_passing_epoch_count": _mean([float(row["gate_passing_epoch_count"]) for row in rows]),
        "selected_epoch_passes_gate_rate": _mean([float(row["selected_epoch_passes_gate_flag"]) for row in rows]),
        "selector_fallback_used_rate": _mean([float(row["selector_fallback_used_flag"]) for row in rows]),
        "seed_gate_positive_rate": _mean([1.0 if float(row["gate_passing_epoch_count"]) > 0.0 else 0.0 for row in rows]),
        "mean_val_transported_final_energy": _mean([float(row["val_transported_final_energy"]) for row in rows]),
        "mean_val_report_output_mse": _mean([float(row["val_report_output_mse"]) for row in rows]),
        "std_val_report_output_mse": _std([float(row["val_report_output_mse"]) for row in rows]),
        "mean_val_terminal_rowspace_rms": _mean([float(row["val_terminal_rowspace_rms"]) for row in rows]),
        "mean_val_terminal_rowspace_fraction": _mean([float(row["val_terminal_rowspace_fraction"]) for row in rows]),
        "mean_val_preterminal_rowspace_share": _mean([float(row["val_preterminal_rowspace_share"]) for row in rows]),
        "mean_val_terminal_rowspace_share": _mean([float(row["val_terminal_rowspace_share"]) for row in rows]),
        "mean_val_preterminal_output_share": _mean([float(row["val_preterminal_output_share"]) for row in rows]),
        "mean_val_terminal_output_share": _mean([float(row["val_terminal_output_share"]) for row in rows]),
        "mean_test_preterminal_rowspace_share": _mean([float(row["test_preterminal_rowspace_share"]) for row in rows]),
        "mean_test_terminal_rowspace_share": _mean([float(row["test_terminal_rowspace_share"]) for row in rows]),
        "mean_runtime_proxy_seconds": _mean([float(row["runtime_proxy_seconds"]) for row in rows]),
        "mean_validation_rowspace_rms_by_knot": [
            _mean(
                [
                    float(knot_row["hidden_state_rms_gap_rowspace"])
                    for knot_row in val_case_rows
                    if int(knot_row["knot_index"]) == knot_index
                ]
            )
            for knot_index in knot_indices
        ],
        "mean_test_rowspace_rms_by_knot": [
            _mean(
                [
                    float(knot_row["hidden_state_rms_gap_rowspace"])
                    for knot_row in test_case_rows
                    if int(knot_row["knot_index"]) == knot_index
                ]
            )
            for knot_index in knot_indices
        ],
        "rollout_knot_indices": [int(index) for index in knot_indices],
        "rollout_knot_times": [
            _mean([float(knot_row["knot_time"]) for knot_row in val_case_rows if int(knot_row["knot_index"]) == knot_index])
            for knot_index in knot_indices
        ],
    }


def _diagnose_and_recommend(
    config: FMPCTF2LateRolloutDriftControlSuiteConfig,
    by_case: dict[str, dict[str, Any]],
    pairwise_vs_control: dict[str, dict[str, float]],
) -> tuple[str, dict[str, Any], str]:
    control = by_case["adopted_control_terminal_only"]
    non_control_case_names = [name for name in by_case if name != "adopted_control_terminal_only"]
    reduced_drift_cases = [
        name
        for name in non_control_case_names
        if float(pairwise_vs_control[name]["mean_val_terminal_rowspace_rms_delta"]) < 0.0
        and float(pairwise_vs_control[name]["mean_val_preterminal_rowspace_share_delta"]) < 0.0
        and float(pairwise_vs_control[name]["mean_val_preterminal_output_share_delta"]) < 0.0
    ]
    adoption_ready_cases = [
        name
        for name in reduced_drift_cases
        if float(pairwise_vs_control[name]["mean_val_accuracy_delta"]) >= float(config.material_val_gain_threshold)
        and float(pairwise_vs_control[name]["mean_gate_passing_epoch_count_delta"]) >= -float(
            config.max_gate_count_drop_for_adoption
        )
        and float(by_case[name]["selected_epoch_passes_gate_rate"]) >= float(control["selected_epoch_passes_gate_rate"])
        and float(by_case[name]["selector_fallback_used_rate"]) <= float(control["selector_fallback_used_rate"])
    ]
    if adoption_ready_cases:
        recommended_case = max(
            adoption_ready_cases,
            key=lambda name: (
                float(pairwise_vs_control[name]["mean_val_accuracy_delta"]),
                -abs(float(pairwise_vs_control[name]["mean_gate_passing_epoch_count_delta"])),
            ),
        )
        return (
            "late_rollout_drift_is_reduced_by_earlier_same_geometry_control",
            {
                "reduced_drift_cases": reduced_drift_cases,
                "adoption_ready_cases": adoption_ready_cases,
                "recommended_case": recommended_case,
            },
            "run one narrow adoption-level confirmation for the smallest earlier same-geometry late-rollout control that clears the current robustness contract",
        )
    if reduced_drift_cases:
        best_case = max(
            reduced_drift_cases,
            key=lambda name: float(pairwise_vs_control[name]["mean_val_accuracy_delta"]),
        )
        return (
            "mixed_result_but_not_adoption_level",
            {
                "reduced_drift_cases": reduced_drift_cases,
                "best_case": best_case,
                "best_case_pairwise_vs_control": pairwise_vs_control[best_case],
            },
            "run one narrow source-localization diagnostic on the preterminal update formulation itself rather than another cone-family sweep",
        )
    return (
        "late_rollout_drift_is_not_recoverable_by_simple_earlier_same_geometry_control",
        {
            "reduced_drift_cases": [],
            "control_mean_val_terminal_rowspace_rms": float(control["mean_val_terminal_rowspace_rms"]),
        },
        "run one narrow source-localization diagnostic on the preterminal update formulation itself rather than another cone-family sweep",
    )


def run_fmpc_tf2_late_rollout_drift_control_suite(
    config: FMPCTF2LateRolloutDriftControlSuiteConfig,
) -> FMPCTF2LateRolloutDriftControlSuiteRunResult:
    run_dir = _prepare_run_dir(
        _resolve_run_dir(
            config.output_root,
            config.experiment_name,
            config.resolved_run_id(),
            config.output_layout,
        )
    )
    _write_json(run_dir / "config.json", _suite_config_payload(config))

    tf2_root = run_dir / "tf2_runs"
    tf2_root.mkdir(parents=True, exist_ok=True)

    aggregate_rows: list[dict[str, Any]] = []
    knot_rows: list[dict[str, Any]] = []

    for case_spec in config.case_specs():
        for seed in config.seeds:
            tf2_config = build_tf2_corrective_transport_terminal_angleclip_default_config(
                experiment_name="tf2",
                output_root=tf2_root,
                output_layout="run_id_subdir",
                run_id=_run_id(case_spec.case_name, int(seed)),
                run_seed=int(seed),
                data_seed=int(seed),
                model_init_seed=int(seed),
                psi_init_seed=int(seed),
                batch_order_seed=int(seed),
                epochs=int(config.epochs),
                batch_size=int(config.batch_size),
                eval_steps=int(config.eval_steps),
                layer_dims=tuple(config.layer_dims),
                terminal_local_field_intervention_step_offsets=tuple(case_spec.intervention_step_offsets),
            )
            result = run_fmpc_tf2_experiment(tf2_config)
            if result.model is None or result.psi_network is None:
                raise ValueError("Late-rollout drift-control suite requires runtime model and psi network objects.")

            split = load_digits_split(
                split_seed=int(tf2_config.data_seed),
                train_fraction=float(tf2_config.train_fraction),
                val_fraction=float(tf2_config.val_fraction),
                test_fraction=float(tf2_config.test_fraction),
            )

            weight = np.asarray(result.model.layers[-1].weight, dtype=np.float64)
            bias = np.asarray(result.model.layers[-1].bias, dtype=np.float64)
            basis = _rowspace_basis(weight)

            case_row: dict[str, Any] = {
                "case_name": case_spec.case_name,
                "seed": int(seed),
                "run_id": str(result.run_dir.name),
                "run_summary_path": _relative_posix(run_dir, result.run_dir / "summary.json"),
                "intervention_step_offsets": ",".join(str(value) for value in case_spec.intervention_step_offsets),
                "selected_epoch": int(result.summary["best_epoch"]),
                "val_accuracy": float(result.summary["val_accuracy"]),
                "test_accuracy": float(result.summary["test_accuracy"]),
                "gate_passing_epoch_count": int(result.summary["gate_passing_epoch_count"]),
                "selected_epoch_passes_gate": bool(result.summary["selected_epoch_passes_gate"]),
                "selected_epoch_passes_gate_flag": 1.0 if bool(result.summary["selected_epoch_passes_gate"]) else 0.0,
                "selector_fallback_used": bool(result.summary["selector_fallback_used"]),
                "selector_fallback_used_flag": 1.0 if bool(result.summary["selector_fallback_used"]) else 0.0,
                "val_transported_final_energy": float(result.summary["val_transported_final_energy"]),
                "val_report_output_mse": float(result.summary["val_loss"]),
                "runtime_proxy_seconds": _runtime_proxy_seconds(result.summary),
            }

            for split_name, x_split, y_split in (
                ("validation", split.x_val, split.y_val),
                ("test", split.x_test, split.y_test),
            ):
                context = build_tf1_context(result.model, x_split, y_split)
                transported_knots, knot_times = _transport_penultimate_by_knot(
                    result.model,
                    result.psi_network,
                    tf2_config,
                    x_split,
                    y_split,
                )
                slow_pc_knots, slow_pc_step_indices = _slow_pc_penultimate_by_knot(result.model, context, knot_times)
                rowspace_gap_series: list[float] = []
                output_gap_series: list[float] = []

                for knot_index, (knot_time, slow_pc_step_index, transported_features, slow_pc_features) in enumerate(
                    zip(knot_times, slow_pc_step_indices, transported_knots, slow_pc_knots, strict=True)
                ):
                    delta_geometry = _delta_geometry(transported_features, slow_pc_features, y_split, basis)
                    output_gap = _rms_gap(
                        transported_features @ weight.T + bias,
                        slow_pc_features @ weight.T + bias,
                    )
                    rowspace_gap_series.append(float(delta_geometry["delta_h_rms_rowspace"]))
                    output_gap_series.append(float(output_gap))
                    knot_rows.append(
                        {
                            "case_name": case_spec.case_name,
                            "seed": int(seed),
                            "split": split_name,
                            "knot_index": int(knot_index),
                            "knot_time": float(knot_time),
                            "slow_pc_step_index": int(slow_pc_step_index),
                            "hidden_state_rms_gap_total": float(delta_geometry["delta_h_rms_total"]),
                            "hidden_state_rms_gap_rowspace": float(delta_geometry["delta_h_rms_rowspace"]),
                            "hidden_state_rms_gap_orthogonal": float(delta_geometry["delta_h_rms_orthogonal"]),
                            "hidden_state_gap_rowspace_fraction": float(delta_geometry["delta_h_rowspace_fraction"]),
                            "output_state_rms_gap": float(output_gap),
                        }
                    )

                rowspace_contrib = _contribution_from_series(rowspace_gap_series)
                output_contrib = _contribution_from_series(output_gap_series)
                prefix = "val" if split_name == "validation" else "test"
                case_row[f"{prefix}_terminal_rowspace_rms"] = float(rowspace_gap_series[-1])
                case_row[f"{prefix}_terminal_rowspace_fraction"] = float(
                    _delta_geometry(transported_knots[-1], slow_pc_knots[-1], y_split, basis)["delta_h_rowspace_fraction"]
                )
                case_row[f"{prefix}_preterminal_rowspace_share"] = float(rowspace_contrib["preterminal_share"])
                case_row[f"{prefix}_terminal_rowspace_share"] = float(rowspace_contrib["terminal_share"])
                case_row[f"{prefix}_preterminal_output_share"] = float(output_contrib["preterminal_share"])
                case_row[f"{prefix}_terminal_output_share"] = float(output_contrib["terminal_share"])

            aggregate_rows.append(case_row)

    _write_csv(run_dir / "aggregate_runs.csv", aggregate_rows)
    _write_csv(run_dir / "knotwise_drift_metrics.csv", knot_rows)

    by_case: dict[str, dict[str, Any]] = {}
    pairwise_vs_control: dict[str, dict[str, float]] = {}
    control_rows = _rows_for_case(aggregate_rows, "adopted_control_terminal_only")
    for case_spec in config.case_specs():
        case_rows = _rows_for_case(aggregate_rows, case_spec.case_name)
        case_knot_rows = [row for row in knot_rows if str(row["case_name"]) == case_spec.case_name]
        by_case[case_spec.case_name] = _case_summary(case_rows, case_knot_rows)
        if case_spec.case_name != "adopted_control_terminal_only":
            pairwise_vs_control[case_spec.case_name] = _pairwise_delta(case_rows, control_rows)

    diagnosis, diagnosis_evidence, recommended_next_move = _diagnose_and_recommend(
        config,
        by_case,
        pairwise_vs_control,
    )

    summary = {
        "phase": "FMPC Stage 04 Incremental Bridge",
        "stage": "adopted_package_late_rollout_drift_control_diagnostic",
        "num_runs": len(aggregate_rows),
        "by_case": by_case,
        "pairwise_vs_control": pairwise_vs_control,
        "diagnosis": diagnosis,
        "diagnosis_evidence": diagnosis_evidence,
        "recommended_next_narrow_tf2_move": recommended_next_move,
        "aggregate_runs_csv_path": "aggregate_runs.csv",
        "knotwise_drift_metrics_csv_path": "knotwise_drift_metrics.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF2LateRolloutDriftControlSuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config),
        aggregate_rows=aggregate_rows,
        knot_rows=knot_rows,
        summary=summary,
    )
