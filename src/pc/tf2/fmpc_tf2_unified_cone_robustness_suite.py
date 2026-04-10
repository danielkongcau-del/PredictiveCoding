from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .fmpc_tf2_unified_cone_shape_suite import (
    _mean,
    _prepare_run_dir,
    _resolve_run_dir,
    _std,
    _write_csv,
    _write_json,
)


@dataclass(frozen=True)
class _CaseSpec:
    case_name: str
    description: str
    source_root: str | Path
    source_case_name: str


@dataclass
class FMPCTF2UnifiedConeRobustnessSuiteConfig:
    """Run a confirmation-level robustness diagnosis for the unified-cone family."""

    experiment_name: str = "fmpc_tf2_unified_cone_robustness_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    hard_shape_source_root: str | Path = "outputs/tf2/fmpc_tf2_unified_cone_shape_suite"
    smooth_source_root: str | Path = "outputs/tf2/fmpc_tf2_smooth_unified_cone_suite"
    volatility_ratio_threshold: float = 1.25
    severe_epoch_fraction_threshold: float = 0.10
    gate_count_recovery_threshold: float = 2.0

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"

    def case_specs(self) -> tuple[_CaseSpec, ...]:
        return (
            _CaseSpec(
                case_name="adopted_control",
                description="Current hard full-vector 30 degree adopted cone.",
                source_root=self.smooth_source_root,
                source_case_name="adopted_control",
            ),
            _CaseSpec(
                case_name="hard_interior_reference_20",
                description="Current non-adopted hard 20 degree interior-margin reference.",
                source_root=self.hard_shape_source_root,
                source_case_name="unified_cone_interior_margin_20",
            ),
            _CaseSpec(
                case_name="smooth_unified_reference_30",
                description="Current non-adopted smooth unified-cone reference.",
                source_root=self.smooth_source_root,
                source_case_name="smooth_unified_cone_projection_30",
            ),
        )


@dataclass
class FMPCTF2UnifiedConeRobustnessSuiteRunResult:
    run_dir: Path
    config: dict[str, Any]
    aggregate_rows: list[dict[str, Any]]
    epoch_rows: list[dict[str, Any]]
    summary: dict[str, Any]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _suite_config_payload(config: FMPCTF2UnifiedConeRobustnessSuiteConfig) -> dict[str, Any]:
    return {
        "phase": "Phase TF2",
        "stage": "adopted_package_unified_cone_robustness_diagnostic",
        "base_preset_name": "tf2_corrective_transport_terminal_angleclip_default",
        "reuse_existing_outputs": True,
        "source_roots": {
            "hard_shape_source_root": str(config.hard_shape_source_root),
            "smooth_source_root": str(config.smooth_source_root),
        },
        "candidate_specs": [
            {
                "case_name": spec.case_name,
                "description": spec.description,
                "source_root": str(spec.source_root),
                "source_case_name": spec.source_case_name,
            }
            for spec in config.case_specs()
        ],
        "thresholds": {
            "volatility_ratio_threshold": float(config.volatility_ratio_threshold),
            "severe_epoch_fraction_threshold": float(config.severe_epoch_fraction_threshold),
            "gate_count_recovery_threshold": float(config.gate_count_recovery_threshold),
        },
    }


def _to_float(value: Any) -> float:
    return float(value)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "1.0"}:
        return True
    if text in {"false", "0", "0.0"}:
        return False
    raise ValueError(f"Cannot parse boolean value from {value!r}.")


def _load_case_source_rows(case: _CaseSpec) -> list[dict[str, str]]:
    source_root = Path(case.source_root)
    aggregate_path = source_root / "aggregate_runs.csv"
    if not aggregate_path.exists():
        raise FileNotFoundError(f"Missing aggregate_runs.csv for {case.case_name}: {aggregate_path}")
    rows = [row for row in _read_csv(aggregate_path) if str(row.get("case_name", "")) == case.source_case_name]
    if not rows:
        raise ValueError(f"No rows found for source case '{case.source_case_name}' in {aggregate_path}.")
    return rows


def _gate_failure_type(*, accuracy_pass: bool, energy_pass: bool) -> str:
    if accuracy_pass and energy_pass:
        return "pass"
    if (not accuracy_pass) and energy_pass:
        return "accuracy_only"
    if accuracy_pass and (not energy_pass):
        return "energy_only"
    return "both"


def _selector_score(val_accuracy: float, gate_pass: bool) -> float:
    return float(val_accuracy) if gate_pass else float("-inf")


def _mean_abs_diff(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(sum(abs(float(values[index + 1]) - float(values[index])) for index in range(len(values) - 1)) / (len(values) - 1))


def _case_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Case summary requires at least one row.")

    def mean_metric(metric_name: str) -> float:
        return _mean([float(row[metric_name]) for row in rows])

    def std_metric(metric_name: str) -> float:
        return _std([float(row[metric_name]) for row in rows])

    return {
        "num_runs": len(rows),
        "mean_val_accuracy": mean_metric("val_accuracy"),
        "std_val_accuracy": std_metric("val_accuracy"),
        "mean_test_accuracy": mean_metric("test_accuracy"),
        "std_test_accuracy": std_metric("test_accuracy"),
        "mean_gate_passing_epoch_count": mean_metric("gate_passing_epoch_count"),
        "selected_epoch_passes_gate_rate": mean_metric("selected_epoch_passes_gate"),
        "selector_fallback_used_rate": mean_metric("selector_fallback_used"),
        "seed_gate_positive_rate": mean_metric("seed_gate_positive"),
        "mean_val_report_output_mse": mean_metric("val_report_output_mse"),
        "std_val_report_output_mse": std_metric("val_report_output_mse"),
        "mean_val_supervised_transport_output_mse": mean_metric("val_supervised_transport_output_mse"),
        "std_val_supervised_transport_output_mse": std_metric("val_supervised_transport_output_mse"),
        "mean_val_delta_h_rms_rowspace": mean_metric("val_delta_h_rms_rowspace"),
        "mean_val_delta_h_rowspace_fraction": mean_metric("val_delta_h_rowspace_fraction"),
        "mean_runtime_proxy_seconds": mean_metric("runtime_proxy_seconds"),
        "mean_gate_epoch_rate": mean_metric("gate_epoch_rate"),
        "mean_val_accuracy_margin_mean": mean_metric("val_accuracy_margin_mean"),
        "mean_val_accuracy_margin_min": mean_metric("val_accuracy_margin_min"),
        "mean_gate_energy_margin_mean": mean_metric("gate_energy_margin_mean"),
        "mean_gate_energy_margin_min": mean_metric("gate_energy_margin_min"),
        "mean_val_accuracy_volatility": mean_metric("val_accuracy_volatility"),
        "mean_val_transported_final_energy_volatility": mean_metric("val_transported_final_energy_volatility"),
        "mean_best_vs_selected_val_accuracy_gap": mean_metric("best_vs_selected_val_accuracy_gap"),
        "mean_best_vs_selected_energy_gap": mean_metric("best_vs_selected_energy_gap"),
        "mean_gate_failure_accuracy_only_fraction": mean_metric("gate_failure_accuracy_only_fraction"),
        "mean_gate_failure_energy_only_fraction": mean_metric("gate_failure_energy_only_fraction"),
        "mean_gate_failure_both_fraction": mean_metric("gate_failure_both_fraction"),
        "mean_negative_accuracy_margin_epoch_fraction": mean_metric("negative_accuracy_margin_epoch_fraction"),
        "mean_negative_gate_energy_margin_epoch_fraction": mean_metric("negative_gate_energy_margin_epoch_fraction"),
        "mean_severe_bad_epoch_fraction": mean_metric("severe_bad_epoch_fraction"),
    }


def _pairwise_delta(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]) -> dict[str, Any]:
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
        "shared_seeds": [int(seed) for seed in shared_seeds],
        "mean_val_accuracy_delta": mean_delta("val_accuracy"),
        "mean_test_accuracy_delta": mean_delta("test_accuracy"),
        "mean_gate_passing_epoch_count_delta": mean_delta("gate_passing_epoch_count"),
        "mean_selected_epoch_passes_gate_rate_delta": mean_delta("selected_epoch_passes_gate"),
        "mean_selector_fallback_used_rate_delta": mean_delta("selector_fallback_used"),
        "mean_seed_gate_positive_rate_delta": mean_delta("seed_gate_positive"),
        "mean_val_report_output_mse_delta": mean_delta("val_report_output_mse"),
        "mean_val_supervised_transport_output_mse_delta": mean_delta("val_supervised_transport_output_mse"),
        "mean_val_delta_h_rms_rowspace_delta": mean_delta("val_delta_h_rms_rowspace"),
        "mean_val_delta_h_rowspace_fraction_delta": mean_delta("val_delta_h_rowspace_fraction"),
        "mean_val_accuracy_volatility_delta": mean_delta("val_accuracy_volatility"),
        "mean_val_transported_final_energy_volatility_delta": mean_delta("val_transported_final_energy_volatility"),
        "mean_negative_accuracy_margin_epoch_fraction_delta": mean_delta("negative_accuracy_margin_epoch_fraction"),
        "mean_negative_gate_energy_margin_epoch_fraction_delta": mean_delta("negative_gate_energy_margin_epoch_fraction"),
        "mean_gate_failure_accuracy_only_fraction_delta": mean_delta("gate_failure_accuracy_only_fraction"),
        "mean_gate_failure_energy_only_fraction_delta": mean_delta("gate_failure_energy_only_fraction"),
        "mean_gate_failure_both_fraction_delta": mean_delta("gate_failure_both_fraction"),
        "mean_severe_bad_epoch_fraction_delta": mean_delta("severe_bad_epoch_fraction"),
        "mean_runtime_proxy_seconds_delta": mean_delta("runtime_proxy_seconds"),
    }


def _per_run_robustness(case_name: str, row: dict[str, str], epoch_rows: list[dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected_epoch = int(row["selected_epoch"])
    parsed_epochs: list[dict[str, Any]] = []
    for epoch_row in epoch_rows:
        epoch = int(epoch_row["epoch"])
        val_accuracy = _to_float(epoch_row["val_accuracy"])
        val_baseline_accuracy = _to_float(epoch_row["val_baseline_accuracy"])
        val_transported_final_energy = _to_float(epoch_row["val_transported_final_energy"])
        val_identity_final_energy = _to_float(epoch_row["val_identity_final_energy"])
        val_local_field_only_final_energy = _to_float(epoch_row["val_local_field_only_final_energy"])
        accuracy_margin = float(val_accuracy - val_baseline_accuracy)
        energy_margin_identity = float(val_identity_final_energy - val_transported_final_energy)
        energy_margin_local = float(val_local_field_only_final_energy - val_transported_final_energy)
        gate_energy_margin = float(min(energy_margin_identity, energy_margin_local))
        accuracy_pass = bool(accuracy_margin > 0.0)
        energy_pass = bool((energy_margin_identity > 0.0) and (energy_margin_local >= 0.0))
        gate_pass = bool(accuracy_pass and energy_pass)
        failure_type = _gate_failure_type(accuracy_pass=accuracy_pass, energy_pass=energy_pass)
        parsed_epochs.append(
            {
                "case_name": case_name,
                "seed": int(row["seed"]),
                "epoch": epoch,
                "stage": str(epoch_row["stage"]),
                "val_accuracy": val_accuracy,
                "val_baseline_accuracy": val_baseline_accuracy,
                "val_accuracy_margin": accuracy_margin,
                "val_transported_final_energy": val_transported_final_energy,
                "val_identity_final_energy": val_identity_final_energy,
                "val_local_field_only_final_energy": val_local_field_only_final_energy,
                "val_energy_margin_vs_identity": energy_margin_identity,
                "val_energy_margin_vs_local_field_only": energy_margin_local,
                "gate_energy_margin": gate_energy_margin,
                "gate_pass": 1.0 if gate_pass else 0.0,
                "gate_failure_type": failure_type,
                "selector_score": float(val_accuracy),
                "gate_constrained_selector_score": _selector_score(val_accuracy, gate_pass),
                "selected_epoch_flag": 1.0 if epoch == selected_epoch else 0.0,
            }
        )

    if not parsed_epochs:
        raise ValueError(f"No epoch rows parsed for case '{case_name}' seed {row['seed']}.")

    best_accuracy_epoch = max(parsed_epochs, key=lambda item: float(item["val_accuracy"]))
    best_energy_epoch = min(parsed_epochs, key=lambda item: float(item["val_transported_final_energy"]))
    for epoch_entry in parsed_epochs:
        epoch_entry["best_accuracy_epoch_flag"] = 1.0 if int(epoch_entry["epoch"]) == int(best_accuracy_epoch["epoch"]) else 0.0
        epoch_entry["best_energy_epoch_flag"] = 1.0 if int(epoch_entry["epoch"]) == int(best_energy_epoch["epoch"]) else 0.0

    selected_entry = next(item for item in parsed_epochs if int(item["epoch"]) == selected_epoch)
    fail_epochs = [item for item in parsed_epochs if float(item["gate_pass"]) < 0.5]
    fail_count = max(len(fail_epochs), 1)
    total_epoch_count = len(parsed_epochs)
    gate_pass_count = sum(1 for item in parsed_epochs if float(item["gate_pass"]) > 0.5)
    energy_margin_values = [float(item["gate_energy_margin"]) for item in parsed_epochs]
    accuracy_values = [float(item["val_accuracy"]) for item in parsed_epochs]
    transported_energy_values = [float(item["val_transported_final_energy"]) for item in parsed_epochs]
    severe_bad_epoch_cutoff = float(min(energy_margin_values) * 0.5) if min(energy_margin_values) < 0.0 else -1e-12
    severe_bad_epoch_fraction = float(
        sum(1 for value in energy_margin_values if float(value) <= severe_bad_epoch_cutoff) / float(total_epoch_count)
    )

    aggregate_row = {
        "case_name": case_name,
        "seed": int(row["seed"]),
        "run_id": str(row["run_id"]),
        "run_summary_path": str(row["run_summary_path"]),
        "selected_epoch": selected_epoch,
        "val_accuracy": _to_float(row["val_accuracy"]),
        "test_accuracy": _to_float(row["test_accuracy"]),
        "gate_passing_epoch_count": int(float(row["gate_passing_epoch_count"])),
        "selected_epoch_passes_gate": 1.0 if _to_bool(row["selected_epoch_passes_gate"]) else 0.0,
        "selector_fallback_used": 1.0 if _to_bool(row["selector_fallback_used"]) else 0.0,
        "seed_gate_positive": 1.0 if int(float(row["gate_passing_epoch_count"])) > 0 else 0.0,
        "val_report_output_mse": _to_float(row["val_report_output_mse"]),
        "val_supervised_transport_output_mse": _to_float(row["val_supervised_transport_output_mse"]),
        "val_delta_h_rms_rowspace": _to_float(row["val_delta_h_rms_rowspace"]),
        "val_delta_h_rowspace_fraction": _to_float(row["val_delta_h_rowspace_fraction"]),
        "runtime_proxy_seconds": _to_float(row["runtime_proxy_seconds"]),
        "gate_epoch_rate": float(gate_pass_count / float(total_epoch_count)),
        "val_accuracy_margin_mean": _mean([float(item["val_accuracy_margin"]) for item in parsed_epochs]),
        "val_accuracy_margin_min": min(float(item["val_accuracy_margin"]) for item in parsed_epochs),
        "gate_energy_margin_mean": _mean(energy_margin_values),
        "gate_energy_margin_min": min(energy_margin_values),
        "val_accuracy_volatility": _mean_abs_diff(accuracy_values),
        "val_transported_final_energy_volatility": _mean_abs_diff(transported_energy_values),
        "best_vs_selected_val_accuracy_gap": float(float(best_accuracy_epoch["val_accuracy"]) - float(selected_entry["val_accuracy"])),
        "best_vs_selected_energy_gap": float(
            float(selected_entry["val_transported_final_energy"]) - float(best_energy_epoch["val_transported_final_energy"])
        ),
        "gate_failure_accuracy_only_fraction": float(
            sum(1 for item in fail_epochs if str(item["gate_failure_type"]) == "accuracy_only") / float(fail_count)
        ),
        "gate_failure_energy_only_fraction": float(
            sum(1 for item in fail_epochs if str(item["gate_failure_type"]) == "energy_only") / float(fail_count)
        ),
        "gate_failure_both_fraction": float(
            sum(1 for item in fail_epochs if str(item["gate_failure_type"]) == "both") / float(fail_count)
        ),
        "negative_accuracy_margin_epoch_fraction": float(
            sum(1 for item in parsed_epochs if float(item["val_accuracy_margin"]) <= 0.0) / float(total_epoch_count)
        ),
        "negative_gate_energy_margin_epoch_fraction": float(
            sum(1 for item in parsed_epochs if float(item["gate_energy_margin"]) <= 0.0) / float(total_epoch_count)
        ),
        "severe_bad_epoch_fraction": severe_bad_epoch_fraction,
        "per_epoch_gate_pass_sequence": "".join("1" if float(item["gate_pass"]) > 0.5 else "0" for item in parsed_epochs),
    }
    return aggregate_row, parsed_epochs


def _load_case(case: _CaseSpec) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_root = Path(case.source_root)
    source_rows = _load_case_source_rows(case)
    aggregate_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    for row in source_rows:
        run_summary_path = source_root / str(row["run_summary_path"])
        run_dir = run_summary_path.parent
        epoch_metrics_path = run_dir / "epoch_metrics.csv"
        if not epoch_metrics_path.exists():
            raise FileNotFoundError(f"Missing epoch_metrics.csv: {epoch_metrics_path}")
        aggregate_row, parsed_epochs = _per_run_robustness(case.case_name, row, _read_csv(epoch_metrics_path))
        aggregate_rows.append(aggregate_row)
        epoch_rows.extend(parsed_epochs)
    return aggregate_rows, epoch_rows


def _variant_diagnosis(
    control_summary: dict[str, Any],
    variant_summary: dict[str, Any],
    variant_vs_control: dict[str, Any],
    config: FMPCTF2UnifiedConeRobustnessSuiteConfig,
) -> str:
    energy_side_fraction = float(variant_summary["mean_gate_failure_energy_only_fraction"]) + float(
        variant_summary["mean_gate_failure_both_fraction"]
    )
    accuracy_side_fraction = float(variant_summary["mean_gate_failure_accuracy_only_fraction"]) + float(
        variant_summary["mean_gate_failure_both_fraction"]
    )
    accuracy_volatility_ratio = float(variant_summary["mean_val_accuracy_volatility"]) / max(
        float(control_summary["mean_val_accuracy_volatility"]), 1e-12
    )
    energy_volatility_ratio = float(variant_summary["mean_val_transported_final_energy_volatility"]) / max(
        float(control_summary["mean_val_transported_final_energy_volatility"]), 1e-12
    )
    if (
        float(variant_vs_control["mean_negative_gate_energy_margin_epoch_fraction_delta"]) > 0.0
        and energy_side_fraction >= accuracy_side_fraction
        and accuracy_volatility_ratio < float(config.volatility_ratio_threshold)
        and energy_volatility_ratio < float(config.volatility_ratio_threshold)
    ):
        return "systematic_threshold_margin_collapse_mainly_on_energy_side"
    if (
        accuracy_volatility_ratio >= float(config.volatility_ratio_threshold)
        or energy_volatility_ratio >= float(config.volatility_ratio_threshold)
    ):
        return "higher_temporal_volatility_or_instability"
    if float(variant_summary["mean_severe_bad_epoch_fraction"]) >= float(config.severe_epoch_fraction_threshold):
        return "rare_but_severe_bad_epochs"
    return "mixed_picture"


def _diagnose(
    summary_by_candidate: dict[str, dict[str, Any]],
    pairwise_vs_control: dict[str, dict[str, Any]],
    config: FMPCTF2UnifiedConeRobustnessSuiteConfig,
) -> tuple[str, dict[str, Any], str]:
    control = summary_by_candidate["adopted_control"]
    hard20 = summary_by_candidate["hard_interior_reference_20"]
    smooth = summary_by_candidate["smooth_unified_reference_30"]
    hard20_diag = _variant_diagnosis(control, hard20, pairwise_vs_control["hard_interior_reference_20"], config)
    smooth_diag = _variant_diagnosis(control, smooth, pairwise_vs_control["smooth_unified_reference_30"], config)
    evidence = {
        "control_mean_gate_passing_epoch_count": float(control["mean_gate_passing_epoch_count"]),
        "hard20_mean_gate_passing_epoch_count": float(hard20["mean_gate_passing_epoch_count"]),
        "smooth_mean_gate_passing_epoch_count": float(smooth["mean_gate_passing_epoch_count"]),
        "control_seed_gate_positive_rate": float(control["seed_gate_positive_rate"]),
        "hard20_seed_gate_positive_rate": float(hard20["seed_gate_positive_rate"]),
        "smooth_seed_gate_positive_rate": float(smooth["seed_gate_positive_rate"]),
        "control_selected_epoch_passes_gate_rate": float(control["selected_epoch_passes_gate_rate"]),
        "hard20_selected_epoch_passes_gate_rate": float(hard20["selected_epoch_passes_gate_rate"]),
        "smooth_selected_epoch_passes_gate_rate": float(smooth["selected_epoch_passes_gate_rate"]),
        "control_selector_fallback_used_rate": float(control["selector_fallback_used_rate"]),
        "hard20_selector_fallback_used_rate": float(hard20["selector_fallback_used_rate"]),
        "smooth_selector_fallback_used_rate": float(smooth["selector_fallback_used_rate"]),
        "hard20_variant_diagnosis": hard20_diag,
        "smooth_variant_diagnosis": smooth_diag,
    }
    diagnosis = "unified_cone_family_is_locally_saturated_under_the_fixed_gate_and_selector_contract"
    recommended_next_move = (
        "treat the unified-cone family as locally saturated and move future TF2 work to a different remaining "
        "package-internal issue rather than another confirmation inside this family"
    )
    if (
        hard20_diag == "rare_but_severe_bad_epochs"
        or smooth_diag == "rare_but_severe_bad_epochs"
        or (
            float(smooth["mean_gate_passing_epoch_count"]) >= float(hard20["mean_gate_passing_epoch_count"])
            + float(config.gate_count_recovery_threshold)
            and float(smooth["mean_selected_epoch_passes_gate_rate"]) >= float(control["selected_epoch_passes_gate_rate"])
        )
    ):
        diagnosis = "there_is_still_one_confirmation_level_reason_to_probe_robustness_inside_the_unified_cone_family"
        recommended_next_move = (
            "if TF2 work continues inside this family, allow at most one final confirmation-level robustness follow-up "
            "focused on the diagnosed failure structure rather than another new cone variant"
        )
    return diagnosis, evidence, recommended_next_move


def run_fmpc_tf2_unified_cone_robustness_suite(
    config: FMPCTF2UnifiedConeRobustnessSuiteConfig,
) -> FMPCTF2UnifiedConeRobustnessSuiteRunResult:
    run_dir = _prepare_run_dir(
        _resolve_run_dir(
            config.output_root,
            config.experiment_name,
            config.resolved_run_id(),
            config.output_layout,
        )
    )
    _write_json(run_dir / "config.json", _suite_config_payload(config))

    aggregate_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    for case in config.case_specs():
        case_aggregate_rows, case_epoch_rows = _load_case(case)
        aggregate_rows.extend(case_aggregate_rows)
        epoch_rows.extend(case_epoch_rows)

    _write_csv(run_dir / "aggregate_runs.csv", aggregate_rows)
    _write_csv(run_dir / "epoch_robustness_diagnostics.csv", epoch_rows)

    summary_by_candidate = {
        case.case_name: _case_summary([row for row in aggregate_rows if str(row["case_name"]) == case.case_name])
        for case in config.case_specs()
    }
    control_rows = [row for row in aggregate_rows if str(row["case_name"]) == "adopted_control"]
    pairwise_vs_control = {
        case.case_name: _pairwise_delta([row for row in aggregate_rows if str(row["case_name"]) == case.case_name], control_rows)
        for case in config.case_specs()
        if case.case_name != "adopted_control"
    }
    diagnosis, evidence, recommended_next_move = _diagnose(summary_by_candidate, pairwise_vs_control, config)

    summary = {
        "phase": "Phase TF2",
        "stage": "adopted_package_unified_cone_robustness_diagnostic",
        "num_runs": len(aggregate_rows),
        "by_candidate": summary_by_candidate,
        "pairwise_vs_control": pairwise_vs_control,
        "diagnosis": diagnosis,
        "diagnosis_evidence": evidence,
        "recommended_next_move": recommended_next_move,
        "should_continue_inside_unified_cone_family": bool(
            diagnosis == "there_is_still_one_confirmation_level_reason_to_probe_robustness_inside_the_unified_cone_family"
        ),
        "decision": (
            "one_last_confirmation_level_followup_still_justified"
            if diagnosis == "there_is_still_one_confirmation_level_reason_to_probe_robustness_inside_the_unified_cone_family"
            else "treat_unified_cone_family_as_locally_saturated"
        ),
        "aggregate_csv_path": "aggregate_runs.csv",
        "epoch_robustness_csv_path": "epoch_robustness_diagnostics.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF2UnifiedConeRobustnessSuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config),
        aggregate_rows=aggregate_rows,
        epoch_rows=epoch_rows,
        summary=summary,
    )
