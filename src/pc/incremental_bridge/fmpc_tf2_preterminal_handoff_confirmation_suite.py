from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from .fmpc_tf2_preterminal_source_localization_suite import (
    _CaseSpec,
    _case_rows,
    _case_summary,
    _pairwise_delta,
    _prepare_run_dir,
    _resolve_run_dir,
    _run_one_case_seed,
    _write_csv,
    _write_json,
)

_Diagnosis = Literal[
    "minimal_cached_handoff_reformulation_is_adoption_viable",
    "handoff_reformulation_recovers_partially_but_not_adoption_level",
    "preterminal_handoff_blocker_persists",
]


@dataclass
class FMPCTF2PreterminalHandoffConfirmationSuiteConfig:
    """Run a very narrow confirmation on the smallest preterminal handoff reformulation."""

    experiment_name: str = "fmpc_tf2_preterminal_handoff_confirmation_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 60
    batch_size: int = 128
    eval_steps: int = 15
    layer_dims: tuple[int, ...] = (64, 64, 10)
    viable_seed_gate_positive_rate_threshold: float = 0.8
    viable_selected_gate_rate_threshold: float = 0.8
    viable_max_selector_fallback_used_rate: float = 0.2
    viable_min_accuracy_recovery_fraction: float = 0.5
    viable_min_rowspace_recovery_fraction: float = 0.5
    viable_max_val_accuracy_regression_vs_control: float = 0.005
    viable_max_test_accuracy_regression_vs_control: float = 0.005

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"

    def case_specs(self) -> tuple[_CaseSpec, ...]:
        return (
            _CaseSpec(
                case_name="adopted_control_terminal_only",
                description="Current adopted package: full-vector hard 30 degree cone on the terminal micro-step only.",
                intervention_step_offsets=(-1,),
            ),
            _CaseSpec(
                case_name="failed_penultimate_plus_terminal_live",
                description="Earlier-control failure anchor: same adopted full-vector hard 30 degree cone on the penultimate and terminal micro-steps.",
                intervention_step_offsets=(-2, -1),
            ),
            _CaseSpec(
                case_name="minimal_cached_handoff_reformulation",
                description="Earlier-control package with only the preterminal on-policy handoff state swapped back to the cached batch-start successor handoff.",
                intervention_step_offsets=(-2, -1),
                handoff_mode="cached_onpolicy_handoff",
            ),
        )


@dataclass
class FMPCTF2PreterminalHandoffConfirmationSuiteRunResult:
    run_dir: Path
    config: dict[str, Any]
    aggregate_rows: list[dict[str, Any]]
    summary: dict[str, Any]


def _suite_config_payload(config: FMPCTF2PreterminalHandoffConfirmationSuiteConfig) -> dict[str, Any]:
    return {
        "phase": "Phase Incremental Bridge",
        "stage": "adopted_package_preterminal_handoff_confirmation",
        "base_preset_name": "tf2_corrective_transport_terminal_angleclip_default",
        "transport_family_fixed": True,
        "candidate_specs": [
            {
                "case_name": spec.case_name,
                "description": spec.description,
                "intervention_step_offsets": [int(value) for value in spec.intervention_step_offsets],
                "direction_source_mode": spec.direction_source_mode,
                "norm_handling_mode": spec.norm_handling_mode,
                "handoff_mode": spec.handoff_mode,
            }
            for spec in config.case_specs()
        ],
        "seeds": [int(seed) for seed in config.seeds],
        "selector_contract_fixed": True,
        "diagnostic_only": True,
        "viability_thresholds": {
            "seed_gate_positive_rate_threshold": float(config.viable_seed_gate_positive_rate_threshold),
            "selected_gate_rate_threshold": float(config.viable_selected_gate_rate_threshold),
            "max_selector_fallback_used_rate": float(config.viable_max_selector_fallback_used_rate),
            "min_accuracy_recovery_fraction": float(config.viable_min_accuracy_recovery_fraction),
            "min_rowspace_recovery_fraction": float(config.viable_min_rowspace_recovery_fraction),
            "max_val_accuracy_regression_vs_control": float(config.viable_max_val_accuracy_regression_vs_control),
            "max_test_accuracy_regression_vs_control": float(config.viable_max_test_accuracy_regression_vs_control),
        },
    }


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _recovery_fraction_higher_is_better(control: float, anchor: float, candidate: float) -> float:
    anchor_gain = float(anchor) - float(control)
    if anchor_gain <= 0.0:
        return 0.0
    return (float(candidate) - float(control)) / anchor_gain


def _recovery_fraction_lower_is_better(control: float, anchor: float, candidate: float) -> float:
    anchor_gain = float(control) - float(anchor)
    if anchor_gain <= 0.0:
        return 0.0
    return (float(control) - float(candidate)) / anchor_gain


def _diagnose_and_recommend(
    config: FMPCTF2PreterminalHandoffConfirmationSuiteConfig,
    by_case: dict[str, dict[str, Any]],
) -> tuple[_Diagnosis, dict[str, Any], str]:
    control = by_case["adopted_control_terminal_only"]
    failed_anchor = by_case["failed_penultimate_plus_terminal_live"]
    candidate = by_case["minimal_cached_handoff_reformulation"]

    gate_recovered = (
        _safe_float(candidate["seed_gate_positive_rate"]) >= float(config.viable_seed_gate_positive_rate_threshold)
        and _safe_float(candidate["selected_epoch_passes_gate_rate"])
        >= float(config.viable_selected_gate_rate_threshold)
        and _safe_float(candidate["selector_fallback_used_rate"])
        <= float(config.viable_max_selector_fallback_used_rate)
    )
    control_competitive = (
        _safe_float(candidate["mean_val_accuracy"]) + float(config.viable_max_val_accuracy_regression_vs_control)
        >= _safe_float(control["mean_val_accuracy"])
        and _safe_float(candidate["mean_test_accuracy"]) + float(config.viable_max_test_accuracy_regression_vs_control)
        >= _safe_float(control["mean_test_accuracy"])
    )
    accuracy_recovery_fraction = _recovery_fraction_higher_is_better(
        _safe_float(control["mean_val_accuracy"]),
        _safe_float(failed_anchor["mean_val_accuracy"]),
        _safe_float(candidate["mean_val_accuracy"]),
    )
    rowspace_recovery_fraction = _recovery_fraction_lower_is_better(
        _safe_float(control["mean_val_terminal_rowspace_rms"]),
        _safe_float(failed_anchor["mean_val_terminal_rowspace_rms"]),
        _safe_float(candidate["mean_val_terminal_rowspace_rms"]),
    )
    output_mse_recovery_fraction = _recovery_fraction_lower_is_better(
        _safe_float(control["mean_val_report_output_mse"]),
        _safe_float(failed_anchor["mean_val_report_output_mse"]),
        _safe_float(candidate["mean_val_report_output_mse"]),
    )

    evidence = {
        "gate_recovered": bool(gate_recovered),
        "control_competitive": bool(control_competitive),
        "accuracy_recovery_fraction_vs_failed_anchor": float(accuracy_recovery_fraction),
        "rowspace_recovery_fraction_vs_failed_anchor": float(rowspace_recovery_fraction),
        "output_mse_recovery_fraction_vs_failed_anchor": float(output_mse_recovery_fraction),
        "control_case": control,
        "failed_anchor_case": failed_anchor,
        "candidate_case": candidate,
    }

    if (
        gate_recovered
        and control_competitive
        and accuracy_recovery_fraction >= float(config.viable_min_accuracy_recovery_fraction)
        and rowspace_recovery_fraction >= float(config.viable_min_rowspace_recovery_fraction)
    ):
        return (
            "minimal_cached_handoff_reformulation_is_adoption_viable",
            evidence,
            "run one narrow adoption-level confirmation against the current adopted control before promoting the minimal cached-handoff reformulation inside the adopted TF2 package",
        )
    if gate_recovered:
        return (
            "handoff_reformulation_recovers_partially_but_not_adoption_level",
            evidence,
            "run one narrower handoff-state source-localization step on the preterminal successor handoff itself rather than another cone-family sweep",
        )
    return (
        "preterminal_handoff_blocker_persists",
        evidence,
        "run one narrower handoff-state source-localization step on the preterminal successor handoff itself rather than another cone-family sweep",
    )


def run_fmpc_tf2_preterminal_handoff_confirmation_suite(
    config: FMPCTF2PreterminalHandoffConfirmationSuiteConfig,
) -> FMPCTF2PreterminalHandoffConfirmationSuiteRunResult:
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
    for case_spec in config.case_specs():
        for seed in config.seeds:
            aggregate_rows.append(_run_one_case_seed(run_dir, config, case_spec, int(seed)))
    _write_csv(run_dir / "aggregate_runs.csv", aggregate_rows)

    by_case: dict[str, dict[str, Any]] = {}
    pairwise_vs_control: dict[str, dict[str, Any]] = {}
    pairwise_vs_failed_anchor: dict[str, dict[str, Any]] = {}
    control_rows = _case_rows(aggregate_rows, "adopted_control_terminal_only")
    failed_anchor_rows = _case_rows(aggregate_rows, "failed_penultimate_plus_terminal_live")
    for case_spec in config.case_specs():
        case_rows_all = [row for row in aggregate_rows if str(row["case_name"]) == case_spec.case_name]
        case_rows = _case_rows(aggregate_rows, case_spec.case_name)
        by_case[case_spec.case_name] = _case_summary(case_rows, case_rows_all)
        if case_spec.case_name != "adopted_control_terminal_only":
            pairwise_vs_control[case_spec.case_name] = _pairwise_delta(case_rows, control_rows)
        if case_spec.case_name != "failed_penultimate_plus_terminal_live":
            pairwise_vs_failed_anchor[case_spec.case_name] = _pairwise_delta(case_rows, failed_anchor_rows)

    diagnosis, diagnosis_evidence, recommended_next_move = _diagnose_and_recommend(config, by_case)
    summary = {
        "phase": "Phase Incremental Bridge",
        "stage": "adopted_package_preterminal_handoff_confirmation",
        "num_runs": len(aggregate_rows),
        "by_case": by_case,
        "pairwise_vs_control": pairwise_vs_control,
        "pairwise_vs_failed_anchor": pairwise_vs_failed_anchor,
        "diagnosis": diagnosis,
        "diagnosis_evidence": diagnosis_evidence,
        "recommended_next_narrow_tf2_move": recommended_next_move,
        "aggregate_runs_csv_path": "aggregate_runs.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF2PreterminalHandoffConfirmationSuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config),
        aggregate_rows=aggregate_rows,
        summary=summary,
    )
