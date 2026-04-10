from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from .fmpc_tf2_successor_value_confirmation_suite import (
    _CaseSpec,
    _case_recovery_metrics,
    _case_rows,
    _case_summary,
    _failure_row,
    _pairwise_delta,
    _prepare_run_dir,
    _run_one_case_seed,
    _successor_value_audit_table,
    _write_csv,
    _write_json,
)
from .fmpc_tf2_preterminal_source_localization_suite import _resolve_run_dir


_Diagnosis = Literal[
    "local_successor_value_refinement_is_adoption_viable",
    "local_successor_value_refinement_improves_but_not_to_adoption_level",
    "low-live_successor-value_regime_is_locally_saturated",
]


@dataclass(frozen=True)
class FMPCTF2SuccessorValueFollowupSuiteConfig:
    """Run a tiny local follow-up around the best low-live successor-value blend."""

    experiment_name: str = "fmpc_tf2_successor_value_followup_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 60
    batch_size: int = 128
    eval_steps: int = 15
    layer_dims: tuple[int, ...] = (64, 64, 10)
    contract_seed_gate_positive_rate: float = 1.0
    contract_selected_gate_rate: float = 1.0
    contract_max_selector_fallback_used_rate: float = 0.0
    viable_max_val_accuracy_regression_vs_control: float = 0.005
    viable_max_test_accuracy_regression_vs_control: float = 0.005
    material_accuracy_retention_gain_over_anchor_025: float = 0.05
    material_rowspace_retention_gain_over_anchor_025: float = 0.05

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
                description="Reference earlier-control package: same adopted full-vector hard 30 degree cone on the penultimate and terminal micro-steps with the live on-policy successor value.",
                intervention_step_offsets=(-2, -1),
            ),
            _CaseSpec(
                case_name="blended_successor_value_alpha_020",
                description="Penultimate successor value blended 20% live / 80% cached, with all terminal angle-clip semantics unchanged.",
                intervention_step_offsets=(-2, -1),
                successor_value_blend_alpha=0.20,
            ),
            _CaseSpec(
                case_name="blended_successor_value_alpha_025",
                description="Penultimate successor value blended 25% live / 75% cached, kept as the current best narrow reference anchor.",
                intervention_step_offsets=(-2, -1),
                successor_value_blend_alpha=0.25,
            ),
            _CaseSpec(
                case_name="blended_successor_value_alpha_030",
                description="Penultimate successor value blended 30% live / 70% cached, with all terminal angle-clip semantics unchanged.",
                intervention_step_offsets=(-2, -1),
                successor_value_blend_alpha=0.30,
            ),
        )


@dataclass
class FMPCTF2SuccessorValueFollowupSuiteRunResult:
    run_dir: Path
    config: dict[str, Any]
    aggregate_rows: list[dict[str, Any]]
    summary: dict[str, Any]


def _suite_config_payload(config: FMPCTF2SuccessorValueFollowupSuiteConfig) -> dict[str, Any]:
    return {
        "phase": "Phase TF2",
        "stage": "adopted_package_preterminal_successor_value_followup",
        "base_preset_name": "tf2_corrective_transport_terminal_angleclip_default",
        "transport_family_fixed": True,
        "candidate_specs": [
            {
                "case_name": spec.case_name,
                "description": spec.description,
                "intervention_step_offsets": [int(value) for value in spec.intervention_step_offsets],
                "successor_value_blend_alpha": (
                    None if spec.successor_value_blend_alpha is None else float(spec.successor_value_blend_alpha)
                ),
            }
            for spec in config.case_specs()
        ],
        "successor_value_audit_table": _successor_value_audit_table(),
        "seeds": [int(seed) for seed in config.seeds],
        "selector_contract_fixed": True,
        "diagnostic_only": True,
    }


def _control_relative_metrics(control: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    control_accuracy = float(control["mean_val_accuracy"])
    control_gate = float(control["mean_gate_passing_epoch_count"])
    control_rowspace_rms = float(control["mean_val_terminal_rowspace_rms"])
    return {
        "val_accuracy_fraction_of_control": (
            float(candidate["mean_val_accuracy"]) / control_accuracy if abs(control_accuracy) > 1e-12 else 0.0
        ),
        "gate_robustness_fraction_of_control": (
            float(candidate["mean_gate_passing_epoch_count"]) / control_gate if abs(control_gate) > 1e-12 else 0.0
        ),
        "terminal_rowspace_rms_improvement_fraction_vs_control": (
            (control_rowspace_rms - float(candidate["mean_val_terminal_rowspace_rms"])) / control_rowspace_rms
            if abs(control_rowspace_rms) > 1e-12
            else 0.0
        ),
    }


def _gate_contract_intact(
    config: FMPCTF2SuccessorValueFollowupSuiteConfig,
    case_summary: dict[str, Any],
) -> bool:
    return (
        float(case_summary["seed_gate_positive_rate"]) >= float(config.contract_seed_gate_positive_rate)
        and float(case_summary["selected_epoch_passes_gate_rate"]) >= float(config.contract_selected_gate_rate)
        and float(case_summary["selector_fallback_used_rate"])
        <= float(config.contract_max_selector_fallback_used_rate)
    )


def _diagnose_and_recommend(
    config: FMPCTF2SuccessorValueFollowupSuiteConfig,
    by_case: dict[str, dict[str, Any]],
    recovery_vs_failed_anchor: dict[str, dict[str, Any]],
    control_relative_recovery: dict[str, dict[str, Any]],
) -> tuple[_Diagnosis, dict[str, Any], str]:
    control = by_case["adopted_control_terminal_only"]
    failed_anchor = by_case["failed_penultimate_plus_terminal_live"]
    anchor_025 = by_case["blended_successor_value_alpha_025"]
    candidate_names = [
        "blended_successor_value_alpha_020",
        "blended_successor_value_alpha_030",
    ]
    ranked = sorted(
        candidate_names,
        key=lambda name: (
            float(by_case[name]["seed_gate_positive_rate"]),
            float(by_case[name]["selected_epoch_passes_gate_rate"]),
            -float(by_case[name]["selector_fallback_used_rate"]),
            float(recovery_vs_failed_anchor[name]["accuracy_gain_retained_fraction_from_control_to_failed_anchor"]),
            float(
                recovery_vs_failed_anchor[name][
                    "terminal_rowspace_rms_gain_retained_fraction_from_control_to_failed_anchor"
                ]
            ),
            float(by_case[name]["mean_val_accuracy"]),
            -float(by_case[name]["mean_val_terminal_rowspace_rms"]),
        ),
        reverse=True,
    )
    best_name = ranked[0]
    best = by_case[best_name]
    best_recovery = recovery_vs_failed_anchor[best_name]
    anchor_recovery = recovery_vs_failed_anchor["blended_successor_value_alpha_025"]
    gate_contract_intact = _gate_contract_intact(config, best)
    control_competitive = (
        float(best["mean_val_accuracy"]) + float(config.viable_max_val_accuracy_regression_vs_control)
        >= float(control["mean_val_accuracy"])
        and float(best["mean_test_accuracy"]) + float(config.viable_max_test_accuracy_regression_vs_control)
        >= float(control["mean_test_accuracy"])
    )
    materially_better_than_anchor = (
        float(best_recovery["accuracy_gain_retained_fraction_from_control_to_failed_anchor"])
        >= float(anchor_recovery["accuracy_gain_retained_fraction_from_control_to_failed_anchor"])
        + float(config.material_accuracy_retention_gain_over_anchor_025)
        and float(best_recovery["terminal_rowspace_rms_gain_retained_fraction_from_control_to_failed_anchor"])
        >= float(anchor_recovery["terminal_rowspace_rms_gain_retained_fraction_from_control_to_failed_anchor"])
        + float(config.material_rowspace_retention_gain_over_anchor_025)
    )
    evidence = {
        "control_case": control,
        "failed_anchor_case": failed_anchor,
        "anchor_025_case": anchor_025,
        "ranked_candidates": ranked,
        "best_candidate_name": best_name,
        "best_candidate_case": best,
        "best_candidate_recovery_vs_failed_anchor": best_recovery,
        "best_candidate_recovery_vs_control": control_relative_recovery[best_name],
        "anchor_025_recovery_vs_failed_anchor": anchor_recovery,
        "anchor_025_recovery_vs_control": control_relative_recovery["blended_successor_value_alpha_025"],
    }
    if gate_contract_intact and materially_better_than_anchor and control_competitive:
        return (
            "local_successor_value_refinement_is_adoption_viable",
            evidence,
            "run one minimal adoption/defaultization confirmation against the current adopted control before promoting the refined successor-value blend",
        )
    anchor_contract_intact = _gate_contract_intact(config, anchor_025)
    any_positive_local_improvement = any(
        float(recovery_vs_failed_anchor[name]["accuracy_gain_retained_fraction_from_control_to_failed_anchor"])
        > float(anchor_recovery["accuracy_gain_retained_fraction_from_control_to_failed_anchor"])
        or float(
            recovery_vs_failed_anchor[name][
                "terminal_rowspace_rms_gain_retained_fraction_from_control_to_failed_anchor"
            ]
        )
        > float(anchor_recovery["terminal_rowspace_rms_gain_retained_fraction_from_control_to_failed_anchor"])
        for name in candidate_names
    )
    if anchor_contract_intact and any_positive_local_improvement:
        return (
            "local_successor_value_refinement_improves_but_not_to_adoption_level",
            evidence,
            "run one deeper diagnostic on the live successor-value formulation itself rather than another broader low-live blend sweep",
        )
    return (
        "low-live_successor-value_regime_is_locally_saturated",
        evidence,
        "treat the low-live successor-value regime as locally saturated and move to a deeper diagnostic on the live successor-value formulation itself",
    )


def run_fmpc_tf2_successor_value_followup_suite(
    config: FMPCTF2SuccessorValueFollowupSuiteConfig,
) -> FMPCTF2SuccessorValueFollowupSuiteRunResult:
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
            try:
                aggregate_rows.append(_run_one_case_seed(run_dir, config, case_spec, int(seed)))
            except Exception as error:  # pragma: no cover
                aggregate_rows.append(
                    _failure_row(
                        type(
                            "_CompatFailureCaseSpec",
                            (),
                            {
                                "case_name": case_spec.case_name,
                                "intervention_step_offsets": case_spec.intervention_step_offsets,
                                "direction_source_mode": "detached_local_field",
                                "norm_handling_mode": "keep_live_norm",
                                "handoff_mode": (
                                    "live_onpolicy_successor"
                                    if case_spec.successor_value_blend_alpha is None
                                    else f"successor_value_blend_{float(case_spec.successor_value_blend_alpha):.2f}"
                                ),
                            },
                        )(),
                        int(seed),
                        error,
                    )
                )

    _write_csv(run_dir / "aggregate_runs.csv", aggregate_rows)

    by_case: dict[str, dict[str, Any]] = {}
    pairwise_vs_control: dict[str, dict[str, Any]] = {}
    pairwise_vs_failed_anchor: dict[str, dict[str, Any]] = {}
    recovery_fractions_vs_failed_anchor: dict[str, dict[str, Any]] = {}
    control_relative_recovery: dict[str, dict[str, Any]] = {}
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

    control_summary = by_case["adopted_control_terminal_only"]
    failed_anchor_summary = by_case["failed_penultimate_plus_terminal_live"]
    for case_name in (
        "blended_successor_value_alpha_020",
        "blended_successor_value_alpha_025",
        "blended_successor_value_alpha_030",
    ):
        recovery_fractions_vs_failed_anchor[case_name] = _case_recovery_metrics(
            control_summary,
            failed_anchor_summary,
            by_case[case_name],
        )
        control_relative_recovery[case_name] = _control_relative_metrics(
            control_summary,
            by_case[case_name],
        )

    diagnosis, diagnosis_evidence, recommended_next_move = _diagnose_and_recommend(
        config,
        by_case,
        recovery_fractions_vs_failed_anchor,
        control_relative_recovery,
    )
    summary = {
        "phase": "Phase TF2",
        "stage": "adopted_package_preterminal_successor_value_followup",
        "num_runs": len(aggregate_rows),
        "successor_value_audit_table": _successor_value_audit_table(),
        "by_case": by_case,
        "pairwise_vs_control": pairwise_vs_control,
        "pairwise_vs_failed_anchor": pairwise_vs_failed_anchor,
        "recovery_fractions_vs_failed_anchor": recovery_fractions_vs_failed_anchor,
        "control_relative_recovery": control_relative_recovery,
        "diagnosis": diagnosis,
        "diagnosis_evidence": diagnosis_evidence,
        "recommended_next_narrow_tf2_move": recommended_next_move,
        "aggregate_runs_csv_path": "aggregate_runs.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF2SuccessorValueFollowupSuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config),
        aggregate_rows=aggregate_rows,
        summary=summary,
    )
