from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from .fmpc_tf2_preterminal_source_localization_suite import (
    _case_rows,
    _case_summary,
    _failure_row,
    _pairwise_delta,
    _prepare_run_dir,
    _resolve_run_dir,
    _write_csv,
    _write_json,
)
from .fmpc_tf2_successor_increment_confirmation_suite import (
    _case_recovery_metrics,
    _control_relative_recovery,
)
from .fmpc_tf2_successor_increment_source_suite import (
    FMPCTF2SuccessorIncrementSourceSuiteConfig as _BaseConfig,
    _run_one_case_seed,
)


_Diagnosis = Literal[
    "minimal_direction_magnitude_interaction_is_adoption_viable",
    "interaction_recovers_partially_but_not_adoption_level",
    "live_successor_increment_interaction_blocker_persists",
]


@dataclass(frozen=True)
class _CaseSpec:
    case_name: str
    description: str
    intervention_step_offsets: tuple[int, ...]
    increment_mode: str = "live_successor_increment"
    direction_angle_degrees: float | None = None


@dataclass
class FMPCTF2SuccessorIncrementInteractionSuiteConfig:
    """Run a very narrow direction-magnitude interaction diagnostic."""

    experiment_name: str = "fmpc_tf2_successor_increment_interaction_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 60
    batch_size: int = 128
    eval_steps: int = 15
    layer_dims: tuple[int, ...] = (64, 64, 10)
    current_partial_signal_angle_degrees: float = 30.0
    stronger_partial_signal_angle_degrees: float = 20.0
    full_gate_seed_gate_positive_rate: float = 1.0
    full_gate_selected_epoch_passes_gate_rate: float = 1.0
    full_gate_selector_fallback_used_rate: float = 0.0
    viable_min_accuracy_recovery_fraction: float = 0.40
    viable_min_rowspace_recovery_fraction: float = 0.35
    viable_max_val_accuracy_regression_vs_control: float = 0.003
    viable_max_test_accuracy_regression_vs_control: float = 0.003
    partial_recovery_margin_vs_lower_bound: float = 0.10
    partial_combined_score_margin_vs_direction_only: float = 0.05

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
                description="Higher-gain unstable reference: same adopted full-vector hard 30 degree cone on the penultimate and terminal micro-steps with the fully live successor increment.",
                intervention_step_offsets=(-2, -1),
                increment_mode="live_successor_increment",
            ),
            _CaseSpec(
                case_name="exact_cached_direction_live_magnitude_lower_bound",
                description="Safe lower-bound reference: replace only the live successor-increment direction with the exact cached direction while keeping the live increment magnitude unchanged.",
                intervention_step_offsets=(-2, -1),
                increment_mode="exact_cached_direction_live_magnitude",
            ),
            _CaseSpec(
                case_name="trust_region_direction_30",
                description="Current partial-signal reference: 30 degree trust-region toward the cached increment direction while keeping the live increment magnitude unchanged.",
                intervention_step_offsets=(-2, -1),
                increment_mode="increment_direction_trust_region_keep_live_norm_30",
                direction_angle_degrees=float(self.current_partial_signal_angle_degrees),
            ),
            _CaseSpec(
                case_name="trust_region_direction_20",
                description="Current stronger partial-signal reference: 20 degree trust-region toward the cached increment direction while keeping the live increment magnitude unchanged.",
                intervention_step_offsets=(-2, -1),
                increment_mode="increment_direction_trust_region_keep_live_norm_30",
                direction_angle_degrees=float(self.stronger_partial_signal_angle_degrees),
            ),
            _CaseSpec(
                case_name="interaction_direction_30_cached_magnitude",
                description="Minimal interaction candidate: 30 degree trust-region toward the cached increment direction with cached increment magnitude.",
                intervention_step_offsets=(-2, -1),
                increment_mode="increment_direction_trust_region_cached_magnitude",
                direction_angle_degrees=float(self.current_partial_signal_angle_degrees),
            ),
            _CaseSpec(
                case_name="interaction_direction_20_cached_magnitude",
                description="Stronger minimal interaction candidate: 20 degree trust-region toward the cached increment direction with cached increment magnitude.",
                intervention_step_offsets=(-2, -1),
                increment_mode="increment_direction_trust_region_cached_magnitude",
                direction_angle_degrees=float(self.stronger_partial_signal_angle_degrees),
            ),
        )


@dataclass
class FMPCTF2SuccessorIncrementInteractionSuiteRunResult:
    run_dir: Path
    config: dict[str, Any]
    aggregate_rows: list[dict[str, Any]]
    summary: dict[str, Any]


def _interaction_component_table() -> list[dict[str, Any]]:
    return [
        {
            "component_name": "delta_on_live",
            "where_produced": "derived from the live preterminal successor value as z_on_next_live - z_on_k_live",
            "source_type": "live_on_policy",
            "used_for": "preterminal live successor increment under joint direction-magnitude reformulation",
        },
        {
            "component_name": "delta_on_cached",
            "where_produced": "derived from the cached preterminal successor value as z_on_next_cached - z_on_k_cached",
            "source_type": "cached_batch_start_reference",
            "used_for": "cached direction and cached magnitude anchors",
        },
        {
            "component_name": "dir_live",
            "where_produced": "normalize(delta_on_live)",
            "source_type": "live_on_policy",
            "used_for": "direction source before trust-region regularization",
        },
        {
            "component_name": "dir_cached",
            "where_produced": "normalize(delta_on_cached)",
            "source_type": "cached_batch_start_reference",
            "used_for": "direction anchor for the interaction candidates",
        },
        {
            "component_name": "norm_live",
            "where_produced": "||delta_on_live||",
            "source_type": "live_on_policy",
            "used_for": "live-magnitude baseline retained by the safe lower-bound and direction-only references",
        },
        {
            "component_name": "norm_cached",
            "where_produced": "||delta_on_cached||",
            "source_type": "cached_batch_start_reference",
            "used_for": "safer magnitude treatment combined with the trust-region direction in the interaction candidates",
        },
    ]


def _suite_config_payload(config: FMPCTF2SuccessorIncrementInteractionSuiteConfig) -> dict[str, Any]:
    return {
        "phase": "Phase Incremental Bridge",
        "stage": "adopted_package_preterminal_successor_increment_direction_magnitude_interaction",
        "base_preset_name": "tf2_corrective_transport_terminal_angleclip_default",
        "transport_family_fixed": True,
        "candidate_specs": [
            {
                "case_name": spec.case_name,
                "description": spec.description,
                "intervention_step_offsets": [int(value) for value in spec.intervention_step_offsets],
                "increment_mode": spec.increment_mode,
                "direction_angle_degrees": spec.direction_angle_degrees,
            }
            for spec in config.case_specs()
        ],
        "increment_direction_magnitude_component_table": _interaction_component_table(),
        "selector_contract_fixed": True,
        "diagnostic_only": True,
        "seeds": [int(seed) for seed in config.seeds],
    }


def _case_base_config(
    config: FMPCTF2SuccessorIncrementInteractionSuiteConfig,
    *,
    direction_angle_degrees: float,
) -> _BaseConfig:
    return _BaseConfig(
        experiment_name=config.experiment_name,
        output_root=config.output_root,
        run_id=config.run_id,
        output_layout=config.output_layout,
        seeds=config.seeds,
        epochs=config.epochs,
        batch_size=config.batch_size,
        eval_steps=config.eval_steps,
        layer_dims=config.layer_dims,
        increment_direction_trust_region_angle_degrees=float(direction_angle_degrees),
        diagnosis_dominance_margin=0.25,
    )


def _case_config_for_spec(
    config: FMPCTF2SuccessorIncrementInteractionSuiteConfig,
    case_spec: _CaseSpec,
) -> _BaseConfig:
    angle = (
        float(case_spec.direction_angle_degrees)
        if case_spec.direction_angle_degrees is not None
        else float(config.current_partial_signal_angle_degrees)
    )
    return _case_base_config(config, direction_angle_degrees=angle)


def _gate_contract_intact(
    config: FMPCTF2SuccessorIncrementInteractionSuiteConfig,
    case_summary: dict[str, Any],
) -> bool:
    return (
        float(case_summary["seed_gate_positive_rate"]) >= float(config.full_gate_seed_gate_positive_rate)
        and float(case_summary["selected_epoch_passes_gate_rate"])
        >= float(config.full_gate_selected_epoch_passes_gate_rate)
        and float(case_summary["selector_fallback_used_rate"]) <= float(config.full_gate_selector_fallback_used_rate)
    )


def _combined_recovery_score(recovery: dict[str, Any]) -> float:
    return float(
        (
            float(recovery["accuracy_gain_retained_fraction_from_control_to_failed_anchor"])
            + float(recovery["gate_robustness_recovery_fraction_from_failed_anchor_to_control"])
            + float(recovery["terminal_rowspace_rms_gain_retained_fraction_from_control_to_failed_anchor"])
        )
        / 3.0
    )


def _diagnose_and_recommend(
    config: FMPCTF2SuccessorIncrementInteractionSuiteConfig,
    by_case: dict[str, dict[str, Any]],
    recovery_vs_failed_anchor: dict[str, dict[str, Any]],
    control_relative_recovery: dict[str, dict[str, Any]],
) -> tuple[_Diagnosis, dict[str, Any], str]:
    control = by_case["adopted_control_terminal_only"]
    failed_anchor = by_case["failed_penultimate_plus_terminal_live"]
    lower_bound = by_case["exact_cached_direction_live_magnitude_lower_bound"]
    ref_30 = by_case["trust_region_direction_30"]
    ref_20 = by_case["trust_region_direction_20"]
    interaction_cases = {
        "interaction_direction_30_cached_magnitude": by_case["interaction_direction_30_cached_magnitude"],
        "interaction_direction_20_cached_magnitude": by_case["interaction_direction_20_cached_magnitude"],
    }

    best_gate_safe_name: str | None = None
    best_gate_safe_case: dict[str, Any] | None = None
    best_gate_safe_recovery: dict[str, Any] | None = None
    for name, case in interaction_cases.items():
        if not _gate_contract_intact(config, case):
            continue
        recovery = recovery_vs_failed_anchor[name]
        if best_gate_safe_case is None or _combined_recovery_score(recovery) > _combined_recovery_score(best_gate_safe_recovery):  # type: ignore[arg-type]
            best_gate_safe_name = name
            best_gate_safe_case = case
            best_gate_safe_recovery = recovery

    best_interaction_name = max(
        interaction_cases,
        key=lambda name: _combined_recovery_score(recovery_vs_failed_anchor[name]),
    )
    best_interaction_recovery = recovery_vs_failed_anchor[best_interaction_name]
    best_direction_only_score = max(
        _combined_recovery_score(recovery_vs_failed_anchor["trust_region_direction_30"]),
        _combined_recovery_score(recovery_vs_failed_anchor["trust_region_direction_20"]),
    )

    evidence = {
        "control_case": control,
        "failed_anchor_case": failed_anchor,
        "safe_lower_bound_case": lower_bound,
        "direction_only_reference_cases": {
            "trust_region_direction_30": ref_30,
            "trust_region_direction_20": ref_20,
        },
        "interaction_cases": interaction_cases,
        "recovery_vs_failed_anchor": recovery_vs_failed_anchor,
        "control_relative_recovery": control_relative_recovery,
        "best_gate_safe_candidate_name": best_gate_safe_name,
        "best_interaction_candidate_name": best_interaction_name,
    }

    lower_bound_accuracy = float(
        recovery_vs_failed_anchor["exact_cached_direction_live_magnitude_lower_bound"][
            "accuracy_gain_retained_fraction_from_control_to_failed_anchor"
        ]
    )
    lower_bound_rowspace = float(
        recovery_vs_failed_anchor["exact_cached_direction_live_magnitude_lower_bound"][
            "terminal_rowspace_rms_gain_retained_fraction_from_control_to_failed_anchor"
        ]
    )

    if best_gate_safe_case is not None and best_gate_safe_recovery is not None:
        materially_beats_lower_bound = (
            float(best_gate_safe_recovery["accuracy_gain_retained_fraction_from_control_to_failed_anchor"])
            >= lower_bound_accuracy + float(config.partial_recovery_margin_vs_lower_bound)
            and float(best_gate_safe_recovery["terminal_rowspace_rms_gain_retained_fraction_from_control_to_failed_anchor"])
            >= lower_bound_rowspace + float(config.partial_recovery_margin_vs_lower_bound)
        )
        control_competitive = (
            float(best_gate_safe_case["mean_val_accuracy"]) + float(config.viable_max_val_accuracy_regression_vs_control)
            >= float(control["mean_val_accuracy"])
            and float(best_gate_safe_case["mean_test_accuracy"]) + float(config.viable_max_test_accuracy_regression_vs_control)
            >= float(control["mean_test_accuracy"])
        )
        if (
            materially_beats_lower_bound
            and control_competitive
            and float(best_gate_safe_recovery["accuracy_gain_retained_fraction_from_control_to_failed_anchor"])
            >= float(config.viable_min_accuracy_recovery_fraction)
            and float(best_gate_safe_recovery["terminal_rowspace_rms_gain_retained_fraction_from_control_to_failed_anchor"])
            >= float(config.viable_min_rowspace_recovery_fraction)
        ):
            return (
                "minimal_direction_magnitude_interaction_is_adoption_viable",
                evidence,
                "run one minimal defaultization confirmation for the strongest interaction candidate against the current adopted control",
            )
        if materially_beats_lower_bound:
            return (
                "interaction_recovers_partially_but_not_adoption_level",
                evidence,
                "run one deeper diagnostic on the live successor increment formulation itself rather than another broader successor-value or cone-family sweep",
            )

    if _combined_recovery_score(best_interaction_recovery) >= (
        best_direction_only_score + float(config.partial_combined_score_margin_vs_direction_only)
    ):
        return (
            "interaction_recovers_partially_but_not_adoption_level",
            evidence,
            "run one deeper diagnostic on the live successor increment formulation itself rather than another broader successor-value or cone-family sweep",
        )

    return (
        "live_successor_increment_interaction_blocker_persists",
        evidence,
        "run one deeper diagnostic on the live successor increment formulation itself rather than another broader successor-value or cone-family sweep",
    )


def run_fmpc_tf2_successor_increment_interaction_suite(
    config: FMPCTF2SuccessorIncrementInteractionSuiteConfig,
) -> FMPCTF2SuccessorIncrementInteractionSuiteRunResult:
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
        case_config = _case_config_for_spec(config, case_spec)
        for seed in config.seeds:
            try:
                aggregate_rows.append(_run_one_case_seed(run_dir, case_config, case_spec, int(seed)))
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
                                "handoff_mode": case_spec.increment_mode,
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
        "exact_cached_direction_live_magnitude_lower_bound",
        "trust_region_direction_30",
        "trust_region_direction_20",
        "interaction_direction_30_cached_magnitude",
        "interaction_direction_20_cached_magnitude",
    ):
        recovery_fractions_vs_failed_anchor[case_name] = _case_recovery_metrics(
            control_summary,
            failed_anchor_summary,
            by_case[case_name],
        )
        control_relative_recovery[case_name] = _control_relative_recovery(control_summary, by_case[case_name])

    diagnosis, diagnosis_evidence, recommended_next_move = _diagnose_and_recommend(
        config,
        by_case,
        recovery_fractions_vs_failed_anchor,
        control_relative_recovery,
    )
    summary = {
        "phase": "Phase Incremental Bridge",
        "stage": "adopted_package_preterminal_successor_increment_direction_magnitude_interaction",
        "num_runs": len(aggregate_rows),
        "increment_direction_magnitude_component_table": _interaction_component_table(),
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
    return FMPCTF2SuccessorIncrementInteractionSuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config),
        aggregate_rows=aggregate_rows,
        summary=summary,
    )
