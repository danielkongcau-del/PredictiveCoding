from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .fmpc_tf2 import (
    build_tf2_corrective_transport_terminal_angleclip_default_config,
    run_fmpc_tf2_experiment,
)
from .fmpc_tf2_gap_decomposition_suite import _evaluate_tf2_supervised_alignment


@dataclass(frozen=True)
class _CandidateSpec:
    candidate_name: str
    description: str
    transported_output_alignment_weight: float
    transported_output_alignment_schedule: str
    intervention_strength_rank: int


@dataclass
class FMPCTF2ReadoutAlignmentSuiteConfig:
    """Run a narrow adopted-package readout-alignment confirmation pass."""

    experiment_name: str = "fmpc_tf2_readout_alignment_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 60
    batch_size: int = 128
    eval_steps: int = 15
    layer_dims: tuple[int, ...] = (64, 64, 10)
    material_test_gain_threshold: float = 0.005
    material_supervised_output_mse_gain_threshold: float = 0.001
    max_gate_rate_drop: float = 0.2
    max_selector_fallback_increase: float = 0.2
    reference_summary_path: str | Path = "outputs/stage_04_incremental_bridge/fmpc_tf2_gap_decomposition_suite/aggregate_summary.json"

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"

    def candidate_specs(self) -> tuple[_CandidateSpec, ...]:
        return (
            _CandidateSpec(
                candidate_name="adopted_control",
                description="Current adopted package with no readout-alignment aid.",
                transported_output_alignment_weight=0.0,
                transported_output_alignment_schedule="none",
                intervention_strength_rank=0,
            ),
            _CandidateSpec(
                candidate_name="readout_align_final_w050",
                description="Adopted package plus a final-micro-step transported readout-alignment weight.",
                transported_output_alignment_weight=0.5,
                transported_output_alignment_schedule="final_micro_step_only",
                intervention_strength_rank=1,
            ),
            _CandidateSpec(
                candidate_name="readout_align_every_w050",
                description="Adopted package plus the same transported readout-alignment weight on every micro-step.",
                transported_output_alignment_weight=0.5,
                transported_output_alignment_schedule="every_micro_step",
                intervention_strength_rank=2,
            ),
        )


@dataclass
class FMPCTF2ReadoutAlignmentSuiteRunResult:
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


def _rate(values: list[bool]) -> float:
    if not values:
        raise ValueError("Rate requires at least one value.")
    return float(sum(1.0 for value in values if bool(value)) / float(len(values)))


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _bool_or_none(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    return bool(value)


def _suite_config_payload(config: FMPCTF2ReadoutAlignmentSuiteConfig) -> dict[str, Any]:
    return {
        "phase": "FMPC Stage 04 Incremental Bridge",
        "stage": "adopted_package_readout_alignment_confirmation",
        "base_preset_name": "tf2_corrective_transport_terminal_angleclip_default",
        "candidate_specs": [
            {
                "candidate_name": spec.candidate_name,
                "description": spec.description,
                "transported_output_alignment_weight": float(spec.transported_output_alignment_weight),
                "transported_output_alignment_schedule": spec.transported_output_alignment_schedule,
                "intervention_strength_rank": int(spec.intervention_strength_rank),
            }
            for spec in config.candidate_specs()
        ],
        "seeds": [int(seed) for seed in config.seeds],
        "thresholds": {
            "material_test_gain_threshold": float(config.material_test_gain_threshold),
            "material_supervised_output_mse_gain_threshold": float(config.material_supervised_output_mse_gain_threshold),
            "max_gate_rate_drop": float(config.max_gate_rate_drop),
            "max_selector_fallback_increase": float(config.max_selector_fallback_increase),
        },
    }


def _candidate_run_id(candidate_name: str, seed: int) -> str:
    return f"{candidate_name}_s{seed}"


def _load_reference_context(reference_summary_path: str | Path) -> dict[str, Any] | None:
    path = Path(reference_summary_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _row(
    run_index: int,
    run_dir: Path,
    candidate: _CandidateSpec,
    seed: int,
    result: Any,
    val_alignment: Any,
    test_alignment: Any,
) -> dict[str, Any]:
    summary = result.summary
    timing = dict(summary.get("timing", {}))
    return {
        "run_index": int(run_index),
        "candidate_name": candidate.candidate_name,
        "seed": int(seed),
        "run_id": str(result.run_dir.name),
        "run_summary_path": _relative_posix(run_dir, result.run_dir / "summary.json"),
        "transported_output_alignment_weight": float(candidate.transported_output_alignment_weight),
        "transported_output_alignment_schedule": candidate.transported_output_alignment_schedule,
        "val_accuracy": float(summary["val_accuracy"]),
        "test_accuracy": float(summary["test_accuracy"]),
        "selected_epoch": int(summary["best_epoch"]),
        "gate_passing_epoch_count": int(summary["gate_passing_epoch_count"]),
        "selected_epoch_passes_gate": bool(summary["selected_epoch_passes_gate"]),
        "selector_fallback_used": bool(summary["selector_fallback_used"]),
        "val_report_output_mse": float(summary["val_loss"]),
        "test_report_output_mse": float(summary["test_loss"]),
        "val_transported_final_energy": float(summary["val_transported_final_energy"]),
        "test_transported_final_energy": float(summary["test_transported_final_energy"]),
        "val_supervised_transported_output_mse": float(val_alignment.transport_output_mse),
        "test_supervised_transported_output_mse": float(test_alignment.transport_output_mse),
        "val_internal_slow_pc_output_mse": float(val_alignment.internal_slow_pc_output_mse),
        "test_internal_slow_pc_output_mse": float(test_alignment.internal_slow_pc_output_mse),
        "val_transport_minus_internal_slow_pc_output_mse": float(
            val_alignment.transport_output_mse - val_alignment.internal_slow_pc_output_mse
        ),
        "test_transport_minus_internal_slow_pc_output_mse": float(
            test_alignment.transport_output_mse - test_alignment.internal_slow_pc_output_mse
        ),
        "runtime_proxy_seconds": float(timing.get("train_wall_time_seconds", 0.0))
        + float(timing.get("final_evaluation_wall_time_seconds", 0.0)),
    }


def _candidate_rows(rows: list[dict[str, Any]], candidate_name: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["candidate_name"]) == candidate_name]


def _candidate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Candidate summary requires at least one row.")
    val_accuracies = [float(row["val_accuracy"]) for row in rows]
    test_accuracies = [float(row["test_accuracy"]) for row in rows]
    selected_epochs = [float(row["selected_epoch"]) for row in rows]
    gate_counts = [float(row["gate_passing_epoch_count"]) for row in rows]
    gate_pass_rate = [bool(row["selected_epoch_passes_gate"]) for row in rows]
    fallback_rate = [bool(row["selector_fallback_used"]) for row in rows]
    runtime_values = [float(row["runtime_proxy_seconds"]) for row in rows]
    val_report_mse = [float(row["val_report_output_mse"]) for row in rows]
    test_report_mse = [float(row["test_report_output_mse"]) for row in rows]
    val_supervised_transport_mse = [float(row["val_supervised_transported_output_mse"]) for row in rows]
    test_supervised_transport_mse = [float(row["test_supervised_transported_output_mse"]) for row in rows]
    val_internal_slow_pc_mse = [float(row["val_internal_slow_pc_output_mse"]) for row in rows]
    test_internal_slow_pc_mse = [float(row["test_internal_slow_pc_output_mse"]) for row in rows]
    val_transport_gap = [float(row["val_transport_minus_internal_slow_pc_output_mse"]) for row in rows]
    test_transport_gap = [float(row["test_transport_minus_internal_slow_pc_output_mse"]) for row in rows]
    val_transport_energy = [float(row["val_transported_final_energy"]) for row in rows]

    return {
        "num_runs": len(rows),
        "mean_val_accuracy": _mean(val_accuracies),
        "std_val_accuracy": _std(val_accuracies),
        "mean_test_accuracy": _mean(test_accuracies),
        "std_test_accuracy": _std(test_accuracies),
        "mean_selected_epoch": _mean(selected_epochs),
        "mean_gate_passing_epoch_count": _mean(gate_counts),
        "selected_epoch_passes_gate_rate": _rate(gate_pass_rate),
        "selector_fallback_used_rate": _rate(fallback_rate),
        "seed_gate_positive_rate": _rate([value > 0.0 for value in gate_counts]),
        "mean_val_transported_final_energy": _mean(val_transport_energy),
        "mean_val_report_output_mse": _mean(val_report_mse),
        "std_val_report_output_mse": _std(val_report_mse),
        "mean_test_report_output_mse": _mean(test_report_mse),
        "std_test_report_output_mse": _std(test_report_mse),
        "mean_val_supervised_transported_output_mse": _mean(val_supervised_transport_mse),
        "std_val_supervised_transported_output_mse": _std(val_supervised_transport_mse),
        "mean_test_supervised_transported_output_mse": _mean(test_supervised_transport_mse),
        "std_test_supervised_transported_output_mse": _std(test_supervised_transport_mse),
        "mean_val_internal_slow_pc_output_mse": _mean(val_internal_slow_pc_mse),
        "std_val_internal_slow_pc_output_mse": _std(val_internal_slow_pc_mse),
        "mean_test_internal_slow_pc_output_mse": _mean(test_internal_slow_pc_mse),
        "std_test_internal_slow_pc_output_mse": _std(test_internal_slow_pc_mse),
        "mean_val_transport_minus_internal_slow_pc_output_mse": _mean(val_transport_gap),
        "std_val_transport_minus_internal_slow_pc_output_mse": _std(val_transport_gap),
        "mean_test_transport_minus_internal_slow_pc_output_mse": _mean(test_transport_gap),
        "std_test_transport_minus_internal_slow_pc_output_mse": _std(test_transport_gap),
        "mean_runtime_proxy_seconds": _mean(runtime_values),
        "std_runtime_proxy_seconds": _std(runtime_values),
    }


def _pairwise_delta(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]) -> dict[str, Any]:
    left_by_seed = {int(row["seed"]): row for row in left_rows}
    right_by_seed = {int(row["seed"]): row for row in right_rows}
    shared_seeds = sorted(set(left_by_seed) & set(right_by_seed))
    if not shared_seeds:
        raise ValueError("Pairwise comparison requires at least one shared seed.")

    def mean_delta(metric_name: str) -> float:
        values = [float(left_by_seed[seed][metric_name]) - float(right_by_seed[seed][metric_name]) for seed in shared_seeds]
        return _mean(values)

    def rate_delta(metric_name: str) -> float:
        left_values = [bool(left_by_seed[seed][metric_name]) for seed in shared_seeds]
        right_values = [bool(right_by_seed[seed][metric_name]) for seed in shared_seeds]
        return _rate(left_values) - _rate(right_values)

    return {
        "shared_seeds": [int(seed) for seed in shared_seeds],
        "mean_val_accuracy_delta": mean_delta("val_accuracy"),
        "mean_test_accuracy_delta": mean_delta("test_accuracy"),
        "mean_gate_passing_epoch_count_delta": mean_delta("gate_passing_epoch_count"),
        "selected_epoch_passes_gate_rate_delta": rate_delta("selected_epoch_passes_gate"),
        "selector_fallback_used_rate_delta": rate_delta("selector_fallback_used"),
        "mean_val_transported_final_energy_delta": mean_delta("val_transported_final_energy"),
        "mean_val_report_output_mse_delta": mean_delta("val_report_output_mse"),
        "mean_test_report_output_mse_delta": mean_delta("test_report_output_mse"),
        "mean_val_supervised_transported_output_mse_delta": mean_delta("val_supervised_transported_output_mse"),
        "mean_test_supervised_transported_output_mse_delta": mean_delta("test_supervised_transported_output_mse"),
        "mean_val_internal_slow_pc_output_mse_delta": mean_delta("val_internal_slow_pc_output_mse"),
        "mean_test_internal_slow_pc_output_mse_delta": mean_delta("test_internal_slow_pc_output_mse"),
        "mean_val_transport_minus_internal_slow_pc_output_mse_delta": mean_delta(
            "val_transport_minus_internal_slow_pc_output_mse"
        ),
        "mean_test_transport_minus_internal_slow_pc_output_mse_delta": mean_delta(
            "test_transport_minus_internal_slow_pc_output_mse"
        ),
        "mean_runtime_proxy_seconds_delta": mean_delta("runtime_proxy_seconds"),
    }


def _report_only_delta_vs_reference(candidate_summary: dict[str, Any], reference_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean_val_accuracy_delta": float(candidate_summary["mean_val_accuracy"] - reference_summary["mean_val_accuracy"]),
        "mean_test_accuracy_delta": float(candidate_summary["mean_test_accuracy"] - reference_summary["mean_test_accuracy"]),
        "mean_val_report_output_mse_delta": float(
            candidate_summary["mean_val_report_output_mse"] - reference_summary["mean_val_report_output_mse"]
        ),
        "mean_test_report_output_mse_delta": float(
            candidate_summary["mean_test_report_output_mse"] - reference_summary["mean_test_report_output_mse"]
        ),
        "mean_runtime_proxy_seconds_delta": float(
            candidate_summary["mean_runtime_proxy_seconds"] - reference_summary["mean_runtime_proxy_seconds"]
        ),
    }


def _qualifies_for_adoption(
    pairwise: dict[str, Any],
    config: FMPCTF2ReadoutAlignmentSuiteConfig,
) -> bool:
    return (
        float(pairwise["mean_test_accuracy_delta"]) >= float(config.material_test_gain_threshold)
        and float(pairwise["mean_val_accuracy_delta"]) >= 0.0
        and float(pairwise["mean_val_supervised_transported_output_mse_delta"])
        <= -float(config.material_supervised_output_mse_gain_threshold)
        and float(pairwise["mean_test_supervised_transported_output_mse_delta"])
        <= -float(config.material_supervised_output_mse_gain_threshold)
        and float(pairwise["selected_epoch_passes_gate_rate_delta"]) >= -float(config.max_gate_rate_drop)
        and float(pairwise["selector_fallback_used_rate_delta"]) <= float(config.max_selector_fallback_increase)
    )


def run_fmpc_tf2_readout_alignment_suite(
    config: FMPCTF2ReadoutAlignmentSuiteConfig,
) -> FMPCTF2ReadoutAlignmentSuiteRunResult:
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
    run_index = 0
    for candidate in config.candidate_specs():
        for seed in config.seeds:
            tf2_config = build_tf2_corrective_transport_terminal_angleclip_default_config(
                experiment_name="tf2",
                output_root=tf2_root,
                output_layout="run_id_subdir",
                run_id=_candidate_run_id(candidate.candidate_name, int(seed)),
                run_seed=int(seed),
                data_seed=int(seed),
                model_init_seed=int(seed),
                psi_init_seed=int(seed),
                batch_order_seed=int(seed),
                epochs=int(config.epochs),
                batch_size=int(config.batch_size),
                eval_steps=int(config.eval_steps),
                layer_dims=tuple(config.layer_dims),
                transported_output_alignment_weight=float(candidate.transported_output_alignment_weight),
                transported_output_alignment_schedule=candidate.transported_output_alignment_schedule,
            )
            result = run_fmpc_tf2_experiment(tf2_config)
            if result.model is None or result.psi_network is None:
                raise ValueError("Readout-alignment suite requires runtime model and psi network objects.")
            from ..datasets import load_digits_split

            split = load_digits_split(
                split_seed=int(tf2_config.data_seed),
                train_fraction=float(tf2_config.train_fraction),
                val_fraction=float(tf2_config.val_fraction),
                test_fraction=float(tf2_config.test_fraction),
            )
            val_alignment = _evaluate_tf2_supervised_alignment(
                result.model,
                result.psi_network,
                tf2_config,
                split.x_val,
                split.y_val,
            )
            test_alignment = _evaluate_tf2_supervised_alignment(
                result.model,
                result.psi_network,
                tf2_config,
                split.x_test,
                split.y_test,
            )
            aggregate_rows.append(
                _row(
                    run_index=run_index,
                    run_dir=run_dir,
                    candidate=candidate,
                    seed=int(seed),
                    result=result,
                    val_alignment=val_alignment,
                    test_alignment=test_alignment,
                )
            )
            run_index += 1

    _write_csv(run_dir / "aggregate_runs.csv", aggregate_rows)

    by_candidate = {
        candidate.candidate_name: _candidate_summary(_candidate_rows(aggregate_rows, candidate.candidate_name))
        for candidate in config.candidate_specs()
    }
    control_name = "adopted_control"
    control_rows = _candidate_rows(aggregate_rows, control_name)
    pairwise_vs_control = {
        candidate.candidate_name: _pairwise_delta(_candidate_rows(aggregate_rows, candidate.candidate_name), control_rows)
        for candidate in config.candidate_specs()
        if candidate.candidate_name != control_name
    }

    qualifying_candidates = [
        candidate
        for candidate in config.candidate_specs()
        if candidate.candidate_name != control_name
        and _qualifies_for_adoption(pairwise_vs_control[candidate.candidate_name], config)
    ]
    qualifying_candidates.sort(
        key=lambda candidate: (
            int(candidate.intervention_strength_rank),
            -float(pairwise_vs_control[candidate.candidate_name]["mean_test_accuracy_delta"]),
        )
    )
    promoted_candidate_name = None if not qualifying_candidates else qualifying_candidates[0].candidate_name

    reference_context = _load_reference_context(config.reference_summary_path)
    report_only_vs_slow_pc: dict[str, Any] | None = None
    report_only_vs_historical: dict[str, Any] | None = None
    if reference_context is not None:
        by_method = dict(reference_context.get("by_method", {}))
        slow_pc_reference = by_method.get("canonical_slow_pc_digits_baseline")
        historical_reference = by_method.get("tf2_corrective_transport_default")
        if slow_pc_reference is not None:
            report_only_vs_slow_pc = {
                candidate_name: _report_only_delta_vs_reference(summary, slow_pc_reference)
                for candidate_name, summary in by_candidate.items()
            }
        if historical_reference is not None:
            report_only_vs_historical = {
                candidate_name: _report_only_delta_vs_reference(summary, historical_reference)
                for candidate_name, summary in by_candidate.items()
            }

    if promoted_candidate_name is not None:
        adoption_decision = "promote_readout_alignment_variant"
        remaining_open_question = (
            "whether the promoted readout-alignment aid remains stable enough for package-level confirmation "
            "against the current adopted default"
        )
    else:
        adoption_decision = "do_not_promote_readout_alignment_variant"
        remaining_open_question = (
            "whether the remaining slow-PC gap should now be pursued through a different narrow issue "
            "inside the adopted package"
        )

    summary = {
        "phase": "FMPC Stage 04 Incremental Bridge",
        "stage": "adopted_package_readout_alignment_confirmation",
        "num_runs": len(aggregate_rows),
        "reference_context_reused": reference_context is not None,
        "reference_context_path": None if reference_context is None else str(Path(config.reference_summary_path).as_posix()),
        "mean_std_val_accuracy_by_candidate": {
            candidate_name: {
                "mean": payload["mean_val_accuracy"],
                "std": payload["std_val_accuracy"],
            }
            for candidate_name, payload in by_candidate.items()
        },
        "mean_std_test_accuracy_by_candidate": {
            candidate_name: {
                "mean": payload["mean_test_accuracy"],
                "std": payload["std_test_accuracy"],
            }
            for candidate_name, payload in by_candidate.items()
        },
        "mean_gate_passing_epoch_count_by_candidate": {
            candidate_name: payload["mean_gate_passing_epoch_count"] for candidate_name, payload in by_candidate.items()
        },
        "selected_epoch_passes_gate_rate_by_candidate": {
            candidate_name: payload["selected_epoch_passes_gate_rate"] for candidate_name, payload in by_candidate.items()
        },
        "selector_fallback_used_rate_by_candidate": {
            candidate_name: payload["selector_fallback_used_rate"] for candidate_name, payload in by_candidate.items()
        },
        "mean_val_transported_final_energy_by_candidate": {
            candidate_name: payload["mean_val_transported_final_energy"] for candidate_name, payload in by_candidate.items()
        },
        "mean_std_report_output_mse_by_candidate": {
            candidate_name: {
                "mean_val": payload["mean_val_report_output_mse"],
                "std_val": payload["std_val_report_output_mse"],
                "mean_test": payload["mean_test_report_output_mse"],
                "std_test": payload["std_test_report_output_mse"],
            }
            for candidate_name, payload in by_candidate.items()
        },
        "mean_std_supervised_transported_output_mse_by_candidate": {
            candidate_name: {
                "mean_val": payload["mean_val_supervised_transported_output_mse"],
                "std_val": payload["std_val_supervised_transported_output_mse"],
                "mean_test": payload["mean_test_supervised_transported_output_mse"],
                "std_test": payload["std_test_supervised_transported_output_mse"],
            }
            for candidate_name, payload in by_candidate.items()
        },
        "mean_std_internal_slow_pc_output_mse_by_candidate": {
            candidate_name: {
                "mean_val": payload["mean_val_internal_slow_pc_output_mse"],
                "std_val": payload["std_val_internal_slow_pc_output_mse"],
                "mean_test": payload["mean_test_internal_slow_pc_output_mse"],
                "std_test": payload["std_test_internal_slow_pc_output_mse"],
            }
            for candidate_name, payload in by_candidate.items()
        },
        "mean_std_transport_minus_internal_slow_pc_output_mse_by_candidate": {
            candidate_name: {
                "mean_val": payload["mean_val_transport_minus_internal_slow_pc_output_mse"],
                "std_val": payload["std_val_transport_minus_internal_slow_pc_output_mse"],
                "mean_test": payload["mean_test_transport_minus_internal_slow_pc_output_mse"],
                "std_test": payload["std_test_transport_minus_internal_slow_pc_output_mse"],
            }
            for candidate_name, payload in by_candidate.items()
        },
        "mean_runtime_proxy_seconds_by_candidate": {
            candidate_name: payload["mean_runtime_proxy_seconds"] for candidate_name, payload in by_candidate.items()
        },
        "by_candidate": by_candidate,
        "pairwise_vs_adopted_control": pairwise_vs_control,
        "report_only_vs_canonical_slow_pc_digits_baseline": report_only_vs_slow_pc,
        "report_only_vs_historical_corrective_reference": report_only_vs_historical,
        "qualifying_candidates": [candidate.candidate_name for candidate in qualifying_candidates],
        "should_promote_readout_alignment_variant": promoted_candidate_name is not None,
        "promoted_candidate_name": promoted_candidate_name,
        "adoption_decision": adoption_decision,
        "remaining_open_question": remaining_open_question,
        "aggregate_csv_path": "aggregate_runs.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF2ReadoutAlignmentSuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config),
        aggregate_rows=aggregate_rows,
        summary=summary,
    )
