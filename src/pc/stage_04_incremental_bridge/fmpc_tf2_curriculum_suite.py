from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from .fmpc_tf2 import build_tf2_corrective_transport_default_config, run_fmpc_tf2_experiment


CurriculumAxis = Literal["default", "identity", "warmup", "ramp", "combined"]


@dataclass(frozen=True)
class FMPCTF2CurriculumCandidate:
    """One narrow bootstrap↔identity curriculum candidate."""

    key: str
    stage: int
    axis: CurriculumAxis
    description: str
    identity_loss_weight: float
    warmup_epochs: int
    hybrid_ramp_epochs: int


@dataclass
class FMPCTF2CurriculumSuiteConfig:
    """Narrow TF2 curriculum study inside the corrective default."""

    experiment_name: str = "fmpc_tf2_curriculum_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 60
    batch_size: int = 128
    eval_steps: int = 15
    layer_dims: tuple[int, ...] = (64, 64, 10)
    current_default_identity_loss_weight: float = 0.2
    current_default_warmup_epochs: int = 5
    current_default_hybrid_ramp_epochs: int = 10
    identity_loss_weight_options: tuple[float, ...] = (0.1, 0.2, 0.4)
    warmup_epochs_options: tuple[int, ...] = (0, 5, 10)
    hybrid_ramp_epochs_options: tuple[int, ...] = (0, 10, 20, 40)
    stage2_material_test_gain: float = 0.005
    stage2_material_val_gain: float = 0.0
    slow_pc_reference_runs_path: str | Path = "outputs/stage_04_incremental_bridge/fmpc_tf2_default_adoption_suite/aggregate_runs.csv"
    slow_pc_reference_name: str = "canonical_slow_pc_digits_baseline"

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"


@dataclass
class FMPCTF2CurriculumSuiteRunResult:
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


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Required TF2 curriculum reference artifact is missing: {path_obj}")
    with path_obj.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _candidate_key(prefix: str, value: float | int) -> str:
    if isinstance(value, float):
        return f"{prefix}_{str(value).replace('.', 'p')}"
    return f"{prefix}_{int(value)}"


def _stage1_candidates(config: FMPCTF2CurriculumSuiteConfig) -> list[FMPCTF2CurriculumCandidate]:
    default = FMPCTF2CurriculumCandidate(
        key="default",
        stage=1,
        axis="default",
        description="current corrective default curriculum",
        identity_loss_weight=float(config.current_default_identity_loss_weight),
        warmup_epochs=int(config.current_default_warmup_epochs),
        hybrid_ramp_epochs=int(config.current_default_hybrid_ramp_epochs),
    )
    candidates: list[FMPCTF2CurriculumCandidate] = [default]
    for identity_loss_weight in config.identity_loss_weight_options:
        if float(identity_loss_weight) == float(config.current_default_identity_loss_weight):
            continue
        candidates.append(
            FMPCTF2CurriculumCandidate(
                key=_candidate_key("id", float(identity_loss_weight)),
                stage=1,
                axis="identity",
                description=f"identity_loss_weight={float(identity_loss_weight):.3g}",
                identity_loss_weight=float(identity_loss_weight),
                warmup_epochs=int(config.current_default_warmup_epochs),
                hybrid_ramp_epochs=int(config.current_default_hybrid_ramp_epochs),
            )
        )
    for warmup_epochs in config.warmup_epochs_options:
        if int(warmup_epochs) == int(config.current_default_warmup_epochs):
            continue
        candidates.append(
            FMPCTF2CurriculumCandidate(
                key=_candidate_key("warmup", int(warmup_epochs)),
                stage=1,
                axis="warmup",
                description=f"warmup_epochs={int(warmup_epochs)}",
                identity_loss_weight=float(config.current_default_identity_loss_weight),
                warmup_epochs=int(warmup_epochs),
                hybrid_ramp_epochs=int(config.current_default_hybrid_ramp_epochs),
            )
        )
    for hybrid_ramp_epochs in config.hybrid_ramp_epochs_options:
        if int(hybrid_ramp_epochs) == int(config.current_default_hybrid_ramp_epochs):
            continue
        candidates.append(
            FMPCTF2CurriculumCandidate(
                key=_candidate_key("ramp", int(hybrid_ramp_epochs)),
                stage=1,
                axis="ramp",
                description=f"hybrid_ramp_epochs={int(hybrid_ramp_epochs)}",
                identity_loss_weight=float(config.current_default_identity_loss_weight),
                warmup_epochs=int(config.current_default_warmup_epochs),
                hybrid_ramp_epochs=int(hybrid_ramp_epochs),
            )
        )
    return candidates


def _suite_config_payload(
    config: FMPCTF2CurriculumSuiteConfig,
    candidates: list[FMPCTF2CurriculumCandidate],
) -> dict[str, Any]:
    return {
        "phase": "FMPC Stage 04 Incremental Bridge",
        "stage": "corrective_transport_curriculum_suite",
        "tested_candidate_keys": [candidate.key for candidate in candidates],
        "seeds": [int(seed) for seed in config.seeds],
        "fixed": {
            "preset_name": "tf2_corrective_transport_default",
            "use_teacher_free_features": True,
            "feature_aware_tangents": False,
            "micro_steps": 4,
            "incremental_weight_updates": False,
            "supervision_policy": "local_only",
            "theta_update_cadence": "terminal_only",
            "theta_update_budget": "matched",
            "bootstrap_integrator": "rk2",
            "bootstrap_substeps": 4,
            "checkpoint_selector": "gate_constrained_accuracy_then_val_accuracy",
            "family_lineage": "tf1_mlp_aug",
        },
        "current_default": {
            "identity_loss_weight": float(config.current_default_identity_loss_weight),
            "warmup_epochs": int(config.current_default_warmup_epochs),
            "hybrid_ramp_epochs": int(config.current_default_hybrid_ramp_epochs),
        },
        "staged_search_rule": {
            "stage_1": "single-axis sweeps only around the current default",
            "stage_2": "run exactly one combined candidate only if one or more axis winners beat the default materially",
            "stage_2_material_test_gain": float(config.stage2_material_test_gain),
            "stage_2_material_val_gain": float(config.stage2_material_val_gain),
        },
        "slow_pc_reference_runs_path": str(config.slow_pc_reference_runs_path),
        "slow_pc_reference_name": str(config.slow_pc_reference_name),
    }


def _load_slow_pc_reference_by_seed(
    rows: list[dict[str, str]],
    *,
    reference_name: str,
    seeds: tuple[int, ...],
) -> dict[int, dict[str, float]]:
    filtered = [row for row in rows if str(row.get("preset_name", "")) == reference_name]
    if not filtered:
        raise ValueError(f"No slow-PC reference rows found for '{reference_name}'.")
    by_seed = {
        int(row["seed"]): {
            "val_accuracy": float(row["val_accuracy"]),
            "test_accuracy": float(row["test_accuracy"]),
        }
        for row in filtered
    }
    missing = [seed for seed in seeds if seed not in by_seed]
    if missing:
        raise ValueError(f"Slow-PC reference rows are missing seeds: {missing}.")
    return {int(seed): by_seed[int(seed)] for seed in seeds}


def _run_id(candidate: FMPCTF2CurriculumCandidate, seed: int) -> str:
    return f"{candidate.key}_seed{seed}"


def _success_row(
    *,
    run_index: int,
    candidate: FMPCTF2CurriculumCandidate,
    seed: int,
    result: Any,
    run_dir: Path,
) -> dict[str, Any]:
    summary = result.summary
    total_wall_time_seconds = float(summary["timing"]["train_wall_time_seconds"]) + float(
        summary["timing"]["final_evaluation_wall_time_seconds"]
    )
    return {
        "run_index": int(run_index),
        "candidate_key": candidate.key,
        "stage": int(candidate.stage),
        "axis": candidate.axis,
        "description": candidate.description,
        "seed": int(seed),
        "identity_loss_weight": float(candidate.identity_loss_weight),
        "warmup_epochs": int(candidate.warmup_epochs),
        "hybrid_ramp_epochs": int(candidate.hybrid_ramp_epochs),
        "checkpoint_selector": str(summary["checkpoint_selector"]),
        "val_accuracy": float(summary["val_accuracy"]),
        "test_accuracy": float(summary["test_accuracy"]),
        "gate_passing_epoch_count": int(summary["gate_passing_epoch_count"]),
        "val_transported_final_energy": float(summary["val_transported_final_energy"]),
        "selected_epoch": int(summary["best_epoch"]),
        "selected_epoch_passes_gate": bool(summary["selected_epoch_passes_gate"]),
        "selector_fallback_used": bool(summary["selector_fallback_used"]),
        "train_wall_time_seconds": float(summary["timing"]["train_wall_time_seconds"]),
        "total_wall_time_seconds": float(total_wall_time_seconds),
        "run_status": "completed",
        "nan_or_inf_failure": False,
        "failure_type": "",
        "failure_message": "",
        "run_summary_path": _relative_posix(run_dir, result.run_dir / "summary.json"),
    }


def _failure_row(
    *,
    run_index: int,
    candidate: FMPCTF2CurriculumCandidate,
    seed: int,
    error: Exception,
) -> dict[str, Any]:
    message = str(error)
    lowered = message.lower()
    return {
        "run_index": int(run_index),
        "candidate_key": candidate.key,
        "stage": int(candidate.stage),
        "axis": candidate.axis,
        "description": candidate.description,
        "seed": int(seed),
        "identity_loss_weight": float(candidate.identity_loss_weight),
        "warmup_epochs": int(candidate.warmup_epochs),
        "hybrid_ramp_epochs": int(candidate.hybrid_ramp_epochs),
        "checkpoint_selector": "",
        "val_accuracy": None,
        "test_accuracy": None,
        "gate_passing_epoch_count": None,
        "val_transported_final_energy": None,
        "selected_epoch": None,
        "selected_epoch_passes_gate": None,
        "selector_fallback_used": None,
        "train_wall_time_seconds": None,
        "total_wall_time_seconds": None,
        "run_status": "failed",
        "nan_or_inf_failure": bool("nan" in lowered or "inf" in lowered),
        "failure_type": type(error).__name__,
        "failure_message": message,
        "run_summary_path": "",
    }


def _rows_for_candidate(rows: list[dict[str, Any]], candidate_key: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["candidate_key"]) == candidate_key]


def _successful_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["run_status"]) == "completed"]


def _candidate_summary(
    rows: list[dict[str, Any]],
    *,
    slow_pc_ref: dict[int, dict[str, float]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Candidate summary requires at least one row.")
    exemplar = rows[0]
    successes = _successful_rows(rows)
    payload: dict[str, Any] = {
        "candidate_key": str(exemplar["candidate_key"]),
        "stage": int(exemplar["stage"]),
        "axis": str(exemplar["axis"]),
        "description": str(exemplar["description"]),
        "identity_loss_weight": float(exemplar["identity_loss_weight"]),
        "warmup_epochs": int(exemplar["warmup_epochs"]),
        "hybrid_ramp_epochs": int(exemplar["hybrid_ramp_epochs"]),
        "num_runs": int(len(rows)),
        "num_completed_runs": int(len(successes)),
        "num_failures": int(sum(1 for row in rows if str(row["run_status"]) == "failed")),
        "nan_or_inf_failure_count": int(sum(1 for row in rows if bool(row["nan_or_inf_failure"]))),
    }
    if not successes:
        payload.update(
            {
                "mean_val_accuracy": None,
                "std_val_accuracy": None,
                "mean_test_accuracy": None,
                "std_test_accuracy": None,
                "mean_gate_passing_epoch_count": None,
                "mean_val_transported_final_energy": None,
                "mean_total_wall_time_seconds": None,
                "std_total_wall_time_seconds": None,
                "mean_test_accuracy_gap_to_slow_pc": None,
            }
        )
        return payload

    val_accuracies = [float(row["val_accuracy"]) for row in successes]
    test_accuracies = [float(row["test_accuracy"]) for row in successes]
    gate_counts = [float(row["gate_passing_epoch_count"]) for row in successes]
    energies = [float(row["val_transported_final_energy"]) for row in successes]
    total_times = [float(row["total_wall_time_seconds"]) for row in successes]
    slow_test_accuracies = [float(slow_pc_ref[int(row["seed"])]["test_accuracy"]) for row in successes]
    payload.update(
        {
            "mean_val_accuracy": _mean(val_accuracies),
            "std_val_accuracy": _std(val_accuracies),
            "mean_test_accuracy": _mean(test_accuracies),
            "std_test_accuracy": _std(test_accuracies),
            "mean_gate_passing_epoch_count": _mean(gate_counts),
            "mean_val_transported_final_energy": _mean(energies),
            "mean_total_wall_time_seconds": _mean(total_times),
            "std_total_wall_time_seconds": _std(total_times),
            "mean_test_accuracy_gap_to_slow_pc": _mean(test_accuracies) - _mean(slow_test_accuracies),
        }
    )
    return payload


def _pairwise_delta(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if reference["mean_test_accuracy"] is None or candidate["mean_test_accuracy"] is None:
        return {
            "mean_val_accuracy_delta": None,
            "mean_test_accuracy_delta": None,
            "mean_gate_passing_epoch_count_delta": None,
            "mean_val_transported_final_energy_delta": None,
            "mean_total_wall_time_seconds_delta": None,
            "mean_test_gap_to_slow_pc_delta": None,
        }
    return {
        "mean_val_accuracy_delta": float(candidate["mean_val_accuracy"]) - float(reference["mean_val_accuracy"]),
        "mean_test_accuracy_delta": float(candidate["mean_test_accuracy"]) - float(reference["mean_test_accuracy"]),
        "mean_gate_passing_epoch_count_delta": float(candidate["mean_gate_passing_epoch_count"])
        - float(reference["mean_gate_passing_epoch_count"]),
        "mean_val_transported_final_energy_delta": float(candidate["mean_val_transported_final_energy"])
        - float(reference["mean_val_transported_final_energy"]),
        "mean_total_wall_time_seconds_delta": float(candidate["mean_total_wall_time_seconds"])
        - float(reference["mean_total_wall_time_seconds"]),
        "mean_test_gap_to_slow_pc_delta": float(candidate["mean_test_accuracy_gap_to_slow_pc"])
        - float(reference["mean_test_accuracy_gap_to_slow_pc"]),
    }


def _is_material_axis_winner(
    default_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    *,
    test_gain: float,
    val_gain: float,
) -> bool:
    if default_summary["mean_test_accuracy"] is None or candidate_summary["mean_test_accuracy"] is None:
        return False
    if int(candidate_summary["num_failures"]) > int(default_summary["num_failures"]):
        return False
    if int(candidate_summary["nan_or_inf_failure_count"]) > int(default_summary["nan_or_inf_failure_count"]):
        return False
    return (
        float(candidate_summary["mean_test_accuracy"]) - float(default_summary["mean_test_accuracy"]) >= float(test_gain)
        and float(candidate_summary["mean_val_accuracy"]) - float(default_summary["mean_val_accuracy"]) >= float(val_gain)
    )


def _best_summary(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        summaries,
        key=lambda item: (
            float("-inf") if item["mean_test_accuracy"] is None else float(item["mean_test_accuracy"]),
            float("-inf") if item["mean_val_accuracy"] is None else float(item["mean_val_accuracy"]),
        ),
    )


def _build_stage2_candidate(
    default_candidate: FMPCTF2CurriculumCandidate,
    *,
    axis_winners: dict[str, dict[str, Any]],
) -> FMPCTF2CurriculumCandidate | None:
    identity_loss_weight = float(default_candidate.identity_loss_weight)
    warmup_epochs = int(default_candidate.warmup_epochs)
    hybrid_ramp_epochs = int(default_candidate.hybrid_ramp_epochs)
    changed = False
    if "identity" in axis_winners:
        identity_loss_weight = float(axis_winners["identity"]["identity_loss_weight"])
        changed = True
    if "warmup" in axis_winners:
        warmup_epochs = int(axis_winners["warmup"]["warmup_epochs"])
        changed = True
    if "ramp" in axis_winners:
        hybrid_ramp_epochs = int(axis_winners["ramp"]["hybrid_ramp_epochs"])
        changed = True
    if not changed:
        return None
    if (
        identity_loss_weight == float(default_candidate.identity_loss_weight)
        and warmup_epochs == int(default_candidate.warmup_epochs)
        and hybrid_ramp_epochs == int(default_candidate.hybrid_ramp_epochs)
    ):
        return None
    return FMPCTF2CurriculumCandidate(
        key="combined_axis_winners",
        stage=2,
        axis="combined",
        description="combined non-default axis winners from stage 1",
        identity_loss_weight=identity_loss_weight,
        warmup_epochs=warmup_epochs,
        hybrid_ramp_epochs=hybrid_ramp_epochs,
    )


def run_fmpc_tf2_curriculum_suite(
    config: FMPCTF2CurriculumSuiteConfig,
) -> FMPCTF2CurriculumSuiteRunResult:
    """Run a narrow curriculum study inside the current corrective TF2 default."""

    stage1_candidates = _stage1_candidates(config)
    slow_pc_rows = _read_csv(config.slow_pc_reference_runs_path)
    slow_pc_ref = _load_slow_pc_reference_by_seed(
        slow_pc_rows,
        reference_name=config.slow_pc_reference_name,
        seeds=config.seeds,
    )

    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, config.resolved_run_id(), config.output_layout)
    )
    _write_json(run_dir / "config.json", _suite_config_payload(config, stage1_candidates))
    runs_root = run_dir / "runs"

    rows: list[dict[str, Any]] = []
    run_index = 0
    for candidate in stage1_candidates:
        for seed in config.seeds:
            run_index += 1
            run_config = build_tf2_corrective_transport_default_config(
                output_root=runs_root,
                output_layout="run_id_subdir",
                run_id=_run_id(candidate, seed),
                run_seed=seed,
                data_seed=seed,
                model_init_seed=seed,
                psi_init_seed=seed,
                batch_order_seed=seed,
                epochs=config.epochs,
                batch_size=config.batch_size,
                eval_steps=config.eval_steps,
                layer_dims=config.layer_dims,
                identity_loss_weight=candidate.identity_loss_weight,
                warmup_epochs=candidate.warmup_epochs,
                hybrid_ramp_epochs=candidate.hybrid_ramp_epochs,
            )
            try:
                result = run_fmpc_tf2_experiment(run_config)
            except Exception as error:  # pragma: no cover - failure path
                rows.append(
                    _failure_row(
                        run_index=run_index,
                        candidate=candidate,
                        seed=seed,
                        error=error,
                    )
                )
            else:
                rows.append(
                    _success_row(
                        run_index=run_index,
                        candidate=candidate,
                        seed=seed,
                        result=result,
                        run_dir=run_dir,
                    )
                )

    default_candidate = next(candidate for candidate in stage1_candidates if candidate.key == "default")
    stage1_summaries = {
        candidate.key: _candidate_summary(_rows_for_candidate(rows, candidate.key), slow_pc_ref=slow_pc_ref)
        for candidate in stage1_candidates
    }
    default_summary = stage1_summaries["default"]

    axis_winners: dict[str, dict[str, Any]] = {}
    for axis in ("identity", "warmup", "ramp"):
        axis_summaries = [
            summary
            for summary in stage1_summaries.values()
            if str(summary["axis"]) == axis
        ]
        if not axis_summaries:
            continue
        winner = _best_summary(axis_summaries)
        if _is_material_axis_winner(
            default_summary,
            winner,
            test_gain=config.stage2_material_test_gain,
            val_gain=config.stage2_material_val_gain,
        ):
            axis_winners[axis] = winner

    stage2_candidate = _build_stage2_candidate(default_candidate, axis_winners=axis_winners)
    tested_candidates = list(stage1_candidates)
    if stage2_candidate is not None:
        tested_candidates.append(stage2_candidate)
        for seed in config.seeds:
            run_index += 1
            run_config = build_tf2_corrective_transport_default_config(
                output_root=runs_root,
                output_layout="run_id_subdir",
                run_id=_run_id(stage2_candidate, seed),
                run_seed=seed,
                data_seed=seed,
                model_init_seed=seed,
                psi_init_seed=seed,
                batch_order_seed=seed,
                epochs=config.epochs,
                batch_size=config.batch_size,
                eval_steps=config.eval_steps,
                layer_dims=config.layer_dims,
                identity_loss_weight=stage2_candidate.identity_loss_weight,
                warmup_epochs=stage2_candidate.warmup_epochs,
                hybrid_ramp_epochs=stage2_candidate.hybrid_ramp_epochs,
            )
            try:
                result = run_fmpc_tf2_experiment(run_config)
            except Exception as error:  # pragma: no cover - failure path
                rows.append(
                    _failure_row(
                        run_index=run_index,
                        candidate=stage2_candidate,
                        seed=seed,
                        error=error,
                    )
                )
            else:
                rows.append(
                    _success_row(
                        run_index=run_index,
                        candidate=stage2_candidate,
                        seed=seed,
                        result=result,
                        run_dir=run_dir,
                    )
                )

    _write_json(run_dir / "config.json", _suite_config_payload(config, tested_candidates))

    candidate_summaries = {
        candidate.key: _candidate_summary(_rows_for_candidate(rows, candidate.key), slow_pc_ref=slow_pc_ref)
        for candidate in tested_candidates
    }
    best_summary = _best_summary(list(candidate_summaries.values()))

    mean_std_val_accuracy_by_configuration: dict[str, dict[str, Any]] = {}
    mean_std_test_accuracy_by_configuration: dict[str, dict[str, Any]] = {}
    mean_gate_passing_epoch_count_by_configuration: dict[str, Any] = {}
    mean_val_transported_final_energy_by_configuration: dict[str, Any] = {}
    mean_wall_clock_runtime_by_configuration: dict[str, Any] = {}
    gap_to_canonical_slow_pc_by_configuration: dict[str, Any] = {}
    pairwise_comparison_against_current_corrective_default: dict[str, dict[str, Any]] = {}
    for candidate in tested_candidates:
        summary = candidate_summaries[candidate.key]
        key = candidate.key
        mean_std_val_accuracy_by_configuration[key] = {
            "mean": summary["mean_val_accuracy"],
            "std": summary["std_val_accuracy"],
        }
        mean_std_test_accuracy_by_configuration[key] = {
            "mean": summary["mean_test_accuracy"],
            "std": summary["std_test_accuracy"],
        }
        mean_gate_passing_epoch_count_by_configuration[key] = summary["mean_gate_passing_epoch_count"]
        mean_val_transported_final_energy_by_configuration[key] = summary["mean_val_transported_final_energy"]
        mean_wall_clock_runtime_by_configuration[key] = {
            "mean_total_wall_time_seconds": summary["mean_total_wall_time_seconds"],
            "std_total_wall_time_seconds": summary["std_total_wall_time_seconds"],
        }
        gap_to_canonical_slow_pc_by_configuration[key] = {
            "mean_test_accuracy_gap": summary["mean_test_accuracy_gap_to_slow_pc"],
        }
        pairwise_comparison_against_current_corrective_default[key] = _pairwise_delta(default_summary, summary)

    tested_non_default_summaries = [
        summary for key, summary in candidate_summaries.items() if key != "default" and summary["mean_test_accuracy"] is not None
    ]
    curriculum_improves = bool(
        tested_non_default_summaries
        and float(best_summary["mean_test_accuracy"]) - float(default_summary["mean_test_accuracy"]) >= 0.005
        and float(best_summary["mean_val_accuracy"]) >= float(default_summary["mean_val_accuracy"])
        and int(best_summary["num_failures"]) == 0
        and int(best_summary["nan_or_inf_failure_count"]) == 0
        and str(best_summary["candidate_key"]) != "default"
    )

    if curriculum_improves:
        should_change_default = True
        interpretation = (
            "the tested curriculum knobs improve fixed-4-step corrective transport quality over the current default"
        )
        next_move = "adopt the improved curriculum inside tf2_corrective_transport_default and validate it against the slow-PC baseline with a narrow multiseed confirmation pass"
    else:
        should_change_default = False
        interpretation = (
            "no tested bootstrap↔identity curriculum beats the current fixed-4-step corrective default by a material margin"
        )
        next_move = "keep the current corrective default and narrow the next move to bootstrap-target fidelity rather than curriculum"

    stage1_trace = {
        "identity_sweep": [candidate.key for candidate in stage1_candidates if candidate.axis in {"default", "identity"}],
        "warmup_sweep": [candidate.key for candidate in stage1_candidates if candidate.axis in {"default", "warmup"}],
        "ramp_sweep": [candidate.key for candidate in stage1_candidates if candidate.axis in {"default", "ramp"}],
    }
    stage2_trace = {
        "material_axis_winners": {
            axis: {
                "candidate_key": winner["candidate_key"],
                "identity_loss_weight": winner["identity_loss_weight"],
                "warmup_epochs": winner["warmup_epochs"],
                "hybrid_ramp_epochs": winner["hybrid_ramp_epochs"],
            }
            for axis, winner in axis_winners.items()
        },
        "combined_candidate_tested": None
        if stage2_candidate is None
        else {
            "candidate_key": stage2_candidate.key,
            "identity_loss_weight": stage2_candidate.identity_loss_weight,
            "warmup_epochs": stage2_candidate.warmup_epochs,
            "hybrid_ramp_epochs": stage2_candidate.hybrid_ramp_epochs,
        },
    }

    csv_rows = [
        {
            **row,
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

    summary = {
        "phase": "FMPC Stage 04 Incremental Bridge",
        "stage": "corrective_transport_curriculum_suite",
        "num_runs": int(len(rows)),
        "tested_configuration_order": [candidate.key for candidate in tested_candidates],
        "staged_search_trace": {
            "stage_1": stage1_trace,
            "stage_2": stage2_trace,
        },
        "mean_std_val_accuracy_by_configuration": mean_std_val_accuracy_by_configuration,
        "mean_std_test_accuracy_by_configuration": mean_std_test_accuracy_by_configuration,
        "mean_gate_passing_epoch_count_by_configuration": mean_gate_passing_epoch_count_by_configuration,
        "mean_val_transported_final_energy_by_configuration": mean_val_transported_final_energy_by_configuration,
        "mean_wall_clock_runtime_by_configuration": mean_wall_clock_runtime_by_configuration,
        "gap_to_canonical_slow_pc_by_configuration": gap_to_canonical_slow_pc_by_configuration,
        "pairwise_comparison_against_current_corrective_default": pairwise_comparison_against_current_corrective_default,
        "best_configuration_by_mean_test_accuracy": best_summary,
        "does_better_bootstrap_identity_curriculum_improve_fixed_4_step_transport_quality": curriculum_improves,
        "should_corrective_default_change": bool(should_change_default),
        "if_no_improvement_next_move_is_bootstrap_target_fidelity": bool(not curriculum_improves),
        "bootstrap_identity_curriculum_interpretation": interpretation,
        "next_single_narrow_research_move": next_move,
        "aggregate_csv_path": "aggregate_runs.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF2CurriculumSuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config, tested_candidates),
        aggregate_rows=rows,
        summary=summary,
    )
