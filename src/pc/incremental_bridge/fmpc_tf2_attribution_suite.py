from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .fmpc_tf2 import (
    TF2InterleavingStart,
    TF2PresetName,
    TF2SupervisionPolicy,
    TF2ThetaUpdateCadence,
    build_tf2_preset_config,
    run_fmpc_tf2_experiment,
)


@dataclass(frozen=True)
class FMPCTF2AttributionCandidate:
    """One narrow TF2 factor-attribution candidate configuration."""

    key: str
    source_preset: TF2PresetName
    description: str
    micro_steps: int
    incremental_weight_updates: bool
    supervision_policy: TF2SupervisionPolicy
    theta_update_cadence: TF2ThetaUpdateCadence
    interleaving_start: TF2InterleavingStart


@dataclass
class FMPCTF2AttributionSuiteConfig:
    """Small factor-attribution suite for the active TF2 bridge presets."""

    experiment_name: str = "fmpc_tf2_attribution_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    candidate_keys: tuple[str, ...] = (
        "canonical_base",
        "canonical_local_only",
        "canonical_every_2",
        "canonical_terminal_mixed",
        "canonical_after_warmup",
        "canonical_micro_2",
        "corrective_base",
        "corrective_every_2_local",
        "corrective_every_micro_local",
        "corrective_micro_2",
    )
    seeds: tuple[int, ...] = (0, 1, 2)
    epochs: int = 60
    batch_size: int = 128
    eval_steps: int = 15
    layer_dims: tuple[int, ...] = (64, 64, 10)
    slow_pc_reference_runs_path: str | Path = "outputs/incremental_bridge/fmpc_tf2_default_adoption_suite/aggregate_runs.csv"
    slow_pc_reference_name: str = "canonical_slow_pc_digits_baseline"

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"


@dataclass
class FMPCTF2AttributionSuiteRunResult:
    run_dir: Path
    config: dict[str, Any]
    aggregate_rows: list[dict[str, Any]]
    summary: dict[str, Any]


def _candidate_registry() -> dict[str, FMPCTF2AttributionCandidate]:
    return {
        "canonical_base": FMPCTF2AttributionCandidate(
            key="canonical_base",
            source_preset="tf2_canonical",
            description="tf2_canonical baseline",
            micro_steps=4,
            incremental_weight_updates=True,
            supervision_policy="mixed",
            theta_update_cadence="every_micro_step",
            interleaving_start="epoch_0",
        ),
        "canonical_local_only": FMPCTF2AttributionCandidate(
            key="canonical_local_only",
            source_preset="tf2_canonical",
            description="canonical with local_only supervision",
            micro_steps=4,
            incremental_weight_updates=True,
            supervision_policy="local_only",
            theta_update_cadence="every_micro_step",
            interleaving_start="epoch_0",
        ),
        "canonical_every_2": FMPCTF2AttributionCandidate(
            key="canonical_every_2",
            source_preset="tf2_canonical",
            description="canonical with every_2_micro_steps cadence",
            micro_steps=4,
            incremental_weight_updates=True,
            supervision_policy="mixed",
            theta_update_cadence="every_2_micro_steps",
            interleaving_start="epoch_0",
        ),
        "canonical_terminal_mixed": FMPCTF2AttributionCandidate(
            key="canonical_terminal_mixed",
            source_preset="tf2_canonical",
            description="canonical with terminal_only theta updates",
            micro_steps=4,
            incremental_weight_updates=False,
            supervision_policy="mixed",
            theta_update_cadence="terminal_only",
            interleaving_start="epoch_0",
        ),
        "canonical_after_warmup": FMPCTF2AttributionCandidate(
            key="canonical_after_warmup",
            source_preset="tf2_canonical",
            description="canonical with delayed interleaving start",
            micro_steps=4,
            incremental_weight_updates=True,
            supervision_policy="mixed",
            theta_update_cadence="every_micro_step",
            interleaving_start="after_warmup",
        ),
        "canonical_micro_2": FMPCTF2AttributionCandidate(
            key="canonical_micro_2",
            source_preset="tf2_canonical",
            description="canonical with micro_steps=2",
            micro_steps=2,
            incremental_weight_updates=True,
            supervision_policy="mixed",
            theta_update_cadence="every_micro_step",
            interleaving_start="epoch_0",
        ),
        "corrective_base": FMPCTF2AttributionCandidate(
            key="corrective_base",
            source_preset="tf2_corrective_transport_default",
            description="tf2_corrective_transport_default baseline",
            micro_steps=4,
            incremental_weight_updates=False,
            supervision_policy="local_only",
            theta_update_cadence="terminal_only",
            interleaving_start="epoch_0",
        ),
        "corrective_every_2_local": FMPCTF2AttributionCandidate(
            key="corrective_every_2_local",
            source_preset="tf2_corrective_transport_default",
            description="corrective family with every_2_micro_steps cadence",
            micro_steps=4,
            incremental_weight_updates=True,
            supervision_policy="local_only",
            theta_update_cadence="every_2_micro_steps",
            interleaving_start="epoch_0",
        ),
        "corrective_every_micro_local": FMPCTF2AttributionCandidate(
            key="corrective_every_micro_local",
            source_preset="tf2_corrective_transport_default",
            description="corrective family with every_micro_step cadence",
            micro_steps=4,
            incremental_weight_updates=True,
            supervision_policy="local_only",
            theta_update_cadence="every_micro_step",
            interleaving_start="epoch_0",
        ),
        "corrective_micro_2": FMPCTF2AttributionCandidate(
            key="corrective_micro_2",
            source_preset="tf2_corrective_transport_default",
            description="corrective family with micro_steps=2",
            micro_steps=2,
            incremental_weight_updates=False,
            supervision_policy="local_only",
            theta_update_cadence="terminal_only",
            interleaving_start="epoch_0",
        ),
    }


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
        raise FileNotFoundError(f"Required TF2 attribution reference artifact is missing: {path_obj}")
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


def _suite_config_payload(
    config: FMPCTF2AttributionSuiteConfig,
    candidates: tuple[FMPCTF2AttributionCandidate, ...],
) -> dict[str, Any]:
    return {
        "phase": "Phase Incremental Bridge",
        "stage": "corrective_transport_attribution_suite",
        "candidate_keys": [candidate.key for candidate in candidates],
        "seeds": [int(seed) for seed in config.seeds],
        "fixed": {
            "feature_aware_tangents": False,
            "theta_update_budget": "matched",
            "checkpoint_selector": "gate_constrained_accuracy_then_val_accuracy",
            "family_lineage": "tf1_mlp_aug",
            "dataset_name": "digits",
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


def _candidate_run_id(candidate_key: str, seed: int) -> str:
    return f"{candidate_key}_seed{seed}"


def _failure_row(
    *,
    run_index: int,
    candidate: FMPCTF2AttributionCandidate,
    seed: int,
    error: Exception,
) -> dict[str, Any]:
    message = str(error)
    lowered = message.lower()
    return {
        "run_index": int(run_index),
        "candidate_key": candidate.key,
        "preset_name": str(candidate.source_preset),
        "description": candidate.description,
        "seed": int(seed),
        "micro_steps": int(candidate.micro_steps),
        "incremental_weight_updates": bool(candidate.incremental_weight_updates),
        "supervision_policy": str(candidate.supervision_policy),
        "theta_update_cadence": str(candidate.theta_update_cadence),
        "interleaving_start": str(candidate.interleaving_start),
        "theta_update_budget": "matched",
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
    candidate: FMPCTF2AttributionCandidate,
    seed: int,
    result: Any,
    run_dir: Path,
) -> dict[str, Any]:
    summary = result.summary
    return {
        "run_index": int(run_index),
        "candidate_key": candidate.key,
        "preset_name": str(summary["preset_name"]),
        "description": candidate.description,
        "seed": int(seed),
        "micro_steps": int(summary["micro_steps"]),
        "incremental_weight_updates": bool(summary["incremental_weight_updates"]),
        "supervision_policy": str(summary["supervision_policy"]),
        "theta_update_cadence": str(summary["theta_update_cadence"]),
        "interleaving_start": str(summary["interleaving_start"]),
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


def _successful_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["run_status"]) == "completed"]


def _group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["candidate_key"]), []).append(row)
    return grouped


def _candidate_summary(
    rows: list[dict[str, Any]],
    *,
    slow_pc_ref: dict[int, dict[str, float]],
) -> dict[str, Any]:
    successes = _successful_rows(rows)
    exemplar = rows[0]
    payload: dict[str, Any] = {
        "candidate_key": str(exemplar["candidate_key"]),
        "preset_name": str(exemplar["preset_name"]),
        "description": str(exemplar["description"]),
        "micro_steps": int(exemplar["micro_steps"]),
        "incremental_weight_updates": bool(exemplar["incremental_weight_updates"]),
        "supervision_policy": str(exemplar["supervision_policy"]),
        "theta_update_cadence": str(exemplar["theta_update_cadence"]),
        "interleaving_start": str(exemplar["interleaving_start"]),
        "theta_update_budget": str(exemplar["theta_update_budget"]),
        "num_runs": int(len(rows)),
        "num_completed_runs": int(len(successes)),
        "num_failures": int(len(rows) - len(successes)),
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
                "mean_test_accuracy_gap_to_slow_pc": None,
            }
        )
        return payload

    val_accuracies = [float(row["val_accuracy"]) for row in successes]
    test_accuracies = [float(row["test_accuracy"]) for row in successes]
    gate_counts = [float(row["gate_passing_epoch_count"]) for row in successes]
    energies = [float(row["val_transported_final_energy"]) for row in successes]
    slow_test_accuracies = [float(slow_pc_ref[int(row["seed"])]["test_accuracy"]) for row in successes]
    payload.update(
        {
            "mean_val_accuracy": _mean(val_accuracies),
            "std_val_accuracy": _std(val_accuracies),
            "mean_test_accuracy": _mean(test_accuracies),
            "std_test_accuracy": _std(test_accuracies),
            "mean_gate_passing_epoch_count": _mean(gate_counts),
            "mean_val_transported_final_energy": _mean(energies),
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
            "mean_test_gap_to_slow_pc_delta": None,
        }
    return {
        "mean_val_accuracy_delta": float(candidate["mean_val_accuracy"]) - float(reference["mean_val_accuracy"]),
        "mean_test_accuracy_delta": float(candidate["mean_test_accuracy"]) - float(reference["mean_test_accuracy"]),
        "mean_gate_passing_epoch_count_delta": float(candidate["mean_gate_passing_epoch_count"])
        - float(reference["mean_gate_passing_epoch_count"]),
        "mean_test_gap_to_slow_pc_delta": float(candidate["mean_test_accuracy_gap_to_slow_pc"])
        - float(reference["mean_test_accuracy_gap_to_slow_pc"]),
    }


def _delta(best: dict[str, Any], base: dict[str, Any], field: str) -> float | None:
    if best[field] is None or base[field] is None:
        return None
    return float(best[field]) - float(base[field])


def _better(delta: float | None, threshold: float = 0.0) -> bool:
    return delta is not None and float(delta) > float(threshold)


def _safe_delta_by_keys(
    summaries: dict[str, dict[str, Any]],
    better_key: str,
    base_key: str,
    field: str,
) -> float | None:
    if better_key not in summaries or base_key not in summaries:
        return None
    return _delta(summaries[better_key], summaries[base_key], field)


def _attribution_interpretation(summaries: dict[str, dict[str, Any]]) -> tuple[str, str, str]:
    terminal_gain_mixed = _safe_delta_by_keys(
        summaries,
        "canonical_terminal_mixed",
        "canonical_base",
        "mean_test_accuracy",
    )
    terminal_gain_local = _safe_delta_by_keys(
        summaries,
        "corrective_base",
        "corrective_every_micro_local",
        "mean_test_accuracy",
    )
    supervision_gain_terminal = _safe_delta_by_keys(
        summaries,
        "corrective_base",
        "canonical_terminal_mixed",
        "mean_test_accuracy",
    )
    supervision_gain_every_micro = _safe_delta_by_keys(
        summaries,
        "canonical_local_only",
        "canonical_base",
        "mean_test_accuracy",
    )
    micro_gain_corrective = _safe_delta_by_keys(
        summaries,
        "corrective_base",
        "corrective_micro_2",
        "mean_test_accuracy",
    )
    micro_gain_canonical = _safe_delta_by_keys(
        summaries,
        "canonical_base",
        "canonical_micro_2",
        "mean_test_accuracy",
    )
    every_2_gain_local = _safe_delta_by_keys(
        summaries,
        "corrective_every_2_local",
        "corrective_every_micro_local",
        "mean_test_accuracy",
    )

    if _better(terminal_gain_mixed, 0.01) and _better(terminal_gain_local, 0.01):
        mechanism = "current TF2 advantage is primarily explained by removing frequent in-loop theta updates and keeping terminal-only theta updates under matched budget"
        if _better(supervision_gain_terminal, 0.003):
            mechanism += "; local_only supervision adds a smaller secondary gain once cadence is already terminal-only"
        elif _better(supervision_gain_every_micro, 0.003):
            mechanism += "; local_only supervision helps mainly by reducing damage in the every-micro-step regime"
        else:
            mechanism += "; supervision-policy differences look secondary relative to cadence"
    else:
        mechanism = "current TF2 advantage is interaction-dominated and is not explained by a single cadence change alone"

    default_interpretation = "keep tf2_corrective_transport_default as the empirical TF2 preset and keep tf2_canonical as a hypothesis-driven iFMPC candidate"

    if _better(micro_gain_corrective, 0.01) and _better(micro_gain_canonical, 0.005):
        next_step = "increase micro_steps inside the current corrective-transport default while keeping terminal_only, local_only, matched budget, and the current selector fixed"
    elif _better(supervision_gain_terminal, 0.005):
        next_step = "probe a very small local_only-to-low-ratio-mixed supervision relaxation on top of the corrective transport default"
    elif _better(every_2_gain_local, 0.005):
        next_step = "probe every_2_micro_steps cadence inside the local_only corrective family"
    else:
        next_step = "keep the corrective transport default and run one tiny micro-step-count extension before any broader TF2 semantic change"

    return mechanism, default_interpretation, next_step


def run_fmpc_tf2_attribution_suite(
    config: FMPCTF2AttributionSuiteConfig,
) -> FMPCTF2AttributionSuiteRunResult:
    """Run a narrow factor-attribution suite around the active TF2 presets."""

    registry = _candidate_registry()
    candidates = tuple(registry[key] for key in config.candidate_keys)
    slow_pc_rows = _read_csv(config.slow_pc_reference_runs_path)
    slow_pc_ref = _load_slow_pc_reference_by_seed(
        slow_pc_rows,
        reference_name=config.slow_pc_reference_name,
        seeds=config.seeds,
    )

    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, config.resolved_run_id(), config.output_layout)
    )
    _write_json(run_dir / "config.json", _suite_config_payload(config, candidates))

    rows: list[dict[str, Any]] = []
    run_index = 0
    runs_root = run_dir / "runs"
    for candidate in candidates:
        for seed in config.seeds:
            run_index += 1
            tf2_config = build_tf2_preset_config(
                candidate.source_preset,
                output_root=runs_root,
                output_layout="run_id_subdir",
                run_id=_candidate_run_id(candidate.key, seed),
                run_seed=seed,
                data_seed=seed,
                model_init_seed=seed,
                psi_init_seed=seed,
                batch_order_seed=seed,
                epochs=config.epochs,
                batch_size=config.batch_size,
                eval_steps=config.eval_steps,
                layer_dims=config.layer_dims,
                feature_aware_tangents=False,
                theta_update_budget="matched",
                checkpoint_selector="gate_constrained_accuracy_then_val_accuracy",
                micro_steps=candidate.micro_steps,
                incremental_weight_updates=candidate.incremental_weight_updates,
                supervision_policy=candidate.supervision_policy,
                theta_update_cadence=candidate.theta_update_cadence,
                interleaving_start=candidate.interleaving_start,
            )
            try:
                result = run_fmpc_tf2_experiment(tf2_config)
            except Exception as error:  # pragma: no cover - stability path
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

    csv_rows = [
        {
            **row,
            "incremental_weight_updates": str(bool(row["incremental_weight_updates"])),
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

    grouped = _group_rows(rows)
    summaries = {
        key: _candidate_summary(grouped[key], slow_pc_ref=slow_pc_ref)
        for key in config.candidate_keys
    }
    anchor_summary = summaries["corrective_base"]

    pairwise_vs_current_default = {
        key: _pairwise_delta(anchor_summary, summary)
        for key, summary in summaries.items()
    }

    mechanism_attribution, default_interpretation, next_step = _attribution_interpretation(summaries)
    best_summary = max(
        summaries.values(),
        key=lambda item: (
            float("-inf") if item["mean_test_accuracy"] is None else float(item["mean_test_accuracy"]),
            float("-inf") if item["mean_val_accuracy"] is None else float(item["mean_val_accuracy"]),
        ),
    )

    factor_deltas = {
        "cadence_effect_mixed_micro_steps_4": {
            "terminal_only_minus_every_micro_step_test_accuracy": _safe_delta_by_keys(
                summaries,
                "canonical_terminal_mixed",
                "canonical_base",
                "mean_test_accuracy",
            ),
            "every_2_micro_steps_minus_every_micro_step_test_accuracy": _safe_delta_by_keys(
                summaries,
                "canonical_every_2",
                "canonical_base",
                "mean_test_accuracy",
            ),
        },
        "cadence_effect_local_only_micro_steps_4": {
            "terminal_only_minus_every_micro_step_test_accuracy": _safe_delta_by_keys(
                summaries,
                "corrective_base",
                "corrective_every_micro_local",
                "mean_test_accuracy",
            ),
            "every_2_micro_steps_minus_every_micro_step_test_accuracy": _safe_delta_by_keys(
                summaries,
                "corrective_every_2_local",
                "corrective_every_micro_local",
                "mean_test_accuracy",
            ),
        },
        "supervision_effect_terminal_only_micro_steps_4": {
            "local_only_minus_mixed_test_accuracy": _safe_delta_by_keys(
                summaries,
                "corrective_base",
                "canonical_terminal_mixed",
                "mean_test_accuracy",
            )
        },
        "supervision_effect_every_micro_step_micro_steps_4": {
            "local_only_minus_mixed_test_accuracy": _safe_delta_by_keys(
                summaries,
                "canonical_local_only",
                "canonical_base",
                "mean_test_accuracy",
            )
        },
        "interleaving_start_effect_every_micro_mixed_micro_steps_4": {
            "after_warmup_minus_epoch_0_test_accuracy": _safe_delta_by_keys(
                summaries,
                "canonical_after_warmup",
                "canonical_base",
                "mean_test_accuracy",
            )
        },
        "micro_steps_effect_every_micro_mixed": {
            "micro_steps_4_minus_micro_steps_2_test_accuracy": _safe_delta_by_keys(
                summaries,
                "canonical_base",
                "canonical_micro_2",
                "mean_test_accuracy",
            )
        },
        "micro_steps_effect_terminal_only_local_only": {
            "micro_steps_4_minus_micro_steps_2_test_accuracy": _safe_delta_by_keys(
                summaries,
                "corrective_base",
                "corrective_micro_2",
                "mean_test_accuracy",
            )
        },
    }

    any_narrows_gap = any(
        summary["mean_test_accuracy_gap_to_slow_pc"] is not None
        and anchor_summary["mean_test_accuracy_gap_to_slow_pc"] is not None
        and float(summary["mean_test_accuracy_gap_to_slow_pc"])
        > float(anchor_summary["mean_test_accuracy_gap_to_slow_pc"])
        for summary in summaries.values()
    )

    summary = {
        "phase": "Phase Incremental Bridge",
        "stage": "corrective_transport_attribution_suite",
        "num_runs": int(len(rows)),
        "mean_std_val_accuracy_by_configuration": {
            key: {"mean": value["mean_val_accuracy"], "std": value["std_val_accuracy"]}
            for key, value in summaries.items()
        },
        "mean_std_test_accuracy_by_configuration": {
            key: {"mean": value["mean_test_accuracy"], "std": value["std_test_accuracy"]}
            for key, value in summaries.items()
        },
        "mean_gate_passing_epoch_count_by_configuration": {
            key: value["mean_gate_passing_epoch_count"] for key, value in summaries.items()
        },
        "pairwise_comparison_against_current_tf2_default": pairwise_vs_current_default,
        "gap_to_canonical_slow_pc_by_configuration": {
            key: {"mean_test_accuracy_gap": value["mean_test_accuracy_gap_to_slow_pc"]}
            for key, value in summaries.items()
        },
        "factor_deltas": factor_deltas,
        "candidate_summaries": summaries,
        "best_configuration_by_mean_test_accuracy": best_summary,
        "what_explains_the_current_empirical_default": mechanism_attribution,
        "what_should_remain_default": default_interpretation,
        "smallest_promising_next_research_step": next_step,
        "whether_any_tested_configuration_narrows_the_slow_pc_gap_below_current_default": bool(any_narrows_gap),
        "aggregate_csv_path": "aggregate_runs.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)

    return FMPCTF2AttributionSuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config, candidates),
        aggregate_rows=rows,
        summary=summary,
    )
