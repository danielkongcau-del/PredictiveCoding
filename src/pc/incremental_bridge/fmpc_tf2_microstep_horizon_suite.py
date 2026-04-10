from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from .fmpc_tf2 import build_tf2_corrective_transport_default_config, run_fmpc_tf2_experiment


TF2MicrostepProtocol = Literal["fixed_outer_training", "matched_inner_compute"]


@dataclass
class FMPCTF2MicrostepHorizonSuiteConfig:
    """Narrow corrective-family micro-step horizon study."""

    experiment_name: str = "fmpc_tf2_microstep_horizon_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    protocols: tuple[TF2MicrostepProtocol, ...] = (
        "fixed_outer_training",
        "matched_inner_compute",
    )
    micro_steps_options: tuple[int, ...] = (4, 6, 8, 10)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    base_epochs: int = 60
    batch_size: int = 128
    eval_steps: int = 15
    layer_dims: tuple[int, ...] = (64, 64, 10)
    slow_pc_reference_runs_path: str | Path = "outputs/incremental_bridge/fmpc_tf2_default_adoption_suite/aggregate_runs.csv"
    slow_pc_reference_name: str = "canonical_slow_pc_digits_baseline"
    matched_budget_reference_micro_steps: int = 4

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"


@dataclass
class FMPCTF2MicrostepHorizonSuiteRunResult:
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
        raise FileNotFoundError(f"Required TF2 micro-step reference artifact is missing: {path_obj}")
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


def _epochs_for_protocol(
    protocol: TF2MicrostepProtocol,
    *,
    micro_steps: int,
    base_epochs: int,
    reference_micro_steps: int,
) -> int:
    if protocol == "fixed_outer_training":
        return int(base_epochs)
    if protocol == "matched_inner_compute":
        matched_epochs = round(float(base_epochs) * float(reference_micro_steps) / float(micro_steps))
        return max(1, int(matched_epochs))
    raise ValueError(f"Unsupported protocol '{protocol}'.")


def _effective_microstep_budget(epochs: int, micro_steps: int) -> int:
    return int(epochs) * int(micro_steps)


def _protocol_label(protocol: TF2MicrostepProtocol) -> str:
    return str(protocol)


def _suite_config_payload(config: FMPCTF2MicrostepHorizonSuiteConfig) -> dict[str, Any]:
    return {
        "phase": "Phase Incremental Bridge",
        "stage": "corrective_microstep_horizon_suite",
        "protocols": [str(protocol) for protocol in config.protocols],
        "micro_steps_options": [int(value) for value in config.micro_steps_options],
        "seeds": [int(seed) for seed in config.seeds],
        "fixed": {
            "preset_name": "tf2_corrective_transport_default",
            "feature_aware_tangents": False,
            "incremental_weight_updates": False,
            "supervision_policy": "local_only",
            "theta_update_cadence": "terminal_only",
            "theta_update_budget": "matched",
            "checkpoint_selector": "gate_constrained_accuracy_then_val_accuracy",
            "family_lineage": "tf1_mlp_aug",
        },
        "matching_rule": {
            "reference_micro_steps": int(config.matched_budget_reference_micro_steps),
            "base_epochs": int(config.base_epochs),
            "matched_inner_compute_epochs_rule": "round(base_epochs * reference_micro_steps / micro_steps)",
            "effective_microstep_budget": "epochs * micro_steps",
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


def _run_id(protocol: TF2MicrostepProtocol, micro_steps: int, seed: int) -> str:
    return f"{protocol}_m{micro_steps}_seed{seed}"


def _failure_row(
    *,
    run_index: int,
    protocol: TF2MicrostepProtocol,
    micro_steps: int,
    epochs_used: int,
    seed: int,
    error: Exception,
) -> dict[str, Any]:
    message = str(error)
    lowered = message.lower()
    return {
        "run_index": int(run_index),
        "protocol": _protocol_label(protocol),
        "seed": int(seed),
        "micro_steps": int(micro_steps),
        "epochs_used": int(epochs_used),
        "effective_microstep_budget": int(_effective_microstep_budget(epochs_used, micro_steps)),
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


def _skipped_row(
    *,
    run_index: int,
    protocol: TF2MicrostepProtocol,
    micro_steps: int,
    epochs_used: int,
    seed: int,
    instability_start_micro_steps: int,
) -> dict[str, Any]:
    return {
        "run_index": int(run_index),
        "protocol": _protocol_label(protocol),
        "seed": int(seed),
        "micro_steps": int(micro_steps),
        "epochs_used": int(epochs_used),
        "effective_microstep_budget": int(_effective_microstep_budget(epochs_used, micro_steps)),
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
        "run_status": "skipped_after_instability",
        "nan_or_inf_failure": False,
        "failure_type": "",
        "failure_message": f"skipped because instability first appeared at micro_steps={int(instability_start_micro_steps)}",
        "run_summary_path": "",
    }


def _success_row(
    *,
    run_index: int,
    protocol: TF2MicrostepProtocol,
    micro_steps: int,
    epochs_used: int,
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
        "protocol": _protocol_label(protocol),
        "seed": int(seed),
        "micro_steps": int(micro_steps),
        "epochs_used": int(epochs_used),
        "effective_microstep_budget": int(_effective_microstep_budget(epochs_used, micro_steps)),
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


def _successful_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["run_status"]) == "completed"]


def _rows_for_key(rows: list[dict[str, Any]], protocol: TF2MicrostepProtocol, micro_steps: int) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row["protocol"]) == _protocol_label(protocol) and int(row["micro_steps"]) == int(micro_steps)
    ]


def _group_summary(
    rows: list[dict[str, Any]],
    *,
    slow_pc_ref: dict[int, dict[str, float]],
) -> dict[str, Any]:
    successes = _successful_rows(rows)
    exemplar = rows[0]
    payload: dict[str, Any] = {
        "protocol": str(exemplar["protocol"]),
        "micro_steps": int(exemplar["micro_steps"]),
        "epochs_used": int(exemplar["epochs_used"]),
        "effective_microstep_budget": int(exemplar["effective_microstep_budget"]),
        "num_runs": int(len(rows)),
        "num_completed_runs": int(len(successes)),
        "num_failures": int(sum(1 for row in rows if str(row["run_status"]) == "failed")),
        "num_skipped_runs": int(sum(1 for row in rows if str(row["run_status"]) == "skipped_after_instability")),
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
                "mean_train_wall_time_seconds": None,
                "std_train_wall_time_seconds": None,
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
    train_times = [float(row["train_wall_time_seconds"]) for row in successes]
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
            "mean_train_wall_time_seconds": _mean(train_times),
            "std_train_wall_time_seconds": _std(train_times),
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


def _material_matched_win(reference: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if reference["mean_test_accuracy"] is None or candidate["mean_test_accuracy"] is None:
        return False
    test_delta = float(candidate["mean_test_accuracy"]) - float(reference["mean_test_accuracy"])
    val_delta = float(candidate["mean_val_accuracy"]) - float(reference["mean_val_accuracy"])
    failures = int(candidate["num_failures"]) + int(candidate["nan_or_inf_failure_count"])
    return test_delta >= 0.005 and val_delta >= 0.0 and failures == 0


def _best_completed_summary(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    return max(
        summaries,
        key=lambda item: (
            float("-inf") if item["mean_test_accuracy"] is None else float(item["mean_test_accuracy"]),
            float("-inf") if item["mean_val_accuracy"] is None else float(item["mean_val_accuracy"]),
        ),
    )


def run_fmpc_tf2_microstep_horizon_suite(
    config: FMPCTF2MicrostepHorizonSuiteConfig,
) -> FMPCTF2MicrostepHorizonSuiteRunResult:
    """Run a compute-aware micro-step horizon study inside the corrective TF2 family."""

    slow_pc_rows = _read_csv(config.slow_pc_reference_runs_path)
    slow_pc_ref = _load_slow_pc_reference_by_seed(
        slow_pc_rows,
        reference_name=config.slow_pc_reference_name,
        seeds=config.seeds,
    )

    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, config.resolved_run_id(), config.output_layout)
    )
    _write_json(run_dir / "config.json", _suite_config_payload(config))

    rows: list[dict[str, Any]] = []
    run_index = 0
    runs_root = run_dir / "runs"
    instability_start_by_protocol: dict[str, int | None] = {str(protocol): None for protocol in config.protocols}

    for protocol in config.protocols:
        protocol_label = _protocol_label(protocol)
        for micro_steps in config.micro_steps_options:
            epochs_used = _epochs_for_protocol(
                protocol,
                micro_steps=micro_steps,
                base_epochs=config.base_epochs,
                reference_micro_steps=config.matched_budget_reference_micro_steps,
            )
            if instability_start_by_protocol[protocol_label] is not None:
                for seed in config.seeds:
                    run_index += 1
                    rows.append(
                        _skipped_row(
                            run_index=run_index,
                            protocol=protocol,
                            micro_steps=micro_steps,
                            epochs_used=epochs_used,
                            seed=seed,
                            instability_start_micro_steps=int(instability_start_by_protocol[protocol_label]),
                        )
                    )
                continue

            protocol_failed = False
            for seed in config.seeds:
                run_index += 1
                run_config = build_tf2_corrective_transport_default_config(
                    output_root=runs_root,
                    output_layout="run_id_subdir",
                    run_id=_run_id(protocol, micro_steps, seed),
                    run_seed=seed,
                    data_seed=seed,
                    model_init_seed=seed,
                    psi_init_seed=seed,
                    batch_order_seed=seed,
                    epochs=epochs_used,
                    batch_size=config.batch_size,
                    eval_steps=config.eval_steps,
                    layer_dims=config.layer_dims,
                    micro_steps=micro_steps,
                    feature_aware_tangents=False,
                    incremental_weight_updates=False,
                    supervision_policy="local_only",
                    theta_update_cadence="terminal_only",
                    theta_update_budget="matched",
                    checkpoint_selector="gate_constrained_accuracy_then_val_accuracy",
                )
                try:
                    result = run_fmpc_tf2_experiment(run_config)
                except Exception as error:  # pragma: no cover - instability path
                    rows.append(
                        _failure_row(
                            run_index=run_index,
                            protocol=protocol,
                            micro_steps=micro_steps,
                            epochs_used=epochs_used,
                            seed=seed,
                            error=error,
                        )
                    )
                    instability_start_by_protocol[protocol_label] = int(micro_steps)
                    protocol_failed = True
                    break
                else:
                    rows.append(
                        _success_row(
                            run_index=run_index,
                            protocol=protocol,
                            micro_steps=micro_steps,
                            epochs_used=epochs_used,
                            seed=seed,
                            result=result,
                            run_dir=run_dir,
                        )
                    )
            if protocol_failed:
                remaining_options = [value for value in config.micro_steps_options if int(value) > int(micro_steps)]
                for remaining_micro_steps in remaining_options:
                    remaining_epochs = _epochs_for_protocol(
                        protocol,
                        micro_steps=remaining_micro_steps,
                        base_epochs=config.base_epochs,
                        reference_micro_steps=config.matched_budget_reference_micro_steps,
                    )
                    for seed in config.seeds:
                        run_index += 1
                        rows.append(
                            _skipped_row(
                                run_index=run_index,
                                protocol=protocol,
                                micro_steps=remaining_micro_steps,
                                epochs_used=remaining_epochs,
                                seed=seed,
                                instability_start_micro_steps=int(micro_steps),
                            )
                        )
                break

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

    grouped_summaries = {
        (_protocol_label(protocol), int(micro_steps)): _group_summary(
            _rows_for_key(rows, protocol, micro_steps),
            slow_pc_ref=slow_pc_ref,
        )
        for protocol in config.protocols
        for micro_steps in config.micro_steps_options
    }

    mean_std_val_accuracy_by_protocol_and_micro_steps: dict[str, dict[str, Any]] = {}
    mean_std_test_accuracy_by_protocol_and_micro_steps: dict[str, dict[str, Any]] = {}
    mean_gate_passing_epoch_count_by_protocol_and_micro_steps: dict[str, dict[str, Any]] = {}
    mean_val_transported_final_energy_by_protocol_and_micro_steps: dict[str, dict[str, Any]] = {}
    mean_wall_clock_runtime_by_protocol_and_micro_steps: dict[str, dict[str, Any]] = {}
    effective_microstep_budget_by_protocol_and_micro_steps: dict[str, dict[str, Any]] = {}
    gap_to_canonical_slow_pc_by_protocol_and_micro_steps: dict[str, dict[str, Any]] = {}
    pairwise_comparison_vs_micro_steps_4_by_protocol: dict[str, dict[str, Any]] = {}

    for protocol in config.protocols:
        protocol_label = _protocol_label(protocol)
        mean_std_val_accuracy_by_protocol_and_micro_steps[protocol_label] = {}
        mean_std_test_accuracy_by_protocol_and_micro_steps[protocol_label] = {}
        mean_gate_passing_epoch_count_by_protocol_and_micro_steps[protocol_label] = {}
        mean_val_transported_final_energy_by_protocol_and_micro_steps[protocol_label] = {}
        mean_wall_clock_runtime_by_protocol_and_micro_steps[protocol_label] = {}
        effective_microstep_budget_by_protocol_and_micro_steps[protocol_label] = {}
        gap_to_canonical_slow_pc_by_protocol_and_micro_steps[protocol_label] = {}
        pairwise_comparison_vs_micro_steps_4_by_protocol[protocol_label] = {}
        reference_summary = grouped_summaries[(protocol_label, 4)]
        for micro_steps in config.micro_steps_options:
            summary = grouped_summaries[(protocol_label, int(micro_steps))]
            key = str(int(micro_steps))
            mean_std_val_accuracy_by_protocol_and_micro_steps[protocol_label][key] = {
                "mean": summary["mean_val_accuracy"],
                "std": summary["std_val_accuracy"],
            }
            mean_std_test_accuracy_by_protocol_and_micro_steps[protocol_label][key] = {
                "mean": summary["mean_test_accuracy"],
                "std": summary["std_test_accuracy"],
            }
            mean_gate_passing_epoch_count_by_protocol_and_micro_steps[protocol_label][key] = (
                summary["mean_gate_passing_epoch_count"]
            )
            mean_val_transported_final_energy_by_protocol_and_micro_steps[protocol_label][key] = (
                summary["mean_val_transported_final_energy"]
            )
            mean_wall_clock_runtime_by_protocol_and_micro_steps[protocol_label][key] = {
                "mean_total_wall_time_seconds": summary["mean_total_wall_time_seconds"],
                "std_total_wall_time_seconds": summary["std_total_wall_time_seconds"],
                "mean_train_wall_time_seconds": summary["mean_train_wall_time_seconds"],
                "std_train_wall_time_seconds": summary["std_train_wall_time_seconds"],
            }
            effective_microstep_budget_by_protocol_and_micro_steps[protocol_label][key] = {
                "epochs_used": summary["epochs_used"],
                "effective_microstep_budget": summary["effective_microstep_budget"],
            }
            gap_to_canonical_slow_pc_by_protocol_and_micro_steps[protocol_label][key] = {
                "mean_test_accuracy_gap": summary["mean_test_accuracy_gap_to_slow_pc"],
            }
            pairwise_comparison_vs_micro_steps_4_by_protocol[protocol_label][key] = _pairwise_delta(
                reference_summary,
                summary,
            )

    fixed_summaries = [
        grouped_summaries[("fixed_outer_training", int(micro_steps))]
        for micro_steps in config.micro_steps_options
        if grouped_summaries[("fixed_outer_training", int(micro_steps))]["num_completed_runs"] > 0
    ]
    matched_summaries = [
        grouped_summaries[("matched_inner_compute", int(micro_steps))]
        for micro_steps in config.micro_steps_options
        if grouped_summaries[("matched_inner_compute", int(micro_steps))]["num_completed_runs"] > 0
    ]
    best_fixed = _best_completed_summary(fixed_summaries)
    best_matched = _best_completed_summary(matched_summaries)
    matched_reference = grouped_summaries[("matched_inner_compute", 4)]
    matched_winner_is_gt4 = int(best_matched["micro_steps"]) > 4 and _material_matched_win(
        matched_reference,
        best_matched,
    )

    if matched_winner_is_gt4:
        recommended_micro_steps_default = int(best_matched["micro_steps"])
        interpretation = (
            "micro_steps > 4 retains a stable gain under matched inner compute, so the benefit is best read as a genuine transport-horizon gain rather than only extra training compute"
        )
    else:
        recommended_micro_steps_default = 4
        if int(best_fixed["micro_steps"]) > 4 and float(best_fixed["mean_test_accuracy"]) > float(
            grouped_summaries[("fixed_outer_training", 4)]["mean_test_accuracy"]
        ):
            interpretation = (
                "larger micro-step counts help under fixed outer training but do not survive matched inner compute, so the observed gain is mainly a compute-budget effect"
            )
        else:
            interpretation = (
                "micro_steps = 4 remains the best tested setting under both protocols, so there is no evidence yet that a longer transport horizon beats the current default"
            )

    next_narrow_move = (
        "adopt the matched-inner-compute winner as the new corrective default and validate it against the slow-PC baseline with a narrow multiseed pass"
        if matched_winner_is_gt4
        else "keep micro_steps = 4 and test one narrow transport-quality change inside the current corrective default, rather than adding more micro-step compute"
    )

    summary = {
        "phase": "Phase Incremental Bridge",
        "stage": "corrective_microstep_horizon_suite",
        "num_runs": int(len(rows)),
        "matching_rule": _suite_config_payload(config)["matching_rule"],
        "mean_std_val_accuracy_by_protocol_and_micro_steps": mean_std_val_accuracy_by_protocol_and_micro_steps,
        "mean_std_test_accuracy_by_protocol_and_micro_steps": mean_std_test_accuracy_by_protocol_and_micro_steps,
        "mean_gate_passing_epoch_count_by_protocol_and_micro_steps": mean_gate_passing_epoch_count_by_protocol_and_micro_steps,
        "mean_val_transported_final_energy_by_protocol_and_micro_steps": mean_val_transported_final_energy_by_protocol_and_micro_steps,
        "mean_wall_clock_runtime_by_protocol_and_micro_steps": mean_wall_clock_runtime_by_protocol_and_micro_steps,
        "effective_microstep_budget_by_protocol_and_micro_steps": effective_microstep_budget_by_protocol_and_micro_steps,
        "gap_to_canonical_slow_pc_by_protocol_and_micro_steps": gap_to_canonical_slow_pc_by_protocol_and_micro_steps,
        "instability_start_micro_steps_by_protocol": instability_start_by_protocol,
        "pairwise_comparison_vs_micro_steps_4_by_protocol": pairwise_comparison_vs_micro_steps_4_by_protocol,
        "best_configuration_by_fixed_outer_training": best_fixed,
        "best_configuration_by_matched_inner_compute": best_matched,
        "does_micro_steps_greater_than_4_win_under_matched_inner_compute": bool(matched_winner_is_gt4),
        "recommended_micro_steps_default": int(recommended_micro_steps_default),
        "transport_horizon_vs_compute_interpretation": interpretation,
        "next_single_narrow_research_move": next_narrow_move,
        "aggregate_csv_path": "aggregate_runs.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF2MicrostepHorizonSuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config),
        aggregate_rows=rows,
        summary=summary,
    )
