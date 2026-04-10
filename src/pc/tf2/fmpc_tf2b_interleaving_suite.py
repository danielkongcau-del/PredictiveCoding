from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .fmpc_tf2 import (
    FMPCTF2RunResult,
    TF2InterleavingStart,
    TF2ThetaUpdateCadence,
    build_tf2_corrective_transport_default_config,
    run_fmpc_tf2_experiment,
)


@dataclass
class FMPCTF2BInterleavingSuiteConfig:
    """Narrow TF2B rescue study around the current corrective transport default."""

    experiment_name: str = "fmpc_tf2b_interleaving_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    theta_update_cadences: tuple[TF2ThetaUpdateCadence, ...] = (
        "terminal_only",
        "every_2_micro_steps",
        "every_micro_step",
    )
    onpolicy_mix_ratios: tuple[float, ...] = (0.0, 0.25, 0.5)
    interleaving_start_options: tuple[TF2InterleavingStart, ...] = ("epoch_0", "after_warmup")
    seeds: tuple[int, ...] = (0, 1, 2)
    epochs: int = 60
    batch_size: int = 128
    eval_steps: int = 15
    layer_dims: tuple[int, ...] = (64, 64, 10)
    slow_pc_reference_runs_path: str | Path = "outputs/fmpc_tf1_external_comparison_suite/aggregate_runs.csv"
    slow_pc_reference_method_name: str = "canonical_slow_pc_digits_baseline"

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"


@dataclass
class FMPCTF2BInterleavingSuiteRunResult:
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
        raise FileNotFoundError(f"Required TF2B reference artifact is missing: {path_obj}")
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


def _ratio_tag(ratio: float) -> str:
    text = f"{float(ratio):.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _config_key(
    theta_update_cadence: str,
    onpolicy_mix_ratio: float,
    interleaving_start: str,
) -> str:
    return f"cad_{theta_update_cadence}_mix_{_ratio_tag(onpolicy_mix_ratio)}_start_{interleaving_start}"


def _run_id(
    theta_update_cadence: str,
    onpolicy_mix_ratio: float,
    interleaving_start: str,
    seed: int,
) -> str:
    return f"{_config_key(theta_update_cadence, onpolicy_mix_ratio, interleaving_start)}_seed{seed}"


def _suite_config_payload(config: FMPCTF2BInterleavingSuiteConfig) -> dict[str, Any]:
    return {
        "phase": "Phase TF2B",
        "stage": "interleaving_rescue_suite",
        "anchor_preset": "tf2_corrective_transport_default",
        "theta_update_cadences": [str(value) for value in config.theta_update_cadences],
        "onpolicy_mix_ratios": [float(value) for value in config.onpolicy_mix_ratios],
        "interleaving_start_options": [str(value) for value in config.interleaving_start_options],
        "seeds": [int(seed) for seed in config.seeds],
        "fixed": {
            "preset_name": "tf2_corrective_transport_default",
            "micro_steps": 4,
            "theta_update_budget": "matched",
            "checkpoint_selector": "gate_constrained_accuracy_then_val_accuracy",
            "family_lineage": "tf1_mlp_aug",
            "feature_aware_tangents": False,
        },
        "slow_pc_reference_runs_path": str(config.slow_pc_reference_runs_path),
        "slow_pc_reference_method_name": str(config.slow_pc_reference_method_name),
    }


def _load_slow_pc_reference_by_seed(
    rows: list[dict[str, str]],
    *,
    method_name: str,
    seeds: tuple[int, ...],
) -> dict[int, dict[str, float]]:
    filtered = [row for row in rows if str(row.get("method_name", "")) == method_name]
    by_seed = {
        int(row["seed"]): {
            "val_accuracy": float(row["val_accuracy"]),
            "test_accuracy": float(row["test_accuracy"]),
        }
        for row in filtered
    }
    missing = [seed for seed in seeds if seed not in by_seed]
    if missing:
        raise ValueError(
            f"Slow-PC reference rows for method '{method_name}' are missing seeds: {missing}."
        )
    return {int(seed): by_seed[int(seed)] for seed in seeds}


def _aggregate_row(
    run_index: int,
    result: FMPCTF2RunResult,
    *,
    seed: int,
    theta_update_cadence: str,
    onpolicy_mix_ratio: float,
    interleaving_start: str,
    run_dir: Path,
) -> dict[str, Any]:
    summary = result.summary
    return {
        "run_index": int(run_index),
        "config_key": _config_key(theta_update_cadence, onpolicy_mix_ratio, interleaving_start),
        "preset_name": str(summary["preset_name"]),
        "seed": int(seed),
        "theta_update_cadence": str(theta_update_cadence),
        "onpolicy_mix_ratio": float(onpolicy_mix_ratio),
        "interleaving_start": str(interleaving_start),
        "micro_steps": int(summary["micro_steps"]),
        "theta_update_budget": str(summary["theta_update_budget"]),
        "checkpoint_selector": str(summary["checkpoint_selector"]),
        "val_accuracy": float(summary["val_accuracy"]),
        "test_accuracy": float(summary["test_accuracy"]),
        "gate_passing_epoch_count": int(summary["gate_passing_epoch_count"]),
        "val_transported_final_energy": float(summary["val_transported_final_energy"]),
        "selected_epoch": int(summary["best_epoch"]),
        "selected_epoch_passes_gate": bool(summary["selected_epoch_passes_gate"]),
        "selector_fallback_used": bool(summary["selector_fallback_used"]),
        "run_summary_path": _relative_posix(run_dir, result.run_dir / "summary.json"),
    }


def _rows_for_key(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["config_key"]) == key]


def _config_summary(
    rows: list[dict[str, Any]],
    *,
    slow_pc_ref: dict[int, dict[str, float]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Config summary requires at least one row.")
    exemplar = rows[0]
    val_accuracies = [float(row["val_accuracy"]) for row in rows]
    test_accuracies = [float(row["test_accuracy"]) for row in rows]
    gate_counts = [float(row["gate_passing_epoch_count"]) for row in rows]
    selected_epochs = [float(row["selected_epoch"]) for row in rows]
    slow_val_accuracies = [float(slow_pc_ref[int(row["seed"])]["val_accuracy"]) for row in rows]
    slow_test_accuracies = [float(slow_pc_ref[int(row["seed"])]["test_accuracy"]) for row in rows]
    mean_val = _mean(val_accuracies)
    mean_test = _mean(test_accuracies)
    mean_slow_val = _mean(slow_val_accuracies)
    mean_slow_test = _mean(slow_test_accuracies)
    return {
        "config_key": str(exemplar["config_key"]),
        "preset_name": str(exemplar["preset_name"]),
        "theta_update_cadence": str(exemplar["theta_update_cadence"]),
        "onpolicy_mix_ratio": float(exemplar["onpolicy_mix_ratio"]),
        "interleaving_start": str(exemplar["interleaving_start"]),
        "micro_steps": int(exemplar["micro_steps"]),
        "theta_update_budget": str(exemplar["theta_update_budget"]),
        "checkpoint_selector": str(exemplar["checkpoint_selector"]),
        "num_runs": int(len(rows)),
        "mean_val_accuracy": float(mean_val),
        "std_val_accuracy": float(_std(val_accuracies)),
        "mean_test_accuracy": float(mean_test),
        "std_test_accuracy": float(_std(test_accuracies)),
        "mean_gate_passing_epoch_count": float(_mean(gate_counts)),
        "mean_selected_epoch": float(_mean(selected_epochs)),
        "mean_val_accuracy_gap_to_slow_pc": float(mean_val - mean_slow_val),
        "mean_test_accuracy_gap_to_slow_pc": float(mean_test - mean_slow_test),
    }


def _pairwise_delta(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    return {
        "mean_val_accuracy_delta": float(candidate["mean_val_accuracy"]) - float(reference["mean_val_accuracy"]),
        "mean_test_accuracy_delta": float(candidate["mean_test_accuracy"]) - float(reference["mean_test_accuracy"]),
        "mean_gate_passing_epoch_count_delta": float(candidate["mean_gate_passing_epoch_count"])
        - float(reference["mean_gate_passing_epoch_count"]),
        "mean_test_gap_to_slow_pc_delta": float(candidate["mean_test_accuracy_gap_to_slow_pc"])
        - float(reference["mean_test_accuracy_gap_to_slow_pc"]),
    }


def _matched_deltas(
    config_summaries: dict[str, dict[str, Any]],
    *,
    transform: callable,
) -> list[float]:
    deltas: list[float] = []
    for summary in config_summaries.values():
        pair = transform(summary)
        if pair is None:
            continue
        base_key, candidate_key = pair
        if base_key not in config_summaries or candidate_key not in config_summaries:
            continue
        deltas.append(
            float(config_summaries[candidate_key]["mean_test_accuracy"])
            - float(config_summaries[base_key]["mean_test_accuracy"])
        )
    return deltas


def _helps_from_deltas(deltas: list[float]) -> bool:
    return bool(
        deltas
        and _mean(deltas) > 0.0
        and sum(delta > 0.0 for delta in deltas) >= max(1, (len(deltas) // 2) + 1)
    )


def _best_config(config_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return max(
        config_summaries.values(),
        key=lambda item: (
            float(item["mean_test_accuracy"]),
            float(item["mean_val_accuracy"]),
            float(item["mean_gate_passing_epoch_count"]),
        ),
    )


def _recommended_next_step(
    anchor: dict[str, Any],
    best: dict[str, Any],
    *,
    gentle_interleaving_helps: bool,
    low_ratio_onpolicy_supervision_helps: bool,
) -> str:
    if float(best["mean_test_accuracy"]) <= float(anchor["mean_test_accuracy"]):
        return "keep corrective transport default"
    cadence_changed = str(best["theta_update_cadence"]) != "terminal_only"
    mix_changed = float(best["onpolicy_mix_ratio"]) > 0.0
    if cadence_changed and mix_changed and gentle_interleaving_helps and low_ratio_onpolicy_supervision_helps:
        return "both"
    if cadence_changed and gentle_interleaving_helps:
        return "adopt softened interleaving"
    if mix_changed and low_ratio_onpolicy_supervision_helps:
        return "adopt softened on-policy supervision"
    return "keep corrective transport default"


def run_fmpc_tf2b_interleaving_suite(
    config: FMPCTF2BInterleavingSuiteConfig,
) -> FMPCTF2BInterleavingSuiteRunResult:
    """Run the narrow TF2B interleaving-rescue study."""

    slow_pc_rows = _read_csv(config.slow_pc_reference_runs_path)
    slow_pc_ref = _load_slow_pc_reference_by_seed(
        slow_pc_rows,
        method_name=config.slow_pc_reference_method_name,
        seeds=config.seeds,
    )

    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, config.resolved_run_id(), config.output_layout)
    )
    _write_json(run_dir / "config.json", _suite_config_payload(config))

    rows: list[dict[str, Any]] = []
    run_index = 0
    runs_root = run_dir / "runs"
    for theta_update_cadence in config.theta_update_cadences:
        for onpolicy_mix_ratio in config.onpolicy_mix_ratios:
            for interleaving_start in config.interleaving_start_options:
                for seed in config.seeds:
                    run_index += 1
                    run_config = build_tf2_corrective_transport_default_config(
                        output_root=runs_root,
                        output_layout="run_id_subdir",
                        run_id=_run_id(theta_update_cadence, onpolicy_mix_ratio, interleaving_start, seed),
                        run_seed=seed,
                        data_seed=seed,
                        model_init_seed=seed,
                        psi_init_seed=seed,
                        batch_order_seed=seed,
                        epochs=config.epochs,
                        batch_size=config.batch_size,
                        eval_steps=config.eval_steps,
                        layer_dims=config.layer_dims,
                        micro_steps=4,
                        theta_update_budget="matched",
                        checkpoint_selector="gate_constrained_accuracy_then_val_accuracy",
                        incremental_weight_updates=(theta_update_cadence != "terminal_only"),
                        supervision_policy="local_only" if float(onpolicy_mix_ratio) <= 0.0 else "mixed",
                        theta_update_cadence=theta_update_cadence,
                        onpolicy_mix_ratio=float(onpolicy_mix_ratio),
                        interleaving_start=interleaving_start,
                    )
                    result = run_fmpc_tf2_experiment(run_config)
                    rows.append(
                        _aggregate_row(
                            run_index,
                            result,
                            seed=seed,
                            theta_update_cadence=theta_update_cadence,
                            onpolicy_mix_ratio=onpolicy_mix_ratio,
                            interleaving_start=interleaving_start,
                            run_dir=run_dir,
                        )
                    )

    csv_rows = [
        {
            **row,
            "selected_epoch_passes_gate": str(bool(row["selected_epoch_passes_gate"])),
            "selector_fallback_used": str(bool(row["selector_fallback_used"])),
        }
        for row in rows
    ]
    _write_csv(run_dir / "aggregate_runs.csv", csv_rows)

    config_summaries = {
        key: _config_summary(_rows_for_key(rows, key), slow_pc_ref=slow_pc_ref)
        for key in sorted({str(row["config_key"]) for row in rows})
    }
    anchor_key = _config_key("terminal_only", 0.0, "epoch_0")
    if anchor_key not in config_summaries:
        raise ValueError("The TF2B study must include the corrective transport anchor configuration.")
    anchor_summary = config_summaries[anchor_key]
    best_summary = _best_config(config_summaries)

    pairwise_vs_anchor = {
        key: _pairwise_delta(anchor_summary, value) for key, value in config_summaries.items()
    }

    gentle_interleaving_deltas: list[float] = []
    for onpolicy_mix_ratio in config.onpolicy_mix_ratios:
        for interleaving_start in config.interleaving_start_options:
            base_key = _config_key("terminal_only", onpolicy_mix_ratio, interleaving_start)
            gentle_key = _config_key("every_2_micro_steps", onpolicy_mix_ratio, interleaving_start)
            if base_key in config_summaries and gentle_key in config_summaries:
                gentle_interleaving_deltas.append(
                    float(config_summaries[gentle_key]["mean_test_accuracy"])
                    - float(config_summaries[base_key]["mean_test_accuracy"])
                )
    low_ratio_deltas: list[float] = []
    for theta_update_cadence in config.theta_update_cadences:
        for interleaving_start in config.interleaving_start_options:
            base_key = _config_key(theta_update_cadence, 0.0, interleaving_start)
            low_key = _config_key(theta_update_cadence, 0.25, interleaving_start)
            if base_key in config_summaries and low_key in config_summaries:
                low_ratio_deltas.append(
                    float(config_summaries[low_key]["mean_test_accuracy"])
                    - float(config_summaries[base_key]["mean_test_accuracy"])
                )
    delayed_interleaving_deltas: list[float] = []
    for theta_update_cadence in config.theta_update_cadences:
        for onpolicy_mix_ratio in config.onpolicy_mix_ratios:
            if theta_update_cadence == "terminal_only" and float(onpolicy_mix_ratio) == 0.0:
                continue
            epoch0_key = _config_key(theta_update_cadence, onpolicy_mix_ratio, "epoch_0")
            delayed_key = _config_key(theta_update_cadence, onpolicy_mix_ratio, "after_warmup")
            if epoch0_key in config_summaries and delayed_key in config_summaries:
                delayed_interleaving_deltas.append(
                    float(config_summaries[delayed_key]["mean_test_accuracy"])
                    - float(config_summaries[epoch0_key]["mean_test_accuracy"])
                )

    gentle_interleaving_helps = _helps_from_deltas(gentle_interleaving_deltas)
    low_ratio_onpolicy_supervision_helps = _helps_from_deltas(low_ratio_deltas)
    delayed_interleaving_helps = _helps_from_deltas(delayed_interleaving_deltas)

    any_narrows_gap = any(
        float(summary["mean_test_accuracy_gap_to_slow_pc"]) > float(anchor_summary["mean_test_accuracy_gap_to_slow_pc"])
        for summary in config_summaries.values()
    )

    summary = {
        "phase": "Phase TF2B",
        "stage": "interleaving_rescue_suite",
        "num_runs": int(len(rows)),
        "anchor_preset_name": "tf2_corrective_transport_default",
        "current_tf2_corrective_transport_default_reference": anchor_summary,
        "mean_std_val_accuracy_by_configuration": {
            key: {"mean": float(value["mean_val_accuracy"]), "std": float(value["std_val_accuracy"])}
            for key, value in config_summaries.items()
        },
        "mean_std_test_accuracy_by_configuration": {
            key: {"mean": float(value["mean_test_accuracy"]), "std": float(value["std_test_accuracy"])}
            for key, value in config_summaries.items()
        },
        "mean_gate_passing_epoch_count_by_configuration": {
            key: float(value["mean_gate_passing_epoch_count"]) for key, value in config_summaries.items()
        },
        "pairwise_comparison_against_current_tf2_corrective_transport_default": pairwise_vs_anchor,
        "gap_to_canonical_slow_pc_by_configuration": {
            key: {
                "mean_val_accuracy_gap": float(value["mean_val_accuracy_gap_to_slow_pc"]),
                "mean_test_accuracy_gap": float(value["mean_test_accuracy_gap_to_slow_pc"]),
            }
            for key, value in config_summaries.items()
        },
        "whether_gentle_interleaving_helps": bool(gentle_interleaving_helps),
        "whether_low_ratio_onpolicy_supervision_helps": bool(low_ratio_onpolicy_supervision_helps),
        "whether_delayed_interleaving_helps": bool(delayed_interleaving_helps),
        "whether_any_configuration_narrows_the_slow_pc_test_gap_below_current_tf2_default": bool(any_narrows_gap),
        "best_configuration_by_mean_test_accuracy": best_summary,
        "recommended_next_tf2_step": _recommended_next_step(
            anchor_summary,
            best_summary,
            gentle_interleaving_helps=gentle_interleaving_helps,
            low_ratio_onpolicy_supervision_helps=low_ratio_onpolicy_supervision_helps,
        ),
        "aggregate_csv_path": "aggregate_runs.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF2BInterleavingSuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config),
        aggregate_rows=rows,
        summary=summary,
    )
