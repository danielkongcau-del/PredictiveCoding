from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .fmpc_tf2 import TF2PresetName, build_tf2_preset_config, run_fmpc_tf2_experiment


@dataclass
class FMPCTF2DefaultAdoptionSuiteConfig:
    """Validate adoption of the current empirical TF2 working default."""

    experiment_name: str = "fmpc_tf2_default_adoption_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    tf2_preset_names: tuple[TF2PresetName, ...] = (
        "tf2_canonical",
        "tf2_corrective_transport_default",
    )
    tf1_reference_preset_name: str = "baseline_working_default"
    slow_pc_reference_name: str = "canonical_slow_pc_digits_baseline"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 60
    batch_size: int = 128
    eval_steps: int = 15
    layer_dims: tuple[int, ...] = (64, 64, 10)
    tf1_reference_summary_path: str | Path = "outputs/fmpc_tf1_default_adoption_suite/aggregate_summary.json"
    tf1_reference_runs_path: str | Path = "outputs/fmpc_tf1_default_adoption_suite/aggregate_runs.csv"
    slow_pc_reference_summary_path: str | Path = "outputs/fmpc_tf1_external_comparison_suite/aggregate_summary.json"
    slow_pc_reference_runs_path: str | Path = "outputs/fmpc_tf1_external_comparison_suite/aggregate_runs.csv"
    jpc_probe_summary_path: str | Path = "outputs/tf2/tf2_jpc_probe/summary.json"

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"


@dataclass
class FMPCTF2DefaultAdoptionSuiteRunResult:
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


def _read_json(path: str | Path) -> dict[str, Any]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Required TF2 adoption reference artifact is missing: {path_obj}")
    with path_obj.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Required TF2 adoption reference artifact is missing: {path_obj}")
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


def _suite_config_payload(config: FMPCTF2DefaultAdoptionSuiteConfig) -> dict[str, Any]:
    return {
        "phase": "Phase TF2",
        "stage": "ifmpc_bridge_default_adoption_suite",
        "tf2_preset_names": [str(name) for name in config.tf2_preset_names],
        "tf1_reference_preset_name": str(config.tf1_reference_preset_name),
        "slow_pc_reference_name": str(config.slow_pc_reference_name),
        "seeds": [int(seed) for seed in config.seeds],
        "jpc_probe_summary_path": str(config.jpc_probe_summary_path),
        "validation_only_gating": True,
        "test_report_only": True,
        "no_selector_override": True,
    }


def _tf2_run_id(preset_name: TF2PresetName, seed: int) -> str:
    return f"{preset_name}_seed{seed}"


def _float_or_none(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def _int_or_none(value: Any) -> int | None:
    if value in (None, "", "None"):
        return None
    return int(value)


def _bool_or_none(value: Any) -> bool | None:
    if value in (None, "", "None"):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"Unsupported boolean-like value '{value}'.")


def _tf2_row(run_index: int, result: Any, preset_name: TF2PresetName, seed: int, run_dir: Path) -> dict[str, Any]:
    summary = result.summary
    return {
        "run_index": int(run_index),
        "preset_name": str(preset_name),
        "family_or_reference": "tf2",
        "seed": int(seed),
        "checkpoint_selector": str(summary["checkpoint_selector"]),
        "micro_steps": int(summary["micro_steps"]),
        "incremental_weight_updates": bool(summary["incremental_weight_updates"]),
        "supervision_policy": str(summary["supervision_policy"]),
        "theta_update_budget": str(summary["theta_update_budget"]),
        "val_accuracy": float(summary["val_accuracy"]),
        "test_accuracy": float(summary["test_accuracy"]),
        "gate_passing_epoch_count": int(summary["gate_passing_epoch_count"]),
        "val_transported_final_energy": float(summary["val_transported_final_energy"]),
        "selected_epoch": int(summary["best_epoch"]),
        "selected_epoch_passes_gate": bool(summary["selected_epoch_passes_gate"]),
        "selector_fallback_used": bool(summary["selector_fallback_used"]),
        "run_summary_path": _relative_posix(run_dir, result.run_dir / "summary.json"),
    }


def _tf1_reference_rows(
    rows: list[dict[str, str]],
    *,
    preset_name: str,
) -> list[dict[str, Any]]:
    filtered = [row for row in rows if str(row.get("preset_name", "")) == preset_name]
    if not filtered:
        raise ValueError(f"No TF1 reference rows found for preset '{preset_name}'.")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(filtered, start=1):
        normalized.append(
            {
                "run_index": int(index),
                "preset_name": str(preset_name),
                "family_or_reference": "sealed_tf1",
                "seed": int(row["seed"]),
                "checkpoint_selector": str(row["checkpoint_selector"]),
                "micro_steps": None,
                "incremental_weight_updates": None,
                "supervision_policy": None,
                "theta_update_budget": None,
                "val_accuracy": float(row["val_accuracy"]),
                "test_accuracy": float(row["test_accuracy"]),
                "gate_passing_epoch_count": int(row["gate_passing_epoch_count"]),
                "val_transported_final_energy": float(row["val_transported_final_energy"]),
                "selected_epoch": int(row["selected_epoch"]),
                "selected_epoch_passes_gate": _bool_or_none(row["selected_epoch_passes_gate"]),
                "selector_fallback_used": _bool_or_none(row["selector_fallback_used"]),
                "run_summary_path": row.get("run_summary_path", ""),
            }
        )
    return normalized


def _slow_pc_reference_rows(
    rows: list[dict[str, str]],
    *,
    method_name: str,
) -> list[dict[str, Any]]:
    filtered = [row for row in rows if str(row.get("method_name", "")) == method_name]
    if not filtered:
        raise ValueError(f"No slow-PC reference rows found for method '{method_name}'.")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(filtered, start=1):
        normalized.append(
            {
                "run_index": int(index),
                "preset_name": str(method_name),
                "family_or_reference": "slow_pc_reference",
                "seed": int(row["seed"]),
                "checkpoint_selector": str(row.get("checkpoint_selector", "")),
                "micro_steps": None,
                "incremental_weight_updates": None,
                "supervision_policy": None,
                "theta_update_budget": None,
                "val_accuracy": float(row["val_accuracy"]),
                "test_accuracy": float(row["test_accuracy"]),
                "gate_passing_epoch_count": None,
                "val_transported_final_energy": _float_or_none(row.get("val_transported_final_energy")),
                "selected_epoch": None,
                "selected_epoch_passes_gate": None,
                "selector_fallback_used": None,
                "run_summary_path": row.get("run_summary_path", ""),
            }
        )
    return normalized


def _rows_for_preset(rows: list[dict[str, Any]], preset_name: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["preset_name"]) == preset_name]


def _preset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Preset summary requires at least one row.")
    val_accuracies = [float(row["val_accuracy"]) for row in rows]
    test_accuracies = [float(row["test_accuracy"]) for row in rows]
    gate_counts = [
        float(row["gate_passing_epoch_count"])
        for row in rows
        if row["gate_passing_epoch_count"] not in (None, "")
    ]
    selected_epochs = [
        float(row["selected_epoch"])
        for row in rows
        if row["selected_epoch"] not in (None, "")
    ]
    selectors = sorted({str(row["checkpoint_selector"]) for row in rows if str(row["checkpoint_selector"])})
    return {
        "num_runs": int(len(rows)),
        "checkpoint_selectors": selectors,
        "mean_val_accuracy": _mean(val_accuracies),
        "std_val_accuracy": _std(val_accuracies),
        "mean_test_accuracy": _mean(test_accuracies),
        "std_test_accuracy": _std(test_accuracies),
        "mean_gate_passing_epoch_count": None if not gate_counts else _mean(gate_counts),
        "mean_selected_epoch": None if not selected_epochs else _mean(selected_epochs),
    }


def _pairwise_delta(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    return {
        "mean_val_accuracy_delta": float(candidate["mean_val_accuracy"]) - float(reference["mean_val_accuracy"]),
        "mean_test_accuracy_delta": float(candidate["mean_test_accuracy"]) - float(reference["mean_test_accuracy"]),
    }


def _interpretation_label(
    canonical_summary: dict[str, Any],
    corrective_summary: dict[str, Any],
    tf1_summary: dict[str, Any],
) -> str:
    corrective_test = float(corrective_summary["mean_test_accuracy"])
    canonical_test = float(canonical_summary["mean_test_accuracy"])
    tf1_test = float(tf1_summary["mean_test_accuracy"])
    tolerance = 0.01
    if corrective_test > tf1_test and corrective_test > canonical_test + tolerance:
        return "corrective transport bridge"
    if canonical_test > tf1_test and canonical_test > corrective_test + tolerance:
        return "full incremental iFMPC"
    if corrective_test > tf1_test and canonical_test > tf1_test:
        return "both"
    return "neither"


def run_fmpc_tf2_default_adoption_suite(
    config: FMPCTF2DefaultAdoptionSuiteConfig,
) -> FMPCTF2DefaultAdoptionSuiteRunResult:
    """Run a narrow TF2 default-adoption validation pass."""

    tf1_reference_summary = _read_json(config.tf1_reference_summary_path)
    slow_pc_reference_summary = _read_json(config.slow_pc_reference_summary_path)
    jpc_probe_summary = _read_json(config.jpc_probe_summary_path)
    tf1_reference_rows = _read_csv(config.tf1_reference_runs_path)
    slow_pc_reference_rows = _read_csv(config.slow_pc_reference_runs_path)

    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, config.resolved_run_id(), config.output_layout)
    )
    _write_json(run_dir / "config.json", _suite_config_payload(config))

    rows: list[dict[str, Any]] = []
    runs_root = run_dir / "runs"
    run_index = 0
    for preset_name in config.tf2_preset_names:
        for seed in config.seeds:
            run_index += 1
            tf2_config = build_tf2_preset_config(
                preset_name,
                output_root=runs_root,
                output_layout="run_id_subdir",
                run_id=_tf2_run_id(preset_name, seed),
                run_seed=seed,
                data_seed=seed,
                model_init_seed=seed,
                psi_init_seed=seed,
                batch_order_seed=seed,
                epochs=config.epochs,
                batch_size=config.batch_size,
                eval_steps=config.eval_steps,
                layer_dims=config.layer_dims,
            )
            result = run_fmpc_tf2_experiment(tf2_config)
            rows.append(_tf2_row(run_index, result, preset_name, seed, run_dir))

    tf1_rows = _tf1_reference_rows(tf1_reference_rows, preset_name=config.tf1_reference_preset_name)
    slow_rows = _slow_pc_reference_rows(slow_pc_reference_rows, method_name=config.slow_pc_reference_name)

    next_index = len(rows) + 1
    for offset, row in enumerate(tf1_rows):
        row["run_index"] = int(next_index + offset)
    next_index += len(tf1_rows)
    for offset, row in enumerate(slow_rows):
        row["run_index"] = int(next_index + offset)

    rows.extend(tf1_rows)
    rows.extend(slow_rows)

    csv_rows = []
    for row in rows:
        csv_rows.append(
            {
                **row,
                "incremental_weight_updates": "" if row["incremental_weight_updates"] is None else str(bool(row["incremental_weight_updates"])),
                "selected_epoch_passes_gate": "" if row["selected_epoch_passes_gate"] is None else str(bool(row["selected_epoch_passes_gate"])),
                "selector_fallback_used": "" if row["selector_fallback_used"] is None else str(bool(row["selector_fallback_used"])),
                "micro_steps": "" if row["micro_steps"] is None else str(int(row["micro_steps"])),
                "gate_passing_epoch_count": "" if row["gate_passing_epoch_count"] is None else str(int(row["gate_passing_epoch_count"])),
                "val_transported_final_energy": "" if row["val_transported_final_energy"] is None else str(float(row["val_transported_final_energy"])),
                "selected_epoch": "" if row["selected_epoch"] is None else str(int(row["selected_epoch"])),
            }
        )
    _write_csv(run_dir / "aggregate_runs.csv", csv_rows)

    preset_names = [
        *config.tf2_preset_names,
        config.tf1_reference_preset_name,
        config.slow_pc_reference_name,
    ]
    by_preset = {preset_name: _preset_summary(_rows_for_preset(rows, preset_name)) for preset_name in preset_names}

    canonical_summary = by_preset["tf2_canonical"]
    corrective_summary = by_preset["tf2_corrective_transport_default"]
    tf1_summary = by_preset[config.tf1_reference_preset_name]
    slow_summary = by_preset[config.slow_pc_reference_name]

    corrective_vs_canonical = _pairwise_delta(canonical_summary, corrective_summary)
    corrective_vs_tf1 = _pairwise_delta(tf1_summary, corrective_summary)
    corrective_vs_slow = {
        "mean_val_accuracy_gap": float(corrective_summary["mean_val_accuracy"]) - float(slow_summary["mean_val_accuracy"]),
        "mean_test_accuracy_gap": float(corrective_summary["mean_test_accuracy"]) - float(slow_summary["mean_test_accuracy"]),
    }
    corrective_becomes_main = bool(
        float(corrective_summary["mean_test_accuracy"]) > float(canonical_summary["mean_test_accuracy"])
        and float(corrective_summary["mean_val_accuracy"]) >= float(canonical_summary["mean_val_accuracy"])
    )
    interpretation = _interpretation_label(canonical_summary, corrective_summary, tf1_summary)
    jpc_supports_incremental_priority = bool(
        jpc_probe_summary.get("recommended_tf2_emphasis") == "incremental scheduling"
    )

    summary = {
        "phase": "Phase TF2",
        "stage": "ifmpc_bridge_default_adoption_suite",
        "num_runs": int(len(rows)),
        "mean_std_val_accuracy_by_preset": {
            preset_name: {
                "mean": float(preset_summary["mean_val_accuracy"]),
                "std": float(preset_summary["std_val_accuracy"]),
            }
            for preset_name, preset_summary in by_preset.items()
        },
        "mean_std_test_accuracy_by_preset": {
            preset_name: {
                "mean": float(preset_summary["mean_test_accuracy"]),
                "std": float(preset_summary["std_test_accuracy"]),
            }
            for preset_name, preset_summary in by_preset.items()
        },
        "mean_gate_passing_epoch_count_by_preset": {
            preset_name: preset_summary["mean_gate_passing_epoch_count"]
            for preset_name, preset_summary in by_preset.items()
        },
        "by_preset": by_preset,
        "pairwise_tf2_corrective_transport_default_vs_tf2_canonical": corrective_vs_canonical,
        "pairwise_tf2_corrective_transport_default_vs_sealed_tf1_working_default": corrective_vs_tf1,
        "pairwise_tf2_corrective_transport_default_vs_canonical_slow_pc_digits_baseline": corrective_vs_slow,
        "tf2_corrective_transport_default_should_become_main_tf2_preset": corrective_becomes_main,
        "current_tf2_evidence_interpretation": interpretation,
        "jpc_probe_reference": {
            "summary_path": str(config.jpc_probe_summary_path),
            "recommended_tf2_emphasis": jpc_probe_summary.get("recommended_tf2_emphasis"),
            "whether_mupc_like_scaling_appears_to_improve_forward_init_stability": jpc_probe_summary.get(
                "whether_mupc_like_scaling_appears_to_improve_forward_init_stability"
            ),
            "whether_many_step_inference_still_materially_outperforms_1_step_in_jpc_standard_pc": jpc_probe_summary.get(
                "whether_many_step_inference_still_materially_outperforms_1_step_in_jpc_standard_pc"
            ),
        },
        "jpc_probe_supports_prioritizing_incremental_scheduling": jpc_supports_incremental_priority,
        "recommended_main_tf2_preset_after_adoption_validation": {
            "preset_name": "tf2_corrective_transport_default"
            if corrective_becomes_main
            else "tf2_canonical",
            "checkpoint_selector": "gate_constrained_accuracy_then_val_accuracy",
            "mechanistic_interpretation": interpretation,
        },
        "aggregate_csv_path": "aggregate_runs.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)

    return FMPCTF2DefaultAdoptionSuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config),
        aggregate_rows=rows,
        summary=summary,
    )
