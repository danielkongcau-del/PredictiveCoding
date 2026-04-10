from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .fmpc_tf2 import FMPCTF2RunResult, build_tf2_canonical_config, run_fmpc_tf2_experiment


@dataclass
class FMPCTF2SuiteConfig:
    """Configuration for the narrow TF2 bridge-validation suite."""

    experiment_name: str = "fmpc_tf2_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    incremental_weight_updates_options: tuple[bool, ...] = (False, True)
    supervision_policies: tuple[str, ...] = ("local_only", "mixed")
    micro_steps_options: tuple[int, ...] = (2, 4)
    seeds: tuple[int, ...] = (0, 1, 2)
    epochs: int = 60
    batch_size: int = 128
    eval_steps: int = 15
    layer_dims: tuple[int, ...] = (64, 64, 10)
    sealed_tf1_summary_path: str | Path = "outputs/fmpc_tf1_default_adoption_suite/aggregate_summary.json"
    slow_pc_summary_path: str | Path = "outputs/fmpc_tf1_external_comparison_suite/aggregate_summary.json"

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"


@dataclass
class FMPCTF2SuiteRunResult:
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
    return float(sum(values) / float(len(values)))


def _std(values: list[float]) -> float:
    mean_value = _mean(values)
    variance = sum((float(value) - mean_value) ** 2 for value in values) / float(len(values))
    return float(variance ** 0.5)


def _suite_config_payload(config: FMPCTF2SuiteConfig) -> dict[str, Any]:
    return {
        "phase": "FMPC Stage 04 Incremental Bridge",
        "stage": "ifmpc_bridge_stage_suite",
        "incremental_weight_updates_options": [bool(v) for v in config.incremental_weight_updates_options],
        "supervision_policies": [str(v) for v in config.supervision_policies],
        "micro_steps_options": [int(v) for v in config.micro_steps_options],
        "seeds": [int(v) for v in config.seeds],
        "fixed": {
            "family_lineage": "tf1_mlp_aug",
            "feature_aware_tangents": False,
            "identity_loss_weight": 0.2,
            "hybrid_ramp_epochs": 10,
            "bootstrap_substeps": 4,
            "checkpoint_selector": "gate_constrained_accuracy_then_val_accuracy",
            "theta_update_budget": "matched",
        },
        "sealed_tf1_summary_path": str(config.sealed_tf1_summary_path),
        "slow_pc_summary_path": str(config.slow_pc_summary_path),
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Required TF2 reference artifact is missing: {path_obj}")
    with path_obj.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sealed_tf1_reference(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "mean_val_accuracy": float(summary["mean_std_val_accuracy_by_preset"]["baseline_working_default"]["mean"]),
        "mean_test_accuracy": float(summary["mean_std_test_accuracy_by_preset"]["baseline_working_default"]["mean"]),
    }


def _slow_pc_reference(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "mean_val_accuracy": float(
            summary["mean_std_val_accuracy_by_method"]["canonical_slow_pc_digits_baseline"]["mean"]
        ),
        "mean_test_accuracy": float(
            summary["mean_std_test_accuracy_by_method"]["canonical_slow_pc_digits_baseline"]["mean"]
        ),
    }


def _candidate_run_id(incremental_weight_updates: bool, supervision_policy: str, micro_steps: int, seed: int) -> str:
    incremental_tag = "inc1" if incremental_weight_updates else "inc0"
    return f"{incremental_tag}_{supervision_policy}_m{micro_steps}_seed{seed}"


def _config_key(row: dict[str, Any]) -> str:
    incremental_tag = "inc1" if bool(row["incremental_weight_updates"]) else "inc0"
    return f"{incremental_tag}_{row['supervision_policy']}_m{int(row['micro_steps'])}"


def _aggregate_row(run_index: int, result: FMPCTF2RunResult, seed: int, run_dir: Path) -> dict[str, Any]:
    summary = result.summary
    return {
        "run_index": int(run_index),
        "config_key": _config_key(summary),
        "seed": int(seed),
        "incremental_weight_updates": bool(summary["incremental_weight_updates"]),
        "supervision_policy": str(summary["supervision_policy"]),
        "micro_steps": int(summary["micro_steps"]),
        "theta_update_budget": str(summary["theta_update_budget"]),
        "theta_micro_lr": float(summary["theta_micro_lr"]),
        "theta_micro_bias_lr": float(summary["theta_micro_bias_lr"]),
        "val_accuracy": float(summary["val_accuracy"]),
        "test_accuracy": float(summary["test_accuracy"]),
        "gate_passing_epoch_count": int(summary["gate_passing_epoch_count"]),
        "val_transported_final_energy": float(summary["val_transported_final_energy"]),
        "selected_epoch": int(summary["best_epoch"]),
        "selected_epoch_passes_gate": bool(summary["selected_epoch_passes_gate"]),
        "selector_fallback_used": bool(summary["selector_fallback_used"]),
        "forward_init_stability_metrics_json": json.dumps(summary["forward_init_stability_metrics"], sort_keys=True),
        "run_summary_path": _relative_posix(run_dir, result.run_dir / "summary.json"),
    }


def _rows_for_key(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["config_key"]) == key]


def _config_summary(rows: list[dict[str, Any]], key: str, tf1_ref: dict[str, float], slow_ref: dict[str, float]) -> dict[str, Any]:
    val_accuracies = [float(row["val_accuracy"]) for row in rows]
    test_accuracies = [float(row["test_accuracy"]) for row in rows]
    gate_counts = [float(row["gate_passing_epoch_count"]) for row in rows]
    selected_epochs = [float(row["selected_epoch"]) for row in rows]
    exemplar = rows[0]
    mean_val = _mean(val_accuracies)
    mean_test = _mean(test_accuracies)
    return {
        "config_key": key,
        "num_runs": int(len(rows)),
        "incremental_weight_updates": bool(exemplar["incremental_weight_updates"]),
        "supervision_policy": str(exemplar["supervision_policy"]),
        "micro_steps": int(exemplar["micro_steps"]),
        "theta_update_budget": str(exemplar["theta_update_budget"]),
        "theta_micro_lr": float(exemplar["theta_micro_lr"]),
        "theta_micro_bias_lr": float(exemplar["theta_micro_bias_lr"]),
        "mean_val_accuracy": mean_val,
        "std_val_accuracy": _std(val_accuracies),
        "mean_test_accuracy": mean_test,
        "std_test_accuracy": _std(test_accuracies),
        "mean_gate_passing_epoch_count": _mean(gate_counts),
        "mean_selected_epoch": _mean(selected_epochs),
        "pairwise_vs_sealed_tf1": {
            "mean_val_accuracy_delta": float(mean_val - tf1_ref["mean_val_accuracy"]),
            "mean_test_accuracy_delta": float(mean_test - tf1_ref["mean_test_accuracy"]),
        },
        "gap_to_canonical_slow_pc": {
            "mean_val_accuracy_gap": float(mean_val - slow_ref["mean_val_accuracy"]),
            "mean_test_accuracy_gap": float(mean_test - slow_ref["mean_test_accuracy"]),
        },
    }


def _best_config(config_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return max(
        config_summaries.values(),
        key=lambda item: (
            float(item["mean_test_accuracy"]),
            float(item["mean_val_accuracy"]),
            float(item["mean_gate_passing_epoch_count"]),
        ),
    )


def _matched_pair_deltas(config_summaries: dict[str, dict[str, Any]], *, compare_axis: str) -> list[float]:
    deltas: list[float] = []
    if compare_axis == "supervision_policy":
        for incremental_weight_updates in (False, True):
            for micro_steps in (2, 4):
                base_key = f"{'inc1' if incremental_weight_updates else 'inc0'}_local_only_m{micro_steps}"
                mixed_key = f"{'inc1' if incremental_weight_updates else 'inc0'}_mixed_m{micro_steps}"
                if base_key not in config_summaries or mixed_key not in config_summaries:
                    continue
                base = config_summaries[base_key]
                mixed = config_summaries[mixed_key]
                deltas.append(float(mixed["mean_test_accuracy"]) - float(base["mean_test_accuracy"]))
        return deltas
    if compare_axis == "incremental_weight_updates":
        for supervision_policy in ("local_only", "mixed"):
            for micro_steps in (2, 4):
                base_key = f"inc0_{supervision_policy}_m{micro_steps}"
                inc_key = f"inc1_{supervision_policy}_m{micro_steps}"
                if base_key not in config_summaries or inc_key not in config_summaries:
                    continue
                base = config_summaries[base_key]
                inc = config_summaries[inc_key]
                deltas.append(float(inc["mean_test_accuracy"]) - float(base["mean_test_accuracy"]))
        return deltas
    raise ValueError(f"Unsupported compare_axis '{compare_axis}'.")


def _recommended_next_stage(best_summary: dict[str, Any], tf1_ref: dict[str, float], slow_ref: dict[str, float], materially_narrowed: bool) -> str:
    if float(best_summary["mean_test_accuracy"]) <= float(tf1_ref["mean_test_accuracy"]):
        return "strengthen substrate scaling later"
    slow_gap = float(slow_ref["mean_test_accuracy"]) - float(best_summary["mean_test_accuracy"])
    if materially_narrowed and slow_gap <= 0.05:
        return "move toward generalized TF3 later"
    if materially_narrowed or float(best_summary["mean_test_accuracy"]) > float(tf1_ref["mean_test_accuracy"]):
        return "continue TF2 bridge"
    return "strengthen substrate scaling later"


def run_fmpc_tf2_suite(config: FMPCTF2SuiteConfig) -> FMPCTF2SuiteRunResult:
    """Run the narrow TF2 bridge-validation suite."""

    sealed_tf1_summary = _load_json(config.sealed_tf1_summary_path)
    slow_pc_summary = _load_json(config.slow_pc_summary_path)
    tf1_ref = _sealed_tf1_reference(sealed_tf1_summary)
    slow_ref = _slow_pc_reference(slow_pc_summary)

    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, config.resolved_run_id(), config.output_layout)
    )
    _write_json(run_dir / "config.json", _suite_config_payload(config))

    rows: list[dict[str, Any]] = []
    runs_root = run_dir / "runs"
    run_index = 0
    for incremental_weight_updates in config.incremental_weight_updates_options:
        for supervision_policy in config.supervision_policies:
            for micro_steps in config.micro_steps_options:
                for seed in config.seeds:
                    run_index += 1
                    run_config = build_tf2_canonical_config(
                        output_root=runs_root,
                        output_layout="run_id_subdir",
                        run_id=_candidate_run_id(incremental_weight_updates, supervision_policy, micro_steps, seed),
                        run_seed=seed,
                        data_seed=seed,
                        model_init_seed=seed,
                        psi_init_seed=seed,
                        batch_order_seed=seed,
                        incremental_weight_updates=incremental_weight_updates,
                        supervision_policy=supervision_policy,
                        micro_steps=micro_steps,
                        theta_update_budget="matched",
                        epochs=config.epochs,
                        batch_size=config.batch_size,
                        eval_steps=config.eval_steps,
                        layer_dims=config.layer_dims,
                    )
                    result = run_fmpc_tf2_experiment(run_config)
                    rows.append(_aggregate_row(run_index, result, seed, run_dir))

    csv_rows = [
        {
            **row,
            "incremental_weight_updates": str(bool(row["incremental_weight_updates"])),
            "selected_epoch_passes_gate": str(bool(row["selected_epoch_passes_gate"])),
            "selector_fallback_used": str(bool(row["selector_fallback_used"])),
        }
        for row in rows
    ]
    _write_csv(run_dir / "aggregate_runs.csv", csv_rows)

    config_keys = sorted({str(row["config_key"]) for row in rows})
    config_summaries = {
        key: _config_summary(_rows_for_key(rows, key), key, tf1_ref, slow_ref) for key in config_keys
    }
    best_summary = _best_config(config_summaries)

    mixed_policy_deltas = _matched_pair_deltas(config_summaries, compare_axis="supervision_policy")
    incremental_deltas = _matched_pair_deltas(config_summaries, compare_axis="incremental_weight_updates")
    mixed_policy_supervision_helps = bool(
        mixed_policy_deltas
        and _mean(mixed_policy_deltas) > 0.0
        and sum(delta > 0.0 for delta in mixed_policy_deltas) >= max(1, (len(mixed_policy_deltas) // 2) + 1)
    )
    incremental_theta_updates_help = bool(
        incremental_deltas
        and _mean(incremental_deltas) > 0.0
        and sum(delta > 0.0 for delta in incremental_deltas) >= max(1, (len(incremental_deltas) // 2) + 1)
    )

    tf1_gap = float(slow_ref["mean_test_accuracy"] - tf1_ref["mean_test_accuracy"])
    best_gap = float(slow_ref["mean_test_accuracy"] - best_summary["mean_test_accuracy"])
    gap_reduction = float(tf1_gap - best_gap)
    materially_narrowed = bool(gap_reduction >= 0.1 * tf1_gap)

    summary = {
        "phase": "FMPC Stage 04 Incremental Bridge",
        "stage": "ifmpc_bridge_stage_suite",
        "num_runs": int(len(rows)),
        "sealed_tf1_working_default_reference": tf1_ref,
        "canonical_slow_pc_digits_baseline_reference": slow_ref,
        "mean_std_val_accuracy_by_config": {
            key: {"mean": float(value["mean_val_accuracy"]), "std": float(value["std_val_accuracy"])}
            for key, value in config_summaries.items()
        },
        "mean_std_test_accuracy_by_config": {
            key: {"mean": float(value["mean_test_accuracy"]), "std": float(value["std_test_accuracy"])}
            for key, value in config_summaries.items()
        },
        "mean_gate_passing_epoch_count_by_config": {
            key: float(value["mean_gate_passing_epoch_count"]) for key, value in config_summaries.items()
        },
        "pairwise_comparison_against_sealed_tf1_working_default": {
            key: value["pairwise_vs_sealed_tf1"] for key, value in config_summaries.items()
        },
        "pairwise_gap_to_canonical_slow_pc_digits_baseline": {
            key: value["gap_to_canonical_slow_pc"] for key, value in config_summaries.items()
        },
        "by_config": config_summaries,
        "mixed_policy_supervision_helps": mixed_policy_supervision_helps,
        "incremental_theta_updates_help": incremental_theta_updates_help,
        "matched_budget_tf2_narrows_slow_pc_gap_materially": materially_narrowed,
        "best_configuration_by_mean_test_accuracy": best_summary,
        "recommended_next_stage": _recommended_next_stage(best_summary, tf1_ref, slow_ref, materially_narrowed),
        "aggregate_csv_path": "aggregate_runs.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF2SuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config),
        aggregate_rows=rows,
        summary=summary,
    )
