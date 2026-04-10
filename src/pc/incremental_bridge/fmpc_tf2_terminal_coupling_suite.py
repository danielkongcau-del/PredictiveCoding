from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..datasets import load_digits_split
from .fmpc_tf2 import (
    TF2TerminalLocalFieldDirectionIntervention,
    build_tf2_corrective_transport_terminal_angleclip_default_config,
    run_fmpc_tf2_experiment,
)
from .fmpc_tf2_endpoint_basis_suite import _delta_geometry, _interface_geometry, _rowspace_basis
from .fmpc_tf2_readout_refit_suite import _build_feature_bundle


@dataclass(frozen=True)
class _CaseSpec:
    case_name: str
    description: str
    terminal_intervention: TF2TerminalLocalFieldDirectionIntervention


@dataclass
class FMPCTF2TerminalCouplingSuiteConfig:
    """Run a narrow adopted-package terminal row-space / orthogonal coupling diagnostic."""

    experiment_name: str = "fmpc_tf2_terminal_coupling_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 60
    batch_size: int = 128
    eval_steps: int = 15
    layer_dims: tuple[int, ...] = (64, 64, 10)
    material_test_gain_threshold: float = 0.005
    max_gate_count_drop: float = 2.0
    max_selected_gate_rate_drop: float = 0.20
    max_selector_fallback_increase: float = 0.20

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"

    def case_specs(self) -> tuple[_CaseSpec, ...]:
        return (
            _CaseSpec(
                case_name="adopted_control",
                description="Current adopted full-vector terminal angle clip.",
                terminal_intervention="local_field_direction_angle_clip_keep_live_norm",
            ),
            _CaseSpec(
                case_name="rowspace_only_angle_clip",
                description="Clip only the terminal readout-row-space component; keep the orthogonal component unchanged.",
                terminal_intervention="local_field_direction_angle_clip_keep_live_norm_rowspace_only",
            ),
            _CaseSpec(
                case_name="orthogonal_only_angle_clip",
                description="Clip only the terminal orthogonal component; keep the readout-row-space component unchanged.",
                terminal_intervention="local_field_direction_angle_clip_keep_live_norm_orthogonal_only",
            ),
        )


@dataclass
class FMPCTF2TerminalCouplingSuiteRunResult:
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


def _suite_config_payload(config: FMPCTF2TerminalCouplingSuiteConfig) -> dict[str, Any]:
    return {
        "phase": "Phase Incremental Bridge",
        "stage": "adopted_package_terminal_rowspace_orthogonal_coupling",
        "base_preset_name": "tf2_corrective_transport_terminal_angleclip_default",
        "transport_family_fixed": True,
        "candidate_specs": [
            {
                "case_name": spec.case_name,
                "description": spec.description,
                "terminal_intervention": spec.terminal_intervention,
            }
            for spec in config.case_specs()
        ],
        "seeds": [int(seed) for seed in config.seeds],
        "thresholds": {
            "material_test_gain_threshold": float(config.material_test_gain_threshold),
            "max_gate_count_drop": float(config.max_gate_count_drop),
            "max_selected_gate_rate_drop": float(config.max_selected_gate_rate_drop),
            "max_selector_fallback_increase": float(config.max_selector_fallback_increase),
        },
    }


def _case_run_id(case_name: str, seed: int) -> str:
    short_name = {
        "adopted_control": "control",
        "rowspace_only_angle_clip": "rowclip",
        "orthogonal_only_angle_clip": "orthclip",
    }[case_name]
    return f"{short_name}_s{seed}"


def _runtime_proxy_seconds(summary: dict[str, Any]) -> float:
    timing = dict(summary.get("timing", {}))
    return float(timing.get("train_wall_time_seconds", 0.0)) + float(
        timing.get("final_evaluation_wall_time_seconds", 0.0)
    )


def _pairwise_delta(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]) -> dict[str, Any]:
    left_by_seed = {int(row["seed"]): row for row in left_rows}
    right_by_seed = {int(row["seed"]): row for row in right_rows}
    shared_seeds = sorted(set(left_by_seed) & set(right_by_seed))
    if not shared_seeds:
        raise ValueError("Pairwise comparison requires at least one shared seed.")

    def mean_delta(metric_name: str) -> float:
        return _mean(
            [float(left_by_seed[seed][metric_name]) - float(right_by_seed[seed][metric_name]) for seed in shared_seeds]
        )

    return {
        "shared_seeds": [int(seed) for seed in shared_seeds],
        "mean_val_accuracy_delta": mean_delta("val_accuracy"),
        "mean_test_accuracy_delta": mean_delta("test_accuracy"),
        "mean_gate_passing_epoch_count_delta": mean_delta("gate_passing_epoch_count"),
        "mean_selected_epoch_passes_gate_rate_delta": mean_delta("selected_epoch_passes_gate"),
        "mean_selector_fallback_used_rate_delta": mean_delta("selector_fallback_used"),
        "mean_val_transported_final_energy_delta": mean_delta("val_transported_final_energy"),
        "mean_val_report_output_mse_delta": mean_delta("val_report_output_mse"),
        "mean_test_report_output_mse_delta": mean_delta("test_report_output_mse"),
        "mean_val_supervised_transport_output_mse_delta": mean_delta("val_supervised_transport_output_mse"),
        "mean_test_supervised_transport_output_mse_delta": mean_delta("test_supervised_transport_output_mse"),
        "mean_val_delta_h_rms_total_delta": mean_delta("val_delta_h_rms_total"),
        "mean_val_delta_h_rms_rowspace_delta": mean_delta("val_delta_h_rms_rowspace"),
        "mean_val_delta_h_rms_orthogonal_delta": mean_delta("val_delta_h_rms_orthogonal"),
        "mean_val_delta_h_rowspace_fraction_delta": mean_delta("val_delta_h_rowspace_fraction"),
        "mean_test_delta_h_rms_total_delta": mean_delta("test_delta_h_rms_total"),
        "mean_test_delta_h_rms_rowspace_delta": mean_delta("test_delta_h_rms_rowspace"),
        "mean_test_delta_h_rms_orthogonal_delta": mean_delta("test_delta_h_rms_orthogonal"),
        "mean_test_delta_h_rowspace_fraction_delta": mean_delta("test_delta_h_rowspace_fraction"),
        "mean_runtime_proxy_seconds_delta": mean_delta("runtime_proxy_seconds"),
    }


def _case_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Case summary requires at least one row.")
    return {
        "num_runs": len(rows),
        "mean_val_accuracy": _mean([float(row["val_accuracy"]) for row in rows]),
        "std_val_accuracy": _std([float(row["val_accuracy"]) for row in rows]),
        "mean_test_accuracy": _mean([float(row["test_accuracy"]) for row in rows]),
        "std_test_accuracy": _std([float(row["test_accuracy"]) for row in rows]),
        "mean_gate_passing_epoch_count": _mean([float(row["gate_passing_epoch_count"]) for row in rows]),
        "selected_epoch_passes_gate_rate": _mean([float(row["selected_epoch_passes_gate"]) for row in rows]),
        "selector_fallback_used_rate": _mean([float(row["selector_fallback_used"]) for row in rows]),
        "mean_val_transported_final_energy": _mean([float(row["val_transported_final_energy"]) for row in rows]),
        "mean_val_report_output_mse": _mean([float(row["val_report_output_mse"]) for row in rows]),
        "std_val_report_output_mse": _std([float(row["val_report_output_mse"]) for row in rows]),
        "mean_test_report_output_mse": _mean([float(row["test_report_output_mse"]) for row in rows]),
        "std_test_report_output_mse": _std([float(row["test_report_output_mse"]) for row in rows]),
        "mean_val_supervised_transport_output_mse": _mean(
            [float(row["val_supervised_transport_output_mse"]) for row in rows]
        ),
        "std_val_supervised_transport_output_mse": _std(
            [float(row["val_supervised_transport_output_mse"]) for row in rows]
        ),
        "mean_test_supervised_transport_output_mse": _mean(
            [float(row["test_supervised_transport_output_mse"]) for row in rows]
        ),
        "std_test_supervised_transport_output_mse": _std(
            [float(row["test_supervised_transport_output_mse"]) for row in rows]
        ),
        "mean_val_delta_h_rms_total": _mean([float(row["val_delta_h_rms_total"]) for row in rows]),
        "mean_val_delta_h_rms_rowspace": _mean([float(row["val_delta_h_rms_rowspace"]) for row in rows]),
        "mean_val_delta_h_rms_orthogonal": _mean([float(row["val_delta_h_rms_orthogonal"]) for row in rows]),
        "mean_val_delta_h_rowspace_fraction": _mean([float(row["val_delta_h_rowspace_fraction"]) for row in rows]),
        "mean_test_delta_h_rms_total": _mean([float(row["test_delta_h_rms_total"]) for row in rows]),
        "mean_test_delta_h_rms_rowspace": _mean([float(row["test_delta_h_rms_rowspace"]) for row in rows]),
        "mean_test_delta_h_rms_orthogonal": _mean([float(row["test_delta_h_rms_orthogonal"]) for row in rows]),
        "mean_test_delta_h_rowspace_fraction": _mean([float(row["test_delta_h_rowspace_fraction"]) for row in rows]),
        "mean_selected_epoch": _mean([float(row["selected_epoch"]) for row in rows]),
        "mean_runtime_proxy_seconds": _mean([float(row["runtime_proxy_seconds"]) for row in rows]),
    }


def _qualifies_for_adoption(
    pairwise_vs_control: dict[str, Any],
    config: FMPCTF2TerminalCouplingSuiteConfig,
) -> bool:
    return (
        float(pairwise_vs_control["mean_test_accuracy_delta"]) >= float(config.material_test_gain_threshold)
        and float(pairwise_vs_control["mean_val_accuracy_delta"]) >= 0.0
        and float(pairwise_vs_control["mean_gate_passing_epoch_count_delta"]) >= -float(config.max_gate_count_drop)
        and float(pairwise_vs_control["mean_selected_epoch_passes_gate_rate_delta"]) >= -float(config.max_selected_gate_rate_drop)
        and float(pairwise_vs_control["mean_selector_fallback_used_rate_delta"]) <= float(config.max_selector_fallback_increase)
    )


def run_fmpc_tf2_terminal_coupling_suite(
    config: FMPCTF2TerminalCouplingSuiteConfig,
) -> FMPCTF2TerminalCouplingSuiteRunResult:
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

    for case in config.case_specs():
        for seed in config.seeds:
            tf2_config = build_tf2_corrective_transport_terminal_angleclip_default_config(
                experiment_name="tf2",
                output_root=tf2_root,
                output_layout="run_id_subdir",
                run_id=_case_run_id(case.case_name, int(seed)),
                run_seed=int(seed),
                data_seed=int(seed),
                model_init_seed=int(seed),
                psi_init_seed=int(seed),
                batch_order_seed=int(seed),
                epochs=int(config.epochs),
                batch_size=int(config.batch_size),
                eval_steps=int(config.eval_steps),
                layer_dims=tuple(config.layer_dims),
                terminal_local_field_direction_intervention=case.terminal_intervention,
            )
            result = run_fmpc_tf2_experiment(tf2_config)
            if result.model is None or result.psi_network is None:
                raise ValueError("Terminal coupling suite requires runtime model and psi network objects.")

            split = load_digits_split(
                split_seed=int(tf2_config.data_seed),
                train_fraction=float(tf2_config.train_fraction),
                val_fraction=float(tf2_config.val_fraction),
                test_fraction=float(tf2_config.test_fraction),
            )
            val_bundle = _build_feature_bundle(result.model, result.psi_network, tf2_config, split.x_val, split.y_val)
            test_bundle = _build_feature_bundle(result.model, result.psi_network, tf2_config, split.x_test, split.y_test)
            weight = result.model.layers[-1].weight
            bias = result.model.layers[-1].bias
            basis = _rowspace_basis(weight)
            transported_val = _interface_geometry(val_bundle.transported_penultimate, split.y_val, weight, bias)
            transported_test = _interface_geometry(test_bundle.transported_penultimate, split.y_test, weight, bias)
            delta_val = _delta_geometry(val_bundle.transported_penultimate, val_bundle.slow_pc_penultimate, split.y_val, basis)
            delta_test = _delta_geometry(
                test_bundle.transported_penultimate,
                test_bundle.slow_pc_penultimate,
                split.y_test,
                basis,
            )
            summary = result.summary
            aggregate_rows.append(
                {
                    "case_name": case.case_name,
                    "seed": int(seed),
                    "run_id": str(result.run_dir.name),
                    "run_summary_path": _relative_posix(run_dir, result.run_dir / "summary.json"),
                    "terminal_local_field_direction_intervention": case.terminal_intervention,
                    "selected_epoch": int(summary["best_epoch"]),
                    "val_accuracy": float(summary["val_accuracy"]),
                    "test_accuracy": float(summary["test_accuracy"]),
                    "gate_passing_epoch_count": int(summary["gate_passing_epoch_count"]),
                    "selected_epoch_passes_gate": 1.0 if bool(summary["selected_epoch_passes_gate"]) else 0.0,
                    "selector_fallback_used": 1.0 if bool(summary["selector_fallback_used"]) else 0.0,
                    "val_transported_final_energy": float(summary["val_transported_final_energy"]),
                    "val_report_output_mse": float(summary["val_loss"]),
                    "test_report_output_mse": float(summary["test_loss"]),
                    "val_supervised_transport_output_mse": float(transported_val["frozen_head_output_mse"]),
                    "test_supervised_transport_output_mse": float(transported_test["frozen_head_output_mse"]),
                    "val_delta_h_rms_total": float(delta_val["delta_h_rms_total"]),
                    "val_delta_h_rms_rowspace": float(delta_val["delta_h_rms_rowspace"]),
                    "val_delta_h_rms_orthogonal": float(delta_val["delta_h_rms_orthogonal"]),
                    "val_delta_h_rowspace_fraction": float(delta_val["delta_h_rowspace_fraction"]),
                    "test_delta_h_rms_total": float(delta_test["delta_h_rms_total"]),
                    "test_delta_h_rms_rowspace": float(delta_test["delta_h_rms_rowspace"]),
                    "test_delta_h_rms_orthogonal": float(delta_test["delta_h_rms_orthogonal"]),
                    "test_delta_h_rowspace_fraction": float(delta_test["delta_h_rowspace_fraction"]),
                    "runtime_proxy_seconds": _runtime_proxy_seconds(summary),
                }
            )

    _write_csv(run_dir / "aggregate_runs.csv", aggregate_rows)

    by_candidate = {
        case.case_name: _case_summary([row for row in aggregate_rows if str(row["case_name"]) == case.case_name])
        for case in config.case_specs()
    }
    control_rows = [row for row in aggregate_rows if str(row["case_name"]) == "adopted_control"]
    pairwise_vs_control = {
        case.case_name: _pairwise_delta([row for row in aggregate_rows if str(row["case_name"]) == case.case_name], control_rows)
        for case in config.case_specs()
        if case.case_name != "adopted_control"
    }

    adoption_candidates = [
        case.case_name
        for case in config.case_specs()
        if case.case_name != "adopted_control" and _qualifies_for_adoption(pairwise_vs_control[case.case_name], config)
    ]
    promoted_candidate_name = adoption_candidates[0] if adoption_candidates else None

    rowspace_pair = pairwise_vs_control.get("rowspace_only_angle_clip", {})
    orthogonal_pair = pairwise_vs_control.get("orthogonal_only_angle_clip", {})
    coupling_effect_supported = (
        promoted_candidate_name is None
        and float(rowspace_pair.get("mean_val_accuracy_delta", 0.0)) < 0.0
        and float(rowspace_pair.get("mean_test_accuracy_delta", 0.0)) < 0.0
        and float(orthogonal_pair.get("mean_val_accuracy_delta", 0.0)) < 0.0
        and float(orthogonal_pair.get("mean_test_accuracy_delta", 0.0)) < 0.0
    )

    summary = {
        "phase": "Phase Incremental Bridge",
        "stage": "adopted_package_terminal_rowspace_orthogonal_coupling",
        "num_runs": len(aggregate_rows),
        "by_candidate": by_candidate,
        "pairwise_vs_control": pairwise_vs_control,
        "coupling_effect_supported": coupling_effect_supported,
        "coupling_diagnosis": (
            "full_vector_gain_depends_on_coupled_rowspace_and_orthogonal_control"
            if coupling_effect_supported
            else "decomposed_terminal_control_remains_competitive"
        ),
        "should_promote_decomposed_terminal_intervention": promoted_candidate_name is not None,
        "promoted_candidate_name": promoted_candidate_name,
        "decision": (
            "promote_decomposed_terminal_intervention"
            if promoted_candidate_name is not None
            else "keep_current_adopted_default_unchanged"
        ),
        "aggregate_csv_path": "aggregate_runs.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF2TerminalCouplingSuiteRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config),
        aggregate_rows=aggregate_rows,
        summary=summary,
    )
