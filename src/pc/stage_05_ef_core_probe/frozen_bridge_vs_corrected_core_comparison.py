from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..datasets import load_digits_split
from ..metrics import classification_accuracy
from ..stage_03_transport_core_v1.fmpc_tf1_flow import (
    build_tf1_context,
    hidden_energy_from_state,
    hidden_local_flow,
    rollout_hidden_transport,
)
from ..stage_04_incremental_bridge.fmpc_tf2 import (
    FMPCTF2Config,
    build_tf2_corrective_transport_terminal_angleclip_default_config,
    run_fmpc_tf2_experiment,
)
from .fmpc_ef_exploratory_probe import (
    FMPCEFExploratoryProbeConfig,
    ProbeMechanismMetrics,
    build_fmpc_ef_exploratory_probe_config,
    run_fmpc_ef_exploratory_probe,
)


ComparisonMethodName = Literal[
    "stage_04_frozen_bridge",
    "stage_05_corrected_residual_core",
]
OutputLayout = Literal["single_dir", "run_id_subdir"]

STAGE04_METHOD_NAME: ComparisonMethodName = "stage_04_frozen_bridge"
STAGE05_METHOD_NAME: ComparisonMethodName = "stage_05_corrected_residual_core"
JUSTIFY_V2_DECISION_NAME = "stage05_corrected_residual_core_justifies_v2_charter"


@dataclass
class FrozenBridgeVsCorrectedCoreComparisonConfig:
    """Compare the frozen Stage 04 bridge against the corrected Stage 05 core."""

    experiment_name: str = "frozen_bridge_vs_corrected_core_comparison"
    output_root: str | Path = "outputs/stage_05_ef_core_probe"
    run_id: str | None = None
    output_layout: OutputLayout = "single_dir"
    dataset_name: str = "digits"
    seeds: tuple[int, ...] = (0, 1, 2)
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    batch_size: int = 128
    shuffle_batches: bool = True
    stage04_epochs: int = 60
    stage04_eval_steps: int = 15
    stage04_layer_dims: tuple[int, ...] = (64, 64, 10)
    stage05_epochs: int = 12
    stage05_eval_steps: int = 15
    stage05_layer_dims: tuple[int, ...] = (64, 16, 10)

    def __post_init__(self) -> None:
        if self.dataset_name != "digits":
            raise ValueError("The frozen-bridge vs corrected-core comparison currently supports digits only.")
        if not self.seeds:
            raise ValueError("seeds must contain at least one seed.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.stage04_epochs <= 0 or self.stage05_epochs <= 0:
            raise ValueError("stage04_epochs and stage05_epochs must be positive.")
        if self.stage04_eval_steps <= 0 or self.stage05_eval_steps <= 0:
            raise ValueError("stage04_eval_steps and stage05_eval_steps must be positive.")

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_{self.seeds[0]}"


@dataclass
class FrozenBridgeVsCorrectedCoreComparisonRunResult:
    run_dir: Path
    config: dict[str, Any]
    aggregate_rows: list[dict[str, Any]]
    summary: dict[str, Any]
    comparison_report: dict[str, Any]


def _resolve_run_dir(
    output_root: str | Path,
    experiment_name: str,
    run_id: str,
    output_layout: OutputLayout,
) -> Path:
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


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


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


def _hidden_residual_rms(context: Any, z: np.ndarray) -> float:
    flow = hidden_local_flow(context, z)
    return float(np.sqrt(np.mean(flow * flow)))


def _stage04_mechanism_metrics(
    model: Any,
    psi_network: Any,
    config: FMPCTF2Config,
    x_split: np.ndarray,
    y_split: np.ndarray,
    *,
    transport_steps: int,
) -> ProbeMechanismMetrics:
    from ..stage_04_incremental_bridge.fmpc_tf2 import _learned_velocity_fn as _tf2_learned_velocity_fn

    context = build_tf1_context(model, x_split, y_split)
    initial_energy = hidden_energy_from_state(context, context.z0)
    initial_residual_rms = _hidden_residual_rms(context, context.z0)
    identity = rollout_hidden_transport(
        context,
        context.z0,
        transport_steps=transport_steps,
        mode="identity",
    )
    local_field = rollout_hidden_transport(
        context,
        context.z0,
        transport_steps=transport_steps,
        mode="local_field_only",
    )
    learned = rollout_hidden_transport(
        context,
        context.z0,
        transport_steps=transport_steps,
        mode="learned",
        velocity_fn=_tf2_learned_velocity_fn(context, psi_network, config),
    )
    identity_residual = _hidden_residual_rms(context, identity.z_knots[-1])
    local_field_residual = _hidden_residual_rms(context, local_field.z_knots[-1])
    learned_residual = _hidden_residual_rms(context, learned.z_knots[-1])
    return ProbeMechanismMetrics(
        transport_steps=int(transport_steps),
        initial_energy=float(initial_energy),
        identity_final_energy=float(identity.final_energy),
        local_field_only_final_energy=float(local_field.final_energy),
        transported_final_energy=float(learned.final_energy),
        energy_delta_vs_identity=float(learned.final_energy - identity.final_energy),
        energy_delta_vs_local_field_only=float(learned.final_energy - local_field.final_energy),
        initial_fixed_point_residual_rms=float(initial_residual_rms),
        identity_final_fixed_point_residual_rms=float(identity_residual),
        local_field_only_final_fixed_point_residual_rms=float(local_field_residual),
        transported_final_fixed_point_residual_rms=float(learned_residual),
        fixed_point_residual_delta_vs_identity=float(learned_residual - identity_residual),
        fixed_point_residual_delta_vs_local_field_only=float(learned_residual - local_field_residual),
    )


def _slow_pc_metrics(model: Any, x_split: np.ndarray, y_split: np.ndarray) -> tuple[float, float]:
    predictions = model.predict(x_split)
    output_mse = float(np.mean((predictions - y_split) ** 2))
    accuracy = classification_accuracy(predictions, y_split)
    return output_mse, accuracy


def _stage04_config(
    config: FrozenBridgeVsCorrectedCoreComparisonConfig,
    *,
    seed: int,
    output_root: Path,
) -> FMPCTF2Config:
    return build_tf2_corrective_transport_terminal_angleclip_default_config(
        output_root=output_root,
        experiment_name=STAGE04_METHOD_NAME,
        output_layout="run_id_subdir",
        run_id=f"seed_{seed}",
        run_seed=seed,
        data_seed=seed,
        model_init_seed=seed,
        psi_init_seed=seed,
        batch_order_seed=seed,
        train_fraction=float(config.train_fraction),
        val_fraction=float(config.val_fraction),
        test_fraction=float(config.test_fraction),
        batch_size=int(config.batch_size),
        shuffle_batches=bool(config.shuffle_batches),
        epochs=int(config.stage04_epochs),
        eval_steps=int(config.stage04_eval_steps),
        layer_dims=config.stage04_layer_dims,
    )


def _stage05_config(
    config: FrozenBridgeVsCorrectedCoreComparisonConfig,
    *,
    seed: int,
    output_root: Path,
) -> FMPCEFExploratoryProbeConfig:
    return build_fmpc_ef_exploratory_probe_config(
        output_root=output_root,
        experiment_name=STAGE05_METHOD_NAME,
        output_layout="run_id_subdir",
        run_id=f"seed_{seed}",
        run_seed=seed,
        data_seed=seed,
        model_init_seed=seed,
        psi_init_seed=seed,
        batch_order_seed=seed,
        train_fraction=float(config.train_fraction),
        val_fraction=float(config.val_fraction),
        test_fraction=float(config.test_fraction),
        batch_size=int(config.batch_size),
        shuffle_batches=bool(config.shuffle_batches),
        epochs=int(config.stage05_epochs),
        eval_steps=int(config.stage05_eval_steps),
        layer_dims=config.stage05_layer_dims,
    )


def _artifact_checks(
    run_dir: Path,
    *,
    seed: int,
    expected_dataset_name: str,
    expected_batch_size: int,
    expected_shuffle_batches: bool,
) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    summary_path = run_dir / "summary.json"
    epoch_metrics_path = run_dir / "epoch_metrics.csv"
    checks = {
        "config_json_exists": bool(config_path.exists()),
        "summary_json_exists": bool(summary_path.exists()),
        "epoch_metrics_csv_exists": bool(epoch_metrics_path.exists()),
        "seed_matches": False,
        "dataset_matches": False,
        "batch_protocol_matches": False,
    }
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        dataset_payload = payload.get("dataset", {})
        run_payload = payload.get("run", {})
        checks["seed_matches"] = (
            int(run_payload.get("run_seed", -1)) == int(seed)
            and int(dataset_payload.get("data_seed", -1)) == int(seed)
        )
        checks["dataset_matches"] = str(dataset_payload.get("dataset_name", "")) == str(expected_dataset_name)
        checks["batch_protocol_matches"] = (
            int(run_payload.get("batch_size", -1)) == int(expected_batch_size)
            and bool(run_payload.get("shuffle_batches", False)) == bool(expected_shuffle_batches)
        )
    passed = all(bool(value) for value in checks.values())
    return {
        **checks,
        "deterministic_artifact_checks_passed": bool(passed),
    }


def _stage04_row(
    *,
    run_index: int,
    suite_run_dir: Path,
    seed: int,
    result: Any,
    config: FMPCTF2Config,
) -> dict[str, Any]:
    split = load_digits_split(
        split_seed=config.data_seed,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
    )
    val_one_step = _stage04_mechanism_metrics(
        result.model,
        result.psi_network,
        config,
        split.x_val,
        split.y_val,
        transport_steps=1,
    )
    val_configured = _stage04_mechanism_metrics(
        result.model,
        result.psi_network,
        config,
        split.x_val,
        split.y_val,
        transport_steps=config.micro_steps,
    )
    val_output_mse, val_accuracy = _slow_pc_metrics(result.model, split.x_val, split.y_val)
    test_output_mse, test_accuracy = _slow_pc_metrics(result.model, split.x_test, split.y_test)
    artifact_checks = _artifact_checks(
        result.run_dir,
        seed=seed,
        expected_dataset_name=config.dataset_name,
        expected_batch_size=config.batch_size,
        expected_shuffle_batches=config.shuffle_batches,
    )
    timing = dict(result.summary.get("timing", {}))
    runtime_proxy = float(timing.get("train_wall_time_seconds", 0.0)) + float(
        timing.get("final_evaluation_wall_time_seconds", 0.0)
    )
    return {
        "run_index": int(run_index),
        "method_name": STAGE04_METHOD_NAME,
        "stage_name": "FMPC Stage 04 Incremental Bridge",
        "seed": int(seed),
        "run_id": str(result.run_dir.name),
        "run_config_path": _relative_posix(suite_run_dir, result.run_dir / "config.json"),
        "run_summary_path": _relative_posix(suite_run_dir, result.run_dir / "summary.json"),
        "configured_transport_steps": int(config.micro_steps),
        "one_step_energy_delta_vs_identity": float(val_one_step.energy_delta_vs_identity),
        "configured_step_energy_delta_vs_identity": float(val_configured.energy_delta_vs_identity),
        "configured_step_fixed_point_residual_delta_vs_identity": float(
            val_configured.fixed_point_residual_delta_vs_identity
        ),
        "one_step_energy_delta_vs_local_field_only": float(val_one_step.energy_delta_vs_local_field_only),
        "configured_step_energy_delta_vs_local_field_only": float(
            val_configured.energy_delta_vs_local_field_only
        ),
        "configured_step_fixed_point_residual_delta_vs_local_field_only": float(
            val_configured.fixed_point_residual_delta_vs_local_field_only
        ),
        "val_accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy),
        "val_output_mse": float(val_output_mse),
        "test_output_mse": float(test_output_mse),
        "runtime_proxy_seconds": float(runtime_proxy),
        "acceptance_contract": "mechanism_first_comparison_context",
        "mechanism_signal_positive": bool(
            float(val_one_step.energy_delta_vs_identity) < 0.0
            and float(val_configured.energy_delta_vs_identity) < 0.0
            and float(val_configured.fixed_point_residual_delta_vs_identity) < 0.0
        ),
        **artifact_checks,
    }


def _stage05_row(
    *,
    run_index: int,
    suite_run_dir: Path,
    seed: int,
    result: Any,
    config: FMPCEFExploratoryProbeConfig,
) -> dict[str, Any]:
    summary = result.summary
    val_one_step = summary["mechanism_metrics"]["one_step"]
    val_configured = summary["mechanism_metrics"]["configured_steps"]
    artifact_checks = _artifact_checks(
        result.run_dir,
        seed=seed,
        expected_dataset_name=config.dataset_name,
        expected_batch_size=config.batch_size,
        expected_shuffle_batches=config.shuffle_batches,
    )
    runtime_proxy = float(summary.get("train_wall_time_seconds", 0.0)) + float(
        summary.get("evaluation_wall_time_seconds", 0.0)
    )
    return {
        "run_index": int(run_index),
        "method_name": STAGE05_METHOD_NAME,
        "stage_name": "FMPC Stage 05 EF Core Probe",
        "seed": int(seed),
        "run_id": str(result.run_dir.name),
        "run_config_path": _relative_posix(suite_run_dir, result.run_dir / "config.json"),
        "run_summary_path": _relative_posix(suite_run_dir, result.run_dir / "summary.json"),
        "configured_transport_steps": int(config.transport_steps),
        "one_step_energy_delta_vs_identity": float(val_one_step["energy_delta_vs_identity"]),
        "configured_step_energy_delta_vs_identity": float(val_configured["energy_delta_vs_identity"]),
        "configured_step_fixed_point_residual_delta_vs_identity": float(
            val_configured["fixed_point_residual_delta_vs_identity"]
        ),
        "one_step_energy_delta_vs_local_field_only": float(
            val_one_step["energy_delta_vs_local_field_only"]
        ),
        "configured_step_energy_delta_vs_local_field_only": float(
            val_configured["energy_delta_vs_local_field_only"]
        ),
        "configured_step_fixed_point_residual_delta_vs_local_field_only": float(
            val_configured["fixed_point_residual_delta_vs_local_field_only"]
        ),
        "val_accuracy": float(summary["val_accuracy"]),
        "test_accuracy": float(summary["test_accuracy"]),
        "val_output_mse": float(summary["val_output_mse"]),
        "test_output_mse": float(summary["test_output_mse"]),
        "runtime_proxy_seconds": float(runtime_proxy),
        "acceptance_contract": str(summary["acceptance_contract"]),
        "mechanism_signal_positive": bool(
            float(val_one_step["energy_delta_vs_identity"]) < 0.0
            and float(val_configured["energy_delta_vs_identity"]) < 0.0
            and float(val_configured["fixed_point_residual_delta_vs_identity"]) < 0.0
        ),
        **artifact_checks,
    }


def _method_rows(rows: list[dict[str, Any]], method_name: ComparisonMethodName) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["method_name"]) == str(method_name)]


def _method_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Method summary requires at least one row.")

    def _metric_summary(field: str) -> dict[str, float]:
        values = [float(row[field]) for row in rows]
        return {"mean": _mean(values), "std": _std(values)}

    artifact_checks = [bool(row["deterministic_artifact_checks_passed"]) for row in rows]
    mechanism_checks = [bool(row["mechanism_signal_positive"]) for row in rows]
    return {
        "num_runs": int(len(rows)),
        "configured_transport_steps": int(rows[0]["configured_transport_steps"]),
        "deterministic_artifact_check_rate": _rate(artifact_checks),
        "mechanism_signal_positive_rate": _rate(mechanism_checks),
        "one_step_energy_delta_vs_identity": _metric_summary("one_step_energy_delta_vs_identity"),
        "configured_step_energy_delta_vs_identity": _metric_summary(
            "configured_step_energy_delta_vs_identity"
        ),
        "configured_step_fixed_point_residual_delta_vs_identity": _metric_summary(
            "configured_step_fixed_point_residual_delta_vs_identity"
        ),
        "one_step_energy_delta_vs_local_field_only": _metric_summary(
            "one_step_energy_delta_vs_local_field_only"
        ),
        "configured_step_energy_delta_vs_local_field_only": _metric_summary(
            "configured_step_energy_delta_vs_local_field_only"
        ),
        "configured_step_fixed_point_residual_delta_vs_local_field_only": _metric_summary(
            "configured_step_fixed_point_residual_delta_vs_local_field_only"
        ),
        "val_accuracy": _metric_summary("val_accuracy"),
        "test_accuracy": _metric_summary("test_accuracy"),
        "val_output_mse": _metric_summary("val_output_mse"),
        "test_output_mse": _metric_summary("test_output_mse"),
        "runtime_proxy_seconds": _metric_summary("runtime_proxy_seconds"),
    }


def _pairwise_summary(
    rows: list[dict[str, Any]],
    *,
    candidate_method: ComparisonMethodName,
    reference_method: ComparisonMethodName,
) -> dict[str, Any]:
    candidate_by_seed = {int(row["seed"]): row for row in _method_rows(rows, candidate_method)}
    reference_by_seed = {int(row["seed"]): row for row in _method_rows(rows, reference_method)}
    shared_seeds = sorted(set(candidate_by_seed).intersection(reference_by_seed))
    if not shared_seeds:
        raise ValueError("Pairwise summary requires at least one shared seed.")

    def _delta(field: str) -> dict[str, float]:
        values = [
            float(candidate_by_seed[seed][field]) - float(reference_by_seed[seed][field])
            for seed in shared_seeds
        ]
        return {"mean": _mean(values), "std": _std(values)}

    return {
        "shared_seeds": [int(seed) for seed in shared_seeds],
        "one_step_energy_delta_vs_identity_delta": _delta("one_step_energy_delta_vs_identity"),
        "configured_step_energy_delta_vs_identity_delta": _delta(
            "configured_step_energy_delta_vs_identity"
        ),
        "configured_step_fixed_point_residual_delta_vs_identity_delta": _delta(
            "configured_step_fixed_point_residual_delta_vs_identity"
        ),
        "configured_step_energy_delta_vs_local_field_only_delta": _delta(
            "configured_step_energy_delta_vs_local_field_only"
        ),
        "val_accuracy_delta": _delta("val_accuracy"),
        "test_accuracy_delta": _delta("test_accuracy"),
        "runtime_proxy_seconds_delta": _delta("runtime_proxy_seconds"),
    }


def _comparison_protocol_payload(config: FrozenBridgeVsCorrectedCoreComparisonConfig) -> dict[str, Any]:
    return {
        "dataset_name": config.dataset_name,
        "seeds": [int(seed) for seed in config.seeds],
        "train_fraction": float(config.train_fraction),
        "val_fraction": float(config.val_fraction),
        "test_fraction": float(config.test_fraction),
        "shared_batch_size": int(config.batch_size),
        "shared_shuffle_batches": bool(config.shuffle_batches),
        "stage_04_control": {
            "method_name": STAGE04_METHOD_NAME,
            "preset_name": "tf2_corrective_transport_terminal_angleclip_default",
            "configured_transport_steps": 4,
            "epochs": int(config.stage04_epochs),
            "eval_steps": int(config.stage04_eval_steps),
            "layer_dims": [int(value) for value in config.stage04_layer_dims],
        },
        "stage_05_candidate": {
            "method_name": STAGE05_METHOD_NAME,
            "transport_family": "residual_meanflow_core",
            "residual_identity_mode": "residual_corrected_meanflow",
            "configured_transport_steps": 2,
            "epochs": int(config.stage05_epochs),
            "eval_steps": int(config.stage05_eval_steps),
            "layer_dims": [int(value) for value in config.stage05_layer_dims],
        },
        "decision_rule": {
            "primary_split": "validation",
            "requires_all_stage05_artifact_checks": True,
            "requires_stage05_one_step_energy_delta_vs_identity_negative_on_every_seed": True,
            "requires_stage05_configured_step_energy_delta_vs_identity_negative_on_every_seed": True,
            "requires_stage05_configured_step_fixed_point_residual_delta_vs_identity_negative_on_every_seed": True,
            "task_accuracy_is_report_only": True,
        },
    }


def _stage05_v2_charter_decision(rows: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    stage05_rows = _method_rows(rows, STAGE05_METHOD_NAME)
    artifact_pass = all(bool(row["deterministic_artifact_checks_passed"]) for row in stage05_rows)
    one_step_pass = all(float(row["one_step_energy_delta_vs_identity"]) < 0.0 for row in stage05_rows)
    configured_energy_pass = all(
        float(row["configured_step_energy_delta_vs_identity"]) < 0.0 for row in stage05_rows
    )
    configured_residual_pass = all(
        float(row["configured_step_fixed_point_residual_delta_vs_identity"]) < 0.0
        for row in stage05_rows
    )
    justified = bool(artifact_pass and one_step_pass and configured_energy_pass and configured_residual_pass)
    return justified, {
        "artifact_checks_all_pass": bool(artifact_pass),
        "one_step_energy_delta_vs_identity_negative_on_every_seed": bool(one_step_pass),
        "configured_step_energy_delta_vs_identity_negative_on_every_seed": bool(
            configured_energy_pass
        ),
        "configured_step_fixed_point_residual_delta_vs_identity_negative_on_every_seed": bool(
            configured_residual_pass
        ),
    }


def _supports_lines(
    *,
    justified: bool,
    decision_detail: dict[str, Any],
    by_method: dict[str, dict[str, Any]],
) -> list[str]:
    stage05 = by_method[STAGE05_METHOD_NAME]
    lines = [
        "Stage 05 comparison artifacts are reproducible under the shared dataset/seed/batch protocol."
        if decision_detail["artifact_checks_all_pass"]
        else "Stage 05 comparison artifacts do not yet fully satisfy the shared artifact protocol.",
        (
            "Stage 05 shows stable negative validation one-step energy delta vs identity across all comparison seeds."
            if decision_detail["one_step_energy_delta_vs_identity_negative_on_every_seed"]
            else "Stage 05 does not keep validation one-step energy delta vs identity negative on every comparison seed."
        ),
        (
            "Stage 05 keeps configured-step validation fixed-point residual delta vs identity negative across all comparison seeds."
            if decision_detail["configured_step_fixed_point_residual_delta_vs_identity_negative_on_every_seed"]
            else "Stage 05 does not keep configured-step validation fixed-point residual delta vs identity negative on every comparison seed."
        ),
    ]
    if justified:
        lines.append(
            "The corrected residual core has enough mechanism-first signal to justify a Stage 05 v2 charter."
        )
    else:
        lines.append(
            "The corrected residual core does not yet clear the stricter multiseed mechanism-first rule for a Stage 05 v2 charter."
        )
    lines.append("Stage 05 report-only accuracy remains a contextual metric, not the gate, in this comparison.")
    lines.append(
        f"Stage 05 mean validation accuracy is {stage05['val_accuracy']['mean']:.6f}, which is reported but not used as the charter gate."
    )
    return lines


def _does_not_support_lines(
    *,
    by_method: dict[str, dict[str, Any]],
    pairwise_stage05_vs_stage04: dict[str, Any],
) -> list[str]:
    stage04 = by_method[STAGE04_METHOD_NAME]
    stage05 = by_method[STAGE05_METHOD_NAME]
    lines = [
        "This comparison does not promote Stage 05 to replace the frozen Stage 04 bridge on main.",
        "This comparison does not reopen any Stage 04 package-internal stabilizer search.",
        "This comparison does not claim that Stage 05 has solved the task-accuracy gap to the frozen bridge.",
    ]
    if float(stage05["test_accuracy"]["mean"]) <= float(stage04["test_accuracy"]["mean"]):
        lines.append("Stage 05 remains below the frozen bridge on report-only test accuracy in the current comparison.")
    if float(pairwise_stage05_vs_stage04["configured_step_energy_delta_vs_identity_delta"]["mean"]) >= 0.0:
        lines.append(
            "Stage 05 does not outperform the frozen bridge on configured-step energy delta vs identity in this comparison."
        )
    return lines


def _comparison_report_markdown(report: dict[str, Any]) -> str:
    protocol = report["comparison_protocol"]
    decision = report["decision"]
    supports = report["supports"]
    does_not_support = report["does_not_support"]
    lines = [
        "# Frozen Bridge vs Corrected Residual Core",
        "",
        "## Protocol",
        f"- dataset: `{protocol['dataset_name']}`",
        f"- seeds: `{protocol['seeds']}`",
        f"- shared batch size: `{protocol['shared_batch_size']}`",
        f"- shared shuffle_batches: `{protocol['shared_shuffle_batches']}`",
        "",
        "## Decision",
        f"- `{JUSTIFY_V2_DECISION_NAME}`: `{decision[JUSTIFY_V2_DECISION_NAME]}`",
        f"- rationale: `{decision['decision_rationale']}`",
        "",
        "## Supports",
    ]
    for item in supports:
        lines.append(f"- {item}")
    lines.extend(["", "## Does Not Support"])
    for item in does_not_support:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _suite_config_payload(config: FrozenBridgeVsCorrectedCoreComparisonConfig) -> dict[str, Any]:
    return {
        "phase": "FMPC Stage 05 EF Core Probe",
        "stage": "frozen_bridge_vs_corrected_residual_core_comparison",
        "comparison_protocol": _comparison_protocol_payload(config),
        "artifacts": {
            "aggregate_runs_csv": "aggregate_runs.csv",
            "aggregate_summary_json": "aggregate_summary.json",
            "comparison_report_json": "comparison_report.json",
            "comparison_report_md": "comparison_report.md",
        },
    }


def run_frozen_bridge_vs_corrected_core_comparison(
    config: FrozenBridgeVsCorrectedCoreComparisonConfig,
) -> FrozenBridgeVsCorrectedCoreComparisonRunResult:
    """Run the formal frozen-bridge vs corrected-core comparison."""

    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, config.resolved_run_id(), config.output_layout)
    )
    _write_json(run_dir / "config.json", _suite_config_payload(config))

    rows: list[dict[str, Any]] = []
    runs_root = run_dir / "runs"
    run_index = 0

    for seed in config.seeds:
        run_index += 1
        stage04_config = _stage04_config(config, seed=seed, output_root=runs_root)
        stage04_result = run_fmpc_tf2_experiment(stage04_config)
        rows.append(
            _stage04_row(
                run_index=run_index,
                suite_run_dir=run_dir,
                seed=seed,
                result=stage04_result,
                config=stage04_config,
            )
        )

        run_index += 1
        stage05_config = _stage05_config(config, seed=seed, output_root=runs_root)
        stage05_result = run_fmpc_ef_exploratory_probe(stage05_config)
        rows.append(
            _stage05_row(
                run_index=run_index,
                suite_run_dir=run_dir,
                seed=seed,
                result=stage05_result,
                config=stage05_config,
            )
        )

    csv_rows = [
        {
            **row,
            "deterministic_artifact_checks_passed": str(bool(row["deterministic_artifact_checks_passed"])),
            "mechanism_signal_positive": str(bool(row["mechanism_signal_positive"])),
            "config_json_exists": str(bool(row["config_json_exists"])),
            "summary_json_exists": str(bool(row["summary_json_exists"])),
            "epoch_metrics_csv_exists": str(bool(row["epoch_metrics_csv_exists"])),
            "seed_matches": str(bool(row["seed_matches"])),
            "dataset_matches": str(bool(row["dataset_matches"])),
            "batch_protocol_matches": str(bool(row["batch_protocol_matches"])),
        }
        for row in rows
    ]
    _write_csv(run_dir / "aggregate_runs.csv", csv_rows)

    by_method = {
        STAGE04_METHOD_NAME: _method_summary(_method_rows(rows, STAGE04_METHOD_NAME)),
        STAGE05_METHOD_NAME: _method_summary(_method_rows(rows, STAGE05_METHOD_NAME)),
    }
    pairwise_stage05_vs_stage04 = _pairwise_summary(
        rows,
        candidate_method=STAGE05_METHOD_NAME,
        reference_method=STAGE04_METHOD_NAME,
    )
    justified, decision_detail = _stage05_v2_charter_decision(rows)
    decision_rationale = (
        "Stage 05 clears the multiseed mechanism-first comparison rule."
        if justified
        else "Stage 05 remains below the stricter multiseed mechanism-first charter rule."
    )

    summary = {
        "phase": "FMPC Stage 05 EF Core Probe",
        "stage": "frozen_bridge_vs_corrected_residual_core_comparison",
        "num_runs": int(len(rows)),
        "comparison_protocol": _comparison_protocol_payload(config),
        "by_method": by_method,
        "pairwise_stage05_vs_stage04": pairwise_stage05_vs_stage04,
        JUSTIFY_V2_DECISION_NAME: bool(justified),
        "stage05_v2_charter_decision_detail": decision_detail,
        "stage05_v2_charter_decision_rationale": decision_rationale,
        "aggregate_runs_csv_path": "aggregate_runs.csv",
        "comparison_report_json_path": "comparison_report.json",
        "comparison_report_md_path": "comparison_report.md",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)

    report = {
        "comparison_protocol": _comparison_protocol_payload(config),
        "decision": {
            JUSTIFY_V2_DECISION_NAME: bool(justified),
            "decision_detail": decision_detail,
            "decision_rationale": decision_rationale,
        },
        "supports": _supports_lines(
            justified=justified,
            decision_detail=decision_detail,
            by_method=by_method,
        ),
        "does_not_support": _does_not_support_lines(
            by_method=by_method,
            pairwise_stage05_vs_stage04=pairwise_stage05_vs_stage04,
        ),
    }
    _write_json(run_dir / "comparison_report.json", report)
    _write_text(run_dir / "comparison_report.md", _comparison_report_markdown(report))

    return FrozenBridgeVsCorrectedCoreComparisonRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config),
        aggregate_rows=rows,
        summary=summary,
        comparison_report=report,
    )
