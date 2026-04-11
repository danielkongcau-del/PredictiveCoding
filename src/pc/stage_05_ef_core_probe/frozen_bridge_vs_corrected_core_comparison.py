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
    "stage_05_corrected_residual_core_v1",
    "stage_05_two_branch_corrected_residual_core_v2",
    "stage_05_two_branch_corrected_residual_core_v2_current_budget",
    "stage_05_two_branch_corrected_residual_core_v2_longer_training",
]
OutputLayout = Literal["single_dir", "run_id_subdir"]

STAGE04_METHOD_NAME: ComparisonMethodName = "stage_04_frozen_bridge"
STAGE05_METHOD_NAME: ComparisonMethodName = "stage_05_corrected_residual_core"
STAGE05_V1_METHOD_NAME: ComparisonMethodName = "stage_05_corrected_residual_core_v1"
STAGE05_V2_METHOD_NAME: ComparisonMethodName = "stage_05_two_branch_corrected_residual_core_v2"
STAGE05_V2_CURRENT_BUDGET_METHOD_NAME: ComparisonMethodName = (
    "stage_05_two_branch_corrected_residual_core_v2_current_budget"
)
STAGE05_V2_LONGER_TRAINING_METHOD_NAME: ComparisonMethodName = (
    "stage_05_two_branch_corrected_residual_core_v2_longer_training"
)
JUSTIFY_V2_DECISION_NAME = "stage05_corrected_residual_core_justifies_v2_charter"
STAGE05_V2_FAVORABLE_DECISION_NAME = "stage05_v2_improves_mechanism_magnitude_over_v1"
STAGE05_V2_LONGER_TRAINING_DECISION_NAME = (
    "stage05_v2_longer_training_materially_improves_configured_step_mechanism"
)
STAGE05_V2_LONGER_TRAINING_ACCURACY_DECISION_NAME = (
    "stage05_v2_longer_training_materially_improves_report_only_accuracy"
)


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


@dataclass
class FrozenBridgeVsStage05V2ComparisonConfig:
    """Compare the frozen Stage 04 bridge against the Stage 05 v2 two-branch core."""

    experiment_name: str = "frozen_bridge_vs_two_branch_corrected_core_comparison"
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
    stage05_transport_steps: int = 2

    def __post_init__(self) -> None:
        if self.dataset_name != "digits":
            raise ValueError("The frozen-bridge vs Stage 05 v2 comparison currently supports digits only.")
        if not self.seeds:
            raise ValueError("seeds must contain at least one seed.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.stage04_epochs <= 0 or self.stage05_epochs <= 0:
            raise ValueError("stage04_epochs and stage05_epochs must be positive.")
        if self.stage04_eval_steps <= 0 or self.stage05_eval_steps <= 0:
            raise ValueError("stage04_eval_steps and stage05_eval_steps must be positive.")
        if self.stage05_transport_steps <= 0:
            raise ValueError("stage05_transport_steps must be positive.")

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_{self.seeds[0]}"


@dataclass
class CorrectedResidualCoreV1VsV2ComparisonConfig:
    """Compare the Stage 05 corrected residual core v1 against the Stage 05 v2 two-branch core."""

    experiment_name: str = "corrected_residual_core_v1_vs_v2_comparison"
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
    stage05_epochs: int = 12
    stage05_eval_steps: int = 15
    stage05_layer_dims: tuple[int, ...] = (64, 16, 10)
    stage05_transport_steps: int = 2

    def __post_init__(self) -> None:
        if self.dataset_name != "digits":
            raise ValueError("The Stage 05 v1 vs v2 comparison currently supports digits only.")
        if not self.seeds:
            raise ValueError("seeds must contain at least one seed.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.stage05_epochs <= 0 or self.stage05_eval_steps <= 0:
            raise ValueError("stage05_epochs and stage05_eval_steps must be positive.")
        if self.stage05_transport_steps <= 0:
            raise ValueError("stage05_transport_steps must be positive.")

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_{self.seeds[0]}"


@dataclass
class Stage05V2LongerTrainingValidationConfig:
    """Compare the current Stage 05 v2 budget against a longer-training budget."""

    experiment_name: str = "stage05_v2_longer_training_validation"
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
    current_stage05_epochs: int = 12
    longer_stage05_epochs: int = 24
    stage05_eval_steps: int = 15
    stage05_layer_dims: tuple[int, ...] = (64, 16, 10)
    stage05_transport_steps: int = 2
    configured_step_improvement_fraction_threshold: float = 0.05
    report_accuracy_improvement_threshold: float = 0.01

    def __post_init__(self) -> None:
        if self.dataset_name != "digits":
            raise ValueError("The Stage 05 v2 longer-training validation currently supports digits only.")
        if not self.seeds:
            raise ValueError("seeds must contain at least one seed.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.current_stage05_epochs <= 0 or self.longer_stage05_epochs <= 0:
            raise ValueError("current_stage05_epochs and longer_stage05_epochs must be positive.")
        if self.longer_stage05_epochs <= self.current_stage05_epochs:
            raise ValueError("longer_stage05_epochs must be greater than current_stage05_epochs.")
        if self.stage05_eval_steps <= 0:
            raise ValueError("stage05_eval_steps must be positive.")
        if self.stage05_transport_steps <= 0:
            raise ValueError("stage05_transport_steps must be positive.")
        if self.configured_step_improvement_fraction_threshold < 0.0:
            raise ValueError("configured_step_improvement_fraction_threshold must be non-negative.")
        if self.report_accuracy_improvement_threshold < 0.0:
            raise ValueError("report_accuracy_improvement_threshold must be non-negative.")

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_{self.seeds[0]}"


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


def _stage05_v1_config(
    config: CorrectedResidualCoreV1VsV2ComparisonConfig,
    *,
    seed: int,
    output_root: Path,
) -> FMPCEFExploratoryProbeConfig:
    return build_fmpc_ef_exploratory_probe_config(
        output_root=output_root,
        experiment_name=STAGE05_V1_METHOD_NAME,
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
        transport_steps=int(config.stage05_transport_steps),
        use_two_branch_residual_core=False,
        feature_aware_state_branch_tangents=False,
    )


def _build_stage05_v2_config(
    *,
    output_root: Path,
    experiment_name: str,
    seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    batch_size: int,
    shuffle_batches: bool,
    epochs: int,
    eval_steps: int,
    layer_dims: tuple[int, ...],
    transport_steps: int,
) -> FMPCEFExploratoryProbeConfig:
    return build_fmpc_ef_exploratory_probe_config(
        output_root=output_root,
        experiment_name=experiment_name,
        output_layout="run_id_subdir",
        run_id=f"seed_{seed}",
        run_seed=seed,
        data_seed=seed,
        model_init_seed=seed,
        psi_init_seed=seed,
        batch_order_seed=seed,
        train_fraction=float(train_fraction),
        val_fraction=float(val_fraction),
        test_fraction=float(test_fraction),
        batch_size=int(batch_size),
        shuffle_batches=bool(shuffle_batches),
        epochs=int(epochs),
        eval_steps=int(eval_steps),
        layer_dims=layer_dims,
        transport_steps=int(transport_steps),
        use_two_branch_residual_core=True,
        feature_aware_state_branch_tangents=True,
    )


def _stage05_v2_config(
    config: CorrectedResidualCoreV1VsV2ComparisonConfig,
    *,
    seed: int,
    output_root: Path,
) -> FMPCEFExploratoryProbeConfig:
    return _build_stage05_v2_config(
        output_root=output_root,
        experiment_name=STAGE05_V2_METHOD_NAME,
        seed=seed,
        train_fraction=float(config.train_fraction),
        val_fraction=float(config.val_fraction),
        test_fraction=float(config.test_fraction),
        batch_size=int(config.batch_size),
        shuffle_batches=bool(config.shuffle_batches),
        epochs=int(config.stage05_epochs),
        eval_steps=int(config.stage05_eval_steps),
        layer_dims=config.stage05_layer_dims,
        transport_steps=int(config.stage05_transport_steps),
    )


def _stage05_v2_bridge_config(
    config: FrozenBridgeVsStage05V2ComparisonConfig,
    *,
    seed: int,
    output_root: Path,
) -> FMPCEFExploratoryProbeConfig:
    return _build_stage05_v2_config(
        output_root=output_root,
        experiment_name=STAGE05_V2_METHOD_NAME,
        seed=seed,
        train_fraction=float(config.train_fraction),
        val_fraction=float(config.val_fraction),
        test_fraction=float(config.test_fraction),
        batch_size=int(config.batch_size),
        shuffle_batches=bool(config.shuffle_batches),
        epochs=int(config.stage05_epochs),
        eval_steps=int(config.stage05_eval_steps),
        layer_dims=config.stage05_layer_dims,
        transport_steps=int(config.stage05_transport_steps),
    )


def _stage05_v2_budget_config(
    config: Stage05V2LongerTrainingValidationConfig,
    *,
    seed: int,
    output_root: Path,
    experiment_name: str,
    epochs: int,
) -> FMPCEFExploratoryProbeConfig:
    return _build_stage05_v2_config(
        output_root=output_root,
        experiment_name=experiment_name,
        seed=seed,
        train_fraction=float(config.train_fraction),
        val_fraction=float(config.val_fraction),
        test_fraction=float(config.test_fraction),
        batch_size=int(config.batch_size),
        shuffle_batches=bool(config.shuffle_batches),
        epochs=int(epochs),
        eval_steps=int(config.stage05_eval_steps),
        layer_dims=config.stage05_layer_dims,
        transport_steps=int(config.stage05_transport_steps),
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
    selected_epoch = int(result.summary.get("selected_epoch", config.epochs))
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
        "transport_family": "stage04_frozen_bridge_control",
        "residual_branch_structure": "not_applicable",
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
        "selected_epoch": int(selected_epoch),
        "total_training_epochs": int(config.epochs),
        "selection_hits_final_training_boundary": bool(int(selected_epoch) >= int(config.epochs)),
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
    selected_epoch = int(summary["selected_epoch"])
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
        "selected_epoch": int(selected_epoch),
        "total_training_epochs": int(config.epochs),
        "selection_hits_final_training_boundary": bool(int(selected_epoch) >= int(config.epochs)),
        "runtime_proxy_seconds": float(runtime_proxy),
        "acceptance_contract": str(summary["acceptance_contract"]),
        "mechanism_signal_positive": bool(
            float(val_one_step["energy_delta_vs_identity"]) < 0.0
            and float(val_configured["energy_delta_vs_identity"]) < 0.0
            and float(val_configured["fixed_point_residual_delta_vs_identity"]) < 0.0
        ),
        **artifact_checks,
    }


def _stage05_core_row(
    *,
    run_index: int,
    suite_run_dir: Path,
    seed: int,
    result: Any,
    config: FMPCEFExploratoryProbeConfig,
    method_name: ComparisonMethodName,
    stage_name: str,
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
        "method_name": method_name,
        "stage_name": stage_name,
        "seed": int(seed),
        "run_id": str(result.run_dir.name),
        "run_config_path": _relative_posix(suite_run_dir, result.run_dir / "config.json"),
        "run_summary_path": _relative_posix(suite_run_dir, result.run_dir / "summary.json"),
        "transport_family": str(summary["transport_family"]),
        "residual_branch_structure": str(summary["residual_branch_structure"]),
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
        "selected_epoch": int(summary["selected_epoch"]),
        "total_training_epochs": int(config.epochs),
        "selection_hits_final_training_boundary": bool(int(summary["selected_epoch"]) >= int(config.epochs)),
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
        "total_training_epochs": int(rows[0].get("total_training_epochs", 0)),
        "selected_epoch": _metric_summary("selected_epoch"),
        "selection_hits_final_training_boundary_rate": _rate(
            [bool(row.get("selection_hits_final_training_boundary", False)) for row in rows]
        ),
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


def _stage05_v2_vs_stage04_protocol_payload(
    config: FrozenBridgeVsStage05V2ComparisonConfig,
) -> dict[str, Any]:
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
            "method_name": STAGE05_V2_METHOD_NAME,
            "transport_family": "two_branch_residual_meanflow_core",
            "residual_identity_mode": "residual_corrected_meanflow",
            "residual_branch_structure": "two_branch",
            "feature_aware_state_branch_tangents": True,
            "configured_transport_steps": int(config.stage05_transport_steps),
            "epochs": int(config.stage05_epochs),
            "eval_steps": int(config.stage05_eval_steps),
            "layer_dims": [int(value) for value in config.stage05_layer_dims],
        },
        "decision_rule": {
            "primary_split": "validation",
            "requires_all_stage05_v2_artifact_checks": True,
            "requires_stage05_v2_one_step_energy_delta_vs_identity_negative_on_every_seed": True,
            "requires_stage05_v2_configured_step_energy_delta_vs_identity_negative_on_every_seed": True,
            "requires_stage05_v2_configured_step_fixed_point_residual_delta_vs_identity_negative_on_every_seed": True,
            "task_accuracy_is_report_only": True,
            "replacement_claim_expected": False,
        },
    }


def _mechanism_strength_label(
    *,
    candidate_energy_mean: float,
    reference_energy_mean: float,
    candidate_residual_mean: float | None = None,
    reference_residual_mean: float | None = None,
) -> str:
    energy_better = float(candidate_energy_mean) < float(reference_energy_mean)
    if candidate_residual_mean is None or reference_residual_mean is None:
        return "stronger" if energy_better else "weaker"
    residual_better = float(candidate_residual_mean) < float(reference_residual_mean)
    return "stronger" if (energy_better and residual_better) else "weaker"


def _report_accuracy_strength_label(
    *,
    candidate_val_accuracy_mean: float,
    reference_val_accuracy_mean: float,
    candidate_test_accuracy_mean: float,
    reference_test_accuracy_mean: float,
) -> str:
    if (
        float(candidate_val_accuracy_mean) >= float(reference_val_accuracy_mean)
        and float(candidate_test_accuracy_mean) >= float(reference_test_accuracy_mean)
    ):
        return "stronger"
    return "weaker"


def _stage05_v2_vs_stage04_decision(
    rows: list[dict[str, Any]],
    *,
    by_method: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, str]]:
    stage05_rows = _method_rows(rows, STAGE05_V2_METHOD_NAME)
    artifact_pass = all(bool(row["deterministic_artifact_checks_passed"]) for row in stage05_rows)
    one_step_pass = all(float(row["one_step_energy_delta_vs_identity"]) < 0.0 for row in stage05_rows)
    configured_energy_pass = all(
        float(row["configured_step_energy_delta_vs_identity"]) < 0.0 for row in stage05_rows
    )
    configured_residual_pass = all(
        float(row["configured_step_fixed_point_residual_delta_vs_identity"]) < 0.0
        for row in stage05_rows
    )
    stage04 = by_method[STAGE04_METHOD_NAME]
    stage05 = by_method[STAGE05_V2_METHOD_NAME]

    one_step_strength = _mechanism_strength_label(
        candidate_energy_mean=stage05["one_step_energy_delta_vs_identity"]["mean"],
        reference_energy_mean=stage04["one_step_energy_delta_vs_identity"]["mean"],
    )
    configured_step_strength = _mechanism_strength_label(
        candidate_energy_mean=stage05["configured_step_energy_delta_vs_identity"]["mean"],
        reference_energy_mean=stage04["configured_step_energy_delta_vs_identity"]["mean"],
        candidate_residual_mean=stage05["configured_step_fixed_point_residual_delta_vs_identity"]["mean"],
        reference_residual_mean=stage04["configured_step_fixed_point_residual_delta_vs_identity"]["mean"],
    )
    accuracy_strength = _report_accuracy_strength_label(
        candidate_val_accuracy_mean=stage05["val_accuracy"]["mean"],
        reference_val_accuracy_mean=stage04["val_accuracy"]["mean"],
        candidate_test_accuracy_mean=stage05["test_accuracy"]["mean"],
        reference_test_accuracy_mean=stage04["test_accuracy"]["mean"],
    )

    justifies_continued_exploration = bool(
        artifact_pass and one_step_pass and configured_energy_pass and configured_residual_pass
    )
    as_new_reference = bool(justifies_continued_exploration)
    replaces_frozen_bridge = bool(
        justifies_continued_exploration
        and one_step_strength == "stronger"
        and configured_step_strength == "stronger"
        and accuracy_strength == "stronger"
    )

    decision = {
        "stage05_v2_justifies_continued_exploration": bool(justifies_continued_exploration),
        "stage05_v2_as_new_exploratory_reference": bool(as_new_reference),
        "stage05_v2_replaces_frozen_bridge_on_main": bool(replaces_frozen_bridge),
        "stage05_v2_decision_detail": {
            "artifact_checks_all_pass": bool(artifact_pass),
            "one_step_energy_delta_vs_identity_negative_on_every_seed": bool(one_step_pass),
            "configured_step_energy_delta_vs_identity_negative_on_every_seed": bool(
                configured_energy_pass
            ),
            "configured_step_fixed_point_residual_delta_vs_identity_negative_on_every_seed": bool(
                configured_residual_pass
            ),
        },
    }
    comparisons = {
        "one_step_mechanism_vs_stage04": one_step_strength,
        "configured_step_mechanism_vs_stage04": configured_step_strength,
        "report_only_accuracy_vs_stage04": accuracy_strength,
    }
    return decision, comparisons


def _stage05_v2_vs_stage04_decision_rationale(
    *,
    decision: dict[str, Any],
    comparisons: dict[str, str],
) -> str:
    if not bool(decision["stage05_v2_justifies_continued_exploration"]):
        return "Stage 05 v2 keeps positive mechanism signal but does not clear the refreshed multiseed exploration rule."
    if bool(decision["stage05_v2_replaces_frozen_bridge_on_main"]):
        return "Stage 05 v2 clears the refreshed exploration rule and also clears the much stronger replacement rule."
    return (
        "Stage 05 v2 clears the refreshed mechanism-first exploration rule, "
        f"is {comparisons['one_step_mechanism_vs_stage04']} on one-step mechanism, "
        f"is {comparisons['configured_step_mechanism_vs_stage04']} on configured-step mechanism, "
        f"and is {comparisons['report_only_accuracy_vs_stage04']} on report-only accuracy versus the frozen bridge."
    )


def _stage05_v2_vs_stage04_supports_lines(
    *,
    decision: dict[str, Any],
    comparisons: dict[str, str],
    by_method: dict[str, dict[str, Any]],
) -> list[str]:
    stage05 = by_method[STAGE05_V2_METHOD_NAME]
    lines = [
        (
            "Stage 05 v2 artifacts are reproducible under the shared dataset/seed/batch protocol."
            if decision["stage05_v2_decision_detail"]["artifact_checks_all_pass"]
            else "Stage 05 v2 artifacts do not yet fully satisfy the shared artifact protocol."
        ),
        (
            "Stage 05 v2 keeps one-step validation energy delta vs identity negative on every comparison seed."
            if decision["stage05_v2_decision_detail"]["one_step_energy_delta_vs_identity_negative_on_every_seed"]
            else "Stage 05 v2 does not keep one-step validation energy delta vs identity negative on every comparison seed."
        ),
        (
            "Stage 05 v2 keeps configured-step validation energy delta vs identity negative on every comparison seed."
            if decision["stage05_v2_decision_detail"]["configured_step_energy_delta_vs_identity_negative_on_every_seed"]
            else "Stage 05 v2 does not keep configured-step validation energy delta vs identity negative on every comparison seed."
        ),
        (
            "Stage 05 v2 keeps configured-step validation fixed-point residual delta vs identity negative on every comparison seed."
            if decision["stage05_v2_decision_detail"]["configured_step_fixed_point_residual_delta_vs_identity_negative_on_every_seed"]
            else "Stage 05 v2 does not keep configured-step validation fixed-point residual delta vs identity negative on every comparison seed."
        ),
        f"Relative to the frozen bridge, Stage 05 v2 is {comparisons['one_step_mechanism_vs_stage04']} on one-step mechanism.",
        f"Relative to the frozen bridge, Stage 05 v2 is {comparisons['configured_step_mechanism_vs_stage04']} on configured-step mechanism.",
        f"Relative to the frozen bridge, Stage 05 v2 is {comparisons['report_only_accuracy_vs_stage04']} on report-only accuracy.",
    ]
    if bool(decision["stage05_v2_justifies_continued_exploration"]):
        lines.append("The refreshed comparison supports continued Stage 05 mechanism-first exploration.")
    if bool(decision["stage05_v2_as_new_exploratory_reference"]):
        lines.append("The refreshed comparison supports using Stage 05 v2 as the new exploratory reference.")
    lines.append(
        f"Stage 05 v2 mean validation accuracy is {stage05['val_accuracy']['mean']:.6f}; accuracy remains report-only in this comparison."
    )
    return lines


def _stage05_v2_vs_stage04_does_not_support_lines(
    *,
    decision: dict[str, Any],
) -> list[str]:
    lines = [
        "This refreshed comparison supports keeping Stage 04 frozen on main.",
        "This refreshed comparison supports keeping Stage 05 mechanism-first.",
        "This refreshed comparison does not reopen any Stage 04 package-internal work.",
        "This refreshed comparison does not promote task accuracy to the Stage 05 gate.",
    ]
    if not bool(decision["stage05_v2_replaces_frozen_bridge_on_main"]):
        lines.append("This refreshed comparison does not support replacing the frozen Stage 04 bridge on main.")
    return lines


def _stage05_v2_vs_stage04_report_markdown(report: dict[str, Any]) -> str:
    protocol = report["comparison_protocol"]
    decision = report["decision"]
    supports = report["supports"]
    does_not_support = report["does_not_support"]
    lines = [
        "# Frozen Bridge vs Two-Branch Corrected Residual Core",
        "",
        "## Protocol",
        f"- dataset: `{protocol['dataset_name']}`",
        f"- seeds: `{protocol['seeds']}`",
        f"- shared batch size: `{protocol['shared_batch_size']}`",
        f"- shared shuffle_batches: `{protocol['shared_shuffle_batches']}`",
        "",
        "## Decision",
        f"- `stage05_v2_justifies_continued_exploration`: `{decision['stage05_v2_justifies_continued_exploration']}`",
        f"- `stage05_v2_as_new_exploratory_reference`: `{decision['stage05_v2_as_new_exploratory_reference']}`",
        f"- `stage05_v2_replaces_frozen_bridge_on_main`: `{decision['stage05_v2_replaces_frozen_bridge_on_main']}`",
        f"- rationale: `{decision['stage05_v2_vs_stage04_decision_rationale']}`",
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


def _stage05_v2_vs_stage04_suite_config_payload(
    config: FrozenBridgeVsStage05V2ComparisonConfig,
) -> dict[str, Any]:
    return {
        "phase": "FMPC Stage 05 EF Core Probe",
        "stage": "frozen_bridge_vs_two_branch_corrected_residual_core_comparison",
        "comparison_protocol": _stage05_v2_vs_stage04_protocol_payload(config),
        "artifacts": {
            "aggregate_runs_csv": "aggregate_runs.csv",
            "aggregate_summary_json": "aggregate_summary.json",
            "comparison_report_json": "comparison_report.json",
            "comparison_report_md": "comparison_report.md",
        },
    }


def run_frozen_bridge_vs_stage05_v2_comparison(
    config: FrozenBridgeVsStage05V2ComparisonConfig,
) -> FrozenBridgeVsCorrectedCoreComparisonRunResult:
    """Run the refreshed frozen-bridge vs Stage 05 v2 comparison."""

    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, config.resolved_run_id(), config.output_layout)
    )
    _write_json(run_dir / "config.json", _stage05_v2_vs_stage04_suite_config_payload(config))

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
        stage05_config = _stage05_v2_bridge_config(config, seed=seed, output_root=runs_root)
        stage05_result = run_fmpc_ef_exploratory_probe(stage05_config)
        rows.append(
            _stage05_core_row(
                run_index=run_index,
                suite_run_dir=run_dir,
                seed=seed,
                result=stage05_result,
                config=stage05_config,
                method_name=STAGE05_V2_METHOD_NAME,
                stage_name="FMPC Stage 05 EF Core Probe v2",
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
        STAGE05_V2_METHOD_NAME: _method_summary(_method_rows(rows, STAGE05_V2_METHOD_NAME)),
    }
    pairwise_stage05_v2_vs_stage04 = _pairwise_summary(
        rows,
        candidate_method=STAGE05_V2_METHOD_NAME,
        reference_method=STAGE04_METHOD_NAME,
    )
    decision, comparisons = _stage05_v2_vs_stage04_decision(rows, by_method=by_method)
    decision_rationale = _stage05_v2_vs_stage04_decision_rationale(
        decision=decision,
        comparisons=comparisons,
    )

    summary = {
        "phase": "FMPC Stage 05 EF Core Probe",
        "stage": "frozen_bridge_vs_two_branch_corrected_residual_core_comparison",
        "num_runs": int(len(rows)),
        "comparison_protocol": _stage05_v2_vs_stage04_protocol_payload(config),
        "by_method": by_method,
        "pairwise_stage05_v2_vs_stage04": pairwise_stage05_v2_vs_stage04,
        **decision,
        **comparisons,
        "stage05_v2_vs_stage04_decision_rationale": decision_rationale,
        "aggregate_runs_csv_path": "aggregate_runs.csv",
        "comparison_report_json_path": "comparison_report.json",
        "comparison_report_md_path": "comparison_report.md",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)

    report = {
        "comparison_protocol": _stage05_v2_vs_stage04_protocol_payload(config),
        "decision": {
            **decision,
            **comparisons,
            "stage05_v2_vs_stage04_decision_rationale": decision_rationale,
        },
        "supports": _stage05_v2_vs_stage04_supports_lines(
            decision=decision,
            comparisons=comparisons,
            by_method=by_method,
        ),
        "does_not_support": _stage05_v2_vs_stage04_does_not_support_lines(
            decision=decision,
        ),
    }
    _write_json(run_dir / "comparison_report.json", report)
    _write_text(run_dir / "comparison_report.md", _stage05_v2_vs_stage04_report_markdown(report))

    return FrozenBridgeVsCorrectedCoreComparisonRunResult(
        run_dir=run_dir,
        config=_stage05_v2_vs_stage04_suite_config_payload(config),
        aggregate_rows=rows,
        summary=summary,
        comparison_report=report,
    )


def _stage05_v1_vs_v2_protocol_payload(
    config: CorrectedResidualCoreV1VsV2ComparisonConfig,
) -> dict[str, Any]:
    return {
        "dataset_name": config.dataset_name,
        "seeds": [int(seed) for seed in config.seeds],
        "train_fraction": float(config.train_fraction),
        "val_fraction": float(config.val_fraction),
        "test_fraction": float(config.test_fraction),
        "shared_batch_size": int(config.batch_size),
        "shared_shuffle_batches": bool(config.shuffle_batches),
        "stage_05_v1_reference": {
            "method_name": STAGE05_V1_METHOD_NAME,
            "transport_family": "residual_meanflow_core",
            "residual_branch_structure": "single_branch",
            "configured_transport_steps": int(config.stage05_transport_steps),
            "epochs": int(config.stage05_epochs),
            "eval_steps": int(config.stage05_eval_steps),
            "layer_dims": [int(value) for value in config.stage05_layer_dims],
        },
        "stage_05_v2_candidate": {
            "method_name": STAGE05_V2_METHOD_NAME,
            "transport_family": "two_branch_residual_meanflow_core",
            "residual_branch_structure": "two_branch",
            "configured_transport_steps": int(config.stage05_transport_steps),
            "epochs": int(config.stage05_epochs),
            "eval_steps": int(config.stage05_eval_steps),
            "layer_dims": [int(value) for value in config.stage05_layer_dims],
            "feature_aware_state_branch_tangents": True,
        },
        "decision_rule": {
            "primary_split": "validation",
            "requires_all_stage05_v2_artifact_checks": True,
            "requires_stage05_v2_one_step_energy_delta_vs_identity_negative_on_every_seed": True,
            "requires_stage05_v2_configured_step_energy_delta_vs_identity_negative_on_every_seed": True,
            "requires_stage05_v2_configured_step_fixed_point_residual_delta_vs_identity_negative_on_every_seed": True,
            "requires_stage05_v2_mean_configured_step_energy_delta_vs_identity_more_negative_than_v1": True,
            "requires_stage05_v2_mean_configured_step_fixed_point_residual_delta_vs_identity_more_negative_than_v1": True,
            "task_accuracy_is_report_only": True,
        },
    }


def _stage05_v2_vs_v1_decision(
    rows: list[dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    v1_rows = _method_rows(rows, STAGE05_V1_METHOD_NAME)
    v2_rows = _method_rows(rows, STAGE05_V2_METHOD_NAME)
    artifact_pass = all(bool(row["deterministic_artifact_checks_passed"]) for row in v2_rows)
    one_step_pass = all(float(row["one_step_energy_delta_vs_identity"]) < 0.0 for row in v2_rows)
    configured_energy_pass = all(
        float(row["configured_step_energy_delta_vs_identity"]) < 0.0 for row in v2_rows
    )
    configured_residual_pass = all(
        float(row["configured_step_fixed_point_residual_delta_vs_identity"]) < 0.0 for row in v2_rows
    )
    v1_mean_configured_energy = _mean(
        [float(row["configured_step_energy_delta_vs_identity"]) for row in v1_rows]
    )
    v2_mean_configured_energy = _mean(
        [float(row["configured_step_energy_delta_vs_identity"]) for row in v2_rows]
    )
    v1_mean_configured_residual = _mean(
        [float(row["configured_step_fixed_point_residual_delta_vs_identity"]) for row in v1_rows]
    )
    v2_mean_configured_residual = _mean(
        [float(row["configured_step_fixed_point_residual_delta_vs_identity"]) for row in v2_rows]
    )
    mean_configured_energy_better = bool(v2_mean_configured_energy < v1_mean_configured_energy)
    mean_configured_residual_better = bool(
        v2_mean_configured_residual < v1_mean_configured_residual
    )
    favorable = bool(
        artifact_pass
        and one_step_pass
        and configured_energy_pass
        and configured_residual_pass
        and mean_configured_energy_better
        and mean_configured_residual_better
    )
    return favorable, {
        "artifact_checks_all_pass": bool(artifact_pass),
        "one_step_energy_delta_vs_identity_negative_on_every_seed": bool(one_step_pass),
        "configured_step_energy_delta_vs_identity_negative_on_every_seed": bool(
            configured_energy_pass
        ),
        "configured_step_fixed_point_residual_delta_vs_identity_negative_on_every_seed": bool(
            configured_residual_pass
        ),
        "mean_configured_step_energy_delta_vs_identity_more_negative_than_v1": bool(
            mean_configured_energy_better
        ),
        "mean_configured_step_fixed_point_residual_delta_vs_identity_more_negative_than_v1": bool(
            mean_configured_residual_better
        ),
    }


def _stage05_v1_vs_v2_supports_lines(
    *,
    favorable: bool,
    decision_detail: dict[str, Any],
    by_method: dict[str, dict[str, Any]],
) -> list[str]:
    v1 = by_method[STAGE05_V1_METHOD_NAME]
    v2 = by_method[STAGE05_V2_METHOD_NAME]
    lines = [
        (
            "Stage 05 v2 artifacts are reproducible under the shared dataset/seed/batch protocol."
            if decision_detail["artifact_checks_all_pass"]
            else "Stage 05 v2 artifacts do not yet fully satisfy the shared artifact protocol."
        ),
        (
            "Stage 05 v2 keeps one-step validation energy delta vs identity negative on every comparison seed."
            if decision_detail["one_step_energy_delta_vs_identity_negative_on_every_seed"]
            else "Stage 05 v2 does not keep one-step validation energy delta vs identity negative on every comparison seed."
        ),
        (
            "Stage 05 v2 keeps configured-step validation fixed-point residual delta vs identity negative on every comparison seed."
            if decision_detail["configured_step_fixed_point_residual_delta_vs_identity_negative_on_every_seed"]
            else "Stage 05 v2 does not keep configured-step validation fixed-point residual delta vs identity negative on every comparison seed."
        ),
        (
            "Stage 05 v2 improves mean configured-step validation energy delta vs identity over v1."
            if decision_detail["mean_configured_step_energy_delta_vs_identity_more_negative_than_v1"]
            else "Stage 05 v2 does not improve mean configured-step validation energy delta vs identity over v1."
        ),
        (
            "Stage 05 v2 improves mean configured-step validation fixed-point residual delta vs identity over v1."
            if decision_detail["mean_configured_step_fixed_point_residual_delta_vs_identity_more_negative_than_v1"]
            else "Stage 05 v2 does not improve mean configured-step validation fixed-point residual delta vs identity over v1."
        ),
    ]
    if favorable:
        lines.append(
            "The two-branch corrected residual core is favorable on mechanism-first grounds for the next narrow Stage 05 step."
        )
    else:
        lines.append(
            "The two-branch corrected residual core does not yet improve mechanism magnitude over v1 under the narrow Stage 05 v2 rule."
        )
    lines.append(
        f"Stage 05 v1 mean validation accuracy is {v1['val_accuracy']['mean']:.6f} and Stage 05 v2 mean validation accuracy is {v2['val_accuracy']['mean']:.6f}; accuracy remains report-only."
    )
    return lines


def _stage05_v1_vs_v2_does_not_support_lines(
    *,
    pairwise_v2_vs_v1: dict[str, Any],
) -> list[str]:
    lines = [
        "This comparison does not claim that Stage 05 replaces the frozen Stage 04 bridge result on main.",
        "This comparison does not reopen any Stage 04 package-internal work.",
        "This comparison does not promote task accuracy to a gate.",
    ]
    if (
        float(pairwise_v2_vs_v1["configured_step_energy_delta_vs_identity_delta"]["mean"]) >= 0.0
    ):
        lines.append(
            "Stage 05 v2 does not improve configured-step energy delta vs identity over v1 in this comparison."
        )
    if (
        float(
            pairwise_v2_vs_v1[
                "configured_step_fixed_point_residual_delta_vs_identity_delta"
            ]["mean"]
        )
        >= 0.0
    ):
        lines.append(
            "Stage 05 v2 does not improve configured-step fixed-point residual delta vs identity over v1 in this comparison."
        )
    return lines


def _stage05_v1_vs_v2_report_markdown(report: dict[str, Any]) -> str:
    protocol = report["comparison_protocol"]
    decision = report["decision"]
    supports = report["supports"]
    does_not_support = report["does_not_support"]
    lines = [
        "# Stage 05 Corrected Residual Core v1 vs v2",
        "",
        "## Protocol",
        f"- dataset: `{protocol['dataset_name']}`",
        f"- seeds: `{protocol['seeds']}`",
        f"- shared batch size: `{protocol['shared_batch_size']}`",
        f"- shared shuffle_batches: `{protocol['shared_shuffle_batches']}`",
        "",
        "## Decision",
        f"- `{STAGE05_V2_FAVORABLE_DECISION_NAME}`: `{decision[STAGE05_V2_FAVORABLE_DECISION_NAME]}`",
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


def _stage05_v1_vs_v2_suite_config_payload(
    config: CorrectedResidualCoreV1VsV2ComparisonConfig,
) -> dict[str, Any]:
    return {
        "phase": "FMPC Stage 05 EF Core Probe",
        "stage": "corrected_residual_core_v1_vs_v2_comparison",
        "comparison_protocol": _stage05_v1_vs_v2_protocol_payload(config),
        "artifacts": {
            "aggregate_runs_csv": "aggregate_runs.csv",
            "aggregate_summary_json": "aggregate_summary.json",
            "comparison_report_json": "comparison_report.json",
            "comparison_report_md": "comparison_report.md",
        },
    }


def run_corrected_residual_core_v1_vs_v2_comparison(
    config: CorrectedResidualCoreV1VsV2ComparisonConfig,
) -> FrozenBridgeVsCorrectedCoreComparisonRunResult:
    """Run the formal Stage 05 corrected residual core v1 vs v2 comparison."""

    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, config.resolved_run_id(), config.output_layout)
    )
    _write_json(run_dir / "config.json", _stage05_v1_vs_v2_suite_config_payload(config))

    rows: list[dict[str, Any]] = []
    runs_root = run_dir / "runs"
    run_index = 0

    for seed in config.seeds:
        run_index += 1
        v1_config = _stage05_v1_config(config, seed=seed, output_root=runs_root)
        v1_result = run_fmpc_ef_exploratory_probe(v1_config)
        rows.append(
            _stage05_core_row(
                run_index=run_index,
                suite_run_dir=run_dir,
                seed=seed,
                result=v1_result,
                config=v1_config,
                method_name=STAGE05_V1_METHOD_NAME,
                stage_name="FMPC Stage 05 EF Core Probe v1",
            )
        )

        run_index += 1
        v2_config = _stage05_v2_config(config, seed=seed, output_root=runs_root)
        v2_result = run_fmpc_ef_exploratory_probe(v2_config)
        rows.append(
            _stage05_core_row(
                run_index=run_index,
                suite_run_dir=run_dir,
                seed=seed,
                result=v2_result,
                config=v2_config,
                method_name=STAGE05_V2_METHOD_NAME,
                stage_name="FMPC Stage 05 EF Core Probe v2",
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
        STAGE05_V1_METHOD_NAME: _method_summary(_method_rows(rows, STAGE05_V1_METHOD_NAME)),
        STAGE05_V2_METHOD_NAME: _method_summary(_method_rows(rows, STAGE05_V2_METHOD_NAME)),
    }
    pairwise_v2_vs_v1 = _pairwise_summary(
        rows,
        candidate_method=STAGE05_V2_METHOD_NAME,
        reference_method=STAGE05_V1_METHOD_NAME,
    )
    favorable, decision_detail = _stage05_v2_vs_v1_decision(rows)
    decision_rationale = (
        "Stage 05 v2 improves mechanism magnitude over v1 under the narrow multiseed rule."
        if favorable
        else "Stage 05 v2 does not yet improve mechanism magnitude over v1 under the narrow multiseed rule."
    )

    summary = {
        "phase": "FMPC Stage 05 EF Core Probe",
        "stage": "corrected_residual_core_v1_vs_v2_comparison",
        "num_runs": int(len(rows)),
        "comparison_protocol": _stage05_v1_vs_v2_protocol_payload(config),
        "by_method": by_method,
        "pairwise_stage05_v2_vs_v1": pairwise_v2_vs_v1,
        STAGE05_V2_FAVORABLE_DECISION_NAME: bool(favorable),
        "stage05_v2_vs_v1_decision_detail": decision_detail,
        "stage05_v2_vs_v1_decision_rationale": decision_rationale,
        "aggregate_runs_csv_path": "aggregate_runs.csv",
        "comparison_report_json_path": "comparison_report.json",
        "comparison_report_md_path": "comparison_report.md",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)

    report = {
        "comparison_protocol": _stage05_v1_vs_v2_protocol_payload(config),
        "decision": {
            STAGE05_V2_FAVORABLE_DECISION_NAME: bool(favorable),
            "decision_detail": decision_detail,
            "decision_rationale": decision_rationale,
        },
        "supports": _stage05_v1_vs_v2_supports_lines(
            favorable=favorable,
            decision_detail=decision_detail,
            by_method=by_method,
        ),
        "does_not_support": _stage05_v1_vs_v2_does_not_support_lines(
            pairwise_v2_vs_v1=pairwise_v2_vs_v1,
        ),
    }
    _write_json(run_dir / "comparison_report.json", report)
    _write_text(run_dir / "comparison_report.md", _stage05_v1_vs_v2_report_markdown(report))

    return FrozenBridgeVsCorrectedCoreComparisonRunResult(
        run_dir=run_dir,
        config=_stage05_v1_vs_v2_suite_config_payload(config),
        aggregate_rows=rows,
        summary=summary,
        comparison_report=report,
    )


def _negative_magnitude_relative_gain(
    *,
    current_value: float,
    candidate_value: float,
) -> float:
    baseline = abs(float(current_value))
    candidate = abs(float(candidate_value))
    scale = max(baseline, 1e-12)
    return float((candidate - baseline) / scale)


def _stage05_v2_longer_training_protocol_payload(
    config: Stage05V2LongerTrainingValidationConfig,
) -> dict[str, Any]:
    return {
        "dataset_name": config.dataset_name,
        "seeds": [int(seed) for seed in config.seeds],
        "train_fraction": float(config.train_fraction),
        "val_fraction": float(config.val_fraction),
        "test_fraction": float(config.test_fraction),
        "shared_batch_size": int(config.batch_size),
        "shared_shuffle_batches": bool(config.shuffle_batches),
        "current_budget_reference": {
            "method_name": STAGE05_V2_CURRENT_BUDGET_METHOD_NAME,
            "transport_family": "two_branch_residual_meanflow_core",
            "residual_branch_structure": "two_branch",
            "feature_aware_state_branch_tangents": True,
            "epochs": int(config.current_stage05_epochs),
            "eval_steps": int(config.stage05_eval_steps),
            "configured_transport_steps": int(config.stage05_transport_steps),
            "layer_dims": [int(value) for value in config.stage05_layer_dims],
        },
        "longer_training_candidate": {
            "method_name": STAGE05_V2_LONGER_TRAINING_METHOD_NAME,
            "transport_family": "two_branch_residual_meanflow_core",
            "residual_branch_structure": "two_branch",
            "feature_aware_state_branch_tangents": True,
            "epochs": int(config.longer_stage05_epochs),
            "eval_steps": int(config.stage05_eval_steps),
            "configured_transport_steps": int(config.stage05_transport_steps),
            "layer_dims": [int(value) for value in config.stage05_layer_dims],
        },
        "decision_rule": {
            "primary_split": "validation",
            "task_accuracy_is_report_only": True,
            "configured_step_improvement_fraction_threshold": float(
                config.configured_step_improvement_fraction_threshold
            ),
            "report_accuracy_improvement_threshold": float(
                config.report_accuracy_improvement_threshold
            ),
            "longer_budget_selection_rule_unchanged": True,
        },
    }


def _stage05_v2_longer_training_decision(
    *,
    rows: list[dict[str, Any]],
    config: Stage05V2LongerTrainingValidationConfig,
    by_method: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    current_rows = _method_rows(rows, STAGE05_V2_CURRENT_BUDGET_METHOD_NAME)
    longer_rows = _method_rows(rows, STAGE05_V2_LONGER_TRAINING_METHOD_NAME)
    current_by_seed = {int(row["seed"]): row for row in current_rows}
    longer_by_seed = {int(row["seed"]): row for row in longer_rows}
    shared_seeds = sorted(set(current_by_seed).intersection(longer_by_seed))
    if not shared_seeds:
        raise ValueError("Longer-training decision requires shared seeds.")

    current_summary = by_method[STAGE05_V2_CURRENT_BUDGET_METHOD_NAME]
    longer_summary = by_method[STAGE05_V2_LONGER_TRAINING_METHOD_NAME]
    current_energy_mean = float(
        current_summary["configured_step_energy_delta_vs_identity"]["mean"]
    )
    longer_energy_mean = float(
        longer_summary["configured_step_energy_delta_vs_identity"]["mean"]
    )
    current_residual_mean = float(
        current_summary["configured_step_fixed_point_residual_delta_vs_identity"]["mean"]
    )
    longer_residual_mean = float(
        longer_summary["configured_step_fixed_point_residual_delta_vs_identity"]["mean"]
    )
    current_val_accuracy_mean = float(current_summary["val_accuracy"]["mean"])
    longer_val_accuracy_mean = float(longer_summary["val_accuracy"]["mean"])
    current_test_accuracy_mean = float(current_summary["test_accuracy"]["mean"])
    longer_test_accuracy_mean = float(longer_summary["test_accuracy"]["mean"])

    configured_energy_gain_fraction = _negative_magnitude_relative_gain(
        current_value=current_energy_mean,
        candidate_value=longer_energy_mean,
    )
    configured_residual_gain_fraction = _negative_magnitude_relative_gain(
        current_value=current_residual_mean,
        candidate_value=longer_residual_mean,
    )
    energy_seed_improvement_rate = _rate(
        [
            float(longer_by_seed[seed]["configured_step_energy_delta_vs_identity"])
            < float(current_by_seed[seed]["configured_step_energy_delta_vs_identity"])
            for seed in shared_seeds
        ]
    )
    residual_seed_improvement_rate = _rate(
        [
            float(longer_by_seed[seed]["configured_step_fixed_point_residual_delta_vs_identity"])
            < float(current_by_seed[seed]["configured_step_fixed_point_residual_delta_vs_identity"])
            for seed in shared_seeds
        ]
    )
    configured_step_mechanism_improved_materially = bool(
        configured_energy_gain_fraction
        >= float(config.configured_step_improvement_fraction_threshold)
        and configured_residual_gain_fraction
        >= float(config.configured_step_improvement_fraction_threshold)
        and energy_seed_improvement_rate >= 0.5
        and residual_seed_improvement_rate >= 0.5
    )
    val_accuracy_gain = float(longer_val_accuracy_mean - current_val_accuracy_mean)
    test_accuracy_gain = float(longer_test_accuracy_mean - current_test_accuracy_mean)
    report_only_accuracy_improved_materially = bool(
        val_accuracy_gain >= float(config.report_accuracy_improvement_threshold)
        and test_accuracy_gain >= float(config.report_accuracy_improvement_threshold)
    )
    current_budget_boundary_all = bool(
        all(bool(row["selection_hits_final_training_boundary"]) for row in current_rows)
    )
    longer_budget_boundary_all = bool(
        all(bool(row["selection_hits_final_training_boundary"]) for row in longer_rows)
    )
    longer_budget_boundary_rate = _rate(
        [bool(row["selection_hits_final_training_boundary"]) for row in longer_rows]
    )
    recommended_next_move = (
        "continue_with_budget"
        if longer_budget_boundary_all
        else "open_stage05_v3_charter"
    )
    if longer_budget_boundary_all:
        rationale = (
            "The stronger Stage 05 v2 budget still selects the final training epoch on every seed, "
            "so the budget question is not yet closed."
        )
    elif configured_step_mechanism_improved_materially or report_only_accuracy_improved_materially:
        rationale = (
            "The stronger Stage 05 v2 budget improves the current v2 reference without still selecting "
            "the final training epoch on every seed, so the budget question is now better answered "
            "and the next step can move to a true v3 charter if needed."
        )
    else:
        rationale = (
            "The stronger Stage 05 v2 budget no longer looks boundary-limited and still does not "
            "materially improve the current v2 reference, so a true v3 mechanism charter is now justified."
        )
    decision = {
        STAGE05_V2_LONGER_TRAINING_DECISION_NAME: bool(
            configured_step_mechanism_improved_materially
        ),
        STAGE05_V2_LONGER_TRAINING_ACCURACY_DECISION_NAME: bool(
            report_only_accuracy_improved_materially
        ),
        "configured_step_energy_gain_fraction": float(configured_energy_gain_fraction),
        "configured_step_residual_gain_fraction": float(configured_residual_gain_fraction),
        "configured_step_energy_seed_improvement_rate": float(energy_seed_improvement_rate),
        "configured_step_residual_seed_improvement_rate": float(residual_seed_improvement_rate),
        "val_accuracy_gain": float(val_accuracy_gain),
        "test_accuracy_gain": float(test_accuracy_gain),
        "current_budget_selection_hits_final_training_boundary_on_all_seeds": bool(
            current_budget_boundary_all
        ),
        "longer_budget_selection_hits_final_training_boundary_on_all_seeds": bool(
            longer_budget_boundary_all
        ),
        "longer_budget_selection_hits_final_training_boundary_rate": float(
            longer_budget_boundary_rate
        ),
        "recommended_next_move": recommended_next_move,
    }
    return decision, rationale


def _stage05_v2_longer_training_supports_lines(
    *,
    decision: dict[str, Any],
    by_method: dict[str, dict[str, Any]],
) -> list[str]:
    current_summary = by_method[STAGE05_V2_CURRENT_BUDGET_METHOD_NAME]
    longer_summary = by_method[STAGE05_V2_LONGER_TRAINING_METHOD_NAME]
    return [
        (
            "The longer-training Stage 05 v2 candidate materially improves configured-step mechanism magnitude over the current budget."
            if decision[STAGE05_V2_LONGER_TRAINING_DECISION_NAME]
            else "The longer-training Stage 05 v2 candidate does not yet materially improve configured-step mechanism magnitude over the current budget."
        ),
        (
            "The longer-training Stage 05 v2 candidate materially improves report-only accuracy over the current budget."
            if decision[STAGE05_V2_LONGER_TRAINING_ACCURACY_DECISION_NAME]
            else "The longer-training Stage 05 v2 candidate does not yet materially improve report-only accuracy over the current budget."
        ),
        (
            "The stronger budget still hits the final training boundary on every seed."
            if decision["longer_budget_selection_hits_final_training_boundary_on_all_seeds"]
            else "The stronger budget no longer hits the final training boundary on every seed."
        ),
        (
            f"Current-budget configured-step validation energy delta vs identity mean: {current_summary['configured_step_energy_delta_vs_identity']['mean']:.12f}."
        ),
        (
            f"Longer-budget configured-step validation energy delta vs identity mean: {longer_summary['configured_step_energy_delta_vs_identity']['mean']:.12f}."
        ),
        (
            f"Current-budget validation/test accuracy means: {current_summary['val_accuracy']['mean']:.6f} / {current_summary['test_accuracy']['mean']:.6f}."
        ),
        (
            f"Longer-budget validation/test accuracy means: {longer_summary['val_accuracy']['mean']:.6f} / {longer_summary['test_accuracy']['mean']:.6f}."
        ),
    ]


def _stage05_v2_longer_training_does_not_support_lines(
    *,
    decision: dict[str, Any],
) -> list[str]:
    lines = [
        "This validation does not reopen Stage 04 package-internal work.",
        "This validation does not change the Stage 05 v2 transport family, objective family, or selection rule.",
        "This validation does not claim that Stage 05 replaces the frozen Stage 04 bridge result on main.",
    ]
    if decision["recommended_next_move"] == "continue_with_budget":
        lines.append(
            "This validation does not yet justify opening a true Stage 05 v3 mechanism charter."
        )
    else:
        lines.append("This validation does not imply any new Stage 04 replacement claim.")
    return lines


def _stage05_v2_longer_training_report_markdown(report: dict[str, Any]) -> str:
    protocol = report["comparison_protocol"]
    decision = report["decision"]
    lines = [
        "# Stage 05 V2 Longer-Training Validation",
        "",
        "## Protocol",
        f"- dataset: `{protocol['dataset_name']}`",
        f"- seeds: `{protocol['seeds']}`",
        f"- shared batch size: `{protocol['shared_batch_size']}`",
        f"- shared shuffle_batches: `{protocol['shared_shuffle_batches']}`",
        f"- current budget epochs: `{protocol['current_budget_reference']['epochs']}`",
        f"- longer budget epochs: `{protocol['longer_training_candidate']['epochs']}`",
        "",
        "## Decision",
        f"- `{STAGE05_V2_LONGER_TRAINING_DECISION_NAME}`: `{decision[STAGE05_V2_LONGER_TRAINING_DECISION_NAME]}`",
        f"- `{STAGE05_V2_LONGER_TRAINING_ACCURACY_DECISION_NAME}`: `{decision[STAGE05_V2_LONGER_TRAINING_ACCURACY_DECISION_NAME]}`",
        f"- longer budget still hits final training boundary on all seeds: `{decision['longer_budget_selection_hits_final_training_boundary_on_all_seeds']}`",
        f"- recommended next move: `{decision['recommended_next_move']}`",
        f"- rationale: `{decision['decision_rationale']}`",
        "",
        "## Supports",
    ]
    for item in report["supports"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Does Not Support"])
    for item in report["does_not_support"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _stage05_v2_longer_training_suite_config_payload(
    config: Stage05V2LongerTrainingValidationConfig,
) -> dict[str, Any]:
    return {
        "phase": "FMPC Stage 05 EF Core Probe",
        "stage": "stage05_v2_longer_training_validation",
        "comparison_protocol": _stage05_v2_longer_training_protocol_payload(config),
        "artifacts": {
            "aggregate_runs_csv": "aggregate_runs.csv",
            "aggregate_summary_json": "aggregate_summary.json",
            "comparison_report_json": "comparison_report.json",
            "comparison_report_md": "comparison_report.md",
        },
    }


def run_stage05_v2_longer_training_validation(
    config: Stage05V2LongerTrainingValidationConfig,
) -> FrozenBridgeVsCorrectedCoreComparisonRunResult:
    """Run a narrow longer-training validation on the existing Stage 05 v2 family."""

    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, config.resolved_run_id(), config.output_layout)
    )
    _write_json(run_dir / "config.json", _stage05_v2_longer_training_suite_config_payload(config))

    rows: list[dict[str, Any]] = []
    runs_root = run_dir / "runs"
    run_index = 0

    for seed in config.seeds:
        run_index += 1
        current_config = _stage05_v2_budget_config(
            config,
            seed=seed,
            output_root=runs_root,
            experiment_name=STAGE05_V2_CURRENT_BUDGET_METHOD_NAME,
            epochs=int(config.current_stage05_epochs),
        )
        current_result = run_fmpc_ef_exploratory_probe(current_config)
        rows.append(
            _stage05_core_row(
                run_index=run_index,
                suite_run_dir=run_dir,
                seed=seed,
                result=current_result,
                config=current_config,
                method_name=STAGE05_V2_CURRENT_BUDGET_METHOD_NAME,
                stage_name="FMPC Stage 05 EF Core Probe v2 Current Budget",
            )
        )

        run_index += 1
        longer_config = _stage05_v2_budget_config(
            config,
            seed=seed,
            output_root=runs_root,
            experiment_name=STAGE05_V2_LONGER_TRAINING_METHOD_NAME,
            epochs=int(config.longer_stage05_epochs),
        )
        longer_result = run_fmpc_ef_exploratory_probe(longer_config)
        rows.append(
            _stage05_core_row(
                run_index=run_index,
                suite_run_dir=run_dir,
                seed=seed,
                result=longer_result,
                config=longer_config,
                method_name=STAGE05_V2_LONGER_TRAINING_METHOD_NAME,
                stage_name="FMPC Stage 05 EF Core Probe v2 Longer Training",
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
            "selection_hits_final_training_boundary": str(
                bool(row["selection_hits_final_training_boundary"])
            ),
        }
        for row in rows
    ]
    _write_csv(run_dir / "aggregate_runs.csv", csv_rows)

    by_method = {
        STAGE05_V2_CURRENT_BUDGET_METHOD_NAME: _method_summary(
            _method_rows(rows, STAGE05_V2_CURRENT_BUDGET_METHOD_NAME)
        ),
        STAGE05_V2_LONGER_TRAINING_METHOD_NAME: _method_summary(
            _method_rows(rows, STAGE05_V2_LONGER_TRAINING_METHOD_NAME)
        ),
    }
    pairwise_longer_vs_current = _pairwise_summary(
        rows,
        candidate_method=STAGE05_V2_LONGER_TRAINING_METHOD_NAME,
        reference_method=STAGE05_V2_CURRENT_BUDGET_METHOD_NAME,
    )
    decision, decision_rationale = _stage05_v2_longer_training_decision(
        rows=rows,
        config=config,
        by_method=by_method,
    )

    summary = {
        "phase": "FMPC Stage 05 EF Core Probe",
        "stage": "stage05_v2_longer_training_validation",
        "num_runs": int(len(rows)),
        "comparison_protocol": _stage05_v2_longer_training_protocol_payload(config),
        "by_method": by_method,
        "pairwise_longer_budget_vs_current_budget": pairwise_longer_vs_current,
        **decision,
        "decision_rationale": decision_rationale,
        "aggregate_runs_csv_path": "aggregate_runs.csv",
        "comparison_report_json_path": "comparison_report.json",
        "comparison_report_md_path": "comparison_report.md",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)

    report = {
        "comparison_protocol": _stage05_v2_longer_training_protocol_payload(config),
        "decision": {
            **decision,
            "decision_rationale": decision_rationale,
        },
        "supports": _stage05_v2_longer_training_supports_lines(
            decision=decision,
            by_method=by_method,
        ),
        "does_not_support": _stage05_v2_longer_training_does_not_support_lines(
            decision=decision,
        ),
    }
    _write_json(run_dir / "comparison_report.json", report)
    _write_text(run_dir / "comparison_report.md", _stage05_v2_longer_training_report_markdown(report))

    return FrozenBridgeVsCorrectedCoreComparisonRunResult(
        run_dir=run_dir,
        config=_stage05_v2_longer_training_suite_config_payload(config),
        aggregate_rows=rows,
        summary=summary,
        comparison_report=report,
    )
