from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .benchmark_specs import (
    MLPTrainingSpec,
    PCTrainingSpec,
    ToyBenchmarkSpec,
    get_benchmark_spec,
    run_pc_benchmark,
)
from .experiment import OutputLayout
from .metrics import metric_higher_is_better
from .utils import set_seed

PHASE2C_BENCHMARK_NAMES = (
    "toy_regression",
    "toy_sine_regression",
)
PHASE2C_DEFAULT_SEED_VALUES: dict[str, tuple[int, ...]] = {
    "toy_regression": (0, 1, 2, 3, 4),
    "toy_sine_regression": (3, 4, 5, 6, 7),
}
PHASE2_TUNED_SOURCE_LEGACY = "phase2c_legacy"
PHASE2_TUNED_SOURCE_PHASE2F = "phase2f_joint_search"
PHASE2_TUNED_SOURCE_PHASE2G = "phase2g_matched_search"
PHASE2_TUNED_SOURCE_PHASE2G1 = "phase2g1_boundary_check"
PHASE2_MLP_SOURCE_DEFAULT = "benchmark_default"
PHASE2_MLP_SOURCE_PHASE2G = "phase2g_matched_search"
PHASE2_MLP_SOURCE_PHASE2G1 = "phase2g1_boundary_check"
PHASE2C_LEGACY_TUNED_PC_TRAINING: dict[str, PCTrainingSpec] = {
    "toy_regression": PCTrainingSpec(
        eta_x=0.2,
        eta_w=0.10,
        eta_b=0.10,
        train_steps=25,
        eval_steps=25,
        state_init="forward",
    ),
    "toy_sine_regression": PCTrainingSpec(
        eta_x=0.15,
        eta_w=0.06,
        eta_b=0.06,
        train_steps=30,
        eval_steps=30,
        state_init="forward",
    ),
}
TIE_RTOL = 1.0e-12
TIE_ATOL = 1.0e-12


@dataclass(frozen=True)
class TunedPCSelection:
    """Resolved tuned-PC configuration source for refreshed downstream studies."""

    source: str
    name: str
    pc_training: PCTrainingSpec
    epochs: int
    selection_artifact_path: str | None
    selection_config_id: str | None
    selection_run_id: str | None
    selection_val_metric: float | None
    selection_test_metric: float | None

    @property
    def selection_eval_metric(self) -> float | None:
        return self.selection_val_metric


@dataclass(frozen=True)
class MLPSelection:
    """Resolved MLP configuration source for refreshed downstream studies."""

    source: str
    name: str
    mlp_training: MLPTrainingSpec
    epochs: int
    selection_artifact_path: str | None
    selection_config_id: str | None
    selection_run_id: str | None
    selection_val_metric: float | None
    selection_test_metric: float | None

    @property
    def selection_eval_metric(self) -> float | None:
        return self.selection_val_metric


@dataclass
class PCMultiSeedRunResult:
    """Materialized outputs of one Phase 2c multi-seed study."""

    run_dir: Path
    study_config: dict[str, Any]
    seed_records: list[dict[str, Any]]
    aggregate_summary: dict[str, Any]


def _resolve_run_id(run_id: str | None) -> str:
    if run_id is not None:
        return run_id
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _multiseed_experiment_name(
    benchmark_name: str,
    *,
    tuned_source: str,
    mlp_source: str,
) -> str:
    if (
        tuned_source == PHASE2_TUNED_SOURCE_LEGACY
        and mlp_source == PHASE2_MLP_SOURCE_DEFAULT
    ):
        return f"pc_multiseed_{benchmark_name}"
    if (
        tuned_source == PHASE2_TUNED_SOURCE_PHASE2F
        and mlp_source == PHASE2_MLP_SOURCE_DEFAULT
    ):
        return f"pc_multiseed_phase2f_{benchmark_name}"
    if (
        tuned_source == PHASE2_TUNED_SOURCE_PHASE2G
        and mlp_source == PHASE2_MLP_SOURCE_PHASE2G
    ):
        return f"pc_multiseed_phase2g_{benchmark_name}"
    if (
        tuned_source == PHASE2_TUNED_SOURCE_PHASE2G1
        and mlp_source == PHASE2_MLP_SOURCE_PHASE2G1
    ):
        return f"pc_multiseed_phase2g1_{benchmark_name}"
    return f"pc_multiseed_custom_{benchmark_name}"


def _multiseed_phase_label(*, tuned_source: str, mlp_source: str) -> str:
    if (
        tuned_source == PHASE2_TUNED_SOURCE_LEGACY
        and mlp_source == PHASE2_MLP_SOURCE_DEFAULT
    ):
        return "Phase 2c"
    if (
        tuned_source == PHASE2_TUNED_SOURCE_PHASE2F
        and mlp_source == PHASE2_MLP_SOURCE_DEFAULT
    ):
        return "Phase 2c-refresh"
    if (
        tuned_source == PHASE2_TUNED_SOURCE_PHASE2G
        and mlp_source == PHASE2_MLP_SOURCE_PHASE2G
    ):
        return "Phase 2g-refresh"
    if (
        tuned_source == PHASE2_TUNED_SOURCE_PHASE2G1
        and mlp_source == PHASE2_MLP_SOURCE_PHASE2G1
    ):
        return "Phase 2g.1-refresh"
    return "Phase 2-refresh-custom"


def _resolve_multiseed_root(
    output_root: str | Path,
    benchmark_name: str,
    run_id: str,
    output_layout: OutputLayout,
    tuned_source: str,
    mlp_source: str,
) -> Path:
    experiment_name = _multiseed_experiment_name(
        benchmark_name,
        tuned_source=tuned_source,
        mlp_source=mlp_source,
    )
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


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, indent=2)


def _write_seed_records(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "seed_index",
        "run_seed",
        "data_seed",
        "model_init_seed",
        "default_pc_status",
        "default_pc_failure_reason",
        "default_pc_primary_metric_value",
        "default_pc_final_pre_update_energy",
        "default_pc_summary_path",
        "tuned_pc_status",
        "tuned_pc_failure_reason",
        "tuned_pc_primary_metric_value",
        "tuned_pc_final_pre_update_energy",
        "tuned_pc_summary_path",
        "mlp_status",
        "mlp_failure_reason",
        "mlp_primary_metric_value",
        "mlp_final_loss",
        "mlp_summary_path",
        "primary_metric_delta_tuned_pc_minus_default_pc",
        "primary_metric_delta_mlp_minus_tuned_pc",
        "tuned_pc_beats_default_pc",
        "tuned_pc_beats_mlp",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_mlp_epoch_metrics(
    path: Path,
    rows: list[dict[str, Any]],
    num_layers: int,
    task_name: str,
) -> None:
    fieldnames = ["epoch", "loss"]
    fieldnames.extend([f"weight_norm_l{layer_index}" for layer_index in range(1, num_layers + 1)])
    fieldnames.extend([f"bias_norm_l{layer_index}" for layer_index in range(1, num_layers + 1)])
    if task_name == "regression":
        fieldnames.extend(
            [
                "train_mse",
                "val_mse",
                "train_baseline_mse",
                "val_baseline_mse",
                "mse",
                "baseline_mse",
            ]
        )
    elif task_name == "classification":
        fieldnames.extend(
            [
                "train_accuracy",
                "val_accuracy",
                "train_baseline_accuracy",
                "val_baseline_accuracy",
                "accuracy",
                "baseline_accuracy",
            ]
        )
    else:
        raise ValueError(f"Unsupported task '{task_name}'.")

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _training_dict(training: PCTrainingSpec, *, epochs: int) -> dict[str, Any]:
    return {
        "epochs": epochs,
        "eta_x": training.eta_x,
        "eta_w": training.eta_w,
        "eta_b": training.eta_b,
        "train_steps": training.train_steps,
        "eval_steps": training.eval_steps,
        "state_init": training.state_init,
    }


def _mlp_training_dict(training: MLPTrainingSpec, *, epochs: int) -> dict[str, Any]:
    return {
        "epochs": epochs,
        "eta_w": training.eta_w,
        "eta_b": training.eta_b,
    }


def get_phase2c_tuned_pc_training(benchmark_name: str) -> PCTrainingSpec:
    """Return the fixed legacy Phase 2c tuned PC config for one regression benchmark."""
    try:
        return PHASE2C_LEGACY_TUNED_PC_TRAINING[benchmark_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported Phase 2c benchmark '{benchmark_name}'.") from exc


def _phase2f_best_config_summary_path(
    joint_search_output_root: str | Path,
    benchmark_name: str,
) -> Path:
    return (
        Path(joint_search_output_root)
        / "phase2f_joint_search"
        / benchmark_name
        / "best_config_summary.json"
    )


def _phase2g_best_pc_config_summary_path(
    joint_search_output_root: str | Path,
    benchmark_name: str,
) -> Path:
    return (
        Path(joint_search_output_root)
        / "phase2g_matched_search"
        / benchmark_name
        / "best_pc_config_summary.json"
    )


def _phase2g_best_mlp_config_summary_path(
    joint_search_output_root: str | Path,
    benchmark_name: str,
) -> Path:
    return (
        Path(joint_search_output_root)
        / "phase2g_matched_search"
        / benchmark_name
        / "best_mlp_config_summary.json"
    )


def _phase2g1_best_pc_config_summary_path(
    joint_search_output_root: str | Path,
    benchmark_name: str,
) -> Path:
    return (
        Path(joint_search_output_root)
        / "phase2g1_boundary_check"
        / benchmark_name
        / "best_pc_config_summary.json"
    )


def _phase2g1_best_mlp_config_summary_path(
    joint_search_output_root: str | Path,
    benchmark_name: str,
) -> Path:
    return (
        Path(joint_search_output_root)
        / "phase2g1_boundary_check"
        / benchmark_name
        / "best_mlp_config_summary.json"
    )


def _selection_config_id(payload: dict[str, Any]) -> str:
    value = payload.get("best_config_id", payload.get("boundary_check_best_config_id"))
    if value is None:
        raise KeyError("Selection payload is missing a best-config id.")
    return str(value)


def _selection_best_config(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("best_config", payload.get("boundary_check_best_config"))
    if value is None:
        raise KeyError("Selection payload is missing a best-config body.")
    return dict(value)


def _selection_val_metric_value(payload: dict[str, Any]) -> float | None:
    value = payload.get(
        "selection_metric_value",
        payload.get(
            "boundary_check_val_metric",
            payload.get("val_metric", payload.get("eval_metric")),
        ),
    )
    return None if value is None else float(value)


def _selection_test_metric_value(payload: dict[str, Any]) -> float | None:
    value = payload.get(
        "report_metric_value",
        payload.get("boundary_check_test_metric", payload.get("test_metric")),
    )
    return None if value is None else float(value)


def _combined_config_source(*, tuned_source: str, mlp_source: str) -> str:
    if tuned_source == mlp_source:
        return tuned_source
    return "mixed_config_sources"


def resolve_tuned_pc_selection(
    benchmark_name: str,
    *,
    tuned_source: str = PHASE2_TUNED_SOURCE_PHASE2F,
    joint_search_output_root: str | Path = "outputs",
) -> TunedPCSelection:
    """Return the tuned-PC config for either legacy or Phase 2f-refreshed studies."""
    if tuned_source == PHASE2_TUNED_SOURCE_LEGACY:
        return TunedPCSelection(
            source=tuned_source,
            name="eta_w_double",
            pc_training=get_phase2c_tuned_pc_training(benchmark_name),
            epochs=get_benchmark_spec(benchmark_name).epochs,
            selection_artifact_path=None,
            selection_config_id=None,
            selection_run_id=None,
            selection_val_metric=None,
            selection_test_metric=None,
        )

    if tuned_source == PHASE2_TUNED_SOURCE_PHASE2F:
        best_config_path = _phase2f_best_config_summary_path(
            joint_search_output_root,
            benchmark_name,
        )
    elif tuned_source == PHASE2_TUNED_SOURCE_PHASE2G:
        best_config_path = _phase2g_best_pc_config_summary_path(
            joint_search_output_root,
            benchmark_name,
        )
    elif tuned_source == PHASE2_TUNED_SOURCE_PHASE2G1:
        best_config_path = _phase2g1_best_pc_config_summary_path(
            joint_search_output_root,
            benchmark_name,
        )
    else:
        raise ValueError(f"Unsupported tuned_source '{tuned_source}'.")

    if not best_config_path.exists():
        raise FileNotFoundError(
            f"Selected tuned-PC summary was not found at "
            f"'{best_config_path}'. Run the corresponding Phase 2 search first or pass "
            "`joint_search_output_root` explicitly."
        )

    with best_config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    best_config_id = _selection_config_id(payload)
    best_config = _selection_best_config(payload)
    return TunedPCSelection(
        source=tuned_source,
        name=best_config_id,
        pc_training=PCTrainingSpec(
            eta_x=float(best_config["eta_x"]),
            eta_w=float(best_config["eta_w"]),
            eta_b=float(best_config["eta_b"]),
            train_steps=int(best_config["train_steps"]),
            eval_steps=int(best_config["eval_steps"]),
            state_init=str(best_config["state_init"]),
        ),
        epochs=int(best_config["epochs"]),
        selection_artifact_path=str(best_config_path),
        selection_config_id=best_config_id,
        selection_run_id=str(payload["run_id"]),
        selection_val_metric=_selection_val_metric_value(payload),
        selection_test_metric=_selection_test_metric_value(payload),
    )


def resolve_mlp_selection(
    benchmark_name: str,
    *,
    mlp_source: str = PHASE2_MLP_SOURCE_DEFAULT,
    joint_search_output_root: str | Path = "outputs",
) -> MLPSelection:
    """Return the MLP config for either benchmark-default or Phase 2g-refreshed studies."""
    base_spec = get_benchmark_spec(benchmark_name)
    if mlp_source == PHASE2_MLP_SOURCE_DEFAULT:
        return MLPSelection(
            source=mlp_source,
            name="benchmark_default",
            mlp_training=base_spec.mlp_training,
            epochs=base_spec.epochs,
            selection_artifact_path=None,
            selection_config_id=None,
            selection_run_id=None,
            selection_val_metric=None,
            selection_test_metric=None,
        )

    if mlp_source == PHASE2_MLP_SOURCE_PHASE2G:
        best_config_path = _phase2g_best_mlp_config_summary_path(
            joint_search_output_root,
            benchmark_name,
        )
    elif mlp_source == PHASE2_MLP_SOURCE_PHASE2G1:
        best_config_path = _phase2g1_best_mlp_config_summary_path(
            joint_search_output_root,
            benchmark_name,
        )
    else:
        raise ValueError(f"Unsupported mlp_source '{mlp_source}'.")
    if not best_config_path.exists():
        raise FileNotFoundError(
            "Selected MLP summary was not found at "
            f"'{best_config_path}'. Run the Phase 2g matched search first or pass "
            "`joint_search_output_root` explicitly."
        )

    with best_config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    best_config_id = _selection_config_id(payload)
    best_config = _selection_best_config(payload)
    return MLPSelection(
        source=mlp_source,
        name=best_config_id,
        mlp_training=MLPTrainingSpec(
            eta_w=float(best_config["eta_w"]),
            eta_b=float(best_config["eta_b"]),
        ),
        epochs=int(best_config["epochs"]),
        selection_artifact_path=str(best_config_path),
        selection_config_id=best_config_id,
        selection_run_id=str(payload["run_id"]),
        selection_val_metric=_selection_val_metric_value(payload),
        selection_test_metric=_selection_test_metric_value(payload),
    )


def default_seed_values_for_benchmark(benchmark_name: str) -> tuple[int, ...]:
    """Return the fixed default seed set for one Phase 2c benchmark."""
    try:
        return PHASE2C_DEFAULT_SEED_VALUES[benchmark_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported Phase 2c benchmark '{benchmark_name}'.") from exc


def _seeded_spec(
    base_spec: ToyBenchmarkSpec,
    *,
    seed: int,
    pc_training: PCTrainingSpec | None = None,
    mlp_training: MLPTrainingSpec | None = None,
    epochs: int | None = None,
) -> ToyBenchmarkSpec:
    return replace(
        base_spec,
        run_seed=seed,
        model_init_seed=seed,
        data_seed=base_spec.data_seed,
        pc_training=base_spec.pc_training if pc_training is None else pc_training,
        mlp_training=base_spec.mlp_training if mlp_training is None else mlp_training,
        epochs=base_spec.epochs if epochs is None else epochs,
    )


def _metric_value_is_better(metric_name: str, left_value: float, right_value: float) -> bool:
    if np.isclose(left_value, right_value, rtol=TIE_RTOL, atol=TIE_ATOL):
        return False
    if metric_higher_is_better(metric_name):
        return left_value > right_value
    return left_value < right_value


def _compare_metric_values(metric_name: str, left_value: float, right_value: float) -> str:
    if np.isclose(left_value, right_value, rtol=TIE_RTOL, atol=TIE_ATOL):
        return "tie"
    if metric_higher_is_better(metric_name):
        return "left" if left_value > right_value else "right"
    return "left" if left_value < right_value else "right"


def _headline_test_comparison(
    *,
    metric_name: str,
    pc_value: float | None,
    mlp_value: float | None,
) -> dict[str, Any]:
    if pc_value is None or mlp_value is None:
        return {
            "headline_test_comparison_target": "selected_pc_vs_selected_mlp",
            "headline_test_comparison_split": "test",
            "headline_test_winner": None,
            "headline_test_winner_reason": None,
            "headline_test_pc_metric_mean": pc_value,
            "headline_test_mlp_metric_mean": mlp_value,
            "headline_test_metric_difference_mlp_minus_pc": None,
            "headline_test_pc_beats_mlp": None,
        }

    comparison = _compare_metric_values(metric_name, pc_value, mlp_value)
    if comparison == "left":
        winner = "pc"
    elif comparison == "right":
        winner = "mlp"
    else:
        winner = "tie"

    if comparison == "tie":
        reason = "metrics_equal_within_tolerance"
    elif metric_higher_is_better(metric_name):
        reason = f"higher_is_better: {winner}_metric_mean is larger on held-out test"
    else:
        reason = f"lower_is_better: {winner}_metric_mean is smaller on held-out test"

    return {
        "headline_test_comparison_target": "selected_pc_vs_selected_mlp",
        "headline_test_comparison_split": "test",
        "headline_test_winner": winner,
        "headline_test_winner_reason": reason,
        "headline_test_pc_metric_mean": pc_value,
        "headline_test_mlp_metric_mean": mlp_value,
        "headline_test_metric_difference_mlp_minus_pc": mlp_value - pc_value,
        "headline_test_pc_beats_mlp": winner == "pc",
    }


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _std_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=0))


def _mlp_config_payload(
    spec: ToyBenchmarkSpec,
    run_id: str,
    output_root: str | Path,
    output_layout: OutputLayout,
    split,
) -> dict[str, Any]:
    return {
        "experiment_name": "mlp",
        "run_id": run_id,
        "seed": spec.run_seed,
        "run_seed": spec.run_seed,
        "data_seed": spec.data_seed,
        "model_init_seed": spec.model_init_seed,
        "seeds": {
            "run_seed": spec.run_seed,
            "data_seed": spec.data_seed,
            "model_init_seed": spec.model_init_seed,
        },
        "task": spec.task_config(),
        "data": spec.data_config(split),
        "model": spec.mlp_model_config(),
        "training": {
            "epochs": spec.epochs,
            "eta_w": spec.mlp_training.eta_w,
            "eta_b": spec.mlp_training.eta_b,
            "loss_name": "mse",
            "run_seed": spec.run_seed,
        },
        "logging": {
            "output_root": str(output_root),
            "output_layout": output_layout,
            "plot_energy": False,
            "trace_policy": "not_applicable",
        },
    }


def _run_mlp_variant(
    spec: ToyBenchmarkSpec,
    *,
    variant_dir: Path,
    run_id: str,
    output_layout: OutputLayout,
    split,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    variant_dir.mkdir(parents=True, exist_ok=True)
    set_seed(spec.run_seed)
    model = spec.build_mlp_model()
    config_payload = _mlp_config_payload(spec, run_id, variant_dir.parent, output_layout, split)
    _write_json(variant_dir / "config.json", config_payload)

    x_train = split.x_train
    y_train = split.y_train
    x_val = split.x_val
    y_val = split.y_val
    x_test = split.x_test
    y_test = split.y_test
    train_baseline_metric_value = spec.baseline_metric_fn(y_train)
    val_baseline_metric_value = spec.baseline_metric_fn(y_val)
    test_baseline_metric_value = spec.baseline_metric_fn(y_test)
    epoch_rows: list[dict[str, Any]] = []

    for epoch in range(1, spec.epochs + 1):
        batch_result = model.train_batch(x_train, y_train)
        train_predictions = model.predict(x_train)
        val_predictions = model.predict(x_val)
        train_metric_value = spec.primary_metric_fn(train_predictions, y_train)
        val_metric_value = spec.primary_metric_fn(val_predictions, y_val)
        row: dict[str, Any] = {
            "epoch": epoch,
            "loss": batch_result.loss,
        }
        for layer_index, value in enumerate(batch_result.parameter_norms["weight_norms"], start=1):
            row[f"weight_norm_l{layer_index}"] = value
        for layer_index, value in enumerate(batch_result.parameter_norms["bias_norms"], start=1):
            row[f"bias_norm_l{layer_index}"] = value
        if spec.task_name == "regression":
            row["train_mse"] = train_metric_value
            row["val_mse"] = val_metric_value
            row["train_baseline_mse"] = train_baseline_metric_value
            row["val_baseline_mse"] = val_baseline_metric_value
            row["mse"] = val_metric_value
            row["baseline_mse"] = val_baseline_metric_value
        elif spec.task_name == "classification":
            row["train_accuracy"] = train_metric_value
            row["val_accuracy"] = val_metric_value
            row["train_baseline_accuracy"] = train_baseline_metric_value
            row["val_baseline_accuracy"] = val_baseline_metric_value
            row["accuracy"] = val_metric_value
            row["baseline_accuracy"] = val_baseline_metric_value
        else:
            raise ValueError(f"Unsupported task '{spec.task_name}'.")
        epoch_rows.append(row)

    test_predictions = model.predict(x_test)
    test_metric_value = spec.primary_metric_fn(test_predictions, y_test)
    best_row = (
        max(epoch_rows, key=lambda row: row[spec.primary_metric_name])
        if spec.primary_metric_higher_is_better
        else min(epoch_rows, key=lambda row: row[spec.primary_metric_name])
    )
    final_row = epoch_rows[-1]
    summary = {
        "experiment_name": "mlp",
        "run_id": run_id,
        "phase": "Phase 2c",
        "model_family": "mlp",
        "seed": spec.run_seed,
        "run_seed": spec.run_seed,
        "data_seed": spec.data_seed,
        "model_init_seed": spec.model_init_seed,
        "seeds": {
            "run_seed": spec.run_seed,
            "data_seed": spec.data_seed,
            "model_init_seed": spec.model_init_seed,
        },
        "final_epoch": final_row["epoch"],
        "final_loss": final_row["loss"],
        "metric_name": spec.primary_metric_name,
        "metric_higher_is_better": spec.primary_metric_higher_is_better,
        "train_metric": final_row[f"train_{spec.primary_metric_name}"],
        "val_metric": final_row[f"val_{spec.primary_metric_name}"],
        "test_metric": test_metric_value,
        "eval_metric": final_row[f"val_{spec.primary_metric_name}"],
        "primary_metric_name": spec.primary_metric_name,
        "primary_metric_value": test_metric_value,
        "primary_metric_higher_is_better": spec.primary_metric_higher_is_better,
        "selection_metric_source": "val_metric",
        "selection_metric_value": final_row[f"val_{spec.primary_metric_name}"],
        "report_metric_source": "test_metric",
        "report_metric_value": test_metric_value,
        "baseline_metric_name": spec.baseline_metric_name,
        "train_baseline_metric": train_baseline_metric_value,
        "val_baseline_metric": val_baseline_metric_value,
        "test_baseline_metric": test_baseline_metric_value,
        "eval_baseline_metric": val_baseline_metric_value,
        "baseline_metric_value": test_baseline_metric_value,
        "best_epoch": best_row["epoch"],
        "best_val_metric": best_row[f"val_{spec.primary_metric_name}"],
    }
    _write_mlp_epoch_metrics(
        variant_dir / "epoch_metrics.csv",
        epoch_rows,
        num_layers=len(model.layers),
        task_name=spec.task_name,
    )
    _write_json(variant_dir / "summary.json", summary)
    return variant_dir, epoch_rows, summary


def _clean_failed_variant_dir(variant_dir: Path) -> None:
    if variant_dir.exists():
        shutil.rmtree(variant_dir)
    variant_dir.mkdir(parents=True, exist_ok=True)


def _failure_reason(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _default_success(row: dict[str, Any]) -> bool:
    return row["default_pc_status"] == "ok"


def _tuned_success(row: dict[str, Any]) -> bool:
    return row["tuned_pc_status"] == "ok"


def _mlp_success(row: dict[str, Any]) -> bool:
    return row["mlp_status"] == "ok"


def _paired_default_vs_tuned(row: dict[str, Any]) -> bool:
    return _default_success(row) and _tuned_success(row)


def _paired_tuned_vs_mlp(row: dict[str, Any]) -> bool:
    return _tuned_success(row) and _mlp_success(row)


def _build_study_config(
    base_spec: ToyBenchmarkSpec,
    *,
    run_id: str,
    output_layout: OutputLayout,
    seed_values: Sequence[int],
    tuned_selection: TunedPCSelection,
    mlp_selection: MLPSelection,
    split,
) -> dict[str, Any]:
    tuned_training = tuned_selection.pc_training
    experiment_name = _multiseed_experiment_name(
        base_spec.benchmark_name,
        tuned_source=tuned_selection.source,
        mlp_source=mlp_selection.source,
    )
    return {
        "experiment_name": experiment_name,
        "run_id": run_id,
        "phase": _multiseed_phase_label(
            tuned_source=tuned_selection.source,
            mlp_source=mlp_selection.source,
        ),
        "benchmark_name": base_spec.benchmark_name,
        "task_name": base_spec.task_name,
        "config_source": _combined_config_source(
            tuned_source=tuned_selection.source,
            mlp_source=mlp_selection.source,
        ),
        "selected_by_metric_source": "val_metric",
        "final_report_metric_source": "test_metric",
        "seed_values": [int(seed) for seed in seed_values],
        "seed_semantics": {
            "data_seed": "fixed",
            "run_seed": "varies_with_seed",
            "model_init_seed": "varies_with_seed",
            "notes": "This phase mainly measures initialization stability, not dataset sampling variability.",
        },
        "data_seed_fixed": base_spec.data_seed,
        "data": base_spec.data_config(split),
        "default_pc_training": _training_dict(base_spec.pc_training, epochs=base_spec.epochs),
        "tuned_pc_source": tuned_selection.source,
        "tuned_pc_preset_name": tuned_selection.name,
        "tuned_pc_selection_config_id": tuned_selection.selection_config_id,
        "tuned_pc_selection_run_id": tuned_selection.selection_run_id,
        "tuned_pc_selection_val_metric": tuned_selection.selection_val_metric,
        "tuned_pc_selection_test_metric": tuned_selection.selection_test_metric,
        "tuned_pc_selection_eval_metric": tuned_selection.selection_eval_metric,
        "tuned_pc_selection_artifact_path": tuned_selection.selection_artifact_path,
        "tuned_pc_training": _training_dict(tuned_training, epochs=tuned_selection.epochs),
        "mlp_source": mlp_selection.source,
        "mlp_preset_name": mlp_selection.name,
        "mlp_selection_config_id": mlp_selection.selection_config_id,
        "mlp_selection_run_id": mlp_selection.selection_run_id,
        "mlp_selection_val_metric": mlp_selection.selection_val_metric,
        "mlp_selection_test_metric": mlp_selection.selection_test_metric,
        "mlp_selection_artifact_path": mlp_selection.selection_artifact_path,
        "mlp_training": _mlp_training_dict(mlp_selection.mlp_training, epochs=mlp_selection.epochs),
        "output_layout": output_layout,
    }


def _build_aggregate_summary(
    *,
    base_spec: ToyBenchmarkSpec,
    run_id: str,
    seed_values: Sequence[int],
    seed_records: list[dict[str, Any]],
    tuned_selection: TunedPCSelection,
    mlp_selection: MLPSelection,
) -> dict[str, Any]:
    default_values = [
        float(row["default_pc_primary_metric_value"])
        for row in seed_records
        if row["default_pc_primary_metric_value"] is not None
    ]
    tuned_values = [
        float(row["tuned_pc_primary_metric_value"])
        for row in seed_records
        if row["tuned_pc_primary_metric_value"] is not None
    ]
    mlp_values = [
        float(row["mlp_primary_metric_value"])
        for row in seed_records
        if row["mlp_primary_metric_value"] is not None
    ]
    default_vs_tuned_deltas = [
        float(row["primary_metric_delta_tuned_pc_minus_default_pc"])
        for row in seed_records
        if row["primary_metric_delta_tuned_pc_minus_default_pc"] is not None
    ]
    tuned_vs_mlp_deltas = [
        float(row["primary_metric_delta_mlp_minus_tuned_pc"])
        for row in seed_records
        if row["primary_metric_delta_mlp_minus_tuned_pc"] is not None
    ]

    tuned_better_than_default = 0
    default_better_than_tuned = 0
    tuned_default_ties = 0
    tuned_better_than_mlp = 0
    mlp_better_than_tuned = 0
    tuned_mlp_ties = 0

    for row in seed_records:
        if _paired_default_vs_tuned(row):
            comparison = _compare_metric_values(
                base_spec.primary_metric_name,
                float(row["tuned_pc_primary_metric_value"]),
                float(row["default_pc_primary_metric_value"]),
            )
            if comparison == "left":
                tuned_better_than_default += 1
            elif comparison == "right":
                default_better_than_tuned += 1
            else:
                tuned_default_ties += 1

        if _paired_tuned_vs_mlp(row):
            comparison = _compare_metric_values(
                base_spec.primary_metric_name,
                float(row["tuned_pc_primary_metric_value"]),
                float(row["mlp_primary_metric_value"]),
            )
            if comparison == "left":
                tuned_better_than_mlp += 1
            elif comparison == "right":
                mlp_better_than_tuned += 1
            else:
                tuned_mlp_ties += 1

    default_mean = _mean_or_none(default_values)
    tuned_mean = _mean_or_none(tuned_values)
    mlp_mean = _mean_or_none(mlp_values)
    headline_test = _headline_test_comparison(
        metric_name=base_spec.primary_metric_name,
        pc_value=tuned_mean,
        mlp_value=mlp_mean,
    )

    return {
        "experiment_name": _multiseed_experiment_name(
            base_spec.benchmark_name,
            tuned_source=tuned_selection.source,
            mlp_source=mlp_selection.source,
        ),
        "run_id": run_id,
        "phase": _multiseed_phase_label(
            tuned_source=tuned_selection.source,
            mlp_source=mlp_selection.source,
        ),
        "benchmark_name": base_spec.benchmark_name,
        "task_name": base_spec.task_name,
        "primary_metric_name": base_spec.primary_metric_name,
        "primary_metric_higher_is_better": base_spec.primary_metric_higher_is_better,
        "config_source": _combined_config_source(
            tuned_source=tuned_selection.source,
            mlp_source=mlp_selection.source,
        ),
        "selected_by_metric_source": "val_metric",
        "final_report_metric_source": "test_metric",
        "seed_values": [int(seed) for seed in seed_values],
        "planned_seed_count": len(seed_values),
        "default_pc_success_count": sum(1 for row in seed_records if _default_success(row)),
        "tuned_pc_success_count": sum(1 for row in seed_records if _tuned_success(row)),
        "mlp_success_count": sum(1 for row in seed_records if _mlp_success(row)),
        "paired_default_vs_tuned_count": sum(
            1 for row in seed_records if _paired_default_vs_tuned(row)
        ),
        "paired_tuned_vs_mlp_count": sum(1 for row in seed_records if _paired_tuned_vs_mlp(row)),
        "default_pc_primary_metric_mean": default_mean,
        "default_pc_primary_metric_std": _std_or_none(default_values),
        "tuned_pc_primary_metric_mean": tuned_mean,
        "tuned_pc_primary_metric_std": _std_or_none(tuned_values),
        "mlp_primary_metric_mean": mlp_mean,
        "mlp_primary_metric_std": _std_or_none(mlp_values),
        "primary_metric_delta_tuned_pc_minus_default_pc_mean": _mean_or_none(default_vs_tuned_deltas),
        "primary_metric_delta_tuned_pc_minus_default_pc_std": _std_or_none(default_vs_tuned_deltas),
        "primary_metric_delta_mlp_minus_tuned_pc_mean": _mean_or_none(tuned_vs_mlp_deltas),
        "primary_metric_delta_mlp_minus_tuned_pc_std": _std_or_none(tuned_vs_mlp_deltas),
        "tuned_pc_better_than_default_pc_seed_count": tuned_better_than_default,
        "default_pc_better_than_tuned_pc_seed_count": default_better_than_tuned,
        "tuned_pc_vs_default_pc_tie_seed_count": tuned_default_ties,
        "tuned_pc_better_than_mlp_seed_count": tuned_better_than_mlp,
        "mlp_better_than_tuned_pc_seed_count": mlp_better_than_tuned,
        "tuned_pc_vs_mlp_tie_seed_count": tuned_mlp_ties,
        "tuned_pc_mean_beats_default_pc": (
            None
            if tuned_mean is None or default_mean is None
            else _metric_value_is_better(base_spec.primary_metric_name, tuned_mean, default_mean)
        ),
        "tuned_pc_mean_beats_mlp": (
            None
            if tuned_mean is None or mlp_mean is None
            else _metric_value_is_better(base_spec.primary_metric_name, tuned_mean, mlp_mean)
        ),
        "tuned_pc_source": tuned_selection.source,
        "tuned_pc_preset_name": tuned_selection.name,
        "selected_pc_config": _training_dict(tuned_selection.pc_training, epochs=tuned_selection.epochs),
        "tuned_pc_selection_config_id": tuned_selection.selection_config_id,
        "tuned_pc_selection_run_id": tuned_selection.selection_run_id,
        "tuned_pc_selection_val_metric": tuned_selection.selection_val_metric,
        "tuned_pc_selection_test_metric": tuned_selection.selection_test_metric,
        "tuned_pc_selection_eval_metric": tuned_selection.selection_eval_metric,
        "tuned_pc_selection_artifact_path": tuned_selection.selection_artifact_path,
        "mlp_source": mlp_selection.source,
        "mlp_preset_name": mlp_selection.name,
        "selected_mlp_config": _mlp_training_dict(mlp_selection.mlp_training, epochs=mlp_selection.epochs),
        "mlp_selection_config_id": mlp_selection.selection_config_id,
        "mlp_selection_run_id": mlp_selection.selection_run_id,
        "mlp_selection_val_metric": mlp_selection.selection_val_metric,
        "mlp_selection_test_metric": mlp_selection.selection_test_metric,
        "mlp_selection_artifact_path": mlp_selection.selection_artifact_path,
        "selection_split": "validation",
        "final_report_split": "test",
        "tie_rtol": TIE_RTOL,
        "tie_atol": TIE_ATOL,
        **headline_test,
        "notes": {
            "seed_semantics": "data_seed stays fixed while run_seed and model_init_seed vary together.",
            "interpretation": "This phase mainly measures initialization stability rather than dataset sampling variability.",
            "primary_metric_delta_mlp_minus_tuned_pc": "mlp_primary_metric_value - tuned_pc_primary_metric_value",
            "refresh": (
                "This run uses the legacy Phase 2c tuned preset and the benchmark-default MLP config."
                if (
                    tuned_selection.source == PHASE2_TUNED_SOURCE_LEGACY
                    and mlp_selection.source == PHASE2_MLP_SOURCE_DEFAULT
                )
                else (
                    "This run uses the best eval-ranked Phase 2f joint-search PC config and the benchmark-default MLP config."
                    if (
                        tuned_selection.source == PHASE2_TUNED_SOURCE_PHASE2F
                        and mlp_selection.source == PHASE2_MLP_SOURCE_DEFAULT
                    )
                    else (
                        "This run uses the Phase 2g matched-search selected PC and MLP configs."
                        if (
                            tuned_selection.source == PHASE2_TUNED_SOURCE_PHASE2G
                            and mlp_selection.source == PHASE2_MLP_SOURCE_PHASE2G
                        )
                        else (
                            "This run uses the Phase 2g.1 boundary-check refined PC and MLP configs, with final comparisons reported on held-out test metrics."
                            if (
                                tuned_selection.source == PHASE2_TUNED_SOURCE_PHASE2G1
                                and mlp_selection.source == PHASE2_MLP_SOURCE_PHASE2G1
                            )
                            else "This run uses a custom combination of selected PC/MLP configs."
                        )
                    )
                )
            ),
        },
    }


def _plot_multiseed_summary(
    run_dir: Path,
    base_spec: ToyBenchmarkSpec,
    seed_records: list[dict[str, Any]],
    aggregate_summary: dict[str, Any],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "plot_summary=True requires matplotlib to be installed in the current environment."
        ) from exc

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    seed_labels = [str(row["run_seed"]) for row in seed_records]
    default_values = [
        np.nan if row["default_pc_primary_metric_value"] is None else float(row["default_pc_primary_metric_value"])
        for row in seed_records
    ]
    tuned_values = [
        np.nan if row["tuned_pc_primary_metric_value"] is None else float(row["tuned_pc_primary_metric_value"])
        for row in seed_records
    ]
    mlp_values = [
        np.nan if row["mlp_primary_metric_value"] is None else float(row["mlp_primary_metric_value"])
        for row in seed_records
    ]

    figure = plt.figure(figsize=(10, 4))
    axis = figure.add_subplot(1, 1, 1)
    axis.plot(seed_labels, default_values, marker="o", label="default_pc")
    axis.plot(seed_labels, tuned_values, marker="o", label="tuned_pc")
    axis.plot(seed_labels, mlp_values, marker="o", label="mlp")
    axis.set_xlabel("Seed")
    axis.set_ylabel(base_spec.primary_metric_name)
    axis.set_title(base_spec.benchmark_name)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots_dir / "primary_metric_by_seed.png")
    plt.close(figure)

    labels = ["default_pc", "tuned_pc", "mlp"]
    means = [
        aggregate_summary["default_pc_primary_metric_mean"],
        aggregate_summary["tuned_pc_primary_metric_mean"],
        aggregate_summary["mlp_primary_metric_mean"],
    ]
    stds = [
        aggregate_summary["default_pc_primary_metric_std"],
        aggregate_summary["tuned_pc_primary_metric_std"],
        aggregate_summary["mlp_primary_metric_std"],
    ]
    figure = plt.figure(figsize=(8, 4))
    axis = figure.add_subplot(1, 1, 1)
    axis.bar(labels, means, yerr=stds, capsize=4)
    axis.set_ylabel(base_spec.primary_metric_name)
    axis.set_title(f"{base_spec.benchmark_name} mean ± std")
    figure.tight_layout()
    figure.savefig(plots_dir / "variant_mean_std.png")
    plt.close(figure)


def run_pc_multiseed_study(
    benchmark_name: str,
    *,
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    output_layout: OutputLayout = "single_dir",
    plot_energy: bool = False,
    plot_summary: bool = False,
    seed_values: Sequence[int] | None = None,
    tuned_source: str = PHASE2_TUNED_SOURCE_PHASE2G1,
    mlp_source: str | None = None,
    joint_search_output_root: str | Path = "outputs",
) -> PCMultiSeedRunResult:
    """Run the fixed Phase 2c multi-seed study for one regression benchmark."""
    if benchmark_name not in PHASE2C_BENCHMARK_NAMES:
        raise ValueError(
            f"Phase 2c currently supports only {PHASE2C_BENCHMARK_NAMES}, got '{benchmark_name}'."
        )

    base_spec = get_benchmark_spec(benchmark_name)
    split = base_spec.make_dataset_split()
    resolved_mlp_source = (
        PHASE2_MLP_SOURCE_PHASE2G1
        if mlp_source is None and tuned_source == PHASE2_TUNED_SOURCE_PHASE2G1
        else PHASE2_MLP_SOURCE_PHASE2G
        if mlp_source is None and tuned_source == PHASE2_TUNED_SOURCE_PHASE2G
        else PHASE2_MLP_SOURCE_DEFAULT
        if mlp_source is None
        else mlp_source
    )
    tuned_selection = resolve_tuned_pc_selection(
        benchmark_name,
        tuned_source=tuned_source,
        joint_search_output_root=joint_search_output_root,
    )
    mlp_selection = resolve_mlp_selection(
        benchmark_name,
        mlp_source=resolved_mlp_source,
        joint_search_output_root=joint_search_output_root,
    )
    resolved_run_id = _resolve_run_id(run_id)
    resolved_seed_values = (
        list(seed_values)
        if seed_values is not None
        else list(default_seed_values_for_benchmark(benchmark_name))
    )
    run_dir = _prepare_run_dir(
        _resolve_multiseed_root(
            output_root,
            benchmark_name,
            resolved_run_id,
            output_layout,
            tuned_selection.source,
            mlp_selection.source,
        )
    )
    study_config = _build_study_config(
        base_spec,
        run_id=resolved_run_id,
        output_layout=output_layout,
        seed_values=resolved_seed_values,
        tuned_selection=tuned_selection,
        mlp_selection=mlp_selection,
        split=split,
    )
    _write_json(run_dir / "study_config.json", study_config)

    seed_records: list[dict[str, Any]] = []
    tuned_training = tuned_selection.pc_training

    for seed_index, seed in enumerate(resolved_seed_values):
        seed_dir = run_dir / "seeds" / f"seed_{int(seed):04d}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        default_spec = _seeded_spec(base_spec, seed=int(seed))
        tuned_spec = _seeded_spec(
            base_spec,
            seed=int(seed),
            pc_training=tuned_training,
            epochs=tuned_selection.epochs,
        )
        mlp_spec = _seeded_spec(
            base_spec,
            seed=int(seed),
            mlp_training=mlp_selection.mlp_training,
            epochs=mlp_selection.epochs,
        )

        row: dict[str, Any] = {
            "seed_index": seed_index,
            "run_seed": int(seed),
            "data_seed": base_spec.data_seed,
            "model_init_seed": int(seed),
            "default_pc_status": "failed",
            "default_pc_failure_reason": "",
            "default_pc_primary_metric_value": None,
            "default_pc_final_pre_update_energy": None,
            "default_pc_summary_path": "",
            "tuned_pc_status": "failed",
            "tuned_pc_failure_reason": "",
            "tuned_pc_primary_metric_value": None,
            "tuned_pc_final_pre_update_energy": None,
            "tuned_pc_summary_path": "",
            "mlp_status": "failed",
            "mlp_failure_reason": "",
            "mlp_primary_metric_value": None,
            "mlp_final_loss": None,
            "mlp_summary_path": "",
            "primary_metric_delta_tuned_pc_minus_default_pc": None,
            "primary_metric_delta_mlp_minus_tuned_pc": None,
            "tuned_pc_beats_default_pc": None,
            "tuned_pc_beats_mlp": None,
        }

        try:
            default_result = run_pc_benchmark(
                default_spec,
                output_root=seed_dir,
                run_id=resolved_run_id,
                plot_energy=plot_energy,
                output_layout="single_dir",
                experiment_name="default_pc",
                split=split,
            )
            row["default_pc_status"] = "ok"
            row["default_pc_primary_metric_value"] = float(default_result.summary["test_metric"])
            row["default_pc_final_pre_update_energy"] = float(
                default_result.summary["final_pre_update_energy"]
            )
            row["default_pc_summary_path"] = (
                Path("seeds") / f"seed_{int(seed):04d}" / "default_pc" / "summary.json"
            ).as_posix()
        except Exception as exc:
            _clean_failed_variant_dir(seed_dir / "default_pc")
            row["default_pc_failure_reason"] = _failure_reason(exc)

        try:
            tuned_result = run_pc_benchmark(
                tuned_spec,
                output_root=seed_dir,
                run_id=resolved_run_id,
                plot_energy=plot_energy,
                output_layout="single_dir",
                experiment_name="tuned_pc",
                split=split,
            )
            row["tuned_pc_status"] = "ok"
            row["tuned_pc_primary_metric_value"] = float(tuned_result.summary["test_metric"])
            row["tuned_pc_final_pre_update_energy"] = float(
                tuned_result.summary["final_pre_update_energy"]
            )
            row["tuned_pc_summary_path"] = (
                Path("seeds") / f"seed_{int(seed):04d}" / "tuned_pc" / "summary.json"
            ).as_posix()
        except Exception as exc:
            _clean_failed_variant_dir(seed_dir / "tuned_pc")
            row["tuned_pc_failure_reason"] = _failure_reason(exc)

        try:
            _, _, mlp_summary = _run_mlp_variant(
                mlp_spec,
                variant_dir=seed_dir / "mlp",
                run_id=resolved_run_id,
                output_layout="single_dir",
                split=split,
            )
            row["mlp_status"] = "ok"
            row["mlp_primary_metric_value"] = float(mlp_summary["test_metric"])
            row["mlp_final_loss"] = float(mlp_summary["final_loss"])
            row["mlp_summary_path"] = (
                Path("seeds") / f"seed_{int(seed):04d}" / "mlp" / "summary.json"
            ).as_posix()
        except Exception as exc:
            _clean_failed_variant_dir(seed_dir / "mlp")
            row["mlp_failure_reason"] = _failure_reason(exc)

        if _paired_default_vs_tuned(row):
            tuned_value = float(row["tuned_pc_primary_metric_value"])
            default_value = float(row["default_pc_primary_metric_value"])
            row["primary_metric_delta_tuned_pc_minus_default_pc"] = tuned_value - default_value
            row["tuned_pc_beats_default_pc"] = _metric_value_is_better(
                base_spec.primary_metric_name,
                tuned_value,
                default_value,
            )

        if _paired_tuned_vs_mlp(row):
            tuned_value = float(row["tuned_pc_primary_metric_value"])
            mlp_value = float(row["mlp_primary_metric_value"])
            row["primary_metric_delta_mlp_minus_tuned_pc"] = mlp_value - tuned_value
            row["tuned_pc_beats_mlp"] = _metric_value_is_better(
                base_spec.primary_metric_name,
                tuned_value,
                mlp_value,
            )

        seed_records.append(row)

    _write_seed_records(run_dir / "seed_records.csv", seed_records)
    aggregate_summary = _build_aggregate_summary(
        base_spec=base_spec,
        run_id=resolved_run_id,
        seed_values=resolved_seed_values,
        seed_records=seed_records,
        tuned_selection=tuned_selection,
        mlp_selection=mlp_selection,
    )
    _write_json(run_dir / "aggregate_summary.json", aggregate_summary)

    if plot_summary:
        _plot_multiseed_summary(run_dir, base_spec, seed_records, aggregate_summary)

    return PCMultiSeedRunResult(
        run_dir=run_dir,
        study_config=study_config,
        seed_records=seed_records,
        aggregate_summary=aggregate_summary,
    )
