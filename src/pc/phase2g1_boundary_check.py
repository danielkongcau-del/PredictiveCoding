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
from .comparison import (
    WINNER_TOLERANCE_ATOL,
    WINNER_TOLERANCE_RTOL,
    select_comparison_winner,
)
from .experiment import OutputLayout
from .metrics import metric_higher_is_better
from .utils import set_seed

PHASE2G1_BENCHMARK_NAMES = (
    "toy_regression",
    "toy_sine_regression",
)

DEFAULT_PC_BOUNDARY_EXTENSIONS: dict[str, dict[str, list[float | int]]] = {
    "toy_regression": {
        "eta_x": [0.025],
        "eta_w": [0.6],
        "epochs": [480],
    },
    "toy_sine_regression": {
        "eta_x": [0.2],
        "eta_w": [0.3],
        "train_steps": [180],
        "epochs": [480],
    },
}

DEFAULT_MLP_BOUNDARY_EXTENSIONS: dict[str, dict[str, list[float | int]]] = {
    "toy_regression": {
        "eta_w": [0.3, 0.4],
        "epochs": [480],
    },
    "toy_sine_regression": {
        "eta_w": [0.3],
        "epochs": [480],
    },
}


@dataclass(frozen=True)
class BoundaryCheckPCTrial:
    """One local predictive-coding configuration for the Phase 2g.1 boundary check."""

    config_id: str
    label: str
    changed_fields: tuple[str, ...]
    uses_extended_boundary_values: bool
    eta_x: float
    eta_w: float
    train_steps: int
    epochs: int
    state_init: str

    @property
    def eta_b(self) -> float:
        return self.eta_w

    @property
    def eval_steps(self) -> int:
        return self.train_steps

    def to_pc_training_spec(self) -> PCTrainingSpec:
        return PCTrainingSpec(
            eta_x=self.eta_x,
            eta_w=self.eta_w,
            eta_b=self.eta_b,
            train_steps=self.train_steps,
            eval_steps=self.eval_steps,
            state_init=self.state_init,
        )

    def config_dict(self) -> dict[str, Any]:
        return {
            "eta_x": self.eta_x,
            "eta_w": self.eta_w,
            "eta_b": self.eta_b,
            "train_steps": self.train_steps,
            "eval_steps": self.eval_steps,
            "epochs": self.epochs,
            "state_init": self.state_init,
        }


@dataclass(frozen=True)
class BoundaryCheckMLPTrial:
    """One local MLP configuration for the Phase 2g.1 boundary check."""

    config_id: str
    label: str
    changed_fields: tuple[str, ...]
    uses_extended_boundary_values: bool
    eta_w: float
    epochs: int

    @property
    def eta_b(self) -> float:
        return self.eta_w

    def to_mlp_training_spec(self) -> MLPTrainingSpec:
        return MLPTrainingSpec(
            eta_w=self.eta_w,
            eta_b=self.eta_b,
        )

    def config_dict(self) -> dict[str, Any]:
        return {
            "eta_w": self.eta_w,
            "eta_b": self.eta_b,
            "epochs": self.epochs,
        }


@dataclass
class Phase2G1BoundaryCheckRunResult:
    """Materialized outputs of one Phase 2g.1 local boundary-check study."""

    run_dir: Path
    study_config: dict[str, Any]
    pc_search_rows: list[dict[str, Any]]
    mlp_search_rows: list[dict[str, Any]]
    aggregate_summary: dict[str, Any]
    best_pc_config_summary: dict[str, Any]
    best_mlp_config_summary: dict[str, Any]


def _resolve_run_id(run_id: str | None) -> str:
    if run_id is not None:
        return run_id
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_run_root(
    output_root: str | Path,
    benchmark_name: str,
    run_id: str,
    output_layout: OutputLayout,
) -> Path:
    base_dir = Path(output_root) / "phase2g1_boundary_check" / benchmark_name
    if output_layout == "single_dir":
        return base_dir
    if output_layout == "run_id_subdir":
        return base_dir / run_id
    raise ValueError(f"Unsupported output_layout '{output_layout}'.")


def _prepare_run_dir(run_dir: Path) -> Path:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, indent=2)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _format_value_token(value: float | int) -> str:
    text = str(value)
    return text.replace("-", "neg_").replace(".", "p")


def _selection_val_metric(payload: dict[str, Any]) -> float:
    value = payload.get("val_metric", payload.get("selection_metric_value", payload.get("eval_metric")))
    if value is None:
        raise ValueError("Expected a validation metric in the Phase 2g summary payload.")
    return float(value)


def _report_test_metric(payload: dict[str, Any]) -> float:
    value = payload.get("test_metric", payload.get("report_metric_value", payload.get("primary_metric_value")))
    if value is None:
        raise ValueError("Expected a test metric in the Phase 2g summary payload.")
    return float(value)


def _metric_delta(candidate_value: float, reference_value: float) -> float:
    return float(candidate_value - reference_value)


def _metric_improvement_amount(metric_name: str, reference_value: float, candidate_value: float) -> float:
    if metric_higher_is_better(metric_name):
        return float(candidate_value - reference_value)
    return float(reference_value - candidate_value)


def _metric_value_is_better(metric_name: str, candidate_value: float, reference_value: float) -> bool:
    if np.isclose(
        candidate_value,
        reference_value,
        rtol=WINNER_TOLERANCE_RTOL,
        atol=WINNER_TOLERANCE_ATOL,
    ):
        return False
    if metric_higher_is_better(metric_name):
        return candidate_value > reference_value
    return candidate_value < reference_value


def _rank_rows(rows: list[dict[str, Any]], metric_name: str) -> list[dict[str, Any]]:
    successful_rows = [row for row in rows if row["status"] == "ok"]
    if not successful_rows:
        return [dict(row, selection_metric_rank=None, selection_metric_delta_vs_best=None) for row in rows]

    if metric_higher_is_better(metric_name):
        ranked_successful = sorted(
            successful_rows,
            key=lambda row: (-float(row["selection_metric_value"]), str(row["config_id"])),
        )
    else:
        ranked_successful = sorted(
            successful_rows,
            key=lambda row: (float(row["selection_metric_value"]), str(row["config_id"])),
        )

    best_selection_metric = float(ranked_successful[0]["selection_metric_value"])
    rank_mapping = {row["config_id"]: index for index, row in enumerate(ranked_successful, start=1)}
    ranked_rows: list[dict[str, Any]] = []
    for row in rows:
        ranked_row = dict(row)
        if row["status"] != "ok":
            ranked_row["selection_metric_rank"] = None
            ranked_row["selection_metric_delta_vs_best"] = None
        else:
            ranked_row["selection_metric_rank"] = rank_mapping[row["config_id"]]
            ranked_row["selection_metric_delta_vs_best"] = (
                float(row["selection_metric_value"]) - best_selection_metric
            )
        ranked_rows.append(ranked_row)
    return ranked_rows


def _select_best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful_rows = [row for row in rows if row["status"] == "ok"]
    if not successful_rows:
        raise ValueError("At least one successful configuration is required.")
    return min(successful_rows, key=lambda row: int(row["selection_metric_rank"]))


def _selection_reason(metric_name: str, successful_count: int) -> str:
    if metric_higher_is_better(metric_name):
        return f"selected highest val_{metric_name} across {successful_count} successful boundary-check configurations"
    return f"selected lowest val_{metric_name} across {successful_count} successful boundary-check configurations"


def _top_ranked_configs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "config_id": row["config_id"],
            "trial_label": row["trial_label"],
            "val_metric": row["val_metric"],
            "test_metric": row["test_metric"],
            "train_metric": row["train_metric"],
            "uses_extended_boundary_values": row["uses_extended_boundary_values"],
            "changed_fields": row["changed_fields"],
            "selection_metric_rank": row["selection_metric_rank"],
        }
        for row in sorted(
            [row for row in rows if row["status"] == "ok"],
            key=lambda item: int(item["selection_metric_rank"]),
        )[:5]
    ]


def _write_search_results(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "config_id",
        "trial_label",
        "model_family",
        "changed_fields",
        "uses_extended_boundary_values",
        "eta_x",
        "eta_w",
        "eta_b",
        "train_steps",
        "eval_steps",
        "epochs",
        "state_init",
        "status",
        "failure_reason",
        "metric_name",
        "metric_higher_is_better",
        "train_metric",
        "val_metric",
        "test_metric",
        "selection_metric_source",
        "selection_metric_value",
        "report_metric_source",
        "report_metric_value",
        "selection_metric_rank",
        "selection_metric_delta_vs_best",
        "train_baseline_metric",
        "val_baseline_metric",
        "test_baseline_metric",
        "beats_val_baseline",
        "best_epoch",
        "final_pre_update_energy",
        "final_post_update_energy",
        "final_loss",
        "summary_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_payload = dict(row)
            row_payload["changed_fields"] = "|".join(row["changed_fields"])
            writer.writerow(row_payload)


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


def _clean_failed_trial_dir(trial_dir: Path) -> None:
    if trial_dir.exists():
        shutil.rmtree(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)


def _trial_failure_reason(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _phase2g_reference_root(
    previous_search_output_root: str | Path,
    benchmark_name: str,
    previous_run_id: str | None,
) -> Path:
    base_root = Path(previous_search_output_root)
    primary = base_root / "phase2g_matched_search" / benchmark_name
    secondary = base_root / benchmark_name
    for candidate_base in (primary, secondary):
        candidate = candidate_base / previous_run_id if previous_run_id is not None else candidate_base
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find Phase 2g reference artifacts for benchmark '{benchmark_name}' under '{base_root}'."
    )


def _load_phase2g_reference_artifacts(
    benchmark_name: str,
    *,
    previous_search_output_root: str | Path,
    previous_run_id: str | None,
) -> dict[str, Any]:
    reference_root = _phase2g_reference_root(previous_search_output_root, benchmark_name, previous_run_id)
    aggregate_summary = _load_json(reference_root / "aggregate_summary.json")
    study_config = _load_json(reference_root / "study_config.json")
    best_pc_summary = _load_json(reference_root / "best_pc_config_summary.json")
    best_mlp_summary = _load_json(reference_root / "best_mlp_config_summary.json")
    return {
        "reference_root": reference_root,
        "aggregate_summary": aggregate_summary,
        "study_config": study_config,
        "best_pc_summary": best_pc_summary,
        "best_mlp_summary": best_mlp_summary,
    }


def _is_boundary_value(value: float | int, boundary_value: float | int) -> bool:
    return bool(np.isclose(float(value), float(boundary_value), rtol=1e-12, atol=1e-12))


def identify_boundary_dimensions(
    *,
    search_space: dict[str, Sequence[float] | Sequence[int]],
    best_config: dict[str, Any],
    dimensions: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Return per-dimension boundary metadata for a selected config."""
    boundary_dimensions: dict[str, dict[str, Any]] = {}
    for dimension in dimensions:
        if dimension not in search_space or dimension not in best_config:
            continue
        values = list(search_space[dimension])
        if len(values) == 0:
            continue
        lower_value = min(values)
        upper_value = max(values)
        best_value = best_config[dimension]
        at_lower_edge = _is_boundary_value(best_value, lower_value)
        at_upper_edge = _is_boundary_value(best_value, upper_value)
        if at_lower_edge or at_upper_edge:
            boundary_dimensions[dimension] = {
                "value": best_value,
                "search_values": values,
                "lower_edge_value": lower_value,
                "upper_edge_value": upper_value,
                "at_lower_edge": at_lower_edge,
                "at_upper_edge": at_upper_edge,
            }
    return boundary_dimensions


def _resolve_extension_spec(
    benchmark_name: str,
    *,
    default_extensions: dict[str, dict[str, list[float | int]]],
    override: dict[str, list[float | int]] | None,
) -> dict[str, list[float | int]]:
    resolved = {
        key: list(values)
        for key, values in default_extensions.get(benchmark_name, {}).items()
    }
    if override is None:
        return resolved
    for key, values in override.items():
        if len(values) == 0:
            raise ValueError(f"Boundary extension key '{key}' must contain at least one value.")
        resolved[key] = list(values)
    return resolved


def _combined_extension_value(
    boundary_info: dict[str, Any],
    extension_values: Sequence[float | int],
) -> float | int:
    if boundary_info["at_lower_edge"] and not boundary_info["at_upper_edge"]:
        return min(extension_values)
    if boundary_info["at_upper_edge"] and not boundary_info["at_lower_edge"]:
        return max(extension_values)
    current_value = float(boundary_info["value"])
    return max(extension_values, key=lambda candidate: abs(float(candidate) - current_value))


def _pc_trial_signature(
    *,
    eta_x: float,
    eta_w: float,
    train_steps: int,
    epochs: int,
    state_init: str,
) -> tuple[Any, ...]:
    return (eta_x, eta_w, train_steps, epochs, state_init)


def _mlp_trial_signature(*, eta_w: float, epochs: int) -> tuple[Any, ...]:
    return (eta_w, epochs)


def build_pc_boundary_trials(
    spec: ToyBenchmarkSpec,
    *,
    phase2g_best_summary: dict[str, Any],
    phase2g_study_config: dict[str, Any],
    extension_override: dict[str, list[float | int]] | None = None,
) -> tuple[list[BoundaryCheckPCTrial], dict[str, Any]]:
    """Return a compact local neighborhood around the Phase 2g best PC config."""
    previous_best_config = dict(phase2g_best_summary["best_config"])
    original_search_space = {
        key: list(values)
        for key, values in phase2g_study_config["pc_search_space"].items()
    }
    boundary_dimensions = identify_boundary_dimensions(
        search_space=original_search_space,
        best_config=previous_best_config,
        dimensions=("eta_x", "eta_w", "train_steps", "epochs"),
    )
    extension_spec = _resolve_extension_spec(
        spec.benchmark_name,
        default_extensions=DEFAULT_PC_BOUNDARY_EXTENSIONS,
        override=extension_override,
    )
    probed_dimensions = [
        dimension
        for dimension in ("eta_x", "eta_w", "train_steps", "epochs")
        if dimension in boundary_dimensions and dimension in extension_spec
    ]
    unprobed_boundary_dimensions = [
        dimension for dimension in boundary_dimensions if dimension not in probed_dimensions
    ]

    trials: list[BoundaryCheckPCTrial] = []
    seen_signatures: set[tuple[Any, ...]] = set()

    def append_trial(
        *,
        label: str,
        changed_fields: tuple[str, ...],
        uses_extended_boundary_values: bool,
        eta_x: float,
        eta_w: float,
        train_steps: int,
        epochs: int,
    ) -> None:
        signature = _pc_trial_signature(
            eta_x=eta_x,
            eta_w=eta_w,
            train_steps=train_steps,
            epochs=epochs,
            state_init=previous_best_config["state_init"],
        )
        if signature in seen_signatures:
            return
        seen_signatures.add(signature)
        trials.append(
            BoundaryCheckPCTrial(
                config_id=f"cfg_{len(trials) + 1:03d}",
                label=label,
                changed_fields=changed_fields,
                uses_extended_boundary_values=uses_extended_boundary_values,
                eta_x=float(eta_x),
                eta_w=float(eta_w),
                train_steps=int(train_steps),
                epochs=int(epochs),
                state_init=str(previous_best_config["state_init"]),
            )
        )

    append_trial(
        label="phase2g_best",
        changed_fields=tuple(),
        uses_extended_boundary_values=False,
        eta_x=float(previous_best_config["eta_x"]),
        eta_w=float(previous_best_config["eta_w"]),
        train_steps=int(previous_best_config["train_steps"]),
        epochs=int(previous_best_config["epochs"]),
    )

    for dimension in probed_dimensions:
        for extension_value in extension_spec[dimension]:
            trial_kwargs = {
                "eta_x": float(previous_best_config["eta_x"]),
                "eta_w": float(previous_best_config["eta_w"]),
                "train_steps": int(previous_best_config["train_steps"]),
                "epochs": int(previous_best_config["epochs"]),
            }
            trial_kwargs[dimension] = extension_value
            append_trial(
                label=f"probe_{dimension}_{_format_value_token(extension_value)}",
                changed_fields=(dimension,),
                uses_extended_boundary_values=True,
                eta_x=float(trial_kwargs["eta_x"]),
                eta_w=float(trial_kwargs["eta_w"]),
                train_steps=int(trial_kwargs["train_steps"]),
                epochs=int(trial_kwargs["epochs"]),
            )

    if len(probed_dimensions) >= 2:
        combined_values: dict[str, float | int] = {}
        for dimension in probed_dimensions:
            combined_values[dimension] = _combined_extension_value(
                boundary_dimensions[dimension],
                extension_spec[dimension],
            )
        append_trial(
            label="probe_combined_boundary",
            changed_fields=tuple(probed_dimensions),
            uses_extended_boundary_values=True,
            eta_x=float(combined_values.get("eta_x", previous_best_config["eta_x"])),
            eta_w=float(combined_values.get("eta_w", previous_best_config["eta_w"])),
            train_steps=int(combined_values.get("train_steps", previous_best_config["train_steps"])),
            epochs=int(combined_values.get("epochs", previous_best_config["epochs"])),
        )

    boundary_report = {
        "original_search_space": original_search_space,
        "previous_best_config": previous_best_config,
        "boundary_dimensions": boundary_dimensions,
        "probed_dimensions": probed_dimensions,
        "unprobed_boundary_dimensions": unprobed_boundary_dimensions,
        "extension_spec": extension_spec,
        "trial_count": len(trials),
        "trial_labels": [trial.label for trial in trials],
    }
    return trials, boundary_report


def build_mlp_boundary_trials(
    spec: ToyBenchmarkSpec,
    *,
    phase2g_best_summary: dict[str, Any],
    phase2g_study_config: dict[str, Any],
    extension_override: dict[str, list[float | int]] | None = None,
) -> tuple[list[BoundaryCheckMLPTrial], dict[str, Any]]:
    """Return a compact local neighborhood around the Phase 2g best MLP config."""
    previous_best_config = dict(phase2g_best_summary["best_config"])
    original_search_space = {
        key: list(values)
        for key, values in phase2g_study_config["mlp_search_space"].items()
    }
    boundary_dimensions = identify_boundary_dimensions(
        search_space=original_search_space,
        best_config=previous_best_config,
        dimensions=("eta_w", "epochs"),
    )
    extension_spec = _resolve_extension_spec(
        spec.benchmark_name,
        default_extensions=DEFAULT_MLP_BOUNDARY_EXTENSIONS,
        override=extension_override,
    )
    probed_dimensions = [
        dimension
        for dimension in ("eta_w", "epochs")
        if dimension in boundary_dimensions and dimension in extension_spec
    ]
    unprobed_boundary_dimensions = [
        dimension for dimension in boundary_dimensions if dimension not in probed_dimensions
    ]

    trials: list[BoundaryCheckMLPTrial] = []
    seen_signatures: set[tuple[Any, ...]] = set()

    def append_trial(
        *,
        label: str,
        changed_fields: tuple[str, ...],
        uses_extended_boundary_values: bool,
        eta_w: float,
        epochs: int,
    ) -> None:
        signature = _mlp_trial_signature(eta_w=eta_w, epochs=epochs)
        if signature in seen_signatures:
            return
        seen_signatures.add(signature)
        trials.append(
            BoundaryCheckMLPTrial(
                config_id=f"cfg_{len(trials) + 1:03d}",
                label=label,
                changed_fields=changed_fields,
                uses_extended_boundary_values=uses_extended_boundary_values,
                eta_w=float(eta_w),
                epochs=int(epochs),
            )
        )

    append_trial(
        label="phase2g_best",
        changed_fields=tuple(),
        uses_extended_boundary_values=False,
        eta_w=float(previous_best_config["eta_w"]),
        epochs=int(previous_best_config["epochs"]),
    )

    for dimension in probed_dimensions:
        for extension_value in extension_spec[dimension]:
            trial_kwargs = {
                "eta_w": float(previous_best_config["eta_w"]),
                "epochs": int(previous_best_config["epochs"]),
            }
            trial_kwargs[dimension] = extension_value
            append_trial(
                label=f"probe_{dimension}_{_format_value_token(extension_value)}",
                changed_fields=(dimension,),
                uses_extended_boundary_values=True,
                eta_w=float(trial_kwargs["eta_w"]),
                epochs=int(trial_kwargs["epochs"]),
            )

    if len(probed_dimensions) >= 2:
        combined_values: dict[str, float | int] = {}
        for dimension in probed_dimensions:
            combined_values[dimension] = _combined_extension_value(
                boundary_dimensions[dimension],
                extension_spec[dimension],
            )
        append_trial(
            label="probe_combined_boundary",
            changed_fields=tuple(probed_dimensions),
            uses_extended_boundary_values=True,
            eta_w=float(combined_values.get("eta_w", previous_best_config["eta_w"])),
            epochs=int(combined_values.get("epochs", previous_best_config["epochs"])),
        )

    boundary_report = {
        "original_search_space": original_search_space,
        "previous_best_config": previous_best_config,
        "boundary_dimensions": boundary_dimensions,
        "probed_dimensions": probed_dimensions,
        "unprobed_boundary_dimensions": unprobed_boundary_dimensions,
        "extension_spec": extension_spec,
        "trial_count": len(trials),
        "trial_labels": [trial.label for trial in trials],
    }
    return trials, boundary_report


def _mlp_config_payload(
    spec: ToyBenchmarkSpec,
    trial: BoundaryCheckMLPTrial,
    *,
    run_id: str,
    search_root: Path,
    output_layout: OutputLayout,
    split,
) -> dict[str, Any]:
    return {
        "phase": "Phase 2g.1",
        "experiment_name": trial.config_id,
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
            "epochs": trial.epochs,
            "eta_w": trial.eta_w,
            "eta_b": trial.eta_b,
            "loss_name": "mse",
            "run_seed": spec.run_seed,
        },
        "logging": {
            "output_root": str(search_root),
            "output_layout": output_layout,
            "plot_energy": False,
            "trace_policy": "not_applicable",
        },
    }


def _run_mlp_trial(
    spec: ToyBenchmarkSpec,
    trial: BoundaryCheckMLPTrial,
    *,
    trial_root: Path,
    run_id: str,
    output_layout: OutputLayout,
    split,
) -> dict[str, Any]:
    trial_root.mkdir(parents=True, exist_ok=True)
    set_seed(spec.run_seed)
    model = spec.build_mlp_model()
    _write_json(
        trial_root / "config.json",
        _mlp_config_payload(
            spec,
            trial,
            run_id=run_id,
            search_root=trial_root.parent,
            output_layout=output_layout,
            split=split,
        ),
    )

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

    final_row = epoch_rows[-1]
    if spec.primary_metric_higher_is_better:
        best_row = max(epoch_rows, key=lambda row: float(row[f"val_{spec.primary_metric_name}"]))
    else:
        best_row = min(epoch_rows, key=lambda row: float(row[f"val_{spec.primary_metric_name}"]))

    test_predictions = model.predict(x_test)
    test_metric_value = spec.primary_metric_fn(test_predictions, y_test)
    summary = {
        "phase": "Phase 2g.1",
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
        "final_loss": final_row["loss"],
    }

    _write_mlp_epoch_metrics(
        trial_root / "epoch_metrics.csv",
        epoch_rows,
        num_layers=len(model.layers),
        task_name=spec.task_name,
    )
    _write_json(trial_root / "summary.json", summary)
    return summary


def _pc_row_from_summary(
    trial: BoundaryCheckPCTrial,
    summary: dict[str, Any],
    *,
    summary_path: str,
) -> dict[str, Any]:
    return {
        "config_id": trial.config_id,
        "trial_label": trial.label,
        "model_family": "predictive_coding",
        "changed_fields": list(trial.changed_fields),
        "uses_extended_boundary_values": trial.uses_extended_boundary_values,
        "eta_x": trial.eta_x,
        "eta_w": trial.eta_w,
        "eta_b": trial.eta_b,
        "train_steps": trial.train_steps,
        "eval_steps": trial.eval_steps,
        "epochs": trial.epochs,
        "state_init": trial.state_init,
        "status": "ok",
        "failure_reason": "",
        "metric_name": summary["metric_name"],
        "metric_higher_is_better": summary["metric_higher_is_better"],
        "train_metric": float(summary["train_metric"]),
        "val_metric": float(summary["val_metric"]),
        "test_metric": float(summary["test_metric"]),
        "selection_metric_source": "val_metric",
        "selection_metric_value": float(summary["val_metric"]),
        "report_metric_source": "test_metric",
        "report_metric_value": float(summary["test_metric"]),
        "selection_metric_rank": None,
        "selection_metric_delta_vs_best": None,
        "train_baseline_metric": float(summary["train_baseline_metric"]),
        "val_baseline_metric": float(summary["val_baseline_metric"]),
        "test_baseline_metric": float(summary["test_baseline_metric"]),
        "beats_val_baseline": _metric_value_is_better(
            str(summary["metric_name"]),
            float(summary["val_metric"]),
            float(summary["val_baseline_metric"]),
        ),
        "best_epoch": int(summary["best_epoch"]),
        "final_pre_update_energy": float(summary["final_pre_update_energy"]),
        "final_post_update_energy": float(summary["final_post_update_energy"]),
        "final_loss": None,
        "summary_path": summary_path,
    }


def _mlp_row_from_summary(
    trial: BoundaryCheckMLPTrial,
    summary: dict[str, Any],
    *,
    summary_path: str,
) -> dict[str, Any]:
    return {
        "config_id": trial.config_id,
        "trial_label": trial.label,
        "model_family": "mlp",
        "changed_fields": list(trial.changed_fields),
        "uses_extended_boundary_values": trial.uses_extended_boundary_values,
        "eta_x": None,
        "eta_w": trial.eta_w,
        "eta_b": trial.eta_b,
        "train_steps": None,
        "eval_steps": None,
        "epochs": trial.epochs,
        "state_init": None,
        "status": "ok",
        "failure_reason": "",
        "metric_name": summary["metric_name"],
        "metric_higher_is_better": summary["metric_higher_is_better"],
        "train_metric": float(summary["train_metric"]),
        "val_metric": float(summary["val_metric"]),
        "test_metric": float(summary["test_metric"]),
        "selection_metric_source": "val_metric",
        "selection_metric_value": float(summary["val_metric"]),
        "report_metric_source": "test_metric",
        "report_metric_value": float(summary["test_metric"]),
        "selection_metric_rank": None,
        "selection_metric_delta_vs_best": None,
        "train_baseline_metric": float(summary["train_baseline_metric"]),
        "val_baseline_metric": float(summary["val_baseline_metric"]),
        "test_baseline_metric": float(summary["test_baseline_metric"]),
        "beats_val_baseline": _metric_value_is_better(
            str(summary["metric_name"]),
            float(summary["val_metric"]),
            float(summary["val_baseline_metric"]),
        ),
        "best_epoch": int(summary["best_epoch"]),
        "final_pre_update_energy": None,
        "final_post_update_energy": None,
        "final_loss": float(summary["final_loss"]),
        "summary_path": summary_path,
    }


def _failed_pc_row(
    trial: BoundaryCheckPCTrial,
    spec: ToyBenchmarkSpec,
    split,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "config_id": trial.config_id,
        "trial_label": trial.label,
        "model_family": "predictive_coding",
        "changed_fields": list(trial.changed_fields),
        "uses_extended_boundary_values": trial.uses_extended_boundary_values,
        "eta_x": trial.eta_x,
        "eta_w": trial.eta_w,
        "eta_b": trial.eta_b,
        "train_steps": trial.train_steps,
        "eval_steps": trial.eval_steps,
        "epochs": trial.epochs,
        "state_init": trial.state_init,
        "status": "failed",
        "failure_reason": _trial_failure_reason(exc),
        "metric_name": spec.primary_metric_name,
        "metric_higher_is_better": spec.primary_metric_higher_is_better,
        "train_metric": None,
        "val_metric": None,
        "test_metric": None,
        "selection_metric_source": "val_metric",
        "selection_metric_value": None,
        "report_metric_source": "test_metric",
        "report_metric_value": None,
        "selection_metric_rank": None,
        "selection_metric_delta_vs_best": None,
        "train_baseline_metric": spec.baseline_metric_fn(split.y_train),
        "val_baseline_metric": spec.baseline_metric_fn(split.y_val),
        "test_baseline_metric": spec.baseline_metric_fn(split.y_test),
        "beats_val_baseline": None,
        "best_epoch": None,
        "final_pre_update_energy": None,
        "final_post_update_energy": None,
        "final_loss": None,
        "summary_path": "",
    }


def _failed_mlp_row(
    trial: BoundaryCheckMLPTrial,
    spec: ToyBenchmarkSpec,
    split,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "config_id": trial.config_id,
        "trial_label": trial.label,
        "model_family": "mlp",
        "changed_fields": list(trial.changed_fields),
        "uses_extended_boundary_values": trial.uses_extended_boundary_values,
        "eta_x": None,
        "eta_w": trial.eta_w,
        "eta_b": trial.eta_b,
        "train_steps": None,
        "eval_steps": None,
        "epochs": trial.epochs,
        "state_init": None,
        "status": "failed",
        "failure_reason": _trial_failure_reason(exc),
        "metric_name": spec.primary_metric_name,
        "metric_higher_is_better": spec.primary_metric_higher_is_better,
        "train_metric": None,
        "val_metric": None,
        "test_metric": None,
        "selection_metric_source": "val_metric",
        "selection_metric_value": None,
        "report_metric_source": "test_metric",
        "report_metric_value": None,
        "selection_metric_rank": None,
        "selection_metric_delta_vs_best": None,
        "train_baseline_metric": spec.baseline_metric_fn(split.y_train),
        "val_baseline_metric": spec.baseline_metric_fn(split.y_val),
        "test_baseline_metric": spec.baseline_metric_fn(split.y_test),
        "beats_val_baseline": None,
        "best_epoch": None,
        "final_pre_update_energy": None,
        "final_post_update_energy": None,
        "final_loss": None,
        "summary_path": "",
    }


def _moved_beyond_original_boundary(
    *,
    config: dict[str, Any],
    boundary_dimensions: dict[str, dict[str, Any]],
) -> tuple[bool, list[str]]:
    moved_fields: list[str] = []
    for dimension, boundary_info in boundary_dimensions.items():
        value = config.get(dimension)
        if value is None:
            continue
        lower_value = boundary_info["lower_edge_value"]
        upper_value = boundary_info["upper_edge_value"]
        if float(value) < float(lower_value) or float(value) > float(upper_value):
            moved_fields.append(dimension)
    return len(moved_fields) > 0, moved_fields


def _family_best_summary(
    *,
    benchmark_name: str,
    run_id: str,
    task_name: str,
    metric_name: str,
    model_family: str,
    previous_best_summary: dict[str, Any],
    best_row: dict[str, Any],
    search_rows: list[dict[str, Any]],
    boundary_report: dict[str, Any],
) -> dict[str, Any]:
    successful_count = len([row for row in search_rows if row["status"] == "ok"])
    moved_beyond_boundary, moved_fields = _moved_beyond_original_boundary(
        config=best_row,
        boundary_dimensions=boundary_report["boundary_dimensions"],
    )
    if model_family == "predictive_coding":
        best_config = {
            "eta_x": best_row["eta_x"],
            "eta_w": best_row["eta_w"],
            "eta_b": best_row["eta_b"],
            "train_steps": best_row["train_steps"],
            "eval_steps": best_row["eval_steps"],
            "epochs": best_row["epochs"],
            "state_init": best_row["state_init"],
        }
    else:
        best_config = {
            "eta_w": best_row["eta_w"],
            "eta_b": best_row["eta_b"],
            "epochs": best_row["epochs"],
        }

    previous_val_metric = _selection_val_metric(previous_best_summary)
    previous_test_metric = _report_test_metric(previous_best_summary)
    return {
        "experiment_name": f"phase2g1_boundary_check_{benchmark_name}",
        "run_id": run_id,
        "phase": "Phase 2g.1",
        "benchmark_name": benchmark_name,
        "task_name": task_name,
        "model_family": model_family,
        "metric_name": metric_name,
        "metric_higher_is_better": metric_higher_is_better(metric_name),
        "selected_by_metric_source": "val_metric",
        "final_report_metric_source": "test_metric",
        "previous_phase2g_best_config_id": previous_best_summary["best_config_id"],
        "previous_phase2g_best_config": previous_best_summary["best_config"],
        "previous_phase2g_val_metric": previous_val_metric,
        "previous_phase2g_test_metric": previous_test_metric,
        "boundary_check_best_config_id": best_row["config_id"],
        "boundary_check_best_trial_label": best_row["trial_label"],
        "boundary_check_best_config": best_config,
        "boundary_check_train_metric": best_row["train_metric"],
        "boundary_check_val_metric": best_row["val_metric"],
        "boundary_check_test_metric": best_row["test_metric"],
        "boundary_check_best_epoch": best_row["best_epoch"],
        "boundary_check_summary_path": best_row["summary_path"],
        "boundary_check_uses_extended_boundary_values": best_row["uses_extended_boundary_values"],
        "boundary_check_changed_fields": best_row["changed_fields"],
        "moved_beyond_original_boundary": moved_beyond_boundary,
        "moved_beyond_original_boundary_fields": moved_fields,
        "val_metric_delta_vs_previous_phase2g_best": _metric_delta(
            float(best_row["val_metric"]),
            previous_val_metric,
        ),
        "test_metric_delta_vs_previous_phase2g_best": _metric_delta(
            float(best_row["test_metric"]),
            previous_test_metric,
        ),
        "val_metric_improvement_vs_previous_phase2g_best": _metric_improvement_amount(
            metric_name,
            previous_val_metric,
            float(best_row["val_metric"]),
        ),
        "test_metric_improvement_vs_previous_phase2g_best": _metric_improvement_amount(
            metric_name,
            previous_test_metric,
            float(best_row["test_metric"]),
        ),
        "successful_trial_count": successful_count,
        "failed_trial_count": len([row for row in search_rows if row["status"] == "failed"]),
        "selection_reason": _selection_reason(metric_name, successful_count),
        "boundary_dimensions": boundary_report["boundary_dimensions"],
        "probed_dimensions": boundary_report["probed_dimensions"],
        "unprobed_boundary_dimensions": boundary_report["unprobed_boundary_dimensions"],
        "top_ranked_configs": _top_ranked_configs(search_rows),
    }


def _build_aggregate_summary(
    *,
    benchmark_name: str,
    run_id: str,
    task_name: str,
    metric_name: str,
    previous_reference_artifacts: dict[str, Any],
    pc_search_rows: list[dict[str, Any]],
    mlp_search_rows: list[dict[str, Any]],
    pc_boundary_report: dict[str, Any],
    mlp_boundary_report: dict[str, Any],
) -> dict[str, Any]:
    previous_aggregate = previous_reference_artifacts["aggregate_summary"]
    previous_best_pc = previous_reference_artifacts["best_pc_summary"]
    previous_best_mlp = previous_reference_artifacts["best_mlp_summary"]
    best_pc_row = _select_best_row(pc_search_rows)
    best_mlp_row = _select_best_row(mlp_search_rows)

    previous_pc_test_metric = _report_test_metric(previous_best_pc)
    previous_mlp_test_metric = _report_test_metric(previous_best_mlp)
    boundary_check_pc_test_metric = float(best_pc_row["test_metric"])
    boundary_check_mlp_test_metric = float(best_mlp_row["test_metric"])
    previous_winner = str(previous_aggregate["test_winner"])
    previous_winner_reason = str(previous_aggregate["test_winner_reason"])
    boundary_check_winner, boundary_check_winner_reason = select_comparison_winner(
        metric_name,
        boundary_check_pc_test_metric,
        boundary_check_mlp_test_metric,
    )

    pc_moved_beyond_boundary, pc_moved_fields = _moved_beyond_original_boundary(
        config=best_pc_row,
        boundary_dimensions=pc_boundary_report["boundary_dimensions"],
    )
    mlp_moved_beyond_boundary, mlp_moved_fields = _moved_beyond_original_boundary(
        config=best_mlp_row,
        boundary_dimensions=mlp_boundary_report["boundary_dimensions"],
    )
    headline_conclusion_changed = previous_winner != boundary_check_winner
    boundary_sensitive = pc_moved_beyond_boundary or mlp_moved_beyond_boundary
    if not boundary_sensitive:
        further_search_reason = (
            "No selected Phase 2g.1 best config moved outside the original Phase 2g search bounds."
        )
    else:
        moved_families = []
        if pc_moved_beyond_boundary:
            moved_families.append("predictive_coding")
        if mlp_moved_beyond_boundary:
            moved_families.append("mlp")
        further_search_reason = (
            "Boundary-check best config moved beyond the original search bounds for "
            + ", ".join(moved_families)
            + "."
        )

    return {
        "experiment_name": f"phase2g1_boundary_check_{benchmark_name}",
        "run_id": run_id,
        "phase": "Phase 2g.1",
        "benchmark_name": benchmark_name,
        "task_name": task_name,
        "search_target": "local_boundary_check_matched_pc_and_mlp",
        "metric_name": metric_name,
        "metric_higher_is_better": metric_higher_is_better(metric_name),
        "selected_by_metric_source": "val_metric",
        "selection_split": "validation",
        "final_report_metric_source": "test_metric",
        "final_report_split": "test",
        "test_metric_used_for_selection": False,
        "previous_phase2g_reference_root": str(previous_reference_artifacts["reference_root"]),
        "previous_phase2g_run_id": previous_aggregate["run_id"],
        "previous_phase2g_test_winner": previous_winner,
        "previous_phase2g_test_winner_reason": previous_winner_reason,
        "previous_phase2g_best_pc_config_id": previous_best_pc["best_config_id"],
        "previous_phase2g_best_pc_config": previous_best_pc["best_config"],
        "previous_phase2g_best_pc_val_metric": _selection_val_metric(previous_best_pc),
        "previous_phase2g_best_pc_test_metric": previous_pc_test_metric,
        "previous_phase2g_best_mlp_config_id": previous_best_mlp["best_config_id"],
        "previous_phase2g_best_mlp_config": previous_best_mlp["best_config"],
        "previous_phase2g_best_mlp_val_metric": _selection_val_metric(previous_best_mlp),
        "previous_phase2g_best_mlp_test_metric": previous_mlp_test_metric,
        "pc_trial_count": len(pc_search_rows),
        "pc_successful_trial_count": len([row for row in pc_search_rows if row["status"] == "ok"]),
        "pc_failed_trial_count": len([row for row in pc_search_rows if row["status"] == "failed"]),
        "mlp_trial_count": len(mlp_search_rows),
        "mlp_successful_trial_count": len([row for row in mlp_search_rows if row["status"] == "ok"]),
        "mlp_failed_trial_count": len([row for row in mlp_search_rows if row["status"] == "failed"]),
        "boundary_check_best_pc_config_id": best_pc_row["config_id"],
        "boundary_check_best_pc_trial_label": best_pc_row["trial_label"],
        "boundary_check_best_pc_config": {
            "eta_x": best_pc_row["eta_x"],
            "eta_w": best_pc_row["eta_w"],
            "eta_b": best_pc_row["eta_b"],
            "train_steps": best_pc_row["train_steps"],
            "eval_steps": best_pc_row["eval_steps"],
            "epochs": best_pc_row["epochs"],
            "state_init": best_pc_row["state_init"],
        },
        "boundary_check_best_pc_val_metric": best_pc_row["val_metric"],
        "boundary_check_best_pc_test_metric": best_pc_row["test_metric"],
        "boundary_check_best_pc_summary_path": best_pc_row["summary_path"],
        "boundary_check_best_mlp_config_id": best_mlp_row["config_id"],
        "boundary_check_best_mlp_trial_label": best_mlp_row["trial_label"],
        "boundary_check_best_mlp_config": {
            "eta_w": best_mlp_row["eta_w"],
            "eta_b": best_mlp_row["eta_b"],
            "epochs": best_mlp_row["epochs"],
        },
        "boundary_check_best_mlp_val_metric": best_mlp_row["val_metric"],
        "boundary_check_best_mlp_test_metric": best_mlp_row["test_metric"],
        "boundary_check_best_mlp_summary_path": best_mlp_row["summary_path"],
        "boundary_check_test_winner": boundary_check_winner,
        "boundary_check_test_winner_reason": boundary_check_winner_reason,
        "headline_conclusion_changed": headline_conclusion_changed,
        "prior_conclusion_survived_boundary_check": not headline_conclusion_changed,
        "pc_val_metric_improvement_vs_previous_phase2g_best": _metric_improvement_amount(
            metric_name,
            _selection_val_metric(previous_best_pc),
            float(best_pc_row["val_metric"]),
        ),
        "pc_test_metric_improvement_vs_previous_phase2g_best": _metric_improvement_amount(
            metric_name,
            previous_pc_test_metric,
            float(best_pc_row["test_metric"]),
        ),
        "mlp_val_metric_improvement_vs_previous_phase2g_best": _metric_improvement_amount(
            metric_name,
            _selection_val_metric(previous_best_mlp),
            float(best_mlp_row["val_metric"]),
        ),
        "mlp_test_metric_improvement_vs_previous_phase2g_best": _metric_improvement_amount(
            metric_name,
            previous_mlp_test_metric,
            float(best_mlp_row["test_metric"]),
        ),
        "previous_test_metric_difference_mlp_minus_pc": previous_mlp_test_metric - previous_pc_test_metric,
        "boundary_check_test_metric_difference_mlp_minus_pc": (
            boundary_check_mlp_test_metric - boundary_check_pc_test_metric
        ),
        "pc_best_config_moved_beyond_original_boundary": pc_moved_beyond_boundary,
        "pc_moved_beyond_original_boundary_fields": pc_moved_fields,
        "mlp_best_config_moved_beyond_original_boundary": mlp_moved_beyond_boundary,
        "mlp_moved_beyond_original_boundary_fields": mlp_moved_fields,
        "pc_boundary_sensitivity_detected": pc_moved_beyond_boundary,
        "mlp_boundary_sensitivity_detected": mlp_moved_beyond_boundary,
        "boundary_sensitivity_detected": boundary_sensitive,
        "further_search_still_warranted": boundary_sensitive,
        "further_search_reason": further_search_reason,
        "pc_boundary_dimensions": pc_boundary_report["boundary_dimensions"],
        "pc_probed_dimensions": pc_boundary_report["probed_dimensions"],
        "pc_unprobed_boundary_dimensions": pc_boundary_report["unprobed_boundary_dimensions"],
        "mlp_boundary_dimensions": mlp_boundary_report["boundary_dimensions"],
        "mlp_probed_dimensions": mlp_boundary_report["probed_dimensions"],
        "mlp_unprobed_boundary_dimensions": mlp_boundary_report["unprobed_boundary_dimensions"],
        "top_ranked_pc_configs": _top_ranked_configs(pc_search_rows),
        "top_ranked_mlp_configs": _top_ranked_configs(mlp_search_rows),
        "winner_tolerance_rtol": WINNER_TOLERANCE_RTOL,
        "winner_tolerance_atol": WINNER_TOLERANCE_ATOL,
        "notes": {
            "study_shape": "Small local boundary-check neighborhood around the Phase 2g selected configs; not a full new Cartesian search.",
            "selection_rule": "Configurations are always ranked by validation metric.",
            "report_rule": "Headline method comparison is always based on held-out test metric.",
        },
    }


def _study_config_payload(
    spec: ToyBenchmarkSpec,
    *,
    run_id: str,
    output_layout: OutputLayout,
    split,
    previous_reference_artifacts: dict[str, Any],
    pc_boundary_report: dict[str, Any],
    mlp_boundary_report: dict[str, Any],
) -> dict[str, Any]:
    previous_aggregate = previous_reference_artifacts["aggregate_summary"]
    previous_best_pc = previous_reference_artifacts["best_pc_summary"]
    previous_best_mlp = previous_reference_artifacts["best_mlp_summary"]
    return {
        "experiment_name": f"phase2g1_boundary_check_{spec.benchmark_name}",
        "run_id": run_id,
        "phase": "Phase 2g.1",
        "benchmark_name": spec.benchmark_name,
        "task_name": spec.task_name,
        "search_target": "local_boundary_check_matched_pc_and_mlp",
        "metric_name": spec.primary_metric_name,
        "metric_higher_is_better": spec.primary_metric_higher_is_better,
        "selected_by_metric_source": "val_metric",
        "selection_split": "validation",
        "final_report_metric_source": "test_metric",
        "final_report_split": "test",
        "test_metric_used_for_selection": False,
        "seeds": {
            "run_seed": spec.run_seed,
            "data_seed": spec.data_seed,
            "model_init_seed": spec.model_init_seed,
        },
        "data": spec.data_config(split),
        "previous_phase2g_reference_root": str(previous_reference_artifacts["reference_root"]),
        "previous_phase2g_run_id": previous_aggregate["run_id"],
        "previous_phase2g_test_winner": previous_aggregate["test_winner"],
        "previous_phase2g_best_pc_config_id": previous_best_pc["best_config_id"],
        "previous_phase2g_best_pc_config": previous_best_pc["best_config"],
        "previous_phase2g_best_pc_val_metric": _selection_val_metric(previous_best_pc),
        "previous_phase2g_best_pc_test_metric": _report_test_metric(previous_best_pc),
        "previous_phase2g_best_mlp_config_id": previous_best_mlp["best_config_id"],
        "previous_phase2g_best_mlp_config": previous_best_mlp["best_config"],
        "previous_phase2g_best_mlp_val_metric": _selection_val_metric(previous_best_mlp),
        "previous_phase2g_best_mlp_test_metric": _report_test_metric(previous_best_mlp),
        "pc_boundary_report": pc_boundary_report,
        "mlp_boundary_report": mlp_boundary_report,
        "output_layout": output_layout,
        "notes": {
            "scope": "Local boundary extensions only; this is not a redesigned search protocol.",
            "selection_rule": "Validation metric only.",
            "report_rule": "Held-out test metric only.",
            "runtime_tradeoff": "Toy regression PC train_steps lower-edge was detected but not extended because the protocol kept the study small and did not introduce an arbitrary smaller train-step grid.",
        },
    }


def run_phase2g1_boundary_check(
    benchmark_name: str,
    *,
    output_root: str | Path = "outputs",
    previous_search_output_root: str | Path = "outputs",
    previous_run_id: str | None = None,
    run_id: str | None = None,
    output_layout: OutputLayout = "single_dir",
    plot_energy: bool = False,
    pc_boundary_extensions_override: dict[str, list[float | int]] | None = None,
    mlp_boundary_extensions_override: dict[str, list[float | int]] | None = None,
) -> Phase2G1BoundaryCheckRunResult:
    """Run a compact local boundary check around the current Phase 2g best configs."""
    if benchmark_name not in PHASE2G1_BENCHMARK_NAMES:
        raise ValueError(
            f"Phase 2g.1 currently supports only {PHASE2G1_BENCHMARK_NAMES}, got '{benchmark_name}'."
        )

    spec = get_benchmark_spec(benchmark_name)
    split = spec.make_dataset_split()
    resolved_run_id = _resolve_run_id(run_id)
    previous_reference_artifacts = _load_phase2g_reference_artifacts(
        benchmark_name,
        previous_search_output_root=previous_search_output_root,
        previous_run_id=previous_run_id,
    )
    pc_trials, pc_boundary_report = build_pc_boundary_trials(
        spec,
        phase2g_best_summary=previous_reference_artifacts["best_pc_summary"],
        phase2g_study_config=previous_reference_artifacts["study_config"],
        extension_override=pc_boundary_extensions_override,
    )
    mlp_trials, mlp_boundary_report = build_mlp_boundary_trials(
        spec,
        phase2g_best_summary=previous_reference_artifacts["best_mlp_summary"],
        phase2g_study_config=previous_reference_artifacts["study_config"],
        extension_override=mlp_boundary_extensions_override,
    )
    run_dir = _prepare_run_dir(
        _resolve_run_root(output_root, benchmark_name, resolved_run_id, output_layout)
    )

    study_config = _study_config_payload(
        spec,
        run_id=resolved_run_id,
        output_layout=output_layout,
        split=split,
        previous_reference_artifacts=previous_reference_artifacts,
        pc_boundary_report=pc_boundary_report,
        mlp_boundary_report=mlp_boundary_report,
    )
    _write_json(run_dir / "study_config.json", study_config)

    pc_search_rows: list[dict[str, Any]] = []
    for trial in pc_trials:
        trial_spec = replace(
            spec,
            epochs=trial.epochs,
            pc_training=trial.to_pc_training_spec(),
        )
        trial_dir = run_dir / "pc_trials" / trial.config_id
        try:
            result = run_pc_benchmark(
                trial_spec,
                output_root=run_dir / "pc_trials",
                run_id=resolved_run_id,
                plot_energy=plot_energy,
                output_layout="single_dir",
                experiment_name=trial.config_id,
                split=split,
            )
            pc_search_rows.append(
                _pc_row_from_summary(
                    trial,
                    result.summary,
                    summary_path=(Path("pc_trials") / trial.config_id / "summary.json").as_posix(),
                )
            )
        except Exception as exc:
            _clean_failed_trial_dir(trial_dir)
            pc_search_rows.append(_failed_pc_row(trial, spec, split, exc))

    mlp_search_rows: list[dict[str, Any]] = []
    for trial in mlp_trials:
        trial_spec = replace(
            spec,
            epochs=trial.epochs,
            mlp_training=trial.to_mlp_training_spec(),
        )
        trial_dir = run_dir / "mlp_trials" / trial.config_id
        try:
            summary = _run_mlp_trial(
                trial_spec,
                trial,
                trial_root=trial_dir,
                run_id=resolved_run_id,
                output_layout="single_dir",
                split=split,
            )
            mlp_search_rows.append(
                _mlp_row_from_summary(
                    trial,
                    summary,
                    summary_path=(Path("mlp_trials") / trial.config_id / "summary.json").as_posix(),
                )
            )
        except Exception as exc:
            _clean_failed_trial_dir(trial_dir)
            mlp_search_rows.append(_failed_mlp_row(trial, spec, split, exc))

    pc_search_rows = _rank_rows(pc_search_rows, spec.primary_metric_name)
    mlp_search_rows = _rank_rows(mlp_search_rows, spec.primary_metric_name)
    _write_search_results(run_dir / "pc_boundary_results.csv", pc_search_rows)
    _write_search_results(run_dir / "mlp_boundary_results.csv", mlp_search_rows)

    best_pc_row = _select_best_row(pc_search_rows)
    best_mlp_row = _select_best_row(mlp_search_rows)
    best_pc_config_summary = _family_best_summary(
        benchmark_name=benchmark_name,
        run_id=resolved_run_id,
        task_name=spec.task_name,
        metric_name=spec.primary_metric_name,
        model_family="predictive_coding",
        previous_best_summary=previous_reference_artifacts["best_pc_summary"],
        best_row=best_pc_row,
        search_rows=pc_search_rows,
        boundary_report=pc_boundary_report,
    )
    best_mlp_config_summary = _family_best_summary(
        benchmark_name=benchmark_name,
        run_id=resolved_run_id,
        task_name=spec.task_name,
        metric_name=spec.primary_metric_name,
        model_family="mlp",
        previous_best_summary=previous_reference_artifacts["best_mlp_summary"],
        best_row=best_mlp_row,
        search_rows=mlp_search_rows,
        boundary_report=mlp_boundary_report,
    )
    aggregate_summary = _build_aggregate_summary(
        benchmark_name=benchmark_name,
        run_id=resolved_run_id,
        task_name=spec.task_name,
        metric_name=spec.primary_metric_name,
        previous_reference_artifacts=previous_reference_artifacts,
        pc_search_rows=pc_search_rows,
        mlp_search_rows=mlp_search_rows,
        pc_boundary_report=pc_boundary_report,
        mlp_boundary_report=mlp_boundary_report,
    )

    _write_json(run_dir / "best_pc_config_summary.json", best_pc_config_summary)
    _write_json(run_dir / "best_mlp_config_summary.json", best_mlp_config_summary)
    _write_json(run_dir / "aggregate_summary.json", aggregate_summary)

    return Phase2G1BoundaryCheckRunResult(
        run_dir=run_dir,
        study_config=study_config,
        pc_search_rows=pc_search_rows,
        mlp_search_rows=mlp_search_rows,
        aggregate_summary=aggregate_summary,
        best_pc_config_summary=best_pc_config_summary,
        best_mlp_config_summary=best_mlp_config_summary,
    )
