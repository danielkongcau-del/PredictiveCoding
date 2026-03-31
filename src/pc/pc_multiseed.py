from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .benchmark_specs import PCTrainingSpec, ToyBenchmarkSpec, get_benchmark_spec, run_pc_benchmark
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
PHASE2C_TUNED_PC_TRAINING: dict[str, PCTrainingSpec] = {
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


def _resolve_multiseed_root(
    output_root: str | Path,
    benchmark_name: str,
    run_id: str,
    output_layout: OutputLayout,
) -> Path:
    experiment_name = f"pc_multiseed_{benchmark_name}"
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
        fieldnames.extend(["mse", "baseline_mse"])
    elif task_name == "classification":
        fieldnames.extend(["accuracy", "baseline_accuracy"])
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


def get_phase2c_tuned_pc_training(benchmark_name: str) -> PCTrainingSpec:
    """Return the fixed Phase 2c tuned PC config for one regression benchmark."""
    try:
        return PHASE2C_TUNED_PC_TRAINING[benchmark_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported Phase 2c benchmark '{benchmark_name}'.") from exc


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
) -> ToyBenchmarkSpec:
    return replace(
        base_spec,
        run_seed=seed,
        model_init_seed=seed,
        data_seed=base_spec.data_seed,
        pc_training=base_spec.pc_training if pc_training is None else pc_training,
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
    x: np.ndarray,
    y: np.ndarray,
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
        "data": spec.data_config(x, y),
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
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    variant_dir.mkdir(parents=True, exist_ok=True)
    set_seed(spec.run_seed)
    model = spec.build_mlp_model()
    config_payload = _mlp_config_payload(spec, run_id, variant_dir.parent, output_layout, x, y)
    _write_json(variant_dir / "config.json", config_payload)

    baseline_metric_value = spec.baseline_metric_fn(y)
    epoch_rows: list[dict[str, Any]] = []

    for epoch in range(1, spec.epochs + 1):
        batch_result = model.train_batch(x, y)
        predictions = model.predict(x)
        primary_metric_value = spec.primary_metric_fn(predictions, y)
        row: dict[str, Any] = {
            "epoch": epoch,
            "loss": batch_result.loss,
        }
        for layer_index, value in enumerate(batch_result.parameter_norms["weight_norms"], start=1):
            row[f"weight_norm_l{layer_index}"] = value
        for layer_index, value in enumerate(batch_result.parameter_norms["bias_norms"], start=1):
            row[f"bias_norm_l{layer_index}"] = value
        if spec.task_name == "regression":
            row["mse"] = primary_metric_value
            row["baseline_mse"] = baseline_metric_value
        elif spec.task_name == "classification":
            row["accuracy"] = primary_metric_value
            row["baseline_accuracy"] = baseline_metric_value
        else:
            raise ValueError(f"Unsupported task '{spec.task_name}'.")
        epoch_rows.append(row)

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
        "primary_metric_name": spec.primary_metric_name,
        "primary_metric_value": final_row[spec.primary_metric_name],
        "primary_metric_higher_is_better": spec.primary_metric_higher_is_better,
        "baseline_metric_name": spec.baseline_metric_name,
        "baseline_metric_value": baseline_metric_value,
        "best_epoch": best_row["epoch"],
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
) -> dict[str, Any]:
    tuned_training = get_phase2c_tuned_pc_training(base_spec.benchmark_name)
    return {
        "experiment_name": f"pc_multiseed_{base_spec.benchmark_name}",
        "run_id": run_id,
        "phase": "Phase 2c",
        "benchmark_name": base_spec.benchmark_name,
        "task_name": base_spec.task_name,
        "seed_values": [int(seed) for seed in seed_values],
        "seed_semantics": {
            "data_seed": "fixed",
            "run_seed": "varies_with_seed",
            "model_init_seed": "varies_with_seed",
            "notes": "This phase mainly measures initialization stability, not dataset sampling variability.",
        },
        "data_seed_fixed": base_spec.data_seed,
        "default_pc_training": _training_dict(base_spec.pc_training, epochs=base_spec.epochs),
        "tuned_pc_preset_name": "eta_w_double",
        "tuned_pc_training": _training_dict(tuned_training, epochs=base_spec.epochs),
        "mlp_training": {
            "epochs": base_spec.epochs,
            "eta_w": base_spec.mlp_training.eta_w,
            "eta_b": base_spec.mlp_training.eta_b,
        },
        "output_layout": output_layout,
    }


def _build_aggregate_summary(
    *,
    base_spec: ToyBenchmarkSpec,
    run_id: str,
    seed_values: Sequence[int],
    seed_records: list[dict[str, Any]],
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

    return {
        "experiment_name": f"pc_multiseed_{base_spec.benchmark_name}",
        "run_id": run_id,
        "phase": "Phase 2c",
        "benchmark_name": base_spec.benchmark_name,
        "task_name": base_spec.task_name,
        "primary_metric_name": base_spec.primary_metric_name,
        "primary_metric_higher_is_better": base_spec.primary_metric_higher_is_better,
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
        "tuned_pc_preset_name": "eta_w_double",
        "tie_rtol": TIE_RTOL,
        "tie_atol": TIE_ATOL,
        "notes": {
            "seed_semantics": "data_seed stays fixed while run_seed and model_init_seed vary together.",
            "interpretation": "This phase mainly measures initialization stability rather than dataset sampling variability.",
            "primary_metric_delta_mlp_minus_tuned_pc": "mlp_primary_metric_value - tuned_pc_primary_metric_value",
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
) -> PCMultiSeedRunResult:
    """Run the fixed Phase 2c multi-seed study for one regression benchmark."""
    if benchmark_name not in PHASE2C_BENCHMARK_NAMES:
        raise ValueError(
            f"Phase 2c currently supports only {PHASE2C_BENCHMARK_NAMES}, got '{benchmark_name}'."
        )

    base_spec = get_benchmark_spec(benchmark_name)
    resolved_run_id = _resolve_run_id(run_id)
    resolved_seed_values = (
        list(seed_values)
        if seed_values is not None
        else list(default_seed_values_for_benchmark(benchmark_name))
    )
    run_dir = _prepare_run_dir(
        _resolve_multiseed_root(output_root, benchmark_name, resolved_run_id, output_layout)
    )
    study_config = _build_study_config(
        base_spec,
        run_id=resolved_run_id,
        output_layout=output_layout,
        seed_values=resolved_seed_values,
    )
    _write_json(run_dir / "study_config.json", study_config)

    x, y = base_spec.make_data()
    seed_records: list[dict[str, Any]] = []
    tuned_training = get_phase2c_tuned_pc_training(benchmark_name)

    for seed_index, seed in enumerate(resolved_seed_values):
        seed_dir = run_dir / "seeds" / f"seed_{int(seed):04d}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        default_spec = _seeded_spec(base_spec, seed=int(seed))
        tuned_spec = _seeded_spec(base_spec, seed=int(seed), pc_training=tuned_training)
        mlp_spec = _seeded_spec(base_spec, seed=int(seed))

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
                x=x,
                y=y,
            )
            row["default_pc_status"] = "ok"
            row["default_pc_primary_metric_value"] = float(default_result.summary["primary_metric_value"])
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
                x=x,
                y=y,
            )
            row["tuned_pc_status"] = "ok"
            row["tuned_pc_primary_metric_value"] = float(tuned_result.summary["primary_metric_value"])
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
                x=x,
                y=y,
            )
            row["mlp_status"] = "ok"
            row["mlp_primary_metric_value"] = float(mlp_summary["primary_metric_value"])
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
