from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from .real_pc import RealPCConfig, RealPCRunResult, run_real_pc_experiment

OutputLayout = Literal["single_dir", "run_id_subdir"]
InferenceMethod = Literal["euler", "rk2"]
InferenceBackend = Literal["pc_euler", "pc_rk2"]


@dataclass(frozen=True)
class RealPCInferenceBaselineCandidate:
    """One standalone real-data PC inference-baseline trial."""

    candidate_id: str
    inference_method: InferenceMethod
    train_steps: int
    eval_steps: int | None = None
    note: str = ""

    def resolved_eval_steps(self) -> int:
        if self.eval_steps is None:
            return self.train_steps
        return self.eval_steps

    def resolved_inference_backend(self) -> InferenceBackend:
        return "pc_euler" if self.inference_method == "euler" else "pc_rk2"


@dataclass
class RealPCInferenceBaselineStudyConfig:
    """Configuration for a narrow digits PC inference-baseline study."""

    experiment_name: str = "digits_pc_inference_baselines"
    dataset_name: str = "digits"
    run_seed: int = 0
    data_seed: int = 0
    model_init_seed: int = 0
    batch_order_seed: int = 0
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: OutputLayout = "single_dir"
    plot_curves: bool = False
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    layer_dims: tuple[int, ...] = (64, 64, 10)
    hidden_activation: str = "tanh"
    output_activation: str = "identity"
    weight_scale: float = 0.05
    sigma2: float | tuple[float, ...] = 1.0
    eta_x: float = 0.10
    eta_w: float = 0.02
    eta_b: float | None = 0.02
    state_init: str = "forward"
    epochs: int = 60
    batch_size: int = 64
    shuffle_batches: bool = True
    logging: dict[str, Any] = field(default_factory=dict)
    candidates: tuple[RealPCInferenceBaselineCandidate, ...] = (
        RealPCInferenceBaselineCandidate("euler_steps_01", "euler", 1),
        RealPCInferenceBaselineCandidate("euler_steps_03", "euler", 3),
        RealPCInferenceBaselineCandidate("euler_steps_05", "euler", 5),
        RealPCInferenceBaselineCandidate("euler_steps_30", "euler", 30),
        RealPCInferenceBaselineCandidate("rk2_steps_01", "rk2", 1, note="same nominal step count as euler_1"),
        RealPCInferenceBaselineCandidate("rk2_steps_03", "rk2", 3, note="same nominal step count as euler_3"),
        RealPCInferenceBaselineCandidate("rk2_steps_05", "rk2", 5, note="same nominal step count as euler_5"),
        RealPCInferenceBaselineCandidate(
            "rk2_steps_15",
            "rk2",
            15,
            note="roughly matched gradient-evaluation budget to euler_30",
        ),
    )

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_{self.run_seed}"


@dataclass
class RealPCInferenceBaselineStudyResult:
    run_dir: Path
    study_config: dict[str, Any]
    trial_rows: list[dict[str, Any]]
    aggregate_summary: dict[str, Any]


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


def _write_trial_table(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "candidate_id",
        "inference_backend",
        "inference_method",
        "train_steps",
        "eval_steps",
        "best_epoch",
        "val_metric",
        "test_metric",
        "test_baseline_metric",
        "epochs",
        "batch_size",
        "batches_per_epoch",
        "note",
        "trial_run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _candidate_trial_config(
    config: RealPCInferenceBaselineStudyConfig,
    candidate: RealPCInferenceBaselineCandidate,
    trials_root: Path,
) -> RealPCConfig:
    return RealPCConfig(
        experiment_name=candidate.candidate_id,
        dataset_name=config.dataset_name,
        task_name="classification",
        run_seed=config.run_seed,
        data_seed=config.data_seed,
        model_init_seed=config.model_init_seed,
        batch_order_seed=config.batch_order_seed,
        output_root=trials_root,
        run_id=None,
        output_layout="single_dir",
        plot_curves=config.plot_curves,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
        layer_dims=config.layer_dims,
        hidden_activation=config.hidden_activation,
        output_activation=config.output_activation,
        weight_scale=config.weight_scale,
        sigma2=config.sigma2,
        eta_x=config.eta_x,
        eta_w=config.eta_w,
        eta_b=config.eta_b,
        train_steps=candidate.train_steps,
        eval_steps=candidate.resolved_eval_steps(),
        inference_backend=candidate.resolved_inference_backend(),
        inference_method=candidate.inference_method,
        state_init=config.state_init,
        epochs=config.epochs,
        batch_size=config.batch_size,
        shuffle_batches=config.shuffle_batches,
        logging=dict(config.logging),
    )


def _trial_row(
    candidate: RealPCInferenceBaselineCandidate,
    result: RealPCRunResult,
) -> dict[str, Any]:
    summary = result.summary
    return {
        "candidate_id": candidate.candidate_id,
        "inference_backend": candidate.resolved_inference_backend(),
        "inference_method": candidate.inference_method,
        "train_steps": candidate.train_steps,
        "eval_steps": candidate.resolved_eval_steps(),
        "best_epoch": summary["best_epoch"],
        "val_metric": summary["val_metric"],
        "test_metric": summary["test_metric"],
        "test_baseline_metric": summary["test_baseline_metric"],
        "epochs": summary["epochs"],
        "batch_size": summary["batch_size"],
        "batches_per_epoch": summary["batches_per_epoch"],
        "note": candidate.note,
        "trial_run_dir": str(result.run_dir),
    }


def _study_config_payload(config: RealPCInferenceBaselineStudyConfig, run_id: str) -> dict[str, Any]:
    return {
        "experiment_name": config.experiment_name,
        "run_id": run_id,
        "phase": "Phase 4",
        "study_type": "standalone_real_data_inference_baselines",
        "dataset_name": config.dataset_name,
        "selected_by": "val_metric",
        "primary_metric_name": "accuracy",
        "primary_metric_higher_is_better": True,
        "comparison_note": (
            "This study strengthens standalone predictive-coding inference baselines only. "
            "It is not a formal real-data comparison pipeline."
        ),
        "seeds": {
            "run_seed": config.run_seed,
            "data_seed": config.data_seed,
            "model_init_seed": config.model_init_seed,
            "batch_order_seed": config.batch_order_seed,
        },
        "base_training": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "eta_x": config.eta_x,
            "eta_w": config.eta_w,
            "eta_b": config.eta_b,
            "state_init": config.state_init,
            "layer_dims": list(config.layer_dims),
            "hidden_activation": config.hidden_activation,
            "output_activation": config.output_activation,
            "weight_scale": config.weight_scale,
            "sigma2": config.sigma2 if not isinstance(config.sigma2, tuple) else list(config.sigma2),
        },
        "candidates": [asdict(candidate) for candidate in config.candidates],
    }


def run_real_pc_inference_baseline_study(
    config: RealPCInferenceBaselineStudyConfig,
) -> RealPCInferenceBaselineStudyResult:
    """Run a narrow standalone real-data PC inference-baseline study."""
    if len(config.candidates) == 0:
        raise ValueError("At least one inference-baseline candidate is required.")

    run_id = config.resolved_run_id()
    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, run_id, config.output_layout)
    )
    trials_root = run_dir / "trials"
    trials_root.mkdir(parents=True, exist_ok=True)

    study_config = _study_config_payload(config, run_id)
    _write_json(run_dir / "study_config.json", study_config)

    trial_rows: list[dict[str, Any]] = []
    for candidate in config.candidates:
        trial_config = _candidate_trial_config(config, candidate, trials_root)
        trial_result = run_real_pc_experiment(trial_config)
        trial_rows.append(_trial_row(candidate, trial_result))

    _write_trial_table(run_dir / "trial_table.csv", trial_rows)

    sorted_rows = sorted(trial_rows, key=lambda row: (-float(row["val_metric"]), row["candidate_id"]))
    selected = sorted_rows[0]
    aggregate_summary = {
        "experiment_name": config.experiment_name,
        "run_id": run_id,
        "phase": "Phase 4",
        "summary_type": "standalone_real_data_inference_baselines",
        "dataset_name": config.dataset_name,
        "selected_by": "val_metric",
        "primary_metric_name": "accuracy",
        "primary_metric_higher_is_better": True,
        "selected_candidate_id": selected["candidate_id"],
        "selected_candidate": dict(selected),
        "num_candidates": len(trial_rows),
        "candidates": trial_rows,
        "comparison_note": (
            "Standalone inference-baseline hardening only. "
            "Not a formal real-data PC-vs-MLP comparison."
        ),
    }
    _write_json(run_dir / "aggregate_summary.json", aggregate_summary)

    return RealPCInferenceBaselineStudyResult(
        run_dir=run_dir,
        study_config=study_config,
        trial_rows=trial_rows,
        aggregate_summary=aggregate_summary,
    )
