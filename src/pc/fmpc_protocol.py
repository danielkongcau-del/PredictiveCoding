from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .minibatch import iter_minibatches
from .real_pc import OutputLayout, RealPCConfig, RealPCRunResult, run_real_pc_experiment
from .utils import set_seed

TeacherBackend = Literal["pc_euler", "pc_rk2"]
StudentPlaceholderBackend = Literal["fmpc"]
RefinementPlaceholderBackend = Literal["none", "pc_euler", "pc_rk2"]


@dataclass
class FMPCPreparationConfig:
    """Teacher-only FMPC-v0 preparation settings.

    This scaffold is intentionally conservative:
    - it trains a standard real-data PC teacher using the current baseline path
    - it exports teacher supervision targets under target-clamped train-mode semantics
    - it records future student/transport settings only as placeholders
    """

    dataset_name: Literal["digits", "fashion_mnist"] = "digits"
    output_root: str | Path = "outputs"
    experiment_name: str | None = None
    run_id: str | None = None
    output_layout: OutputLayout = "single_dir"
    teacher_pc_config: RealPCConfig = field(default_factory=RealPCConfig)
    teacher_export_backend: TeacherBackend = "pc_euler"
    teacher_export_steps: int | None = None
    teacher_export_batch_size: int | None = None
    export_trajectory: bool = False
    student_backend_placeholder: StudentPlaceholderBackend = "fmpc"
    student_transport_steps_placeholder: int | None = None
    optional_refinement_backend_placeholder: RefinementPlaceholderBackend = "none"
    optional_refinement_steps_placeholder: int = 0

    def resolved_experiment_name(self) -> str:
        if self.experiment_name is not None:
            return self.experiment_name
        return f"fmpc_v0_prepare_{self.dataset_name}"

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{self.dataset_name}"


@dataclass
class FMPCPreparationRunResult:
    """Materialized teacher-only FMPC-v0 preparation outputs."""

    run_dir: Path
    config: dict[str, Any]
    summary: dict[str, Any]
    teacher_model_result: RealPCRunResult
    teacher_targets_manifest: dict[str, Any]


def _resolve_run_dir(
    output_root: str | Path,
    experiment_name: str,
    run_id: str,
    output_layout: OutputLayout,
) -> Path:
    root = Path(output_root)
    if output_layout == "single_dir":
        return root / experiment_name
    if output_layout == "run_id_subdir":
        return root / experiment_name / run_id
    raise ValueError(f"Unsupported output_layout '{output_layout}'.")


def _prepare_run_dir(run_dir: Path) -> Path:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, indent=2)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _teacher_pc_run_config(config: FMPCPreparationConfig, run_dir: Path) -> RealPCConfig:
    return replace(
        config.teacher_pc_config,
        dataset_name=config.dataset_name,
        output_root=run_dir,
        experiment_name="teacher_model",
        run_id="teacher_model",
        output_layout="single_dir",
    )


def _teacher_export_steps(config: FMPCPreparationConfig, teacher_model_result: RealPCRunResult) -> int:
    if teacher_model_result.model is None:
        raise RuntimeError("teacher_model_result.model is required for teacher export.")
    if config.teacher_export_steps is not None:
        return config.teacher_export_steps
    if teacher_model_result.model.train_steps is None:
        raise ValueError("Teacher export requires a concrete train_steps value.")
    return int(teacher_model_result.model.train_steps)


def _teacher_export_batch_size(config: FMPCPreparationConfig, teacher_model_result: RealPCRunResult) -> int:
    batch_size = config.teacher_export_batch_size
    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError("teacher_export_batch_size must be positive when provided.")
        return batch_size
    return int(teacher_model_result.summary["batch_size"])


def _split_arrays(split: Any, split_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if split_name == "train":
        x_split = split.x_train
        y_split = split.y_train
        absolute_indices = np.asarray(split.metadata["train_indices"], dtype=np.int64)
    elif split_name == "val":
        x_split = split.x_val
        y_split = split.y_val
        absolute_indices = np.asarray(split.metadata["val_indices"], dtype=np.int64)
    elif split_name == "test":
        x_split = split.x_test
        y_split = split.y_test
        absolute_indices = np.asarray(split.metadata["test_indices"], dtype=np.int64)
    else:
        raise ValueError(f"Unsupported split_name '{split_name}'.")

    return (
        np.asarray(x_split, dtype=np.float64),
        np.asarray(y_split, dtype=np.float64),
        absolute_indices,
    )


def _export_teacher_targets_for_split(
    teacher_model_result: RealPCRunResult,
    *,
    split_name: str,
    teacher_backend: TeacherBackend,
    teacher_steps: int,
    export_batch_size: int,
    export_trajectory: bool,
    targets_dir: Path,
) -> dict[str, Any]:
    if teacher_model_result.model is None or teacher_model_result.split is None:
        raise RuntimeError("Teacher model runtime objects are required for FMPC preparation export.")

    model = teacher_model_result.model
    split = teacher_model_result.split
    x_split, y_split, absolute_indices = _split_arrays(split, split_name)

    sample_indices_batches: list[np.ndarray] = []
    targets_batches: list[np.ndarray] = []
    z0_batches: list[np.ndarray] = []
    z_star_batches: list[np.ndarray] = []
    trajectory_batches: list[np.ndarray] = []

    for x_batch, y_batch, batch_relative_indices in iter_minibatches(
        x_split,
        y_split,
        export_batch_size,
        shuffle=False,
        return_indices=True,
    ):
        teacher_export = model.export_teacher_targets(
            x_batch,
            y_batch,
            record_trace=False,
            record_trajectory=export_trajectory,
        )
        if teacher_backend != model.inference_backend or teacher_steps != model.train_steps:
            from .inference import run_teacher_inference_export

            teacher_export = run_teacher_inference_export(
                model.layers,
                x_batch,
                y=y_batch,
                init=model.state_init,
                mode="train",
                eta_x=model.eta_x,
                steps=teacher_steps,
                backend=teacher_backend,
                record_trace=False,
                record_trajectory=export_trajectory,
            )

        sample_indices_batches.append(absolute_indices[batch_relative_indices])
        targets_batches.append(np.asarray(y_batch, dtype=np.float64))
        z0_batches.append(teacher_export.z0)
        z_star_batches.append(teacher_export.z_star)

        if export_trajectory:
            if teacher_export.z_trajectory is None:
                raise RuntimeError("Requested trajectory export but teacher_export.z_trajectory is missing.")
            trajectory = np.stack(teacher_export.z_trajectory, axis=0).transpose(1, 0, 2)
            trajectory_batches.append(trajectory.astype(np.float64, copy=False))

    sample_indices = np.concatenate(sample_indices_batches, axis=0)
    targets = np.concatenate(targets_batches, axis=0)
    z0 = np.concatenate(z0_batches, axis=0)
    z_star = np.concatenate(z_star_batches, axis=0)

    payload: dict[str, np.ndarray] = {
        "sample_indices": sample_indices.astype(np.int64, copy=False),
        "targets": targets.astype(np.float64, copy=False),
        "z0": z0.astype(np.float64, copy=False),
        "z_star": z_star.astype(np.float64, copy=False),
    }
    if export_trajectory:
        payload["z_trajectory"] = np.concatenate(trajectory_batches, axis=0).astype(np.float64, copy=False)

    split_path = targets_dir / f"{split_name}.npz"
    np.savez_compressed(split_path, **payload)

    return {
        "split_name": split_name,
        "path": str(split_path),
        "num_samples": int(sample_indices.shape[0]),
        "sample_indices_shape": list(sample_indices.shape),
        "targets_shape": list(targets.shape),
        "z0_shape": list(z0.shape),
        "z_star_shape": list(z_star.shape),
        "z_trajectory_shape": None if "z_trajectory" not in payload else list(payload["z_trajectory"].shape),
        "dtype": "float64",
        "teacher_mode": "train",
        "teacher_target_semantics": "target-clamped supervised teacher",
        "teacher_backend": teacher_backend,
        "teacher_steps": int(teacher_steps),
    }


def _config_payload(
    config: FMPCPreparationConfig,
    run_id: str,
    teacher_pc_config: RealPCConfig,
) -> dict[str, Any]:
    return {
        "experiment_name": config.resolved_experiment_name(),
        "run_id": run_id,
        "phase": "Phase 4",
        "protocol_name": "fmpc_v0_preparation",
        "mode": "teacher_only_preparation",
        "dataset_name": config.dataset_name,
        "teacher_model_config": _jsonable(asdict(teacher_pc_config)),
        "teacher_export": {
            "teacher_backend": config.teacher_export_backend,
            "teacher_steps": config.teacher_export_steps,
            "teacher_export_batch_size": config.teacher_export_batch_size,
            "export_trajectory": config.export_trajectory,
            "teacher_mode": "train",
            "teacher_target_semantics": "target-clamped supervised teacher",
        },
        "student_placeholder": {
            "backend": config.student_backend_placeholder,
            "implemented": False,
            "transport_steps": config.student_transport_steps_placeholder,
            "optional_refinement_backend": config.optional_refinement_backend_placeholder,
            "optional_refinement_steps": config.optional_refinement_steps_placeholder,
        },
        "notes": [
            "This scaffold exports teacher-only preparation artifacts for future FMPC-v0 work.",
            "No transporter/student has been implemented yet.",
            "No FMPC result is produced by this run.",
        ],
    }


def run_fmpc_v0_preparation(config: FMPCPreparationConfig) -> FMPCPreparationRunResult:
    """Run a teacher-only FMPC-v0 preparation protocol.

    The current conservative interpretation is:
    - train a standard real-data PC teacher first
    - export `z0` and `z_star` under target-clamped train-mode teacher semantics
    - reserve student transport/refinement settings only as placeholders
    """
    set_seed(config.teacher_pc_config.run_seed)
    run_id = config.resolved_run_id()
    run_dir = _prepare_run_dir(
        _resolve_run_dir(
            config.output_root,
            config.resolved_experiment_name(),
            run_id,
            config.output_layout,
        )
    )

    teacher_pc_config = _teacher_pc_run_config(config, run_dir)
    config_payload = _config_payload(config, run_id, teacher_pc_config)
    _write_json(run_dir / "config.json", config_payload)

    teacher_model_result = run_real_pc_experiment(
        teacher_pc_config,
        return_runtime_objects=True,
    )
    teacher_steps = _teacher_export_steps(config, teacher_model_result)
    export_batch_size = _teacher_export_batch_size(config, teacher_model_result)

    teacher_targets_dir = run_dir / "teacher_targets"
    teacher_targets_dir.mkdir(parents=True, exist_ok=True)
    split_manifests = {
        split_name: _export_teacher_targets_for_split(
            teacher_model_result,
            split_name=split_name,
            teacher_backend=config.teacher_export_backend,
            teacher_steps=teacher_steps,
            export_batch_size=export_batch_size,
            export_trajectory=config.export_trajectory,
            targets_dir=teacher_targets_dir,
        )
        for split_name in ("train", "val", "test")
    }

    teacher_targets_manifest = {
        "phase": "Phase 4",
        "protocol_name": "fmpc_v0_preparation",
        "dataset_name": config.dataset_name,
        "teacher_backend": config.teacher_export_backend,
        "teacher_steps": int(teacher_steps),
        "teacher_export_batch_size": int(export_batch_size),
        "teacher_mode": "train",
        "teacher_target_semantics": "target-clamped supervised teacher",
        "export_trajectory": bool(config.export_trajectory),
        "splits": split_manifests,
    }
    _write_json(teacher_targets_dir / "manifest.json", teacher_targets_manifest)

    summary = {
        "phase": "Phase 4",
        "protocol_name": "fmpc_v0_preparation",
        "mode": "teacher_only_preparation",
        "dataset_name": config.dataset_name,
        "teacher_model_artifact_dir": str(teacher_model_result.run_dir),
        "teacher_summary_path": str(teacher_model_result.run_dir / "summary.json"),
        "teacher_targets_manifest_path": str(teacher_targets_dir / "manifest.json"),
        "teacher": {
            "training_backend": teacher_model_result.summary["inference_backend"],
            "training_eval_steps": teacher_model_result.summary["eval_steps"],
            "export_backend": config.teacher_export_backend,
            "export_steps": int(teacher_steps),
            "teacher_mode": "train",
            "teacher_target_semantics": "target-clamped supervised teacher",
        },
        "student_placeholder": {
            "backend": config.student_backend_placeholder,
            "implemented": False,
            "transport_steps": config.student_transport_steps_placeholder,
            "optional_refinement_backend": config.optional_refinement_backend_placeholder,
            "optional_refinement_steps": config.optional_refinement_steps_placeholder,
        },
        "artifacts": {
            "config_path": str(run_dir / "config.json"),
            "summary_path": str(run_dir / "summary.json"),
            "teacher_model_dir": str(teacher_model_result.run_dir),
            "teacher_targets_dir": str(teacher_targets_dir),
        },
        "notes": [
            "Teacher-only preparation scaffold for future FMPC-v0 runs.",
            "No transporter/student metrics are written.",
            "This artifact should not be read as a completed FMPC method or a formal comparison.",
        ],
    }
    _write_json(run_dir / "summary.json", summary)

    return FMPCPreparationRunResult(
        run_dir=run_dir,
        config=config_payload,
        summary=summary,
        teacher_model_result=teacher_model_result,
        teacher_targets_manifest=teacher_targets_manifest,
    )
