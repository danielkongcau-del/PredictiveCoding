from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..datasets import load_digits_split, load_fashion_mnist_split
from ..layers import PCLayerParams
from ..minibatch import iter_minibatches
from ..models import PCNetwork
from ..real_pc import OutputLayout, RealPCConfig, RealPCRunResult, run_real_pc_experiment
from ..toy_data import SupervisedDataSplit
from ..utils import set_seed

TeacherBackend = Literal["pc_euler", "pc_rk2"]
StudentPlaceholderBackend = Literal["fmpc"]
RefinementPlaceholderBackend = Literal["none", "pc_euler", "pc_rk2"]

FMPC_PREPARATION_SCHEMA_VERSION = "phase5_portable_v1"
PC_TEACHER_CHECKPOINT_FORMAT = "pc_teacher_checkpoint_npz"


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


def resolve_fmpc_teacher_manifest_path(path: str | Path) -> Path:
    """Resolve a teacher-target manifest path from a file, directory, or preparation run dir."""
    path_obj = Path(path)
    if path_obj.is_file():
        if path_obj.name != "manifest.json":
            raise ValueError("Expected a teacher-target manifest named 'manifest.json'.")
        return path_obj
    if path_obj.is_dir():
        if (path_obj / "manifest.json").exists():
            return path_obj / "manifest.json"
        if (path_obj / "teacher_targets" / "manifest.json").exists():
            return path_obj / "teacher_targets" / "manifest.json"
    raise ValueError(
        "Expected a teacher-target manifest path, a teacher_targets directory, "
        "or an FMPC preparation run directory containing teacher_targets/manifest.json."
    )


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


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _relative_artifact_path(from_dir: Path, target_path: Path) -> str:
    return Path(os.path.relpath(target_path, start=from_dir)).as_posix()


def _teacher_pc_run_config(config: FMPCPreparationConfig, run_dir: Path) -> RealPCConfig:
    return replace(
        config.teacher_pc_config,
        dataset_name=config.dataset_name,
        output_root=run_dir,
        experiment_name="teacher_model",
        run_id="teacher_model",
        output_layout="single_dir",
    )


def _teacher_checkpoint_payload(model: PCNetwork) -> dict[str, np.ndarray]:
    layer_dims = [int(model.layers[0].weight.shape[1])]
    layer_dims.extend(int(layer.weight.shape[0]) for layer in model.layers)
    activation_names = [str(layer.activation_name) for layer in model.layers]
    sigma2_values = [float(layer.sigma2) for layer in model.layers]

    payload: dict[str, np.ndarray] = {
        "format": np.asarray(PC_TEACHER_CHECKPOINT_FORMAT),
        "dtype": np.asarray("float64"),
        "num_layers": np.asarray(len(model.layers), dtype=np.int64),
        "layer_dims": np.asarray(layer_dims, dtype=np.int64),
        "activation_names": np.asarray(activation_names),
        "sigma2": np.asarray(sigma2_values, dtype=np.float64),
        "eta_x": np.asarray(model.eta_x, dtype=np.float64),
        "eta_w": np.asarray(model.eta_w, dtype=np.float64),
        "eta_b": np.asarray(model.eta_b, dtype=np.float64),
        "train_steps": np.asarray(model.train_steps, dtype=np.int64),
        "eval_steps": np.asarray(model.eval_steps, dtype=np.int64),
        "inference_backend": np.asarray(str(model.inference_backend)),
        "inference_method": np.asarray("" if model.inference_method is None else str(model.inference_method)),
        "state_init": np.asarray(str(model.state_init)),
    }
    for layer_index, layer in enumerate(model.layers):
        payload[f"layer_{layer_index}_weight"] = np.asarray(layer.weight, dtype=np.float64)
        payload[f"layer_{layer_index}_bias"] = np.asarray(layer.bias, dtype=np.float64)
    return payload


def save_pc_teacher_checkpoint(
    model: PCNetwork,
    path: str | Path,
) -> dict[str, Any]:
    """Serialize an exact predictive-coding teacher checkpoint to a portable `.npz` file."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _teacher_checkpoint_payload(model)
    np.savez_compressed(checkpoint_path, **payload)
    return {
        "format": PC_TEACHER_CHECKPOINT_FORMAT,
        "dtype": "float64",
        "num_layers": int(len(model.layers)),
        "layer_dims": [int(payload["layer_dims"][index]) for index in range(payload["layer_dims"].shape[0])],
        "activation_names": [str(value) for value in payload["activation_names"].tolist()],
        "train_steps": int(model.train_steps),
        "eval_steps": int(model.eval_steps),
        "inference_backend": str(model.inference_backend),
        "inference_method": None if model.inference_method is None else str(model.inference_method),
        "state_init": str(model.state_init),
        "path": str(checkpoint_path),
    }


def load_pc_teacher_checkpoint(path: str | Path) -> PCNetwork:
    """Load an exact predictive-coding teacher checkpoint written by `save_pc_teacher_checkpoint`."""
    checkpoint_path = Path(path)
    with np.load(checkpoint_path, allow_pickle=False) as payload:
        checkpoint_format = str(payload["format"].item())
        if checkpoint_format != PC_TEACHER_CHECKPOINT_FORMAT:
            raise ValueError(
                f"Unsupported teacher checkpoint format '{checkpoint_format}'."
            )
        num_layers = int(payload["num_layers"])
        activation_names = [str(value) for value in payload["activation_names"].tolist()]
        sigma2_values = [float(value) for value in payload["sigma2"].tolist()]
        layers: list[PCLayerParams] = []
        for layer_index in range(num_layers):
            layers.append(
                PCLayerParams(
                    weight=np.asarray(payload[f"layer_{layer_index}_weight"], dtype=np.float64),
                    bias=np.asarray(payload[f"layer_{layer_index}_bias"], dtype=np.float64),
                    sigma2=float(sigma2_values[layer_index]),
                    activation_name=str(activation_names[layer_index]),
                )
            )
        inference_method_raw = str(payload["inference_method"].item())
        inference_method = None if inference_method_raw == "" else inference_method_raw
        return PCNetwork(
            layers=layers,
            eta_x=float(payload["eta_x"]),
            eta_w=float(payload["eta_w"]),
            eta_b=float(payload["eta_b"]),
            train_steps=int(payload["train_steps"]),
            eval_steps=int(payload["eval_steps"]),
            inference_backend=str(payload["inference_backend"].item()),
            inference_method=inference_method,  # type: ignore[arg-type]
            state_init=str(payload["state_init"].item()),
        )


def _load_split_from_teacher_config_payload(payload: dict[str, Any]) -> SupervisedDataSplit:
    data_payload = payload["data"]
    dataset_name = str(data_payload["dataset_name"])
    split_fractions = data_payload["split_fractions"]
    kwargs = {
        "split_seed": int(data_payload["split_seed"]),
        "train_fraction": float(split_fractions["train"]),
        "val_fraction": float(split_fractions["val"]),
        "test_fraction": float(split_fractions["test"]),
    }
    if dataset_name == "digits":
        return load_digits_split(**kwargs)
    if dataset_name == "fashion_mnist":
        return load_fashion_mnist_split(**kwargs)
    raise ValueError(f"Unsupported dataset_name '{dataset_name}' in teacher config payload.")


def load_prepared_teacher_runtime(path: str | Path) -> tuple[PCNetwork, SupervisedDataSplit]:
    """Load a prepared FMPC teacher exactly from its serialized checkpoint and split config."""
    manifest_path = resolve_fmpc_teacher_manifest_path(path)
    manifest = _read_json(manifest_path)
    checkpoint_payload = manifest.get("teacher_checkpoint")
    if not isinstance(checkpoint_payload, dict):
        raise ValueError("Teacher manifest is missing required teacher_checkpoint metadata.")
    checkpoint_reference = checkpoint_payload.get("relative_path") or checkpoint_payload.get("path")
    if checkpoint_reference is None:
        raise ValueError("Teacher checkpoint metadata must contain relative_path or path.")
    checkpoint_path = manifest_path.parent / Path(str(checkpoint_reference))
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found at '{checkpoint_path}'.")

    teacher_model_dir = checkpoint_path.parent
    config_path = teacher_model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Teacher config.json not found at '{config_path}'.")

    model = load_pc_teacher_checkpoint(checkpoint_path)
    split = _load_split_from_teacher_config_payload(_read_json(config_path))
    return model, split


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


def _delta_z_stats(z0: np.ndarray, z_star: np.ndarray) -> dict[str, float]:
    delta_z = np.asarray(z_star - z0, dtype=np.float64)
    sample_l2 = np.linalg.norm(delta_z, axis=1)
    return {
        "delta_z_l2_mean": float(np.mean(sample_l2)),
        "delta_z_rms": float(np.sqrt(np.mean(delta_z**2))),
        "delta_z_max_abs": float(np.max(np.abs(delta_z))),
    }


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
            from ..inference import run_teacher_inference_export

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
    delta_z_stats = _delta_z_stats(z0, z_star)

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
        "relative_path": split_path.name,
        "path": split_path.name,
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
        **delta_z_stats,
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
    if teacher_model_result.model is None:
        raise RuntimeError("Teacher preparation requires runtime model objects for checkpoint export.")
    checkpoint_path = teacher_model_result.run_dir / "checkpoint.npz"
    checkpoint_metadata = save_pc_teacher_checkpoint(teacher_model_result.model, checkpoint_path)
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
        "schema_version": FMPC_PREPARATION_SCHEMA_VERSION,
        "phase": "Phase 4",
        "protocol_name": "fmpc_v0_preparation",
        "dataset_name": config.dataset_name,
        "teacher_backend": config.teacher_export_backend,
        "teacher_steps": int(teacher_steps),
        "teacher_export_batch_size": int(export_batch_size),
        "teacher_mode": "train",
        "teacher_target_semantics": "target-clamped supervised teacher",
        "export_trajectory": bool(config.export_trajectory),
        "trajectory_includes_endpoints": bool(config.export_trajectory),
        "trajectory_axis_semantics": None if not config.export_trajectory else "(batch, step, z_dim)",
        "tau_definition": None if not config.export_trajectory else "tau_k = k / teacher_steps",
        "teacher_checkpoint": {
            "format": checkpoint_metadata["format"],
            "dtype": checkpoint_metadata["dtype"],
            "relative_path": _relative_artifact_path(teacher_targets_dir, checkpoint_path),
            "layer_dims": checkpoint_metadata["layer_dims"],
            "train_steps": checkpoint_metadata["train_steps"],
            "eval_steps": checkpoint_metadata["eval_steps"],
            "inference_backend": checkpoint_metadata["inference_backend"],
            "inference_method": checkpoint_metadata["inference_method"],
            "state_init": checkpoint_metadata["state_init"],
        },
        "splits": split_manifests,
    }
    _write_json(teacher_targets_dir / "manifest.json", teacher_targets_manifest)

    summary = {
        "phase": "Phase 4",
        "protocol_name": "fmpc_v0_preparation",
        "mode": "teacher_only_preparation",
        "dataset_name": config.dataset_name,
        "teacher_model_artifact_dir": "teacher_model",
        "teacher_summary_path": "teacher_model/summary.json",
        "teacher_checkpoint_path": "teacher_model/checkpoint.npz",
        "teacher_targets_manifest_path": "teacher_targets/manifest.json",
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
            "config_path": "config.json",
            "summary_path": "summary.json",
            "teacher_model_dir": "teacher_model",
            "teacher_targets_dir": "teacher_targets",
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
