from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Literal

import numpy as np

from ..energy import compute_cache, total_energy
from .fmpc_student_data import FMPCStudentDataset, FMPCStudentSplit, load_fmpc_student_dataset
from .fmpc_protocol import load_prepared_teacher_runtime
from ..inference import build_clamped_mask, initialize_states
from ..metrics import (
    energy_gap_to_teacher,
    hidden_state_l2_distance,
    hidden_state_rms_gap,
    regression_mse,
    state_update_direction_cosine,
)
from ..minibatch import iter_minibatches
from ..mlp_baseline import MLPNetwork, init_mlp_baseline_layers
from ..real_pc import OutputLayout, RealPCConfig, RealPCRunResult, run_real_pc_experiment
from ..state_io import unflatten_hidden_states
from ..toy_data import SupervisedDataSplit
from ..utils import set_seed


# Loading an exact teacher checkpoint should reproduce exported z0 / z_star up to
# normal float64 save/load noise. This tolerance is intentionally strict.
_TEACHER_CHECKPOINT_ATOL = 1e-12
_LEGACY_TEACHER_RETRAIN_ATOL = 5e-3


@dataclass
class FMPCStudentConfig:
    """Configuration for the offline FMPC-v0 student on digits."""

    experiment_name: str = "fmpc_v0_student"
    dataset_name: str = "digits"
    teacher_preparation_path: str | Path = "outputs/fmpc_v0_prepare_digits"
    run_seed: int = 0
    model_init_seed: int = 0
    batch_order_seed: int = 0
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: OutputLayout = "single_dir"
    hidden_dims: tuple[int, ...] = (64,)
    hidden_activation: str = "tanh"
    output_activation: str = "identity"
    weight_scale: float = 0.05
    eta_w: float = 0.05
    eta_b: float | None = None
    epochs: int = 20
    batch_size: int = 64
    shuffle_batches: bool = True
    save_split_predictions: bool = False
    allow_teacher_retrain: bool = False
    refinement_enabled: bool = False
    refinement_steps: int = 0
    refinement_gamma: float = 0.0
    logging: dict[str, Any] = field(default_factory=dict)

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_{self.run_seed}"


@dataclass
class FMPCStudentRunResult:
    """Materialized outputs of an offline FMPC-v0 student run."""

    run_dir: Path
    config: dict[str, Any]
    epoch_metrics: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass
class FMPCStudentTransporter:
    """Offline NumPy student transporter for endpoint hidden-state prediction.

    Shape contract:
    - `student_inputs`: `(batch, z_dim + target_dim)`
    - `delta_z_hat`: `(batch, z_dim)`
    - `z_hat`: `(batch, z_dim)`
    """

    network: MLPNetwork
    z_dim: int
    target_dim: int

    def predict_delta_z(self, student_inputs: np.ndarray) -> np.ndarray:
        """Predict `delta_z_hat` from batch-first student inputs."""
        inputs = np.asarray(student_inputs, dtype=np.float64)
        if inputs.ndim != 2:
            raise ValueError("student_inputs must be shaped (batch, z_dim + target_dim).")
        expected_dim = self.z_dim + self.target_dim
        if inputs.shape[1] != expected_dim:
            raise ValueError(f"student_inputs feature dimension must be {expected_dim}.")
        predictions = self.network.predict(inputs)
        if predictions.shape != (inputs.shape[0], self.z_dim):
            raise RuntimeError("Student transporter produced an unexpected delta_z_hat shape.")
        return np.asarray(predictions, dtype=np.float64)

    def predict_z_hat(self, split: FMPCStudentSplit) -> np.ndarray:
        """Return `z_hat = z0 + delta_z_hat` for one validated offline student split."""
        delta_z_hat = self.predict_delta_z(split.student_inputs)
        return (split.z0 + delta_z_hat).astype(np.float64, copy=False)

    def train_batch(self, student_inputs: np.ndarray, delta_z: np.ndarray) -> float:
        """Train the student on one batch and return batch MSE on `delta_z`."""
        result = self.network.train_batch(student_inputs, delta_z)
        return float(result.loss)


@dataclass(frozen=True)
class _TeacherSplitReference:
    split_name: str
    x: np.ndarray
    y: np.ndarray
    teacher_energy: float
    teacher_wall_time_seconds: float
    teacher_export_batch_size: int


@dataclass(frozen=True)
class _StudentSplitEvaluation:
    delta_mse: float
    state_l2_gap: float
    state_rms_gap: float
    teacher_energy: float
    student_energy: float
    energy_gap_to_teacher: float
    update_direction_cosine: float | None
    transport_wall_time_seconds: float
    teacher_wall_time_seconds: float
    speedup_vs_teacher: float | None
    delta_z_hat: np.ndarray
    z_hat: np.ndarray


def init_fmpc_student_transporter(
    *,
    z_dim: int,
    target_dim: int,
    hidden_dims: tuple[int, ...],
    hidden_activation: str,
    output_activation: str,
    weight_scale: float,
    eta_w: float,
    eta_b: float | None,
    seed: int,
) -> FMPCStudentTransporter:
    """Initialize a NumPy FMPC student transporter.

    Shape contract:
    - transporter input dim is `z_dim + target_dim`
    - transporter output dim is `z_dim`
    """

    input_dim = int(z_dim + target_dim)
    layer_dims = (input_dim, *hidden_dims, int(z_dim))
    network = MLPNetwork(
        layers=init_mlp_baseline_layers(
            layer_dims,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            weight_scale=weight_scale,
            seed=seed,
            dtype=np.float64,
        ),
        eta_w=eta_w,
        eta_b=eta_b,
    )
    return FMPCStudentTransporter(network=network, z_dim=int(z_dim), target_dim=int(target_dim))


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
        import shutil

        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, indent=2)


def _relative_artifact_reference(from_dir: Path, target: str | Path | None) -> str | None:
    if target is None:
        return None
    return Path(
        os.path.relpath(Path(target).resolve(), start=from_dir.resolve())
    ).as_posix()


def _write_epoch_metrics(path: Path, rows: list[dict[str, Any]], num_layers: int) -> None:
    fieldnames = [
        "epoch",
        "batch_size",
        "batches_per_epoch",
        "train_loss",
        "val_loss",
        "train_state_l2_gap",
        "train_state_rms_gap",
        "val_state_l2_gap",
        "val_state_rms_gap",
        "train_teacher_energy",
        "train_student_energy",
        "train_energy_gap_to_teacher",
        "val_teacher_energy",
        "val_student_energy",
        "val_energy_gap_to_teacher",
        "train_update_direction_cosine",
        "val_update_direction_cosine",
        "train_transport_wall_time_seconds",
        "val_transport_wall_time_seconds",
        "train_teacher_wall_time_seconds",
        "val_teacher_wall_time_seconds",
        "train_speedup_vs_teacher",
        "val_speedup_vs_teacher",
    ]
    fieldnames.extend([f"weight_norm_l{layer_index}" for layer_index in range(1, num_layers + 1)])
    fieldnames.extend([f"bias_norm_l{layer_index}" for layer_index in range(1, num_layers + 1)])

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_split_predictions(path: Path, split: FMPCStudentSplit, evaluation: _StudentSplitEvaluation) -> None:
    np.savez_compressed(
        path,
        sample_indices=split.sample_indices,
        target_onehot=split.target_onehot,
        z0=split.z0,
        z_star=split.z_star,
        delta_z=split.delta_z,
        delta_z_hat=evaluation.delta_z_hat,
        z_hat=evaluation.z_hat,
    )


def _snapshot_parameters(model: FMPCStudentTransporter) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(layer.weight.copy(), layer.bias.copy()) for layer in model.network.layers]


def _restore_parameters(model: FMPCStudentTransporter, snapshot: list[tuple[np.ndarray, np.ndarray]]) -> None:
    if len(snapshot) != len(model.network.layers):
        raise ValueError("Parameter snapshot must align with transporter layers.")
    for layer, (weight, bias) in zip(model.network.layers, snapshot, strict=True):
        layer.weight = weight.copy()
        layer.bias = bias.copy()


def _parameter_norms(model: FMPCStudentTransporter) -> dict[str, list[float]]:
    return {
        "weight_norms": [float(np.linalg.norm(layer.weight)) for layer in model.network.layers],
        "bias_norms": [float(np.linalg.norm(layer.bias)) for layer in model.network.layers],
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _teacher_model_dir_from_manifest(manifest_path: Path) -> Path:
    teacher_model_dir = manifest_path.parent.parent / "teacher_model"
    if not (teacher_model_dir / "config.json").exists():
        raise ValueError(
            "Offline FMPC student requires a preparation run directory with teacher_model/config.json."
        )
    return teacher_model_dir


def _real_pc_config_from_saved_artifact(path: Path) -> RealPCConfig:
    payload = _read_json(path)
    split_fractions = payload["data"]["split_fractions"]
    sigma2_payload = payload["model"]["sigma2"]
    sigma2: float | tuple[float, ...]
    if isinstance(sigma2_payload, list):
        sigma2 = tuple(float(value) for value in sigma2_payload)
    else:
        sigma2 = float(sigma2_payload)
    return RealPCConfig(
        experiment_name=str(payload["experiment_name"]),
        dataset_name=str(payload["data"]["dataset_name"]),
        task_name=str(payload["task"]["name"]),
        run_seed=int(payload["run_seed"]),
        data_seed=int(payload["data_seed"]),
        model_init_seed=int(payload["model_init_seed"]),
        batch_order_seed=int(payload["batch_order_seed"]),
        output_root="outputs",
        run_id=str(payload["run_id"]),
        output_layout=str(payload["logging"]["output_layout"]),
        plot_curves=bool(payload["logging"]["plot_curves"]),
        train_fraction=float(split_fractions["train"]),
        val_fraction=float(split_fractions["val"]),
        test_fraction=float(split_fractions["test"]),
        layer_dims=tuple(int(value) for value in payload["model"]["layer_dims"]),
        hidden_activation=str(payload["model"]["hidden_activation"]),
        output_activation=str(payload["model"]["output_activation"]),
        weight_scale=float(payload["model"]["weight_scale"]),
        sigma2=sigma2,
        eta_x=float(payload["training"]["eta_x"]),
        eta_w=float(payload["training"]["eta_w"]),
        eta_b=float(payload["training"]["eta_b"]),
        train_steps=int(payload["training"]["train_steps"]),
        eval_steps=int(payload["training"]["eval_steps"]),
        inference_backend=str(payload["training"]["inference_backend"]),
        inference_method=payload["training"]["inference_method"],
        state_init=str(payload["training"]["state_init"]),
        epochs=int(payload["training"]["epochs"]),
        batch_size=int(payload["training"]["batch_size"]),
        shuffle_batches=bool(payload["training"]["shuffle_batches"]),
        teacher_reference_backend=None,
        teacher_reference_eval_steps=None,
        logging={},
    )


def _reconstruct_teacher_runtime(dataset: FMPCStudentDataset) -> tuple[PCNetwork, SupervisedDataSplit]:
    from ..models import PCNetwork

    teacher_model_dir = _teacher_model_dir_from_manifest(dataset.teacher_manifest_path)
    teacher_config = _real_pc_config_from_saved_artifact(teacher_model_dir / "config.json")

    with TemporaryDirectory(prefix="fmpc_teacher_rebuild_") as temp_dir:
        rebuilt_config = RealPCConfig(
            **{
                **teacher_config.__dict__,
                "output_root": temp_dir,
                "experiment_name": "teacher_rebuild",
                "run_id": "teacher_rebuild",
                "output_layout": "single_dir",
                "plot_curves": False,
                "logging": {},
            }
        )
        result: RealPCRunResult = run_real_pc_experiment(rebuilt_config, return_runtime_objects=True)

    if result.model is None or result.split is None:
        raise RuntimeError("Teacher reconstruction must return runtime objects.")
    return result.model, result.split


def _load_teacher_runtime(
    dataset: FMPCStudentDataset,
    *,
    allow_teacher_retrain: bool,
) -> tuple[PCNetwork, SupervisedDataSplit, bool]:
    if dataset.teacher_checkpoint_path is not None:
        try:
            model, split = load_prepared_teacher_runtime(dataset.teacher_manifest_path)
            return model, split, False
        except FileNotFoundError:
            if not allow_teacher_retrain:
                raise FileNotFoundError(
                    "Offline FMPC student evaluation requires an exact serialized teacher checkpoint. "
                    "Re-run teacher preparation with checkpoint export, or set allow_teacher_retrain=True "
                    "only as an explicit legacy fallback."
                ) from None
    if allow_teacher_retrain:
        model, split = _reconstruct_teacher_runtime(dataset)
        return model, split, True
    raise FileNotFoundError(
        "Offline FMPC student evaluation requires an exact serialized teacher checkpoint. "
        "Re-run teacher preparation with checkpoint export, or set allow_teacher_retrain=True "
        "only as an explicit legacy fallback."
    )


def _teacher_split_arrays(
    split: SupervisedDataSplit,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if split_name == "train":
        return (
            np.asarray(split.x_train, dtype=np.float64),
            np.asarray(split.y_train, dtype=np.float64),
            np.asarray(split.metadata["train_indices"], dtype=np.int64),
        )
    if split_name == "val":
        return (
            np.asarray(split.x_val, dtype=np.float64),
            np.asarray(split.y_val, dtype=np.float64),
            np.asarray(split.metadata["val_indices"], dtype=np.int64),
        )
    if split_name == "test":
        return (
            np.asarray(split.x_test, dtype=np.float64),
            np.asarray(split.y_test, dtype=np.float64),
            np.asarray(split.metadata["test_indices"], dtype=np.int64),
        )
    raise ValueError(f"Unsupported split_name '{split_name}'.")


def _export_teacher_batch(
    teacher_model: PCNetwork,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    *,
    teacher_backend: str,
    teacher_steps: int,
):
    if teacher_backend == teacher_model.inference_backend and teacher_steps == teacher_model.train_steps:
        return teacher_model.export_teacher_targets(
            x_batch,
            y_batch,
            record_trace=True,
            record_trajectory=False,
        )
    from ..inference import run_teacher_inference_export

    return run_teacher_inference_export(
        teacher_model.layers,
        np.asarray(x_batch, dtype=np.float64),
        y=np.asarray(y_batch, dtype=np.float64),
        init=teacher_model.state_init,
        mode="train",
        eta_x=teacher_model.eta_x,
        steps=teacher_steps,
        backend=teacher_backend,
        record_trace=True,
        record_trajectory=False,
    )


def _compute_hidden_state_energy_batch(
    teacher_model: PCNetwork,
    x: np.ndarray,
    y: np.ndarray,
    z_terminal: np.ndarray,
) -> float:
    clamped_mask = build_clamped_mask(len(teacher_model.layers) + 1, mode="train")
    states_template = initialize_states(
        teacher_model.layers,
        np.asarray(x, dtype=np.float64),
        y=np.asarray(y, dtype=np.float64),
        init=teacher_model.state_init,
        mode="train",
    )
    states = unflatten_hidden_states(
        np.asarray(z_terminal, dtype=np.float64),
        states_template,
        clamped_mask,
    )
    cache = compute_cache(states, teacher_model.layers)
    return total_energy(cache, teacher_model.layers, batch_size=int(x.shape[0]))


def _compute_mean_hidden_state_energy(
    teacher_model: PCNetwork,
    x: np.ndarray,
    y: np.ndarray,
    z_terminal: np.ndarray,
    *,
    batch_size: int,
) -> float:
    energies: list[float] = []
    num_samples = int(x.shape[0])
    for start in range(0, num_samples, batch_size):
        stop = min(start + batch_size, num_samples)
        energies.append(
            _compute_hidden_state_energy_batch(
                teacher_model,
                x[start:stop],
                y[start:stop],
                z_terminal[start:stop],
            )
        )
    return float(np.mean(np.asarray(energies, dtype=np.float64)))


def _prepare_teacher_split_reference(
    teacher_model: PCNetwork,
    teacher_split: SupervisedDataSplit,
    student_split: FMPCStudentSplit,
    *,
    export_batch_size: int,
    comparison_atol: float,
) -> _TeacherSplitReference:
    x, y, sample_indices = _teacher_split_arrays(teacher_split, student_split.split_name)
    if not np.array_equal(sample_indices, student_split.sample_indices):
        raise ValueError(
            f"{student_split.split_name} sample_indices do not match the teacher split ordering."
        )
    if not np.allclose(y, student_split.target_onehot):
        raise ValueError(f"{student_split.split_name} one-hot targets do not match the teacher split.")

    teacher_start = perf_counter()
    z0_batches: list[np.ndarray] = []
    z_star_batches: list[np.ndarray] = []
    final_energies: list[float] = []
    for x_batch, y_batch in iter_minibatches(
        x,
        y,
        export_batch_size,
        shuffle=False,
    ):
        teacher_export = _export_teacher_batch(
            teacher_model,
            x_batch,
            y_batch,
            teacher_backend=str(student_split.metadata["teacher_backend"]),
            teacher_steps=int(student_split.metadata["teacher_steps"]),
        )
        z0_batches.append(teacher_export.z0)
        z_star_batches.append(teacher_export.z_star)
        final_energies.append(float(teacher_export.energy_trace[-1]))
    teacher_wall_time_seconds = float(perf_counter() - teacher_start)
    teacher_z0 = np.concatenate(z0_batches, axis=0)
    teacher_z_star = np.concatenate(z_star_batches, axis=0)

    if not np.allclose(
        teacher_z0,
        student_split.z0,
        atol=comparison_atol,
        rtol=comparison_atol,
    ):
        raise ValueError(
            f"{student_split.split_name} z0 does not match the serialized teacher checkpoint export."
        )
    if not np.allclose(
        teacher_z_star,
        student_split.z_star,
        atol=comparison_atol,
        rtol=comparison_atol,
    ):
        raise ValueError(
            f"{student_split.split_name} z_star does not match the serialized teacher checkpoint export."
        )

    teacher_energy = float(np.mean(np.asarray(final_energies, dtype=np.float64)))
    return _TeacherSplitReference(
        split_name=student_split.split_name,
        x=x,
        y=y,
        teacher_energy=teacher_energy,
        teacher_wall_time_seconds=teacher_wall_time_seconds,
        teacher_export_batch_size=export_batch_size,
    )


def _evaluate_student_split(
    delta_z_hat: np.ndarray,
    split: FMPCStudentSplit,
    reference: _TeacherSplitReference,
    teacher_model: PCNetwork,
    *,
    transport_wall_time_seconds: float,
) -> _StudentSplitEvaluation:
    z_hat = (split.z0 + np.asarray(delta_z_hat, dtype=np.float64)).astype(np.float64, copy=False)

    delta_mse = regression_mse(delta_z_hat, split.delta_z)
    state_l2_gap = hidden_state_l2_distance(z_hat, split.z_star)
    state_rms_gap = hidden_state_rms_gap(z_hat, split.z_star)
    student_energy = _compute_mean_hidden_state_energy(
        teacher_model,
        reference.x,
        reference.y,
        z_hat,
        batch_size=reference.teacher_export_batch_size,
    )
    energy_gap = energy_gap_to_teacher(student_energy, reference.teacher_energy)
    update_cosine = state_update_direction_cosine(split.z0, z_hat, split.z0, split.z_star)
    speedup_vs_teacher = None
    if transport_wall_time_seconds > 0.0:
        speedup_vs_teacher = float(reference.teacher_wall_time_seconds / transport_wall_time_seconds)

    return _StudentSplitEvaluation(
        delta_mse=delta_mse,
        state_l2_gap=state_l2_gap,
        state_rms_gap=state_rms_gap,
        teacher_energy=float(reference.teacher_energy),
        student_energy=float(student_energy),
        energy_gap_to_teacher=float(energy_gap),
        update_direction_cosine=update_cosine,
        transport_wall_time_seconds=transport_wall_time_seconds,
        teacher_wall_time_seconds=float(reference.teacher_wall_time_seconds),
        speedup_vs_teacher=speedup_vs_teacher,
        delta_z_hat=np.asarray(delta_z_hat, dtype=np.float64),
        z_hat=z_hat,
    )


def _evaluate_transporter_split(
    transporter: FMPCStudentTransporter,
    split: FMPCStudentSplit,
    reference: _TeacherSplitReference,
    teacher_model: PCNetwork,
) -> _StudentSplitEvaluation:
    transport_start = perf_counter()
    delta_z_hat = transporter.predict_delta_z(split.student_inputs)
    transport_wall_time_seconds = float(perf_counter() - transport_start)
    return _evaluate_student_split(
        delta_z_hat,
        split,
        reference,
        teacher_model,
        transport_wall_time_seconds=transport_wall_time_seconds,
    )


def _evaluate_identity_baseline_split(
    split: FMPCStudentSplit,
    reference: _TeacherSplitReference,
    teacher_model: PCNetwork,
) -> _StudentSplitEvaluation:
    identity_start = perf_counter()
    delta_z_hat = np.zeros_like(split.delta_z, dtype=np.float64)
    transport_wall_time_seconds = float(perf_counter() - identity_start)
    return _evaluate_student_split(
        delta_z_hat,
        split,
        reference,
        teacher_model,
        transport_wall_time_seconds=transport_wall_time_seconds,
    )


def load_fmpc_student_teacher_runtime(
    dataset: FMPCStudentDataset,
    *,
    allow_teacher_retrain: bool,
) -> tuple[PCNetwork, SupervisedDataSplit, bool, float]:
    """Load the exact teacher runtime used for offline FMPC student evaluation.

    Returns:
    - `teacher_model`
    - `teacher_split`
    - `used_teacher_retrain_fallback`
    - `comparison_atol`
    """

    teacher_model, teacher_split, used_teacher_retrain_fallback = _load_teacher_runtime(
        dataset,
        allow_teacher_retrain=allow_teacher_retrain,
    )
    comparison_atol = (
        _LEGACY_TEACHER_RETRAIN_ATOL if used_teacher_retrain_fallback else _TEACHER_CHECKPOINT_ATOL
    )
    return teacher_model, teacher_split, used_teacher_retrain_fallback, comparison_atol


def prepare_fmpc_student_teacher_references(
    dataset: FMPCStudentDataset,
    teacher_model: PCNetwork,
    teacher_split: SupervisedDataSplit,
    *,
    comparison_atol: float,
) -> dict[str, _TeacherSplitReference]:
    """Prepare split-aligned teacher references for offline FMPC student evaluation."""

    export_batch_size = int(dataset.metadata["teacher_export_batch_size"])
    return {
        "train": _prepare_teacher_split_reference(
            teacher_model,
            teacher_split,
            dataset.train,
            export_batch_size=export_batch_size,
            comparison_atol=comparison_atol,
        ),
        "val": _prepare_teacher_split_reference(
            teacher_model,
            teacher_split,
            dataset.val,
            export_batch_size=export_batch_size,
            comparison_atol=comparison_atol,
        ),
        "test": _prepare_teacher_split_reference(
            teacher_model,
            teacher_split,
            dataset.test,
            export_batch_size=export_batch_size,
            comparison_atol=comparison_atol,
        ),
    }


def evaluate_fmpc_delta_predictions(
    delta_z_hat: np.ndarray,
    split: FMPCStudentSplit,
    reference: _TeacherSplitReference,
    teacher_model: PCNetwork,
    *,
    transport_wall_time_seconds: float,
) -> _StudentSplitEvaluation:
    """Evaluate batch-first `delta_z_hat` predictions in the original hidden-state space."""

    return _evaluate_student_split(
        delta_z_hat,
        split,
        reference,
        teacher_model,
        transport_wall_time_seconds=transport_wall_time_seconds,
    )


def evaluate_fmpc_identity_baseline(
    split: FMPCStudentSplit,
    reference: _TeacherSplitReference,
    teacher_model: PCNetwork,
) -> _StudentSplitEvaluation:
    """Evaluate the explicit identity / zero-delta baseline `z_hat = z0`."""

    return _evaluate_identity_baseline_split(split, reference, teacher_model)


def fmpc_split_evaluation_metrics_payload(evaluation: _StudentSplitEvaluation) -> dict[str, Any]:
    """Convert one offline FMPC split evaluation into a JSON-serializable metric payload."""

    return {
        "state_l2_gap": evaluation.state_l2_gap,
        "state_rms_gap": evaluation.state_rms_gap,
        "teacher_energy": evaluation.teacher_energy,
        "predicted_energy": evaluation.student_energy,
        "energy_gap_to_teacher": evaluation.energy_gap_to_teacher,
        "update_direction_cosine": evaluation.update_direction_cosine,
        "transport_wall_time_seconds": evaluation.transport_wall_time_seconds,
        "teacher_inference_wall_time_seconds": evaluation.teacher_wall_time_seconds,
        "speedup_vs_teacher": evaluation.speedup_vs_teacher,
    }


def _config_payload(
    config: FMPCStudentConfig,
    run_id: str,
    run_dir: Path,
    dataset: FMPCStudentDataset,
    batches_per_epoch: int,
    *,
    teacher_checkpoint_loaded: bool,
    teacher_reference_comparison_atol: float,
) -> dict[str, Any]:
    return {
        "experiment_name": config.experiment_name,
        "run_id": run_id,
        "phase": "Phase 5",
        "stage": "offline_fmpc_v0_student",
        "dataset_name": config.dataset_name,
        "teacher_artifact_dir": _relative_artifact_reference(run_dir, config.teacher_preparation_path),
        "teacher_manifest_path": _relative_artifact_reference(run_dir, dataset.teacher_manifest_path),
        "teacher_checkpoint_path": _relative_artifact_reference(run_dir, dataset.teacher_checkpoint_path),
        "teacher_mode": dataset.teacher_mode,
        "teacher_target_semantics": dataset.teacher_target_semantics,
        "run_seed": config.run_seed,
        "model_init_seed": config.model_init_seed,
        "batch_order_seed": config.batch_order_seed,
        "teacher_recovery": {
            "allow_teacher_retrain": config.allow_teacher_retrain,
            "teacher_checkpoint_required_by_default": True,
            "teacher_checkpoint_loaded": teacher_checkpoint_loaded,
            "teacher_reference_comparison_atol": teacher_reference_comparison_atol,
        },
        "student": {
            "model_family": "fmpc_student",
            "hidden_dims": list(config.hidden_dims),
            "hidden_activation": config.hidden_activation,
            "output_activation": config.output_activation,
            "weight_scale": config.weight_scale,
            "student_input_definition": dataset.student_input_definition,
            "student_target_definition": dataset.student_target_definition,
            "student_output_definition": "delta_z_hat",
            "transporter_output_definition": "z_hat = z0 + delta_z_hat",
        },
        "training": {
            "loss_name": "mse_delta_z",
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "batches_per_epoch": batches_per_epoch,
            "eta_w": config.eta_w,
            "eta_b": config.eta_w if config.eta_b is None else config.eta_b,
            "shuffle_batches": config.shuffle_batches,
            "selection_metric_name": "state_rms_gap",
            "selection_metric_source": "val_metric",
            "report_metric_source": "test_metric",
        },
        "evaluation": {
            "metrics": [
                "state_l2_gap",
                "state_rms_gap",
                "teacher_energy",
                "predicted_energy",
                "energy_gap_to_teacher",
                "update_direction_cosine",
                "transport_wall_time_seconds",
                "teacher_inference_wall_time_seconds",
                "speedup_vs_teacher",
            ],
            "identity_baseline_enabled": True,
            "refinement_enabled": config.refinement_enabled,
            "refinement_steps": config.refinement_steps,
            "refinement_gamma": config.refinement_gamma,
        },
        "logging": {
            "output_root": str(config.output_root),
            "output_layout": config.output_layout,
            "save_split_predictions": config.save_split_predictions,
            **config.logging,
        },
    }


def run_fmpc_student_experiment(config: FMPCStudentConfig) -> FMPCStudentRunResult:
    """Run the minimal offline FMPC-v0 student on teacher-exported digits targets."""

    if config.dataset_name != "digits":
        raise ValueError("The minimal offline FMPC-v0 student currently supports digits only.")
    if config.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.refinement_enabled:
        raise ValueError("refinement is only a placeholder in v0 and must remain disabled.")
    if config.refinement_steps != 0 or config.refinement_gamma != 0.0:
        raise ValueError("refinement placeholders must stay at zero in the minimal v0 student.")

    set_seed(config.run_seed)
    dataset = load_fmpc_student_dataset(config.teacher_preparation_path, expected_dataset_name="digits")
    teacher_model, teacher_split, used_teacher_retrain_fallback = _load_teacher_runtime(
        dataset,
        allow_teacher_retrain=config.allow_teacher_retrain,
    )
    comparison_atol = (
        _LEGACY_TEACHER_RETRAIN_ATOL if used_teacher_retrain_fallback else _TEACHER_CHECKPOINT_ATOL
    )

    transporter = init_fmpc_student_transporter(
        z_dim=dataset.z_dim,
        target_dim=dataset.target_dim,
        hidden_dims=config.hidden_dims,
        hidden_activation=config.hidden_activation,
        output_activation=config.output_activation,
        weight_scale=config.weight_scale,
        eta_w=config.eta_w,
        eta_b=config.eta_b,
        seed=config.model_init_seed,
    )

    export_batch_size = int(dataset.metadata["teacher_export_batch_size"])
    train_reference = _prepare_teacher_split_reference(
        teacher_model,
        teacher_split,
        dataset.train,
        export_batch_size=export_batch_size,
        comparison_atol=comparison_atol,
    )
    val_reference = _prepare_teacher_split_reference(
        teacher_model,
        teacher_split,
        dataset.val,
        export_batch_size=export_batch_size,
        comparison_atol=comparison_atol,
    )
    test_reference = _prepare_teacher_split_reference(
        teacher_model,
        teacher_split,
        dataset.test,
        export_batch_size=export_batch_size,
        comparison_atol=comparison_atol,
    )

    batches_per_epoch = sum(
        1
        for _ in iter_minibatches(
            dataset.train.student_inputs,
            dataset.train.delta_z,
            config.batch_size,
            shuffle=config.shuffle_batches,
            seed=config.batch_order_seed,
        )
    )

    run_id = config.resolved_run_id()
    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, run_id, config.output_layout)
    )
    config_payload = _config_payload(
        config,
        run_id,
        run_dir,
        dataset,
        batches_per_epoch,
        teacher_checkpoint_loaded=not used_teacher_retrain_fallback,
        teacher_reference_comparison_atol=comparison_atol,
    )
    _write_json(run_dir / "config.json", config_payload)

    best_snapshot: list[tuple[np.ndarray, np.ndarray]] | None = None
    best_epoch = 0
    best_val_metric = np.inf
    best_val_loss = np.inf
    epoch_rows: list[dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        minibatches = iter_minibatches(
            dataset.train.student_inputs,
            dataset.train.delta_z,
            config.batch_size,
            shuffle=config.shuffle_batches,
            seed=config.batch_order_seed + (epoch - 1),
        )
        for x_batch, y_batch in minibatches:
            transporter.train_batch(x_batch, y_batch)

        train_eval = _evaluate_transporter_split(transporter, dataset.train, train_reference, teacher_model)
        val_eval = _evaluate_transporter_split(transporter, dataset.val, val_reference, teacher_model)
        norms = _parameter_norms(transporter)

        row: dict[str, Any] = {
            "epoch": epoch,
            "batch_size": config.batch_size,
            "batches_per_epoch": batches_per_epoch,
            "train_loss": train_eval.delta_mse,
            "val_loss": val_eval.delta_mse,
            "train_state_l2_gap": train_eval.state_l2_gap,
            "train_state_rms_gap": train_eval.state_rms_gap,
            "val_state_l2_gap": val_eval.state_l2_gap,
            "val_state_rms_gap": val_eval.state_rms_gap,
            "train_teacher_energy": train_eval.teacher_energy,
            "train_student_energy": train_eval.student_energy,
            "train_energy_gap_to_teacher": train_eval.energy_gap_to_teacher,
            "val_teacher_energy": val_eval.teacher_energy,
            "val_student_energy": val_eval.student_energy,
            "val_energy_gap_to_teacher": val_eval.energy_gap_to_teacher,
            "train_update_direction_cosine": train_eval.update_direction_cosine,
            "val_update_direction_cosine": val_eval.update_direction_cosine,
            "train_transport_wall_time_seconds": train_eval.transport_wall_time_seconds,
            "val_transport_wall_time_seconds": val_eval.transport_wall_time_seconds,
            "train_teacher_wall_time_seconds": train_eval.teacher_wall_time_seconds,
            "val_teacher_wall_time_seconds": val_eval.teacher_wall_time_seconds,
            "train_speedup_vs_teacher": train_eval.speedup_vs_teacher,
            "val_speedup_vs_teacher": val_eval.speedup_vs_teacher,
        }
        for layer_index, value in enumerate(norms["weight_norms"], start=1):
            row[f"weight_norm_l{layer_index}"] = value
        for layer_index, value in enumerate(norms["bias_norms"], start=1):
            row[f"bias_norm_l{layer_index}"] = value
        epoch_rows.append(row)

        if val_eval.state_rms_gap < best_val_metric:
            best_val_metric = val_eval.state_rms_gap
            best_val_loss = val_eval.delta_mse
            best_epoch = epoch
            best_snapshot = _snapshot_parameters(transporter)

    if best_snapshot is None:
        raise RuntimeError("No best student checkpoint was recorded.")
    _restore_parameters(transporter, best_snapshot)

    train_eval = _evaluate_transporter_split(transporter, dataset.train, train_reference, teacher_model)
    val_eval = _evaluate_transporter_split(transporter, dataset.val, val_reference, teacher_model)
    test_eval = _evaluate_transporter_split(transporter, dataset.test, test_reference, teacher_model)
    identity_train = _evaluate_identity_baseline_split(dataset.train, train_reference, teacher_model)
    identity_val = _evaluate_identity_baseline_split(dataset.val, val_reference, teacher_model)
    identity_test = _evaluate_identity_baseline_split(dataset.test, test_reference, teacher_model)

    if config.save_split_predictions:
        predictions_dir = run_dir / "split_predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        _write_split_predictions(predictions_dir / "train.npz", dataset.train, train_eval)
        _write_split_predictions(predictions_dir / "val.npz", dataset.val, val_eval)
        _write_split_predictions(predictions_dir / "test.npz", dataset.test, test_eval)

    summary = {
        "experiment_name": config.experiment_name,
        "run_id": run_id,
        "phase": "Phase 5",
        "stage": "offline_fmpc_v0_student",
        "dataset_name": config.dataset_name,
        "teacher_artifact_dir": _relative_artifact_reference(run_dir, config.teacher_preparation_path),
        "teacher_manifest_path": _relative_artifact_reference(run_dir, dataset.teacher_manifest_path),
        "teacher_checkpoint_path": _relative_artifact_reference(run_dir, dataset.teacher_checkpoint_path),
        "teacher_mode": dataset.teacher_mode,
        "teacher_target_semantics": dataset.teacher_target_semantics,
        "student_input_definition": dataset.student_input_definition,
        "student_target_definition": dataset.student_target_definition,
        "student_output_definition": "delta_z_hat",
        "transporter_output_definition": "z_hat = z0 + delta_z_hat",
        "loss_name": "mse_delta_z",
        "run_seed": config.run_seed,
        "model_init_seed": config.model_init_seed,
        "batch_order_seed": config.batch_order_seed,
        "allow_teacher_retrain": config.allow_teacher_retrain,
        "teacher_checkpoint_loaded": not used_teacher_retrain_fallback,
        "teacher_reference_comparison_atol": comparison_atol,
        "metric_name": "state_rms_gap",
        "primary_metric_name": "state_rms_gap",
        "primary_metric_higher_is_better": False,
        "selection_metric_source": "val_metric",
        "report_metric_source": "test_metric",
        "train_metric": train_eval.state_rms_gap,
        "val_metric": val_eval.state_rms_gap,
        "test_metric": test_eval.state_rms_gap,
        "train_loss": train_eval.delta_mse,
        "val_loss": val_eval.delta_mse,
        "test_loss": test_eval.delta_mse,
        "best_epoch": best_epoch,
        "best_val_metric": best_val_metric,
        "best_val_loss": best_val_loss,
        "batch_size": config.batch_size,
        "batches_per_epoch": batches_per_epoch,
        "epochs": config.epochs,
        "hidden_dims": list(config.hidden_dims),
        "hidden_activation": config.hidden_activation,
        "output_activation": config.output_activation,
        "weight_scale": config.weight_scale,
        "eta_w": config.eta_w,
        "eta_b": config.eta_w if config.eta_b is None else config.eta_b,
        "refinement_enabled": False,
        "refinement_steps": 0,
        "refinement_gamma": 0.0,
        "train_state_l2_gap": train_eval.state_l2_gap,
        "val_state_l2_gap": val_eval.state_l2_gap,
        "test_state_l2_gap": test_eval.state_l2_gap,
        "train_state_rms_gap": train_eval.state_rms_gap,
        "val_state_rms_gap": val_eval.state_rms_gap,
        "test_state_rms_gap": test_eval.state_rms_gap,
        "train_teacher_energy": train_eval.teacher_energy,
        "val_teacher_energy": val_eval.teacher_energy,
        "test_teacher_energy": test_eval.teacher_energy,
        "train_student_energy": train_eval.student_energy,
        "val_student_energy": val_eval.student_energy,
        "test_student_energy": test_eval.student_energy,
        "train_predicted_energy": train_eval.student_energy,
        "val_predicted_energy": val_eval.student_energy,
        "test_predicted_energy": test_eval.student_energy,
        "train_energy_gap_to_teacher": train_eval.energy_gap_to_teacher,
        "val_energy_gap_to_teacher": val_eval.energy_gap_to_teacher,
        "test_energy_gap_to_teacher": test_eval.energy_gap_to_teacher,
        "train_update_direction_cosine": train_eval.update_direction_cosine,
        "val_update_direction_cosine": val_eval.update_direction_cosine,
        "test_update_direction_cosine": test_eval.update_direction_cosine,
        "train_transport_wall_time_seconds": train_eval.transport_wall_time_seconds,
        "val_transport_wall_time_seconds": val_eval.transport_wall_time_seconds,
        "test_transport_wall_time_seconds": test_eval.transport_wall_time_seconds,
        "train_teacher_wall_time_seconds": train_eval.teacher_wall_time_seconds,
        "val_teacher_wall_time_seconds": val_eval.teacher_wall_time_seconds,
        "test_teacher_wall_time_seconds": test_eval.teacher_wall_time_seconds,
        "train_teacher_inference_wall_time_seconds": train_eval.teacher_wall_time_seconds,
        "val_teacher_inference_wall_time_seconds": val_eval.teacher_wall_time_seconds,
        "test_teacher_inference_wall_time_seconds": test_eval.teacher_wall_time_seconds,
        "train_speedup_vs_teacher": train_eval.speedup_vs_teacher,
        "val_speedup_vs_teacher": val_eval.speedup_vs_teacher,
        "test_speedup_vs_teacher": test_eval.speedup_vs_teacher,
        "identity_baseline": {
            "train_state_l2_gap": identity_train.state_l2_gap,
            "val_state_l2_gap": identity_val.state_l2_gap,
            "test_state_l2_gap": identity_test.state_l2_gap,
            "train_state_rms_gap": identity_train.state_rms_gap,
            "val_state_rms_gap": identity_val.state_rms_gap,
            "test_state_rms_gap": identity_test.state_rms_gap,
            "train_teacher_energy": identity_train.teacher_energy,
            "val_teacher_energy": identity_val.teacher_energy,
            "test_teacher_energy": identity_test.teacher_energy,
            "train_predicted_energy": identity_train.student_energy,
            "val_predicted_energy": identity_val.student_energy,
            "test_predicted_energy": identity_test.student_energy,
            "train_energy_gap_to_teacher": identity_train.energy_gap_to_teacher,
            "val_energy_gap_to_teacher": identity_val.energy_gap_to_teacher,
            "test_energy_gap_to_teacher": identity_test.energy_gap_to_teacher,
            "train_update_direction_cosine": identity_train.update_direction_cosine,
            "val_update_direction_cosine": identity_val.update_direction_cosine,
            "test_update_direction_cosine": identity_test.update_direction_cosine,
            "train_transport_wall_time_seconds": identity_train.transport_wall_time_seconds,
            "val_transport_wall_time_seconds": identity_val.transport_wall_time_seconds,
            "test_transport_wall_time_seconds": identity_test.transport_wall_time_seconds,
            "train_teacher_inference_wall_time_seconds": identity_train.teacher_wall_time_seconds,
            "val_teacher_inference_wall_time_seconds": identity_val.teacher_wall_time_seconds,
            "test_teacher_inference_wall_time_seconds": identity_test.teacher_wall_time_seconds,
            "train_speedup_vs_teacher": identity_train.speedup_vs_teacher,
            "val_speedup_vs_teacher": identity_val.speedup_vs_teacher,
            "test_speedup_vs_teacher": identity_test.speedup_vs_teacher,
        },
        "comparison_to_identity": {
            "val_state_rms_gap_delta_student_minus_identity": val_eval.state_rms_gap - identity_val.state_rms_gap,
            "test_state_rms_gap_delta_student_minus_identity": test_eval.state_rms_gap - identity_test.state_rms_gap,
            "student_beats_identity_on_val_metric": bool(val_eval.state_rms_gap < identity_val.state_rms_gap),
            "student_beats_identity_on_test_metric": bool(test_eval.state_rms_gap < identity_test.state_rms_gap),
        },
        "teacher_target_stats": {
            "train": {
                "delta_z_l2_mean": dataset.train.metadata["delta_z_l2_mean"],
                "delta_z_rms": dataset.train.metadata["delta_z_rms"],
                "delta_z_max_abs": dataset.train.metadata["delta_z_max_abs"],
            },
            "val": {
                "delta_z_l2_mean": dataset.val.metadata["delta_z_l2_mean"],
                "delta_z_rms": dataset.val.metadata["delta_z_rms"],
                "delta_z_max_abs": dataset.val.metadata["delta_z_max_abs"],
            },
            "test": {
                "delta_z_l2_mean": dataset.test.metadata["delta_z_l2_mean"],
                "delta_z_rms": dataset.test.metadata["delta_z_rms"],
                "delta_z_max_abs": dataset.test.metadata["delta_z_max_abs"],
            },
        },
    }

    _write_epoch_metrics(run_dir / "epoch_metrics.csv", epoch_rows, num_layers=len(transporter.network.layers))
    _write_json(run_dir / "summary.json", summary)

    return FMPCStudentRunResult(
        run_dir=run_dir,
        config=config_payload,
        epoch_metrics=epoch_rows,
        summary=summary,
    )
