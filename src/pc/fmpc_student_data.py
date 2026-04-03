from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .fmpc_protocol import resolve_fmpc_teacher_manifest_path


@dataclass(frozen=True)
class FMPCStudentSplit:
    """Offline FMPC-v0 student supervision for one split.

    Shape contract:
    - `sample_indices`: `(batch,)`
    - `target_onehot`: `(batch, target_dim)`
    - `z0`: `(batch, z_dim)`
    - `z_star`: `(batch, z_dim)`
    - `delta_z`: `(batch, z_dim)`
    - `student_inputs`: `(batch, z_dim + target_dim)`

    Dtype contract:
    - dense arrays are `float64`
    - `sample_indices` is `int64`
    """

    split_name: str
    sample_indices: np.ndarray
    target_onehot: np.ndarray
    z0: np.ndarray
    z_star: np.ndarray
    delta_z: np.ndarray
    student_inputs: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class FMPCStudentDataset:
    """Validated offline FMPC-v0 student dataset built from teacher-only artifacts."""

    dataset_name: str
    schema_version: str
    teacher_manifest_path: Path
    teacher_checkpoint_path: Path | None
    teacher_mode: str
    teacher_target_semantics: str
    student_input_definition: str
    student_target_definition: str
    z_dim: int
    target_dim: int
    train: FMPCStudentSplit
    val: FMPCStudentSplit
    test: FMPCStudentSplit
    metadata: dict[str, Any]


def build_fmpc_student_inputs(
    z0: np.ndarray,
    target_onehot: np.ndarray,
) -> np.ndarray:
    """Concatenate `z0` and one-hot targets for the offline student.

    Shape contract:
    - `z0`: `(batch, z_dim)`
    - `target_onehot`: `(batch, target_dim)`
    - returns: `(batch, z_dim + target_dim)`

    Dtype contract:
    - output dtype is always `float64`
    """

    z0_array = np.asarray(z0, dtype=np.float64)
    target_array = np.asarray(target_onehot, dtype=np.float64)
    if z0_array.ndim != 2:
        raise ValueError("z0 must be shaped (batch, z_dim).")
    if target_array.ndim != 2:
        raise ValueError("target_onehot must be shaped (batch, target_dim).")
    if z0_array.shape[0] != target_array.shape[0]:
        raise ValueError("z0 and target_onehot must share the same batch dimension.")
    return np.concatenate([z0_array, target_array], axis=1).astype(np.float64, copy=False)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_split_path(manifest_path: Path, split_payload: dict[str, Any]) -> Path:
    path_value = split_payload.get("relative_path", split_payload.get("path"))
    if path_value is None:
        raise ValueError("Split manifest entry must contain relative_path or path.")
    split_path = Path(str(path_value))
    if split_path.is_absolute():
        return split_path
    return (manifest_path.parent / split_path).resolve()


def _resolve_teacher_checkpoint_path(
    manifest_path: Path,
    manifest: dict[str, Any],
) -> Path | None:
    checkpoint_payload = manifest.get("teacher_checkpoint")
    if not isinstance(checkpoint_payload, dict):
        return None
    checkpoint_reference = checkpoint_payload.get("relative_path", checkpoint_payload.get("path"))
    if checkpoint_reference is None:
        return None
    checkpoint_path = Path(str(checkpoint_reference))
    if checkpoint_path.is_absolute():
        return checkpoint_path
    return (manifest_path.parent / checkpoint_path).resolve()


def _validate_one_hot(target_onehot: np.ndarray, split_name: str) -> None:
    if target_onehot.ndim != 2:
        raise ValueError(f"{split_name}.targets must be shaped (batch, target_dim).")
    if not np.all(np.logical_or(target_onehot == 0.0, target_onehot == 1.0)):
        raise ValueError(f"{split_name}.targets must be one-hot encoded with 0/1 entries.")
    row_sums = np.sum(target_onehot, axis=1)
    if not np.allclose(row_sums, 1.0):
        raise ValueError(f"{split_name}.targets must contain exactly one active class per row.")


def _load_split(
    manifest_path: Path,
    split_name: str,
    split_payload: dict[str, Any],
    *,
    expected_target_dim: int | None,
    expected_z_dim: int | None,
) -> FMPCStudentSplit:
    split_path = _resolve_split_path(manifest_path, split_payload)
    with np.load(split_path) as arrays:
        sample_indices = np.asarray(arrays["sample_indices"])
        target_onehot = np.asarray(arrays["targets"])
        z0 = np.asarray(arrays["z0"])
        z_star = np.asarray(arrays["z_star"])

    if sample_indices.ndim != 1:
        raise ValueError(f"{split_name}.sample_indices must be shaped (batch,).")
    if target_onehot.dtype != np.float64:
        raise ValueError(f"{split_name}.targets must be float64.")
    if z0.dtype != np.float64 or z_star.dtype != np.float64:
        raise ValueError(f"{split_name}.z0 and {split_name}.z_star must be float64.")
    if z0.ndim != 2 or z_star.ndim != 2:
        raise ValueError(f"{split_name}.z0 and {split_name}.z_star must be shaped (batch, z_dim).")
    if z0.shape != z_star.shape:
        raise ValueError(f"{split_name}.z0 and {split_name}.z_star must share the same shape.")
    if target_onehot.shape[0] != z0.shape[0] or sample_indices.shape[0] != z0.shape[0]:
        raise ValueError(
            f"{split_name} arrays must agree on the batch dimension across sample_indices, "
            "targets, z0, and z_star."
        )

    _validate_one_hot(target_onehot, split_name)

    target_dim = int(target_onehot.shape[1])
    z_dim = int(z0.shape[1])
    if expected_target_dim is not None and target_dim != expected_target_dim:
        raise ValueError(
            f"{split_name}.targets feature dimension must be {expected_target_dim}, received {target_dim}."
        )
    if expected_z_dim is not None and z_dim != expected_z_dim:
        raise ValueError(f"{split_name}.z feature dimension must be {expected_z_dim}, received {z_dim}.")

    delta_z = (z_star - z0).astype(np.float64, copy=False)
    student_inputs = build_fmpc_student_inputs(z0, target_onehot)

    return FMPCStudentSplit(
        split_name=split_name,
        sample_indices=sample_indices.astype(np.int64, copy=False),
        target_onehot=target_onehot.astype(np.float64, copy=False),
        z0=z0.astype(np.float64, copy=False),
        z_star=z_star.astype(np.float64, copy=False),
        delta_z=delta_z,
        student_inputs=student_inputs,
        metadata={
            "path": str(split_path),
            "teacher_backend": split_payload["teacher_backend"],
            "teacher_steps": int(split_payload["teacher_steps"]),
            "teacher_mode": split_payload["teacher_mode"],
            "teacher_target_semantics": split_payload["teacher_target_semantics"],
            "num_samples": int(sample_indices.shape[0]),
            "delta_z_l2_mean": float(split_payload.get("delta_z_l2_mean", np.mean(np.linalg.norm(delta_z, axis=1)))),
            "delta_z_rms": float(split_payload.get("delta_z_rms", np.sqrt(np.mean(delta_z**2)))),
            "delta_z_max_abs": float(split_payload.get("delta_z_max_abs", np.max(np.abs(delta_z)))),
        },
    )


def load_fmpc_student_dataset(
    path: str | Path,
    *,
    expected_dataset_name: str = "digits",
) -> FMPCStudentDataset:
    """Load and validate offline FMPC-v0 student supervision from Phase 4 teacher artifacts.

    Accepted paths:
    - a `manifest.json` file
    - a `teacher_targets/` directory
    - an FMPC preparation run directory containing `teacher_targets/manifest.json`

    Shape contract:
    - returns split objects whose dense arrays are batch-first
    - `student_inputs` is always `concat([z0, target_onehot])`

    Dtype contract:
    - dense arrays are `float64`
    - sample indices are `int64`
    """

    manifest_path = resolve_fmpc_teacher_manifest_path(path)
    manifest = _read_json(manifest_path)
    teacher_checkpoint_path = _resolve_teacher_checkpoint_path(manifest_path, manifest)

    dataset_name = str(manifest["dataset_name"])
    if dataset_name != expected_dataset_name:
        raise ValueError(
            f"Expected dataset_name '{expected_dataset_name}', received '{dataset_name}'."
        )
    teacher_mode = str(manifest["teacher_mode"])
    teacher_target_semantics = str(manifest["teacher_target_semantics"])
    if teacher_mode != "train":
        raise ValueError("Offline FMPC student targets must come from train-mode teacher export.")
    if teacher_target_semantics != "target-clamped supervised teacher":
        raise ValueError(
            "Offline FMPC student targets must use target-clamped supervised teacher semantics."
        )

    split_payloads = manifest["splits"]
    train_split = _load_split(
        manifest_path,
        "train",
        split_payloads["train"],
        expected_target_dim=None,
        expected_z_dim=None,
    )
    val_split = _load_split(
        manifest_path,
        "val",
        split_payloads["val"],
        expected_target_dim=int(train_split.target_onehot.shape[1]),
        expected_z_dim=int(train_split.z0.shape[1]),
    )
    test_split = _load_split(
        manifest_path,
        "test",
        split_payloads["test"],
        expected_target_dim=int(train_split.target_onehot.shape[1]),
        expected_z_dim=int(train_split.z0.shape[1]),
    )

    return FMPCStudentDataset(
        dataset_name=dataset_name,
        schema_version=str(manifest.get("schema_version", "phase4_legacy_absolute_v0")),
        teacher_manifest_path=manifest_path,
        teacher_checkpoint_path=teacher_checkpoint_path,
        teacher_mode=teacher_mode,
        teacher_target_semantics=teacher_target_semantics,
        student_input_definition="concat([z0, target_onehot])",
        student_target_definition="delta_z = z_star - z0",
        z_dim=int(train_split.z0.shape[1]),
        target_dim=int(train_split.target_onehot.shape[1]),
        train=train_split,
        val=val_split,
        test=test_split,
        metadata={
            "schema_version": str(manifest.get("schema_version", "phase4_legacy_absolute_v0")),
            "teacher_backend": manifest["teacher_backend"],
            "teacher_steps": int(manifest["teacher_steps"]),
            "teacher_export_batch_size": int(manifest["teacher_export_batch_size"]),
            "export_trajectory": bool(manifest["export_trajectory"]),
            "teacher_checkpoint": manifest.get("teacher_checkpoint"),
        },
    )
