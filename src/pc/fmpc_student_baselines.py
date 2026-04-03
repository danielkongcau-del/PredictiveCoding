from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .fmpc_student_data import FMPCStudentSplit, build_fmpc_student_inputs
from .fmpc_student_normalization import FMPCStudentNormalizationStats
from .mlp_baseline import MLPNetwork, init_mlp_baseline_layers


def _as_batch_first(name: str, array: np.ndarray) -> np.ndarray:
    array_float = np.asarray(array, dtype=np.float64)
    if array_float.ndim != 2:
        raise ValueError(f"{name} must be shaped (batch, features).")
    return array_float


def _split_student_inputs(
    student_inputs: np.ndarray,
    *,
    z_dim: int,
    target_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    inputs = _as_batch_first("student_inputs", student_inputs)
    expected_dim = int(z_dim + target_dim)
    if inputs.shape[1] != expected_dim:
        raise ValueError(f"student_inputs feature dimension must be {expected_dim}.")
    return inputs[:, :z_dim], inputs[:, z_dim:]


@dataclass(frozen=True)
class ClassMeanDeltaStudent:
    """Class-conditional prototype baseline using only train-split mean deltas."""

    class_mean_delta: np.ndarray

    @classmethod
    def fit(cls, train_split: FMPCStudentSplit) -> ClassMeanDeltaStudent:
        targets = _as_batch_first("train_split.target_onehot", train_split.target_onehot)
        delta_z = _as_batch_first("train_split.delta_z", train_split.delta_z)
        if targets.shape[0] != delta_z.shape[0]:
            raise ValueError("target_onehot and delta_z must share the same batch dimension.")
        labels = np.argmax(targets, axis=1)
        num_classes = int(targets.shape[1])
        prototypes = np.zeros((num_classes, delta_z.shape[1]), dtype=np.float64)
        for class_index in range(num_classes):
            mask = labels == class_index
            if np.any(mask):
                prototypes[class_index] = np.mean(delta_z[mask], axis=0)
        return cls(class_mean_delta=prototypes)

    def predict_delta_z(self, split: FMPCStudentSplit) -> np.ndarray:
        labels = np.argmax(split.target_onehot, axis=1)
        return np.asarray(self.class_mean_delta[labels], dtype=np.float64)

    def predict_z_hat(self, split: FMPCStudentSplit) -> np.ndarray:
        return (split.z0 + self.predict_delta_z(split)).astype(np.float64, copy=False)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "family": "class_mean_delta",
            "class_mean_delta": self.class_mean_delta.tolist(),
        }


@dataclass(frozen=True)
class RidgeDeltaStudentConfig:
    alpha: float


@dataclass(frozen=True)
class RidgeDeltaStudent:
    """Deterministic closed-form multi-output ridge regression on endpoint targets."""

    config: RidgeDeltaStudentConfig
    normalization: FMPCStudentNormalizationStats
    coefficients: np.ndarray
    bias: np.ndarray
    z_dim: int
    target_dim: int

    @classmethod
    def fit(
        cls,
        train_split: FMPCStudentSplit,
        *,
        normalization: FMPCStudentNormalizationStats,
        config: RidgeDeltaStudentConfig,
    ) -> RidgeDeltaStudent:
        if config.alpha <= 0.0:
            raise ValueError("Ridge alpha must be positive.")
        inputs = normalization.transform_split_inputs(train_split)
        targets = normalization.transform_delta_z(train_split.delta_z)
        design = np.concatenate(
            [inputs, np.ones((inputs.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
        xtx = design.T @ design
        reg = config.alpha * np.eye(design.shape[1], dtype=np.float64)
        xty = design.T @ targets
        solution = np.linalg.solve(xtx + reg, xty)
        coefficients = np.asarray(solution[:-1, :], dtype=np.float64)
        bias = np.asarray(solution[-1, :], dtype=np.float64)
        return cls(
            config=config,
            normalization=normalization,
            coefficients=coefficients,
            bias=bias,
            z_dim=int(train_split.z0.shape[1]),
            target_dim=int(train_split.target_onehot.shape[1]),
        )

    def predict_delta_z(self, split: FMPCStudentSplit) -> np.ndarray:
        inputs = self.normalization.transform_split_inputs(split)
        delta_z_normalized = inputs @ self.coefficients + self.bias
        return self.normalization.inverse_delta_z(delta_z_normalized)

    def predict_z_hat(self, split: FMPCStudentSplit) -> np.ndarray:
        return (split.z0 + self.predict_delta_z(split)).astype(np.float64, copy=False)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "family": "ridge",
            "alpha": float(self.config.alpha),
            "normalization": self.normalization.to_jsonable(),
        }


@dataclass(frozen=True)
class StandardizedMLPStudentConfig:
    hidden_dims: tuple[int, ...]
    hidden_activation: str = "tanh"
    output_activation: str = "identity"
    weight_scale: float = 0.02
    eta_w: float = 0.01
    eta_b: float | None = None
    epochs: int = 40
    batch_size: int = 64
    shuffle_batches: bool = True


@dataclass
class StandardizedMLPStudent:
    """NumPy MLP student trained on normalized endpoint targets."""

    config: StandardizedMLPStudentConfig
    normalization: FMPCStudentNormalizationStats
    network: MLPNetwork
    z_dim: int
    target_dim: int

    @classmethod
    def initialize(
        cls,
        *,
        z_dim: int,
        target_dim: int,
        normalization: FMPCStudentNormalizationStats,
        config: StandardizedMLPStudentConfig,
        seed: int,
    ) -> StandardizedMLPStudent:
        input_dim = int(z_dim + target_dim)
        network = MLPNetwork(
            layers=init_mlp_baseline_layers(
                (input_dim, *config.hidden_dims, int(z_dim)),
                hidden_activation=config.hidden_activation,
                output_activation=config.output_activation,
                weight_scale=config.weight_scale,
                seed=seed,
                dtype=np.float64,
            ),
            eta_w=config.eta_w,
            eta_b=config.eta_b,
        )
        return cls(
            config=config,
            normalization=normalization,
            network=network,
            z_dim=int(z_dim),
            target_dim=int(target_dim),
        )

    def _normalize_student_inputs(self, student_inputs: np.ndarray) -> np.ndarray:
        z0, target_onehot = _split_student_inputs(
            student_inputs,
            z_dim=self.z_dim,
            target_dim=self.target_dim,
        )
        return self.normalization.transform_inputs(z0, target_onehot)

    def predict_delta_z(self, split: FMPCStudentSplit) -> np.ndarray:
        inputs = self.normalization.transform_split_inputs(split)
        delta_z_normalized = self.network.predict(inputs)
        return self.normalization.inverse_delta_z(delta_z_normalized)

    def predict_z_hat(self, split: FMPCStudentSplit) -> np.ndarray:
        return (split.z0 + self.predict_delta_z(split)).astype(np.float64, copy=False)

    def train_batch(self, student_inputs: np.ndarray, delta_z: np.ndarray) -> float:
        normalized_inputs = self._normalize_student_inputs(student_inputs)
        normalized_delta = self.normalization.transform_delta_z(delta_z)
        result = self.network.train_batch(normalized_inputs, normalized_delta)
        return float(result.loss)

    def parameter_norms(self) -> dict[str, list[float]]:
        return {
            "weight_norms": [float(np.linalg.norm(layer.weight)) for layer in self.network.layers],
            "bias_norms": [float(np.linalg.norm(layer.bias)) for layer in self.network.layers],
        }

    def snapshot(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(layer.weight.copy(), layer.bias.copy()) for layer in self.network.layers]

    def restore(self, snapshot: list[tuple[np.ndarray, np.ndarray]]) -> None:
        if len(snapshot) != len(self.network.layers):
            raise ValueError("Parameter snapshot must align with transporter layers.")
        for layer, (weight, bias) in zip(self.network.layers, snapshot, strict=True):
            layer.weight = weight.copy()
            layer.bias = bias.copy()

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "family": "mlp_standardized",
            "hidden_dims": list(self.config.hidden_dims),
            "hidden_activation": self.config.hidden_activation,
            "output_activation": self.config.output_activation,
            "weight_scale": float(self.config.weight_scale),
            "eta_w": float(self.config.eta_w),
            "eta_b": float(self.config.eta_w if self.config.eta_b is None else self.config.eta_b),
            "epochs": int(self.config.epochs),
            "batch_size": int(self.config.batch_size),
            "shuffle_batches": bool(self.config.shuffle_batches),
            "normalization": self.normalization.to_jsonable(),
        }


def identity_delta_prediction(split: FMPCStudentSplit) -> np.ndarray:
    """Return the explicit zero-delta endpoint baseline `delta_z_hat = 0`."""

    return np.zeros_like(split.delta_z, dtype=np.float64)
