from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .inference import (
    InferenceBackendName,
    InferenceResult,
    TeacherInferenceExport,
    build_clamped_mask,
    initialize_states,
    resolve_inference_backend_name,
    run_inference,
    run_teacher_inference_export,
)
from .layers import PCLayerParams
from .training import TrainBatchResult, fit as fit_model, train_batch as train_single_batch


@dataclass
class PCNetwork:
    """Minimal Phase 0 predictive coding network with batch-first public APIs."""

    layers: list[PCLayerParams]
    eta_x: float
    eta_w: float
    eta_b: float | None = None
    train_steps: int = 20
    eval_steps: int | None = None
    inference_backend: InferenceBackendName | str = "pc_euler"
    inference_method: Literal["euler", "rk2"] | None = None
    state_init: str = "forward"

    def __post_init__(self) -> None:
        if len(self.layers) == 0:
            raise ValueError("PCNetwork requires at least one predictive layer.")
        if self.eta_x <= 0.0 or self.eta_w <= 0.0:
            raise ValueError("eta_x and eta_w must be positive.")
        if self.train_steps < 0:
            raise ValueError("train_steps must be non-negative.")
        if self.eval_steps is None:
            self.eval_steps = self.train_steps
        if self.eval_steps < 0:
            raise ValueError("eval_steps must be non-negative.")
        if self.inference_method is not None and self.inference_backend == "pc_euler":
            resolved_backend = resolve_inference_backend_name(method=self.inference_method)
        else:
            resolved_backend = resolve_inference_backend_name(
                self.inference_backend,
                method=self.inference_method,
            )
        self.inference_backend = resolved_backend
        if resolved_backend == "pc_euler":
            self.inference_method = "euler"
        elif resolved_backend == "pc_rk2":
            self.inference_method = "rk2"
        else:
            self.inference_method = None
        if self.eta_b is None:
            self.eta_b = self.eta_w
        if self.eta_b <= 0.0:
            raise ValueError("eta_b must be positive.")

    def infer(
        self,
        x: np.ndarray,
        y: np.ndarray | None = None,
        mode: str = "predict",
        record_trace: bool = True,
    ) -> InferenceResult:
        """Infer free states for x shaped (B, d_0) and optional y shaped (B, d_L)."""
        x_array = np.asarray(x, dtype=np.float64)
        y_array = None if y is None else np.asarray(y, dtype=np.float64)
        states = initialize_states(
            self.layers,
            x_array,
            y=y_array,
            init=self.state_init,
            mode=mode,
        )
        clamped_mask = build_clamped_mask(len(self.layers) + 1, mode=mode)
        steps = self.train_steps if mode == "train" else self.eval_steps
        return run_inference(
            states,
            self.layers,
            clamped_mask,
            eta_x=self.eta_x,
            steps=steps,
            backend=self.inference_backend,
            record_trace=record_trace,
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return x^L for input x shaped (B, d_0) after prediction-mode inference."""
        result = self.infer(x, mode="predict", record_trace=False)
        return result.states[-1]

    def train_batch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        compute_post_update_energy: bool = False,
        return_teacher_export: bool = False,
        record_teacher_trajectory: bool = False,
    ) -> TrainBatchResult:
        """Train on one batch with x shaped (B, d_0) and y shaped (B, d_L)."""
        return train_single_batch(
            self,
            x,
            y,
            compute_post_update_energy=compute_post_update_energy,
            return_teacher_export=return_teacher_export,
            record_teacher_trajectory=record_teacher_trajectory,
        )

    def export_teacher_targets(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        record_trace: bool = True,
        record_trajectory: bool = False,
    ) -> TeacherInferenceExport:
        """Run the current slow teacher on a training batch and export hidden-state targets."""
        return run_teacher_inference_export(
            self.layers,
            np.asarray(x, dtype=np.float64),
            y=np.asarray(y, dtype=np.float64),
            init=self.state_init,
            mode="train",
            eta_x=self.eta_x,
            steps=self.train_steps,
            backend=self.inference_backend,
            record_trace=record_trace,
            record_trajectory=record_trajectory,
        )

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int,
        seed: int | None = None,
    ) -> dict[str, list[float]]:
        """Fit the model on full-batch x and y arrays for a fixed number of epochs."""
        return fit_model(self, x, y, epochs=epochs, seed=seed)
