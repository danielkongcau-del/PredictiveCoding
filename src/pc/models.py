from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .inference import InferenceResult, build_clamped_mask, initialize_states, run_inference
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
    ) -> TrainBatchResult:
        """Train on one batch with x shaped (B, d_0) and y shaped (B, d_L)."""
        return train_single_batch(self, x, y, compute_post_update_energy=compute_post_update_energy)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int,
        seed: int | None = None,
    ) -> dict[str, list[float]]:
        """Fit the model on full-batch x and y arrays for a fixed number of epochs."""
        return fit_model(self, x, y, epochs=epochs, seed=seed)
