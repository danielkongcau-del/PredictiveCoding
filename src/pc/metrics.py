from __future__ import annotations

from collections.abc import Callable

import numpy as np

MetricFn = Callable[[np.ndarray, np.ndarray], float]
BaselineMetricFn = Callable[[np.ndarray], float]


def regression_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Return mean squared error for arrays shaped (batch, features)."""
    return float(np.mean((predictions - targets) ** 2))


def regression_mean_baseline_mse(targets: np.ndarray) -> float:
    """Return the MSE of predicting the per-feature mean target value."""
    mean_prediction = np.mean(targets, axis=0, keepdims=True)
    return float(np.mean((targets - mean_prediction) ** 2))


def classification_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Return argmax accuracy for predictions and one-hot targets shaped (batch, classes)."""
    predicted_labels = np.argmax(predictions, axis=1)
    target_labels = np.argmax(targets, axis=1)
    return float(np.mean(predicted_labels == target_labels))


def majority_class_baseline_accuracy(targets: np.ndarray) -> float:
    """Return the accuracy of always predicting the majority class from one-hot targets."""
    target_labels = np.argmax(targets, axis=1)
    class_counts = np.bincount(target_labels, minlength=targets.shape[1])
    return float(np.max(class_counts) / target_labels.shape[0])


def metric_higher_is_better(metric_name: str) -> bool:
    """Return whether larger metric values indicate better model performance."""
    mapping = {
        "accuracy": True,
        "mse": False,
    }
    try:
        return mapping[metric_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported metric '{metric_name}'.") from exc


def _as_batch_first_latent(name: str, z: np.ndarray) -> np.ndarray:
    """Validate a flattened hidden-state tensor shaped (batch, hidden_dim)."""
    z_array = np.asarray(z, dtype=np.float64)
    if z_array.ndim != 2:
        raise ValueError(f"{name} must be shaped (batch, hidden_dim).")
    return z_array


def hidden_state_l2_distance(
    candidate_z: np.ndarray,
    teacher_z: np.ndarray,
) -> float:
    """Return the Frobenius-distance gap between two batch-first hidden-state tensors.

    Shape contract:
    - `candidate_z`: `(batch, hidden_dim)`
    - `teacher_z`: `(batch, hidden_dim)`
    """
    candidate = _as_batch_first_latent("candidate_z", candidate_z)
    teacher = _as_batch_first_latent("teacher_z", teacher_z)
    if candidate.shape != teacher.shape:
        raise ValueError("candidate_z and teacher_z must share the same shape.")
    return float(np.linalg.norm(candidate - teacher))


def hidden_state_rms_gap(
    candidate_z: np.ndarray,
    teacher_z: np.ndarray,
) -> float:
    """Return the root-mean-square gap between two batch-first hidden-state tensors.

    Shape contract:
    - `candidate_z`: `(batch, hidden_dim)`
    - `teacher_z`: `(batch, hidden_dim)`
    """
    candidate = _as_batch_first_latent("candidate_z", candidate_z)
    teacher = _as_batch_first_latent("teacher_z", teacher_z)
    if candidate.shape != teacher.shape:
        raise ValueError("candidate_z and teacher_z must share the same shape.")
    return float(np.sqrt(np.mean((candidate - teacher) ** 2)))


def energy_gap_to_teacher(
    candidate_energy: float,
    teacher_energy: float,
) -> float:
    """Return candidate minus teacher final energy under the same batch protocol."""
    return float(candidate_energy - teacher_energy)


def update_direction_cosine(
    candidate_direction: np.ndarray,
    teacher_direction: np.ndarray,
) -> float | None:
    """Return cosine alignment between two batch-first flattened update directions.

    Shape contract:
    - `candidate_direction`: `(batch, hidden_dim)`
    - `teacher_direction`: `(batch, hidden_dim)`

    Returns `None` when either direction has zero norm, because the cosine is not
    well defined in that case.
    """
    candidate = _as_batch_first_latent("candidate_direction", candidate_direction)
    teacher = _as_batch_first_latent("teacher_direction", teacher_direction)
    if candidate.shape != teacher.shape:
        raise ValueError("candidate_direction and teacher_direction must share the same shape.")

    candidate_flat = candidate.reshape(-1)
    teacher_flat = teacher.reshape(-1)
    candidate_norm = float(np.linalg.norm(candidate_flat))
    teacher_norm = float(np.linalg.norm(teacher_flat))
    if candidate_norm == 0.0 or teacher_norm == 0.0:
        return None

    cosine = float(np.dot(candidate_flat, teacher_flat) / (candidate_norm * teacher_norm))
    return float(np.clip(cosine, -1.0, 1.0))


def state_update_direction_cosine(
    candidate_z0: np.ndarray,
    candidate_z_terminal: np.ndarray,
    teacher_z0: np.ndarray,
    teacher_z_terminal: np.ndarray,
) -> float | None:
    """Return cosine alignment between candidate and teacher terminal-state displacements.

    Shape contract:
    - all inputs are batch-first flattened hidden states shaped `(batch, hidden_dim)`
    - the compared directions are:
      - `candidate_z_terminal - candidate_z0`
      - `teacher_z_terminal - teacher_z0`
    """
    candidate_start = _as_batch_first_latent("candidate_z0", candidate_z0)
    candidate_terminal = _as_batch_first_latent("candidate_z_terminal", candidate_z_terminal)
    teacher_start = _as_batch_first_latent("teacher_z0", teacher_z0)
    teacher_terminal = _as_batch_first_latent("teacher_z_terminal", teacher_z_terminal)

    if candidate_start.shape != candidate_terminal.shape:
        raise ValueError("candidate_z0 and candidate_z_terminal must share the same shape.")
    if teacher_start.shape != teacher_terminal.shape:
        raise ValueError("teacher_z0 and teacher_z_terminal must share the same shape.")
    if candidate_start.shape != teacher_start.shape:
        raise ValueError("Candidate and teacher hidden-state tensors must share the same shape.")

    return update_direction_cosine(
        candidate_terminal - candidate_start,
        teacher_terminal - teacher_start,
    )


def summarize_teacher_reference_metrics(
    candidate_z0: np.ndarray,
    candidate_z_terminal: np.ndarray,
    candidate_final_energy: float,
    teacher_z0: np.ndarray,
    teacher_z_terminal: np.ndarray,
    teacher_final_energy: float,
) -> dict[str, float | None]:
    """Summarize candidate-vs-teacher terminal-state metrics for one batch-first evaluation.

    Shape contract:
    - `candidate_z0`: `(batch, hidden_dim)`
    - `candidate_z_terminal`: `(batch, hidden_dim)`
    - `teacher_z0`: `(batch, hidden_dim)`
    - `teacher_z_terminal`: `(batch, hidden_dim)`
    """
    return {
        "terminal_state_l2_gap": hidden_state_l2_distance(candidate_z_terminal, teacher_z_terminal),
        "terminal_state_rms_gap": hidden_state_rms_gap(candidate_z_terminal, teacher_z_terminal),
        "candidate_final_energy": float(candidate_final_energy),
        "teacher_final_energy": float(teacher_final_energy),
        "energy_gap_to_teacher": energy_gap_to_teacher(candidate_final_energy, teacher_final_energy),
        "update_direction_cosine": state_update_direction_cosine(
            candidate_z0,
            candidate_z_terminal,
            teacher_z0,
            teacher_z_terminal,
        ),
    }
