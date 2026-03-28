from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def make_rng(seed: int | None) -> np.random.Generator:
    """Create a reproducible NumPy generator from an optional integer seed."""
    return np.random.default_rng(seed)


def set_seed(seed: int | None) -> None:
    """Seed NumPy's legacy RNG for deterministic Phase 0 experiments and tests."""
    if seed is not None:
        np.random.seed(seed)


def assert_shape(array: np.ndarray, expected_shape: tuple[int, ...], name: str) -> None:
    """Raise if an array does not match the expected explicit shape."""
    if array.shape != expected_shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {expected_shape}.")


def ensure_finite_array(array: np.ndarray, name: str) -> None:
    """Raise if an array contains NaN or Inf values."""
    if not np.all(np.isfinite(array)):
        raise FloatingPointError(f"{name} contains non-finite values.")


def ensure_finite_collection(
    arrays: Sequence[np.ndarray | None],
    collection_name: str,
) -> None:
    """Raise if any non-None array in a collection contains NaN or Inf values."""
    for index, array in enumerate(arrays):
        if array is not None:
            ensure_finite_array(array, f"{collection_name}[{index}]")
