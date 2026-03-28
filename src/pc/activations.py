from __future__ import annotations

from collections.abc import Callable

import numpy as np

Array = np.ndarray
ActivationFn = Callable[[Array], Array]


def identity(x: Array) -> Array:
    """Apply the identity activation to an array shaped (batch, features)."""
    return x


def identity_prime(x: Array) -> Array:
    """Return the identity derivative for an array shaped (batch, features)."""
    return np.ones_like(x)


def tanh(x: Array) -> Array:
    """Apply tanh elementwise to an array shaped (batch, features)."""
    return np.tanh(x)


def tanh_prime(x: Array) -> Array:
    """Return tanh' elementwise for an array shaped (batch, features)."""
    tanh_x = np.tanh(x)
    return 1.0 - tanh_x * tanh_x


def relu(x: Array) -> Array:
    """Apply ReLU elementwise to an array shaped (batch, features)."""
    return np.maximum(x, 0.0)


def relu_prime(x: Array) -> Array:
    """Return ReLU' with the Phase 0 convention relu'(0) = 0."""
    return (x > 0.0).astype(x.dtype, copy=False)


def get_activation(name: str) -> tuple[ActivationFn, ActivationFn]:
    """Return activation and derivative functions for arrays shaped (batch, features)."""
    mapping: dict[str, tuple[ActivationFn, ActivationFn]] = {
        "identity": (identity, identity_prime),
        "relu": (relu, relu_prime),
        "tanh": (tanh, tanh_prime),
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported activation '{name}'.") from exc
