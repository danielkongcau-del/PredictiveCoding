from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from .utils import make_rng


def iter_minibatches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int | None,
    *,
    shuffle: bool = False,
    seed: int | None = None,
    drop_last: bool = False,
    return_indices: bool = False,
) -> Iterator[tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield deterministic mini-batches for batch-first arrays.

    Shapes:
    - `x`: `(batch, features)`
    - `y`: `(batch, targets)`
    - yielded `x_batch`: `(batch_i, features)`
    - yielded `y_batch`: `(batch_i, targets)`
    - yielded `indices`: `(batch_i,)` when `return_indices=True`
    """
    x_array = np.asarray(x)
    y_array = np.asarray(y)

    if x_array.ndim != 2 or y_array.ndim != 2:
        raise ValueError("x and y must both be rank-2 batch-first arrays.")
    if x_array.shape[0] != y_array.shape[0]:
        raise ValueError("x and y must have the same batch dimension.")

    num_samples = int(x_array.shape[0])
    if num_samples <= 0:
        raise ValueError("x and y must contain at least one sample.")
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive when provided.")

    if batch_size is None or batch_size >= num_samples:
        indices = np.arange(num_samples, dtype=np.int64)
        if return_indices:
            yield x_array, y_array, indices
        else:
            yield x_array, y_array
        return

    if shuffle:
        indices = make_rng(seed).permutation(num_samples).astype(np.int64, copy=False)
    else:
        indices = np.arange(num_samples, dtype=np.int64)

    for start in range(0, num_samples, batch_size):
        stop = min(start + batch_size, num_samples)
        if drop_last and (stop - start) < batch_size:
            break
        batch_indices = indices[start:stop]
        x_batch = x_array[batch_indices]
        y_batch = y_array[batch_indices]
        if return_indices:
            yield x_batch, y_batch, batch_indices
        else:
            yield x_batch, y_batch
