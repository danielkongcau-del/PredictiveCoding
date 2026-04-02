from __future__ import annotations

import numpy as np

from pc.minibatch import iter_minibatches


def _make_batch_arrays(num_samples: int = 10) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(num_samples * 3, dtype=np.float64).reshape(num_samples, 3)
    y = np.arange(num_samples * 2, dtype=np.float64).reshape(num_samples, 2)
    return x, y


def test_iter_minibatches_preserves_all_samples_without_duplicates() -> None:
    x, y = _make_batch_arrays(num_samples=10)

    batches = list(
        iter_minibatches(
            x,
            y,
            batch_size=4,
            shuffle=False,
            return_indices=True,
        )
    )

    batch_sizes = [x_batch.shape[0] for x_batch, _, _ in batches]
    assert batch_sizes == [4, 4, 2]

    concatenated_indices = np.concatenate([indices for _, _, indices in batches])
    assert np.array_equal(concatenated_indices, np.arange(10, dtype=np.int64))
    assert np.unique(concatenated_indices).shape[0] == 10


def test_iter_minibatches_same_seed_gives_same_shuffled_order() -> None:
    x, y = _make_batch_arrays(num_samples=12)

    first = list(
        iter_minibatches(x, y, batch_size=5, shuffle=True, seed=7, return_indices=True)
    )
    second = list(
        iter_minibatches(x, y, batch_size=5, shuffle=True, seed=7, return_indices=True)
    )

    first_indices = np.concatenate([indices for _, _, indices in first])
    second_indices = np.concatenate([indices for _, _, indices in second])
    assert np.array_equal(first_indices, second_indices)


def test_iter_minibatches_different_seed_changes_shuffled_order() -> None:
    x, y = _make_batch_arrays(num_samples=12)

    first = list(
        iter_minibatches(x, y, batch_size=5, shuffle=True, seed=7, return_indices=True)
    )
    second = list(
        iter_minibatches(x, y, batch_size=5, shuffle=True, seed=11, return_indices=True)
    )

    first_indices = np.concatenate([indices for _, _, indices in first])
    second_indices = np.concatenate([indices for _, _, indices in second])
    assert not np.array_equal(first_indices, second_indices)


def test_iter_minibatches_full_batch_fallback_is_correct() -> None:
    x, y = _make_batch_arrays(num_samples=9)

    none_batches = list(iter_minibatches(x, y, batch_size=None))
    large_batches = list(iter_minibatches(x, y, batch_size=20))

    assert len(none_batches) == 1
    assert len(large_batches) == 1
    assert np.array_equal(none_batches[0][0], x)
    assert np.array_equal(none_batches[0][1], y)
    assert np.array_equal(large_batches[0][0], x)
    assert np.array_equal(large_batches[0][1], y)


def test_iter_minibatches_return_indices_match_batch_contents() -> None:
    x, y = _make_batch_arrays(num_samples=11)

    for x_batch, y_batch, indices in iter_minibatches(
        x,
        y,
        batch_size=4,
        shuffle=True,
        seed=5,
        return_indices=True,
    ):
        assert np.array_equal(x_batch, x[indices])
        assert np.array_equal(y_batch, y[indices])


def test_iter_minibatches_drop_last_omits_incomplete_tail() -> None:
    x, y = _make_batch_arrays(num_samples=10)

    batches = list(
        iter_minibatches(
            x,
            y,
            batch_size=4,
            shuffle=False,
            drop_last=True,
            return_indices=True,
        )
    )

    batch_sizes = [x_batch.shape[0] for x_batch, _, _ in batches]
    assert batch_sizes == [4, 4]
    concatenated_indices = np.concatenate([indices for _, _, indices in batches])
    assert np.array_equal(concatenated_indices, np.arange(8, dtype=np.int64))
