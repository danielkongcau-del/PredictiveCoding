from __future__ import annotations

import numpy as np

from pc.benchmark_specs import BENCHMARK_NAMES, get_benchmark_spec


def test_benchmark_splits_keep_train_and_eval_data_distinct() -> None:
    for benchmark_name in BENCHMARK_NAMES:
        split = get_benchmark_spec(benchmark_name).make_dataset_split()

        assert split.x_train.shape[0] > 0
        assert split.x_val.shape[0] > 0
        assert split.x_test.shape[0] > 0
        assert split.y_train.shape[0] == split.x_train.shape[0]
        assert split.y_val.shape[0] == split.x_val.shape[0]
        assert split.y_test.shape[0] == split.x_test.shape[0]

        if benchmark_name in {"toy_regression", "toy_sine_regression"}:
            assert split.x_val.shape[0] > split.x_train.shape[0]
            assert split.x_test.shape[0] > split.x_train.shape[0]
            assert not np.intersect1d(
                split.x_train.reshape(-1),
                split.x_val.reshape(-1),
            ).size
            assert not np.intersect1d(
                split.x_train.reshape(-1),
                split.x_test.reshape(-1),
            ).size
            assert not np.intersect1d(
                split.x_val.reshape(-1),
                split.x_test.reshape(-1),
            ).size
        else:
            assert split.metadata["train_data_seed"] != split.metadata["val_data_seed"]
            assert split.metadata["train_data_seed"] != split.metadata["test_data_seed"]
            assert split.metadata["val_data_seed"] != split.metadata["test_data_seed"]
            assert not np.array_equal(split.x_train, split.x_val)
            assert not np.array_equal(split.x_train, split.x_test)
            assert not np.array_equal(split.x_val, split.x_test)
