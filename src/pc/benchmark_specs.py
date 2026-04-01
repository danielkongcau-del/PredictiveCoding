from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

from .experiment import ExperimentConfig, ExperimentRunResult, OutputLayout, run_supervised_experiment
from .layers import init_mlp_layers
from .metrics import (
    BaselineMetricFn,
    MetricFn,
    classification_accuracy,
    majority_class_baseline_accuracy,
    metric_higher_is_better,
    regression_mean_baseline_mse,
    regression_mse,
)
from .mlp_baseline import MLPNetwork, init_mlp_baseline_layers
from .models import PCNetwork
from .toy_data import (
    SupervisedDataSplit,
    make_blobs_classification_split,
    make_blobs_classification_data,
    make_linear_regression_split,
    make_linear_regression_data,
    make_sine_regression_split,
    make_sine_regression_data,
)

TaskName = Literal["regression", "classification"]


@dataclass(frozen=True)
class PCTrainingSpec:
    """Fixed predictive-coding training hyperparameters for one benchmark."""

    eta_x: float
    eta_w: float
    eta_b: float
    train_steps: int
    eval_steps: int
    state_init: str = "forward"


@dataclass(frozen=True)
class MLPTrainingSpec:
    """Fixed standard-MLP training hyperparameters for one benchmark."""

    eta_w: float
    eta_b: float


@dataclass(frozen=True)
class ToyBenchmarkSpec:
    """Single source of truth for one toy benchmark and its baseline settings."""

    benchmark_name: str
    task_name: TaskName
    dataset_name: str
    data_split_builder: Callable[..., SupervisedDataSplit]
    data_kwargs: dict[str, Any]
    primary_metric_name: str
    primary_metric_fn: MetricFn
    baseline_metric_name: str
    baseline_metric_fn: BaselineMetricFn
    run_seed: int
    data_seed: int
    model_init_seed: int
    layer_dims: tuple[int, ...]
    hidden_activation: str
    weight_scale: float
    epochs: int
    pc_training: PCTrainingSpec
    mlp_training: MLPTrainingSpec
    output_activation: str = "identity"

    @property
    def primary_metric_higher_is_better(self) -> bool:
        return metric_higher_is_better(self.primary_metric_name)

    def make_dataset_split(self) -> SupervisedDataSplit:
        """Return deterministic train/val/test arrays shaped (batch, features)."""
        return self.data_split_builder(seed=self.data_seed, **self.data_kwargs)

    def make_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the deterministic training arrays shaped (batch, features)."""
        split = self.make_dataset_split()
        return split.x_train, split.y_train

    def task_config(self) -> dict[str, Any]:
        task = {"name": self.task_name}
        if self.task_name == "classification":
            task["num_classes"] = int(self.layer_dims[-1])
        return task

    def data_config(
        self,
        split_or_x: SupervisedDataSplit | np.ndarray,
        y: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if isinstance(split_or_x, SupervisedDataSplit):
            split = split_or_x
        else:
            if y is None:
                raise ValueError("y must be provided when data_config is called with x arrays.")
            split = SupervisedDataSplit(
                x_train=np.asarray(split_or_x, dtype=np.float64),
                y_train=np.asarray(y, dtype=np.float64),
                x_val=np.asarray(split_or_x, dtype=np.float64),
                y_val=np.asarray(y, dtype=np.float64),
                x_test=np.asarray(split_or_x, dtype=np.float64),
                y_test=np.asarray(y, dtype=np.float64),
                metadata={"evaluation_protocol": "train_equals_val_equals_test"},
            )
        data = {
            "dataset_name": self.dataset_name,
            "input_dim": int(split.x_train.shape[1]),
            "target_dim": int(split.y_train.shape[1]),
            "data_seed": self.data_seed,
            "train_size": int(split.x_train.shape[0]),
            "val_size": int(split.x_val.shape[0]),
            "test_size": int(split.x_test.shape[0]),
        }
        data.update(self.data_kwargs)
        data.update(split.metadata)
        return data

    def pc_model_config(self) -> dict[str, Any]:
        return {
            "layer_dims": list(self.layer_dims),
            "hidden_activation": self.hidden_activation,
            "output_activation": self.output_activation,
            "state_init": self.pc_training.state_init,
            "weight_scale": self.weight_scale,
            "model_init_seed": self.model_init_seed,
        }

    def mlp_model_config(self) -> dict[str, Any]:
        return {
            "layer_dims": list(self.layer_dims),
            "hidden_activation": self.hidden_activation,
            "output_activation": self.output_activation,
            "weight_scale": self.weight_scale,
            "model_init_seed": self.model_init_seed,
        }

    def build_pc_model(self) -> PCNetwork:
        """Return the current predictive-coding model for this benchmark."""
        return PCNetwork(
            layers=init_mlp_layers(
                layer_dims=list(self.layer_dims),
                hidden_activation=self.hidden_activation,
                output_activation=self.output_activation,
                weight_scale=self.weight_scale,
                seed=self.model_init_seed,
            ),
            eta_x=self.pc_training.eta_x,
            eta_w=self.pc_training.eta_w,
            eta_b=self.pc_training.eta_b,
            train_steps=self.pc_training.train_steps,
            eval_steps=self.pc_training.eval_steps,
            state_init=self.pc_training.state_init,
        )

    def build_mlp_model(self) -> MLPNetwork:
        """Return the standard backpropagation MLP baseline for this benchmark."""
        return MLPNetwork(
            layers=init_mlp_baseline_layers(
                layer_dims=list(self.layer_dims),
                hidden_activation=self.hidden_activation,
                output_activation=self.output_activation,
                weight_scale=self.weight_scale,
                seed=self.model_init_seed,
            ),
            eta_w=self.mlp_training.eta_w,
            eta_b=self.mlp_training.eta_b,
        )

    def build_pc_config(
        self,
        *,
        output_root: str | Path,
        run_id: str | None,
        plot_energy: bool,
        output_layout: OutputLayout = "single_dir",
        experiment_name: str | None = None,
        split: SupervisedDataSplit | None = None,
    ) -> ExperimentConfig:
        """Return the existing ExperimentConfig for the predictive-coding benchmark run."""
        if split is None:
            split = self.make_dataset_split()
        return ExperimentConfig(
            experiment_name=self.benchmark_name if experiment_name is None else experiment_name,
            seed=self.run_seed,
            data_seed=self.data_seed,
            model_init_seed=self.model_init_seed,
            epochs=self.epochs,
            output_root=output_root,
            run_id=run_id,
            output_layout=output_layout,
            plot_energy=plot_energy,
            task=self.task_config(),
            data=self.data_config(split),
            model=self.pc_model_config(),
            training={
                "epochs": self.epochs,
                "eta_x": self.pc_training.eta_x,
                "eta_w": self.pc_training.eta_w,
                "eta_b": self.pc_training.eta_b,
                "train_steps": self.pc_training.train_steps,
                "eval_steps": self.pc_training.eval_steps,
                "run_seed": self.run_seed,
            },
        )


def run_pc_benchmark(
    spec: ToyBenchmarkSpec,
    *,
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    plot_energy: bool = True,
    output_layout: OutputLayout = "single_dir",
    experiment_name: str | None = None,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    x_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    split: SupervisedDataSplit | None = None,
) -> ExperimentRunResult:
    """Run the existing predictive-coding benchmark path without changing its behavior."""
    if split is None and x is None and y is None:
        split = spec.make_dataset_split()
    if split is not None:
        x = split.x_train
        y = split.y_train
        x_val = split.x_val
        y_val = split.y_val
        x_test = split.x_test
        y_test = split.y_test
    if x is None or y is None:
        raise ValueError("Training data must be provided when split is not used.")
    if x_val is None or y_val is None:
        x_val = x
        y_val = y
    if x_test is None or y_test is None:
        x_test = x_val
        y_test = y_val
    if split is None:
        split = SupervisedDataSplit(
            x_train=np.asarray(x, dtype=np.float64),
            y_train=np.asarray(y, dtype=np.float64),
            x_val=np.asarray(x_val, dtype=np.float64),
            y_val=np.asarray(y_val, dtype=np.float64),
            x_test=np.asarray(x_test, dtype=np.float64),
            y_test=np.asarray(y_test, dtype=np.float64),
            metadata={
                "evaluation_protocol": (
                    "train_equals_val_equals_test"
                    if x_val is x and y_val is y and x_test is x and y_test is y
                    else "explicit_val_test_arrays"
                ),
            },
        )
    config = spec.build_pc_config(
        output_root=output_root,
        run_id=run_id,
        plot_energy=plot_energy,
        output_layout=output_layout,
        experiment_name=experiment_name,
        split=split,
    )
    return run_supervised_experiment(
        config=config,
        model=spec.build_pc_model(),
        x=x,
        y=y,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        task_name=spec.task_name,
        primary_metric_name=spec.primary_metric_name,
        primary_metric_higher_is_better=spec.primary_metric_higher_is_better,
        primary_metric_fn=spec.primary_metric_fn,
        baseline_metric_name=spec.baseline_metric_name,
        baseline_metric_fn=spec.baseline_metric_fn,
    )


BENCHMARK_NAMES = (
    "toy_regression",
    "toy_sine_regression",
    "toy_blobs_classification",
)


def get_benchmark_spec(name: str) -> ToyBenchmarkSpec:
    """Return the fixed benchmark spec for one existing toy experiment."""
    specs: dict[str, ToyBenchmarkSpec] = {
        "toy_regression": ToyBenchmarkSpec(
            benchmark_name="toy_regression",
            task_name="regression",
            dataset_name="linear_regression",
            data_split_builder=make_linear_regression_split,
            data_kwargs={"num_points": 16, "val_num_points": 129, "test_num_points": 129},
            primary_metric_name="mse",
            primary_metric_fn=regression_mse,
            baseline_metric_name="baseline_mse",
            baseline_metric_fn=regression_mean_baseline_mse,
            run_seed=0,
            data_seed=0,
            model_init_seed=0,
            layer_dims=(1, 4, 1),
            hidden_activation="tanh",
            weight_scale=0.15,
            epochs=60,
            pc_training=PCTrainingSpec(
                eta_x=0.2,
                eta_w=0.05,
                eta_b=0.05,
                train_steps=25,
                eval_steps=25,
            ),
            mlp_training=MLPTrainingSpec(
                eta_w=0.1,
                eta_b=0.1,
            ),
        ),
        "toy_sine_regression": ToyBenchmarkSpec(
            benchmark_name="toy_sine_regression",
            task_name="regression",
            dataset_name="sine_regression",
            data_split_builder=make_sine_regression_split,
            data_kwargs={"num_points": 32, "val_num_points": 257, "test_num_points": 257},
            primary_metric_name="mse",
            primary_metric_fn=regression_mse,
            baseline_metric_name="baseline_mse",
            baseline_metric_fn=regression_mean_baseline_mse,
            run_seed=3,
            data_seed=3,
            model_init_seed=3,
            layer_dims=(1, 8, 1),
            hidden_activation="tanh",
            weight_scale=0.12,
            epochs=80,
            pc_training=PCTrainingSpec(
                eta_x=0.15,
                eta_w=0.03,
                eta_b=0.03,
                train_steps=30,
                eval_steps=30,
            ),
            mlp_training=MLPTrainingSpec(
                eta_w=0.05,
                eta_b=0.05,
            ),
        ),
        "toy_blobs_classification": ToyBenchmarkSpec(
            benchmark_name="toy_blobs_classification",
            task_name="classification",
            dataset_name="gaussian_blobs",
            data_split_builder=make_blobs_classification_split,
            data_kwargs={
                "points_per_class": 24,
                "val_points_per_class": 48,
                "test_points_per_class": 48,
            },
            primary_metric_name="accuracy",
            primary_metric_fn=classification_accuracy,
            baseline_metric_name="baseline_accuracy",
            baseline_metric_fn=majority_class_baseline_accuracy,
            run_seed=11,
            data_seed=11,
            model_init_seed=11,
            layer_dims=(2, 10, 3),
            hidden_activation="tanh",
            weight_scale=0.08,
            epochs=70,
            pc_training=PCTrainingSpec(
                eta_x=0.15,
                eta_w=0.05,
                eta_b=0.05,
                train_steps=30,
                eval_steps=30,
            ),
            mlp_training=MLPTrainingSpec(
                eta_w=0.1,
                eta_b=0.1,
            ),
        ),
    }
    try:
        return specs[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported benchmark '{name}'.") from exc
