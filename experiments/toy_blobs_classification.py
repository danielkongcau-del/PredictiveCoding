from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.experiment import ExperimentConfig, ExperimentRunResult, run_supervised_experiment
from pc.layers import init_mlp_layers
from pc.metrics import classification_accuracy, majority_class_baseline_accuracy
from pc.models import PCNetwork
from pc.toy_data import make_blobs_classification_data


def run(
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    plot_energy: bool = True,
) -> ExperimentRunResult:
    """Run the deterministic Gaussian-blobs classification benchmark and save structured outputs."""
    run_seed = 11
    data_seed = 11
    model_init_seed = 11
    x, y = make_blobs_classification_data(seed=data_seed, points_per_class=24)
    model = PCNetwork(
        layers=init_mlp_layers(
            layer_dims=[2, 10, 3],
            hidden_activation="tanh",
            output_activation="identity",
            weight_scale=0.08,
            seed=model_init_seed,
        ),
        eta_x=0.15,
        eta_w=0.05,
        eta_b=0.05,
        train_steps=30,
        eval_steps=30,
        state_init="forward",
    )
    config = ExperimentConfig(
        experiment_name="toy_blobs_classification",
        seed=run_seed,
        data_seed=data_seed,
        model_init_seed=model_init_seed,
        epochs=70,
        output_root=output_root,
        run_id=run_id,
        plot_energy=plot_energy,
        task={"name": "classification", "num_classes": 3},
        data={
            "dataset_name": "gaussian_blobs",
            "points_per_class": 24,
            "input_dim": 2,
            "target_dim": 3,
            "data_seed": data_seed,
        },
        model={
            "layer_dims": [2, 10, 3],
            "hidden_activation": "tanh",
            "output_activation": "identity",
            "state_init": "forward",
            "model_init_seed": model_init_seed,
        },
        training={
            "epochs": 70,
            "eta_x": 0.15,
            "eta_w": 0.05,
            "eta_b": 0.05,
            "train_steps": 30,
            "eval_steps": 30,
            "run_seed": run_seed,
        },
    )
    return run_supervised_experiment(
        config=config,
        model=model,
        x=x,
        y=y,
        task_name="classification",
        primary_metric_name="accuracy",
        primary_metric_higher_is_better=True,
        primary_metric_fn=classification_accuracy,
        baseline_metric_name="baseline_accuracy",
        baseline_metric_fn=majority_class_baseline_accuracy,
    )


def main() -> None:
    """Run the benchmark with default Phase 1 output settings."""
    result = run()
    print("Toy blobs classification completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Final pre-update energy: {result.summary['final_pre_update_energy']:.6f}")
    print(f"Final post-update energy: {result.summary['final_post_update_energy']:.6f}")
    print(f"Final accuracy: {result.summary['primary_metric_value']:.6f}")


if __name__ == "__main__":
    main()
