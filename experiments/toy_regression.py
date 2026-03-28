from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.experiment import ExperimentConfig, ExperimentRunResult, run_supervised_experiment
from pc.layers import init_mlp_layers
from pc.metrics import regression_mean_baseline_mse, regression_mse
from pc.models import PCNetwork
from pc.toy_data import make_linear_regression_data


def run(
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    plot_energy: bool = True,
) -> ExperimentRunResult:
    """Run the deterministic linear regression benchmark and save structured outputs."""
    run_seed = 0
    data_seed = 0
    model_init_seed = 0
    x, y = make_linear_regression_data(seed=data_seed)
    model = PCNetwork(
        layers=init_mlp_layers(
            layer_dims=[1, 4, 1],
            hidden_activation="tanh",
            output_activation="identity",
            weight_scale=0.15,
            seed=model_init_seed,
        ),
        eta_x=0.2,
        eta_w=0.05,
        eta_b=0.05,
        train_steps=25,
        eval_steps=25,
        state_init="forward",
    )
    config = ExperimentConfig(
        experiment_name="toy_regression",
        seed=run_seed,
        data_seed=data_seed,
        model_init_seed=model_init_seed,
        epochs=60,
        output_root=output_root,
        run_id=run_id,
        plot_energy=plot_energy,
        task={"name": "regression"},
        data={
            "dataset_name": "linear_regression",
            "num_points": int(x.shape[0]),
            "input_dim": 1,
            "target_dim": 1,
            "data_seed": data_seed,
        },
        model={
            "layer_dims": [1, 4, 1],
            "hidden_activation": "tanh",
            "output_activation": "identity",
            "state_init": "forward",
            "model_init_seed": model_init_seed,
        },
        training={
            "epochs": 60,
            "eta_x": 0.2,
            "eta_w": 0.05,
            "eta_b": 0.05,
            "train_steps": 25,
            "eval_steps": 25,
            "run_seed": run_seed,
        },
    )
    return run_supervised_experiment(
        config=config,
        model=model,
        x=x,
        y=y,
        task_name="regression",
        primary_metric_name="mse",
        primary_metric_higher_is_better=False,
        primary_metric_fn=regression_mse,
        baseline_metric_name="baseline_mse",
        baseline_metric_fn=regression_mean_baseline_mse,
    )


def main() -> None:
    """Run the benchmark with default Phase 1 output settings."""
    result = run()
    print("Toy regression completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Final pre-update energy: {result.summary['final_pre_update_energy']:.6f}")
    print(f"Final post-update energy: {result.summary['final_post_update_energy']:.6f}")
    print(f"Final MSE: {result.summary['primary_metric_value']:.6f}")


if __name__ == "__main__":
    main()
