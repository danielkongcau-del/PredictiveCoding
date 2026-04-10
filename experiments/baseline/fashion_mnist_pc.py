from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.real_pc import RealPCConfig, RealPCRunResult, run_real_pc_experiment


def run(
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    plot_curves: bool = False,
) -> RealPCRunResult:
    """Run a standalone Fashion-MNIST predictive-coding baseline."""
    config = RealPCConfig(
        experiment_name="fashion_mnist_pc",
        dataset_name="fashion_mnist",
        output_root=output_root,
        run_id=run_id,
        plot_curves=plot_curves,
        layer_dims=(784, 64, 10),
        epochs=30,
        batch_size=128,
    )
    return run_real_pc_experiment(config)


def main() -> None:
    """Run the default Fashion-MNIST predictive-coding baseline and print a short summary."""
    result = run()
    print("Fashion-MNIST PC baseline completed.")
    print(f"Run directory: {result.run_dir}")
    print(
        "Default reruns overwrite the stable outputs/fashion_mnist_pc directory unless you choose a different output target."
    )
    print(f"Best epoch: {result.summary['best_epoch']}")
    print(f"Validation accuracy: {result.summary['val_metric']:.6f}")
    print(f"Test accuracy: {result.summary['test_metric']:.6f}")


if __name__ == "__main__":
    main()
