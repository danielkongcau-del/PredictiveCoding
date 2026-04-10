from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.real_mlp import RealMLPConfig, RealMLPRunResult, run_digits_mlp_experiment


def run(
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    plot_curves: bool = False,
) -> RealMLPRunResult:
    """Run the Phase 3a digits MLP baseline and save structured outputs."""
    config = RealMLPConfig(
        output_root=output_root,
        run_id=run_id,
        plot_curves=plot_curves,
    )
    return run_digits_mlp_experiment(config)


def main() -> None:
    """Run the default digits MLP baseline and print a short completion summary."""
    result = run()
    print("Digits MLP baseline completed.")
    print(f"Run directory: {result.run_dir}")
    print("Default reruns overwrite the stable outputs/digits_mlp directory unless you choose a different output target.")
    print(f"Best epoch: {result.summary['best_epoch']}")
    print(f"Validation accuracy: {result.summary['val_metric']:.6f}")
    print(f"Test accuracy: {result.summary['test_metric']:.6f}")


if __name__ == "__main__":
    main()
