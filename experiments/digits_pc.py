from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.real_pc import RealPCConfig, RealPCRunResult, run_digits_pc_experiment


def run(
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    plot_curves: bool = False,
) -> RealPCRunResult:
    """Run the Phase 3b digits predictive-coding baseline and save structured outputs."""
    config = RealPCConfig(
        output_root=output_root,
        run_id=run_id,
        plot_curves=plot_curves,
    )
    return run_digits_pc_experiment(config)


def main() -> None:
    """Run the default digits predictive-coding baseline and print a short summary."""
    result = run()
    print("Digits PC baseline completed.")
    print(f"Run directory: {result.run_dir}")
    print("Default reruns overwrite the stable outputs/digits_pc directory unless you choose a different output target.")
    print(f"Best epoch: {result.summary['best_epoch']}")
    print(f"Validation accuracy: {result.summary['val_metric']:.6f}")
    print(f"Test accuracy: {result.summary['test_metric']:.6f}")


if __name__ == "__main__":
    main()
