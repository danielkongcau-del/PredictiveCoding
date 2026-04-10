from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.benchmark_specs import get_benchmark_spec, run_pc_benchmark
from pc.experiment import ExperimentRunResult


def run(
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    plot_energy: bool = True,
) -> ExperimentRunResult:
    """Run the deterministic linear regression benchmark and save structured outputs."""
    return run_pc_benchmark(
        get_benchmark_spec("toy_regression"),
        output_root=output_root,
        run_id=run_id,
        plot_energy=plot_energy,
    )


def main() -> None:
    """Run the benchmark with default Phase 1 output settings."""
    result = run()
    print("Toy regression completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Final pre-update energy: {result.summary['final_pre_update_energy']:.6f}")
    print(f"Final post-update energy: {result.summary['final_post_update_energy']:.6f}")
    print(f"Final train MSE: {result.summary['train_metric']:.6f}")
    print(f"Final val MSE: {result.summary['val_metric']:.6f}")
    print(f"Final test MSE: {result.summary['test_metric']:.6f}")


if __name__ == "__main__":
    main()
