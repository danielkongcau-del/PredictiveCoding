from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.benchmark_specs import BENCHMARK_NAMES
from pc.comparison import ComparisonRunResult, run_benchmark_comparison


def run(
    benchmark_name: str,
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    plot_energy: bool = False,
) -> ComparisonRunResult:
    """Run one benchmark comparison with the default Phase 2 output settings."""
    return run_benchmark_comparison(
        benchmark_name,
        output_root=output_root,
        run_id=run_id,
        plot_energy=plot_energy,
    )


def main() -> None:
    """Run all Phase 2 baseline comparisons with default settings."""
    for benchmark_name in BENCHMARK_NAMES:
        result = run(benchmark_name)
        print(f"Comparison completed for {benchmark_name}.")
        print(f"Run directory: {result.run_dir}")
        print(
            f"PC {result.comparison_summary['primary_metric_name']}: "
            f"{result.comparison_summary['pc_primary_metric_value']:.6f}"
        )
        print(
            f"MLP {result.comparison_summary['primary_metric_name']}: "
            f"{result.comparison_summary['mlp_primary_metric_value']:.6f}"
        )
        print(f"Winner: {result.comparison_summary['winner']}")


if __name__ == "__main__":
    main()
