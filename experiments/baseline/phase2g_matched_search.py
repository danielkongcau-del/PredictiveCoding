from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.phase2g_matched_search import (  # noqa: E402
    PHASE2G_BENCHMARK_NAMES,
    Phase2GMatchedSearchRunResult,
    run_phase2g_matched_search,
)


def run(
    benchmark_name: str,
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    plot_energy: bool = False,
) -> Phase2GMatchedSearchRunResult:
    """Run one Phase 2g matched PC+MLP search with default output settings."""
    return run_phase2g_matched_search(
        benchmark_name,
        output_root=output_root,
        run_id=run_id,
        plot_energy=plot_energy,
    )


def main(argv: list[str] | None = None) -> None:
    """Run Phase 2g for one named regression benchmark or both by default."""
    benchmark_names = list(PHASE2G_BENCHMARK_NAMES) if not argv else argv
    for benchmark_name in benchmark_names:
        result = run(benchmark_name)
        print(f"Phase 2g matched search completed for {benchmark_name}.")
        print(f"Run directory: {result.run_dir}")
        print(
            f"Best PC: {result.aggregate_summary['best_pc_config_id']} "
            f"val {result.aggregate_summary['metric_name']} "
            f"{float(result.aggregate_summary['best_pc_val_metric']):.6f}, "
            f"test {float(result.aggregate_summary['best_pc_test_metric']):.6f}"
        )
        print(
            f"Best MLP: {result.aggregate_summary['best_mlp_config_id']} "
            f"val {result.aggregate_summary['metric_name']} "
            f"{float(result.aggregate_summary['best_mlp_val_metric']):.6f}, "
            f"test {float(result.aggregate_summary['best_mlp_test_metric']):.6f}"
        )
        print(
            f"Test winner: {result.aggregate_summary['test_winner']} "
            f"({result.aggregate_summary['test_winner_reason']})"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
