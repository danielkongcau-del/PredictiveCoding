from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.phase2g1_boundary_check import (  # noqa: E402
    PHASE2G1_BENCHMARK_NAMES,
    Phase2G1BoundaryCheckRunResult,
    run_phase2g1_boundary_check,
)


def run(
    benchmark_name: str,
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    plot_energy: bool = False,
) -> Phase2G1BoundaryCheckRunResult:
    """Run one Phase 2g.1 local boundary-check study with default settings."""
    return run_phase2g1_boundary_check(
        benchmark_name,
        output_root=output_root,
        run_id=run_id,
        plot_energy=plot_energy,
    )


def main(argv: list[str] | None = None) -> None:
    """Run Phase 2g.1 for one supported benchmark or both by default."""
    benchmark_names = list(PHASE2G1_BENCHMARK_NAMES) if not argv else argv
    for benchmark_name in benchmark_names:
        result = run(benchmark_name)
        print(f"Phase 2g.1 boundary check completed for {benchmark_name}.")
        print(f"Run directory: {result.run_dir}")
        print(
            f"Previous winner: {result.aggregate_summary['previous_phase2g_test_winner']} -> "
            f"Boundary-check winner: {result.aggregate_summary['boundary_check_test_winner']}"
        )
        print(
            f"Best PC trial: {result.aggregate_summary['boundary_check_best_pc_config_id']} "
            f"val {result.aggregate_summary['metric_name']} "
            f"{float(result.aggregate_summary['boundary_check_best_pc_val_metric']):.6f}, "
            f"test {float(result.aggregate_summary['boundary_check_best_pc_test_metric']):.6f}"
        )
        print(
            f"Best MLP trial: {result.aggregate_summary['boundary_check_best_mlp_config_id']} "
            f"val {result.aggregate_summary['metric_name']} "
            f"{float(result.aggregate_summary['boundary_check_best_mlp_val_metric']):.6f}, "
            f"test {float(result.aggregate_summary['boundary_check_best_mlp_test_metric']):.6f}"
        )
        print(
            f"Conclusion changed: {result.aggregate_summary['headline_conclusion_changed']}; "
            f"Further search warranted: {result.aggregate_summary['further_search_still_warranted']}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
