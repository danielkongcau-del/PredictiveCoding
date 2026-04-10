from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.pc_sensitivity import (
    PHASE2B_BENCHMARK_NAMES,
    PCSensitivityRunResult,
    run_pc_sensitivity_study,
)


def run(
    benchmark_name: str,
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    plot_energy: bool = False,
    plot_summary: bool = False,
) -> PCSensitivityRunResult:
    """Run one Phase 2b sensitivity study with the default output settings."""
    return run_pc_sensitivity_study(
        benchmark_name,
        output_root=output_root,
        run_id=run_id,
        plot_energy=plot_energy,
        plot_summary=plot_summary,
    )


def main(argv: list[str] | None = None) -> None:
    """Run Phase 2b for one named regression benchmark or both by default."""
    benchmark_names = list(PHASE2B_BENCHMARK_NAMES) if not argv else argv
    for benchmark_name in benchmark_names:
        result = run(benchmark_name)
        print(f"PC sensitivity completed for {benchmark_name}.")
        print(f"Run directory: {result.run_dir}")
        print(
            f"Best PC {result.aggregate_summary['primary_metric_name']}: "
            f"{result.aggregate_summary['best_pc_primary_metric_value']:.6f}"
        )
        print(
            f"MLP reference {result.aggregate_summary['primary_metric_name']}: "
            f"{result.aggregate_summary['mlp_reference_primary_metric_value']:.6f}"
        )
        print(
            "Best PC beats MLP reference: "
            f"{result.aggregate_summary['best_pc_beats_mlp_reference']}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
