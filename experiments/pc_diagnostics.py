from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.pc_diagnostics import (
    PHASE2D_BENCHMARK_NAMES,
    PCDiagnosticsRunResult,
    run_pc_diagnostics_study,
)


def run(
    benchmark_name: str,
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    plot_energy: bool = False,
    plot_summary: bool = False,
) -> PCDiagnosticsRunResult:
    """Run one Phase 2d diagnostic study with the default output settings."""
    return run_pc_diagnostics_study(
        benchmark_name,
        output_root=output_root,
        run_id=run_id,
        plot_energy=plot_energy,
        plot_summary=plot_summary,
    )


def main(argv: list[str] | None = None) -> None:
    """Run Phase 2d for one named regression benchmark or both by default."""
    benchmark_names = list(PHASE2D_BENCHMARK_NAMES) if not argv else argv
    for benchmark_name in benchmark_names:
        result = run(benchmark_name)
        print(f"PC diagnostics completed for {benchmark_name}.")
        print(f"Run directory: {result.run_dir}")
        print(
            "Default PC mean metric: "
            f"{result.diagnostic_summary['default_pc_final_metric_mean']:.6f}"
        )
        print(
            "Tuned PC mean metric: "
            f"{result.diagnostic_summary['tuned_pc_final_metric_mean']:.6f}"
        )
        print(f"MLP mean metric: {result.diagnostic_summary['mlp_final_metric_mean']:.6f}")
        print(
            "Budget2x mean metric: "
            f"{result.diagnostic_summary['budget2x_final_metric_mean']:.6f}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
