from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.pc_budget_tradeoff import (
    PHASE2_MLP_SOURCE_PHASE2G1,
    PHASE2_TUNED_SOURCE_PHASE2G1,
    PHASE2E_BENCHMARK_NAMES,
    PCBudgetTradeoffRunResult,
    run_pc_budget_tradeoff_study,
)


def run(
    benchmark_name: str,
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    plot_energy: bool = False,
    plot_summary: bool = False,
    tuned_source: str = PHASE2_TUNED_SOURCE_PHASE2G1,
    mlp_source: str | None = PHASE2_MLP_SOURCE_PHASE2G1,
) -> PCBudgetTradeoffRunResult:
    """Run one Phase 2e budget tradeoff study with the default output settings."""
    return run_pc_budget_tradeoff_study(
        benchmark_name,
        output_root=output_root,
        run_id=run_id,
        plot_energy=plot_energy,
        plot_summary=plot_summary,
        tuned_source=tuned_source,
        mlp_source=mlp_source,
    )


def main(argv: list[str] | None = None) -> None:
    """Run Phase 2e for one named regression benchmark or both by default."""
    benchmark_names = list(PHASE2E_BENCHMARK_NAMES) if not argv else argv
    for benchmark_name in benchmark_names:
        result = run(benchmark_name)
        print(f"PC budget tradeoff completed for {benchmark_name}.")
        print(f"Run directory: {result.run_dir}")
        print(
            "Tuned source: "
            f"{result.aggregate_summary['tuned_pc_source']} "
            f"({result.aggregate_summary['tuned_pc_preset_name']})"
        )
        print(
            "MLP source: "
            f"{result.aggregate_summary['mlp_source']} "
            f"({result.aggregate_summary['mlp_preset_name']})"
        )
        print(
            "1x mean metric: "
            f"{result.aggregate_summary['tuned_pc_1x_primary_metric_mean']:.6f}"
        )
        print(
            "2x mean metric: "
            f"{result.aggregate_summary['tuned_pc_2x_primary_metric_mean']:.6f}"
        )
        print(
            "4x mean metric: "
            f"{result.aggregate_summary['tuned_pc_4x_primary_metric_mean']:.6f}"
        )
        print(f"MLP mean metric: {result.aggregate_summary['mlp_primary_metric_mean']:.6f}")


if __name__ == "__main__":
    main(sys.argv[1:])
