from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.stage_05_ef_core_probe.frozen_bridge_vs_corrected_core_comparison import (
    FrozenBridgeVsCorrectedCoreComparisonConfig,
    FrozenBridgeVsCorrectedCoreComparisonRunResult,
    run_frozen_bridge_vs_corrected_core_comparison,
)


def run(
    output_root: str | Path = "outputs/stage_05_ef_core_probe",
    run_id: str | None = None,
    **overrides: object,
) -> FrozenBridgeVsCorrectedCoreComparisonRunResult:
    """Run the formal frozen-bridge vs corrected-core comparison."""

    config = FrozenBridgeVsCorrectedCoreComparisonConfig(
        output_root=output_root,
        run_id=run_id,
        **overrides,
    )
    return run_frozen_bridge_vs_corrected_core_comparison(config)


def main() -> None:
    result = run()
    print("Frozen bridge vs corrected residual core comparison completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Summary: {result.run_dir / 'aggregate_summary.json'}")
    print(f"Report: {result.run_dir / 'comparison_report.md'}")


if __name__ == "__main__":
    main()
