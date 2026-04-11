from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.stage_05_ef_core_probe.frozen_bridge_vs_corrected_core_comparison import (
    CorrectedResidualCoreV1VsV2ComparisonConfig,
    FrozenBridgeVsCorrectedCoreComparisonConfig,
    FrozenBridgeVsCorrectedCoreComparisonRunResult,
    FrozenBridgeVsStage05V2ComparisonConfig,
    run_frozen_bridge_vs_corrected_core_comparison,
    run_corrected_residual_core_v1_vs_v2_comparison,
    run_frozen_bridge_vs_stage05_v2_comparison,
)


def run(
    output_root: str | Path = "outputs/stage_05_ef_core_probe",
    run_id: str | None = None,
    comparison_variant: str = "stage04_vs_stage05_v1",
    **overrides: object,
) -> FrozenBridgeVsCorrectedCoreComparisonRunResult:
    """Run a formal Stage 05 comparison entrypoint."""

    if comparison_variant == "stage04_vs_stage05_v1":
        config = FrozenBridgeVsCorrectedCoreComparisonConfig(
            output_root=output_root,
            run_id=run_id,
            **overrides,
        )
        return run_frozen_bridge_vs_corrected_core_comparison(config)
    if comparison_variant == "stage05_v1_vs_v2":
        config = CorrectedResidualCoreV1VsV2ComparisonConfig(
            output_root=output_root,
            run_id=run_id,
            **overrides,
        )
        return run_corrected_residual_core_v1_vs_v2_comparison(config)
    if comparison_variant == "stage04_vs_stage05_v2":
        config = FrozenBridgeVsStage05V2ComparisonConfig(
            output_root=output_root,
            run_id=run_id,
            **overrides,
        )
        return run_frozen_bridge_vs_stage05_v2_comparison(config)
    raise ValueError(f"Unsupported comparison_variant '{comparison_variant}'.")


def main() -> None:
    result = run()
    print("Stage 05 comparison completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Summary: {result.run_dir / 'aggregate_summary.json'}")
    print(f"Report: {result.run_dir / 'comparison_report.md'}")


if __name__ == "__main__":
    main()
