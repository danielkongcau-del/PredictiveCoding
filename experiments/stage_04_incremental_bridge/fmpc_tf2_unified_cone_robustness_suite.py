from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.stage_04_incremental_bridge.fmpc_tf2_unified_cone_robustness_suite import (
    FMPCTF2UnifiedConeRobustnessSuiteConfig,
    FMPCTF2UnifiedConeRobustnessSuiteRunResult,
    run_fmpc_tf2_unified_cone_robustness_suite,
)


def run(
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    hard_shape_source_root: str | Path = "outputs/stage_04_incremental_bridge/fmpc_tf2_unified_cone_shape_suite",
    smooth_source_root: str | Path = "outputs/stage_04_incremental_bridge/fmpc_tf2_smooth_unified_cone_suite",
) -> FMPCTF2UnifiedConeRobustnessSuiteRunResult:
    """Run the adopted-package unified-cone robustness-tradeoff diagnostic."""

    config = FMPCTF2UnifiedConeRobustnessSuiteConfig(
        output_root=output_root,
        run_id=run_id,
        hard_shape_source_root=hard_shape_source_root,
        smooth_source_root=smooth_source_root,
    )
    return run_fmpc_tf2_unified_cone_robustness_suite(config)


def main() -> None:
    result = run()
    summary = result.summary
    print("TF2 unified-cone robustness suite completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Decision: {summary['decision']}")
    print(f"Diagnosis: {summary['diagnosis']}")


if __name__ == "__main__":
    main()
