from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.incremental_bridge.fmpc_tf2_external_comparison_suite import (
    FMPCTF2ExternalComparisonSuiteConfig,
    FMPCTF2ExternalComparisonSuiteRunResult,
    run_fmpc_tf2_external_comparison_suite,
)


def run(
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    **overrides: object,
) -> FMPCTF2ExternalComparisonSuiteRunResult:
    """Run the narrow TF2 external comparison against the canonical slow-PC digits baseline."""

    config = FMPCTF2ExternalComparisonSuiteConfig(
        output_root=output_root,
        run_id=run_id,
        **overrides,
    )
    return run_fmpc_tf2_external_comparison_suite(config)


def main() -> None:
    result = run()
    print("TF2 external comparison suite completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Summary: {result.run_dir / 'aggregate_summary.json'}")


if __name__ == "__main__":
    main()
