from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.stage_03_transport_core_v1.fmpc_tf1_gate_coverage_suite import (
    FMPCTF1GateCoverageSuiteConfig,
    FMPCTF1GateCoverageSuiteRunResult,
    run_fmpc_tf1_gate_coverage_suite,
)


def run(
    output_root: str | Path = "outputs",
    run_id: str | None = None,
) -> FMPCTF1GateCoverageSuiteRunResult:
    """Run the narrow TF1 gate-coverage rescue study."""

    return run_fmpc_tf1_gate_coverage_suite(
        FMPCTF1GateCoverageSuiteConfig(
            output_root=output_root,
            run_id=run_id,
        )
    )


def main() -> None:
    result = run()
    print("FMPC Stage 03 Transport Core v1 gate coverage suite completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Aggregate summary: {result.run_dir / 'aggregate_summary.json'}")


if __name__ == "__main__":
    main()
