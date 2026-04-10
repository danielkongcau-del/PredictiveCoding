from __future__ import annotations

from pathlib import Path

from pc.tf1.fmpc_tf1_external_comparison_suite import (
    FMPCTF1ExternalComparisonSuiteConfig,
    FMPCTF1ExternalComparisonSuiteRunResult,
    run_fmpc_tf1_external_comparison_suite,
)


def run(output_root: str | Path = "outputs", run_id: str | None = None) -> FMPCTF1ExternalComparisonSuiteRunResult:
    """Run the narrow TF1 external-comparison validation pass."""

    config = FMPCTF1ExternalComparisonSuiteConfig(output_root=output_root, run_id=run_id)
    return run_fmpc_tf1_external_comparison_suite(config)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
