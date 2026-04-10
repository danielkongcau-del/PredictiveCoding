from __future__ import annotations

from pathlib import Path

from pc.tf1.fmpc_tf1_multiseed_suite import (
    FMPCTF1MultiSeedSuiteConfig,
    FMPCTF1MultiSeedSuiteRunResult,
    run_fmpc_tf1_multiseed_suite,
)


def run(output_root: str | Path = "outputs", run_id: str | None = None) -> FMPCTF1MultiSeedSuiteRunResult:
    """Run the narrow TF1 multiseed confirmation study."""

    config = FMPCTF1MultiSeedSuiteConfig(output_root=output_root, run_id=run_id)
    return run_fmpc_tf1_multiseed_suite(config)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
