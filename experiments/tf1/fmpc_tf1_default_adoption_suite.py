from __future__ import annotations

from pathlib import Path

from pc.tf1.fmpc_tf1_default_adoption_suite import (
    FMPCTF1DefaultAdoptionSuiteConfig,
    FMPCTF1DefaultAdoptionSuiteRunResult,
    run_fmpc_tf1_default_adoption_suite,
)


def run(output_root: str | Path = "outputs", run_id: str | None = None) -> FMPCTF1DefaultAdoptionSuiteRunResult:
    """Run the narrow TF1 default-adoption validation pass."""

    config = FMPCTF1DefaultAdoptionSuiteConfig(output_root=output_root, run_id=run_id)
    return run_fmpc_tf1_default_adoption_suite(config)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
