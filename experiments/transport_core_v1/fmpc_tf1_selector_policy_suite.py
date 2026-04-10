from __future__ import annotations

from pathlib import Path

from pc.transport_core_v1.fmpc_tf1_selector_policy_suite import (
    FMPCTF1SelectorPolicySuiteConfig,
    FMPCTF1SelectorPolicySuiteRunResult,
    run_fmpc_tf1_selector_policy_suite,
)


def run(output_root: str | Path = "outputs", run_id: str | None = None) -> FMPCTF1SelectorPolicySuiteRunResult:
    """Run the narrow TF1 selector-cascade study."""

    config = FMPCTF1SelectorPolicySuiteConfig(output_root=output_root, run_id=run_id)
    return run_fmpc_tf1_selector_policy_suite(config)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
