from __future__ import annotations

from pc.incremental_bridge.fmpc_tf2_successor_value_confirmation_suite import (
    FMPCTF2SuccessorValueConfirmationSuiteConfig,
    run_fmpc_tf2_successor_value_confirmation_suite,
)


def run(**overrides):
    """Run the narrow TF2 preterminal successor-value confirmation suite."""

    config = FMPCTF2SuccessorValueConfirmationSuiteConfig(**overrides)
    return run_fmpc_tf2_successor_value_confirmation_suite(config)


if __name__ == "__main__":
    result = run()
    print("Phase Incremental Bridge successor-value confirmation suite completed.")
    print(f"Output directory: {result.run_dir}")
