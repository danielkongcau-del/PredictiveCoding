from __future__ import annotations

from pc.incremental_bridge.fmpc_tf2_successor_increment_direction_confirmation_suite import (
    FMPCTF2SuccessorIncrementDirectionConfirmationSuiteConfig,
    run_fmpc_tf2_successor_increment_direction_confirmation_suite,
)


def run(**overrides):
    """Run the narrow TF2 preterminal successor-increment direction confirmation suite."""

    config = FMPCTF2SuccessorIncrementDirectionConfirmationSuiteConfig(**overrides)
    return run_fmpc_tf2_successor_increment_direction_confirmation_suite(config)


if __name__ == "__main__":
    result = run()
    print("Phase Incremental Bridge successor-increment direction confirmation suite completed.")
    print(f"Output directory: {result.run_dir}")
