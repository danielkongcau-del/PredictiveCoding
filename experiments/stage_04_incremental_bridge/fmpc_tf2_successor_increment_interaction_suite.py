from __future__ import annotations

from pc.stage_04_incremental_bridge.fmpc_tf2_successor_increment_interaction_suite import (
    FMPCTF2SuccessorIncrementInteractionSuiteConfig,
    run_fmpc_tf2_successor_increment_interaction_suite,
)


def run(**overrides):
    """Run the narrow TF2 successor-increment direction-magnitude interaction suite."""

    config = FMPCTF2SuccessorIncrementInteractionSuiteConfig(**overrides)
    return run_fmpc_tf2_successor_increment_interaction_suite(config)


if __name__ == "__main__":
    result = run()
    print("FMPC Stage 04 Incremental Bridge successor-increment direction-magnitude interaction suite completed.")
    print(f"Output directory: {result.run_dir}")
