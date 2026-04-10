from __future__ import annotations

from pc.tf2.fmpc_tf2_successor_value_followup_suite import (
    FMPCTF2SuccessorValueFollowupSuiteConfig,
    run_fmpc_tf2_successor_value_followup_suite,
)


def run(**overrides):
    """Run the tiny TF2 successor-value local follow-up suite."""

    config = FMPCTF2SuccessorValueFollowupSuiteConfig(**overrides)
    return run_fmpc_tf2_successor_value_followup_suite(config)


if __name__ == "__main__":
    result = run()
    print("Phase TF2 successor-value follow-up suite completed.")
    print(f"Output directory: {result.run_dir}")
