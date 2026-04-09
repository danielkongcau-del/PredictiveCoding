from __future__ import annotations

from pc.fmpc_tf2_successor_handoff_source_suite import (
    FMPCTF2SuccessorHandoffSourceSuiteConfig,
    run_fmpc_tf2_successor_handoff_source_suite,
)


def run(**overrides):
    """Run the narrow TF2 successor-handoff source-localization suite."""

    config = FMPCTF2SuccessorHandoffSourceSuiteConfig(**overrides)
    return run_fmpc_tf2_successor_handoff_source_suite(config)


if __name__ == "__main__":
    result = run()
    print("Phase TF2 successor-handoff source-localization suite completed.")
    print(f"Output directory: {result.run_dir}")
