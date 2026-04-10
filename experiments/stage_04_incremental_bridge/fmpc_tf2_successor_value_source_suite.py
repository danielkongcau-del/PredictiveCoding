from __future__ import annotations

from pc.stage_04_incremental_bridge.fmpc_tf2_successor_value_source_suite import (
    FMPCTF2SuccessorValueSourceSuiteConfig,
    run_fmpc_tf2_successor_value_source_suite,
)


def run(**overrides):
    """Run the narrow TF2 successor-value carry-vs-increment source-localization suite."""

    config = FMPCTF2SuccessorValueSourceSuiteConfig(**overrides)
    return run_fmpc_tf2_successor_value_source_suite(config)


if __name__ == "__main__":
    result = run()
    print("FMPC Stage 04 Incremental Bridge successor-value source-localization suite completed.")
    print(f"Output directory: {result.run_dir}")
