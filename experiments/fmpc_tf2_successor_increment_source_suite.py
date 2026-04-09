from __future__ import annotations

from pc.fmpc_tf2_successor_increment_source_suite import (
    FMPCTF2SuccessorIncrementSourceSuiteConfig,
    run_fmpc_tf2_successor_increment_source_suite,
)


def run(**overrides):
    """Run the narrow TF2 preterminal successor-increment source-localization suite."""

    config = FMPCTF2SuccessorIncrementSourceSuiteConfig(**overrides)
    return run_fmpc_tf2_successor_increment_source_suite(config)


if __name__ == "__main__":
    result = run()
    print("Phase TF2 successor-increment source-localization suite completed.")
    print(f"Output directory: {result.run_dir}")
