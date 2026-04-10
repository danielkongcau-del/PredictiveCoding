from __future__ import annotations

from pc.fmpc_tf2_successor_increment_formulation_suite import (
    FMPCTF2SuccessorIncrementFormulationSuiteConfig,
    run_fmpc_tf2_successor_increment_formulation_suite,
)


def run(**overrides):
    """Run the narrow TF2 successor-increment formulation source-localization suite."""

    config = FMPCTF2SuccessorIncrementFormulationSuiteConfig(**overrides)
    return run_fmpc_tf2_successor_increment_formulation_suite(config)


if __name__ == "__main__":
    result = run()
    print("Phase TF2 successor-increment formulation source-localization suite completed.")
    print(f"Output directory: {result.run_dir}")
