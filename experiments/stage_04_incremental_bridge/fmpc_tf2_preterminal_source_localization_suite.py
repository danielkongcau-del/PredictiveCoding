from __future__ import annotations

from pc.stage_04_incremental_bridge.fmpc_tf2_preterminal_source_localization_suite import (
    FMPCTF2PreterminalSourceLocalizationSuiteConfig,
    run_fmpc_tf2_preterminal_source_localization_suite,
)


def run(**overrides):
    """Run the narrow TF2 preterminal-update source-localization suite."""

    config = FMPCTF2PreterminalSourceLocalizationSuiteConfig(**overrides)
    return run_fmpc_tf2_preterminal_source_localization_suite(config)


if __name__ == "__main__":
    result = run()
    print("FMPC Stage 04 Incremental Bridge preterminal-update source-localization suite completed.")
    print(f"Output directory: {result.run_dir}")
