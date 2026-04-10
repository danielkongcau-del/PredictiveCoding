from __future__ import annotations

from pc.tf2.fmpc_tf2_preterminal_handoff_confirmation_suite import (
    FMPCTF2PreterminalHandoffConfirmationSuiteConfig,
    run_fmpc_tf2_preterminal_handoff_confirmation_suite,
)


def run(**overrides):
    """Run the narrow TF2 preterminal handoff reformulation confirmation suite."""

    config = FMPCTF2PreterminalHandoffConfirmationSuiteConfig(**overrides)
    return run_fmpc_tf2_preterminal_handoff_confirmation_suite(config)


if __name__ == "__main__":
    result = run()
    print("Phase TF2 preterminal handoff confirmation suite completed.")
    print(f"Output directory: {result.run_dir}")
