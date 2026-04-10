from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.stage_04_incremental_bridge.fmpc_tf2_bootstrap_fidelity_suite import (
    FMPCTF2BootstrapFidelitySuiteConfig,
    run_fmpc_tf2_bootstrap_fidelity_suite,
)


def run(**overrides: object):
    """Run the offline-first TF2 bootstrap-target fidelity suite."""

    config = FMPCTF2BootstrapFidelitySuiteConfig(**overrides)
    return run_fmpc_tf2_bootstrap_fidelity_suite(config)


def main() -> None:
    result = run()
    print("FMPC Stage 04 Incremental Bridge bootstrap-target fidelity suite completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Summary: {result.run_dir / 'aggregate_summary.json'}")


if __name__ == "__main__":
    main()
