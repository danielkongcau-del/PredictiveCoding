from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.incremental_bridge.fmpc_tf2_bootstrap_source_bias_suite import (
    FMPCTF2BootstrapSourceBiasSuiteConfig,
    run_fmpc_tf2_bootstrap_source_bias_suite,
)


def run(**overrides: object):
    """Run the offline-first TF2 bootstrap-source bias suite."""

    config = FMPCTF2BootstrapSourceBiasSuiteConfig(**overrides)
    return run_fmpc_tf2_bootstrap_source_bias_suite(config)


def main() -> None:
    result = run()
    print("Phase Incremental Bridge bootstrap-source bias suite completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Summary: {result.run_dir / 'aggregate_summary.json'}")


if __name__ == "__main__":
    main()
