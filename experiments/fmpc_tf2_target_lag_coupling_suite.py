from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.fmpc_tf2_target_lag_coupling_suite import (
    FMPCTF2TargetLagCouplingSuiteConfig,
    run_fmpc_tf2_target_lag_coupling_suite,
)


def run(**overrides: object):
    """Run the diagnostic-only TF2 target-lag coupling suite."""

    config = FMPCTF2TargetLagCouplingSuiteConfig(**overrides)
    return run_fmpc_tf2_target_lag_coupling_suite(config)


def main() -> None:
    result = run()
    print("Phase TF2 target-lag coupling suite completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Summary: {result.run_dir / 'aggregate_summary.json'}")


if __name__ == "__main__":
    main()
