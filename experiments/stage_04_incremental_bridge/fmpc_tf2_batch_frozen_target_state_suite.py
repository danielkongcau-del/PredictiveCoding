from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.stage_04_incremental_bridge.fmpc_tf2_batch_frozen_target_state_suite import (
    FMPCTF2BatchFrozenTargetStateSuiteConfig,
    run_fmpc_tf2_batch_frozen_target_state_suite,
)


def run(**overrides: object):
    """Run the diagnostic-only TF2 batch-frozen target/state coupling suite."""

    config = FMPCTF2BatchFrozenTargetStateSuiteConfig(**overrides)
    return run_fmpc_tf2_batch_frozen_target_state_suite(config)


def main() -> None:
    result = run()
    print("FMPC Stage 04 Incremental Bridge batch-frozen target/state coupling suite completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Summary: {result.run_dir / 'aggregate_summary.json'}")


if __name__ == "__main__":
    main()
