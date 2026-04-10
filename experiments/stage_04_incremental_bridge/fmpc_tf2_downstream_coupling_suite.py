from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.stage_04_incremental_bridge.fmpc_tf2_downstream_coupling_suite import (
    FMPCTF2DownstreamCouplingSuiteConfig,
    run_fmpc_tf2_downstream_coupling_suite,
)


def run(**overrides: object):
    """Run the diagnostic-only TF2 downstream-coupling suite."""

    config = FMPCTF2DownstreamCouplingSuiteConfig(**overrides)
    return run_fmpc_tf2_downstream_coupling_suite(config)


def main() -> None:
    result = run()
    print("FMPC Stage 04 Incremental Bridge downstream-coupling suite completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Summary: {result.run_dir / 'aggregate_summary.json'}")


if __name__ == "__main__":
    main()
