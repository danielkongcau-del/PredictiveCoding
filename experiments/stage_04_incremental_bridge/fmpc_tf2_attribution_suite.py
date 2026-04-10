from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.stage_04_incremental_bridge.fmpc_tf2_attribution_suite import (
    FMPCTF2AttributionSuiteConfig,
    run_fmpc_tf2_attribution_suite,
)


def run(**overrides: object):
    """Run the narrow TF2 corrective-transport attribution suite."""

    config = FMPCTF2AttributionSuiteConfig(**overrides)
    return run_fmpc_tf2_attribution_suite(config)


def main() -> None:
    result = run()
    print("FMPC Stage 04 Incremental Bridge corrective-transport attribution suite completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Summary: {result.run_dir / 'aggregate_summary.json'}")


if __name__ == "__main__":
    main()
