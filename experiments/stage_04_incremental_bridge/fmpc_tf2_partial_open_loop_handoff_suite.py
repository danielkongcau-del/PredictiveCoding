from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.stage_04_incremental_bridge.fmpc_tf2_partial_open_loop_handoff_suite import (
    FMPCTF2PartialOpenLoopHandoffSuiteConfig,
    run_fmpc_tf2_partial_open_loop_handoff_suite,
)


def run(**overrides: object):
    """Run the diagnostic-only TF2 partial-open-loop handoff suite."""

    config = FMPCTF2PartialOpenLoopHandoffSuiteConfig(**overrides)
    return run_fmpc_tf2_partial_open_loop_handoff_suite(config)


def main() -> None:
    result = run()
    print("FMPC Stage 04 Incremental Bridge partial-open-loop handoff suite completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Summary: {result.run_dir / 'aggregate_summary.json'}")


if __name__ == "__main__":
    main()
