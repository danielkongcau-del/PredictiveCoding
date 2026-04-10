from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.stage_04_incremental_bridge.fmpc_tf2_curriculum_suite import (
    FMPCTF2CurriculumSuiteConfig,
    run_fmpc_tf2_curriculum_suite,
)


def run(**overrides: object):
    """Run the narrow TF2 corrective curriculum suite."""

    config = FMPCTF2CurriculumSuiteConfig(**overrides)
    return run_fmpc_tf2_curriculum_suite(config)


def main() -> None:
    result = run()
    print("FMPC Stage 04 Incremental Bridge corrective curriculum suite completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Summary: {result.run_dir / 'aggregate_summary.json'}")


if __name__ == "__main__":
    main()
