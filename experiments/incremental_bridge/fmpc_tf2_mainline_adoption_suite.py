from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.incremental_bridge.fmpc_tf2_mainline_adoption_suite import (
    FMPCTF2MainlineAdoptionSuiteConfig,
    run_fmpc_tf2_mainline_adoption_suite,
)


def run(**overrides: object):
    """Run the integrated TF2 mainline adoption confirmation suite."""

    config = FMPCTF2MainlineAdoptionSuiteConfig(**overrides)
    return run_fmpc_tf2_mainline_adoption_suite(config)


def main() -> None:
    result = run()
    print("Phase Incremental Bridge mainline adoption suite completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Summary: {result.run_dir / 'aggregate_summary.json'}")


if __name__ == "__main__":
    main()
