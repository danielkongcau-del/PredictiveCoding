from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.tf2.fmpc_tf2_psi_expressivity_suite import (
    FMPCTF2PsiExpressivitySuiteConfig,
    run_fmpc_tf2_psi_expressivity_suite,
)


def run(**overrides: object):
    """Run the offline-first TF2 psi expressivity suite."""

    config = FMPCTF2PsiExpressivitySuiteConfig(**overrides)
    return run_fmpc_tf2_psi_expressivity_suite(config)


def main() -> None:
    result = run()
    print("Phase TF2 psi-expressivity suite completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Summary: {result.run_dir / 'aggregate_summary.json'}")


if __name__ == "__main__":
    main()
