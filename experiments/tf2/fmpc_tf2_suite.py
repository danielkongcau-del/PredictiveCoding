from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.tf2.fmpc_tf2_suite import FMPCTF2SuiteConfig, FMPCTF2SuiteRunResult, run_fmpc_tf2_suite


def run(
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    **overrides: object,
) -> FMPCTF2SuiteRunResult:
    """Run the narrow TF2 bridge-validation suite."""

    config = FMPCTF2SuiteConfig(
        output_root=output_root,
        run_id=run_id,
        **overrides,
    )
    return run_fmpc_tf2_suite(config)


def main() -> None:
    result = run()
    print("Phase TF2 iFMPC bridge suite completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Summary: {result.run_dir / 'aggregate_summary.json'}")


if __name__ == "__main__":
    main()
