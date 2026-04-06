from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.fmpc_tf2b_interleaving_suite import (
    FMPCTF2BInterleavingSuiteConfig,
    FMPCTF2BInterleavingSuiteRunResult,
    run_fmpc_tf2b_interleaving_suite,
)


def run(
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    **overrides: object,
) -> FMPCTF2BInterleavingSuiteRunResult:
    """Run the narrow TF2B interleaving-rescue study.

    The study stays inside the current corrective-transport TF2 family and varies
    only:
    - theta update cadence
    - low-ratio on-policy supervision
    - delayed interleaving start
    """

    config = FMPCTF2BInterleavingSuiteConfig(
        output_root=output_root,
        run_id=run_id,
        **overrides,
    )
    return run_fmpc_tf2b_interleaving_suite(config)


def main() -> None:
    result = run()
    print("Phase TF2B interleaving-rescue suite completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Summary: {result.run_dir / 'aggregate_summary.json'}")


if __name__ == "__main__":
    main()
