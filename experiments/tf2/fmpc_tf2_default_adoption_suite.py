from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.tf2.fmpc_tf2_default_adoption_suite import (
    FMPCTF2DefaultAdoptionSuiteConfig,
    FMPCTF2DefaultAdoptionSuiteRunResult,
    run_fmpc_tf2_default_adoption_suite,
)


def run(
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    **overrides: object,
) -> FMPCTF2DefaultAdoptionSuiteRunResult:
    """Run the narrow TF2 default-adoption validation suite.

    This suite compares:
    - the hypothesis-driven TF2 canonical preset
    - the empirical corrective-transport TF2 default
    - the sealed TF1 working default reference
    - the canonical slow-PC digits baseline reference
    """

    config = FMPCTF2DefaultAdoptionSuiteConfig(
        output_root=output_root,
        run_id=run_id,
        **overrides,
    )
    return run_fmpc_tf2_default_adoption_suite(config)


def main() -> None:
    result = run()
    print("Phase TF2 default-adoption suite completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Summary: {result.run_dir / 'aggregate_summary.json'}")


if __name__ == "__main__":
    main()
