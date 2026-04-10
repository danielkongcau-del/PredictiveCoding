from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.incremental_bridge.fmpc_tf2_identity_semantics_suite import (
    FMPCTF2IdentitySemanticsSuiteConfig,
    FMPCTF2IdentitySemanticsSuiteRunResult,
    run_fmpc_tf2_identity_semantics_suite,
)


def run(
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    **overrides: object,
) -> FMPCTF2IdentitySemanticsSuiteRunResult:
    """Run the narrow TF2 identity-semantics decision suite.

    This suite compares:
    - truncated identity semantics:
      - `use_teacher_free_features = true`
      - `feature_aware_tangents = false`
    - feature-aware augmented-input identity semantics:
      - `use_teacher_free_features = true`
      - `feature_aware_tangents = true`

    The comparison stays inside the existing TF2 presets and only toggles the
    identity-tangent semantics.
    """

    config = FMPCTF2IdentitySemanticsSuiteConfig(
        output_root=output_root,
        run_id=run_id,
        **overrides,
    )
    return run_fmpc_tf2_identity_semantics_suite(config)


def main() -> None:
    result = run()
    print("Phase Incremental Bridge identity-semantics suite completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Summary: {result.run_dir / 'aggregate_summary.json'}")


if __name__ == "__main__":
    main()
