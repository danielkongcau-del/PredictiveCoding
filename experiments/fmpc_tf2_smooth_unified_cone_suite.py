from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.fmpc_tf2_smooth_unified_cone_suite import (
    FMPCTF2SmoothUnifiedConeSuiteConfig,
    FMPCTF2SmoothUnifiedConeSuiteRunResult,
    run_fmpc_tf2_smooth_unified_cone_suite,
)


def run(
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    epochs: int = 60,
    batch_size: int = 128,
    eval_steps: int = 15,
    layer_dims: tuple[int, ...] = (64, 64, 10),
) -> FMPCTF2SmoothUnifiedConeSuiteRunResult:
    """Run the adopted-package smooth unified-cone diagnostic."""

    config = FMPCTF2SmoothUnifiedConeSuiteConfig(
        output_root=output_root,
        run_id=run_id,
        seeds=seeds,
        epochs=epochs,
        batch_size=batch_size,
        eval_steps=eval_steps,
        layer_dims=layer_dims,
    )
    return run_fmpc_tf2_smooth_unified_cone_suite(config)


def main() -> None:
    result = run()
    summary = result.summary
    print("TF2 smooth unified-cone suite completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Decision: {summary['decision']}")
    print(f"Diagnosis: {summary['diagnosis']}")


if __name__ == "__main__":
    main()
