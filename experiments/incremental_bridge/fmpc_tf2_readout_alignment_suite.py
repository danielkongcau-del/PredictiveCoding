from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.incremental_bridge.fmpc_tf2_readout_alignment_suite import (
    FMPCTF2ReadoutAlignmentSuiteConfig,
    FMPCTF2ReadoutAlignmentSuiteRunResult,
    run_fmpc_tf2_readout_alignment_suite,
)


def run(
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    epochs: int = 60,
    batch_size: int = 128,
    eval_steps: int = 15,
    layer_dims: tuple[int, ...] = (64, 64, 10),
) -> FMPCTF2ReadoutAlignmentSuiteRunResult:
    """Run the narrow adopted-package readout-alignment confirmation suite."""

    config = FMPCTF2ReadoutAlignmentSuiteConfig(
        output_root=output_root,
        run_id=run_id,
        seeds=seeds,
        epochs=epochs,
        batch_size=batch_size,
        eval_steps=eval_steps,
        layer_dims=layer_dims,
    )
    return run_fmpc_tf2_readout_alignment_suite(config)


def main() -> None:
    """Run the default readout-alignment suite and print a short summary."""

    result = run()
    summary = result.summary
    control = summary["by_candidate"]["adopted_control"]
    print("TF2 readout-alignment suite completed.")
    print(f"Run directory: {result.run_dir}")
    print(
        "Control mean test accuracy: "
        f"{control['mean_test_accuracy']:.6f}"
    )
    print(f"Adoption decision: {summary['adoption_decision']}")
    print(f"Promoted candidate: {summary['promoted_candidate_name']}")


if __name__ == "__main__":
    main()
