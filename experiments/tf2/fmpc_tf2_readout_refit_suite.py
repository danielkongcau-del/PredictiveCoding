from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.tf2.fmpc_tf2_readout_refit_suite import (
    FMPCTF2ReadoutRefitSuiteConfig,
    FMPCTF2ReadoutRefitSuiteRunResult,
    run_fmpc_tf2_readout_refit_suite,
)


def run(
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    epochs: int = 60,
    batch_size: int = 128,
    eval_steps: int = 15,
    layer_dims: tuple[int, ...] = (64, 64, 10),
    reference_summary_path: str | Path = "outputs/tf2/fmpc_tf2_gap_decomposition_suite/aggregate_summary.json",
) -> FMPCTF2ReadoutRefitSuiteRunResult:
    """Run the narrow adopted-package readout-refit / endpoint-separability suite."""

    config = FMPCTF2ReadoutRefitSuiteConfig(
        output_root=output_root,
        run_id=run_id,
        seeds=seeds,
        epochs=epochs,
        batch_size=batch_size,
        eval_steps=eval_steps,
        layer_dims=layer_dims,
        reference_summary_path=reference_summary_path,
    )
    return run_fmpc_tf2_readout_refit_suite(config)


def main() -> None:
    """Run the default readout-refit suite and print a short summary."""

    result = run()
    summary = result.summary
    control = summary["by_case"]["adopted_control"]
    print("TF2 readout-refit suite completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Control mean test accuracy: {control['mean_test_accuracy']:.6f}")
    print(f"Diagnosis: {summary['diagnosis']}")
    print(f"Recommended next move: {summary['recommended_next_narrow_tf2_move']}")


if __name__ == "__main__":
    main()
