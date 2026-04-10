from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.stage_04_incremental_bridge.fmpc_tf2_gap_decomposition_suite import (
    FMPCTF2GapDecompositionSuiteConfig,
    FMPCTF2GapDecompositionSuiteRunResult,
    run_fmpc_tf2_gap_decomposition_suite,
)


def run(
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    tf2_epochs: int = 60,
    tf2_batch_size: int = 128,
    tf2_eval_steps: int = 15,
    tf2_layer_dims: tuple[int, ...] = (64, 64, 10),
    slow_pc_epochs: int = 60,
    slow_pc_batch_size: int = 64,
    slow_pc_train_steps: int = 30,
    slow_pc_eval_steps: int = 30,
    slow_pc_layer_dims: tuple[int, ...] = (64, 64, 10),
) -> FMPCTF2GapDecompositionSuiteRunResult:
    """Run the narrow adopted-package vs slow-PC gap-decomposition suite."""

    config = FMPCTF2GapDecompositionSuiteConfig(
        output_root=output_root,
        run_id=run_id,
        seeds=seeds,
        tf2_epochs=tf2_epochs,
        tf2_batch_size=tf2_batch_size,
        tf2_eval_steps=tf2_eval_steps,
        tf2_layer_dims=tf2_layer_dims,
        slow_pc_epochs=slow_pc_epochs,
        slow_pc_batch_size=slow_pc_batch_size,
        slow_pc_train_steps=slow_pc_train_steps,
        slow_pc_eval_steps=slow_pc_eval_steps,
        slow_pc_layer_dims=slow_pc_layer_dims,
    )
    return run_fmpc_tf2_gap_decomposition_suite(config)


def main() -> None:
    """Run the default gap-decomposition suite and print a short summary."""

    result = run()
    summary = result.summary
    adopted = summary["by_method"]["tf2_corrective_transport_terminal_angleclip_default"]
    slow_pc = summary["by_method"]["canonical_slow_pc_digits_baseline"]
    print("TF2 gap-decomposition suite completed.")
    print(f"Run directory: {result.run_dir}")
    print(
        "Adopted-vs-slow-PC mean test accuracy gap: "
        f"{adopted['mean_test_accuracy'] - slow_pc['mean_test_accuracy']:.6f}"
    )
    print(f"Diagnosis: {summary['remaining_gap_primary_diagnosis']}")
    print(f"Recommended next move: {summary['recommended_next_narrow_tf2_move']}")


if __name__ == "__main__":
    main()
