from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.stage_04_incremental_bridge.fmpc_tf2_output_sensitive_terminal_direction_suite import (
    FMPCTF2OutputSensitiveTerminalDirectionSuiteConfig,
    FMPCTF2OutputSensitiveTerminalDirectionSuiteRunResult,
    run_fmpc_tf2_output_sensitive_terminal_direction_suite,
)


def run(
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    epochs: int = 60,
    batch_size: int = 128,
    eval_steps: int = 15,
    layer_dims: tuple[int, ...] = (64, 64, 10),
    reference_summary_path: str | Path = "outputs/stage_04_incremental_bridge/fmpc_tf2_gap_decomposition_suite/aggregate_summary.json",
) -> FMPCTF2OutputSensitiveTerminalDirectionSuiteRunResult:
    """Run the narrow adopted-package output-sensitive terminal direction suite."""

    config = FMPCTF2OutputSensitiveTerminalDirectionSuiteConfig(
        output_root=output_root,
        run_id=run_id,
        seeds=seeds,
        epochs=epochs,
        batch_size=batch_size,
        eval_steps=eval_steps,
        layer_dims=layer_dims,
        reference_summary_path=reference_summary_path,
    )
    return run_fmpc_tf2_output_sensitive_terminal_direction_suite(config)


def main() -> None:
    result = run()
    summary = result.summary
    print("TF2 output-sensitive terminal direction suite completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Decision: {summary['decision']}")
    print(f"Promoted candidate: {summary['promoted_candidate_name']}")


if __name__ == "__main__":
    main()
