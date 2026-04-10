from __future__ import annotations

from pathlib import Path

from pc.transport_core_v1.fmpc_tf1_accuracy_tuning_suite import (
    FMPCTF1AccuracyTuningSuiteConfig,
    FMPCTF1AccuracyTuningSuiteRunResult,
    run_fmpc_tf1_accuracy_tuning_suite,
)


def run(output_root: str | Path = "outputs", run_id: str | None = None) -> FMPCTF1AccuracyTuningSuiteRunResult:
    """Run the very narrow TF1 accuracy-improvement pass around the working default."""

    config = FMPCTF1AccuracyTuningSuiteConfig(output_root=output_root, run_id=run_id)
    return run_fmpc_tf1_accuracy_tuning_suite(config)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
