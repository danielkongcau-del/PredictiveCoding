from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.jpc_bridge import TF2JPCProbeConfig, TF2JPCProbeRunResult, run_tf2_jpc_probe


def run(
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    batch_size: int = 8,
    inference_steps_horizon: int = 8,
) -> TF2JPCProbeRunResult:
    """Run the minimal TF2 JPC bridge probe.

    This is a diagnostic/reference probe only. It does not modify TF1/TF2
    mainline training paths and does not require JPC to be available.
    """

    config = TF2JPCProbeConfig(
        output_root=output_root,
        run_id=run_id,
        batch_size=batch_size,
        inference_steps_horizon=inference_steps_horizon,
    )
    return run_tf2_jpc_probe(config)


def main() -> None:
    result = run()
    print("TF2 JPC bridge probe completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Summary: {result.run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
