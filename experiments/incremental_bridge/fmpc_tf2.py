from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.incremental_bridge.fmpc_tf2 import (
    FMPCTF2RunResult,
    TF2PresetName,
    build_tf2_preset_config,
    run_fmpc_tf2_experiment,
)


def run(
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    preset_name: TF2PresetName = "tf2_corrective_transport_terminal_angleclip_default",
    **overrides: object,
) -> FMPCTF2RunResult:
    """Run a named TF2 bridge preset on digits.

    Presets:
    - `tf2_corrective_transport_terminal_angleclip_default`: current adopted TF2
      experimental default on `main`.
    - `tf2_corrective_transport_default`: historical plain corrective working
      reference.
    - `tf2_canonical`: hypothesis-driven iFMPC candidate with incremental theta
      updates and mixed supervision.
    """

    config = build_tf2_preset_config(
        preset_name,
        output_root=output_root,
        run_id=run_id,
        **overrides,
    )
    return run_fmpc_tf2_experiment(config)


def main() -> None:
    result = run()
    print("Phase Incremental Bridge iFMPC bridge run completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Preset: {result.summary.get('preset_name')}")
    print(f"Summary: {result.run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
