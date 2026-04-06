from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.fmpc_tf2 import (
    FMPCTF2RunResult,
    TF2PresetName,
    build_tf2_preset_config,
    run_fmpc_tf2_experiment,
)


def run(
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    preset_name: TF2PresetName = "tf2_canonical",
    **overrides: object,
) -> FMPCTF2RunResult:
    """Run a named TF2 bridge preset on digits.

    Presets:
    - `tf2_canonical`: hypothesis-driven iFMPC candidate with incremental
      theta updates and mixed supervision.
    - `tf2_corrective_transport_default`: empirical corrective-transport
      working default from the current TF2 suite.
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
    print("Phase TF2 iFMPC bridge run completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Preset: {result.summary.get('preset_name')}")
    print(f"Summary: {result.run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
