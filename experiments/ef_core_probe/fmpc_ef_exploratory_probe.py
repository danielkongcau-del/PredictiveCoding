from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.ef_core_probe.fmpc_ef_exploratory_probe import (
    FMPCEFExploratoryProbeRunResult,
    build_fmpc_ef_exploratory_probe_config,
    run_fmpc_ef_exploratory_probe,
)


def run(
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    **overrides: object,
) -> FMPCEFExploratoryProbeRunResult:
    """Run the first post-bridge teacher-free exploratory core probe on digits."""

    config = build_fmpc_ef_exploratory_probe_config(
        output_root=output_root,
        run_id=run_id,
        **overrides,
    )
    return run_fmpc_ef_exploratory_probe(config)


def main() -> None:
    result = run()
    print("Post-bridge teacher-free exploratory probe completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Summary: {result.run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
