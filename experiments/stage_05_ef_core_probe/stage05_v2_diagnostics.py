from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.stage_05_ef_core_probe.stage05_v2_diagnostics import (
    Stage05V2DiagnosticsConfig,
    Stage05V2DiagnosticsRunResult,
    run_stage05_v2_diagnostics,
)


def run(
    output_root: str | Path = "outputs/stage_05_ef_core_probe",
    run_id: str | None = None,
    **overrides: object,
) -> Stage05V2DiagnosticsRunResult:
    """Run narrow diagnostics for the current Stage 05 v2 exploratory reference."""

    config = Stage05V2DiagnosticsConfig(
        output_root=output_root,
        run_id=run_id,
        **overrides,
    )
    return run_stage05_v2_diagnostics(config)


def main() -> None:
    result = run()
    print("Stage 05 v2 diagnostics completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Summary: {result.run_dir / 'diagnostic_summary.json'}")


if __name__ == "__main__":
    main()
