from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.stage_06_low_budget_efficiency.fmpc_stage06_objective_curriculum import (
    Stage06ObjectiveCurriculumRunResult,
    build_stage06_v1_objective_curriculum_energydrop_default_config,
    run_stage06_objective_curriculum,
)


def run(
    output_root: str | Path = "outputs/stage_06_low_budget_efficiency",
    run_id: str | None = None,
    **overrides: object,
) -> Stage06ObjectiveCurriculumRunResult:
    config = build_stage06_v1_objective_curriculum_energydrop_default_config(
        output_root=output_root,
        run_id=run_id,
        **overrides,
    )
    return run_stage06_objective_curriculum(config)


def main() -> None:
    result = run()
    print("Stage 06 v1 objective-curriculum probe completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Summary: {result.run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
