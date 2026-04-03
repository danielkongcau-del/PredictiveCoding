from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.fmpc_protocol import FMPCPreparationConfig, FMPCPreparationRunResult, run_fmpc_v0_preparation
from pc.real_pc import RealPCConfig


def run(
    dataset_name: str = "digits",
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    teacher_export_backend: str = "pc_euler",
    teacher_export_steps: int | None = None,
    export_trajectory: bool = False,
) -> FMPCPreparationRunResult:
    """Run the teacher-only FMPC-v0 preparation scaffold."""
    default_layer_dims = (64, 64, 10) if dataset_name == "digits" else (784, 64, 10)
    teacher_pc_config = RealPCConfig(
        dataset_name=dataset_name,
        layer_dims=default_layer_dims,
    )
    config = FMPCPreparationConfig(
        dataset_name=dataset_name,
        output_root=output_root,
        run_id=run_id,
        teacher_pc_config=teacher_pc_config,
        teacher_export_backend=teacher_export_backend,  # type: ignore[arg-type]
        teacher_export_steps=teacher_export_steps,
        export_trajectory=export_trajectory,
    )
    return run_fmpc_v0_preparation(config)


def main() -> None:
    result = run()
    print("FMPC-v0 preparation completed.")
    print("This was a teacher-only preparation run; no transporter was implemented.")
    print(f"Run directory: {result.run_dir}")
    print(f"Teacher model dir: {result.teacher_model_result.run_dir}")
    print(f"Teacher targets manifest: {result.run_dir / 'teacher_targets' / 'manifest.json'}")


if __name__ == "__main__":
    main()
