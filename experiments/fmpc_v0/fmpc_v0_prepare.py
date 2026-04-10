from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.fmpc_v0.fmpc_protocol import FMPCPreparationConfig, FMPCPreparationRunResult, run_fmpc_v0_preparation
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


def run_phase5_digits_validation_teacher(
    output_root: str | Path = "outputs/phase5_validation",
    run_id: str | None = None,
) -> FMPCPreparationRunResult:
    """Run the non-trivial canonical digits teacher recipe used for Phase 5 validation."""
    teacher_pc_config = RealPCConfig(
        dataset_name="digits",
        layer_dims=(64, 64, 10),
        epochs=60,
        batch_size=64,
        train_steps=30,
        eval_steps=30,
        inference_backend="pc_euler",
        state_init="forward",
    )
    return run_fmpc_v0_preparation(
        FMPCPreparationConfig(
            dataset_name="digits",
            output_root=output_root,
            experiment_name="fmpc_v0_prepare_digits_validation",
            run_id=run_id,
            teacher_pc_config=teacher_pc_config,
            teacher_export_backend="pc_euler",
            teacher_export_steps=None,
            export_trajectory=False,
        )
    )


def run_phase5b_digits_validation_teacher(
    output_root: str | Path = "outputs/phase5b_validation",
    run_id: str | None = None,
) -> FMPCPreparationRunResult:
    """Run the canonical digits trajectory teacher recipe used for Phase 5B validation."""

    teacher_pc_config = RealPCConfig(
        dataset_name="digits",
        layer_dims=(64, 64, 10),
        epochs=60,
        batch_size=64,
        train_steps=30,
        eval_steps=30,
        inference_backend="pc_euler",
        state_init="forward",
    )
    return run_fmpc_v0_preparation(
        FMPCPreparationConfig(
            dataset_name="digits",
            output_root=output_root,
            experiment_name="fmpc_v0_prepare_digits_interval_validation",
            run_id=run_id,
            teacher_pc_config=teacher_pc_config,
            teacher_export_backend="pc_euler",
            teacher_export_steps=None,
            export_trajectory=True,
        )
    )


def main() -> None:
    result = run()
    print("FMPC-v0 preparation completed.")
    print("This was a teacher-only preparation run; no transporter was implemented.")
    print(f"Run directory: {result.run_dir}")
    print(f"Teacher model dir: {result.teacher_model_result.run_dir}")
    print(f"Teacher targets manifest: {result.run_dir / 'teacher_targets' / 'manifest.json'}")


if __name__ == "__main__":
    main()
