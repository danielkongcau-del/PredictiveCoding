from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.fmpc_v0.fmpc_student import FMPCStudentConfig, FMPCStudentRunResult, run_fmpc_student_experiment


def run(
    teacher_preparation_path: str | Path = "outputs/fmpc_v0_prepare_digits",
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    hidden_dims: tuple[int, ...] = (64,),
    epochs: int = 20,
    batch_size: int = 64,
    allow_teacher_retrain: bool = False,
) -> FMPCStudentRunResult:
    """Run the minimal offline FMPC-v0 student on digits teacher targets."""

    config = FMPCStudentConfig(
        teacher_preparation_path=teacher_preparation_path,
        output_root=output_root,
        run_id=run_id,
        hidden_dims=hidden_dims,
        epochs=epochs,
        batch_size=batch_size,
        allow_teacher_retrain=allow_teacher_retrain,
    )
    return run_fmpc_student_experiment(config)


def main() -> None:
    result = run()
    print("Offline FMPC-v0 student run completed.")
    print("This run trains a NumPy endpoint transporter on frozen teacher-target artifacts.")
    print("It does not implement a core FMPC backend or a formal comparison pipeline.")
    print(f"Run directory: {result.run_dir}")
    print(f"Summary: {result.run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
