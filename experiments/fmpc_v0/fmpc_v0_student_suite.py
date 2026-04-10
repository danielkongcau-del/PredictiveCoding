from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.fmpc_v0.fmpc_student_suite import (
    FMPCStudentSuiteConfig,
    FMPCStudentSuiteRunResult,
    run_fmpc_student_suite,
)


def run(
    teacher_preparation_path: str | Path = "outputs/fmpc_v0_prepare_digits",
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    ridge_alphas: tuple[float, ...] = (1e-4, 1e-2, 1.0, 100.0),
    mlp_hidden_dims_candidates: tuple[tuple[int, ...], ...] = ((64,), (128,), (128, 128)),
    mlp_epochs_candidates: tuple[int, ...] = (20, 40),
    mlp_eta_w_candidates: tuple[float, ...] = (0.01, 0.05),
    mlp_batch_size: int = 64,
) -> FMPCStudentSuiteRunResult:
    """Run the Phase 5A endpoint student baseline suite on digits teacher targets."""

    config = FMPCStudentSuiteConfig(
        teacher_preparation_path=teacher_preparation_path,
        output_root=output_root,
        run_id=run_id,
        ridge_alphas=ridge_alphas,
        mlp_hidden_dims_candidates=mlp_hidden_dims_candidates,
        mlp_epochs_candidates=mlp_epochs_candidates,
        mlp_eta_w_candidates=mlp_eta_w_candidates,
        mlp_batch_size=mlp_batch_size,
    )
    return run_fmpc_student_suite(config)


def main() -> None:
    result = run()
    print("Phase 5A endpoint student baseline suite completed.")
    print("This run compares identity, class-mean, ridge, and standardized-MLP baselines.")
    print("It stays within endpoint-only offline FMPC-v0 scope.")
    print(f"Run directory: {result.run_dir}")
    print(f"Summary: {result.run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
