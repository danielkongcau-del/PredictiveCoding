from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.stage_02_interval_velocity.fmpc_interval_student import (
    FMPCIntervalSuiteConfig,
    FMPCIntervalSuiteRunResult,
    run_fmpc_interval_suite,
)


def run(
    teacher_preparation_path: str | Path = "outputs/fmpc_v0_prepare_digits",
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    interval_ridge_alphas: tuple[float, ...] = (1e-4, 1e-2, 1.0, 100.0),
    gradient_augmented_feature_names: tuple[str, ...] = ("g_s", "e_out_s", "F_s"),
    augmented_knot_focused_schedule_names: tuple[str, ...] = ("2-step", "3-step"),
    augmented_knot_focus_mixture_candidates: tuple[float, ...] = (0.0, 0.5),
    mlp_hidden_dims_candidates: tuple[tuple[int, ...], ...] = ((64,), (128,), (128, 128)),
    mlp_epochs_candidates: tuple[int, ...] = (20, 40),
    mlp_eta_w_candidates: tuple[float, ...] = (0.01, 0.05),
    mlp_batch_size: int = 64,
    mlp_rollout_aux_schedule_names: tuple[str, ...] = ("2-step", "3-step"),
    mlp_rollout_aux_weight_candidates: tuple[float, ...] = (0.0, 0.25, 0.5),
) -> FMPCIntervalSuiteRunResult:
    """Run the Phase 5B interval-conditioned student suite on trajectory teacher targets."""

    config = FMPCIntervalSuiteConfig(
        teacher_preparation_path=teacher_preparation_path,
        output_root=output_root,
        run_id=run_id,
        interval_ridge_alphas=interval_ridge_alphas,
        gradient_augmented_feature_names=gradient_augmented_feature_names,
        augmented_knot_focused_schedule_names=augmented_knot_focused_schedule_names,
        augmented_knot_focus_mixture_candidates=augmented_knot_focus_mixture_candidates,
        mlp_hidden_dims_candidates=mlp_hidden_dims_candidates,
        mlp_epochs_candidates=mlp_epochs_candidates,
        mlp_eta_w_candidates=mlp_eta_w_candidates,
        mlp_batch_size=mlp_batch_size,
        mlp_rollout_aux_schedule_names=mlp_rollout_aux_schedule_names,
        mlp_rollout_aux_weight_candidates=mlp_rollout_aux_weight_candidates,
    )
    return run_fmpc_interval_suite(config)


def main() -> None:
    result = run()
    print("Phase 5B interval-conditioned student suite completed.")
    print(
        "This run compares identity, carried-forward endpoint ridge, interval ridge, "
        "gradient-augmented ridge variants, and interval standardized MLP."
    )
    print(
        "The gradient-augmented rescue adds frozen-teacher current-state dynamical features, "
        "while staying below MeanFlow / JVP / refinement scope."
    )
    print(f"Run directory: {result.run_dir}")
    print(f"Summary: {result.run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
