from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.stage_02_interval_velocity.fmpc_meanflow_student import (
    FMPCMeanFlowSuiteConfig,
    FMPCMeanFlowSuiteRunResult,
    run_fmpc_meanflow_suite,
)


def run(
    teacher_preparation_path: str | Path = "outputs/fmpc_v0_prepare_digits_interval_validation",
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    hidden_dims_candidates: tuple[tuple[int, ...], ...] = ((64,), (128,)),
    epochs_candidates: tuple[int, ...] = (40,),
    eta_w_candidates: tuple[float, ...] = (0.01, 0.05),
    identity_loss_weight_candidates: tuple[float, ...] = (0.02, 0.05, 0.1),
    rollout_aux_weight_candidates: tuple[float, ...] = (0.0,),
    knot_focus_probability_candidates: tuple[float, ...] = (0.0,),
    feature_tangent_epsilon: float = 1e-3,
) -> FMPCMeanFlowSuiteRunResult:
    """Run the Phase 6A.3 warm-started two-branch MeanFlow teacher-supervised suite on `digits`."""

    config = FMPCMeanFlowSuiteConfig(
        teacher_preparation_path=teacher_preparation_path,
        output_root=output_root,
        run_id=run_id,
        hidden_dims_candidates=hidden_dims_candidates,
        epochs_candidates=epochs_candidates,
        eta_w_candidates=eta_w_candidates,
        identity_loss_weight_candidates=identity_loss_weight_candidates,
        rollout_aux_weight_candidates=rollout_aux_weight_candidates,
        knot_focus_probability_candidates=knot_focus_probability_candidates,
        feature_tangent_epsilon=feature_tangent_epsilon,
    )
    return run_fmpc_meanflow_suite(config)


def main() -> None:
    result = run()
    print("Phase 6A.3 warm-started two-branch MeanFlow suite completed.")
    print(
        "This run compares identity, the carried-forward Phase 5A/5B.2/6A.1/6A.2 baselines, "
        "plus monolithic MeanFlow, diagnostic linear-residual, scratch two-branch, and warm-started two-branch students."
    )
    print(f"Run directory: {result.run_dir}")
    print(f"Summary: {result.run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
