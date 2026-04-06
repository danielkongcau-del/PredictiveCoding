from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.real_pc_inference_baselines import (
    RealPCInferenceBaselineStudyConfig,
    RealPCInferenceBaselineStudyResult,
    run_real_pc_inference_baseline_study,
)


def run(
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    plot_curves: bool = False,
    config: RealPCInferenceBaselineStudyConfig | None = None,
) -> RealPCInferenceBaselineStudyResult:
    """Run a narrow standalone digits PC inference-baseline study."""
    if config is None:
        config = RealPCInferenceBaselineStudyConfig(
            output_root=output_root,
            run_id=run_id,
            plot_curves=plot_curves,
        )
    return run_real_pc_inference_baseline_study(config)


def main() -> None:
    """Run the default digits PC inference-baseline study and print a short summary."""
    result = run()
    selected = result.aggregate_summary["selected_candidate"]
    print("Digits PC inference-baseline study completed.")
    print(f"Run directory: {result.run_dir}")
    print(
        "Default reruns overwrite the stable outputs/digits_pc_inference_baselines directory unless you choose a different output target."
    )
    print(f"Selected by: {result.aggregate_summary['selected_by']}")
    print(f"Selected candidate: {selected['candidate_id']}")
    print(f"Selected inference backend: {selected['inference_backend']}")
    print(f"Selected inference method: {selected['inference_method']}")
    print(f"Selected val accuracy: {float(selected['val_metric']):.6f}")
    print(f"Selected test accuracy: {float(selected['test_metric']):.6f}")


if __name__ == "__main__":
    main()
