from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.stage_05_ef_core_probe.frozen_bridge_vs_corrected_core_comparison import (
    CorrectedResidualCoreV1VsV2ComparisonConfig,
    FrozenBridgeVsCorrectedCoreComparisonConfig,
    FrozenBridgeVsCorrectedCoreComparisonRunResult,
    FrozenBridgeVsStage05V2ComparisonConfig,
    Stage05V3BRefinementDiagnosticConfig,
    Stage05V3CRefinementDiagnosticConfig,
    Stage05V2ActiveV3CFusedComparisonConfig,
    Stage05V2ActiveV3CMidpointReconstructedComparisonConfig,
    Stage05V2PromotedV3BV3CComparisonConfig,
    Stage05V2VsV3AComparisonConfig,
    Stage05V2V3AV3BComparisonConfig,
    Stage05V2BudgetPushValidationConfig,
    Stage05V2EfficiencyDiagnosticConfig,
    Stage05V2LongerTrainingValidationConfig,
    run_frozen_bridge_vs_corrected_core_comparison,
    run_corrected_residual_core_v1_vs_v2_comparison,
    run_frozen_bridge_vs_stage05_v2_comparison,
    run_stage05_v3b_refinement_diagnostic,
    run_stage05_v3c_refinement_diagnostic,
    run_stage05_v2_active_v3c_fused_contract_comparison,
    run_stage05_v2_active_v3c_midpoint_reconstructed_contract_comparison,
    run_stage05_v2_promoted_v3b_v3c_comparison,
    run_stage05_v2_vs_v3a_comparison,
    run_stage05_v2_v3a_v3b_comparison,
    run_stage05_v2_budget_push_validation,
    run_stage05_v2_efficiency_diagnostic,
    run_stage05_v2_longer_training_validation,
)


def run(
    output_root: str | Path = "outputs/stage_05_ef_core_probe",
    run_id: str | None = None,
    comparison_variant: str = "stage04_vs_stage05_v1",
    **overrides: object,
) -> FrozenBridgeVsCorrectedCoreComparisonRunResult:
    """Run a formal Stage 05 comparison entrypoint."""

    if comparison_variant == "stage04_vs_stage05_v1":
        config = FrozenBridgeVsCorrectedCoreComparisonConfig(
            output_root=output_root,
            run_id=run_id,
            **overrides,
        )
        return run_frozen_bridge_vs_corrected_core_comparison(config)
    if comparison_variant == "stage05_v1_vs_v2":
        config = CorrectedResidualCoreV1VsV2ComparisonConfig(
            output_root=output_root,
            run_id=run_id,
            **overrides,
        )
        return run_corrected_residual_core_v1_vs_v2_comparison(config)
    if comparison_variant == "stage05_v2_vs_v3a":
        config = Stage05V2VsV3AComparisonConfig(
            output_root=output_root,
            run_id=run_id,
            **overrides,
        )
        return run_stage05_v2_vs_v3a_comparison(config)
    if comparison_variant == "stage05_v2_v3a_v3b":
        config = Stage05V2V3AV3BComparisonConfig(
            output_root=output_root,
            run_id=run_id,
            **overrides,
        )
        return run_stage05_v2_v3a_v3b_comparison(config)
    if comparison_variant == "stage05_v2_v3a_refined_v3b_recompare":
        config = Stage05V2V3AV3BComparisonConfig(
            output_root=output_root,
            run_id=run_id,
            **overrides,
        )
        return run_stage05_v2_v3a_v3b_comparison(config)
    if comparison_variant == "stage04_vs_stage05_v2":
        config = FrozenBridgeVsStage05V2ComparisonConfig(
            output_root=output_root,
            run_id=run_id,
            **overrides,
        )
        return run_frozen_bridge_vs_stage05_v2_comparison(config)
    if comparison_variant == "stage05_v2_longer_training_validation":
        config = Stage05V2LongerTrainingValidationConfig(
            output_root=output_root,
            run_id=run_id,
            **overrides,
        )
        return run_stage05_v2_longer_training_validation(config)
    if comparison_variant == "stage05_v2_budget_push_validation":
        config = Stage05V2BudgetPushValidationConfig(
            output_root=output_root,
            run_id=run_id,
            **overrides,
        )
        return run_stage05_v2_budget_push_validation(config)
    if comparison_variant == "stage05_v2_efficiency_diagnostic_at_1536":
        config = Stage05V2EfficiencyDiagnosticConfig(
            output_root=output_root,
            run_id=run_id,
            **overrides,
        )
        return run_stage05_v2_efficiency_diagnostic(config)
    if comparison_variant == "stage05_v3b_refinement_diagnostic":
        config = Stage05V3BRefinementDiagnosticConfig(
            output_root=output_root,
            run_id=run_id,
            **overrides,
        )
        return run_stage05_v3b_refinement_diagnostic(config)
    if comparison_variant == "stage05_v3c_refinement_diagnostic":
        config = Stage05V3CRefinementDiagnosticConfig(
            output_root=output_root,
            run_id=run_id,
            **overrides,
        )
        return run_stage05_v3c_refinement_diagnostic(config)
    if comparison_variant == "stage05_v2_promoted_v3b_v3c":
        config = Stage05V2PromotedV3BV3CComparisonConfig(
            output_root=output_root,
            run_id=run_id,
            **overrides,
        )
        return run_stage05_v2_promoted_v3b_v3c_comparison(config)
    if comparison_variant == "stage05_v2_promoted_v3b_refined_v3c_recompare":
        config = Stage05V2PromotedV3BV3CComparisonConfig(
            output_root=output_root,
            run_id=run_id,
            **overrides,
        )
        return run_stage05_v2_promoted_v3b_v3c_comparison(config)
    if comparison_variant == "stage05_v2_active_v3c_fused_contract_comparison":
        config = Stage05V2ActiveV3CFusedComparisonConfig(
            output_root=output_root,
            run_id=run_id,
            **overrides,
        )
        return run_stage05_v2_active_v3c_fused_contract_comparison(config)
    if comparison_variant == "stage05_v2_active_v3c_midpoint_reconstructed_contract_comparison":
        config = Stage05V2ActiveV3CMidpointReconstructedComparisonConfig(
            output_root=output_root,
            run_id=run_id,
            **overrides,
        )
        return run_stage05_v2_active_v3c_midpoint_reconstructed_contract_comparison(config)
    raise ValueError(f"Unsupported comparison_variant '{comparison_variant}'.")


def main() -> None:
    result = run()
    print("Stage 05 comparison completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Summary: {result.run_dir / 'aggregate_summary.json'}")
    print(f"Report: {result.run_dir / 'comparison_report.md'}")


if __name__ == "__main__":
    main()
