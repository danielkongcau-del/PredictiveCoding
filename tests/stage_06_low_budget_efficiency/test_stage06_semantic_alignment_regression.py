from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.stage_06_low_budget_efficiency.fmpc_stage06_objective_curriculum import (
    STAGE05_MATCHED_BUDGET_CONTROL_METHOD_NAME,
    Stage06LowBudgetComparisonConfig,
    _comparison_config_payload,
    _config_payload,
    build_stage06_v1_objective_curriculum_energydrop_default_config,
)


def test_stage06_config_locks_aggregate_two_branch_scaffold_semantics() -> None:
    config = build_stage06_v1_objective_curriculum_energydrop_default_config()
    payload = _config_payload(config)

    assert payload["transport"]["stage05_two_branch_parameterization_preserved"] is True
    assert payload["transport"]["stage05_target_builder_reuse_enabled"] is True
    assert payload["transport"]["stage05_branchwise_supervision_preserved"] is False
    assert payload["transport"]["stage06_supervision_contract_identity"] == (
        "aggregate_residual_supervision_over_stage05_targets"
    )
    assert payload["objective_contract"]["rollout_time_semantics_identity"] == (
        "remaining_horizon_forward_rollout"
    )


def test_stage06_comparison_payload_uses_v3c_matched_budget_control() -> None:
    payload = _comparison_config_payload(
        Stage06LowBudgetComparisonConfig(
            seeds=(0,),
            tier1_epochs=2,
            tier2_epochs=3,
            rescue_epochs=4,
            allow_rescue_tier3=False,
        )
    )

    assert STAGE05_MATCHED_BUDGET_CONTROL_METHOD_NAME == "stage05_v3c_stronger_semigroup_weight"
    assert payload["matched_budget_control"] == STAGE05_MATCHED_BUDGET_CONTROL_METHOD_NAME
