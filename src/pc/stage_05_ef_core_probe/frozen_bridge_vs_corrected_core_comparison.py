from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..datasets import load_digits_split
from ..metrics import classification_accuracy
from ..stage_03_transport_core_v1.fmpc_tf1_flow import (
    build_tf1_context,
    hidden_energy_from_state,
    hidden_local_flow,
    rollout_hidden_transport,
)
from ..stage_04_incremental_bridge.fmpc_tf2 import (
    FMPCTF2Config,
    build_tf2_corrective_transport_terminal_angleclip_default_config,
    run_fmpc_tf2_experiment,
)
from .fmpc_ef_exploratory_probe import (
    FMPCEFExploratoryProbeConfig,
    ProbeMechanismMetrics,
    build_stage05_v3b_stronger_traj_curr_weight_config,
    build_stage05_v3c_endpoint_semigroup_config,
    build_stage05_v3c_stronger_semigroup_weight_config,
    build_fmpc_ef_exploratory_probe_config,
    run_fmpc_ef_exploratory_probe,
)


ComparisonMethodName = Literal[
    "stage_04_frozen_bridge",
    "stage_05_corrected_residual_core",
    "stage_05_corrected_residual_core_v1",
    "stage_05_two_branch_corrected_residual_core_v2",
    "stage05_v3a_explicit_transport_drift_contract",
    "stage05_v3b_trajectory_curriculum_contract",
    "stage05_v3b_alpha_earlier_transition",
    "stage05_v3b_stronger_traj_curr_weight",
    "stage05_v3c_endpoint_semigroup_consistency_contract",
    "stage05_v3c_stronger_semigroup_weight",
    "stage_05_two_branch_corrected_residual_core_v2_current_budget",
    "stage_05_two_branch_corrected_residual_core_v2_longer_training",
    "stage_05_two_branch_corrected_residual_core_v2_budget_reference",
    "stage_05_two_branch_corrected_residual_core_v2_budget_push",
    "stage_05_two_branch_corrected_residual_core_v2_efficiency_reference",
    "stage_05_two_branch_corrected_residual_core_v2_efficiency_candidate",
]
OutputLayout = Literal["single_dir", "run_id_subdir"]
ComparisonScope = Literal["smoke_only", "fixed_budget_comparison", "contextual_comparison"]

STAGE04_METHOD_NAME: ComparisonMethodName = "stage_04_frozen_bridge"
STAGE05_METHOD_NAME: ComparisonMethodName = "stage_05_corrected_residual_core"
STAGE05_V1_METHOD_NAME: ComparisonMethodName = "stage_05_corrected_residual_core_v1"
STAGE05_V2_METHOD_NAME: ComparisonMethodName = "stage_05_two_branch_corrected_residual_core_v2"
STAGE05_V3A_METHOD_NAME: ComparisonMethodName = "stage05_v3a_explicit_transport_drift_contract"
STAGE05_V3B_METHOD_NAME: ComparisonMethodName = "stage05_v3b_trajectory_curriculum_contract"
STAGE05_V3B_ALPHA_EARLIER_METHOD_NAME: ComparisonMethodName = (
    "stage05_v3b_alpha_earlier_transition"
)
STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME: ComparisonMethodName = (
    "stage05_v3b_stronger_traj_curr_weight"
)
STAGE05_V3C_METHOD_NAME: ComparisonMethodName = (
    "stage05_v3c_endpoint_semigroup_consistency_contract"
)
STAGE05_V3C_STRONGER_SEMIGROUP_METHOD_NAME: ComparisonMethodName = (
    "stage05_v3c_stronger_semigroup_weight"
)
STAGE05_V2_CURRENT_BUDGET_METHOD_NAME: ComparisonMethodName = (
    "stage_05_two_branch_corrected_residual_core_v2_current_budget"
)
STAGE05_V2_LONGER_TRAINING_METHOD_NAME: ComparisonMethodName = (
    "stage_05_two_branch_corrected_residual_core_v2_longer_training"
)
STAGE05_V2_BUDGET_REFERENCE_METHOD_NAME: ComparisonMethodName = (
    "stage_05_two_branch_corrected_residual_core_v2_budget_reference"
)
STAGE05_V2_BUDGET_PUSH_METHOD_NAME: ComparisonMethodName = (
    "stage_05_two_branch_corrected_residual_core_v2_budget_push"
)
STAGE05_V2_EFFICIENCY_REFERENCE_METHOD_NAME: ComparisonMethodName = (
    "stage_05_two_branch_corrected_residual_core_v2_efficiency_reference"
)
STAGE05_V2_EFFICIENCY_CANDIDATE_METHOD_NAME: ComparisonMethodName = (
    "stage_05_two_branch_corrected_residual_core_v2_efficiency_candidate"
)
JUSTIFY_V2_DECISION_NAME = "stage05_corrected_residual_core_justifies_v2_charter"
STAGE05_V2_FAVORABLE_DECISION_NAME = "stage05_v2_improves_mechanism_magnitude_over_v1"
STAGE05_V3A_SIGNAL_DECISION_NAME = "stage05_v3a_shows_positive_gap_closure_signal_vs_v2"
STAGE05_V2_LONGER_TRAINING_DECISION_NAME = (
    "stage05_v2_longer_training_materially_improves_configured_step_mechanism"
)
STAGE05_V2_LONGER_TRAINING_ACCURACY_DECISION_NAME = (
    "stage05_v2_longer_training_materially_improves_report_only_accuracy"
)
STAGE05_V2_BUDGET_PUSH_DECISION_NAME = (
    "stage05_v2_budget_push_materially_improves_configured_step_mechanism"
)
STAGE05_V2_BUDGET_PUSH_ACCURACY_DECISION_NAME = (
    "stage05_v2_budget_push_materially_improves_report_only_accuracy"
)
STAGE05_V2_EFFICIENCY_DECISION_NAME = (
    "same_family_efficiency_change_materially_improves_configured_step_mechanism"
)
STAGE05_V2_EFFICIENCY_ACCURACY_DECISION_NAME = (
    "same_family_efficiency_change_materially_improves_report_only_accuracy"
)
STAGE05_V2_EFFICIENCY_GAP_DECISION_NAME = (
    "same_family_efficiency_change_materially_narrows_gap_to_3072_reference"
)
BOUNDARY_LIMITED_INTERPRETATION = "boundary_limited_mechanism_prototype"
STRUCTURALLY_INEFFICIENT_INTERPRETATION = "structurally_inefficient_same_family_line"


@dataclass
class FrozenBridgeVsCorrectedCoreComparisonConfig:
    """Compare the frozen Stage 04 bridge against the corrected Stage 05 core."""

    experiment_name: str = "frozen_bridge_vs_corrected_core_comparison"
    output_root: str | Path = "outputs/stage_05_ef_core_probe"
    run_id: str | None = None
    output_layout: OutputLayout = "single_dir"
    dataset_name: str = "digits"
    seeds: tuple[int, ...] = (0, 1, 2)
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    batch_size: int = 128
    shuffle_batches: bool = True
    stage04_epochs: int = 60
    stage04_eval_steps: int = 15
    stage04_layer_dims: tuple[int, ...] = (64, 64, 10)
    stage05_epochs: int = 12
    stage05_eval_steps: int = 15
    stage05_layer_dims: tuple[int, ...] = (64, 16, 10)

    def __post_init__(self) -> None:
        if self.dataset_name != "digits":
            raise ValueError("The frozen-bridge vs corrected-core comparison currently supports digits only.")
        if not self.seeds:
            raise ValueError("seeds must contain at least one seed.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.stage04_epochs <= 0 or self.stage05_epochs <= 0:
            raise ValueError("stage04_epochs and stage05_epochs must be positive.")
        if self.stage04_eval_steps <= 0 or self.stage05_eval_steps <= 0:
            raise ValueError("stage04_eval_steps and stage05_eval_steps must be positive.")

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_{self.seeds[0]}"


@dataclass
class FrozenBridgeVsCorrectedCoreComparisonRunResult:
    run_dir: Path
    config: dict[str, Any]
    aggregate_rows: list[dict[str, Any]]
    summary: dict[str, Any]
    comparison_report: dict[str, Any]


@dataclass
class FrozenBridgeVsStage05V2ComparisonConfig:
    """Compare the frozen Stage 04 bridge against the Stage 05 v2 two-branch core."""

    experiment_name: str = "frozen_bridge_vs_two_branch_corrected_core_comparison"
    output_root: str | Path = "outputs/stage_05_ef_core_probe"
    run_id: str | None = None
    output_layout: OutputLayout = "single_dir"
    dataset_name: str = "digits"
    seeds: tuple[int, ...] = (0, 1, 2)
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    batch_size: int = 128
    shuffle_batches: bool = True
    stage04_epochs: int = 60
    stage04_eval_steps: int = 15
    stage04_layer_dims: tuple[int, ...] = (64, 64, 10)
    stage05_epochs: int = 12
    stage05_eval_steps: int = 15
    stage05_layer_dims: tuple[int, ...] = (64, 16, 10)
    stage05_transport_steps: int = 2

    def __post_init__(self) -> None:
        if self.dataset_name != "digits":
            raise ValueError("The frozen-bridge vs Stage 05 v2 comparison currently supports digits only.")
        if not self.seeds:
            raise ValueError("seeds must contain at least one seed.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.stage04_epochs <= 0 or self.stage05_epochs <= 0:
            raise ValueError("stage04_epochs and stage05_epochs must be positive.")
        if self.stage04_eval_steps <= 0 or self.stage05_eval_steps <= 0:
            raise ValueError("stage04_eval_steps and stage05_eval_steps must be positive.")
        if self.stage05_transport_steps <= 0:
            raise ValueError("stage05_transport_steps must be positive.")

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_{self.seeds[0]}"


@dataclass
class CorrectedResidualCoreV1VsV2ComparisonConfig:
    """Compare the Stage 05 corrected residual core v1 against the Stage 05 v2 two-branch core."""

    experiment_name: str = "corrected_residual_core_v1_vs_v2_comparison"
    output_root: str | Path = "outputs/stage_05_ef_core_probe"
    run_id: str | None = None
    output_layout: OutputLayout = "single_dir"
    dataset_name: str = "digits"
    seeds: tuple[int, ...] = (0, 1, 2)
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    batch_size: int = 128
    shuffle_batches: bool = True
    stage05_epochs: int = 12
    stage05_eval_steps: int = 15
    stage05_layer_dims: tuple[int, ...] = (64, 16, 10)
    stage05_transport_steps: int = 2

    def __post_init__(self) -> None:
        if self.dataset_name != "digits":
            raise ValueError("The Stage 05 v1 vs v2 comparison currently supports digits only.")
        if not self.seeds:
            raise ValueError("seeds must contain at least one seed.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.stage05_epochs <= 0 or self.stage05_eval_steps <= 0:
            raise ValueError("stage05_epochs and stage05_eval_steps must be positive.")
        if self.stage05_transport_steps <= 0:
            raise ValueError("stage05_transport_steps must be positive.")

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_{self.seeds[0]}"


@dataclass
class Stage05V2VsV3AComparisonConfig:
    """Compare the current Stage 05 v2 reference against the minimal Stage 05 v3-A candidate."""

    experiment_name: str = "stage05_v2_vs_v3a_explicit_transport_drift_comparison"
    output_root: str | Path = "outputs/stage_05_ef_core_probe"
    run_id: str | None = None
    output_layout: OutputLayout = "single_dir"
    comparison_scope: ComparisonScope = "smoke_only"
    dataset_name: str = "digits"
    seeds: tuple[int, ...] = (0,)
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    batch_size: int = 128
    shuffle_batches: bool = True
    stage05_epochs: int = 4
    stage05_eval_steps: int = 8
    stage05_layer_dims: tuple[int, ...] = (64, 16, 10)
    stage05_transport_steps: int = 2
    lambda_drift: float = 1.0
    reuse_stage05_v2_reference_artifacts: bool = False
    reference_artifact_root: str | Path = (
        "outputs/stage_05_ef_core_probe/"
        "stage05_v2_budget_push_validation_1536_to_3072/"
        "runs/stage_05_two_branch_corrected_residual_core_v2_budget_reference"
    )
    contextual_reference_summary_path: str | Path = (
        "outputs/stage_05_ef_core_probe/"
        "stage05_v2_budget_push_validation_1536_to_3072/aggregate_summary.json"
    )
    contextual_reference_stage05_epochs: int = 3072
    configured_step_improvement_fraction_threshold: float = 0.05
    allowed_accuracy_regression_threshold: float = 0.01
    gap_narrowing_fraction_threshold: float = 0.1

    def __post_init__(self) -> None:
        if self.dataset_name != "digits":
            raise ValueError("The Stage 05 v2 vs v3-A comparison currently supports digits only.")
        if not self.seeds:
            raise ValueError("seeds must contain at least one seed.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.stage05_epochs <= 0 or self.stage05_eval_steps <= 0:
            raise ValueError("stage05_epochs and stage05_eval_steps must be positive.")
        if self.stage05_transport_steps <= 0:
            raise ValueError("stage05_transport_steps must be positive.")
        if self.lambda_drift < 0.0:
            raise ValueError("lambda_drift must be non-negative.")
        if self.contextual_reference_stage05_epochs <= 0:
            raise ValueError("contextual_reference_stage05_epochs must be positive.")
        if self.configured_step_improvement_fraction_threshold < 0.0:
            raise ValueError("configured_step_improvement_fraction_threshold must be non-negative.")
        if self.allowed_accuracy_regression_threshold < 0.0:
            raise ValueError("allowed_accuracy_regression_threshold must be non-negative.")
        if self.gap_narrowing_fraction_threshold < 0.0:
            raise ValueError("gap_narrowing_fraction_threshold must be non-negative.")

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_{self.seeds[0]}"


@dataclass
class Stage05V2V3AV3BComparisonConfig:
    """Stage 05 comparison across the current v2 control, v3-A reference, and v3-B candidate."""

    experiment_name: str = "stage05_v2_v3a_v3b_trajectory_curriculum_comparison"
    output_root: str | Path = "outputs/stage_05_ef_core_probe"
    run_id: str | None = None
    output_layout: OutputLayout = "single_dir"
    comparison_scope: ComparisonScope = "smoke_only"
    dataset_name: str = "digits"
    seeds: tuple[int, ...] = (0,)
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    batch_size: int = 128
    shuffle_batches: bool = True
    stage05_epochs: int = 4
    stage05_eval_steps: int = 8
    stage05_layer_dims: tuple[int, ...] = (64, 16, 10)
    stage05_transport_steps: int = 2
    lambda_drift: float = 1.0
    lambda_traj_curr: float = 0.1
    alpha_floor: float = 0.5
    alpha_warmup_epochs: int = 1
    alpha_ramp_epochs: int = 2
    reuse_stage05_v2_reference_artifacts: bool = False
    reference_artifact_root: str | Path = (
        "outputs/stage_05_ef_core_probe/"
        "stage05_v2_budget_push_validation_1536_to_3072/"
        "runs/stage_05_two_branch_corrected_residual_core_v2_budget_reference"
    )
    reuse_stage05_v3a_reference_artifacts: bool = False
    v3a_reference_artifact_root: str | Path = (
        "outputs/stage_05_ef_core_probe/"
        "stage05_v2_vs_v3a_explicit_transport_drift_fixed_budget_comparison/"
        "runs/stage05_v3a_explicit_transport_drift_contract"
    )
    reuse_stage05_v3b_candidate_artifacts: bool = False
    v3b_candidate_artifact_root: str | Path = (
        "outputs/stage_05_ef_core_probe/"
        "stage05_v2_v3a_v3b_fixed_budget_comparison/"
        "runs/stage05_v3b_trajectory_curriculum_contract"
    )
    v3b_candidate_method_name: ComparisonMethodName = STAGE05_V3B_METHOD_NAME
    contextual_reference_summary_path: str | Path = (
        "outputs/stage_05_ef_core_probe/"
        "stage05_v2_budget_push_validation_1536_to_3072/aggregate_summary.json"
    )
    contextual_reference_stage05_epochs: int = 3072
    configured_step_improvement_fraction_threshold: float = 0.05
    allowed_accuracy_regression_threshold: float = 0.01
    gap_narrowing_fraction_threshold: float = 0.1
    gap_closure_gain_threshold: float = 0.05

    def __post_init__(self) -> None:
        if self.dataset_name != "digits":
            raise ValueError("The Stage 05 v2/v3-A/v3-B comparison currently supports digits only.")
        if not self.seeds:
            raise ValueError("seeds must contain at least one seed.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.stage05_epochs <= 0 or self.stage05_eval_steps <= 0:
            raise ValueError("stage05_epochs and stage05_eval_steps must be positive.")
        if self.stage05_transport_steps <= 0:
            raise ValueError("stage05_transport_steps must be positive.")
        if self.lambda_drift < 0.0:
            raise ValueError("lambda_drift must be non-negative.")
        if self.lambda_traj_curr < 0.0:
            raise ValueError("lambda_traj_curr must be non-negative.")
        if not (0.0 < self.alpha_floor < 1.0):
            raise ValueError("alpha_floor must satisfy 0 < alpha_floor < 1.")
        if self.alpha_warmup_epochs < 0 or self.alpha_ramp_epochs < 0:
            raise ValueError("alpha_warmup_epochs and alpha_ramp_epochs must be non-negative.")
        if self.contextual_reference_stage05_epochs <= 0:
            raise ValueError("contextual_reference_stage05_epochs must be positive.")
        if self.configured_step_improvement_fraction_threshold < 0.0:
            raise ValueError("configured_step_improvement_fraction_threshold must be non-negative.")
        if self.allowed_accuracy_regression_threshold < 0.0:
            raise ValueError("allowed_accuracy_regression_threshold must be non-negative.")
        if self.gap_narrowing_fraction_threshold < 0.0:
            raise ValueError("gap_narrowing_fraction_threshold must be non-negative.")
        if self.gap_closure_gain_threshold < 0.0:
            raise ValueError("gap_closure_gain_threshold must be non-negative.")

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_{self.seeds[0]}"


@dataclass
class Stage05V3BRefinementDiagnosticConfig:
    """Run a narrow fixed-budget attribution diagnostic around the current Stage 05 v3-B scaffold."""

    experiment_name: str = "stage05_v3b_refinement_diagnostic"
    output_root: str | Path = "outputs/stage_05_ef_core_probe"
    run_id: str | None = None
    output_layout: OutputLayout = "single_dir"
    comparison_scope: ComparisonScope = "smoke_only"
    dataset_name: str = "digits"
    seeds: tuple[int, ...] = (0,)
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    batch_size: int = 128
    shuffle_batches: bool = True
    stage05_epochs: int = 4
    stage05_eval_steps: int = 8
    stage05_layer_dims: tuple[int, ...] = (64, 16, 10)
    stage05_transport_steps: int = 2
    lambda_drift: float = 1.0
    reuse_stage05_v2_reference_artifacts: bool = False
    reference_artifact_root: str | Path = (
        "outputs/stage_05_ef_core_probe/"
        "stage05_v2_budget_push_validation_1536_to_3072/"
        "runs/stage_05_two_branch_corrected_residual_core_v2_budget_reference"
    )
    reuse_stage05_v3a_reference_artifacts: bool = False
    v3a_reference_artifact_root: str | Path = (
        "outputs/stage_05_ef_core_probe/"
        "stage05_v2_vs_v3a_explicit_transport_drift_fixed_budget_comparison/"
        "runs/stage05_v3a_explicit_transport_drift_contract"
    )
    reuse_stage05_v3b_control_artifacts: bool = False
    v3b_control_artifact_root: str | Path = (
        "outputs/stage_05_ef_core_probe/"
        "stage05_v2_v3a_v3b_fixed_budget_comparison/"
        "runs/stage05_v3b_trajectory_curriculum_contract"
    )
    control_lambda_traj_curr: float = 0.1
    control_alpha_floor: float = 0.5
    control_alpha_warmup_epochs: int = 3
    control_alpha_ramp_epochs: int = 3
    alpha_earlier_transition_alpha_floor: float = 0.25
    alpha_earlier_transition_alpha_warmup_epochs: int = 1
    alpha_earlier_transition_alpha_ramp_epochs: int = 1
    stronger_traj_curr_lambda_traj_curr: float = 0.2
    contextual_reference_summary_path: str | Path = (
        "outputs/stage_05_ef_core_probe/"
        "stage05_v2_budget_push_validation_1536_to_3072/aggregate_summary.json"
    )
    contextual_reference_stage05_epochs: int = 3072
    configured_step_improvement_fraction_threshold: float = 0.05
    allowed_accuracy_regression_threshold: float = 0.01

    def __post_init__(self) -> None:
        if self.dataset_name != "digits":
            raise ValueError("The Stage 05 v3-B refinement diagnostic currently supports digits only.")
        if not self.seeds:
            raise ValueError("seeds must contain at least one seed.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.stage05_epochs <= 0 or self.stage05_eval_steps <= 0:
            raise ValueError("stage05_epochs and stage05_eval_steps must be positive.")
        if self.stage05_transport_steps <= 0:
            raise ValueError("stage05_transport_steps must be positive.")
        if self.lambda_drift < 0.0:
            raise ValueError("lambda_drift must be non-negative.")
        if self.control_lambda_traj_curr < 0.0 or self.stronger_traj_curr_lambda_traj_curr < 0.0:
            raise ValueError("lambda_traj_curr values must be non-negative.")
        for value in (
            self.control_alpha_floor,
            self.alpha_earlier_transition_alpha_floor,
        ):
            if not (0.0 < value < 1.0):
                raise ValueError("alpha_floor values must satisfy 0 < alpha_floor < 1.")
        for value in (
            self.control_alpha_warmup_epochs,
            self.control_alpha_ramp_epochs,
            self.alpha_earlier_transition_alpha_warmup_epochs,
            self.alpha_earlier_transition_alpha_ramp_epochs,
        ):
            if value < 0:
                raise ValueError("alpha warmup/ramp epochs must be non-negative.")
        if self.contextual_reference_stage05_epochs <= 0:
            raise ValueError("contextual_reference_stage05_epochs must be positive.")
        if self.configured_step_improvement_fraction_threshold < 0.0:
            raise ValueError("configured_step_improvement_fraction_threshold must be non-negative.")
        if self.allowed_accuracy_regression_threshold < 0.0:
            raise ValueError("allowed_accuracy_regression_threshold must be non-negative.")

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_{self.seeds[0]}"


@dataclass
class Stage05V2PromotedV3BV3CComparisonConfig:
    """Smoke-ready comparison across the fixed-budget v2 control, promoted refined v3-B, and v3-C."""

    experiment_name: str = "stage05_v2_promoted_v3b_v3c_comparison"
    output_root: str | Path = "outputs/stage_05_ef_core_probe"
    run_id: str | None = None
    output_layout: OutputLayout = "single_dir"
    comparison_scope: ComparisonScope = "smoke_only"
    dataset_name: str = "digits"
    seeds: tuple[int, ...] = (0,)
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    batch_size: int = 128
    shuffle_batches: bool = True
    stage05_epochs: int = 4
    stage05_eval_steps: int = 8
    stage05_layer_dims: tuple[int, ...] = (64, 16, 10)
    stage05_transport_steps: int = 2
    lambda_drift: float = 1.0
    promoted_v3b_lambda_traj_curr: float = 0.2
    promoted_v3b_alpha_floor: float = 0.5
    promoted_v3b_alpha_warmup_epochs: int = 3
    promoted_v3b_alpha_ramp_epochs: int = 3
    lambda_sg: float = 0.05
    reuse_stage05_v2_reference_artifacts: bool = False
    reference_artifact_root: str | Path = (
        "outputs/stage_05_ef_core_probe/"
        "stage05_v2_budget_push_validation_1536_to_3072/"
        "runs/stage_05_two_branch_corrected_residual_core_v2_budget_reference"
    )
    reuse_stage05_v3b_reference_artifacts: bool = False
    v3b_reference_artifact_root: str | Path = (
        "outputs/stage_05_ef_core_probe/"
        "stage05_v3b_refinement_diagnostic/"
        "runs/stage05_v3b_stronger_traj_curr_weight"
    )
    reuse_stage05_v3c_candidate_artifacts: bool = False
    v3c_candidate_method_name: ComparisonMethodName = STAGE05_V3C_METHOD_NAME
    v3c_candidate_artifact_root: str | Path = (
        "outputs/stage_05_ef_core_probe/"
        "stage05_v2_promoted_v3b_v3c_comparison/"
        "runs/stage05_v3c_endpoint_semigroup_consistency_contract"
    )
    contextual_reference_summary_path: str | Path = (
        "outputs/stage_05_ef_core_probe/"
        "stage05_v2_budget_push_validation_1536_to_3072/aggregate_summary.json"
    )
    contextual_reference_stage05_epochs: int = 3072
    configured_step_improvement_fraction_threshold: float = 0.05
    allowed_accuracy_regression_threshold: float = 0.01
    gap_narrowing_fraction_threshold: float = 0.1

    def __post_init__(self) -> None:
        if self.dataset_name != "digits":
            raise ValueError(
                "The Stage 05 v2/promoted-v3-B/v3-C comparison currently supports digits only."
            )
        if not self.seeds:
            raise ValueError("seeds must contain at least one seed.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.stage05_epochs <= 0 or self.stage05_eval_steps <= 0:
            raise ValueError("stage05_epochs and stage05_eval_steps must be positive.")
        if self.stage05_transport_steps <= 0:
            raise ValueError("stage05_transport_steps must be positive.")
        if self.lambda_drift < 0.0:
            raise ValueError("lambda_drift must be non-negative.")
        if self.promoted_v3b_lambda_traj_curr < 0.0:
            raise ValueError("promoted_v3b_lambda_traj_curr must be non-negative.")
        if not (0.0 < self.promoted_v3b_alpha_floor < 1.0):
            raise ValueError("promoted_v3b_alpha_floor must satisfy 0 < alpha_floor < 1.")
        if self.promoted_v3b_alpha_warmup_epochs < 0 or self.promoted_v3b_alpha_ramp_epochs < 0:
            raise ValueError("promoted_v3b alpha warmup/ramp epochs must be non-negative.")
        if self.lambda_sg < 0.0:
            raise ValueError("lambda_sg must be non-negative.")
        if self.v3c_candidate_method_name not in {
            STAGE05_V3C_METHOD_NAME,
            STAGE05_V3C_STRONGER_SEMIGROUP_METHOD_NAME,
        }:
            raise ValueError(
                "v3c_candidate_method_name must be one of "
                "{'stage05_v3c_endpoint_semigroup_consistency_contract', "
                "'stage05_v3c_stronger_semigroup_weight'}."
            )
        if self.contextual_reference_stage05_epochs <= 0:
            raise ValueError("contextual_reference_stage05_epochs must be positive.")
        if self.configured_step_improvement_fraction_threshold < 0.0:
            raise ValueError("configured_step_improvement_fraction_threshold must be non-negative.")
        if self.allowed_accuracy_regression_threshold < 0.0:
            raise ValueError("allowed_accuracy_regression_threshold must be non-negative.")
        if self.gap_narrowing_fraction_threshold < 0.0:
            raise ValueError("gap_narrowing_fraction_threshold must be non-negative.")

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_{self.seeds[0]}"


@dataclass
class Stage05V3CRefinementDiagnosticConfig:
    """Run a narrow fixed-budget attribution diagnostic around the current Stage 05 v3-C probe."""

    experiment_name: str = "stage05_v3c_refinement_diagnostic"
    output_root: str | Path = "outputs/stage_05_ef_core_probe"
    run_id: str | None = None
    output_layout: OutputLayout = "single_dir"
    comparison_scope: ComparisonScope = "smoke_only"
    dataset_name: str = "digits"
    seeds: tuple[int, ...] = (0,)
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    batch_size: int = 128
    shuffle_batches: bool = True
    stage05_epochs: int = 4
    stage05_eval_steps: int = 8
    stage05_layer_dims: tuple[int, ...] = (64, 16, 10)
    stage05_transport_steps: int = 2
    lambda_drift: float = 1.0
    promoted_v3b_lambda_traj_curr: float = 0.2
    promoted_v3b_alpha_floor: float = 0.5
    promoted_v3b_alpha_warmup_epochs: int = 3
    promoted_v3b_alpha_ramp_epochs: int = 3
    control_lambda_sg: float = 0.05
    stronger_semigroup_lambda_sg: float = 0.10
    reuse_stage05_v2_reference_artifacts: bool = False
    reference_artifact_root: str | Path = (
        "outputs/stage_05_ef_core_probe/"
        "stage05_v2_budget_push_validation_1536_to_3072/"
        "runs/stage_05_two_branch_corrected_residual_core_v2_budget_reference"
    )
    reuse_stage05_v3b_reference_artifacts: bool = False
    v3b_reference_artifact_root: str | Path = (
        "outputs/stage_05_ef_core_probe/"
        "stage05_v3b_refinement_diagnostic/"
        "runs/stage05_v3b_stronger_traj_curr_weight"
    )
    reuse_stage05_v3c_control_artifacts: bool = False
    v3c_control_artifact_root: str | Path = (
        "outputs/stage_05_ef_core_probe/"
        "stage05_v2_promoted_v3b_v3c_fixed_budget_comparison/"
        "runs/stage05_v3c_endpoint_semigroup_consistency_contract"
    )
    contextual_reference_summary_path: str | Path = (
        "outputs/stage_05_ef_core_probe/"
        "stage05_v2_budget_push_validation_1536_to_3072/aggregate_summary.json"
    )
    contextual_reference_stage05_epochs: int = 3072
    configured_step_improvement_fraction_threshold: float = 0.05
    allowed_accuracy_regression_threshold: float = 0.01

    def __post_init__(self) -> None:
        if self.dataset_name != "digits":
            raise ValueError("The Stage 05 v3-C refinement diagnostic currently supports digits only.")
        if not self.seeds:
            raise ValueError("seeds must contain at least one seed.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.stage05_epochs <= 0 or self.stage05_eval_steps <= 0:
            raise ValueError("stage05_epochs and stage05_eval_steps must be positive.")
        if self.stage05_transport_steps <= 0:
            raise ValueError("stage05_transport_steps must be positive.")
        if self.lambda_drift < 0.0:
            raise ValueError("lambda_drift must be non-negative.")
        if self.promoted_v3b_lambda_traj_curr < 0.0:
            raise ValueError("promoted_v3b_lambda_traj_curr must be non-negative.")
        if not (0.0 < self.promoted_v3b_alpha_floor < 1.0):
            raise ValueError("promoted_v3b_alpha_floor must satisfy 0 < alpha_floor < 1.")
        if self.promoted_v3b_alpha_warmup_epochs < 0 or self.promoted_v3b_alpha_ramp_epochs < 0:
            raise ValueError("promoted_v3b alpha warmup/ramp epochs must be non-negative.")
        if self.control_lambda_sg < 0.0:
            raise ValueError("control_lambda_sg must be non-negative.")
        if self.stronger_semigroup_lambda_sg < 0.0:
            raise ValueError("stronger_semigroup_lambda_sg must be non-negative.")
        if self.contextual_reference_stage05_epochs <= 0:
            raise ValueError("contextual_reference_stage05_epochs must be positive.")
        if self.configured_step_improvement_fraction_threshold < 0.0:
            raise ValueError("configured_step_improvement_fraction_threshold must be non-negative.")
        if self.allowed_accuracy_regression_threshold < 0.0:
            raise ValueError("allowed_accuracy_regression_threshold must be non-negative.")

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_{self.seeds[0]}"


@dataclass
class Stage05V2LongerTrainingValidationConfig:
    """Compare the current Stage 05 v2 budget against a longer-training budget."""

    experiment_name: str = "stage05_v2_longer_training_validation"
    output_root: str | Path = "outputs/stage_05_ef_core_probe"
    run_id: str | None = None
    output_layout: OutputLayout = "single_dir"
    dataset_name: str = "digits"
    seeds: tuple[int, ...] = (0, 1, 2)
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    batch_size: int = 128
    shuffle_batches: bool = True
    current_stage05_epochs: int = 12
    longer_stage05_epochs: int = 24
    stage05_eval_steps: int = 15
    stage05_layer_dims: tuple[int, ...] = (64, 16, 10)
    stage05_transport_steps: int = 2
    configured_step_improvement_fraction_threshold: float = 0.05
    report_accuracy_improvement_threshold: float = 0.01

    def __post_init__(self) -> None:
        if self.dataset_name != "digits":
            raise ValueError("The Stage 05 v2 longer-training validation currently supports digits only.")
        if not self.seeds:
            raise ValueError("seeds must contain at least one seed.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.current_stage05_epochs <= 0 or self.longer_stage05_epochs <= 0:
            raise ValueError("current_stage05_epochs and longer_stage05_epochs must be positive.")
        if self.longer_stage05_epochs <= self.current_stage05_epochs:
            raise ValueError("longer_stage05_epochs must be greater than current_stage05_epochs.")
        if self.stage05_eval_steps <= 0:
            raise ValueError("stage05_eval_steps must be positive.")
        if self.stage05_transport_steps <= 0:
            raise ValueError("stage05_transport_steps must be positive.")
        if self.configured_step_improvement_fraction_threshold < 0.0:
            raise ValueError("configured_step_improvement_fraction_threshold must be non-negative.")
        if self.report_accuracy_improvement_threshold < 0.0:
            raise ValueError("report_accuracy_improvement_threshold must be non-negative.")

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_{self.seeds[0]}"


@dataclass
class Stage05V2BudgetPushValidationConfig:
    """Compare the current Stage 05 v2 reference budget against a stronger same-family budget push."""

    experiment_name: str = "stage05_v2_budget_push_validation"
    output_root: str | Path = "outputs/stage_05_ef_core_probe"
    run_id: str | None = None
    output_layout: OutputLayout = "single_dir"
    dataset_name: str = "digits"
    seeds: tuple[int, ...] = (0, 1, 2)
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    batch_size: int = 128
    shuffle_batches: bool = True
    reference_stage05_epochs: int = 24
    stronger_stage05_epochs: int = 48
    stage05_eval_steps: int = 15
    stage05_layer_dims: tuple[int, ...] = (64, 16, 10)
    stage05_transport_steps: int = 2
    configured_step_improvement_fraction_threshold: float = 0.05
    report_accuracy_improvement_threshold: float = 0.01

    def __post_init__(self) -> None:
        if self.dataset_name != "digits":
            raise ValueError("The Stage 05 v2 budget-push validation currently supports digits only.")
        if not self.seeds:
            raise ValueError("seeds must contain at least one seed.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.reference_stage05_epochs <= 0 or self.stronger_stage05_epochs <= 0:
            raise ValueError("reference_stage05_epochs and stronger_stage05_epochs must be positive.")
        if self.stronger_stage05_epochs <= self.reference_stage05_epochs:
            raise ValueError("stronger_stage05_epochs must be greater than reference_stage05_epochs.")
        if self.stage05_eval_steps <= 0:
            raise ValueError("stage05_eval_steps must be positive.")
        if self.stage05_transport_steps <= 0:
            raise ValueError("stage05_transport_steps must be positive.")
        if self.configured_step_improvement_fraction_threshold < 0.0:
            raise ValueError("configured_step_improvement_fraction_threshold must be non-negative.")
        if self.report_accuracy_improvement_threshold < 0.0:
            raise ValueError("report_accuracy_improvement_threshold must be non-negative.")

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_{self.seeds[0]}"


@dataclass
class Stage05V2EfficiencyDiagnosticConfig:
    """Diagnose whether a narrow same-family schedule change improves 1536-epoch efficiency."""

    experiment_name: str = "stage05_v2_efficiency_diagnostic_at_1536"
    output_root: str | Path = "outputs/stage_05_ef_core_probe"
    run_id: str | None = None
    output_layout: OutputLayout = "single_dir"
    dataset_name: str = "digits"
    seeds: tuple[int, ...] = (0, 1, 2)
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    batch_size: int = 128
    shuffle_batches: bool = True
    reference_stage05_epochs: int = 1536
    contextual_reference_stage05_epochs: int = 3072
    stage05_eval_steps: int = 15
    stage05_layer_dims: tuple[int, ...] = (64, 16, 10)
    stage05_transport_steps: int = 2
    optimized_lambda_id_warmup_epochs: int = 1
    optimized_lambda_id_ramp_epochs: int = 1
    reference_artifact_root: str | Path = (
        "outputs/stage_05_ef_core_probe/"
        "stage05_v2_budget_push_validation_1536_to_3072/"
        "runs/stage_05_two_branch_corrected_residual_core_v2_budget_reference"
    )
    contextual_reference_summary_path: str | Path = (
        "outputs/stage_05_ef_core_probe/"
        "stage05_v2_budget_push_validation_1536_to_3072/aggregate_summary.json"
    )
    configured_step_improvement_fraction_threshold: float = 0.05
    report_accuracy_improvement_threshold: float = 0.01
    gap_narrowing_fraction_threshold: float = 0.2

    def __post_init__(self) -> None:
        if self.dataset_name != "digits":
            raise ValueError("The Stage 05 v2 efficiency diagnostic currently supports digits only.")
        if not self.seeds:
            raise ValueError("seeds must contain at least one seed.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.reference_stage05_epochs <= 0 or self.contextual_reference_stage05_epochs <= 0:
            raise ValueError(
                "reference_stage05_epochs and contextual_reference_stage05_epochs must be positive."
            )
        if self.contextual_reference_stage05_epochs <= self.reference_stage05_epochs:
            raise ValueError(
                "contextual_reference_stage05_epochs must be greater than reference_stage05_epochs."
            )
        if self.stage05_eval_steps <= 0:
            raise ValueError("stage05_eval_steps must be positive.")
        if self.stage05_transport_steps <= 0:
            raise ValueError("stage05_transport_steps must be positive.")
        if self.optimized_lambda_id_warmup_epochs < 0:
            raise ValueError("optimized_lambda_id_warmup_epochs must be non-negative.")
        if self.optimized_lambda_id_ramp_epochs < 0:
            raise ValueError("optimized_lambda_id_ramp_epochs must be non-negative.")
        if self.configured_step_improvement_fraction_threshold < 0.0:
            raise ValueError("configured_step_improvement_fraction_threshold must be non-negative.")
        if self.report_accuracy_improvement_threshold < 0.0:
            raise ValueError("report_accuracy_improvement_threshold must be non-negative.")
        if self.gap_narrowing_fraction_threshold < 0.0:
            raise ValueError("gap_narrowing_fraction_threshold must be non-negative.")

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_{self.seeds[0]}"


def _resolve_run_dir(
    output_root: str | Path,
    experiment_name: str,
    run_id: str,
    output_layout: OutputLayout,
) -> Path:
    if output_layout == "single_dir":
        return Path(output_root) / experiment_name
    if output_layout == "run_id_subdir":
        return Path(output_root) / experiment_name / run_id
    raise ValueError(f"Unsupported output_layout '{output_layout}'.")


def _prepare_run_dir(run_dir: Path) -> Path:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, indent=2)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at '{path}'.")
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("rows must contain at least one entry.")
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _relative_posix(base_dir: Path, target: Path) -> str:
    return target.relative_to(base_dir).as_posix()


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Mean requires at least one value.")
    return float(sum(values) / float(len(values)))


def _std(values: list[float]) -> float:
    if not values:
        raise ValueError("Std requires at least one value.")
    mean_value = _mean(values)
    variance = sum((float(value) - mean_value) ** 2 for value in values) / float(len(values))
    return float(variance**0.5)


def _rate(values: list[bool]) -> float:
    if not values:
        raise ValueError("Rate requires at least one value.")
    return float(sum(1.0 for value in values if bool(value)) / float(len(values)))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return _repo_root() / path


def _repo_relative_posix(target: Path) -> str:
    try:
        return target.resolve().relative_to(_repo_root().resolve()).as_posix()
    except ValueError:
        return target.as_posix()


def _load_results_md_accuracy_snapshot(name: str) -> dict[str, float | str]:
    results_path = _repo_root() / "RESULTS.md"
    text = results_path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"- `{re.escape(name)}`:\s+"
        rf"- `best_epoch = (?P<best_epoch>[^`]+)`\s+"
        rf"- `val_accuracy = (?P<val_accuracy>[^`]+)`\s+"
        rf"- `test_accuracy = (?P<test_accuracy>[^`]+)`",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"Could not find '{name}' accuracy snapshot in RESULTS.md.")
    return {
        "source": "RESULTS.md",
        "best_epoch": str(match.group("best_epoch")).strip(),
        "val_accuracy": float(match.group("val_accuracy")),
        "test_accuracy": float(match.group("test_accuracy")),
    }


def _load_budget_push_contextual_accuracy_snapshot(
    *,
    stronger_summary: dict[str, Any],
) -> dict[str, Any]:
    repo_root = _repo_root()
    frozen_bridge_comparison_path = (
        repo_root
        / "outputs"
        / "stage_05_ef_core_probe"
        / "frozen_bridge_vs_two_branch_corrected_core_comparison"
        / "aggregate_summary.json"
    )
    if frozen_bridge_comparison_path.exists():
        frozen_bridge_summary = _read_json(frozen_bridge_comparison_path)
        stage04_reference = {
            "source": str(
                Path("outputs")
                / "stage_05_ef_core_probe"
                / "frozen_bridge_vs_two_branch_corrected_core_comparison"
                / "aggregate_summary.json"
            ),
            "val_accuracy": float(
                frozen_bridge_summary["by_method"][STAGE04_METHOD_NAME]["val_accuracy"]["mean"]
            ),
            "test_accuracy": float(
                frozen_bridge_summary["by_method"][STAGE04_METHOD_NAME]["test_accuracy"]["mean"]
            ),
        }
    else:
        stage04_summary = _read_json(
            repo_root / "outputs" / "stage_04_incremental_bridge" / "fmpc_tf2" / "summary.json"
        )
        stage04_reference = {
            "source": "outputs/stage_04_incremental_bridge/fmpc_tf2/summary.json",
            "val_accuracy": float(stage04_summary["val_accuracy"]),
            "test_accuracy": float(stage04_summary["test_accuracy"]),
        }
    digits_pc = _load_results_md_accuracy_snapshot("digits_pc")
    digits_mlp = _load_results_md_accuracy_snapshot("digits_mlp")
    stronger_val_accuracy = float(stronger_summary["val_accuracy"]["mean"])
    stronger_test_accuracy = float(stronger_summary["test_accuracy"]["mean"])
    stronger_pair = (stronger_val_accuracy, stronger_test_accuracy)

    def _relation_phrase(reference: dict[str, float | str], label: str) -> str:
        ref_val = float(reference["val_accuracy"])
        ref_test = float(reference["test_accuracy"])
        if stronger_pair[0] >= ref_val and stronger_pair[1] >= ref_test:
            return f"above the {label} accuracy level"
        if stronger_pair[0] <= ref_val and stronger_pair[1] <= ref_test:
            return f"below the {label} accuracy level"
        return f"mixed relative to the {label} accuracy level"

    note = (
        "Diagnostic-only accuracy context: the stronger Stage 05 v2 budget is "
        f"{_relation_phrase(stage04_reference, 'frozen Stage 04 bridge')}, "
        f"{_relation_phrase(digits_pc, 'standalone digits_pc baseline')}, and "
        f"{_relation_phrase(digits_mlp, 'standalone digits_mlp baseline')}; this comparison informs "
        "budgeting only and does not change the Stage 05 mechanism-first gate."
    )
    return {
        "stage05_v2_stronger_budget": {
            "source": "aggregate_summary.by_method",
            "val_accuracy_mean": float(stronger_val_accuracy),
            "test_accuracy_mean": float(stronger_test_accuracy),
        },
        "frozen_stage04_bridge": stage04_reference,
        "digits_pc": digits_pc,
        "digits_mlp": digits_mlp,
        "note": note,
    }


def _load_stage05_v2_contextual_reference(
    config: (
        Stage05V2EfficiencyDiagnosticConfig
        | Stage05V3BRefinementDiagnosticConfig
        | Stage05V2PromotedV3BV3CComparisonConfig
        | Stage05V3CRefinementDiagnosticConfig
    ),
) -> dict[str, Any]:
    summary_path = _resolve_repo_path(config.contextual_reference_summary_path)
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing contextual Stage 05 reference summary at '{summary_path}'."
        )
    payload = _read_json(summary_path)
    by_method = payload.get("by_method")
    if not isinstance(by_method, dict):
        raise ValueError("Contextual Stage 05 reference summary is missing 'by_method'.")
    contextual_summary = by_method.get(STAGE05_V2_BUDGET_PUSH_METHOD_NAME)
    if not isinstance(contextual_summary, dict):
        raise ValueError(
            "Contextual Stage 05 reference summary is missing the stronger-budget Stage 05 v2 method."
        )
    return {
        "source": _repo_relative_posix(summary_path),
        "method_name": STAGE05_V2_BUDGET_PUSH_METHOD_NAME,
        "epochs": int(config.contextual_reference_stage05_epochs),
        "configured_transport_steps": int(
            contextual_summary.get("configured_transport_steps", config.stage05_transport_steps)
        ),
        "one_step_energy_delta_vs_identity": contextual_summary["one_step_energy_delta_vs_identity"],
        "configured_step_energy_delta_vs_identity": contextual_summary[
            "configured_step_energy_delta_vs_identity"
        ],
        "configured_step_fixed_point_residual_delta_vs_identity": contextual_summary[
            "configured_step_fixed_point_residual_delta_vs_identity"
        ],
        "one_step_energy_delta_vs_local_field_only": contextual_summary[
            "one_step_energy_delta_vs_local_field_only"
        ],
        "configured_step_energy_delta_vs_local_field_only": contextual_summary[
            "configured_step_energy_delta_vs_local_field_only"
        ],
        "configured_step_fixed_point_residual_delta_vs_local_field_only": contextual_summary[
            "configured_step_fixed_point_residual_delta_vs_local_field_only"
        ],
        "val_accuracy": contextual_summary["val_accuracy"],
        "test_accuracy": contextual_summary["test_accuracy"],
        "val_output_mse": contextual_summary["val_output_mse"],
        "test_output_mse": contextual_summary["test_output_mse"],
        "selected_epoch": contextual_summary["selected_epoch"],
        "selection_hits_final_training_boundary_rate": float(
            contextual_summary.get("selection_hits_final_training_boundary_rate", 0.0)
        ),
        "runtime_proxy_seconds": contextual_summary["runtime_proxy_seconds"],
    }


def _hidden_residual_rms(context: Any, z: np.ndarray) -> float:
    flow = hidden_local_flow(context, z)
    return float(np.sqrt(np.mean(flow * flow)))


def _stage04_mechanism_metrics(
    model: Any,
    psi_network: Any,
    config: FMPCTF2Config,
    x_split: np.ndarray,
    y_split: np.ndarray,
    *,
    transport_steps: int,
) -> ProbeMechanismMetrics:
    from ..stage_04_incremental_bridge.fmpc_tf2 import _learned_velocity_fn as _tf2_learned_velocity_fn

    context = build_tf1_context(model, x_split, y_split)
    initial_energy = hidden_energy_from_state(context, context.z0)
    initial_residual_rms = _hidden_residual_rms(context, context.z0)
    identity = rollout_hidden_transport(
        context,
        context.z0,
        transport_steps=transport_steps,
        mode="identity",
    )
    local_field = rollout_hidden_transport(
        context,
        context.z0,
        transport_steps=transport_steps,
        mode="local_field_only",
    )
    learned = rollout_hidden_transport(
        context,
        context.z0,
        transport_steps=transport_steps,
        mode="learned",
        velocity_fn=_tf2_learned_velocity_fn(context, psi_network, config),
    )
    identity_residual = _hidden_residual_rms(context, identity.z_knots[-1])
    local_field_residual = _hidden_residual_rms(context, local_field.z_knots[-1])
    learned_residual = _hidden_residual_rms(context, learned.z_knots[-1])
    return ProbeMechanismMetrics(
        transport_steps=int(transport_steps),
        initial_energy=float(initial_energy),
        identity_final_energy=float(identity.final_energy),
        local_field_only_final_energy=float(local_field.final_energy),
        transported_final_energy=float(learned.final_energy),
        energy_delta_vs_identity=float(learned.final_energy - identity.final_energy),
        energy_delta_vs_local_field_only=float(learned.final_energy - local_field.final_energy),
        initial_fixed_point_residual_rms=float(initial_residual_rms),
        identity_final_fixed_point_residual_rms=float(identity_residual),
        local_field_only_final_fixed_point_residual_rms=float(local_field_residual),
        transported_final_fixed_point_residual_rms=float(learned_residual),
        fixed_point_residual_delta_vs_identity=float(learned_residual - identity_residual),
        fixed_point_residual_delta_vs_local_field_only=float(learned_residual - local_field_residual),
    )


def _slow_pc_metrics(model: Any, x_split: np.ndarray, y_split: np.ndarray) -> tuple[float, float]:
    predictions = model.predict(x_split)
    output_mse = float(np.mean((predictions - y_split) ** 2))
    accuracy = classification_accuracy(predictions, y_split)
    return output_mse, accuracy


def _stage04_config(
    config: FrozenBridgeVsCorrectedCoreComparisonConfig,
    *,
    seed: int,
    output_root: Path,
) -> FMPCTF2Config:
    return build_tf2_corrective_transport_terminal_angleclip_default_config(
        output_root=output_root,
        experiment_name=STAGE04_METHOD_NAME,
        output_layout="run_id_subdir",
        run_id=f"seed_{seed}",
        run_seed=seed,
        data_seed=seed,
        model_init_seed=seed,
        psi_init_seed=seed,
        batch_order_seed=seed,
        train_fraction=float(config.train_fraction),
        val_fraction=float(config.val_fraction),
        test_fraction=float(config.test_fraction),
        batch_size=int(config.batch_size),
        shuffle_batches=bool(config.shuffle_batches),
        epochs=int(config.stage04_epochs),
        eval_steps=int(config.stage04_eval_steps),
        layer_dims=config.stage04_layer_dims,
    )


def _stage05_config(
    config: FrozenBridgeVsCorrectedCoreComparisonConfig,
    *,
    seed: int,
    output_root: Path,
) -> FMPCEFExploratoryProbeConfig:
    return build_fmpc_ef_exploratory_probe_config(
        output_root=output_root,
        experiment_name=STAGE05_METHOD_NAME,
        output_layout="run_id_subdir",
        run_id=f"seed_{seed}",
        run_seed=seed,
        data_seed=seed,
        model_init_seed=seed,
        psi_init_seed=seed,
        batch_order_seed=seed,
        train_fraction=float(config.train_fraction),
        val_fraction=float(config.val_fraction),
        test_fraction=float(config.test_fraction),
        batch_size=int(config.batch_size),
        shuffle_batches=bool(config.shuffle_batches),
        epochs=int(config.stage05_epochs),
        eval_steps=int(config.stage05_eval_steps),
        layer_dims=config.stage05_layer_dims,
    )


def _stage05_v1_config(
    config: CorrectedResidualCoreV1VsV2ComparisonConfig,
    *,
    seed: int,
    output_root: Path,
) -> FMPCEFExploratoryProbeConfig:
    return build_fmpc_ef_exploratory_probe_config(
        output_root=output_root,
        experiment_name=STAGE05_V1_METHOD_NAME,
        output_layout="run_id_subdir",
        run_id=f"seed_{seed}",
        run_seed=seed,
        data_seed=seed,
        model_init_seed=seed,
        psi_init_seed=seed,
        batch_order_seed=seed,
        train_fraction=float(config.train_fraction),
        val_fraction=float(config.val_fraction),
        test_fraction=float(config.test_fraction),
        batch_size=int(config.batch_size),
        shuffle_batches=bool(config.shuffle_batches),
        epochs=int(config.stage05_epochs),
        eval_steps=int(config.stage05_eval_steps),
        layer_dims=config.stage05_layer_dims,
        transport_steps=int(config.stage05_transport_steps),
        use_two_branch_residual_core=False,
        feature_aware_state_branch_tangents=False,
    )


def _build_stage05_v2_config(
    *,
    output_root: Path,
    experiment_name: str,
    seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    batch_size: int,
    shuffle_batches: bool,
    epochs: int,
    eval_steps: int,
    layer_dims: tuple[int, ...],
    transport_steps: int,
    **overrides: Any,
) -> FMPCEFExploratoryProbeConfig:
    return build_fmpc_ef_exploratory_probe_config(
        output_root=output_root,
        experiment_name=experiment_name,
        output_layout="run_id_subdir",
        run_id=f"seed_{seed}",
        run_seed=seed,
        data_seed=seed,
        model_init_seed=seed,
        psi_init_seed=seed,
        batch_order_seed=seed,
        train_fraction=float(train_fraction),
        val_fraction=float(val_fraction),
        test_fraction=float(test_fraction),
        batch_size=int(batch_size),
        shuffle_batches=bool(shuffle_batches),
        epochs=int(epochs),
        eval_steps=int(eval_steps),
        layer_dims=layer_dims,
        transport_steps=int(transport_steps),
        use_two_branch_residual_core=True,
        feature_aware_state_branch_tangents=True,
        **overrides,
    )


def _stage05_v2_config(
    config: CorrectedResidualCoreV1VsV2ComparisonConfig,
    *,
    seed: int,
    output_root: Path,
) -> FMPCEFExploratoryProbeConfig:
    return _build_stage05_v2_config(
        output_root=output_root,
        experiment_name=STAGE05_V2_METHOD_NAME,
        seed=seed,
        train_fraction=float(config.train_fraction),
        val_fraction=float(config.val_fraction),
        test_fraction=float(config.test_fraction),
        batch_size=int(config.batch_size),
        shuffle_batches=bool(config.shuffle_batches),
        epochs=int(config.stage05_epochs),
        eval_steps=int(config.stage05_eval_steps),
        layer_dims=config.stage05_layer_dims,
        transport_steps=int(config.stage05_transport_steps),
    )


def _stage05_v3a_config(
    config: Stage05V2VsV3AComparisonConfig,
    *,
    seed: int,
    output_root: Path,
) -> FMPCEFExploratoryProbeConfig:
    return _build_stage05_v2_config(
        output_root=output_root,
        experiment_name=STAGE05_V3A_METHOD_NAME,
        seed=seed,
        train_fraction=float(config.train_fraction),
        val_fraction=float(config.val_fraction),
        test_fraction=float(config.test_fraction),
        batch_size=int(config.batch_size),
        shuffle_batches=bool(config.shuffle_batches),
        epochs=int(config.stage05_epochs),
        eval_steps=int(config.stage05_eval_steps),
        layer_dims=config.stage05_layer_dims,
        transport_steps=int(config.stage05_transport_steps),
        use_explicit_transport_drift_decomposition=True,
        lambda_drift=float(config.lambda_drift),
    )


def _stage05_v3b_config(
    config: Stage05V2V3AV3BComparisonConfig,
    *,
    seed: int,
    output_root: Path,
) -> FMPCEFExploratoryProbeConfig:
    return _build_stage05_v2_config(
        output_root=output_root,
        experiment_name=str(config.v3b_candidate_method_name),
        seed=seed,
        train_fraction=float(config.train_fraction),
        val_fraction=float(config.val_fraction),
        test_fraction=float(config.test_fraction),
        batch_size=int(config.batch_size),
        shuffle_batches=bool(config.shuffle_batches),
        epochs=int(config.stage05_epochs),
        eval_steps=int(config.stage05_eval_steps),
        layer_dims=config.stage05_layer_dims,
        transport_steps=int(config.stage05_transport_steps),
        use_explicit_transport_drift_decomposition=True,
        use_trajectory_curriculum_contract=True,
        lambda_drift=float(config.lambda_drift),
        lambda_traj_curr=float(config.lambda_traj_curr),
        alpha_floor=float(config.alpha_floor),
        alpha_warmup_epochs=int(config.alpha_warmup_epochs),
        alpha_ramp_epochs=int(config.alpha_ramp_epochs),
    )


def _stage05_v3b_candidate_stage_name(method_name: ComparisonMethodName) -> str:
    if method_name == STAGE05_V3B_METHOD_NAME:
        return "FMPC Stage 05 EF Core Probe v3-B Trajectory Curriculum Candidate"
    if method_name == STAGE05_V3B_ALPHA_EARLIER_METHOD_NAME:
        return "FMPC Stage 05 EF Core Probe v3-B Early-Transition Candidate"
    if method_name == STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME:
        return "FMPC Stage 05 EF Core Probe v3-B Stronger Trajectory-Weight Candidate"
    if method_name == STAGE05_V3C_METHOD_NAME:
        return "FMPC Stage 05 EF Core Probe v3-C Endpoint Semigroup Candidate"
    if method_name == STAGE05_V3C_STRONGER_SEMIGROUP_METHOD_NAME:
        return "FMPC Stage 05 EF Core Probe v3-C Stronger Semigroup-Weight Candidate"
    return f"FMPC Stage 05 EF Core Probe {method_name}"


def _stage05_v3b_variant_config(
    config: Stage05V3BRefinementDiagnosticConfig,
    *,
    seed: int,
    output_root: Path,
    experiment_name: ComparisonMethodName,
    lambda_traj_curr: float,
    alpha_floor: float,
    alpha_warmup_epochs: int,
    alpha_ramp_epochs: int,
) -> FMPCEFExploratoryProbeConfig:
    return _build_stage05_v2_config(
        output_root=output_root,
        experiment_name=experiment_name,
        seed=seed,
        train_fraction=float(config.train_fraction),
        val_fraction=float(config.val_fraction),
        test_fraction=float(config.test_fraction),
        batch_size=int(config.batch_size),
        shuffle_batches=bool(config.shuffle_batches),
        epochs=int(config.stage05_epochs),
        eval_steps=int(config.stage05_eval_steps),
        layer_dims=config.stage05_layer_dims,
        transport_steps=int(config.stage05_transport_steps),
        use_explicit_transport_drift_decomposition=True,
        use_trajectory_curriculum_contract=True,
        lambda_drift=float(config.lambda_drift),
        lambda_traj_curr=float(lambda_traj_curr),
        alpha_floor=float(alpha_floor),
        alpha_warmup_epochs=int(alpha_warmup_epochs),
        alpha_ramp_epochs=int(alpha_ramp_epochs),
        candidate_name_override=str(experiment_name),
    )


def _stage05_promoted_v3b_reference_config(
    config: Stage05V2PromotedV3BV3CComparisonConfig | Stage05V3CRefinementDiagnosticConfig,
    *,
    seed: int,
    output_root: Path,
) -> FMPCEFExploratoryProbeConfig:
    return build_stage05_v3b_stronger_traj_curr_weight_config(
        output_root=output_root,
        experiment_name=STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME,
        output_layout="run_id_subdir",
        run_id=f"seed_{seed}",
        run_seed=seed,
        data_seed=seed,
        model_init_seed=seed,
        psi_init_seed=seed,
        batch_order_seed=seed,
        train_fraction=float(config.train_fraction),
        val_fraction=float(config.val_fraction),
        test_fraction=float(config.test_fraction),
        batch_size=int(config.batch_size),
        shuffle_batches=bool(config.shuffle_batches),
        epochs=int(config.stage05_epochs),
        eval_steps=int(config.stage05_eval_steps),
        layer_dims=config.stage05_layer_dims,
        transport_steps=int(config.stage05_transport_steps),
        lambda_drift=float(config.lambda_drift),
        lambda_traj_curr=float(config.promoted_v3b_lambda_traj_curr),
        alpha_floor=float(config.promoted_v3b_alpha_floor),
        alpha_warmup_epochs=int(config.promoted_v3b_alpha_warmup_epochs),
        alpha_ramp_epochs=int(config.promoted_v3b_alpha_ramp_epochs),
    )


def _stage05_v3c_candidate_config(
    config: Stage05V2PromotedV3BV3CComparisonConfig,
    *,
    seed: int,
    output_root: Path,
) -> FMPCEFExploratoryProbeConfig:
    if config.v3c_candidate_method_name == STAGE05_V3C_METHOD_NAME:
        builder = build_stage05_v3c_endpoint_semigroup_config
    elif config.v3c_candidate_method_name == STAGE05_V3C_STRONGER_SEMIGROUP_METHOD_NAME:
        builder = build_stage05_v3c_stronger_semigroup_weight_config
    else:
        raise ValueError(
            f"Unsupported Stage 05 v3-C candidate '{config.v3c_candidate_method_name}'."
        )
    return builder(
        output_root=output_root,
        experiment_name=str(config.v3c_candidate_method_name),
        output_layout="run_id_subdir",
        run_id=f"seed_{seed}",
        run_seed=seed,
        data_seed=seed,
        model_init_seed=seed,
        psi_init_seed=seed,
        batch_order_seed=seed,
        train_fraction=float(config.train_fraction),
        val_fraction=float(config.val_fraction),
        test_fraction=float(config.test_fraction),
        batch_size=int(config.batch_size),
        shuffle_batches=bool(config.shuffle_batches),
        epochs=int(config.stage05_epochs),
        eval_steps=int(config.stage05_eval_steps),
        layer_dims=config.stage05_layer_dims,
        transport_steps=int(config.stage05_transport_steps),
        lambda_drift=float(config.lambda_drift),
        lambda_traj_curr=float(config.promoted_v3b_lambda_traj_curr),
        alpha_floor=float(config.promoted_v3b_alpha_floor),
        alpha_warmup_epochs=int(config.promoted_v3b_alpha_warmup_epochs),
        alpha_ramp_epochs=int(config.promoted_v3b_alpha_ramp_epochs),
        lambda_sg=float(config.lambda_sg),
    )


def _stage05_v3c_variant_config(
    config: Stage05V3CRefinementDiagnosticConfig,
    *,
    seed: int,
    output_root: Path,
    experiment_name: ComparisonMethodName,
    lambda_sg: float,
) -> FMPCEFExploratoryProbeConfig:
    if experiment_name == STAGE05_V3C_METHOD_NAME:
        builder = build_stage05_v3c_endpoint_semigroup_config
    elif experiment_name == STAGE05_V3C_STRONGER_SEMIGROUP_METHOD_NAME:
        builder = build_stage05_v3c_stronger_semigroup_weight_config
    else:
        raise ValueError(f"Unsupported Stage 05 v3-C variant '{experiment_name}'.")
    return builder(
        output_root=output_root,
        experiment_name=experiment_name,
        output_layout="run_id_subdir",
        run_id=f"seed_{seed}",
        run_seed=seed,
        data_seed=seed,
        model_init_seed=seed,
        psi_init_seed=seed,
        batch_order_seed=seed,
        train_fraction=float(config.train_fraction),
        val_fraction=float(config.val_fraction),
        test_fraction=float(config.test_fraction),
        batch_size=int(config.batch_size),
        shuffle_batches=bool(config.shuffle_batches),
        epochs=int(config.stage05_epochs),
        eval_steps=int(config.stage05_eval_steps),
        layer_dims=config.stage05_layer_dims,
        transport_steps=int(config.stage05_transport_steps),
        lambda_drift=float(config.lambda_drift),
        lambda_traj_curr=float(config.promoted_v3b_lambda_traj_curr),
        alpha_floor=float(config.promoted_v3b_alpha_floor),
        alpha_warmup_epochs=int(config.promoted_v3b_alpha_warmup_epochs),
        alpha_ramp_epochs=int(config.promoted_v3b_alpha_ramp_epochs),
        lambda_sg=float(lambda_sg),
    )


def _stage05_v2_bridge_config(
    config: FrozenBridgeVsStage05V2ComparisonConfig,
    *,
    seed: int,
    output_root: Path,
) -> FMPCEFExploratoryProbeConfig:
    return _build_stage05_v2_config(
        output_root=output_root,
        experiment_name=STAGE05_V2_METHOD_NAME,
        seed=seed,
        train_fraction=float(config.train_fraction),
        val_fraction=float(config.val_fraction),
        test_fraction=float(config.test_fraction),
        batch_size=int(config.batch_size),
        shuffle_batches=bool(config.shuffle_batches),
        epochs=int(config.stage05_epochs),
        eval_steps=int(config.stage05_eval_steps),
        layer_dims=config.stage05_layer_dims,
        transport_steps=int(config.stage05_transport_steps),
    )


def _stage05_v2_budget_config(
    config: (
        Stage05V2LongerTrainingValidationConfig
        | Stage05V2BudgetPushValidationConfig
        | Stage05V2EfficiencyDiagnosticConfig
    ),
    *,
    seed: int,
    output_root: Path,
    experiment_name: str,
    epochs: int,
    **overrides: Any,
) -> FMPCEFExploratoryProbeConfig:
    return _build_stage05_v2_config(
        output_root=output_root,
        experiment_name=experiment_name,
        seed=seed,
        train_fraction=float(config.train_fraction),
        val_fraction=float(config.val_fraction),
        test_fraction=float(config.test_fraction),
        batch_size=int(config.batch_size),
        shuffle_batches=bool(config.shuffle_batches),
        epochs=int(epochs),
        eval_steps=int(config.stage05_eval_steps),
        layer_dims=config.stage05_layer_dims,
        transport_steps=int(config.stage05_transport_steps),
        **overrides,
    )


def _artifact_checks(
    run_dir: Path,
    *,
    seed: int,
    expected_dataset_name: str,
    expected_batch_size: int,
    expected_shuffle_batches: bool,
) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    summary_path = run_dir / "summary.json"
    epoch_metrics_path = run_dir / "epoch_metrics.csv"
    checks = {
        "config_json_exists": bool(config_path.exists()),
        "summary_json_exists": bool(summary_path.exists()),
        "epoch_metrics_csv_exists": bool(epoch_metrics_path.exists()),
        "seed_matches": False,
        "dataset_matches": False,
        "batch_protocol_matches": False,
    }
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        dataset_payload = payload.get("dataset", {})
        run_payload = payload.get("run", {})
        checks["seed_matches"] = (
            int(run_payload.get("run_seed", -1)) == int(seed)
            and int(dataset_payload.get("data_seed", -1)) == int(seed)
        )
        checks["dataset_matches"] = str(dataset_payload.get("dataset_name", "")) == str(expected_dataset_name)
        checks["batch_protocol_matches"] = (
            int(run_payload.get("batch_size", -1)) == int(expected_batch_size)
            and bool(run_payload.get("shuffle_batches", False)) == bool(expected_shuffle_batches)
        )
    passed = all(bool(value) for value in checks.values())
    return {
        **checks,
        "deterministic_artifact_checks_passed": bool(passed),
    }


def _stage04_row(
    *,
    run_index: int,
    suite_run_dir: Path,
    seed: int,
    result: Any,
    config: FMPCTF2Config,
) -> dict[str, Any]:
    split = load_digits_split(
        split_seed=config.data_seed,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
    )
    val_one_step = _stage04_mechanism_metrics(
        result.model,
        result.psi_network,
        config,
        split.x_val,
        split.y_val,
        transport_steps=1,
    )
    val_configured = _stage04_mechanism_metrics(
        result.model,
        result.psi_network,
        config,
        split.x_val,
        split.y_val,
        transport_steps=config.micro_steps,
    )
    val_output_mse, val_accuracy = _slow_pc_metrics(result.model, split.x_val, split.y_val)
    test_output_mse, test_accuracy = _slow_pc_metrics(result.model, split.x_test, split.y_test)
    artifact_checks = _artifact_checks(
        result.run_dir,
        seed=seed,
        expected_dataset_name=config.dataset_name,
        expected_batch_size=config.batch_size,
        expected_shuffle_batches=config.shuffle_batches,
    )
    selected_epoch = int(result.summary.get("selected_epoch", config.epochs))
    timing = dict(result.summary.get("timing", {}))
    runtime_proxy = float(timing.get("train_wall_time_seconds", 0.0)) + float(
        timing.get("final_evaluation_wall_time_seconds", 0.0)
    )
    return {
        "run_index": int(run_index),
        "method_name": STAGE04_METHOD_NAME,
        "stage_name": "FMPC Stage 04 Incremental Bridge",
        "seed": int(seed),
        "run_id": str(result.run_dir.name),
        "run_config_path": _relative_posix(suite_run_dir, result.run_dir / "config.json"),
        "run_summary_path": _relative_posix(suite_run_dir, result.run_dir / "summary.json"),
        "transport_family": "stage04_frozen_bridge_control",
        "residual_branch_structure": "not_applicable",
        "configured_transport_steps": int(config.micro_steps),
        "one_step_energy_delta_vs_identity": float(val_one_step.energy_delta_vs_identity),
        "configured_step_energy_delta_vs_identity": float(val_configured.energy_delta_vs_identity),
        "configured_step_fixed_point_residual_delta_vs_identity": float(
            val_configured.fixed_point_residual_delta_vs_identity
        ),
        "one_step_energy_delta_vs_local_field_only": float(val_one_step.energy_delta_vs_local_field_only),
        "configured_step_energy_delta_vs_local_field_only": float(
            val_configured.energy_delta_vs_local_field_only
        ),
        "configured_step_fixed_point_residual_delta_vs_local_field_only": float(
            val_configured.fixed_point_residual_delta_vs_local_field_only
        ),
        "val_accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy),
        "val_output_mse": float(val_output_mse),
        "test_output_mse": float(test_output_mse),
        "selected_epoch": int(selected_epoch),
        "total_training_epochs": int(config.epochs),
        "selection_hits_final_training_boundary": bool(int(selected_epoch) >= int(config.epochs)),
        "runtime_proxy_seconds": float(runtime_proxy),
        "acceptance_contract": "mechanism_first_comparison_context",
        "mechanism_signal_positive": bool(
            float(val_one_step.energy_delta_vs_identity) < 0.0
            and float(val_configured.energy_delta_vs_identity) < 0.0
            and float(val_configured.fixed_point_residual_delta_vs_identity) < 0.0
        ),
        **artifact_checks,
    }


def _stage05_row(
    *,
    run_index: int,
    suite_run_dir: Path,
    seed: int,
    result: Any,
    config: FMPCEFExploratoryProbeConfig,
) -> dict[str, Any]:
    summary = result.summary
    val_one_step = summary["mechanism_metrics"]["one_step"]
    val_configured = summary["mechanism_metrics"]["configured_steps"]
    artifact_checks = _artifact_checks(
        result.run_dir,
        seed=seed,
        expected_dataset_name=config.dataset_name,
        expected_batch_size=config.batch_size,
        expected_shuffle_batches=config.shuffle_batches,
    )
    selected_epoch = int(summary["selected_epoch"])
    runtime_proxy = float(summary.get("train_wall_time_seconds", 0.0)) + float(
        summary.get("evaluation_wall_time_seconds", 0.0)
    )
    return {
        "run_index": int(run_index),
        "method_name": STAGE05_METHOD_NAME,
        "stage_name": "FMPC Stage 05 EF Core Probe",
        "seed": int(seed),
        "run_id": str(result.run_dir.name),
        "run_config_path": _relative_posix(suite_run_dir, result.run_dir / "config.json"),
        "run_summary_path": _relative_posix(suite_run_dir, result.run_dir / "summary.json"),
        "configured_transport_steps": int(config.transport_steps),
        "one_step_energy_delta_vs_identity": float(val_one_step["energy_delta_vs_identity"]),
        "configured_step_energy_delta_vs_identity": float(val_configured["energy_delta_vs_identity"]),
        "configured_step_fixed_point_residual_delta_vs_identity": float(
            val_configured["fixed_point_residual_delta_vs_identity"]
        ),
        "one_step_energy_delta_vs_local_field_only": float(
            val_one_step["energy_delta_vs_local_field_only"]
        ),
        "configured_step_energy_delta_vs_local_field_only": float(
            val_configured["energy_delta_vs_local_field_only"]
        ),
        "configured_step_fixed_point_residual_delta_vs_local_field_only": float(
            val_configured["fixed_point_residual_delta_vs_local_field_only"]
        ),
        "val_accuracy": float(summary["val_accuracy"]),
        "test_accuracy": float(summary["test_accuracy"]),
        "val_output_mse": float(summary["val_output_mse"]),
        "test_output_mse": float(summary["test_output_mse"]),
        "selected_epoch": int(selected_epoch),
        "total_training_epochs": int(config.epochs),
        "selection_hits_final_training_boundary": bool(int(selected_epoch) >= int(config.epochs)),
        "runtime_proxy_seconds": float(runtime_proxy),
        "acceptance_contract": str(summary["acceptance_contract"]),
        "mechanism_signal_positive": bool(
            float(val_one_step["energy_delta_vs_identity"]) < 0.0
            and float(val_configured["energy_delta_vs_identity"]) < 0.0
            and float(val_configured["fixed_point_residual_delta_vs_identity"]) < 0.0
        ),
        **artifact_checks,
    }


def _stage05_core_row(
    *,
    run_index: int,
    suite_run_dir: Path,
    seed: int,
    result: Any,
    config: FMPCEFExploratoryProbeConfig,
    method_name: ComparisonMethodName,
    stage_name: str,
) -> dict[str, Any]:
    summary = result.summary
    val_one_step = summary["mechanism_metrics"]["one_step"]
    val_configured = summary["mechanism_metrics"]["configured_steps"]
    artifact_checks = _artifact_checks(
        result.run_dir,
        seed=seed,
        expected_dataset_name=config.dataset_name,
        expected_batch_size=config.batch_size,
        expected_shuffle_batches=config.shuffle_batches,
    )
    runtime_proxy = float(summary.get("train_wall_time_seconds", 0.0)) + float(
        summary.get("evaluation_wall_time_seconds", 0.0)
    )
    return {
        "run_index": int(run_index),
        "method_name": method_name,
        "stage_name": stage_name,
        "seed": int(seed),
        "run_id": str(result.run_dir.name),
        "run_config_path": _relative_posix(suite_run_dir, result.run_dir / "config.json"),
        "run_summary_path": _relative_posix(suite_run_dir, result.run_dir / "summary.json"),
        "candidate_name": summary.get("candidate_name"),
        "transport_family": str(summary["transport_family"]),
        "residual_branch_structure": str(summary["residual_branch_structure"]),
        "explicit_transport_drift_decomposition_enabled": bool(
            summary.get("explicit_transport_drift_decomposition_enabled", False)
        ),
        "trajectory_curriculum_enabled": bool(summary.get("trajectory_curriculum_enabled", False)),
        "trajectory_curriculum_schedule_identity": summary.get(
            "trajectory_curriculum_schedule_identity"
        ),
        "lambda_traj_curr": summary.get("lambda_traj_curr"),
        "alpha_floor": summary.get("alpha_floor"),
        "alpha_warmup_epochs": summary.get("alpha_warmup_epochs"),
        "alpha_ramp_epochs": summary.get("alpha_ramp_epochs"),
        "configured_transport_steps": int(config.transport_steps),
        "one_step_energy_delta_vs_identity": float(val_one_step["energy_delta_vs_identity"]),
        "configured_step_energy_delta_vs_identity": float(val_configured["energy_delta_vs_identity"]),
        "configured_step_fixed_point_residual_delta_vs_identity": float(
            val_configured["fixed_point_residual_delta_vs_identity"]
        ),
        "one_step_energy_delta_vs_local_field_only": float(
            val_one_step["energy_delta_vs_local_field_only"]
        ),
        "configured_step_energy_delta_vs_local_field_only": float(
            val_configured["energy_delta_vs_local_field_only"]
        ),
        "configured_step_fixed_point_residual_delta_vs_local_field_only": float(
            val_configured["fixed_point_residual_delta_vs_local_field_only"]
        ),
        "val_accuracy": float(summary["val_accuracy"]),
        "test_accuracy": float(summary["test_accuracy"]),
        "val_output_mse": float(summary["val_output_mse"]),
        "test_output_mse": float(summary["test_output_mse"]),
        "selected_epoch": int(summary["selected_epoch"]),
        "total_training_epochs": int(config.epochs),
        "selection_hits_final_training_boundary": bool(int(summary["selected_epoch"]) >= int(config.epochs)),
        "runtime_proxy_seconds": float(runtime_proxy),
        "acceptance_contract": str(summary["acceptance_contract"]),
        "mechanism_signal_positive": bool(
            float(val_one_step["energy_delta_vs_identity"]) < 0.0
            and float(val_configured["energy_delta_vs_identity"]) < 0.0
            and float(val_configured["fixed_point_residual_delta_vs_identity"]) < 0.0
        ),
        **artifact_checks,
    }


def _load_existing_stage05_core_row(
    *,
    run_index: int,
    existing_run_dir: Path,
    seed: int,
    expected_dataset_name: str,
    expected_batch_size: int,
    expected_shuffle_batches: bool,
    method_name: ComparisonMethodName,
    stage_name: str,
    expected_total_training_epochs: int | None = None,
    expected_eval_steps: int | None = None,
    expected_layer_dims: tuple[int, ...] | None = None,
    expected_transport_steps: int | None = None,
    expected_transport_family: str | None = None,
    expected_explicit_transport_drift_decomposition_enabled: bool | None = None,
    expected_trajectory_curriculum_enabled: bool | None = None,
    expected_endpoint_semigroup_consistency_enabled: bool | None = None,
    expected_lambda_traj_curr: float | None = None,
    expected_alpha_floor: float | None = None,
    expected_lambda_sg: float | None = None,
) -> dict[str, Any]:
    summary = _read_json(existing_run_dir / "summary.json")
    config_payload = _read_json(existing_run_dir / "config.json")
    val_one_step = summary["mechanism_metrics"]["one_step"]
    val_configured = summary["mechanism_metrics"]["configured_steps"]
    run_payload = config_payload["run"]
    model_payload = config_payload["model"]
    transport_payload = config_payload["transport"]
    artifact_checks = _artifact_checks(
        existing_run_dir,
        seed=seed,
        expected_dataset_name=expected_dataset_name,
        expected_batch_size=expected_batch_size,
        expected_shuffle_batches=expected_shuffle_batches,
    )
    expected_layer_dims_list = (
        [int(value) for value in expected_layer_dims] if expected_layer_dims is not None else None
    )
    model_protocol_matches = True
    if expected_total_training_epochs is not None:
        model_protocol_matches = model_protocol_matches and (
            int(run_payload.get("epochs", -1)) == int(expected_total_training_epochs)
        )
    if expected_eval_steps is not None:
        model_protocol_matches = model_protocol_matches and (
            int(model_payload.get("eval_steps", -1)) == int(expected_eval_steps)
        )
    if expected_layer_dims_list is not None:
        model_protocol_matches = model_protocol_matches and (
            [int(value) for value in model_payload.get("layer_dims", [])] == expected_layer_dims_list
        )
    transport_protocol_matches = True
    if expected_transport_steps is not None:
        transport_protocol_matches = transport_protocol_matches and (
            int(transport_payload.get("transport_steps", -1)) == int(expected_transport_steps)
        )
    if expected_transport_family is not None:
        transport_protocol_matches = transport_protocol_matches and (
            str(summary.get("transport_family")) == str(expected_transport_family)
        )
    if expected_explicit_transport_drift_decomposition_enabled is not None:
        transport_protocol_matches = transport_protocol_matches and (
            bool(summary.get("explicit_transport_drift_decomposition_enabled", False))
            == bool(expected_explicit_transport_drift_decomposition_enabled)
        )
    if expected_trajectory_curriculum_enabled is not None:
        transport_protocol_matches = transport_protocol_matches and (
            bool(summary.get("trajectory_curriculum_enabled", False))
            == bool(expected_trajectory_curriculum_enabled)
        )
    if expected_endpoint_semigroup_consistency_enabled is not None:
        transport_protocol_matches = transport_protocol_matches and (
            bool(summary.get("endpoint_semigroup_consistency_enabled", False))
            == bool(expected_endpoint_semigroup_consistency_enabled)
        )
    if expected_lambda_traj_curr is not None:
        transport_protocol_matches = transport_protocol_matches and np.isclose(
            float(summary.get("lambda_traj_curr", 0.0)),
            float(expected_lambda_traj_curr),
        )
    if expected_alpha_floor is not None:
        transport_protocol_matches = transport_protocol_matches and np.isclose(
            float(summary.get("alpha_floor", 0.0)),
            float(expected_alpha_floor),
        )
    if expected_lambda_sg is not None:
        transport_protocol_matches = transport_protocol_matches and np.isclose(
            float(summary.get("lambda_sg", 0.0)),
            float(expected_lambda_sg),
        )
    deterministic_artifact_checks_passed = bool(
        artifact_checks["deterministic_artifact_checks_passed"]
        and model_protocol_matches
        and transport_protocol_matches
    )
    runtime_proxy = float(summary.get("train_wall_time_seconds", 0.0)) + float(
        summary.get("evaluation_wall_time_seconds", 0.0)
    )
    total_training_epochs = int(run_payload["epochs"])
    selected_epoch = int(summary["selected_epoch"])
    return {
        "run_index": int(run_index),
        "method_name": method_name,
        "stage_name": stage_name,
        "seed": int(seed),
        "run_id": str(existing_run_dir.name),
        "run_config_path": _repo_relative_posix(existing_run_dir / "config.json"),
        "run_summary_path": _repo_relative_posix(existing_run_dir / "summary.json"),
        "candidate_name": summary.get("candidate_name"),
        "transport_family": str(summary["transport_family"]),
        "residual_branch_structure": str(summary["residual_branch_structure"]),
        "explicit_transport_drift_decomposition_enabled": bool(
            summary.get("explicit_transport_drift_decomposition_enabled", False)
        ),
        "trajectory_curriculum_enabled": bool(summary.get("trajectory_curriculum_enabled", False)),
        "trajectory_curriculum_schedule_identity": summary.get(
            "trajectory_curriculum_schedule_identity"
        ),
        "lambda_traj_curr": summary.get("lambda_traj_curr"),
        "alpha_floor": summary.get("alpha_floor"),
        "alpha_warmup_epochs": summary.get("alpha_warmup_epochs"),
        "alpha_ramp_epochs": summary.get("alpha_ramp_epochs"),
        "configured_transport_steps": int(transport_payload["transport_steps"]),
        "one_step_energy_delta_vs_identity": float(val_one_step["energy_delta_vs_identity"]),
        "configured_step_energy_delta_vs_identity": float(val_configured["energy_delta_vs_identity"]),
        "configured_step_fixed_point_residual_delta_vs_identity": float(
            val_configured["fixed_point_residual_delta_vs_identity"]
        ),
        "one_step_energy_delta_vs_local_field_only": float(
            val_one_step["energy_delta_vs_local_field_only"]
        ),
        "configured_step_energy_delta_vs_local_field_only": float(
            val_configured["energy_delta_vs_local_field_only"]
        ),
        "configured_step_fixed_point_residual_delta_vs_local_field_only": float(
            val_configured["fixed_point_residual_delta_vs_local_field_only"]
        ),
        "val_accuracy": float(summary["val_accuracy"]),
        "test_accuracy": float(summary["test_accuracy"]),
        "val_output_mse": float(summary["val_output_mse"]),
        "test_output_mse": float(summary["test_output_mse"]),
        "selected_epoch": int(selected_epoch),
        "total_training_epochs": int(total_training_epochs),
        "selection_hits_final_training_boundary": bool(selected_epoch >= total_training_epochs),
        "runtime_proxy_seconds": float(runtime_proxy),
        "acceptance_contract": str(summary["acceptance_contract"]),
        "mechanism_signal_positive": bool(
            float(val_one_step["energy_delta_vs_identity"]) < 0.0
            and float(val_configured["energy_delta_vs_identity"]) < 0.0
            and float(val_configured["fixed_point_residual_delta_vs_identity"]) < 0.0
        ),
        **{
            **artifact_checks,
            "model_protocol_matches": bool(model_protocol_matches),
            "transport_protocol_matches": bool(transport_protocol_matches),
            "deterministic_artifact_checks_passed": bool(deterministic_artifact_checks_passed),
        },
    }


def _positive_gap_closed_fraction(
    *,
    reference_value: float,
    candidate_value: float,
    target_value: float,
) -> float:
    remaining_gap = float(target_value - reference_value)
    if remaining_gap <= 0.0:
        return 1.0 if candidate_value >= reference_value else 0.0
    improvement = max(float(candidate_value - reference_value), 0.0)
    return float(min(improvement / remaining_gap, 1.0))


def _negative_gap_closed_fraction(
    *,
    reference_value: float,
    candidate_value: float,
    target_value: float,
) -> float:
    reference_magnitude = abs(float(reference_value))
    candidate_magnitude = abs(float(candidate_value))
    target_magnitude = abs(float(target_value))
    remaining_gap = float(target_magnitude - reference_magnitude)
    if remaining_gap <= 0.0:
        return 1.0 if candidate_magnitude >= reference_magnitude else 0.0
    improvement = max(float(candidate_magnitude - reference_magnitude), 0.0)
    return float(min(improvement / remaining_gap, 1.0))


def _method_rows(rows: list[dict[str, Any]], method_name: ComparisonMethodName) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["method_name"]) == str(method_name)]


def _method_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Method summary requires at least one row.")

    def _metric_summary(field: str) -> dict[str, float]:
        values = [float(row[field]) for row in rows]
        return {"mean": _mean(values), "std": _std(values)}

    artifact_checks = [bool(row["deterministic_artifact_checks_passed"]) for row in rows]
    mechanism_checks = [bool(row["mechanism_signal_positive"]) for row in rows]
    return {
        "num_runs": int(len(rows)),
        "configured_transport_steps": int(rows[0]["configured_transport_steps"]),
        "total_training_epochs": int(rows[0].get("total_training_epochs", 0)),
        "selected_epoch": _metric_summary("selected_epoch"),
        "selection_hits_final_training_boundary_rate": _rate(
            [bool(row.get("selection_hits_final_training_boundary", False)) for row in rows]
        ),
        "deterministic_artifact_check_rate": _rate(artifact_checks),
        "mechanism_signal_positive_rate": _rate(mechanism_checks),
        "one_step_energy_delta_vs_identity": _metric_summary("one_step_energy_delta_vs_identity"),
        "configured_step_energy_delta_vs_identity": _metric_summary(
            "configured_step_energy_delta_vs_identity"
        ),
        "configured_step_fixed_point_residual_delta_vs_identity": _metric_summary(
            "configured_step_fixed_point_residual_delta_vs_identity"
        ),
        "one_step_energy_delta_vs_local_field_only": _metric_summary(
            "one_step_energy_delta_vs_local_field_only"
        ),
        "configured_step_energy_delta_vs_local_field_only": _metric_summary(
            "configured_step_energy_delta_vs_local_field_only"
        ),
        "configured_step_fixed_point_residual_delta_vs_local_field_only": _metric_summary(
            "configured_step_fixed_point_residual_delta_vs_local_field_only"
        ),
        "val_accuracy": _metric_summary("val_accuracy"),
        "test_accuracy": _metric_summary("test_accuracy"),
        "val_output_mse": _metric_summary("val_output_mse"),
        "test_output_mse": _metric_summary("test_output_mse"),
        "runtime_proxy_seconds": _metric_summary("runtime_proxy_seconds"),
    }


def _pairwise_summary(
    rows: list[dict[str, Any]],
    *,
    candidate_method: ComparisonMethodName,
    reference_method: ComparisonMethodName,
) -> dict[str, Any]:
    candidate_by_seed = {int(row["seed"]): row for row in _method_rows(rows, candidate_method)}
    reference_by_seed = {int(row["seed"]): row for row in _method_rows(rows, reference_method)}
    shared_seeds = sorted(set(candidate_by_seed).intersection(reference_by_seed))
    if not shared_seeds:
        raise ValueError("Pairwise summary requires at least one shared seed.")

    def _delta(field: str) -> dict[str, float]:
        values = [
            float(candidate_by_seed[seed][field]) - float(reference_by_seed[seed][field])
            for seed in shared_seeds
        ]
        return {"mean": _mean(values), "std": _std(values)}

    return {
        "shared_seeds": [int(seed) for seed in shared_seeds],
        "one_step_energy_delta_vs_identity_delta": _delta("one_step_energy_delta_vs_identity"),
        "configured_step_energy_delta_vs_identity_delta": _delta(
            "configured_step_energy_delta_vs_identity"
        ),
        "configured_step_fixed_point_residual_delta_vs_identity_delta": _delta(
            "configured_step_fixed_point_residual_delta_vs_identity"
        ),
        "configured_step_energy_delta_vs_local_field_only_delta": _delta(
            "configured_step_energy_delta_vs_local_field_only"
        ),
        "val_accuracy_delta": _delta("val_accuracy"),
        "test_accuracy_delta": _delta("test_accuracy"),
        "runtime_proxy_seconds_delta": _delta("runtime_proxy_seconds"),
    }


def _comparison_protocol_payload(config: FrozenBridgeVsCorrectedCoreComparisonConfig) -> dict[str, Any]:
    return {
        "dataset_name": config.dataset_name,
        "seeds": [int(seed) for seed in config.seeds],
        "train_fraction": float(config.train_fraction),
        "val_fraction": float(config.val_fraction),
        "test_fraction": float(config.test_fraction),
        "shared_batch_size": int(config.batch_size),
        "shared_shuffle_batches": bool(config.shuffle_batches),
        "stage_04_control": {
            "method_name": STAGE04_METHOD_NAME,
            "preset_name": "tf2_corrective_transport_terminal_angleclip_default",
            "configured_transport_steps": 4,
            "epochs": int(config.stage04_epochs),
            "eval_steps": int(config.stage04_eval_steps),
            "layer_dims": [int(value) for value in config.stage04_layer_dims],
        },
        "stage_05_candidate": {
            "method_name": STAGE05_METHOD_NAME,
            "transport_family": "residual_meanflow_core",
            "residual_identity_mode": "residual_corrected_meanflow",
            "configured_transport_steps": 2,
            "epochs": int(config.stage05_epochs),
            "eval_steps": int(config.stage05_eval_steps),
            "layer_dims": [int(value) for value in config.stage05_layer_dims],
        },
        "decision_rule": {
            "primary_split": "validation",
            "requires_all_stage05_artifact_checks": True,
            "requires_stage05_one_step_energy_delta_vs_identity_negative_on_every_seed": True,
            "requires_stage05_configured_step_energy_delta_vs_identity_negative_on_every_seed": True,
            "requires_stage05_configured_step_fixed_point_residual_delta_vs_identity_negative_on_every_seed": True,
            "task_accuracy_is_report_only": True,
        },
    }


def _stage05_v2_charter_decision(rows: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    stage05_rows = _method_rows(rows, STAGE05_METHOD_NAME)
    artifact_pass = all(bool(row["deterministic_artifact_checks_passed"]) for row in stage05_rows)
    one_step_pass = all(float(row["one_step_energy_delta_vs_identity"]) < 0.0 for row in stage05_rows)
    configured_energy_pass = all(
        float(row["configured_step_energy_delta_vs_identity"]) < 0.0 for row in stage05_rows
    )
    configured_residual_pass = all(
        float(row["configured_step_fixed_point_residual_delta_vs_identity"]) < 0.0
        for row in stage05_rows
    )
    justified = bool(artifact_pass and one_step_pass and configured_energy_pass and configured_residual_pass)
    return justified, {
        "artifact_checks_all_pass": bool(artifact_pass),
        "one_step_energy_delta_vs_identity_negative_on_every_seed": bool(one_step_pass),
        "configured_step_energy_delta_vs_identity_negative_on_every_seed": bool(
            configured_energy_pass
        ),
        "configured_step_fixed_point_residual_delta_vs_identity_negative_on_every_seed": bool(
            configured_residual_pass
        ),
    }


def _supports_lines(
    *,
    justified: bool,
    decision_detail: dict[str, Any],
    by_method: dict[str, dict[str, Any]],
) -> list[str]:
    stage05 = by_method[STAGE05_METHOD_NAME]
    lines = [
        "Stage 05 comparison artifacts are reproducible under the shared dataset/seed/batch protocol."
        if decision_detail["artifact_checks_all_pass"]
        else "Stage 05 comparison artifacts do not yet fully satisfy the shared artifact protocol.",
        (
            "Stage 05 shows stable negative validation one-step energy delta vs identity across all comparison seeds."
            if decision_detail["one_step_energy_delta_vs_identity_negative_on_every_seed"]
            else "Stage 05 does not keep validation one-step energy delta vs identity negative on every comparison seed."
        ),
        (
            "Stage 05 keeps configured-step validation fixed-point residual delta vs identity negative across all comparison seeds."
            if decision_detail["configured_step_fixed_point_residual_delta_vs_identity_negative_on_every_seed"]
            else "Stage 05 does not keep configured-step validation fixed-point residual delta vs identity negative on every comparison seed."
        ),
    ]
    if justified:
        lines.append(
            "The corrected residual core has enough mechanism-first signal to justify a Stage 05 v2 charter."
        )
    else:
        lines.append(
            "The corrected residual core does not yet clear the stricter multiseed mechanism-first rule for a Stage 05 v2 charter."
        )
    lines.append("Stage 05 report-only accuracy remains a contextual metric, not the gate, in this comparison.")
    lines.append(
        f"Stage 05 mean validation accuracy is {stage05['val_accuracy']['mean']:.6f}, which is reported but not used as the charter gate."
    )
    return lines


def _does_not_support_lines(
    *,
    by_method: dict[str, dict[str, Any]],
    pairwise_stage05_vs_stage04: dict[str, Any],
) -> list[str]:
    stage04 = by_method[STAGE04_METHOD_NAME]
    stage05 = by_method[STAGE05_METHOD_NAME]
    lines = [
        "This comparison does not promote Stage 05 to replace the frozen Stage 04 bridge on main.",
        "This comparison does not reopen any Stage 04 package-internal stabilizer search.",
        "This comparison does not claim that Stage 05 has solved the task-accuracy gap to the frozen bridge.",
    ]
    if float(stage05["test_accuracy"]["mean"]) <= float(stage04["test_accuracy"]["mean"]):
        lines.append("Stage 05 remains below the frozen bridge on report-only test accuracy in the current comparison.")
    if float(pairwise_stage05_vs_stage04["configured_step_energy_delta_vs_identity_delta"]["mean"]) >= 0.0:
        lines.append(
            "Stage 05 does not outperform the frozen bridge on configured-step energy delta vs identity in this comparison."
        )
    return lines


def _comparison_report_markdown(report: dict[str, Any]) -> str:
    protocol = report["comparison_protocol"]
    decision = report["decision"]
    supports = report["supports"]
    does_not_support = report["does_not_support"]
    lines = [
        "# Frozen Bridge vs Corrected Residual Core",
        "",
        "## Protocol",
        f"- dataset: `{protocol['dataset_name']}`",
        f"- seeds: `{protocol['seeds']}`",
        f"- shared batch size: `{protocol['shared_batch_size']}`",
        f"- shared shuffle_batches: `{protocol['shared_shuffle_batches']}`",
        "",
        "## Decision",
        f"- `{JUSTIFY_V2_DECISION_NAME}`: `{decision[JUSTIFY_V2_DECISION_NAME]}`",
        f"- rationale: `{decision['decision_rationale']}`",
        "",
        "## Supports",
    ]
    for item in supports:
        lines.append(f"- {item}")
    lines.extend(["", "## Does Not Support"])
    for item in does_not_support:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _suite_config_payload(config: FrozenBridgeVsCorrectedCoreComparisonConfig) -> dict[str, Any]:
    return {
        "phase": "FMPC Stage 05 EF Core Probe",
        "stage": "frozen_bridge_vs_corrected_residual_core_comparison",
        "comparison_protocol": _comparison_protocol_payload(config),
        "artifacts": {
            "aggregate_runs_csv": "aggregate_runs.csv",
            "aggregate_summary_json": "aggregate_summary.json",
            "comparison_report_json": "comparison_report.json",
            "comparison_report_md": "comparison_report.md",
        },
    }


def run_frozen_bridge_vs_corrected_core_comparison(
    config: FrozenBridgeVsCorrectedCoreComparisonConfig,
) -> FrozenBridgeVsCorrectedCoreComparisonRunResult:
    """Run the formal frozen-bridge vs corrected-core comparison."""

    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, config.resolved_run_id(), config.output_layout)
    )
    _write_json(run_dir / "config.json", _suite_config_payload(config))

    rows: list[dict[str, Any]] = []
    runs_root = run_dir / "runs"
    run_index = 0

    for seed in config.seeds:
        run_index += 1
        stage04_config = _stage04_config(config, seed=seed, output_root=runs_root)
        stage04_result = run_fmpc_tf2_experiment(stage04_config)
        rows.append(
            _stage04_row(
                run_index=run_index,
                suite_run_dir=run_dir,
                seed=seed,
                result=stage04_result,
                config=stage04_config,
            )
        )

        run_index += 1
        stage05_config = _stage05_config(config, seed=seed, output_root=runs_root)
        stage05_result = run_fmpc_ef_exploratory_probe(stage05_config)
        rows.append(
            _stage05_row(
                run_index=run_index,
                suite_run_dir=run_dir,
                seed=seed,
                result=stage05_result,
                config=stage05_config,
            )
        )

    csv_rows = [
        {
            **row,
            "deterministic_artifact_checks_passed": str(bool(row["deterministic_artifact_checks_passed"])),
            "mechanism_signal_positive": str(bool(row["mechanism_signal_positive"])),
            "config_json_exists": str(bool(row["config_json_exists"])),
            "summary_json_exists": str(bool(row["summary_json_exists"])),
            "epoch_metrics_csv_exists": str(bool(row["epoch_metrics_csv_exists"])),
            "seed_matches": str(bool(row["seed_matches"])),
            "dataset_matches": str(bool(row["dataset_matches"])),
            "batch_protocol_matches": str(bool(row["batch_protocol_matches"])),
        }
        for row in rows
    ]
    _write_csv(run_dir / "aggregate_runs.csv", csv_rows)

    by_method = {
        STAGE04_METHOD_NAME: _method_summary(_method_rows(rows, STAGE04_METHOD_NAME)),
        STAGE05_METHOD_NAME: _method_summary(_method_rows(rows, STAGE05_METHOD_NAME)),
    }
    pairwise_stage05_vs_stage04 = _pairwise_summary(
        rows,
        candidate_method=STAGE05_METHOD_NAME,
        reference_method=STAGE04_METHOD_NAME,
    )
    justified, decision_detail = _stage05_v2_charter_decision(rows)
    decision_rationale = (
        "Stage 05 clears the multiseed mechanism-first comparison rule."
        if justified
        else "Stage 05 remains below the stricter multiseed mechanism-first charter rule."
    )

    summary = {
        "phase": "FMPC Stage 05 EF Core Probe",
        "stage": "frozen_bridge_vs_corrected_residual_core_comparison",
        "num_runs": int(len(rows)),
        "comparison_protocol": _comparison_protocol_payload(config),
        "by_method": by_method,
        "pairwise_stage05_vs_stage04": pairwise_stage05_vs_stage04,
        JUSTIFY_V2_DECISION_NAME: bool(justified),
        "stage05_v2_charter_decision_detail": decision_detail,
        "stage05_v2_charter_decision_rationale": decision_rationale,
        "aggregate_runs_csv_path": "aggregate_runs.csv",
        "comparison_report_json_path": "comparison_report.json",
        "comparison_report_md_path": "comparison_report.md",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)

    report = {
        "comparison_protocol": _comparison_protocol_payload(config),
        "decision": {
            JUSTIFY_V2_DECISION_NAME: bool(justified),
            "decision_detail": decision_detail,
            "decision_rationale": decision_rationale,
        },
        "supports": _supports_lines(
            justified=justified,
            decision_detail=decision_detail,
            by_method=by_method,
        ),
        "does_not_support": _does_not_support_lines(
            by_method=by_method,
            pairwise_stage05_vs_stage04=pairwise_stage05_vs_stage04,
        ),
    }
    _write_json(run_dir / "comparison_report.json", report)
    _write_text(run_dir / "comparison_report.md", _comparison_report_markdown(report))

    return FrozenBridgeVsCorrectedCoreComparisonRunResult(
        run_dir=run_dir,
        config=_suite_config_payload(config),
        aggregate_rows=rows,
        summary=summary,
        comparison_report=report,
    )


def _stage05_v2_vs_stage04_protocol_payload(
    config: FrozenBridgeVsStage05V2ComparisonConfig,
) -> dict[str, Any]:
    return {
        "dataset_name": config.dataset_name,
        "seeds": [int(seed) for seed in config.seeds],
        "train_fraction": float(config.train_fraction),
        "val_fraction": float(config.val_fraction),
        "test_fraction": float(config.test_fraction),
        "shared_batch_size": int(config.batch_size),
        "shared_shuffle_batches": bool(config.shuffle_batches),
        "stage_04_control": {
            "method_name": STAGE04_METHOD_NAME,
            "preset_name": "tf2_corrective_transport_terminal_angleclip_default",
            "configured_transport_steps": 4,
            "epochs": int(config.stage04_epochs),
            "eval_steps": int(config.stage04_eval_steps),
            "layer_dims": [int(value) for value in config.stage04_layer_dims],
        },
        "stage_05_candidate": {
            "method_name": STAGE05_V2_METHOD_NAME,
            "transport_family": "two_branch_residual_meanflow_core",
            "residual_identity_mode": "residual_corrected_meanflow",
            "residual_branch_structure": "two_branch",
            "feature_aware_state_branch_tangents": True,
            "configured_transport_steps": int(config.stage05_transport_steps),
            "epochs": int(config.stage05_epochs),
            "eval_steps": int(config.stage05_eval_steps),
            "layer_dims": [int(value) for value in config.stage05_layer_dims],
        },
        "decision_rule": {
            "primary_split": "validation",
            "requires_all_stage05_v2_artifact_checks": True,
            "requires_stage05_v2_one_step_energy_delta_vs_identity_negative_on_every_seed": True,
            "requires_stage05_v2_configured_step_energy_delta_vs_identity_negative_on_every_seed": True,
            "requires_stage05_v2_configured_step_fixed_point_residual_delta_vs_identity_negative_on_every_seed": True,
            "task_accuracy_is_report_only": True,
            "replacement_claim_expected": False,
        },
    }


def _mechanism_strength_label(
    *,
    candidate_energy_mean: float,
    reference_energy_mean: float,
    candidate_residual_mean: float | None = None,
    reference_residual_mean: float | None = None,
) -> str:
    energy_better = float(candidate_energy_mean) < float(reference_energy_mean)
    if candidate_residual_mean is None or reference_residual_mean is None:
        return "stronger" if energy_better else "weaker"
    residual_better = float(candidate_residual_mean) < float(reference_residual_mean)
    return "stronger" if (energy_better and residual_better) else "weaker"


def _report_accuracy_strength_label(
    *,
    candidate_val_accuracy_mean: float,
    reference_val_accuracy_mean: float,
    candidate_test_accuracy_mean: float,
    reference_test_accuracy_mean: float,
) -> str:
    if (
        float(candidate_val_accuracy_mean) >= float(reference_val_accuracy_mean)
        and float(candidate_test_accuracy_mean) >= float(reference_test_accuracy_mean)
    ):
        return "stronger"
    return "weaker"


def _stage05_v2_vs_stage04_decision(
    rows: list[dict[str, Any]],
    *,
    by_method: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, str]]:
    stage05_rows = _method_rows(rows, STAGE05_V2_METHOD_NAME)
    artifact_pass = all(bool(row["deterministic_artifact_checks_passed"]) for row in stage05_rows)
    one_step_pass = all(float(row["one_step_energy_delta_vs_identity"]) < 0.0 for row in stage05_rows)
    configured_energy_pass = all(
        float(row["configured_step_energy_delta_vs_identity"]) < 0.0 for row in stage05_rows
    )
    configured_residual_pass = all(
        float(row["configured_step_fixed_point_residual_delta_vs_identity"]) < 0.0
        for row in stage05_rows
    )
    stage04 = by_method[STAGE04_METHOD_NAME]
    stage05 = by_method[STAGE05_V2_METHOD_NAME]

    one_step_strength = _mechanism_strength_label(
        candidate_energy_mean=stage05["one_step_energy_delta_vs_identity"]["mean"],
        reference_energy_mean=stage04["one_step_energy_delta_vs_identity"]["mean"],
    )
    configured_step_strength = _mechanism_strength_label(
        candidate_energy_mean=stage05["configured_step_energy_delta_vs_identity"]["mean"],
        reference_energy_mean=stage04["configured_step_energy_delta_vs_identity"]["mean"],
        candidate_residual_mean=stage05["configured_step_fixed_point_residual_delta_vs_identity"]["mean"],
        reference_residual_mean=stage04["configured_step_fixed_point_residual_delta_vs_identity"]["mean"],
    )
    accuracy_strength = _report_accuracy_strength_label(
        candidate_val_accuracy_mean=stage05["val_accuracy"]["mean"],
        reference_val_accuracy_mean=stage04["val_accuracy"]["mean"],
        candidate_test_accuracy_mean=stage05["test_accuracy"]["mean"],
        reference_test_accuracy_mean=stage04["test_accuracy"]["mean"],
    )

    justifies_continued_exploration = bool(
        artifact_pass and one_step_pass and configured_energy_pass and configured_residual_pass
    )
    as_new_reference = bool(justifies_continued_exploration)
    replaces_frozen_bridge = bool(
        justifies_continued_exploration
        and one_step_strength == "stronger"
        and configured_step_strength == "stronger"
        and accuracy_strength == "stronger"
    )

    decision = {
        "stage05_v2_justifies_continued_exploration": bool(justifies_continued_exploration),
        "stage05_v2_as_new_exploratory_reference": bool(as_new_reference),
        "stage05_v2_replaces_frozen_bridge_on_main": bool(replaces_frozen_bridge),
        "stage05_v2_decision_detail": {
            "artifact_checks_all_pass": bool(artifact_pass),
            "one_step_energy_delta_vs_identity_negative_on_every_seed": bool(one_step_pass),
            "configured_step_energy_delta_vs_identity_negative_on_every_seed": bool(
                configured_energy_pass
            ),
            "configured_step_fixed_point_residual_delta_vs_identity_negative_on_every_seed": bool(
                configured_residual_pass
            ),
        },
    }
    comparisons = {
        "one_step_mechanism_vs_stage04": one_step_strength,
        "configured_step_mechanism_vs_stage04": configured_step_strength,
        "report_only_accuracy_vs_stage04": accuracy_strength,
    }
    return decision, comparisons


def _stage05_v2_vs_stage04_decision_rationale(
    *,
    decision: dict[str, Any],
    comparisons: dict[str, str],
) -> str:
    if not bool(decision["stage05_v2_justifies_continued_exploration"]):
        return "Stage 05 v2 keeps positive mechanism signal but does not clear the refreshed multiseed exploration rule."
    if bool(decision["stage05_v2_replaces_frozen_bridge_on_main"]):
        return "Stage 05 v2 clears the refreshed exploration rule and also clears the much stronger replacement rule."
    return (
        "Stage 05 v2 clears the refreshed mechanism-first exploration rule, "
        f"is {comparisons['one_step_mechanism_vs_stage04']} on one-step mechanism, "
        f"is {comparisons['configured_step_mechanism_vs_stage04']} on configured-step mechanism, "
        f"and is {comparisons['report_only_accuracy_vs_stage04']} on report-only accuracy versus the frozen bridge."
    )


def _stage05_v2_vs_stage04_supports_lines(
    *,
    decision: dict[str, Any],
    comparisons: dict[str, str],
    by_method: dict[str, dict[str, Any]],
) -> list[str]:
    stage05 = by_method[STAGE05_V2_METHOD_NAME]
    lines = [
        (
            "Stage 05 v2 artifacts are reproducible under the shared dataset/seed/batch protocol."
            if decision["stage05_v2_decision_detail"]["artifact_checks_all_pass"]
            else "Stage 05 v2 artifacts do not yet fully satisfy the shared artifact protocol."
        ),
        (
            "Stage 05 v2 keeps one-step validation energy delta vs identity negative on every comparison seed."
            if decision["stage05_v2_decision_detail"]["one_step_energy_delta_vs_identity_negative_on_every_seed"]
            else "Stage 05 v2 does not keep one-step validation energy delta vs identity negative on every comparison seed."
        ),
        (
            "Stage 05 v2 keeps configured-step validation energy delta vs identity negative on every comparison seed."
            if decision["stage05_v2_decision_detail"]["configured_step_energy_delta_vs_identity_negative_on_every_seed"]
            else "Stage 05 v2 does not keep configured-step validation energy delta vs identity negative on every comparison seed."
        ),
        (
            "Stage 05 v2 keeps configured-step validation fixed-point residual delta vs identity negative on every comparison seed."
            if decision["stage05_v2_decision_detail"]["configured_step_fixed_point_residual_delta_vs_identity_negative_on_every_seed"]
            else "Stage 05 v2 does not keep configured-step validation fixed-point residual delta vs identity negative on every comparison seed."
        ),
        f"Relative to the frozen bridge, Stage 05 v2 is {comparisons['one_step_mechanism_vs_stage04']} on one-step mechanism.",
        f"Relative to the frozen bridge, Stage 05 v2 is {comparisons['configured_step_mechanism_vs_stage04']} on configured-step mechanism.",
        f"Relative to the frozen bridge, Stage 05 v2 is {comparisons['report_only_accuracy_vs_stage04']} on report-only accuracy.",
    ]
    if bool(decision["stage05_v2_justifies_continued_exploration"]):
        lines.append("The refreshed comparison supports continued Stage 05 mechanism-first exploration.")
    if bool(decision["stage05_v2_as_new_exploratory_reference"]):
        lines.append("The refreshed comparison supports using Stage 05 v2 as the new exploratory reference.")
    lines.append(
        f"Stage 05 v2 mean validation accuracy is {stage05['val_accuracy']['mean']:.6f}; accuracy remains report-only in this comparison."
    )
    return lines


def _stage05_v2_vs_stage04_does_not_support_lines(
    *,
    decision: dict[str, Any],
) -> list[str]:
    lines = [
        "This refreshed comparison supports keeping Stage 04 frozen on main.",
        "This refreshed comparison supports keeping Stage 05 mechanism-first.",
        "This refreshed comparison does not reopen any Stage 04 package-internal work.",
        "This refreshed comparison does not promote task accuracy to the Stage 05 gate.",
    ]
    if not bool(decision["stage05_v2_replaces_frozen_bridge_on_main"]):
        lines.append("This refreshed comparison does not support replacing the frozen Stage 04 bridge on main.")
    return lines


def _stage05_v2_vs_stage04_report_markdown(report: dict[str, Any]) -> str:
    protocol = report["comparison_protocol"]
    decision = report["decision"]
    supports = report["supports"]
    does_not_support = report["does_not_support"]
    lines = [
        "# Frozen Bridge vs Two-Branch Corrected Residual Core",
        "",
        "## Protocol",
        f"- dataset: `{protocol['dataset_name']}`",
        f"- seeds: `{protocol['seeds']}`",
        f"- shared batch size: `{protocol['shared_batch_size']}`",
        f"- shared shuffle_batches: `{protocol['shared_shuffle_batches']}`",
        "",
        "## Decision",
        f"- `stage05_v2_justifies_continued_exploration`: `{decision['stage05_v2_justifies_continued_exploration']}`",
        f"- `stage05_v2_as_new_exploratory_reference`: `{decision['stage05_v2_as_new_exploratory_reference']}`",
        f"- `stage05_v2_replaces_frozen_bridge_on_main`: `{decision['stage05_v2_replaces_frozen_bridge_on_main']}`",
        f"- rationale: `{decision['stage05_v2_vs_stage04_decision_rationale']}`",
        "",
        "## Supports",
    ]
    for item in supports:
        lines.append(f"- {item}")
    lines.extend(["", "## Does Not Support"])
    for item in does_not_support:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _stage05_v2_vs_stage04_suite_config_payload(
    config: FrozenBridgeVsStage05V2ComparisonConfig,
) -> dict[str, Any]:
    return {
        "phase": "FMPC Stage 05 EF Core Probe",
        "stage": "frozen_bridge_vs_two_branch_corrected_residual_core_comparison",
        "comparison_protocol": _stage05_v2_vs_stage04_protocol_payload(config),
        "artifacts": {
            "aggregate_runs_csv": "aggregate_runs.csv",
            "aggregate_summary_json": "aggregate_summary.json",
            "comparison_report_json": "comparison_report.json",
            "comparison_report_md": "comparison_report.md",
        },
    }


def run_frozen_bridge_vs_stage05_v2_comparison(
    config: FrozenBridgeVsStage05V2ComparisonConfig,
) -> FrozenBridgeVsCorrectedCoreComparisonRunResult:
    """Run the refreshed frozen-bridge vs Stage 05 v2 comparison."""

    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, config.resolved_run_id(), config.output_layout)
    )
    _write_json(run_dir / "config.json", _stage05_v2_vs_stage04_suite_config_payload(config))

    rows: list[dict[str, Any]] = []
    runs_root = run_dir / "runs"
    run_index = 0

    for seed in config.seeds:
        run_index += 1
        stage04_config = _stage04_config(config, seed=seed, output_root=runs_root)
        stage04_result = run_fmpc_tf2_experiment(stage04_config)
        rows.append(
            _stage04_row(
                run_index=run_index,
                suite_run_dir=run_dir,
                seed=seed,
                result=stage04_result,
                config=stage04_config,
            )
        )

        run_index += 1
        stage05_config = _stage05_v2_bridge_config(config, seed=seed, output_root=runs_root)
        stage05_result = run_fmpc_ef_exploratory_probe(stage05_config)
        rows.append(
            _stage05_core_row(
                run_index=run_index,
                suite_run_dir=run_dir,
                seed=seed,
                result=stage05_result,
                config=stage05_config,
                method_name=STAGE05_V2_METHOD_NAME,
                stage_name="FMPC Stage 05 EF Core Probe v2",
            )
        )

    csv_rows = [
        {
            **row,
            "deterministic_artifact_checks_passed": str(bool(row["deterministic_artifact_checks_passed"])),
            "mechanism_signal_positive": str(bool(row["mechanism_signal_positive"])),
            "config_json_exists": str(bool(row["config_json_exists"])),
            "summary_json_exists": str(bool(row["summary_json_exists"])),
            "epoch_metrics_csv_exists": str(bool(row["epoch_metrics_csv_exists"])),
            "seed_matches": str(bool(row["seed_matches"])),
            "dataset_matches": str(bool(row["dataset_matches"])),
            "batch_protocol_matches": str(bool(row["batch_protocol_matches"])),
        }
        for row in rows
    ]
    _write_csv(run_dir / "aggregate_runs.csv", csv_rows)

    by_method = {
        STAGE04_METHOD_NAME: _method_summary(_method_rows(rows, STAGE04_METHOD_NAME)),
        STAGE05_V2_METHOD_NAME: _method_summary(_method_rows(rows, STAGE05_V2_METHOD_NAME)),
    }
    pairwise_stage05_v2_vs_stage04 = _pairwise_summary(
        rows,
        candidate_method=STAGE05_V2_METHOD_NAME,
        reference_method=STAGE04_METHOD_NAME,
    )
    decision, comparisons = _stage05_v2_vs_stage04_decision(rows, by_method=by_method)
    decision_rationale = _stage05_v2_vs_stage04_decision_rationale(
        decision=decision,
        comparisons=comparisons,
    )

    summary = {
        "phase": "FMPC Stage 05 EF Core Probe",
        "stage": "frozen_bridge_vs_two_branch_corrected_residual_core_comparison",
        "num_runs": int(len(rows)),
        "comparison_protocol": _stage05_v2_vs_stage04_protocol_payload(config),
        "by_method": by_method,
        "pairwise_stage05_v2_vs_stage04": pairwise_stage05_v2_vs_stage04,
        **decision,
        **comparisons,
        "stage05_v2_vs_stage04_decision_rationale": decision_rationale,
        "aggregate_runs_csv_path": "aggregate_runs.csv",
        "comparison_report_json_path": "comparison_report.json",
        "comparison_report_md_path": "comparison_report.md",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)

    report = {
        "comparison_protocol": _stage05_v2_vs_stage04_protocol_payload(config),
        "decision": {
            **decision,
            **comparisons,
            "stage05_v2_vs_stage04_decision_rationale": decision_rationale,
        },
        "supports": _stage05_v2_vs_stage04_supports_lines(
            decision=decision,
            comparisons=comparisons,
            by_method=by_method,
        ),
        "does_not_support": _stage05_v2_vs_stage04_does_not_support_lines(
            decision=decision,
        ),
    }
    _write_json(run_dir / "comparison_report.json", report)
    _write_text(run_dir / "comparison_report.md", _stage05_v2_vs_stage04_report_markdown(report))

    return FrozenBridgeVsCorrectedCoreComparisonRunResult(
        run_dir=run_dir,
        config=_stage05_v2_vs_stage04_suite_config_payload(config),
        aggregate_rows=rows,
        summary=summary,
        comparison_report=report,
    )


def _stage05_v1_vs_v2_protocol_payload(
    config: CorrectedResidualCoreV1VsV2ComparisonConfig,
) -> dict[str, Any]:
    return {
        "dataset_name": config.dataset_name,
        "seeds": [int(seed) for seed in config.seeds],
        "train_fraction": float(config.train_fraction),
        "val_fraction": float(config.val_fraction),
        "test_fraction": float(config.test_fraction),
        "shared_batch_size": int(config.batch_size),
        "shared_shuffle_batches": bool(config.shuffle_batches),
        "stage_05_v1_reference": {
            "method_name": STAGE05_V1_METHOD_NAME,
            "transport_family": "residual_meanflow_core",
            "residual_branch_structure": "single_branch",
            "configured_transport_steps": int(config.stage05_transport_steps),
            "epochs": int(config.stage05_epochs),
            "eval_steps": int(config.stage05_eval_steps),
            "layer_dims": [int(value) for value in config.stage05_layer_dims],
        },
        "stage_05_v2_candidate": {
            "method_name": STAGE05_V2_METHOD_NAME,
            "transport_family": "two_branch_residual_meanflow_core",
            "residual_branch_structure": "two_branch",
            "configured_transport_steps": int(config.stage05_transport_steps),
            "epochs": int(config.stage05_epochs),
            "eval_steps": int(config.stage05_eval_steps),
            "layer_dims": [int(value) for value in config.stage05_layer_dims],
            "feature_aware_state_branch_tangents": True,
        },
        "decision_rule": {
            "primary_split": "validation",
            "requires_all_stage05_v2_artifact_checks": True,
            "requires_stage05_v2_one_step_energy_delta_vs_identity_negative_on_every_seed": True,
            "requires_stage05_v2_configured_step_energy_delta_vs_identity_negative_on_every_seed": True,
            "requires_stage05_v2_configured_step_fixed_point_residual_delta_vs_identity_negative_on_every_seed": True,
            "requires_stage05_v2_mean_configured_step_energy_delta_vs_identity_more_negative_than_v1": True,
            "requires_stage05_v2_mean_configured_step_fixed_point_residual_delta_vs_identity_more_negative_than_v1": True,
            "task_accuracy_is_report_only": True,
        },
    }


def _stage05_v2_vs_v1_decision(
    rows: list[dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    v1_rows = _method_rows(rows, STAGE05_V1_METHOD_NAME)
    v2_rows = _method_rows(rows, STAGE05_V2_METHOD_NAME)
    artifact_pass = all(bool(row["deterministic_artifact_checks_passed"]) for row in v2_rows)
    one_step_pass = all(float(row["one_step_energy_delta_vs_identity"]) < 0.0 for row in v2_rows)
    configured_energy_pass = all(
        float(row["configured_step_energy_delta_vs_identity"]) < 0.0 for row in v2_rows
    )
    configured_residual_pass = all(
        float(row["configured_step_fixed_point_residual_delta_vs_identity"]) < 0.0 for row in v2_rows
    )
    v1_mean_configured_energy = _mean(
        [float(row["configured_step_energy_delta_vs_identity"]) for row in v1_rows]
    )
    v2_mean_configured_energy = _mean(
        [float(row["configured_step_energy_delta_vs_identity"]) for row in v2_rows]
    )
    v1_mean_configured_residual = _mean(
        [float(row["configured_step_fixed_point_residual_delta_vs_identity"]) for row in v1_rows]
    )
    v2_mean_configured_residual = _mean(
        [float(row["configured_step_fixed_point_residual_delta_vs_identity"]) for row in v2_rows]
    )
    mean_configured_energy_better = bool(v2_mean_configured_energy < v1_mean_configured_energy)
    mean_configured_residual_better = bool(
        v2_mean_configured_residual < v1_mean_configured_residual
    )
    favorable = bool(
        artifact_pass
        and one_step_pass
        and configured_energy_pass
        and configured_residual_pass
        and mean_configured_energy_better
        and mean_configured_residual_better
    )
    return favorable, {
        "artifact_checks_all_pass": bool(artifact_pass),
        "one_step_energy_delta_vs_identity_negative_on_every_seed": bool(one_step_pass),
        "configured_step_energy_delta_vs_identity_negative_on_every_seed": bool(
            configured_energy_pass
        ),
        "configured_step_fixed_point_residual_delta_vs_identity_negative_on_every_seed": bool(
            configured_residual_pass
        ),
        "mean_configured_step_energy_delta_vs_identity_more_negative_than_v1": bool(
            mean_configured_energy_better
        ),
        "mean_configured_step_fixed_point_residual_delta_vs_identity_more_negative_than_v1": bool(
            mean_configured_residual_better
        ),
    }


def _stage05_v1_vs_v2_supports_lines(
    *,
    favorable: bool,
    decision_detail: dict[str, Any],
    by_method: dict[str, dict[str, Any]],
) -> list[str]:
    v1 = by_method[STAGE05_V1_METHOD_NAME]
    v2 = by_method[STAGE05_V2_METHOD_NAME]
    lines = [
        (
            "Stage 05 v2 artifacts are reproducible under the shared dataset/seed/batch protocol."
            if decision_detail["artifact_checks_all_pass"]
            else "Stage 05 v2 artifacts do not yet fully satisfy the shared artifact protocol."
        ),
        (
            "Stage 05 v2 keeps one-step validation energy delta vs identity negative on every comparison seed."
            if decision_detail["one_step_energy_delta_vs_identity_negative_on_every_seed"]
            else "Stage 05 v2 does not keep one-step validation energy delta vs identity negative on every comparison seed."
        ),
        (
            "Stage 05 v2 keeps configured-step validation fixed-point residual delta vs identity negative on every comparison seed."
            if decision_detail["configured_step_fixed_point_residual_delta_vs_identity_negative_on_every_seed"]
            else "Stage 05 v2 does not keep configured-step validation fixed-point residual delta vs identity negative on every comparison seed."
        ),
        (
            "Stage 05 v2 improves mean configured-step validation energy delta vs identity over v1."
            if decision_detail["mean_configured_step_energy_delta_vs_identity_more_negative_than_v1"]
            else "Stage 05 v2 does not improve mean configured-step validation energy delta vs identity over v1."
        ),
        (
            "Stage 05 v2 improves mean configured-step validation fixed-point residual delta vs identity over v1."
            if decision_detail["mean_configured_step_fixed_point_residual_delta_vs_identity_more_negative_than_v1"]
            else "Stage 05 v2 does not improve mean configured-step validation fixed-point residual delta vs identity over v1."
        ),
    ]
    if favorable:
        lines.append(
            "The two-branch corrected residual core is favorable on mechanism-first grounds for the next narrow Stage 05 step."
        )
    else:
        lines.append(
            "The two-branch corrected residual core does not yet improve mechanism magnitude over v1 under the narrow Stage 05 v2 rule."
        )
    lines.append(
        f"Stage 05 v1 mean validation accuracy is {v1['val_accuracy']['mean']:.6f} and Stage 05 v2 mean validation accuracy is {v2['val_accuracy']['mean']:.6f}; accuracy remains report-only."
    )
    return lines


def _stage05_v1_vs_v2_does_not_support_lines(
    *,
    pairwise_v2_vs_v1: dict[str, Any],
) -> list[str]:
    lines = [
        "This comparison does not claim that Stage 05 replaces the frozen Stage 04 bridge result on main.",
        "This comparison does not reopen any Stage 04 package-internal work.",
        "This comparison does not promote task accuracy to a gate.",
    ]
    if (
        float(pairwise_v2_vs_v1["configured_step_energy_delta_vs_identity_delta"]["mean"]) >= 0.0
    ):
        lines.append(
            "Stage 05 v2 does not improve configured-step energy delta vs identity over v1 in this comparison."
        )
    if (
        float(
            pairwise_v2_vs_v1[
                "configured_step_fixed_point_residual_delta_vs_identity_delta"
            ]["mean"]
        )
        >= 0.0
    ):
        lines.append(
            "Stage 05 v2 does not improve configured-step fixed-point residual delta vs identity over v1 in this comparison."
        )
    return lines


def _stage05_v1_vs_v2_report_markdown(report: dict[str, Any]) -> str:
    protocol = report["comparison_protocol"]
    decision = report["decision"]
    supports = report["supports"]
    does_not_support = report["does_not_support"]
    lines = [
        "# Stage 05 Corrected Residual Core v1 vs v2",
        "",
        "## Protocol",
        f"- dataset: `{protocol['dataset_name']}`",
        f"- seeds: `{protocol['seeds']}`",
        f"- shared batch size: `{protocol['shared_batch_size']}`",
        f"- shared shuffle_batches: `{protocol['shared_shuffle_batches']}`",
        "",
        "## Decision",
        f"- `{STAGE05_V2_FAVORABLE_DECISION_NAME}`: `{decision[STAGE05_V2_FAVORABLE_DECISION_NAME]}`",
        f"- rationale: `{decision['decision_rationale']}`",
        "",
        "## Supports",
    ]
    for item in supports:
        lines.append(f"- {item}")
    lines.extend(["", "## Does Not Support"])
    for item in does_not_support:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _stage05_v1_vs_v2_suite_config_payload(
    config: CorrectedResidualCoreV1VsV2ComparisonConfig,
) -> dict[str, Any]:
    return {
        "phase": "FMPC Stage 05 EF Core Probe",
        "stage": "corrected_residual_core_v1_vs_v2_comparison",
        "comparison_protocol": _stage05_v1_vs_v2_protocol_payload(config),
        "artifacts": {
            "aggregate_runs_csv": "aggregate_runs.csv",
            "aggregate_summary_json": "aggregate_summary.json",
            "comparison_report_json": "comparison_report.json",
            "comparison_report_md": "comparison_report.md",
        },
    }


def run_corrected_residual_core_v1_vs_v2_comparison(
    config: CorrectedResidualCoreV1VsV2ComparisonConfig,
) -> FrozenBridgeVsCorrectedCoreComparisonRunResult:
    """Run the formal Stage 05 corrected residual core v1 vs v2 comparison."""

    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, config.resolved_run_id(), config.output_layout)
    )
    _write_json(run_dir / "config.json", _stage05_v1_vs_v2_suite_config_payload(config))

    rows: list[dict[str, Any]] = []
    runs_root = run_dir / "runs"
    run_index = 0

    for seed in config.seeds:
        run_index += 1
        v1_config = _stage05_v1_config(config, seed=seed, output_root=runs_root)
        v1_result = run_fmpc_ef_exploratory_probe(v1_config)
        rows.append(
            _stage05_core_row(
                run_index=run_index,
                suite_run_dir=run_dir,
                seed=seed,
                result=v1_result,
                config=v1_config,
                method_name=STAGE05_V1_METHOD_NAME,
                stage_name="FMPC Stage 05 EF Core Probe v1",
            )
        )

        run_index += 1
        v2_config = _stage05_v2_config(config, seed=seed, output_root=runs_root)
        v2_result = run_fmpc_ef_exploratory_probe(v2_config)
        rows.append(
            _stage05_core_row(
                run_index=run_index,
                suite_run_dir=run_dir,
                seed=seed,
                result=v2_result,
                config=v2_config,
                method_name=STAGE05_V2_METHOD_NAME,
                stage_name="FMPC Stage 05 EF Core Probe v2",
            )
        )

    csv_rows = [
        {
            **row,
            "deterministic_artifact_checks_passed": str(bool(row["deterministic_artifact_checks_passed"])),
            "mechanism_signal_positive": str(bool(row["mechanism_signal_positive"])),
            "config_json_exists": str(bool(row["config_json_exists"])),
            "summary_json_exists": str(bool(row["summary_json_exists"])),
            "epoch_metrics_csv_exists": str(bool(row["epoch_metrics_csv_exists"])),
            "seed_matches": str(bool(row["seed_matches"])),
            "dataset_matches": str(bool(row["dataset_matches"])),
            "batch_protocol_matches": str(bool(row["batch_protocol_matches"])),
        }
        for row in rows
    ]
    _write_csv(run_dir / "aggregate_runs.csv", csv_rows)

    by_method = {
        STAGE05_V1_METHOD_NAME: _method_summary(_method_rows(rows, STAGE05_V1_METHOD_NAME)),
        STAGE05_V2_METHOD_NAME: _method_summary(_method_rows(rows, STAGE05_V2_METHOD_NAME)),
    }
    pairwise_v2_vs_v1 = _pairwise_summary(
        rows,
        candidate_method=STAGE05_V2_METHOD_NAME,
        reference_method=STAGE05_V1_METHOD_NAME,
    )
    favorable, decision_detail = _stage05_v2_vs_v1_decision(rows)
    decision_rationale = (
        "Stage 05 v2 improves mechanism magnitude over v1 under the narrow multiseed rule."
        if favorable
        else "Stage 05 v2 does not yet improve mechanism magnitude over v1 under the narrow multiseed rule."
    )

    summary = {
        "phase": "FMPC Stage 05 EF Core Probe",
        "stage": "corrected_residual_core_v1_vs_v2_comparison",
        "num_runs": int(len(rows)),
        "comparison_protocol": _stage05_v1_vs_v2_protocol_payload(config),
        "by_method": by_method,
        "pairwise_stage05_v2_vs_v1": pairwise_v2_vs_v1,
        STAGE05_V2_FAVORABLE_DECISION_NAME: bool(favorable),
        "stage05_v2_vs_v1_decision_detail": decision_detail,
        "stage05_v2_vs_v1_decision_rationale": decision_rationale,
        "aggregate_runs_csv_path": "aggregate_runs.csv",
        "comparison_report_json_path": "comparison_report.json",
        "comparison_report_md_path": "comparison_report.md",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)

    report = {
        "comparison_protocol": _stage05_v1_vs_v2_protocol_payload(config),
        "decision": {
            STAGE05_V2_FAVORABLE_DECISION_NAME: bool(favorable),
            "decision_detail": decision_detail,
            "decision_rationale": decision_rationale,
        },
        "supports": _stage05_v1_vs_v2_supports_lines(
            favorable=favorable,
            decision_detail=decision_detail,
            by_method=by_method,
        ),
        "does_not_support": _stage05_v1_vs_v2_does_not_support_lines(
            pairwise_v2_vs_v1=pairwise_v2_vs_v1,
        ),
    }
    _write_json(run_dir / "comparison_report.json", report)
    _write_text(run_dir / "comparison_report.md", _stage05_v1_vs_v2_report_markdown(report))

    return FrozenBridgeVsCorrectedCoreComparisonRunResult(
        run_dir=run_dir,
        config=_stage05_v1_vs_v2_suite_config_payload(config),
        aggregate_rows=rows,
        summary=summary,
        comparison_report=report,
    )

def _load_stage05_v2_vs_v3a_contextual_reference(
    config: Stage05V2VsV3AComparisonConfig,
) -> dict[str, Any]:
    summary_path = _resolve_repo_path(config.contextual_reference_summary_path)
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing contextual Stage 05 reference summary at '{summary_path}'."
        )
    payload = _read_json(summary_path)
    by_method = payload.get("by_method")
    if not isinstance(by_method, dict):
        raise ValueError("Contextual Stage 05 reference summary is missing 'by_method'.")
    contextual_summary = by_method.get(STAGE05_V2_BUDGET_PUSH_METHOD_NAME)
    if not isinstance(contextual_summary, dict):
        raise ValueError(
            "Contextual Stage 05 reference summary is missing the stronger-budget Stage 05 v2 method."
        )
    return {
        "source": _repo_relative_posix(summary_path),
        "method_name": STAGE05_V2_BUDGET_PUSH_METHOD_NAME,
        "epochs": int(config.contextual_reference_stage05_epochs),
        "configured_transport_steps": int(
            contextual_summary.get("configured_transport_steps", config.stage05_transport_steps)
        ),
        "one_step_energy_delta_vs_identity": contextual_summary["one_step_energy_delta_vs_identity"],
        "configured_step_energy_delta_vs_identity": contextual_summary[
            "configured_step_energy_delta_vs_identity"
        ],
        "configured_step_fixed_point_residual_delta_vs_identity": contextual_summary[
            "configured_step_fixed_point_residual_delta_vs_identity"
        ],
        "one_step_energy_delta_vs_local_field_only": contextual_summary[
            "one_step_energy_delta_vs_local_field_only"
        ],
        "configured_step_energy_delta_vs_local_field_only": contextual_summary[
            "configured_step_energy_delta_vs_local_field_only"
        ],
        "configured_step_fixed_point_residual_delta_vs_local_field_only": contextual_summary[
            "configured_step_fixed_point_residual_delta_vs_local_field_only"
        ],
        "val_accuracy": contextual_summary["val_accuracy"],
        "test_accuracy": contextual_summary["test_accuracy"],
        "val_output_mse": contextual_summary["val_output_mse"],
        "test_output_mse": contextual_summary["test_output_mse"],
        "selected_epoch": contextual_summary["selected_epoch"],
        "selection_hits_final_training_boundary_rate": float(
            contextual_summary.get("selection_hits_final_training_boundary_rate", 0.0)
        ),
        "runtime_proxy_seconds": contextual_summary["runtime_proxy_seconds"],
    }


def _load_stage05_v2_v3a_v3b_contextual_reference(
    config: Stage05V2V3AV3BComparisonConfig,
) -> dict[str, Any]:
    summary_path = _resolve_repo_path(config.contextual_reference_summary_path)
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing contextual Stage 05 reference summary at '{summary_path}'."
        )
    payload = _read_json(summary_path)
    by_method = payload.get("by_method")
    if not isinstance(by_method, dict):
        raise ValueError("Contextual Stage 05 reference summary is missing 'by_method'.")
    contextual_summary = by_method.get(STAGE05_V2_BUDGET_PUSH_METHOD_NAME)
    if not isinstance(contextual_summary, dict):
        raise ValueError(
            "Contextual Stage 05 reference summary is missing the stronger-budget Stage 05 v2 method."
        )
    return {
        "source": _repo_relative_posix(summary_path),
        "method_name": STAGE05_V2_BUDGET_PUSH_METHOD_NAME,
        "epochs": int(config.contextual_reference_stage05_epochs),
        "configured_transport_steps": int(
            contextual_summary.get("configured_transport_steps", config.stage05_transport_steps)
        ),
        "one_step_energy_delta_vs_identity": contextual_summary["one_step_energy_delta_vs_identity"],
        "configured_step_energy_delta_vs_identity": contextual_summary[
            "configured_step_energy_delta_vs_identity"
        ],
        "configured_step_fixed_point_residual_delta_vs_identity": contextual_summary[
            "configured_step_fixed_point_residual_delta_vs_identity"
        ],
        "one_step_energy_delta_vs_local_field_only": contextual_summary[
            "one_step_energy_delta_vs_local_field_only"
        ],
        "configured_step_energy_delta_vs_local_field_only": contextual_summary[
            "configured_step_energy_delta_vs_local_field_only"
        ],
        "configured_step_fixed_point_residual_delta_vs_local_field_only": contextual_summary[
            "configured_step_fixed_point_residual_delta_vs_local_field_only"
        ],
        "val_accuracy": contextual_summary["val_accuracy"],
        "test_accuracy": contextual_summary["test_accuracy"],
        "val_output_mse": contextual_summary["val_output_mse"],
        "test_output_mse": contextual_summary["test_output_mse"],
        "selected_epoch": contextual_summary["selected_epoch"],
        "selection_hits_final_training_boundary_rate": float(
            contextual_summary.get("selection_hits_final_training_boundary_rate", 0.0)
        ),
        "runtime_proxy_seconds": contextual_summary["runtime_proxy_seconds"],
    }


def _stage05_v2_vs_v3a_protocol_payload(
    config: Stage05V2VsV3AComparisonConfig,
) -> dict[str, Any]:
    if config.comparison_scope == "smoke_only":
        decision_rule = {
            "purpose": "smoke_ready_v3a_candidate_sanity_check",
            "task_accuracy_is_report_only": True,
            "full_fixed_budget_comparison_still_required": True,
        }
    else:
        decision_rule = {
            "purpose": "fixed_budget_v2_vs_v3a_comparison",
            "primary_split": "validation",
            "task_accuracy_is_report_only": True,
            "configured_step_improvement_fraction_threshold": float(
                config.configured_step_improvement_fraction_threshold
            ),
            "allowed_accuracy_regression_threshold": float(
                config.allowed_accuracy_regression_threshold
            ),
            "gap_narrowing_fraction_threshold": float(
                config.gap_narrowing_fraction_threshold
            ),
            "reference_reused_from_existing_artifacts": bool(
                config.reuse_stage05_v2_reference_artifacts
            ),
        }
    return {
        "comparison_scope": str(config.comparison_scope),
        "dataset_name": config.dataset_name,
        "seeds": [int(seed) for seed in config.seeds],
        "train_fraction": float(config.train_fraction),
        "val_fraction": float(config.val_fraction),
        "test_fraction": float(config.test_fraction),
        "shared_batch_size": int(config.batch_size),
        "shared_shuffle_batches": bool(config.shuffle_batches),
        "stage_05_v2_reference": {
            "method_name": STAGE05_V2_METHOD_NAME,
            "candidate_name": "stage05_v2_two_branch_corrected_residual_meanflow_core",
            "transport_family": "two_branch_residual_meanflow_core",
            "residual_branch_structure": "two_branch",
            "feature_aware_state_branch_tangents": True,
            "explicit_transport_drift_decomposition_enabled": False,
            "reference_reused_from_existing_artifacts": bool(
                config.reuse_stage05_v2_reference_artifacts
            ),
            "source_artifact_root": (
                _repo_relative_posix(_resolve_repo_path(config.reference_artifact_root))
                if config.reuse_stage05_v2_reference_artifacts
                else None
            ),
            "configured_transport_steps": int(config.stage05_transport_steps),
            "epochs": int(config.stage05_epochs),
            "eval_steps": int(config.stage05_eval_steps),
            "layer_dims": [int(value) for value in config.stage05_layer_dims],
        },
        "stage_05_v3a_candidate": {
            "method_name": STAGE05_V3A_METHOD_NAME,
            "candidate_name": "stage05_v3a_explicit_transport_drift_contract",
            "transport_family": "two_branch_residual_meanflow_core",
            "residual_branch_structure": "two_branch",
            "feature_aware_state_branch_tangents": True,
            "explicit_transport_drift_decomposition_enabled": True,
            "lambda_drift": float(config.lambda_drift),
            "configured_transport_steps": int(config.stage05_transport_steps),
            "epochs": int(config.stage05_epochs),
            "eval_steps": int(config.stage05_eval_steps),
            "layer_dims": [int(value) for value in config.stage05_layer_dims],
        },
        "contextual_3072_reference": {
            "source_summary_path": _repo_relative_posix(
                _resolve_repo_path(config.contextual_reference_summary_path)
            ),
            "method_name": STAGE05_V2_BUDGET_PUSH_METHOD_NAME,
            "epochs": int(config.contextual_reference_stage05_epochs),
        },
        "decision_rule": decision_rule,
    }


def _stage05_v2_vs_v3a_decision(
    *,
    rows: list[dict[str, Any]],
    config: Stage05V2VsV3AComparisonConfig,
    by_method: dict[str, dict[str, Any]],
    pairwise_v3a_vs_v2: dict[str, Any],
    contextual_reference: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    v2_rows = _method_rows(rows, STAGE05_V2_METHOD_NAME)
    v3a_rows = _method_rows(rows, STAGE05_V3A_METHOD_NAME)
    v2_by_seed = {int(row["seed"]): row for row in v2_rows}
    v3a_by_seed = {int(row["seed"]): row for row in v3a_rows}
    shared_seeds = sorted(set(v2_by_seed).intersection(v3a_by_seed))
    if not shared_seeds:
        raise ValueError("Stage 05 v2 vs v3-A comparison requires shared seeds.")
    artifact_pass = all(bool(row["deterministic_artifact_checks_passed"]) for row in v3a_rows)
    one_step_positive = all(
        float(row["one_step_energy_delta_vs_identity"]) < 0.0 for row in v3a_rows
    )
    mechanism_positive = all(bool(row["mechanism_signal_positive"]) for row in v3a_rows)
    if config.comparison_scope == "smoke_only":
        positive_gap_signal = bool(
            artifact_pass
            and mechanism_positive
            and float(pairwise_v3a_vs_v2["configured_step_energy_delta_vs_identity_delta"]["mean"])
            < 0.0
            and float(
                pairwise_v3a_vs_v2["configured_step_fixed_point_residual_delta_vs_identity_delta"][
                    "mean"
                ]
            )
            < 0.0
        )
        recommended_next_move = (
            "run_fixed_budget_v2_vs_v3a_comparison"
            if artifact_pass
            else "another_implementation_pass"
        )
        rationale = (
            "The smoke-level v3-A candidate is artifact-stable and ready for a fixed-budget v2 vs v3-A comparison."
            if artifact_pass
            else "The smoke-level v3-A candidate still needs another implementation pass before a fixed-budget comparison."
        )
        return {
            STAGE05_V3A_SIGNAL_DECISION_NAME: bool(positive_gap_signal),
            "deterministic_artifact_checks_all_pass": bool(artifact_pass),
            "stage05_v3a_keeps_one_step_mechanism_positive": bool(one_step_positive),
            "stage05_v3a_mechanism_signal_positive_on_all_smoke_runs": bool(mechanism_positive),
            "stage05_v3a_materially_improves_configured_step_mechanism": bool(
                positive_gap_signal
            ),
            "stage05_v3a_avoids_obvious_report_accuracy_regression": True,
            "pairwise_gap_closure_vs_v2": pairwise_v3a_vs_v2,
            "gap_closure_vs_contextual_3072_reference": None,
            "gap_closure_decision": (
                "positive_smoke_signal" if positive_gap_signal else "signal_not_yet_established"
            ),
            "recommended_next_move": recommended_next_move,
        }, rationale

    v2_summary = by_method[STAGE05_V2_METHOD_NAME]
    v3a_summary = by_method[STAGE05_V3A_METHOD_NAME]
    v2_configured_energy_mean = float(
        v2_summary["configured_step_energy_delta_vs_identity"]["mean"]
    )
    v3a_configured_energy_mean = float(
        v3a_summary["configured_step_energy_delta_vs_identity"]["mean"]
    )
    v2_configured_residual_mean = float(
        v2_summary["configured_step_fixed_point_residual_delta_vs_identity"]["mean"]
    )
    v3a_configured_residual_mean = float(
        v3a_summary["configured_step_fixed_point_residual_delta_vs_identity"]["mean"]
    )
    configured_energy_gain_fraction = _negative_magnitude_relative_gain(
        current_value=v2_configured_energy_mean,
        candidate_value=v3a_configured_energy_mean,
    )
    configured_residual_gain_fraction = _negative_magnitude_relative_gain(
        current_value=v2_configured_residual_mean,
        candidate_value=v3a_configured_residual_mean,
    )
    configured_energy_seed_improvement_rate = _rate(
        [
            float(v3a_by_seed[seed]["configured_step_energy_delta_vs_identity"])
            < float(v2_by_seed[seed]["configured_step_energy_delta_vs_identity"])
            for seed in shared_seeds
        ]
    )
    configured_residual_seed_improvement_rate = _rate(
        [
            float(v3a_by_seed[seed]["configured_step_fixed_point_residual_delta_vs_identity"])
            < float(v2_by_seed[seed]["configured_step_fixed_point_residual_delta_vs_identity"])
            for seed in shared_seeds
        ]
    )
    materially_improves_configured_step = bool(
        configured_energy_gain_fraction
        >= float(config.configured_step_improvement_fraction_threshold)
        and configured_residual_gain_fraction
        >= float(config.configured_step_improvement_fraction_threshold)
        and configured_energy_seed_improvement_rate >= 0.5
        and configured_residual_seed_improvement_rate >= 0.5
    )

    val_accuracy_delta = float(pairwise_v3a_vs_v2["val_accuracy_delta"]["mean"])
    test_accuracy_delta = float(pairwise_v3a_vs_v2["test_accuracy_delta"]["mean"])
    avoids_obvious_accuracy_regression = bool(
        val_accuracy_delta >= -float(config.allowed_accuracy_regression_threshold)
        and test_accuracy_delta >= -float(config.allowed_accuracy_regression_threshold)
    )

    contextual_energy_gap_closed_fraction = _negative_gap_closed_fraction(
        reference_value=v2_configured_energy_mean,
        candidate_value=v3a_configured_energy_mean,
        target_value=float(contextual_reference["configured_step_energy_delta_vs_identity"]["mean"]),
    )
    contextual_residual_gap_closed_fraction = _negative_gap_closed_fraction(
        reference_value=v2_configured_residual_mean,
        candidate_value=v3a_configured_residual_mean,
        target_value=float(
            contextual_reference["configured_step_fixed_point_residual_delta_vs_identity"]["mean"]
        ),
    )
    contextual_val_accuracy_gap_closed_fraction = _positive_gap_closed_fraction(
        reference_value=float(v2_summary["val_accuracy"]["mean"]),
        candidate_value=float(v3a_summary["val_accuracy"]["mean"]),
        target_value=float(contextual_reference["val_accuracy"]["mean"]),
    )
    contextual_test_accuracy_gap_closed_fraction = _positive_gap_closed_fraction(
        reference_value=float(v2_summary["test_accuracy"]["mean"]),
        candidate_value=float(v3a_summary["test_accuracy"]["mean"]),
        target_value=float(contextual_reference["test_accuracy"]["mean"]),
    )
    positive_gap_signal = bool(
        materially_improves_configured_step
        and (
            contextual_energy_gap_closed_fraction > 0.0
            or contextual_residual_gap_closed_fraction > 0.0
        )
    )
    material_gap_signal = bool(
        materially_improves_configured_step
        and contextual_energy_gap_closed_fraction
        >= float(config.gap_narrowing_fraction_threshold)
        and contextual_residual_gap_closed_fraction
        >= float(config.gap_narrowing_fraction_threshold)
    )

    if not artifact_pass or not one_step_positive or not mechanism_positive:
        recommended_next_move = "reject_v3a_and_return_to_chartering"
        gap_closure_decision = "fixed_budget_regression_or_instability"
        rationale = (
            "The fixed-budget v3-A candidate does not preserve the minimum artifact or mechanism stability requirements, "
            "so this contract should not be advanced without re-chartering."
        )
    elif material_gap_signal and avoids_obvious_accuracy_regression:
        recommended_next_move = "proceed_to_stage05_v3b_curriculum_charter"
        gap_closure_decision = "material_positive_gap_closure_signal"
        rationale = (
            "The fixed-budget v3-A candidate materially improves configured-step mechanism over v2, narrows the gap to "
            "the contextual 3072-epoch reference, and does not show an obvious report-only accuracy regression."
        )
    else:
        recommended_next_move = "keep_v3a_and_refine_implementation"
        gap_closure_decision = (
            "positive_but_not_yet_material_gap_closure_signal"
            if positive_gap_signal
            else "no_material_gap_closure_signal"
        )
        rationale = (
            "The fixed-budget v3-A candidate is stable enough to keep as an active implementation branch, but it does "
            "not yet show a strong enough configured-step gap-closure signal over v2 to justify moving on to v3-B."
        )

    return {
        STAGE05_V3A_SIGNAL_DECISION_NAME: bool(positive_gap_signal),
        "deterministic_artifact_checks_all_pass": bool(artifact_pass),
        "stage05_v3a_keeps_one_step_mechanism_positive": bool(one_step_positive),
        "stage05_v3a_mechanism_signal_positive_on_all_runs": bool(mechanism_positive),
        "stage05_v3a_materially_improves_configured_step_mechanism": bool(
            materially_improves_configured_step
        ),
        "stage05_v3a_avoids_obvious_report_accuracy_regression": bool(
            avoids_obvious_accuracy_regression
        ),
        "configured_step_gain_fraction_vs_v2_reference": {
            "energy": float(configured_energy_gain_fraction),
            "residual": float(configured_residual_gain_fraction),
        },
        "configured_step_seed_improvement_rate_vs_v2_reference": {
            "energy": float(configured_energy_seed_improvement_rate),
            "residual": float(configured_residual_seed_improvement_rate),
        },
        "gap_closure_vs_contextual_3072_reference": {
            "configured_step_energy": float(contextual_energy_gap_closed_fraction),
            "configured_step_residual": float(contextual_residual_gap_closed_fraction),
            "val_accuracy": float(contextual_val_accuracy_gap_closed_fraction),
            "test_accuracy": float(contextual_test_accuracy_gap_closed_fraction),
        },
        "pairwise_gap_closure_vs_v2": pairwise_v3a_vs_v2,
        "gap_closure_decision": str(gap_closure_decision),
        "recommended_next_move": str(recommended_next_move),
    }, rationale


def _stage05_v2_vs_v3a_supports_lines(
    *,
    summary: dict[str, Any],
) -> list[str]:
    pairwise = summary["pairwise_deltas_vs_stage05_v2_reference"]
    lines = [
        "The Stage 05 v3-A candidate path writes the standard Stage 05 artifacts.",
        "The v3-A candidate keeps artifact-independent target construction and the existing aggregate residual identity target.",
        "The comparison exposes explicit pairwise deltas versus the current Stage 05 v2 reference.",
    ]
    if summary["comparison_scope"] == "smoke_only":
        if bool(summary["deterministic_artifact_checks_all_pass"]):
            lines.append("The v3-A smoke run passes deterministic artifact checks.")
        return lines
    lines.extend(
        [
            (
                "The fixed-budget v3-A candidate keeps one-step validation energy delta vs identity negative on every seed."
                if bool(summary["stage05_v3a_keeps_one_step_mechanism_positive"])
                else "The fixed-budget v3-A candidate does not keep one-step validation energy delta vs identity negative on every seed."
            ),
            (
                "The fixed-budget v3-A candidate materially improves configured-step mechanism over the v2 reference."
                if bool(summary["stage05_v3a_materially_improves_configured_step_mechanism"])
                else "The fixed-budget v3-A candidate does not materially improve configured-step mechanism over the v2 reference."
            ),
            (
                "The fixed-budget v3-A candidate avoids an obvious report-only accuracy regression."
                if bool(summary["stage05_v3a_avoids_obvious_report_accuracy_regression"])
                else "The fixed-budget v3-A candidate shows an obvious report-only accuracy regression."
            ),
            f"Pairwise configured-step validation energy delta vs identity mean difference vs v2: {pairwise['configured_step_energy_delta_vs_identity_delta']['mean']:.12f}.",
            f"Pairwise configured-step validation fixed-point residual delta vs identity mean difference vs v2: {pairwise['configured_step_fixed_point_residual_delta_vs_identity_delta']['mean']:.12f}.",
        ]
    )
    return lines


def _stage05_v2_vs_v3a_does_not_support_lines(
    *,
    summary: dict[str, Any],
) -> list[str]:
    lines = [
        "This comparison does not justify replacing the frozen Stage 04 bridge on main.",
        "This comparison does not promote task accuracy to the Stage 05 gate.",
    ]
    if summary["comparison_scope"] == "smoke_only":
        lines.insert(0, "This smoke comparison does not establish a formal fixed-budget mechanism win.")
        return lines
    lines.append("This comparison does not reopen Stage 04 package-internal work.")
    if not bool(summary[STAGE05_V3A_SIGNAL_DECISION_NAME]):
        lines.append(
            "The current fixed-budget evidence does not yet show a positive gap-closure signal strong enough to move beyond v3-A."
        )
    return lines


def _stage05_v2_vs_v3a_report_markdown(report: dict[str, Any]) -> str:
    protocol = report["comparison_protocol"]
    decision = report["decision"]
    pairwise = report["pairwise_deltas_vs_stage05_v2_reference"]
    lines = [
        "# Stage 05 v2 vs v3-A Explicit Transport-Drift Contract",
        "",
        "## Protocol",
        f"- comparison scope: `{protocol['comparison_scope']}`",
        f"- dataset: `{protocol['dataset_name']}`",
        f"- seeds: `{protocol['seeds']}`",
        f"- shared batch size: `{protocol['shared_batch_size']}`",
        f"- shared shuffle_batches: `{protocol['shared_shuffle_batches']}`",
        f"- Stage 05 epochs: `{protocol['stage_05_v3a_candidate']['epochs']}`",
        "",
        "## Decision",
        f"- `{STAGE05_V3A_SIGNAL_DECISION_NAME}`: `{decision[STAGE05_V3A_SIGNAL_DECISION_NAME]}`",
        f"- `stage05_v3a_materially_improves_configured_step_mechanism`: `{decision['stage05_v3a_materially_improves_configured_step_mechanism']}`",
        f"- `stage05_v3a_avoids_obvious_report_accuracy_regression`: `{decision['stage05_v3a_avoids_obvious_report_accuracy_regression']}`",
        f"- gap_closure_decision: `{decision['gap_closure_decision']}`",
        f"- recommended next move: `{decision['recommended_next_move']}`",
        f"- rationale: `{decision['decision_rationale']}`",
        "",
        "## Pairwise Deltas Vs V2",
        f"- one-step validation energy delta vs identity delta: `{pairwise['one_step_energy_delta_vs_identity_delta']['mean']}`",
        f"- configured-step validation energy delta vs identity delta: `{pairwise['configured_step_energy_delta_vs_identity_delta']['mean']}`",
        f"- configured-step validation fixed-point residual delta vs identity delta: `{pairwise['configured_step_fixed_point_residual_delta_vs_identity_delta']['mean']}`",
        f"- validation accuracy delta: `{pairwise['val_accuracy_delta']['mean']}`",
        f"- test accuracy delta: `{pairwise['test_accuracy_delta']['mean']}`",
        "",
        "## Supports",
    ]
    for item in report["supports"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Does Not Support"])
    for item in report["does_not_support"]:
        lines.append(f"- {item}")
    if report.get("contextual_3072_reference") is not None:
        contextual = report["contextual_3072_reference"]
        lines.extend(
            [
                "",
                "## Contextual 3072 Reference",
                f"- source: `{contextual['source']}`",
                f"- configured-step validation energy delta vs identity mean: `{contextual['configured_step_energy_delta_vs_identity']['mean']}`",
                f"- configured-step validation fixed-point residual delta vs identity mean: `{contextual['configured_step_fixed_point_residual_delta_vs_identity']['mean']}`",
                f"- validation/test accuracy means: `{contextual['val_accuracy']['mean']}` / `{contextual['test_accuracy']['mean']}`",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _stage05_v2_vs_v3a_suite_config_payload(
    config: Stage05V2VsV3AComparisonConfig,
) -> dict[str, Any]:
    return {
        "phase": "FMPC Stage 05 EF Core Probe",
        "stage": str(config.experiment_name),
        "comparison_protocol": _stage05_v2_vs_v3a_protocol_payload(config),
        "artifacts": {
            "aggregate_runs_csv": "aggregate_runs.csv",
            "aggregate_summary_json": "aggregate_summary.json",
            "comparison_report_json": "comparison_report.json",
            "comparison_report_md": "comparison_report.md",
        },
    }


def run_stage05_v2_vs_v3a_comparison(
    config: Stage05V2VsV3AComparisonConfig,
) -> FrozenBridgeVsCorrectedCoreComparisonRunResult:
    """Run a Stage 05 v2 vs v3-A comparison at smoke or fixed-budget scope."""

    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, config.resolved_run_id(), config.output_layout)
    )
    _write_json(run_dir / "config.json", _stage05_v2_vs_v3a_suite_config_payload(config))

    rows: list[dict[str, Any]] = []
    runs_root = run_dir / "runs"
    run_index = 0

    for seed in config.seeds:
        if config.reuse_stage05_v2_reference_artifacts:
            run_index += 1
            existing_run_dir = _resolve_repo_path(config.reference_artifact_root) / f"seed_{seed}"
            rows.append(
                _load_existing_stage05_core_row(
                    run_index=run_index,
                    existing_run_dir=existing_run_dir,
                    seed=seed,
                    expected_dataset_name=config.dataset_name,
                    expected_batch_size=int(config.batch_size),
                    expected_shuffle_batches=bool(config.shuffle_batches),
                    method_name=STAGE05_V2_METHOD_NAME,
                    stage_name="FMPC Stage 05 EF Core Probe v2 Reference",
                )
            )
        else:
            run_index += 1
            v2_config = _build_stage05_v2_config(
                output_root=runs_root,
                experiment_name=STAGE05_V2_METHOD_NAME,
                seed=seed,
                train_fraction=float(config.train_fraction),
                val_fraction=float(config.val_fraction),
                test_fraction=float(config.test_fraction),
                batch_size=int(config.batch_size),
                shuffle_batches=bool(config.shuffle_batches),
                epochs=int(config.stage05_epochs),
                eval_steps=int(config.stage05_eval_steps),
                layer_dims=config.stage05_layer_dims,
                transport_steps=int(config.stage05_transport_steps),
            )
            v2_result = run_fmpc_ef_exploratory_probe(v2_config)
            rows.append(
                _stage05_core_row(
                    run_index=run_index,
                    suite_run_dir=run_dir,
                    seed=seed,
                    result=v2_result,
                    config=v2_config,
                    method_name=STAGE05_V2_METHOD_NAME,
                    stage_name="FMPC Stage 05 EF Core Probe v2 Reference",
                )
            )

        run_index += 1
        v3a_config = _stage05_v3a_config(config, seed=seed, output_root=runs_root)
        v3a_result = run_fmpc_ef_exploratory_probe(v3a_config)
        rows.append(
            _stage05_core_row(
                run_index=run_index,
                suite_run_dir=run_dir,
                seed=seed,
                result=v3a_result,
                config=v3a_config,
                method_name=STAGE05_V3A_METHOD_NAME,
                stage_name="FMPC Stage 05 EF Core Probe v3-A Explicit Transport-Drift Candidate",
            )
        )

    csv_rows = [
        {
            **row,
            "deterministic_artifact_checks_passed": str(bool(row["deterministic_artifact_checks_passed"])),
            "mechanism_signal_positive": str(bool(row["mechanism_signal_positive"])),
            "config_json_exists": str(bool(row["config_json_exists"])),
            "summary_json_exists": str(bool(row["summary_json_exists"])),
            "epoch_metrics_csv_exists": str(bool(row["epoch_metrics_csv_exists"])),
            "seed_matches": str(bool(row["seed_matches"])),
            "dataset_matches": str(bool(row["dataset_matches"])),
            "batch_protocol_matches": str(bool(row["batch_protocol_matches"])),
            "selection_hits_final_training_boundary": str(
                bool(row["selection_hits_final_training_boundary"])
            ),
        }
        for row in rows
    ]
    _write_csv(run_dir / "aggregate_runs.csv", csv_rows)

    by_method = {
        STAGE05_V2_METHOD_NAME: _method_summary(_method_rows(rows, STAGE05_V2_METHOD_NAME)),
        STAGE05_V3A_METHOD_NAME: _method_summary(_method_rows(rows, STAGE05_V3A_METHOD_NAME)),
    }
    pairwise_v3a_vs_v2 = _pairwise_summary(
        rows,
        candidate_method=STAGE05_V3A_METHOD_NAME,
        reference_method=STAGE05_V2_METHOD_NAME,
    )
    contextual_reference = _load_stage05_v2_vs_v3a_contextual_reference(config)
    decision, decision_rationale = _stage05_v2_vs_v3a_decision(
        rows=rows,
        config=config,
        by_method=by_method,
        pairwise_v3a_vs_v2=pairwise_v3a_vs_v2,
        contextual_reference=contextual_reference,
    )

    summary = {
        "phase": "FMPC Stage 05 EF Core Probe",
        "stage": str(config.experiment_name),
        "comparison_scope": str(config.comparison_scope),
        "num_runs": int(len(rows)),
        "comparison_protocol": _stage05_v2_vs_v3a_protocol_payload(config),
        "by_method": by_method,
        "pairwise_stage05_v3a_vs_v2": pairwise_v3a_vs_v2,
        "pairwise_deltas_vs_stage05_v2_reference": pairwise_v3a_vs_v2,
        "contextual_3072_reference": contextual_reference,
        **decision,
        "decision_rationale": decision_rationale,
        "aggregate_runs_csv_path": "aggregate_runs.csv",
        "comparison_report_json_path": "comparison_report.json",
        "comparison_report_md_path": "comparison_report.md",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)

    report = {
        "comparison_protocol": _stage05_v2_vs_v3a_protocol_payload(config),
        "decision": {
            **decision,
            "decision_rationale": decision_rationale,
        },
        "pairwise_deltas_vs_stage05_v2_reference": pairwise_v3a_vs_v2,
        "contextual_3072_reference": contextual_reference,
        "supports": _stage05_v2_vs_v3a_supports_lines(summary=summary),
        "does_not_support": _stage05_v2_vs_v3a_does_not_support_lines(summary=summary),
    }
    _write_json(run_dir / "comparison_report.json", report)
    _write_text(run_dir / "comparison_report.md", _stage05_v2_vs_v3a_report_markdown(report))

    return FrozenBridgeVsCorrectedCoreComparisonRunResult(
        run_dir=run_dir,
        config=_stage05_v2_vs_v3a_suite_config_payload(config),
        aggregate_rows=rows,
        summary=summary,
        comparison_report=report,
    )


def _stage05_v2_v3a_v3b_protocol_payload(
    config: Stage05V2V3AV3BComparisonConfig,
) -> dict[str, Any]:
    if config.comparison_scope == "smoke_only":
        decision_rule = {
            "purpose": "smoke_ready_v3b_scaffold_check",
            "task_accuracy_is_report_only": True,
            "artifact_checks_required": True,
            "one_step_mechanism_should_remain_positive": True,
            "configured_step_mechanism_should_remain_positive": True,
            "smoke_only": True,
        }
    else:
        decision_rule = {
            "purpose": "fixed_budget_v2_vs_v3a_vs_v3b_comparison",
            "primary_split": "validation",
            "task_accuracy_is_report_only": True,
            "configured_step_improvement_fraction_threshold": float(
                config.configured_step_improvement_fraction_threshold
            ),
            "allowed_accuracy_regression_threshold": float(
                config.allowed_accuracy_regression_threshold
            ),
            "gap_narrowing_fraction_threshold": float(
                config.gap_narrowing_fraction_threshold
            ),
            "gap_closure_gain_threshold": float(config.gap_closure_gain_threshold),
            "reuse_stage05_v2_reference_artifacts": bool(
                config.reuse_stage05_v2_reference_artifacts
            ),
            "reuse_stage05_v3a_reference_artifacts": bool(
                config.reuse_stage05_v3a_reference_artifacts
            ),
            "reuse_stage05_v3b_candidate_artifacts": bool(
                config.reuse_stage05_v3b_candidate_artifacts
            ),
        }
    return {
        "comparison_scope": str(config.comparison_scope),
        "dataset_name": config.dataset_name,
        "seeds": [int(seed) for seed in config.seeds],
        "train_fraction": float(config.train_fraction),
        "val_fraction": float(config.val_fraction),
        "test_fraction": float(config.test_fraction),
        "shared_batch_size": int(config.batch_size),
        "shared_shuffle_batches": bool(config.shuffle_batches),
        "stage_05_v2_reference": {
            "method_name": STAGE05_V2_METHOD_NAME,
            "candidate_name": "stage05_v2_two_branch_corrected_residual_meanflow_core",
            "transport_family": "two_branch_residual_meanflow_core",
            "residual_branch_structure": "two_branch",
            "explicit_transport_drift_decomposition_enabled": False,
            "trajectory_curriculum_enabled": False,
            "reference_reused_from_existing_artifacts": bool(
                config.reuse_stage05_v2_reference_artifacts
            ),
            "source_artifact_root": (
                _repo_relative_posix(_resolve_repo_path(config.reference_artifact_root))
                if config.reuse_stage05_v2_reference_artifacts
                else None
            ),
            "configured_transport_steps": int(config.stage05_transport_steps),
            "epochs": int(config.stage05_epochs),
            "eval_steps": int(config.stage05_eval_steps),
            "layer_dims": [int(value) for value in config.stage05_layer_dims],
        },
        "stage_05_v3a_reference": {
            "method_name": STAGE05_V3A_METHOD_NAME,
            "candidate_name": "stage05_v3a_explicit_transport_drift_contract",
            "transport_family": "two_branch_residual_meanflow_core",
            "residual_branch_structure": "two_branch",
            "explicit_transport_drift_decomposition_enabled": True,
            "trajectory_curriculum_enabled": False,
            "reference_reused_from_existing_artifacts": bool(
                config.reuse_stage05_v3a_reference_artifacts
            ),
            "source_artifact_root": (
                _repo_relative_posix(_resolve_repo_path(config.v3a_reference_artifact_root))
                if config.reuse_stage05_v3a_reference_artifacts
                else None
            ),
            "configured_transport_steps": int(config.stage05_transport_steps),
            "epochs": int(config.stage05_epochs),
            "eval_steps": int(config.stage05_eval_steps),
            "layer_dims": [int(value) for value in config.stage05_layer_dims],
        },
        "stage_05_v3b_candidate": {
            "method_name": str(config.v3b_candidate_method_name),
            "candidate_name": str(config.v3b_candidate_method_name),
            "transport_family": "two_branch_residual_meanflow_core",
            "residual_branch_structure": "two_branch",
            "explicit_transport_drift_decomposition_enabled": True,
            "trajectory_curriculum_enabled": True,
            "trajectory_curriculum_schedule_identity": "warmup_sigmoid_to_alpha_floor",
            "alpha_floor": float(config.alpha_floor),
            "lambda_traj_curr": float(config.lambda_traj_curr),
            "reference_reused_from_existing_artifacts": bool(
                config.reuse_stage05_v3b_candidate_artifacts
            ),
            "source_artifact_root": (
                _repo_relative_posix(_resolve_repo_path(config.v3b_candidate_artifact_root))
                if config.reuse_stage05_v3b_candidate_artifacts
                else None
            ),
            "configured_transport_steps": int(config.stage05_transport_steps),
            "epochs": int(config.stage05_epochs),
            "eval_steps": int(config.stage05_eval_steps),
            "layer_dims": [int(value) for value in config.stage05_layer_dims],
        },
        "contextual_3072_reference": {
            "source_summary_path": _repo_relative_posix(
                _resolve_repo_path(config.contextual_reference_summary_path)
            ),
            "method_name": STAGE05_V2_BUDGET_PUSH_METHOD_NAME,
            "epochs": int(config.contextual_reference_stage05_epochs),
        },
        "decision_rule": decision_rule,
    }


def _stage05_v2_v3a_v3b_suite_config_payload(
    config: Stage05V2V3AV3BComparisonConfig,
) -> dict[str, Any]:
    return {
        "phase": "FMPC Stage 05 EF Core Probe",
        "stage": str(config.experiment_name),
        "comparison_protocol": _stage05_v2_v3a_v3b_protocol_payload(config),
        "artifacts": {
            "aggregate_runs_csv": "aggregate_runs.csv",
            "aggregate_summary_json": "aggregate_summary.json",
            "comparison_report_json": "comparison_report.json",
            "comparison_report_md": "comparison_report.md",
        },
    }


def _stage05_v2_v3a_v3b_decision(
    *,
    rows: list[dict[str, Any]],
    config: Stage05V2V3AV3BComparisonConfig,
    by_method: dict[str, dict[str, Any]],
    pairwise_v3a_vs_v2: dict[str, Any],
    pairwise_v3b_vs_v2: dict[str, Any],
    pairwise_v3b_vs_v3a: dict[str, Any],
    contextual_reference: dict[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    candidate_method_name = str(config.v3b_candidate_method_name)
    promoted_candidate = candidate_method_name != STAGE05_V3B_METHOD_NAME
    v2_rows = _method_rows(rows, STAGE05_V2_METHOD_NAME)
    v3a_rows = _method_rows(rows, STAGE05_V3A_METHOD_NAME)
    v3b_rows = _method_rows(rows, candidate_method_name)
    v2_by_seed = {int(row["seed"]): row for row in v2_rows}
    v3a_by_seed = {int(row["seed"]): row for row in v3a_rows}
    v3b_by_seed = {int(row["seed"]): row for row in v3b_rows}
    shared_seeds = sorted(set(v2_by_seed).intersection(v3a_by_seed).intersection(v3b_by_seed))
    if not shared_seeds:
        raise ValueError("Stage 05 v2/v3-A/v3-B comparison requires shared seeds.")

    artifact_pass = all(bool(row["deterministic_artifact_checks_passed"]) for row in v3b_rows)
    one_step_positive = all(
        float(row["one_step_energy_delta_vs_identity"]) < 0.0 for row in v3b_rows
    )
    mechanism_positive = all(bool(row["mechanism_signal_positive"]) for row in v3b_rows)

    if config.comparison_scope == "smoke_only":
        positive_gap_signal = bool(
            artifact_pass
            and mechanism_positive
            and float(pairwise_v3b_vs_v3a["configured_step_energy_delta_vs_identity_delta"]["mean"])
            < 0.0
            and float(
                pairwise_v3b_vs_v3a["configured_step_fixed_point_residual_delta_vs_identity_delta"][
                    "mean"
                ]
            )
            < 0.0
        )
        recommended_next_move = (
            "run_fixed_budget_v2_vs_v3a_vs_v3b_comparison"
            if artifact_pass
            else "another_v3b_implementation_pass"
        )
        rationale = (
            "The minimal v3-B candidate writes deterministic artifacts and keeps mechanism metrics positive in smoke scope."
            if artifact_pass
            else "The minimal v3-B candidate does not yet clear the smoke-ready artifact or mechanism checks."
        )
        decision = {
            "deterministic_artifact_checks_all_pass": bool(artifact_pass),
            "stage05_v3b_keeps_one_step_mechanism_positive": bool(one_step_positive),
            "stage05_v3b_mechanism_signal_positive_on_all_runs": bool(mechanism_positive),
            "stage05_v3b_improves_configured_step_mechanism_vs_v2": bool(
                float(pairwise_v3b_vs_v2["configured_step_energy_delta_vs_identity_delta"]["mean"])
                < 0.0
                and float(
                    pairwise_v3b_vs_v2["configured_step_fixed_point_residual_delta_vs_identity_delta"][
                        "mean"
                    ]
                )
                < 0.0
            ),
            "stage05_v3b_materially_improves_configured_step_mechanism_vs_v3a": bool(
                positive_gap_signal
            ),
            "stage05_v3b_avoids_obvious_report_accuracy_regression": True,
            "stage05_v3b_shows_positive_gap_closure_signal_vs_v3a": bool(positive_gap_signal),
            "contextual_gap_closure_fractions_vs_3072_reference": None,
            "gap_closure_decision": (
                "positive_smoke_signal_vs_v3a"
                if positive_gap_signal
                else "signal_not_yet_established"
            ),
            "recommended_next_move": recommended_next_move,
        }
        if promoted_candidate:
            decision.update(
                {
                    "promoted_refined_v3b_candidate_name": candidate_method_name,
                    "promoted_refined_v3b_materially_beats_v3a": bool(positive_gap_signal),
                    "promoted_refined_v3b_avoids_obvious_report_accuracy_regression": True,
                }
            )
        return decision, rationale

    if contextual_reference is None:
        raise ValueError("Fixed-budget Stage 05 v2/v3-A/v3-B comparison requires contextual reference.")

    v2_summary = by_method[STAGE05_V2_METHOD_NAME]
    v3a_summary = by_method[STAGE05_V3A_METHOD_NAME]
    v3b_summary = by_method[candidate_method_name]

    v2_configured_energy_mean = float(v2_summary["configured_step_energy_delta_vs_identity"]["mean"])
    v3a_configured_energy_mean = float(v3a_summary["configured_step_energy_delta_vs_identity"]["mean"])
    v3b_configured_energy_mean = float(v3b_summary["configured_step_energy_delta_vs_identity"]["mean"])
    v2_configured_residual_mean = float(
        v2_summary["configured_step_fixed_point_residual_delta_vs_identity"]["mean"]
    )
    v3a_configured_residual_mean = float(
        v3a_summary["configured_step_fixed_point_residual_delta_vs_identity"]["mean"]
    )
    v3b_configured_residual_mean = float(
        v3b_summary["configured_step_fixed_point_residual_delta_vs_identity"]["mean"]
    )

    configured_energy_gain_fraction_vs_v3a = _negative_magnitude_relative_gain(
        current_value=v3a_configured_energy_mean,
        candidate_value=v3b_configured_energy_mean,
    )
    configured_residual_gain_fraction_vs_v3a = _negative_magnitude_relative_gain(
        current_value=v3a_configured_residual_mean,
        candidate_value=v3b_configured_residual_mean,
    )
    configured_energy_gain_fraction_vs_v2 = _negative_magnitude_relative_gain(
        current_value=v2_configured_energy_mean,
        candidate_value=v3b_configured_energy_mean,
    )
    configured_residual_gain_fraction_vs_v2 = _negative_magnitude_relative_gain(
        current_value=v2_configured_residual_mean,
        candidate_value=v3b_configured_residual_mean,
    )

    configured_energy_seed_improvement_rate_vs_v3a = _rate(
        [
            float(v3b_by_seed[seed]["configured_step_energy_delta_vs_identity"])
            < float(v3a_by_seed[seed]["configured_step_energy_delta_vs_identity"])
            for seed in shared_seeds
        ]
    )
    configured_residual_seed_improvement_rate_vs_v3a = _rate(
        [
            float(v3b_by_seed[seed]["configured_step_fixed_point_residual_delta_vs_identity"])
            < float(v3a_by_seed[seed]["configured_step_fixed_point_residual_delta_vs_identity"])
            for seed in shared_seeds
        ]
    )
    configured_energy_seed_improvement_rate_vs_v2 = _rate(
        [
            float(v3b_by_seed[seed]["configured_step_energy_delta_vs_identity"])
            < float(v2_by_seed[seed]["configured_step_energy_delta_vs_identity"])
            for seed in shared_seeds
        ]
    )
    configured_residual_seed_improvement_rate_vs_v2 = _rate(
        [
            float(v3b_by_seed[seed]["configured_step_fixed_point_residual_delta_vs_identity"])
            < float(v2_by_seed[seed]["configured_step_fixed_point_residual_delta_vs_identity"])
            for seed in shared_seeds
        ]
    )

    materially_improves_vs_v3a = bool(
        configured_energy_gain_fraction_vs_v3a
        >= float(config.configured_step_improvement_fraction_threshold)
        and configured_residual_gain_fraction_vs_v3a
        >= float(config.configured_step_improvement_fraction_threshold)
        and configured_energy_seed_improvement_rate_vs_v3a >= 0.5
        and configured_residual_seed_improvement_rate_vs_v3a >= 0.5
    )
    improves_vs_v2 = bool(
        configured_energy_gain_fraction_vs_v2
        >= float(config.configured_step_improvement_fraction_threshold)
        and configured_residual_gain_fraction_vs_v2
        >= float(config.configured_step_improvement_fraction_threshold)
        and configured_energy_seed_improvement_rate_vs_v2 >= 0.5
        and configured_residual_seed_improvement_rate_vs_v2 >= 0.5
    )

    val_accuracy_delta_vs_v3a = float(pairwise_v3b_vs_v3a["val_accuracy_delta"]["mean"])
    test_accuracy_delta_vs_v3a = float(pairwise_v3b_vs_v3a["test_accuracy_delta"]["mean"])
    avoids_obvious_accuracy_regression = bool(
        val_accuracy_delta_vs_v3a >= -float(config.allowed_accuracy_regression_threshold)
        and test_accuracy_delta_vs_v3a >= -float(config.allowed_accuracy_regression_threshold)
    )

    v3a_contextual_energy_gap = _negative_gap_closed_fraction(
        reference_value=v2_configured_energy_mean,
        candidate_value=v3a_configured_energy_mean,
        target_value=float(contextual_reference["configured_step_energy_delta_vs_identity"]["mean"]),
    )
    v3a_contextual_residual_gap = _negative_gap_closed_fraction(
        reference_value=v2_configured_residual_mean,
        candidate_value=v3a_configured_residual_mean,
        target_value=float(
            contextual_reference["configured_step_fixed_point_residual_delta_vs_identity"]["mean"]
        ),
    )
    v3a_contextual_val_accuracy_gap = _positive_gap_closed_fraction(
        reference_value=float(v2_summary["val_accuracy"]["mean"]),
        candidate_value=float(v3a_summary["val_accuracy"]["mean"]),
        target_value=float(contextual_reference["val_accuracy"]["mean"]),
    )
    v3a_contextual_test_accuracy_gap = _positive_gap_closed_fraction(
        reference_value=float(v2_summary["test_accuracy"]["mean"]),
        candidate_value=float(v3a_summary["test_accuracy"]["mean"]),
        target_value=float(contextual_reference["test_accuracy"]["mean"]),
    )
    v3b_contextual_energy_gap = _negative_gap_closed_fraction(
        reference_value=v2_configured_energy_mean,
        candidate_value=v3b_configured_energy_mean,
        target_value=float(contextual_reference["configured_step_energy_delta_vs_identity"]["mean"]),
    )
    v3b_contextual_residual_gap = _negative_gap_closed_fraction(
        reference_value=v2_configured_residual_mean,
        candidate_value=v3b_configured_residual_mean,
        target_value=float(
            contextual_reference["configured_step_fixed_point_residual_delta_vs_identity"]["mean"]
        ),
    )
    v3b_contextual_val_accuracy_gap = _positive_gap_closed_fraction(
        reference_value=float(v2_summary["val_accuracy"]["mean"]),
        candidate_value=float(v3b_summary["val_accuracy"]["mean"]),
        target_value=float(contextual_reference["val_accuracy"]["mean"]),
    )
    v3b_contextual_test_accuracy_gap = _positive_gap_closed_fraction(
        reference_value=float(v2_summary["test_accuracy"]["mean"]),
        candidate_value=float(v3b_summary["test_accuracy"]["mean"]),
        target_value=float(contextual_reference["test_accuracy"]["mean"]),
    )

    contextual_energy_gap_gain_vs_v3a = float(v3b_contextual_energy_gap - v3a_contextual_energy_gap)
    contextual_residual_gap_gain_vs_v3a = float(
        v3b_contextual_residual_gap - v3a_contextual_residual_gap
    )
    positive_gap_signal_vs_v3a = bool(
        materially_improves_vs_v3a
        and (
            contextual_energy_gap_gain_vs_v3a > 0.0
            or contextual_residual_gap_gain_vs_v3a > 0.0
        )
    )
    material_gap_signal_vs_v3a = bool(
        materially_improves_vs_v3a
        and contextual_energy_gap_gain_vs_v3a >= float(config.gap_closure_gain_threshold)
        and contextual_residual_gap_gain_vs_v3a >= float(config.gap_closure_gain_threshold)
    )

    if promoted_candidate:
        materially_beats_v3a = bool(
            artifact_pass
            and one_step_positive
            and mechanism_positive
            and materially_improves_vs_v3a
            and avoids_obvious_accuracy_regression
        )
        if materially_beats_v3a:
            recommended_next_move = "promote_refined_v3b_as_active_reference"
            gap_closure_decision = "promoted_refined_v3b_materially_beats_v3a"
            rationale = (
                f"The promoted refined v3-B candidate `{candidate_method_name}` materially improves configured-step "
                "mechanism over the active fixed-budget v3-A reference, preserves the minimum mechanism checks, and "
                "does not show an obvious report-only accuracy regression."
            )
        else:
            recommended_next_move = "retain_v3a_as_active_reference"
            gap_closure_decision = "promoted_refined_v3b_does_not_materially_beat_v3a"
            rationale = (
                f"The promoted refined v3-B candidate `{candidate_method_name}` does not yet clear the fixed-budget "
                "materiality rule over v3-A strongly enough to displace it as the active improvement reference."
            )
    else:
        if not artifact_pass or not one_step_positive or not mechanism_positive:
            recommended_next_move = "retain_v3a_as_active_reference"
            gap_closure_decision = "fixed_budget_regression_or_instability"
            rationale = (
                "The fixed-budget v3-B candidate does not preserve the minimum artifact or mechanism stability requirements, "
                "so v3-A should remain the active fixed-budget reference."
            )
        elif material_gap_signal_vs_v3a and improves_vs_v2 and avoids_obvious_accuracy_regression:
            recommended_next_move = "proceed_to_stage05_v3c_charter"
            gap_closure_decision = "material_positive_gap_closure_signal_vs_v3a"
            rationale = (
                "The fixed-budget v3-B candidate materially improves configured-step mechanism over v3-A, also clears the "
                "same configured-step materiality rule over the fixed-budget v2 control, and shows a clearly stronger "
                "gap-closure signal relative to the contextual 3072-epoch v2 reference."
            )
        elif positive_gap_signal_vs_v3a:
            recommended_next_move = "keep_v3b_and_refine_implementation"
            gap_closure_decision = "positive_but_not_yet_material_gap_closure_signal_vs_v3a"
            rationale = (
                "The fixed-budget v3-B candidate improves configured-step mechanism over v3-A and nudges gap closure in the "
                "right direction, but not yet strongly enough to justify opening v3-C."
            )
        else:
            recommended_next_move = "retain_v3a_as_active_reference"
            gap_closure_decision = "no_material_gap_closure_signal_vs_v3a"
            rationale = (
                "The fixed-budget v3-B candidate does not yet show a strong enough configured-step advantage over v3-A to "
                "replace it as the active fixed-budget improvement reference."
            )

    decision = {
        "deterministic_artifact_checks_all_pass": bool(artifact_pass),
        "stage05_v3b_keeps_one_step_mechanism_positive": bool(one_step_positive),
        "stage05_v3b_mechanism_signal_positive_on_all_runs": bool(mechanism_positive),
        "stage05_v3b_improves_configured_step_mechanism_vs_v2": bool(improves_vs_v2),
        "stage05_v3b_materially_improves_configured_step_mechanism_vs_v3a": bool(
            materially_improves_vs_v3a
        ),
        "stage05_v3b_avoids_obvious_report_accuracy_regression": bool(
            avoids_obvious_accuracy_regression
        ),
        "stage05_v3b_shows_positive_gap_closure_signal_vs_v3a": bool(
            positive_gap_signal_vs_v3a
        ),
        "configured_step_gain_fraction_vs_stage05_v3a_reference": {
            "energy": float(configured_energy_gain_fraction_vs_v3a),
            "residual": float(configured_residual_gain_fraction_vs_v3a),
        },
        "configured_step_gain_fraction_vs_stage05_v2_control": {
            "energy": float(configured_energy_gain_fraction_vs_v2),
            "residual": float(configured_residual_gain_fraction_vs_v2),
        },
        "configured_step_seed_improvement_rate_vs_stage05_v3a_reference": {
            "energy": float(configured_energy_seed_improvement_rate_vs_v3a),
            "residual": float(configured_residual_seed_improvement_rate_vs_v3a),
        },
        "configured_step_seed_improvement_rate_vs_stage05_v2_control": {
            "energy": float(configured_energy_seed_improvement_rate_vs_v2),
            "residual": float(configured_residual_seed_improvement_rate_vs_v2),
        },
        "contextual_gap_closure_fractions_vs_3072_reference": {
            "stage05_v3a": {
                "configured_step_energy": float(v3a_contextual_energy_gap),
                "configured_step_residual": float(v3a_contextual_residual_gap),
                "val_accuracy": float(v3a_contextual_val_accuracy_gap),
                "test_accuracy": float(v3a_contextual_test_accuracy_gap),
            },
            "stage05_v3b": {
                "configured_step_energy": float(v3b_contextual_energy_gap),
                "configured_step_residual": float(v3b_contextual_residual_gap),
                "val_accuracy": float(v3b_contextual_val_accuracy_gap),
                "test_accuracy": float(v3b_contextual_test_accuracy_gap),
            },
            "v3b_minus_v3a": {
                "configured_step_energy": float(contextual_energy_gap_gain_vs_v3a),
                "configured_step_residual": float(contextual_residual_gap_gain_vs_v3a),
                "val_accuracy": float(v3b_contextual_val_accuracy_gap - v3a_contextual_val_accuracy_gap),
                "test_accuracy": float(v3b_contextual_test_accuracy_gap - v3a_contextual_test_accuracy_gap),
            },
        },
        "gap_closure_decision": str(gap_closure_decision),
        "recommended_next_move": str(recommended_next_move),
    }
    if promoted_candidate:
        decision.update(
            {
                "promoted_refined_v3b_candidate_name": candidate_method_name,
                "promoted_refined_v3b_materially_beats_v3a": bool(
                    artifact_pass
                    and one_step_positive
                    and mechanism_positive
                    and materially_improves_vs_v3a
                    and avoids_obvious_accuracy_regression
                ),
                "promoted_refined_v3b_avoids_obvious_report_accuracy_regression": bool(
                    avoids_obvious_accuracy_regression
                ),
                "promoted_refined_v3b_replaces_v3a_as_active_reference": bool(
                    recommended_next_move == "promote_refined_v3b_as_active_reference"
                ),
            }
        )
    return decision, rationale


def _stage05_v2_v3a_v3b_supports_lines(
    *,
    summary: dict[str, Any],
) -> list[str]:
    candidate_name = str(summary["comparison_protocol"]["stage_05_v3b_candidate"]["candidate_name"])
    lines = [
        "The three-way comparison preserves the Stage 05 mechanism-first contract.",
        "The comparison exposes explicit pairwise deltas versus both the fixed-budget v2 control and the fixed-budget v3-A reference.",
        f"The explicit Stage 05 candidate `{candidate_name}` keeps transport-drift decomposition and makes trajectory curriculum identity explicit in artifacts.",
    ]
    if summary["comparison_scope"] == "smoke_only":
        if bool(summary["deterministic_artifact_checks_all_pass"]):
            lines.append("The v3-B smoke run passes deterministic artifact checks.")
        return lines
    pairwise_v3b_vs_v3a = summary["pairwise_deltas_vs_stage05_v3a_reference"]
    lines.extend(
        [
            (
                "The fixed-budget v3-B candidate keeps one-step validation energy delta vs identity negative on every seed."
                if bool(summary["stage05_v3b_keeps_one_step_mechanism_positive"])
                else "The fixed-budget v3-B candidate does not keep one-step validation energy delta vs identity negative on every seed."
            ),
            (
                f"The fixed-budget candidate `{candidate_name}` materially improves configured-step mechanism over v3-A."
                if bool(summary["stage05_v3b_materially_improves_configured_step_mechanism_vs_v3a"])
                else f"The fixed-budget candidate `{candidate_name}` does not materially improve configured-step mechanism over v3-A."
            ),
            (
                f"The fixed-budget candidate `{candidate_name}` avoids an obvious report-only accuracy regression relative to v3-A."
                if bool(summary["stage05_v3b_avoids_obvious_report_accuracy_regression"])
                else f"The fixed-budget candidate `{candidate_name}` shows an obvious report-only accuracy regression relative to v3-A."
            ),
            f"Pairwise configured-step validation energy delta vs identity mean difference vs v3-A: {pairwise_v3b_vs_v3a['configured_step_energy_delta_vs_identity_delta']['mean']:.12f}.",
            f"Pairwise configured-step validation fixed-point residual delta vs identity mean difference vs v3-A: {pairwise_v3b_vs_v3a['configured_step_fixed_point_residual_delta_vs_identity_delta']['mean']:.12f}.",
        ]
    )
    return lines


def _stage05_v2_v3a_v3b_does_not_support_lines(
    *,
    summary: dict[str, Any],
) -> list[str]:
    return [
        "This comparison does not justify replacing the frozen Stage 04 bridge on main.",
        "This comparison does not promote task accuracy to the Stage 05 gate.",
        "This comparison does not reopen Stage 04 package-internal work.",
    ]


def _stage05_v2_v3a_v3b_report_markdown(report: dict[str, Any]) -> str:
    protocol = report["comparison_protocol"]
    decision = report["decision"]
    candidate_name = str(protocol["stage_05_v3b_candidate"]["candidate_name"])
    lines = [
        "# Stage 05 v2 vs v3-A vs Promoted v3-B Comparison",
        "",
        "## Protocol",
        f"- comparison scope: `{protocol['comparison_scope']}`",
        f"- dataset: `{protocol['dataset_name']}`",
        f"- seeds: `{protocol['seeds']}`",
        f"- shared batch size: `{protocol['shared_batch_size']}`",
        f"- Stage 05 epochs: `{protocol['stage_05_v3b_candidate']['epochs']}`",
        f"- promoted candidate: `{candidate_name}`",
        "",
        "## Decision",
        f"- `stage05_v3b_keeps_one_step_mechanism_positive`: `{decision['stage05_v3b_keeps_one_step_mechanism_positive']}`",
        f"- `stage05_v3b_materially_improves_configured_step_mechanism_vs_v3a`: `{decision['stage05_v3b_materially_improves_configured_step_mechanism_vs_v3a']}`",
        f"- `stage05_v3b_shows_positive_gap_closure_signal_vs_v3a`: `{decision['stage05_v3b_shows_positive_gap_closure_signal_vs_v3a']}`",
        f"- gap_closure_decision: `{decision['gap_closure_decision']}`",
        f"- recommended next move: `{decision['recommended_next_move']}`",
        f"- rationale: `{decision['decision_rationale']}`",
        "",
        "## Pairwise Deltas Vs V2",
        f"- candidate: `{candidate_name}`",
        f"- configured-step validation energy delta vs identity delta: `{report['pairwise_deltas_vs_stage05_v2_reference']['configured_step_energy_delta_vs_identity_delta']['mean']}`",
        f"- configured-step validation fixed-point residual delta vs identity delta: `{report['pairwise_deltas_vs_stage05_v2_reference']['configured_step_fixed_point_residual_delta_vs_identity_delta']['mean']}`",
        "",
        "## Pairwise Deltas Vs V3-A",
        f"- candidate: `{candidate_name}`",
        f"- configured-step validation energy delta vs identity delta: `{report['pairwise_deltas_vs_stage05_v3a_reference']['configured_step_energy_delta_vs_identity_delta']['mean']}`",
        f"- configured-step validation fixed-point residual delta vs identity delta: `{report['pairwise_deltas_vs_stage05_v3a_reference']['configured_step_fixed_point_residual_delta_vs_identity_delta']['mean']}`",
    ]
    if "promoted_refined_v3b_materially_beats_v3a" in decision:
        lines.extend(
            [
                "",
                "## Promoted Candidate Decision",
                f"- `promoted_refined_v3b_materially_beats_v3a`: `{decision['promoted_refined_v3b_materially_beats_v3a']}`",
                f"- `promoted_refined_v3b_avoids_obvious_report_accuracy_regression`: `{decision['promoted_refined_v3b_avoids_obvious_report_accuracy_regression']}`",
                f"- `promoted_refined_v3b_replaces_v3a_as_active_reference`: `{decision['promoted_refined_v3b_replaces_v3a_as_active_reference']}`",
            ]
        )
    contextual_gap = report.get("contextual_gap_closure_fractions_vs_3072_reference")
    if isinstance(contextual_gap, dict):
        lines.extend(
            [
                "",
                "## Contextual 3072 Gap Closure",
                f"- v3-A configured-step energy gap closure: `{contextual_gap['stage05_v3a']['configured_step_energy']}`",
                f"- candidate configured-step energy gap closure: `{contextual_gap['stage05_v3b']['configured_step_energy']}`",
                f"- v3-A configured-step residual gap closure: `{contextual_gap['stage05_v3a']['configured_step_residual']}`",
                f"- candidate configured-step residual gap closure: `{contextual_gap['stage05_v3b']['configured_step_residual']}`",
            ]
        )
    return "\n".join(lines)


def run_stage05_v2_v3a_v3b_comparison(
    config: Stage05V2V3AV3BComparisonConfig,
) -> FrozenBridgeVsCorrectedCoreComparisonRunResult:
    """Run a Stage 05 three-way comparison across the fixed-budget v2 control, v3-A, and v3-B."""

    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, config.resolved_run_id(), config.output_layout)
    )
    _write_json(run_dir / "config.json", _stage05_v2_v3a_v3b_suite_config_payload(config))

    rows: list[dict[str, Any]] = []
    runs_root = run_dir / "runs"
    run_index = 0
    candidate_method_name = str(config.v3b_candidate_method_name)
    reuse_v2_root = _resolve_repo_path(config.reference_artifact_root)
    reuse_v3a_root = _resolve_repo_path(config.v3a_reference_artifact_root)
    reuse_v3b_root = _resolve_repo_path(config.v3b_candidate_artifact_root)

    if config.reuse_stage05_v2_reference_artifacts and not reuse_v2_root.exists():
        raise FileNotFoundError(
            f"Missing Stage 05 v2 reference artifacts at '{reuse_v2_root}'."
        )
    if config.reuse_stage05_v3a_reference_artifacts and not reuse_v3a_root.exists():
        raise FileNotFoundError(
            f"Missing Stage 05 v3-A reference artifacts at '{reuse_v3a_root}'."
        )
    if config.reuse_stage05_v3b_candidate_artifacts and not reuse_v3b_root.exists():
        raise FileNotFoundError(
            f"Missing Stage 05 v3-B candidate artifacts at '{reuse_v3b_root}'."
        )

    for seed in config.seeds:
        run_index += 1
        if config.reuse_stage05_v2_reference_artifacts:
            existing_v2_run_dir = reuse_v2_root / f"seed_{seed}"
            rows.append(
                _load_existing_stage05_core_row(
                    run_index=run_index,
                    existing_run_dir=existing_v2_run_dir,
                    seed=seed,
                    expected_dataset_name=config.dataset_name,
                    expected_batch_size=int(config.batch_size),
                    expected_shuffle_batches=bool(config.shuffle_batches),
                    method_name=STAGE05_V2_METHOD_NAME,
                    stage_name="FMPC Stage 05 EF Core Probe v2 Reference",
                )
            )
        else:
            v2_config = _build_stage05_v2_config(
                output_root=runs_root,
                experiment_name=STAGE05_V2_METHOD_NAME,
                seed=seed,
                train_fraction=float(config.train_fraction),
                val_fraction=float(config.val_fraction),
                test_fraction=float(config.test_fraction),
                batch_size=int(config.batch_size),
                shuffle_batches=bool(config.shuffle_batches),
                epochs=int(config.stage05_epochs),
                eval_steps=int(config.stage05_eval_steps),
                layer_dims=config.stage05_layer_dims,
                transport_steps=int(config.stage05_transport_steps),
            )
            v2_result = run_fmpc_ef_exploratory_probe(v2_config)
            rows.append(
                _stage05_core_row(
                    run_index=run_index,
                    suite_run_dir=run_dir,
                    seed=seed,
                    result=v2_result,
                    config=v2_config,
                    method_name=STAGE05_V2_METHOD_NAME,
                    stage_name="FMPC Stage 05 EF Core Probe v2 Reference",
                )
            )

        run_index += 1
        if config.reuse_stage05_v3a_reference_artifacts:
            existing_v3a_run_dir = reuse_v3a_root / f"seed_{seed}"
            rows.append(
                _load_existing_stage05_core_row(
                    run_index=run_index,
                    existing_run_dir=existing_v3a_run_dir,
                    seed=seed,
                    expected_dataset_name=config.dataset_name,
                    expected_batch_size=int(config.batch_size),
                    expected_shuffle_batches=bool(config.shuffle_batches),
                    method_name=STAGE05_V3A_METHOD_NAME,
                    stage_name="FMPC Stage 05 EF Core Probe v3-A Explicit Transport-Drift Candidate",
                )
            )
        else:
            v3a_config = _stage05_v3a_config(
                Stage05V2VsV3AComparisonConfig(
                    output_root=config.output_root,
                    run_id=config.run_id,
                    output_layout=config.output_layout,
                    comparison_scope=config.comparison_scope,
                    dataset_name=config.dataset_name,
                    seeds=config.seeds,
                    train_fraction=config.train_fraction,
                    val_fraction=config.val_fraction,
                    test_fraction=config.test_fraction,
                    batch_size=config.batch_size,
                    shuffle_batches=config.shuffle_batches,
                    stage05_epochs=config.stage05_epochs,
                    stage05_eval_steps=config.stage05_eval_steps,
                    stage05_layer_dims=config.stage05_layer_dims,
                    stage05_transport_steps=config.stage05_transport_steps,
                    lambda_drift=config.lambda_drift,
                ),
                seed=seed,
                output_root=runs_root,
            )
            v3a_result = run_fmpc_ef_exploratory_probe(v3a_config)
            rows.append(
                _stage05_core_row(
                    run_index=run_index,
                    suite_run_dir=run_dir,
                    seed=seed,
                    result=v3a_result,
                    config=v3a_config,
                    method_name=STAGE05_V3A_METHOD_NAME,
                    stage_name="FMPC Stage 05 EF Core Probe v3-A Explicit Transport-Drift Candidate",
                )
            )

        run_index += 1
        if config.reuse_stage05_v3b_candidate_artifacts:
            existing_v3b_run_dir = reuse_v3b_root / f"seed_{seed}"
            rows.append(
                _load_existing_stage05_core_row(
                    run_index=run_index,
                    existing_run_dir=existing_v3b_run_dir,
                    seed=seed,
                    expected_dataset_name=config.dataset_name,
                    expected_batch_size=int(config.batch_size),
                    expected_shuffle_batches=bool(config.shuffle_batches),
                    method_name=candidate_method_name,
                    stage_name=_stage05_v3b_candidate_stage_name(config.v3b_candidate_method_name),
                )
            )
        else:
            v3b_config = _stage05_v3b_config(config, seed=seed, output_root=runs_root)
            v3b_result = run_fmpc_ef_exploratory_probe(v3b_config)
            rows.append(
                _stage05_core_row(
                    run_index=run_index,
                    suite_run_dir=run_dir,
                    seed=seed,
                    result=v3b_result,
                    config=v3b_config,
                    method_name=candidate_method_name,
                    stage_name=_stage05_v3b_candidate_stage_name(config.v3b_candidate_method_name),
                )
            )

    csv_rows = [
        {
            **row,
            "deterministic_artifact_checks_passed": str(bool(row["deterministic_artifact_checks_passed"])),
            "mechanism_signal_positive": str(bool(row["mechanism_signal_positive"])),
            "config_json_exists": str(bool(row["config_json_exists"])),
            "summary_json_exists": str(bool(row["summary_json_exists"])),
            "epoch_metrics_csv_exists": str(bool(row["epoch_metrics_csv_exists"])),
            "seed_matches": str(bool(row["seed_matches"])),
            "dataset_matches": str(bool(row["dataset_matches"])),
            "batch_protocol_matches": str(bool(row["batch_protocol_matches"])),
            "selection_hits_final_training_boundary": str(
                bool(row["selection_hits_final_training_boundary"])
            ),
        }
        for row in rows
    ]
    _write_csv(run_dir / "aggregate_runs.csv", csv_rows)

    by_method = {
        STAGE05_V2_METHOD_NAME: _method_summary(_method_rows(rows, STAGE05_V2_METHOD_NAME)),
        STAGE05_V3A_METHOD_NAME: _method_summary(_method_rows(rows, STAGE05_V3A_METHOD_NAME)),
        candidate_method_name: _method_summary(_method_rows(rows, candidate_method_name)),
    }
    pairwise_v3a_vs_v2 = _pairwise_summary(
        rows,
        candidate_method=STAGE05_V3A_METHOD_NAME,
        reference_method=STAGE05_V2_METHOD_NAME,
    )
    pairwise_v3b_vs_v2 = _pairwise_summary(
        rows,
        candidate_method=candidate_method_name,
        reference_method=STAGE05_V2_METHOD_NAME,
    )
    pairwise_v3b_vs_v3a = _pairwise_summary(
        rows,
        candidate_method=candidate_method_name,
        reference_method=STAGE05_V3A_METHOD_NAME,
    )
    contextual_reference = (
        None
        if config.comparison_scope == "smoke_only"
        else _load_stage05_v2_v3a_v3b_contextual_reference(config)
    )
    decision, decision_rationale = _stage05_v2_v3a_v3b_decision(
        rows=rows,
        config=config,
        by_method=by_method,
        pairwise_v3a_vs_v2=pairwise_v3a_vs_v2,
        pairwise_v3b_vs_v2=pairwise_v3b_vs_v2,
        pairwise_v3b_vs_v3a=pairwise_v3b_vs_v3a,
        contextual_reference=contextual_reference,
    )
    summary = {
        "phase": "FMPC Stage 05 EF Core Probe",
        "stage": str(config.experiment_name),
        "comparison_scope": str(config.comparison_scope),
        "num_runs": int(len(rows)),
        "promoted_v3b_candidate_name": candidate_method_name,
        "comparison_protocol": _stage05_v2_v3a_v3b_protocol_payload(config),
        "by_method": by_method,
        "pairwise_stage05_v3a_vs_v2": pairwise_v3a_vs_v2,
        "pairwise_stage05_v3b_vs_v2": pairwise_v3b_vs_v2,
        "pairwise_stage05_v3b_vs_v3a": pairwise_v3b_vs_v3a,
        "pairwise_deltas_vs_stage05_v2_reference": pairwise_v3b_vs_v2,
        "pairwise_deltas_vs_stage05_v3a_reference": pairwise_v3b_vs_v3a,
        "pairwise_promoted_refined_v3b_vs_v2": pairwise_v3b_vs_v2,
        "pairwise_promoted_refined_v3b_vs_v3a": pairwise_v3b_vs_v3a,
        "contextual_3072_reference": contextual_reference,
        **decision,
        "decision_rationale": decision_rationale,
        "aggregate_runs_csv_path": "aggregate_runs.csv",
        "comparison_report_json_path": "comparison_report.json",
        "comparison_report_md_path": "comparison_report.md",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)

    report = {
        "comparison_protocol": _stage05_v2_v3a_v3b_protocol_payload(config),
        "decision": {
            **decision,
            "decision_rationale": decision_rationale,
        },
        "promoted_v3b_candidate_name": candidate_method_name,
        "pairwise_stage05_v3a_vs_v2": pairwise_v3a_vs_v2,
        "pairwise_deltas_vs_stage05_v2_reference": pairwise_v3b_vs_v2,
        "pairwise_deltas_vs_stage05_v3a_reference": pairwise_v3b_vs_v3a,
        "pairwise_promoted_refined_v3b_vs_v2": pairwise_v3b_vs_v2,
        "pairwise_promoted_refined_v3b_vs_v3a": pairwise_v3b_vs_v3a,
        "contextual_3072_reference": contextual_reference,
        "contextual_gap_closure_fractions_vs_3072_reference": decision.get(
            "contextual_gap_closure_fractions_vs_3072_reference"
        ),
        "supports": _stage05_v2_v3a_v3b_supports_lines(summary=summary),
        "does_not_support": _stage05_v2_v3a_v3b_does_not_support_lines(summary=summary),
    }
    _write_json(run_dir / "comparison_report.json", report)
    _write_text(run_dir / "comparison_report.md", _stage05_v2_v3a_v3b_report_markdown(report))

    return FrozenBridgeVsCorrectedCoreComparisonRunResult(
        run_dir=run_dir,
        config=_stage05_v2_v3a_v3b_suite_config_payload(config),
        aggregate_rows=rows,
        summary=summary,
        comparison_report=report,
    )


def _stage05_v2_promoted_v3b_v3c_protocol_payload(
    config: Stage05V2PromotedV3BV3CComparisonConfig,
) -> dict[str, Any]:
    if config.comparison_scope == "smoke_only":
        decision_rule = {
            "purpose": "smoke_ready_v3c_semigroup_probe_check",
            "task_accuracy_is_report_only": True,
            "artifact_checks_required": True,
            "one_step_mechanism_should_remain_positive": True,
            "configured_step_mechanism_should_remain_positive": True,
            "smoke_only": True,
        }
    else:
        decision_rule = {
            "purpose": "fixed_budget_v2_vs_promoted_v3b_vs_v3c_comparison",
            "primary_split": "validation",
            "task_accuracy_is_report_only": True,
            "configured_step_improvement_fraction_threshold": float(
                config.configured_step_improvement_fraction_threshold
            ),
            "allowed_accuracy_regression_threshold": float(
                config.allowed_accuracy_regression_threshold
            ),
            "gap_narrowing_fraction_threshold": float(config.gap_narrowing_fraction_threshold),
            "reuse_stage05_v2_reference_artifacts": bool(
                config.reuse_stage05_v2_reference_artifacts
            ),
            "reuse_stage05_v3b_reference_artifacts": bool(
                config.reuse_stage05_v3b_reference_artifacts
            ),
            "reuse_stage05_v3c_candidate_artifacts": bool(
                config.reuse_stage05_v3c_candidate_artifacts
            ),
        }
    return {
        "comparison_scope": str(config.comparison_scope),
        "dataset_name": config.dataset_name,
        "seeds": [int(seed) for seed in config.seeds],
        "train_fraction": float(config.train_fraction),
        "val_fraction": float(config.val_fraction),
        "test_fraction": float(config.test_fraction),
        "shared_batch_size": int(config.batch_size),
        "shared_shuffle_batches": bool(config.shuffle_batches),
        "stage_05_v2_control": {
            "method_name": STAGE05_V2_METHOD_NAME,
            "candidate_name": STAGE05_V2_METHOD_NAME,
            "transport_family": "two_branch_residual_meanflow_core",
            "explicit_transport_drift_decomposition_enabled": False,
            "trajectory_curriculum_enabled": False,
            "endpoint_semigroup_consistency_enabled": False,
            "reference_reused_from_existing_artifacts": bool(
                config.reuse_stage05_v2_reference_artifacts
            ),
            "source_artifact_root": (
                _repo_relative_posix(_resolve_repo_path(config.reference_artifact_root))
                if config.reuse_stage05_v2_reference_artifacts
                else None
            ),
            "configured_transport_steps": int(config.stage05_transport_steps),
            "epochs": int(config.stage05_epochs),
            "eval_steps": int(config.stage05_eval_steps),
            "layer_dims": [int(value) for value in config.stage05_layer_dims],
        },
        "stage_05_promoted_v3b_reference": {
            "method_name": STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME,
            "candidate_name": STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME,
            "transport_family": "two_branch_residual_meanflow_core",
            "explicit_transport_drift_decomposition_enabled": True,
            "trajectory_curriculum_enabled": True,
            "endpoint_semigroup_consistency_enabled": False,
            "trajectory_curriculum_schedule_identity": "warmup_sigmoid_to_alpha_floor",
            "alpha_floor": float(config.promoted_v3b_alpha_floor),
            "lambda_traj_curr": float(config.promoted_v3b_lambda_traj_curr),
            "reference_reused_from_existing_artifacts": bool(
                config.reuse_stage05_v3b_reference_artifacts
            ),
            "source_artifact_root": (
                _repo_relative_posix(_resolve_repo_path(config.v3b_reference_artifact_root))
                if config.reuse_stage05_v3b_reference_artifacts
                else None
            ),
            "configured_transport_steps": int(config.stage05_transport_steps),
            "epochs": int(config.stage05_epochs),
            "eval_steps": int(config.stage05_eval_steps),
            "layer_dims": [int(value) for value in config.stage05_layer_dims],
        },
        "stage_05_v3c_candidate": {
            "method_name": str(config.v3c_candidate_method_name),
            "candidate_name": str(config.v3c_candidate_method_name),
            "transport_family": "two_branch_residual_meanflow_core",
            "explicit_transport_drift_decomposition_enabled": True,
            "trajectory_curriculum_enabled": True,
            "endpoint_semigroup_consistency_enabled": True,
            "trajectory_curriculum_schedule_identity": "warmup_sigmoid_to_alpha_floor",
            "alpha_floor": float(config.promoted_v3b_alpha_floor),
            "lambda_traj_curr": float(config.promoted_v3b_lambda_traj_curr),
            "lambda_sg": float(config.lambda_sg),
            "semigroup_split_identity": "s = t + alpha * r; r_s = (1 - alpha) * r",
            "semigroup_target_mode": "single_sided_detached_split_endpoint",
            "semigroup_target_is_single_sided_detached": True,
            "reference_reused_from_existing_artifacts": bool(
                config.reuse_stage05_v3c_candidate_artifacts
            ),
            "source_artifact_root": (
                _repo_relative_posix(_resolve_repo_path(config.v3c_candidate_artifact_root))
                if config.reuse_stage05_v3c_candidate_artifacts
                else None
            ),
            "configured_transport_steps": int(config.stage05_transport_steps),
            "epochs": int(config.stage05_epochs),
            "eval_steps": int(config.stage05_eval_steps),
            "layer_dims": [int(value) for value in config.stage05_layer_dims],
        },
        "contextual_3072_reference": {
            "source_summary_path": _repo_relative_posix(
                _resolve_repo_path(config.contextual_reference_summary_path)
            ),
            "method_name": STAGE05_V2_BUDGET_PUSH_METHOD_NAME,
            "epochs": int(config.contextual_reference_stage05_epochs),
        },
        "decision_rule": decision_rule,
    }


def _stage05_v2_promoted_v3b_v3c_suite_config_payload(
    config: Stage05V2PromotedV3BV3CComparisonConfig,
) -> dict[str, Any]:
    return {
        "phase": "FMPC Stage 05 EF Core Probe",
        "stage": str(config.experiment_name),
        "comparison_protocol": _stage05_v2_promoted_v3b_v3c_protocol_payload(config),
        "artifacts": {
            "aggregate_runs_csv": "aggregate_runs.csv",
            "aggregate_summary_json": "aggregate_summary.json",
            "comparison_report_json": "comparison_report.json",
            "comparison_report_md": "comparison_report.md",
        },
    }


def _stage05_v2_promoted_v3b_v3c_decision(
    *,
    rows: list[dict[str, Any]],
    config: Stage05V2PromotedV3BV3CComparisonConfig,
    by_method: dict[str, dict[str, Any]],
    pairwise_v3b_vs_v2: dict[str, Any],
    pairwise_v3c_vs_v2: dict[str, Any],
    pairwise_v3c_vs_v3b: dict[str, Any],
    contextual_reference: dict[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    v3c_candidate_method_name = str(config.v3c_candidate_method_name)
    v2_rows = _method_rows(rows, STAGE05_V2_METHOD_NAME)
    v3b_rows = _method_rows(rows, STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME)
    v3c_rows = _method_rows(rows, v3c_candidate_method_name)
    v2_by_seed = {int(row["seed"]): row for row in v2_rows}
    v3b_by_seed = {int(row["seed"]): row for row in v3b_rows}
    v3c_by_seed = {int(row["seed"]): row for row in v3c_rows}
    shared_seeds = sorted(set(v2_by_seed).intersection(v3b_by_seed).intersection(v3c_by_seed))
    if not shared_seeds:
        raise ValueError("Stage 05 v2/promoted-v3-B/v3-C comparison requires shared seeds.")

    artifact_pass = all(
        bool(v2_by_seed[seed]["deterministic_artifact_checks_passed"])
        and bool(v3b_by_seed[seed]["deterministic_artifact_checks_passed"])
        and bool(v3c_by_seed[seed]["deterministic_artifact_checks_passed"])
        for seed in shared_seeds
    )
    one_step_positive = all(
        float(v3c_by_seed[seed]["one_step_energy_delta_vs_identity"]) < 0.0 for seed in shared_seeds
    )
    configured_step_positive = all(
        float(v3c_by_seed[seed]["configured_step_energy_delta_vs_identity"]) < 0.0
        and float(v3c_by_seed[seed]["configured_step_fixed_point_residual_delta_vs_identity"]) < 0.0
        for seed in shared_seeds
    )

    reference_energy_mag = max(
        abs(
            float(
                by_method[STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME][
                    "configured_step_energy_delta_vs_identity"
                ]["mean"]
            )
        ),
        1e-12,
    )
    reference_residual_mag = max(
        abs(
            float(
                by_method[STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME][
                    "configured_step_fixed_point_residual_delta_vs_identity"
                ]["mean"]
            )
        ),
        1e-12,
    )
    energy_delta = float(
        pairwise_v3c_vs_v3b["configured_step_energy_delta_vs_identity_delta"]["mean"]
    )
    residual_delta = float(
        pairwise_v3c_vs_v3b["configured_step_fixed_point_residual_delta_vs_identity_delta"]["mean"]
    )
    materially_improves_vs_v3b = bool(
        energy_delta < 0.0
        and residual_delta < 0.0
        and (
            abs(energy_delta)
            >= float(config.configured_step_improvement_fraction_threshold) * reference_energy_mag
            or abs(residual_delta)
            >= float(config.configured_step_improvement_fraction_threshold) * reference_residual_mag
        )
    )
    avoids_obvious_accuracy_regression = bool(
        float(pairwise_v3c_vs_v3b["val_accuracy_delta"]["mean"])
        >= -float(config.allowed_accuracy_regression_threshold)
        and float(pairwise_v3c_vs_v3b["test_accuracy_delta"]["mean"])
        >= -float(config.allowed_accuracy_regression_threshold)
    )

    gap_closure_payload: dict[str, Any] | None = None
    positive_gap_closure = False
    if contextual_reference is not None:
        v2_energy = float(
            by_method[STAGE05_V2_METHOD_NAME]["configured_step_energy_delta_vs_identity"]["mean"]
        )
        v2_residual = float(
            by_method[STAGE05_V2_METHOD_NAME][
                "configured_step_fixed_point_residual_delta_vs_identity"
            ]["mean"]
        )
        promoted_v3b_energy = float(
            by_method[STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME][
                "configured_step_energy_delta_vs_identity"
            ]["mean"]
        )
        promoted_v3b_residual = float(
            by_method[STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME][
                "configured_step_fixed_point_residual_delta_vs_identity"
            ]["mean"]
        )
        v3c_energy = float(
            by_method[v3c_candidate_method_name]["configured_step_energy_delta_vs_identity"]["mean"]
        )
        v3c_residual = float(
            by_method[v3c_candidate_method_name][
                "configured_step_fixed_point_residual_delta_vs_identity"
            ]["mean"]
        )
        reference_energy = float(contextual_reference["configured_step_energy_delta_vs_identity"]["mean"])
        reference_residual = float(
            contextual_reference["configured_step_fixed_point_residual_delta_vs_identity"]["mean"]
        )

        def _gap_fraction(control: float, candidate: float, reference: float) -> float:
            denominator = control - reference
            if abs(denominator) <= 1e-12:
                return 0.0
            return float((control - candidate) / denominator)

        gap_closure_payload = {
            "promoted_v3b": {
                "configured_step_energy": _gap_fraction(v2_energy, promoted_v3b_energy, reference_energy),
                "configured_step_residual": _gap_fraction(
                    v2_residual, promoted_v3b_residual, reference_residual
                ),
            },
            "stage05_v3c": {
                "configured_step_energy": _gap_fraction(v2_energy, v3c_energy, reference_energy),
                "configured_step_residual": _gap_fraction(v2_residual, v3c_residual, reference_residual),
            },
        }
        gap_closure_payload["refined_v3c"] = dict(gap_closure_payload["stage05_v3c"])
        gap_closure_payload["v3c_minus_promoted_v3b"] = {
            "configured_step_energy": float(
                gap_closure_payload["stage05_v3c"]["configured_step_energy"]
                - gap_closure_payload["promoted_v3b"]["configured_step_energy"]
            ),
            "configured_step_residual": float(
                gap_closure_payload["stage05_v3c"]["configured_step_residual"]
                - gap_closure_payload["promoted_v3b"]["configured_step_residual"]
            ),
        }
        gap_closure_payload["refined_v3c_minus_promoted_v3b"] = dict(
            gap_closure_payload["v3c_minus_promoted_v3b"]
        )
        positive_gap_closure = bool(
            gap_closure_payload["stage05_v3c"]["configured_step_energy"]
            > gap_closure_payload["promoted_v3b"]["configured_step_energy"]
            and gap_closure_payload["stage05_v3c"]["configured_step_residual"]
            >= gap_closure_payload["promoted_v3b"]["configured_step_residual"]
        )

    if config.comparison_scope == "smoke_only":
        recommended_next_move = "run_fixed_budget_v2_vs_promoted_v3b_vs_v3c_comparison"
        gap_closure_decision = "pending_real_fixed_budget_v2_vs_promoted_v3b_vs_v3c_comparison"
        rationale = (
            "The smoke run only verifies that the diagnostic-only v3-C candidate is wired, "
            "deterministic, and comparable against the promoted refined v3-B reference."
        )
    else:
        if (
            artifact_pass
            and one_step_positive
            and configured_step_positive
            and materially_improves_vs_v3b
            and avoids_obvious_accuracy_regression
            and positive_gap_closure
        ):
            recommended_next_move = "promote_refined_v3c_as_active_reference"
            gap_closure_decision = "positive_gap_closure_signal_vs_promoted_v3b"
            rationale = (
                f"The refined v3-C formal comparison candidate `{v3c_candidate_method_name}` "
                "improves configured-step mechanism over the promoted refined v3-B reference, "
                "preserves the mechanism-first gate, and improves contextual gap closure versus "
                "the 3072-epoch same-family reference."
            )
        elif (
            artifact_pass
            and one_step_positive
            and configured_step_positive
            and avoids_obvious_accuracy_regression
        ):
            recommended_next_move = "retain_promoted_v3b_as_active_reference"
            gap_closure_decision = (
                "directional_gain_without_clear_gap_closure"
                if (energy_delta < 0.0 and residual_delta < 0.0)
                else "no_positive_gap_closure_signal_vs_promoted_v3b"
            )
            rationale = (
                f"The refined v3-C formal comparison candidate `{v3c_candidate_method_name}` "
                "preserves the mechanism-first gate but does not yet materially and cleanly "
                "displace the promoted refined v3-B reference."
            )
        else:
            recommended_next_move = "retain_promoted_v3b_as_active_reference"
            gap_closure_decision = "no_positive_gap_closure_signal_vs_promoted_v3b"
            rationale = (
                f"The refined v3-C formal comparison candidate `{v3c_candidate_method_name}` "
                "does not yet show a strong enough configured-step gain over the promoted refined "
                "v3-B reference."
            )

    return (
        {
            "deterministic_artifact_checks_all_pass": bool(artifact_pass),
            "refined_v3c_formal_comparison_candidate_name": str(v3c_candidate_method_name),
            "stage05_v3c_keeps_one_step_mechanism_positive": bool(one_step_positive),
            "stage05_v3c_keeps_configured_step_mechanism_positive": bool(configured_step_positive),
            "stage05_v3c_materially_improves_configured_step_mechanism_vs_promoted_v3b": bool(
                materially_improves_vs_v3b
            ),
            "stage05_v3c_avoids_obvious_report_accuracy_regression": bool(
                avoids_obvious_accuracy_regression
            ),
            "stage05_v3c_shows_positive_gap_closure_signal_vs_promoted_v3b": bool(
                positive_gap_closure
            ),
            "promoted_refined_v3c_materially_beats_promoted_v3b": bool(
                materially_improves_vs_v3b
            ),
            "promoted_refined_v3c_avoids_obvious_report_accuracy_regression": bool(
                avoids_obvious_accuracy_regression
            ),
            "promoted_refined_v3c_replaces_promoted_v3b_as_active_reference": bool(
                recommended_next_move == "promote_refined_v3c_as_active_reference"
            ),
            "contextual_gap_closure_fractions_vs_3072_reference": gap_closure_payload,
            "gap_closure_decision": str(gap_closure_decision),
            "recommended_next_move": str(recommended_next_move),
        },
        rationale,
    )


def _stage05_v2_promoted_v3b_v3c_supports_lines(summary: dict[str, Any]) -> list[str]:
    candidate_name = str(summary["comparison_protocol"]["stage_05_v3c_candidate"]["candidate_name"])
    lines = [
        "The comparison preserves the Stage 05 mechanism-first contract.",
        "The comparison exposes explicit pairwise deltas versus both the fixed-budget v2 control and the promoted refined v3-B reference.",
        f"The diagnostic-only v3-C candidate `{candidate_name}` makes endpoint / semigroup consistency identity explicit in artifacts.",
    ]
    if summary["comparison_scope"] == "smoke_only":
        if bool(summary["deterministic_artifact_checks_all_pass"]):
            lines.append("The v3-C smoke run passes deterministic artifact checks.")
        return lines
    lines.append(
        (
            f"The v3-C formal comparison candidate `{candidate_name}` materially improves configured-step mechanism over the promoted refined v3-B reference."
            if bool(summary["promoted_refined_v3c_materially_beats_promoted_v3b"])
            else f"The v3-C formal comparison candidate `{candidate_name}` does not materially improve configured-step mechanism over the promoted refined v3-B reference."
        )
    )
    return lines


def _stage05_v2_promoted_v3b_v3c_does_not_support_lines() -> list[str]:
    return [
        "This comparison does not justify replacing the frozen Stage 04 bridge on main.",
        "This comparison does not promote task accuracy to the Stage 05 gate.",
        "This comparison does not reopen Stage 04 package-internal work.",
    ]


def _stage05_v2_promoted_v3b_v3c_report_markdown(report: dict[str, Any]) -> str:
    protocol = report["comparison_protocol"]
    decision = report["decision"]
    candidate_name = str(protocol["stage_05_v3c_candidate"]["candidate_name"])
    lines = [
        "# Stage 05 v2 vs Promoted v3-B vs v3-C Comparison",
        "",
        "## Protocol",
        f"- comparison scope: `{protocol['comparison_scope']}`",
        f"- dataset: `{protocol['dataset_name']}`",
        f"- seeds: `{protocol['seeds']}`",
        f"- shared batch size: `{protocol['shared_batch_size']}`",
        f"- Stage 05 epochs: `{protocol['stage_05_v3c_candidate']['epochs']}`",
        "",
        "## Decision",
        f"- refined v3-C formal comparison candidate: `{candidate_name}`",
        f"- `stage05_v3c_keeps_one_step_mechanism_positive`: `{decision['stage05_v3c_keeps_one_step_mechanism_positive']}`",
        f"- `stage05_v3c_materially_improves_configured_step_mechanism_vs_promoted_v3b`: `{decision['stage05_v3c_materially_improves_configured_step_mechanism_vs_promoted_v3b']}`",
        f"- `stage05_v3c_shows_positive_gap_closure_signal_vs_promoted_v3b`: `{decision['stage05_v3c_shows_positive_gap_closure_signal_vs_promoted_v3b']}`",
        f"- gap_closure_decision: `{decision['gap_closure_decision']}`",
        f"- recommended next move: `{decision['recommended_next_move']}`",
        f"- rationale: `{decision['decision_rationale']}`",
        "",
        "## Pairwise Deltas Vs Promoted v3-B",
        f"- configured-step validation energy delta vs identity delta: `{report['pairwise_deltas_vs_promoted_refined_v3b_reference']['configured_step_energy_delta_vs_identity_delta']['mean']}`",
        f"- configured-step validation fixed-point residual delta vs identity delta: `{report['pairwise_deltas_vs_promoted_refined_v3b_reference']['configured_step_fixed_point_residual_delta_vs_identity_delta']['mean']}`",
        "",
        "## Pairwise Deltas Vs V2",
        f"- configured-step validation energy delta vs identity delta: `{report['pairwise_deltas_vs_stage05_v2_reference']['configured_step_energy_delta_vs_identity_delta']['mean']}`",
        f"- configured-step validation fixed-point residual delta vs identity delta: `{report['pairwise_deltas_vs_stage05_v2_reference']['configured_step_fixed_point_residual_delta_vs_identity_delta']['mean']}`",
    ]
    return "\n".join(lines)


def run_stage05_v2_promoted_v3b_v3c_comparison(
    config: Stage05V2PromotedV3BV3CComparisonConfig,
) -> FrozenBridgeVsCorrectedCoreComparisonRunResult:
    """Run the smoke-ready Stage 05 v2 vs promoted refined v3-B vs v3-C comparison."""

    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, config.resolved_run_id(), config.output_layout)
    )
    _write_json(run_dir / "config.json", _stage05_v2_promoted_v3b_v3c_suite_config_payload(config))

    rows: list[dict[str, Any]] = []
    runs_root = run_dir / "runs"
    run_index = 0
    reuse_v2_root = _resolve_repo_path(config.reference_artifact_root)
    reuse_v3b_root = _resolve_repo_path(config.v3b_reference_artifact_root)
    reuse_v3c_root = _resolve_repo_path(config.v3c_candidate_artifact_root)

    if config.reuse_stage05_v2_reference_artifacts and not reuse_v2_root.exists():
        raise FileNotFoundError(f"Missing Stage 05 v2 reference artifacts at '{reuse_v2_root}'.")
    if config.reuse_stage05_v3b_reference_artifacts and not reuse_v3b_root.exists():
        raise FileNotFoundError(
            f"Missing promoted Stage 05 v3-B reference artifacts at '{reuse_v3b_root}'."
        )
    if config.reuse_stage05_v3c_candidate_artifacts and not reuse_v3c_root.exists():
        raise FileNotFoundError(f"Missing Stage 05 v3-C candidate artifacts at '{reuse_v3c_root}'.")

    for seed in config.seeds:
        run_index += 1
        if config.reuse_stage05_v2_reference_artifacts:
            rows.append(
                _load_existing_stage05_core_row(
                    run_index=run_index,
                    existing_run_dir=reuse_v2_root / f"seed_{seed}",
                    seed=seed,
                    expected_dataset_name=config.dataset_name,
                    expected_batch_size=int(config.batch_size),
                    expected_shuffle_batches=bool(config.shuffle_batches),
                    method_name=STAGE05_V2_METHOD_NAME,
                    stage_name="FMPC Stage 05 EF Core Probe v2 Control",
                    expected_total_training_epochs=int(config.stage05_epochs),
                    expected_eval_steps=int(config.stage05_eval_steps),
                    expected_layer_dims=config.stage05_layer_dims,
                    expected_transport_steps=int(config.stage05_transport_steps),
                    expected_transport_family="two_branch_residual_meanflow_core",
                    expected_explicit_transport_drift_decomposition_enabled=False,
                    expected_trajectory_curriculum_enabled=False,
                    expected_endpoint_semigroup_consistency_enabled=False,
                )
            )
        else:
            v2_config = _build_stage05_v2_config(
                output_root=runs_root,
                experiment_name=STAGE05_V2_METHOD_NAME,
                seed=seed,
                train_fraction=float(config.train_fraction),
                val_fraction=float(config.val_fraction),
                test_fraction=float(config.test_fraction),
                batch_size=int(config.batch_size),
                shuffle_batches=bool(config.shuffle_batches),
                epochs=int(config.stage05_epochs),
                eval_steps=int(config.stage05_eval_steps),
                layer_dims=config.stage05_layer_dims,
                transport_steps=int(config.stage05_transport_steps),
            )
            v2_result = run_fmpc_ef_exploratory_probe(v2_config)
            rows.append(
                _stage05_core_row(
                    run_index=run_index,
                    suite_run_dir=run_dir,
                    seed=seed,
                    result=v2_result,
                    config=v2_config,
                    method_name=STAGE05_V2_METHOD_NAME,
                    stage_name="FMPC Stage 05 EF Core Probe v2 Control",
                )
            )

        run_index += 1
        if config.reuse_stage05_v3b_reference_artifacts:
            rows.append(
                _load_existing_stage05_core_row(
                    run_index=run_index,
                    existing_run_dir=reuse_v3b_root / f"seed_{seed}",
                    seed=seed,
                    expected_dataset_name=config.dataset_name,
                    expected_batch_size=int(config.batch_size),
                    expected_shuffle_batches=bool(config.shuffle_batches),
                    method_name=STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME,
                    stage_name="FMPC Stage 05 EF Core Probe Promoted Refined v3-B Reference",
                    expected_total_training_epochs=int(config.stage05_epochs),
                    expected_eval_steps=int(config.stage05_eval_steps),
                    expected_layer_dims=config.stage05_layer_dims,
                    expected_transport_steps=int(config.stage05_transport_steps),
                    expected_transport_family="two_branch_residual_meanflow_core",
                    expected_explicit_transport_drift_decomposition_enabled=True,
                    expected_trajectory_curriculum_enabled=True,
                    expected_endpoint_semigroup_consistency_enabled=False,
                    expected_lambda_traj_curr=float(config.promoted_v3b_lambda_traj_curr),
                    expected_alpha_floor=float(config.promoted_v3b_alpha_floor),
                )
            )
        else:
            promoted_v3b_config = _stage05_promoted_v3b_reference_config(
                config,
                seed=seed,
                output_root=runs_root,
            )
            promoted_v3b_result = run_fmpc_ef_exploratory_probe(promoted_v3b_config)
            rows.append(
                _stage05_core_row(
                    run_index=run_index,
                    suite_run_dir=run_dir,
                    seed=seed,
                    result=promoted_v3b_result,
                    config=promoted_v3b_config,
                    method_name=STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME,
                    stage_name="FMPC Stage 05 EF Core Probe Promoted Refined v3-B Reference",
                )
            )

        run_index += 1
        if config.reuse_stage05_v3c_candidate_artifacts:
            rows.append(
                _load_existing_stage05_core_row(
                    run_index=run_index,
                    existing_run_dir=reuse_v3c_root / f"seed_{seed}",
                    seed=seed,
                    expected_dataset_name=config.dataset_name,
                    expected_batch_size=int(config.batch_size),
                    expected_shuffle_batches=bool(config.shuffle_batches),
                    method_name=str(config.v3c_candidate_method_name),
                    stage_name="FMPC Stage 05 EF Core Probe v3-C Formal Comparison Candidate",
                    expected_total_training_epochs=int(config.stage05_epochs),
                    expected_eval_steps=int(config.stage05_eval_steps),
                    expected_layer_dims=config.stage05_layer_dims,
                    expected_transport_steps=int(config.stage05_transport_steps),
                    expected_transport_family="two_branch_residual_meanflow_core",
                    expected_explicit_transport_drift_decomposition_enabled=True,
                    expected_trajectory_curriculum_enabled=True,
                    expected_endpoint_semigroup_consistency_enabled=True,
                    expected_lambda_traj_curr=float(config.promoted_v3b_lambda_traj_curr),
                    expected_alpha_floor=float(config.promoted_v3b_alpha_floor),
                    expected_lambda_sg=float(config.lambda_sg),
                )
            )
        else:
            v3c_config = _stage05_v3c_candidate_config(config, seed=seed, output_root=runs_root)
            v3c_result = run_fmpc_ef_exploratory_probe(v3c_config)
            rows.append(
                _stage05_core_row(
                    run_index=run_index,
                    suite_run_dir=run_dir,
                    seed=seed,
                    result=v3c_result,
                    config=v3c_config,
                    method_name=str(config.v3c_candidate_method_name),
                    stage_name="FMPC Stage 05 EF Core Probe v3-C Formal Comparison Candidate",
                )
            )

    csv_rows = [
        {
            **row,
            "deterministic_artifact_checks_passed": str(bool(row["deterministic_artifact_checks_passed"])),
            "mechanism_signal_positive": str(bool(row["mechanism_signal_positive"])),
            "config_json_exists": str(bool(row["config_json_exists"])),
            "summary_json_exists": str(bool(row["summary_json_exists"])),
            "epoch_metrics_csv_exists": str(bool(row["epoch_metrics_csv_exists"])),
            "seed_matches": str(bool(row["seed_matches"])),
            "dataset_matches": str(bool(row["dataset_matches"])),
            "batch_protocol_matches": str(bool(row["batch_protocol_matches"])),
            "selection_hits_final_training_boundary": str(
                bool(row["selection_hits_final_training_boundary"])
            ),
        }
        for row in rows
    ]
    _write_csv(run_dir / "aggregate_runs.csv", csv_rows)

    by_method = {
        STAGE05_V2_METHOD_NAME: _method_summary(_method_rows(rows, STAGE05_V2_METHOD_NAME)),
        STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME: _method_summary(
            _method_rows(rows, STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME)
        ),
        str(config.v3c_candidate_method_name): _method_summary(
            _method_rows(rows, str(config.v3c_candidate_method_name))
        ),
    }
    pairwise_v3b_vs_v2 = _pairwise_summary(
        rows,
        candidate_method=STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME,
        reference_method=STAGE05_V2_METHOD_NAME,
    )
    pairwise_v3c_vs_v2 = _pairwise_summary(
        rows,
        candidate_method=str(config.v3c_candidate_method_name),
        reference_method=STAGE05_V2_METHOD_NAME,
    )
    pairwise_v3c_vs_v3b = _pairwise_summary(
        rows,
        candidate_method=str(config.v3c_candidate_method_name),
        reference_method=STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME,
    )
    contextual_reference = (
        None if config.comparison_scope == "smoke_only" else _load_stage05_v2_contextual_reference(config)
    )
    decision, decision_rationale = _stage05_v2_promoted_v3b_v3c_decision(
        rows=rows,
        config=config,
        by_method=by_method,
        pairwise_v3b_vs_v2=pairwise_v3b_vs_v2,
        pairwise_v3c_vs_v2=pairwise_v3c_vs_v2,
        pairwise_v3c_vs_v3b=pairwise_v3c_vs_v3b,
        contextual_reference=contextual_reference,
    )

    configured_step_mechanism_ranking = [
        {
            "method_name": method_name,
            "configured_step_energy_delta_vs_identity_mean": float(
                by_method[method_name]["configured_step_energy_delta_vs_identity"]["mean"]
            ),
            "configured_step_fixed_point_residual_delta_vs_identity_mean": float(
                by_method[method_name]["configured_step_fixed_point_residual_delta_vs_identity"]["mean"]
            ),
        }
        for method_name in sorted(
            (
                STAGE05_V2_METHOD_NAME,
                STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME,
                str(config.v3c_candidate_method_name),
            ),
            key=lambda method_name: (
                float(by_method[method_name]["configured_step_energy_delta_vs_identity"]["mean"]),
                float(
                    by_method[method_name]["configured_step_fixed_point_residual_delta_vs_identity"][
                        "mean"
                    ]
                ),
            ),
        )
    ]

    summary = {
        "phase": "FMPC Stage 05 EF Core Probe",
        "stage": str(config.experiment_name),
        "comparison_scope": str(config.comparison_scope),
        "num_runs": int(len(rows)),
        "comparison_protocol": _stage05_v2_promoted_v3b_v3c_protocol_payload(config),
        "comparison_roles": {
            "immediate_control": STAGE05_V2_METHOD_NAME,
            "active_reference_at_comparison_start": STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME,
            "refined_v3c_formal_comparison_candidate": str(config.v3c_candidate_method_name),
        },
        "candidate_identities": [
            STAGE05_V2_METHOD_NAME,
            STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME,
            str(config.v3c_candidate_method_name),
        ],
        "by_method": by_method,
        "configured_step_mechanism_ranking": configured_step_mechanism_ranking,
        "pairwise_deltas_vs_stage05_v2_reference": pairwise_v3c_vs_v2,
        "pairwise_deltas_vs_promoted_refined_v3b_reference": pairwise_v3c_vs_v3b,
        "pairwise_promoted_refined_v3b_vs_v2": pairwise_v3b_vs_v2,
        "contextual_3072_reference": contextual_reference,
        **decision,
        "decision_rationale": decision_rationale,
        "aggregate_runs_csv_path": "aggregate_runs.csv",
        "comparison_report_json_path": "comparison_report.json",
        "comparison_report_md_path": "comparison_report.md",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)

    report = {
        "comparison_protocol": _stage05_v2_promoted_v3b_v3c_protocol_payload(config),
        "decision": {**decision, "decision_rationale": decision_rationale},
        "pairwise_deltas_vs_stage05_v2_reference": pairwise_v3c_vs_v2,
        "pairwise_deltas_vs_promoted_refined_v3b_reference": pairwise_v3c_vs_v3b,
        "pairwise_promoted_refined_v3b_vs_v2": pairwise_v3b_vs_v2,
        "contextual_3072_reference": contextual_reference,
        "contextual_gap_closure_fractions_vs_3072_reference": decision.get(
            "contextual_gap_closure_fractions_vs_3072_reference"
        ),
        "supports": _stage05_v2_promoted_v3b_v3c_supports_lines(summary),
        "does_not_support": _stage05_v2_promoted_v3b_v3c_does_not_support_lines(),
    }
    _write_json(run_dir / "comparison_report.json", report)
    _write_text(run_dir / "comparison_report.md", _stage05_v2_promoted_v3b_v3c_report_markdown(report))

    return FrozenBridgeVsCorrectedCoreComparisonRunResult(
        run_dir=run_dir,
        config=_stage05_v2_promoted_v3b_v3c_suite_config_payload(config),
        aggregate_rows=rows,
        summary=summary,
        comparison_report=report,
    )


def _stage05_v3b_refinement_variant_specs(
    config: Stage05V3BRefinementDiagnosticConfig,
) -> list[dict[str, Any]]:
    return [
        {
            "method_name": STAGE05_V3B_ALPHA_EARLIER_METHOD_NAME,
            "stage_name": "FMPC Stage 05 EF Core Probe v3-B Alpha Earlier Transition Variant",
            "lambda_traj_curr": float(config.control_lambda_traj_curr),
            "alpha_floor": float(config.alpha_earlier_transition_alpha_floor),
            "alpha_warmup_epochs": int(config.alpha_earlier_transition_alpha_warmup_epochs),
            "alpha_ramp_epochs": int(config.alpha_earlier_transition_alpha_ramp_epochs),
        },
        {
            "method_name": STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME,
            "stage_name": "FMPC Stage 05 EF Core Probe v3-B Stronger Trajectory Curriculum Weight Variant",
            "lambda_traj_curr": float(config.stronger_traj_curr_lambda_traj_curr),
            "alpha_floor": float(config.control_alpha_floor),
            "alpha_warmup_epochs": int(config.control_alpha_warmup_epochs),
            "alpha_ramp_epochs": int(config.control_alpha_ramp_epochs),
        },
    ]


def _stage05_v3b_refinement_variant_payload(
    *,
    method_name: ComparisonMethodName,
    lambda_traj_curr: float,
    alpha_floor: float,
    alpha_warmup_epochs: int,
    alpha_ramp_epochs: int,
) -> dict[str, Any]:
    return {
        "method_name": method_name,
        "candidate_name": method_name,
        "transport_family": "two_branch_residual_meanflow_core",
        "residual_branch_structure": "two_branch",
        "explicit_transport_drift_decomposition_enabled": True,
        "trajectory_curriculum_enabled": True,
        "trajectory_curriculum_schedule_identity": "warmup_sigmoid_to_alpha_floor",
        "lambda_traj_curr": float(lambda_traj_curr),
        "alpha_floor": float(alpha_floor),
        "alpha_warmup_epochs": int(alpha_warmup_epochs),
        "alpha_ramp_epochs": int(alpha_ramp_epochs),
    }


def _stage05_v3b_refinement_protocol_payload(
    config: Stage05V3BRefinementDiagnosticConfig,
) -> dict[str, Any]:
    if config.comparison_scope == "smoke_only":
        decision_rule = {
            "purpose": "smoke_ready_v3b_refinement_scaffold_check",
            "task_accuracy_is_report_only": True,
            "artifact_checks_required": True,
            "smoke_only": True,
        }
    else:
        decision_rule = {
            "purpose": "fixed_budget_v3b_refinement_attribution_diagnostic",
            "primary_split": "validation",
            "task_accuracy_is_report_only": True,
            "configured_step_improvement_fraction_threshold": float(
                config.configured_step_improvement_fraction_threshold
            ),
            "allowed_accuracy_regression_threshold": float(
                config.allowed_accuracy_regression_threshold
            ),
            "reuse_stage05_v2_reference_artifacts": bool(
                config.reuse_stage05_v2_reference_artifacts
            ),
            "reuse_stage05_v3a_reference_artifacts": bool(
                config.reuse_stage05_v3a_reference_artifacts
            ),
            "reuse_stage05_v3b_control_artifacts": bool(
                config.reuse_stage05_v3b_control_artifacts
            ),
        }
    return {
        "comparison_scope": str(config.comparison_scope),
        "dataset_name": config.dataset_name,
        "seeds": [int(seed) for seed in config.seeds],
        "train_fraction": float(config.train_fraction),
        "val_fraction": float(config.val_fraction),
        "test_fraction": float(config.test_fraction),
        "shared_batch_size": int(config.batch_size),
        "shared_shuffle_batches": bool(config.shuffle_batches),
        "fixed_budget_v2_reference": {
            "method_name": STAGE05_V2_METHOD_NAME,
            "reference_reused_from_existing_artifacts": bool(
                config.reuse_stage05_v2_reference_artifacts
            ),
            "source_artifact_root": (
                _repo_relative_posix(_resolve_repo_path(config.reference_artifact_root))
                if config.reuse_stage05_v2_reference_artifacts
                else None
            ),
            "epochs": int(config.stage05_epochs),
            "configured_transport_steps": int(config.stage05_transport_steps),
        },
        "stage05_v3a_reference": {
            "method_name": STAGE05_V3A_METHOD_NAME,
            "reference_reused_from_existing_artifacts": bool(
                config.reuse_stage05_v3a_reference_artifacts
            ),
            "source_artifact_root": (
                _repo_relative_posix(_resolve_repo_path(config.v3a_reference_artifact_root))
                if config.reuse_stage05_v3a_reference_artifacts
                else None
            ),
            "epochs": int(config.stage05_epochs),
            "configured_transport_steps": int(config.stage05_transport_steps),
        },
        "stage05_v3b_control": {
            **_stage05_v3b_refinement_variant_payload(
                method_name=STAGE05_V3B_METHOD_NAME,
                lambda_traj_curr=float(config.control_lambda_traj_curr),
                alpha_floor=float(config.control_alpha_floor),
                alpha_warmup_epochs=int(config.control_alpha_warmup_epochs),
                alpha_ramp_epochs=int(config.control_alpha_ramp_epochs),
            ),
            "reference_reused_from_existing_artifacts": bool(
                config.reuse_stage05_v3b_control_artifacts
            ),
            "source_artifact_root": (
                _repo_relative_posix(_resolve_repo_path(config.v3b_control_artifact_root))
                if config.reuse_stage05_v3b_control_artifacts
                else None
            ),
            "epochs": int(config.stage05_epochs),
            "configured_transport_steps": int(config.stage05_transport_steps),
        },
        "refinement_variants": [
            {
                **_stage05_v3b_refinement_variant_payload(
                    method_name=spec["method_name"],
                    lambda_traj_curr=float(spec["lambda_traj_curr"]),
                    alpha_floor=float(spec["alpha_floor"]),
                    alpha_warmup_epochs=int(spec["alpha_warmup_epochs"]),
                    alpha_ramp_epochs=int(spec["alpha_ramp_epochs"]),
                ),
                "epochs": int(config.stage05_epochs),
                "configured_transport_steps": int(config.stage05_transport_steps),
            }
            for spec in _stage05_v3b_refinement_variant_specs(config)
        ],
        "contextual_3072_reference": {
            "source_summary_path": _repo_relative_posix(
                _resolve_repo_path(config.contextual_reference_summary_path)
            ),
            "method_name": STAGE05_V2_BUDGET_PUSH_METHOD_NAME,
            "epochs": int(config.contextual_reference_stage05_epochs),
        },
        "decision_rule": decision_rule,
    }


def _stage05_v3b_refinement_suite_config_payload(
    config: Stage05V3BRefinementDiagnosticConfig,
) -> dict[str, Any]:
    return {
        "phase": "FMPC Stage 05 EF Core Probe",
        "stage": str(config.experiment_name),
        "comparison_protocol": _stage05_v3b_refinement_protocol_payload(config),
        "artifacts": {
            "aggregate_runs_csv": "aggregate_runs.csv",
            "aggregate_summary_json": "aggregate_summary.json",
            "comparison_report_json": "comparison_report.json",
            "comparison_report_md": "comparison_report.md",
        },
    }


def _negative_magnitude_relative_gain(
    *,
    current_value: float,
    candidate_value: float,
) -> float:
    baseline = abs(float(current_value))
    candidate = abs(float(candidate_value))
    scale = max(baseline, 1e-12)
    return float((candidate - baseline) / scale)


def _stage05_v3b_refinement_contextual_gap_payload(
    *,
    by_method: dict[str, dict[str, Any]],
    contextual_reference: dict[str, Any],
) -> dict[str, Any]:
    v2_summary = by_method[STAGE05_V2_METHOD_NAME]
    v2_configured_energy_mean = float(v2_summary["configured_step_energy_delta_vs_identity"]["mean"])
    v2_configured_residual_mean = float(
        v2_summary["configured_step_fixed_point_residual_delta_vs_identity"]["mean"]
    )
    v2_val_accuracy_mean = float(v2_summary["val_accuracy"]["mean"])
    v2_test_accuracy_mean = float(v2_summary["test_accuracy"]["mean"])

    payload: dict[str, Any] = {}
    for method_name, method_summary in by_method.items():
        if method_name == STAGE05_V2_METHOD_NAME:
            continue
        payload[method_name] = {
            "configured_step_energy": float(
                _negative_gap_closed_fraction(
                    reference_value=v2_configured_energy_mean,
                    candidate_value=float(
                        method_summary["configured_step_energy_delta_vs_identity"]["mean"]
                    ),
                    target_value=float(
                        contextual_reference["configured_step_energy_delta_vs_identity"]["mean"]
                    ),
                )
            ),
            "configured_step_residual": float(
                _negative_gap_closed_fraction(
                    reference_value=v2_configured_residual_mean,
                    candidate_value=float(
                        method_summary["configured_step_fixed_point_residual_delta_vs_identity"][
                            "mean"
                        ]
                    ),
                    target_value=float(
                        contextual_reference[
                            "configured_step_fixed_point_residual_delta_vs_identity"
                        ]["mean"]
                    ),
                )
            ),
            "val_accuracy": float(
                _positive_gap_closed_fraction(
                    reference_value=v2_val_accuracy_mean,
                    candidate_value=float(method_summary["val_accuracy"]["mean"]),
                    target_value=float(contextual_reference["val_accuracy"]["mean"]),
                )
            ),
            "test_accuracy": float(
                _positive_gap_closed_fraction(
                    reference_value=v2_test_accuracy_mean,
                    candidate_value=float(method_summary["test_accuracy"]["mean"]),
                    target_value=float(contextual_reference["test_accuracy"]["mean"]),
                )
            ),
        }
    return payload


def _best_stage05_v3b_refinement_method(
    by_method: dict[str, dict[str, Any]],
) -> ComparisonMethodName:
    ranked = sorted(
        (
            STAGE05_V3B_ALPHA_EARLIER_METHOD_NAME,
            STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME,
        ),
        key=lambda method_name: (
            float(by_method[method_name]["configured_step_energy_delta_vs_identity"]["mean"]),
            float(
                by_method[method_name]["configured_step_fixed_point_residual_delta_vs_identity"][
                    "mean"
                ]
            ),
        ),
    )
    return ranked[0]


def _stage05_v3b_refinement_decision(
    *,
    rows: list[dict[str, Any]],
    config: Stage05V3BRefinementDiagnosticConfig,
    by_method: dict[str, dict[str, Any]],
    pairwise_vs_control: dict[str, dict[str, Any]],
    pairwise_vs_v3a: dict[str, dict[str, Any]],
    contextual_reference: dict[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    best_method = _best_stage05_v3b_refinement_method(by_method)
    best_rows = _method_rows(rows, best_method)
    best_vs_control = pairwise_vs_control[best_method]
    best_vs_v3a = pairwise_vs_v3a[best_method]
    best_by_seed = {int(row["seed"]): row for row in best_rows}
    control_by_seed = {int(row["seed"]): row for row in _method_rows(rows, STAGE05_V3B_METHOD_NAME)}
    v3a_by_seed = {int(row["seed"]): row for row in _method_rows(rows, STAGE05_V3A_METHOD_NAME)}
    shared_control_seeds = sorted(set(best_by_seed).intersection(control_by_seed))
    shared_v3a_seeds = sorted(set(best_by_seed).intersection(v3a_by_seed))

    artifact_pass = all(bool(row["deterministic_artifact_checks_passed"]) for row in best_rows)
    one_step_positive = all(
        float(row["one_step_energy_delta_vs_identity"]) < 0.0 for row in best_rows
    )
    mechanism_positive = all(bool(row["mechanism_signal_positive"]) for row in best_rows)

    energy_gain_vs_control = float(
        _negative_magnitude_relative_gain(
            current_value=float(
                by_method[STAGE05_V3B_METHOD_NAME]["configured_step_energy_delta_vs_identity"][
                    "mean"
                ]
            ),
            candidate_value=float(
                by_method[best_method]["configured_step_energy_delta_vs_identity"]["mean"]
            ),
        )
    )
    residual_gain_vs_control = float(
        _negative_magnitude_relative_gain(
            current_value=float(
                by_method[STAGE05_V3B_METHOD_NAME][
                    "configured_step_fixed_point_residual_delta_vs_identity"
                ]["mean"]
            ),
            candidate_value=float(
                by_method[best_method][
                    "configured_step_fixed_point_residual_delta_vs_identity"
                ]["mean"]
            ),
        )
    )
    energy_gain_vs_v3a = float(
        _negative_magnitude_relative_gain(
            current_value=float(
                by_method[STAGE05_V3A_METHOD_NAME]["configured_step_energy_delta_vs_identity"][
                    "mean"
                ]
            ),
            candidate_value=float(
                by_method[best_method]["configured_step_energy_delta_vs_identity"]["mean"]
            ),
        )
    )
    residual_gain_vs_v3a = float(
        _negative_magnitude_relative_gain(
            current_value=float(
                by_method[STAGE05_V3A_METHOD_NAME][
                    "configured_step_fixed_point_residual_delta_vs_identity"
                ]["mean"]
            ),
            candidate_value=float(
                by_method[best_method][
                    "configured_step_fixed_point_residual_delta_vs_identity"
                ]["mean"]
            ),
        )
    )

    energy_seed_improvement_rate_vs_control = _rate(
        [
            float(best_by_seed[seed]["configured_step_energy_delta_vs_identity"])
            < float(control_by_seed[seed]["configured_step_energy_delta_vs_identity"])
            for seed in shared_control_seeds
        ]
    )
    residual_seed_improvement_rate_vs_control = _rate(
        [
            float(best_by_seed[seed]["configured_step_fixed_point_residual_delta_vs_identity"])
            < float(control_by_seed[seed]["configured_step_fixed_point_residual_delta_vs_identity"])
            for seed in shared_control_seeds
        ]
    )
    energy_seed_improvement_rate_vs_v3a = _rate(
        [
            float(best_by_seed[seed]["configured_step_energy_delta_vs_identity"])
            < float(v3a_by_seed[seed]["configured_step_energy_delta_vs_identity"])
            for seed in shared_v3a_seeds
        ]
    )
    residual_seed_improvement_rate_vs_v3a = _rate(
        [
            float(best_by_seed[seed]["configured_step_fixed_point_residual_delta_vs_identity"])
            < float(v3a_by_seed[seed]["configured_step_fixed_point_residual_delta_vs_identity"])
            for seed in shared_v3a_seeds
        ]
    )

    materially_beats_control = bool(
        energy_gain_vs_control >= float(config.configured_step_improvement_fraction_threshold)
        and residual_gain_vs_control
        >= float(config.configured_step_improvement_fraction_threshold)
        and energy_seed_improvement_rate_vs_control >= 0.5
        and residual_seed_improvement_rate_vs_control >= 0.5
    )
    materially_beats_v3a = bool(
        energy_gain_vs_v3a >= float(config.configured_step_improvement_fraction_threshold)
        and residual_gain_vs_v3a >= float(config.configured_step_improvement_fraction_threshold)
        and energy_seed_improvement_rate_vs_v3a >= 0.5
        and residual_seed_improvement_rate_vs_v3a >= 0.5
    )
    directional_beats_control = bool(
        float(best_vs_control["configured_step_energy_delta_vs_identity_delta"]["mean"]) < 0.0
        and float(
            best_vs_control["configured_step_fixed_point_residual_delta_vs_identity_delta"][
                "mean"
            ]
        )
        < 0.0
    )
    directional_beats_v3a = bool(
        float(best_vs_v3a["configured_step_energy_delta_vs_identity_delta"]["mean"]) < 0.0
        and float(
            best_vs_v3a["configured_step_fixed_point_residual_delta_vs_identity_delta"]["mean"]
        )
        < 0.0
    )
    avoids_obvious_accuracy_regression = bool(
        float(best_vs_v3a["val_accuracy_delta"]["mean"])
        >= -float(config.allowed_accuracy_regression_threshold)
        and float(best_vs_v3a["test_accuracy_delta"]["mean"])
        >= -float(config.allowed_accuracy_regression_threshold)
    )

    contextual_gap = (
        None
        if contextual_reference is None
        else _stage05_v3b_refinement_contextual_gap_payload(
            by_method=by_method,
            contextual_reference=contextual_reference,
        )
    )
    best_contextual_gap_gain_vs_v3a = None
    if contextual_gap is not None:
        best_contextual_gap_gain_vs_v3a = {
            "configured_step_energy": float(
                contextual_gap[best_method]["configured_step_energy"]
                - contextual_gap[STAGE05_V3A_METHOD_NAME]["configured_step_energy"]
            ),
            "configured_step_residual": float(
                contextual_gap[best_method]["configured_step_residual"]
                - contextual_gap[STAGE05_V3A_METHOD_NAME]["configured_step_residual"]
            ),
            "val_accuracy": float(
                contextual_gap[best_method]["val_accuracy"]
                - contextual_gap[STAGE05_V3A_METHOD_NAME]["val_accuracy"]
            ),
            "test_accuracy": float(
                contextual_gap[best_method]["test_accuracy"]
                - contextual_gap[STAGE05_V3A_METHOD_NAME]["test_accuracy"]
            ),
        }

    if not artifact_pass or not one_step_positive or not mechanism_positive:
        recommended_next_move = "retain_v3a_as_active_reference_and_stop_v3b"
        rationale = (
            "The best refined v3-B variant does not preserve the minimum artifact or mechanism "
            "stability checks, so v3-A should remain the active reference and v3-B should stop here."
        )
    elif materially_beats_v3a and avoids_obvious_accuracy_regression:
        recommended_next_move = "promote_refined_v3b_and_recompare"
        rationale = (
            "A narrow v3-B refinement materially improves configured-step mechanism over the active "
            "v3-A reference without an obvious report-only accuracy regression, so it is worth "
            "promoting into a fresh fixed-budget re-comparison."
        )
    elif directional_beats_control or directional_beats_v3a:
        recommended_next_move = "retain_v3a_as_active_reference_but_keep_v3b_as_future_context"
        rationale = (
            "A narrow v3-B refinement nudges configured-step mechanism in the right direction, but "
            "the gain is still not material enough to displace v3-A as the active reference."
        )
    else:
        recommended_next_move = "retain_v3a_as_active_reference_and_stop_v3b"
        rationale = (
            "The tested narrow v3-B refinements do not materially improve configured-step mechanism "
            "relative to v3-A, so continuing this line is not justified from the current evidence."
        )

    return {
        "best_variant_name": str(best_method),
        "best_variant_keeps_one_step_mechanism_positive": bool(one_step_positive),
        "best_variant_mechanism_signal_positive_on_all_runs": bool(mechanism_positive),
        "best_variant_avoids_obvious_report_accuracy_regression": bool(
            avoids_obvious_accuracy_regression
        ),
        "narrow_v3b_refinement_materially_beats_v3b_control": bool(materially_beats_control),
        "narrow_v3b_refinement_materially_beats_v3a_reference": bool(materially_beats_v3a),
        "configured_step_gain_fraction_vs_v3b_control": {
            "energy": float(energy_gain_vs_control),
            "residual": float(residual_gain_vs_control),
        },
        "configured_step_gain_fraction_vs_v3a_reference": {
            "energy": float(energy_gain_vs_v3a),
            "residual": float(residual_gain_vs_v3a),
        },
        "configured_step_seed_improvement_rate_vs_v3b_control": {
            "energy": float(energy_seed_improvement_rate_vs_control),
            "residual": float(residual_seed_improvement_rate_vs_control),
        },
        "configured_step_seed_improvement_rate_vs_v3a_reference": {
            "energy": float(energy_seed_improvement_rate_vs_v3a),
            "residual": float(residual_seed_improvement_rate_vs_v3a),
        },
        "contextual_gap_closure_fractions_vs_3072_reference": contextual_gap,
        "best_variant_contextual_gap_gain_vs_v3a_reference": best_contextual_gap_gain_vs_v3a,
        "recommended_next_move": str(recommended_next_move),
    }, rationale


def _stage05_v3b_refinement_supports_lines(summary: dict[str, Any]) -> list[str]:
    lines = [
        "The diagnostic keeps Stage 05 mechanism-first and reuses the existing fixed-budget references where possible.",
        "The diagnostic stays within the current v3-B family and tests only narrow trajectory-curriculum refinements.",
    ]
    if summary["comparison_scope"] != "smoke_only":
        lines.extend(
            [
                f"Best refined v3-B variant: `{summary['best_variant_name']}`.",
                (
                    "A refined v3-B variant materially beats the v3-A reference on configured-step mechanism."
                    if bool(summary["narrow_v3b_refinement_materially_beats_v3a_reference"])
                    else "No refined v3-B variant materially beats the v3-A reference on configured-step mechanism."
                ),
                (
                    "A refined v3-B variant materially beats the v3-B control on configured-step mechanism."
                    if bool(summary["narrow_v3b_refinement_materially_beats_v3b_control"])
                    else "No refined v3-B variant materially beats the v3-B control on configured-step mechanism."
                ),
            ]
        )
    return lines


def _stage05_v3b_refinement_does_not_support_lines() -> list[str]:
    return [
        "This diagnostic does not reopen Stage 04 package-internal work.",
        "This diagnostic does not introduce a new Stage 05 mechanism family.",
        "This diagnostic does not justify replacing the frozen Stage 04 bridge on main.",
    ]


def _stage05_v3b_refinement_report_markdown(report: dict[str, Any]) -> str:
    protocol = report["comparison_protocol"]
    decision = report["decision"]
    lines = [
        "# Stage 05 v3-B Refinement Diagnostic",
        "",
        "## Protocol",
        f"- comparison scope: `{protocol['comparison_scope']}`",
        f"- dataset: `{protocol['dataset_name']}`",
        f"- seeds: `{protocol['seeds']}`",
        f"- shared batch size: `{protocol['shared_batch_size']}`",
        f"- Stage 05 epochs: `{protocol['stage05_v3b_control']['epochs']}`",
        "",
        "## Decision",
        f"- best variant: `{decision['best_variant_name']}`",
        f"- `narrow_v3b_refinement_materially_beats_v3b_control`: `{decision['narrow_v3b_refinement_materially_beats_v3b_control']}`",
        f"- `narrow_v3b_refinement_materially_beats_v3a_reference`: `{decision['narrow_v3b_refinement_materially_beats_v3a_reference']}`",
        f"- recommended next move: `{decision['recommended_next_move']}`",
        f"- rationale: `{decision['decision_rationale']}`",
        "",
        "## Pairwise Deltas Vs V3-B Control",
    ]
    for method_name, payload in report["pairwise_deltas_vs_v3b_control"].items():
        lines.append(
            f"- `{method_name}` configured-step energy delta vs identity delta: `{payload['configured_step_energy_delta_vs_identity_delta']['mean']}`"
        )
        lines.append(
            f"- `{method_name}` configured-step fixed-point residual delta vs identity delta: `{payload['configured_step_fixed_point_residual_delta_vs_identity_delta']['mean']}`"
        )
    lines.extend(["", "## Pairwise Deltas Vs V3-A"])
    for method_name, payload in report["pairwise_deltas_vs_v3a_reference"].items():
        lines.append(
            f"- `{method_name}` configured-step energy delta vs identity delta: `{payload['configured_step_energy_delta_vs_identity_delta']['mean']}`"
        )
        lines.append(
            f"- `{method_name}` configured-step fixed-point residual delta vs identity delta: `{payload['configured_step_fixed_point_residual_delta_vs_identity_delta']['mean']}`"
        )
    contextual_gap = report.get("contextual_gap_closure_fractions_vs_3072_reference")
    if isinstance(contextual_gap, dict):
        lines.extend(["", "## Contextual 3072 Gap Closure"])
        for method_name, payload in contextual_gap.items():
            lines.append(
                f"- `{method_name}` configured-step energy / residual gap closure: `{payload['configured_step_energy']}` / `{payload['configured_step_residual']}`"
            )
        return "\n".join(lines)
    return "\n".join(lines)


def _stage05_v2_longer_training_protocol_payload(
    config: Stage05V2LongerTrainingValidationConfig,
) -> dict[str, Any]:
    return {
        "dataset_name": config.dataset_name,
        "seeds": [int(seed) for seed in config.seeds],
        "train_fraction": float(config.train_fraction),
        "val_fraction": float(config.val_fraction),
        "test_fraction": float(config.test_fraction),
        "shared_batch_size": int(config.batch_size),
        "shared_shuffle_batches": bool(config.shuffle_batches),
        "current_budget_reference": {
            "method_name": STAGE05_V2_CURRENT_BUDGET_METHOD_NAME,
            "transport_family": "two_branch_residual_meanflow_core",
            "residual_branch_structure": "two_branch",
            "feature_aware_state_branch_tangents": True,
            "epochs": int(config.current_stage05_epochs),
            "eval_steps": int(config.stage05_eval_steps),
            "configured_transport_steps": int(config.stage05_transport_steps),
            "layer_dims": [int(value) for value in config.stage05_layer_dims],
        },
        "longer_training_candidate": {
            "method_name": STAGE05_V2_LONGER_TRAINING_METHOD_NAME,
            "transport_family": "two_branch_residual_meanflow_core",
            "residual_branch_structure": "two_branch",
            "feature_aware_state_branch_tangents": True,
            "epochs": int(config.longer_stage05_epochs),
            "eval_steps": int(config.stage05_eval_steps),
            "configured_transport_steps": int(config.stage05_transport_steps),
            "layer_dims": [int(value) for value in config.stage05_layer_dims],
        },
        "decision_rule": {
            "primary_split": "validation",
            "task_accuracy_is_report_only": True,
            "configured_step_improvement_fraction_threshold": float(
                config.configured_step_improvement_fraction_threshold
            ),
            "report_accuracy_improvement_threshold": float(
                config.report_accuracy_improvement_threshold
            ),
            "longer_budget_selection_rule_unchanged": True,
        },
    }


def _stage05_v2_longer_training_decision(
    *,
    rows: list[dict[str, Any]],
    config: Stage05V2LongerTrainingValidationConfig,
    by_method: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    current_rows = _method_rows(rows, STAGE05_V2_CURRENT_BUDGET_METHOD_NAME)
    longer_rows = _method_rows(rows, STAGE05_V2_LONGER_TRAINING_METHOD_NAME)
    current_by_seed = {int(row["seed"]): row for row in current_rows}
    longer_by_seed = {int(row["seed"]): row for row in longer_rows}
    shared_seeds = sorted(set(current_by_seed).intersection(longer_by_seed))
    if not shared_seeds:
        raise ValueError("Longer-training decision requires shared seeds.")

    current_summary = by_method[STAGE05_V2_CURRENT_BUDGET_METHOD_NAME]
    longer_summary = by_method[STAGE05_V2_LONGER_TRAINING_METHOD_NAME]
    current_energy_mean = float(
        current_summary["configured_step_energy_delta_vs_identity"]["mean"]
    )
    longer_energy_mean = float(
        longer_summary["configured_step_energy_delta_vs_identity"]["mean"]
    )
    current_residual_mean = float(
        current_summary["configured_step_fixed_point_residual_delta_vs_identity"]["mean"]
    )
    longer_residual_mean = float(
        longer_summary["configured_step_fixed_point_residual_delta_vs_identity"]["mean"]
    )
    current_val_accuracy_mean = float(current_summary["val_accuracy"]["mean"])
    longer_val_accuracy_mean = float(longer_summary["val_accuracy"]["mean"])
    current_test_accuracy_mean = float(current_summary["test_accuracy"]["mean"])
    longer_test_accuracy_mean = float(longer_summary["test_accuracy"]["mean"])

    configured_energy_gain_fraction = _negative_magnitude_relative_gain(
        current_value=current_energy_mean,
        candidate_value=longer_energy_mean,
    )
    configured_residual_gain_fraction = _negative_magnitude_relative_gain(
        current_value=current_residual_mean,
        candidate_value=longer_residual_mean,
    )
    energy_seed_improvement_rate = _rate(
        [
            float(longer_by_seed[seed]["configured_step_energy_delta_vs_identity"])
            < float(current_by_seed[seed]["configured_step_energy_delta_vs_identity"])
            for seed in shared_seeds
        ]
    )
    residual_seed_improvement_rate = _rate(
        [
            float(longer_by_seed[seed]["configured_step_fixed_point_residual_delta_vs_identity"])
            < float(current_by_seed[seed]["configured_step_fixed_point_residual_delta_vs_identity"])
            for seed in shared_seeds
        ]
    )
    configured_step_mechanism_improved_materially = bool(
        configured_energy_gain_fraction
        >= float(config.configured_step_improvement_fraction_threshold)
        and configured_residual_gain_fraction
        >= float(config.configured_step_improvement_fraction_threshold)
        and energy_seed_improvement_rate >= 0.5
        and residual_seed_improvement_rate >= 0.5
    )
    val_accuracy_gain = float(longer_val_accuracy_mean - current_val_accuracy_mean)
    test_accuracy_gain = float(longer_test_accuracy_mean - current_test_accuracy_mean)
    report_only_accuracy_improved_materially = bool(
        val_accuracy_gain >= float(config.report_accuracy_improvement_threshold)
        and test_accuracy_gain >= float(config.report_accuracy_improvement_threshold)
    )
    current_budget_boundary_all = bool(
        all(bool(row["selection_hits_final_training_boundary"]) for row in current_rows)
    )
    longer_budget_boundary_all = bool(
        all(bool(row["selection_hits_final_training_boundary"]) for row in longer_rows)
    )
    longer_budget_boundary_rate = _rate(
        [bool(row["selection_hits_final_training_boundary"]) for row in longer_rows]
    )
    recommended_next_move = (
        "continue_with_budget"
        if longer_budget_boundary_all
        else "open_stage05_v3_charter"
    )
    if longer_budget_boundary_all:
        rationale = (
            "The stronger Stage 05 v2 budget still selects the final training epoch on every seed, "
            "so the budget question is not yet closed."
        )
    elif configured_step_mechanism_improved_materially or report_only_accuracy_improved_materially:
        rationale = (
            "The stronger Stage 05 v2 budget improves the current v2 reference without still selecting "
            "the final training epoch on every seed, so the budget question is now better answered "
            "and the next step can move to a true v3 charter if needed."
        )
    else:
        rationale = (
            "The stronger Stage 05 v2 budget no longer looks boundary-limited and still does not "
            "materially improve the current v2 reference, so a true v3 mechanism charter is now justified."
        )
    decision = {
        STAGE05_V2_LONGER_TRAINING_DECISION_NAME: bool(
            configured_step_mechanism_improved_materially
        ),
        STAGE05_V2_LONGER_TRAINING_ACCURACY_DECISION_NAME: bool(
            report_only_accuracy_improved_materially
        ),
        "configured_step_energy_gain_fraction": float(configured_energy_gain_fraction),
        "configured_step_residual_gain_fraction": float(configured_residual_gain_fraction),
        "configured_step_energy_seed_improvement_rate": float(energy_seed_improvement_rate),
        "configured_step_residual_seed_improvement_rate": float(residual_seed_improvement_rate),
        "val_accuracy_gain": float(val_accuracy_gain),
        "test_accuracy_gain": float(test_accuracy_gain),
        "current_budget_selection_hits_final_training_boundary_on_all_seeds": bool(
            current_budget_boundary_all
        ),
        "longer_budget_selection_hits_final_training_boundary_on_all_seeds": bool(
            longer_budget_boundary_all
        ),
        "longer_budget_selection_hits_final_training_boundary_rate": float(
            longer_budget_boundary_rate
        ),
        "recommended_next_move": recommended_next_move,
    }
    return decision, rationale


def _stage05_v2_longer_training_supports_lines(
    *,
    decision: dict[str, Any],
    by_method: dict[str, dict[str, Any]],
) -> list[str]:
    current_summary = by_method[STAGE05_V2_CURRENT_BUDGET_METHOD_NAME]
    longer_summary = by_method[STAGE05_V2_LONGER_TRAINING_METHOD_NAME]
    return [
        (
            "The longer-training Stage 05 v2 candidate materially improves configured-step mechanism magnitude over the current budget."
            if decision[STAGE05_V2_LONGER_TRAINING_DECISION_NAME]
            else "The longer-training Stage 05 v2 candidate does not yet materially improve configured-step mechanism magnitude over the current budget."
        ),
        (
            "The longer-training Stage 05 v2 candidate materially improves report-only accuracy over the current budget."
            if decision[STAGE05_V2_LONGER_TRAINING_ACCURACY_DECISION_NAME]
            else "The longer-training Stage 05 v2 candidate does not yet materially improve report-only accuracy over the current budget."
        ),
        (
            "The stronger budget still hits the final training boundary on every seed."
            if decision["longer_budget_selection_hits_final_training_boundary_on_all_seeds"]
            else "The stronger budget no longer hits the final training boundary on every seed."
        ),
        (
            f"Current-budget configured-step validation energy delta vs identity mean: {current_summary['configured_step_energy_delta_vs_identity']['mean']:.12f}."
        ),
        (
            f"Longer-budget configured-step validation energy delta vs identity mean: {longer_summary['configured_step_energy_delta_vs_identity']['mean']:.12f}."
        ),
        (
            f"Current-budget validation/test accuracy means: {current_summary['val_accuracy']['mean']:.6f} / {current_summary['test_accuracy']['mean']:.6f}."
        ),
        (
            f"Longer-budget validation/test accuracy means: {longer_summary['val_accuracy']['mean']:.6f} / {longer_summary['test_accuracy']['mean']:.6f}."
        ),
    ]


def _stage05_v2_longer_training_does_not_support_lines(
    *,
    decision: dict[str, Any],
) -> list[str]:
    lines = [
        "This validation does not reopen Stage 04 package-internal work.",
        "This validation does not change the Stage 05 v2 transport family, objective family, or selection rule.",
        "This validation does not claim that Stage 05 replaces the frozen Stage 04 bridge result on main.",
    ]
    if decision["recommended_next_move"] == "continue_with_budget":
        lines.append(
            "This validation does not yet justify opening a true Stage 05 v3 mechanism charter."
        )
    else:
        lines.append("This validation does not imply any new Stage 04 replacement claim.")
    return lines


def _stage05_v2_longer_training_report_markdown(report: dict[str, Any]) -> str:
    protocol = report["comparison_protocol"]
    decision = report["decision"]
    lines = [
        "# Stage 05 V2 Longer-Training Validation",
        "",
        "## Protocol",
        f"- dataset: `{protocol['dataset_name']}`",
        f"- seeds: `{protocol['seeds']}`",
        f"- shared batch size: `{protocol['shared_batch_size']}`",
        f"- shared shuffle_batches: `{protocol['shared_shuffle_batches']}`",
        f"- current budget epochs: `{protocol['current_budget_reference']['epochs']}`",
        f"- longer budget epochs: `{protocol['longer_training_candidate']['epochs']}`",
        "",
        "## Decision",
        f"- `{STAGE05_V2_LONGER_TRAINING_DECISION_NAME}`: `{decision[STAGE05_V2_LONGER_TRAINING_DECISION_NAME]}`",
        f"- `{STAGE05_V2_LONGER_TRAINING_ACCURACY_DECISION_NAME}`: `{decision[STAGE05_V2_LONGER_TRAINING_ACCURACY_DECISION_NAME]}`",
        f"- longer budget still hits final training boundary on all seeds: `{decision['longer_budget_selection_hits_final_training_boundary_on_all_seeds']}`",
        f"- recommended next move: `{decision['recommended_next_move']}`",
        f"- rationale: `{decision['decision_rationale']}`",
        "",
        "## Supports",
    ]
    for item in report["supports"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Does Not Support"])
    for item in report["does_not_support"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _stage05_v2_longer_training_suite_config_payload(
    config: Stage05V2LongerTrainingValidationConfig,
) -> dict[str, Any]:
    return {
        "phase": "FMPC Stage 05 EF Core Probe",
        "stage": "stage05_v2_longer_training_validation",
        "comparison_protocol": _stage05_v2_longer_training_protocol_payload(config),
        "artifacts": {
            "aggregate_runs_csv": "aggregate_runs.csv",
            "aggregate_summary_json": "aggregate_summary.json",
            "comparison_report_json": "comparison_report.json",
            "comparison_report_md": "comparison_report.md",
        },
    }


def run_stage05_v2_longer_training_validation(
    config: Stage05V2LongerTrainingValidationConfig,
) -> FrozenBridgeVsCorrectedCoreComparisonRunResult:
    """Run a narrow longer-training validation on the existing Stage 05 v2 family."""

    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, config.resolved_run_id(), config.output_layout)
    )
    _write_json(run_dir / "config.json", _stage05_v2_longer_training_suite_config_payload(config))

    rows: list[dict[str, Any]] = []
    runs_root = run_dir / "runs"
    run_index = 0

    for seed in config.seeds:
        run_index += 1
        current_config = _stage05_v2_budget_config(
            config,
            seed=seed,
            output_root=runs_root,
            experiment_name=STAGE05_V2_CURRENT_BUDGET_METHOD_NAME,
            epochs=int(config.current_stage05_epochs),
        )
        current_result = run_fmpc_ef_exploratory_probe(current_config)
        rows.append(
            _stage05_core_row(
                run_index=run_index,
                suite_run_dir=run_dir,
                seed=seed,
                result=current_result,
                config=current_config,
                method_name=STAGE05_V2_CURRENT_BUDGET_METHOD_NAME,
                stage_name="FMPC Stage 05 EF Core Probe v2 Current Budget",
            )
        )

        run_index += 1
        longer_config = _stage05_v2_budget_config(
            config,
            seed=seed,
            output_root=runs_root,
            experiment_name=STAGE05_V2_LONGER_TRAINING_METHOD_NAME,
            epochs=int(config.longer_stage05_epochs),
        )
        longer_result = run_fmpc_ef_exploratory_probe(longer_config)
        rows.append(
            _stage05_core_row(
                run_index=run_index,
                suite_run_dir=run_dir,
                seed=seed,
                result=longer_result,
                config=longer_config,
                method_name=STAGE05_V2_LONGER_TRAINING_METHOD_NAME,
                stage_name="FMPC Stage 05 EF Core Probe v2 Longer Training",
            )
        )

    csv_rows = [
        {
            **row,
            "deterministic_artifact_checks_passed": str(bool(row["deterministic_artifact_checks_passed"])),
            "mechanism_signal_positive": str(bool(row["mechanism_signal_positive"])),
            "config_json_exists": str(bool(row["config_json_exists"])),
            "summary_json_exists": str(bool(row["summary_json_exists"])),
            "epoch_metrics_csv_exists": str(bool(row["epoch_metrics_csv_exists"])),
            "seed_matches": str(bool(row["seed_matches"])),
            "dataset_matches": str(bool(row["dataset_matches"])),
            "batch_protocol_matches": str(bool(row["batch_protocol_matches"])),
            "selection_hits_final_training_boundary": str(
                bool(row["selection_hits_final_training_boundary"])
            ),
        }
        for row in rows
    ]
    _write_csv(run_dir / "aggregate_runs.csv", csv_rows)

    by_method = {
        STAGE05_V2_CURRENT_BUDGET_METHOD_NAME: _method_summary(
            _method_rows(rows, STAGE05_V2_CURRENT_BUDGET_METHOD_NAME)
        ),
        STAGE05_V2_LONGER_TRAINING_METHOD_NAME: _method_summary(
            _method_rows(rows, STAGE05_V2_LONGER_TRAINING_METHOD_NAME)
        ),
    }
    pairwise_longer_vs_current = _pairwise_summary(
        rows,
        candidate_method=STAGE05_V2_LONGER_TRAINING_METHOD_NAME,
        reference_method=STAGE05_V2_CURRENT_BUDGET_METHOD_NAME,
    )
    decision, decision_rationale = _stage05_v2_longer_training_decision(
        rows=rows,
        config=config,
        by_method=by_method,
    )

    summary = {
        "phase": "FMPC Stage 05 EF Core Probe",
        "stage": "stage05_v2_longer_training_validation",
        "num_runs": int(len(rows)),
        "comparison_protocol": _stage05_v2_longer_training_protocol_payload(config),
        "by_method": by_method,
        "pairwise_longer_budget_vs_current_budget": pairwise_longer_vs_current,
        **decision,
        "decision_rationale": decision_rationale,
        "aggregate_runs_csv_path": "aggregate_runs.csv",
        "comparison_report_json_path": "comparison_report.json",
        "comparison_report_md_path": "comparison_report.md",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)

    report = {
        "comparison_protocol": _stage05_v2_longer_training_protocol_payload(config),
        "decision": {
            **decision,
            "decision_rationale": decision_rationale,
        },
        "supports": _stage05_v2_longer_training_supports_lines(
            decision=decision,
            by_method=by_method,
        ),
        "does_not_support": _stage05_v2_longer_training_does_not_support_lines(
            decision=decision,
        ),
    }
    _write_json(run_dir / "comparison_report.json", report)
    _write_text(run_dir / "comparison_report.md", _stage05_v2_longer_training_report_markdown(report))

    return FrozenBridgeVsCorrectedCoreComparisonRunResult(
        run_dir=run_dir,
        config=_stage05_v2_longer_training_suite_config_payload(config),
        aggregate_rows=rows,
        summary=summary,
        comparison_report=report,
    )


def _stage05_v2_budget_push_protocol_payload(
    config: Stage05V2BudgetPushValidationConfig,
) -> dict[str, Any]:
    return {
        "dataset_name": config.dataset_name,
        "seeds": [int(seed) for seed in config.seeds],
        "train_fraction": float(config.train_fraction),
        "val_fraction": float(config.val_fraction),
        "test_fraction": float(config.test_fraction),
        "shared_batch_size": int(config.batch_size),
        "shared_shuffle_batches": bool(config.shuffle_batches),
        "reference_budget": {
            "method_name": STAGE05_V2_BUDGET_REFERENCE_METHOD_NAME,
            "transport_family": "two_branch_residual_meanflow_core",
            "residual_branch_structure": "two_branch",
            "feature_aware_state_branch_tangents": True,
            "epochs": int(config.reference_stage05_epochs),
            "eval_steps": int(config.stage05_eval_steps),
            "configured_transport_steps": int(config.stage05_transport_steps),
            "layer_dims": [int(value) for value in config.stage05_layer_dims],
        },
        "stronger_budget_candidate": {
            "method_name": STAGE05_V2_BUDGET_PUSH_METHOD_NAME,
            "transport_family": "two_branch_residual_meanflow_core",
            "residual_branch_structure": "two_branch",
            "feature_aware_state_branch_tangents": True,
            "epochs": int(config.stronger_stage05_epochs),
            "eval_steps": int(config.stage05_eval_steps),
            "configured_transport_steps": int(config.stage05_transport_steps),
            "layer_dims": [int(value) for value in config.stage05_layer_dims],
        },
        "decision_rule": {
            "primary_split": "validation",
            "task_accuracy_is_report_only": True,
            "configured_step_improvement_fraction_threshold": float(
                config.configured_step_improvement_fraction_threshold
            ),
            "report_accuracy_improvement_threshold": float(
                config.report_accuracy_improvement_threshold
            ),
            "selection_rule_unchanged": True,
        },
    }


def _stage05_v2_budget_push_decision(
    *,
    rows: list[dict[str, Any]],
    config: Stage05V2BudgetPushValidationConfig,
    by_method: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    reference_rows = _method_rows(rows, STAGE05_V2_BUDGET_REFERENCE_METHOD_NAME)
    stronger_rows = _method_rows(rows, STAGE05_V2_BUDGET_PUSH_METHOD_NAME)
    reference_by_seed = {int(row["seed"]): row for row in reference_rows}
    stronger_by_seed = {int(row["seed"]): row for row in stronger_rows}
    shared_seeds = sorted(set(reference_by_seed).intersection(stronger_by_seed))
    if not shared_seeds:
        raise ValueError("Budget-push decision requires shared seeds.")

    reference_summary = by_method[STAGE05_V2_BUDGET_REFERENCE_METHOD_NAME]
    stronger_summary = by_method[STAGE05_V2_BUDGET_PUSH_METHOD_NAME]
    reference_energy_mean = float(
        reference_summary["configured_step_energy_delta_vs_identity"]["mean"]
    )
    stronger_energy_mean = float(
        stronger_summary["configured_step_energy_delta_vs_identity"]["mean"]
    )
    reference_residual_mean = float(
        reference_summary["configured_step_fixed_point_residual_delta_vs_identity"]["mean"]
    )
    stronger_residual_mean = float(
        stronger_summary["configured_step_fixed_point_residual_delta_vs_identity"]["mean"]
    )
    reference_val_accuracy_mean = float(reference_summary["val_accuracy"]["mean"])
    stronger_val_accuracy_mean = float(stronger_summary["val_accuracy"]["mean"])
    reference_test_accuracy_mean = float(reference_summary["test_accuracy"]["mean"])
    stronger_test_accuracy_mean = float(stronger_summary["test_accuracy"]["mean"])

    configured_energy_gain_fraction = _negative_magnitude_relative_gain(
        current_value=reference_energy_mean,
        candidate_value=stronger_energy_mean,
    )
    configured_residual_gain_fraction = _negative_magnitude_relative_gain(
        current_value=reference_residual_mean,
        candidate_value=stronger_residual_mean,
    )
    energy_seed_improvement_rate = _rate(
        [
            float(stronger_by_seed[seed]["configured_step_energy_delta_vs_identity"])
            < float(reference_by_seed[seed]["configured_step_energy_delta_vs_identity"])
            for seed in shared_seeds
        ]
    )
    residual_seed_improvement_rate = _rate(
        [
            float(stronger_by_seed[seed]["configured_step_fixed_point_residual_delta_vs_identity"])
            < float(reference_by_seed[seed]["configured_step_fixed_point_residual_delta_vs_identity"])
            for seed in shared_seeds
        ]
    )
    configured_step_mechanism_improved_materially = bool(
        configured_energy_gain_fraction
        >= float(config.configured_step_improvement_fraction_threshold)
        and configured_residual_gain_fraction
        >= float(config.configured_step_improvement_fraction_threshold)
        and energy_seed_improvement_rate >= 0.5
        and residual_seed_improvement_rate >= 0.5
    )
    val_accuracy_gain = float(stronger_val_accuracy_mean - reference_val_accuracy_mean)
    test_accuracy_gain = float(stronger_test_accuracy_mean - reference_test_accuracy_mean)
    report_only_accuracy_improved_materially = bool(
        val_accuracy_gain >= float(config.report_accuracy_improvement_threshold)
        and test_accuracy_gain >= float(config.report_accuracy_improvement_threshold)
    )
    reference_boundary_all = bool(
        all(bool(row["selection_hits_final_training_boundary"]) for row in reference_rows)
    )
    stronger_boundary_all = bool(
        all(bool(row["selection_hits_final_training_boundary"]) for row in stronger_rows)
    )
    stronger_boundary_rate = _rate(
        [bool(row["selection_hits_final_training_boundary"]) for row in stronger_rows]
    )
    configured_step_gain_fraction_vs_reference = float(
        min(configured_energy_gain_fraction, configured_residual_gain_fraction)
    )
    report_accuracy_gain_vs_reference = {
        "val_accuracy_delta": float(val_accuracy_gain),
        "test_accuracy_delta": float(test_accuracy_gain),
    }
    budget_line_still_looks_boundary_limited = bool(stronger_boundary_all)
    budget_line_should_continue = bool(
        configured_step_mechanism_improved_materially
        and report_only_accuracy_improved_materially
        and budget_line_still_looks_boundary_limited
    )
    budget_line_should_stop_and_open_v3 = bool(not budget_line_should_continue)
    budget_line_interpretation = (
        BOUNDARY_LIMITED_INTERPRETATION
        if budget_line_should_continue
        else STRUCTURALLY_INEFFICIENT_INTERPRETATION
    )
    recommended_next_move = (
        "continue_with_budget"
        if budget_line_should_continue
        else "open_stage05_v3_charter"
    )
    if budget_line_should_continue:
        rationale = (
            "The stronger Stage 05 v2 budget still materially improves configured-step mechanism and "
            "report-only accuracy while also selecting the final training epoch on every seed, so the "
            "same-family budget line still looks boundary-limited enough to continue."
        )
    elif budget_line_still_looks_boundary_limited:
        rationale = (
            "The stronger Stage 05 v2 budget still reaches the final training boundary on every seed, "
            "but the gains are no longer material enough across both configured-step mechanism and "
            "report-only accuracy, so the budget line should stop and a true Stage 05 v3 charter should open."
        )
    elif configured_step_mechanism_improved_materially or report_only_accuracy_improved_materially:
        rationale = (
            "The stronger Stage 05 v2 budget improves the same-family reference, but it no longer "
            "hits the final training boundary on every seed, so the budget line no longer looks clearly "
            "boundary-limited enough to justify continuing the same-family scaling pass."
        )
    else:
        rationale = (
            "The stronger Stage 05 v2 budget no longer looks boundary-limited and still does not "
            "materially improve the same-family reference, so a true v3 mechanism charter is now justified."
        )
    decision = {
        STAGE05_V2_BUDGET_PUSH_DECISION_NAME: bool(
            configured_step_mechanism_improved_materially
        ),
        STAGE05_V2_BUDGET_PUSH_ACCURACY_DECISION_NAME: bool(
            report_only_accuracy_improved_materially
        ),
        "configured_step_gain_fraction_vs_reference": configured_step_gain_fraction_vs_reference,
        "report_accuracy_gain_vs_reference": report_accuracy_gain_vs_reference,
        "budget_line_still_looks_boundary_limited": bool(
            budget_line_still_looks_boundary_limited
        ),
        "budget_line_should_continue": bool(budget_line_should_continue),
        "budget_line_should_stop_and_open_v3": bool(budget_line_should_stop_and_open_v3),
        "budget_line_interpretation": str(budget_line_interpretation),
        "configured_step_energy_gain_fraction": float(configured_energy_gain_fraction),
        "configured_step_residual_gain_fraction": float(configured_residual_gain_fraction),
        "configured_step_energy_seed_improvement_rate": float(energy_seed_improvement_rate),
        "configured_step_residual_seed_improvement_rate": float(residual_seed_improvement_rate),
        "val_accuracy_gain": float(val_accuracy_gain),
        "test_accuracy_gain": float(test_accuracy_gain),
        "reference_budget_selection_hits_final_training_boundary_on_all_seeds": bool(
            reference_boundary_all
        ),
        "budget_push_selection_hits_final_training_boundary_on_all_seeds": bool(
            stronger_boundary_all
        ),
        "budget_push_selection_hits_final_training_boundary_rate": float(
            stronger_boundary_rate
        ),
        "recommended_next_move": recommended_next_move,
    }
    return decision, rationale


def _stage05_v2_budget_push_supports_lines(
    *,
    decision: dict[str, Any],
    config: Stage05V2BudgetPushValidationConfig,
    by_method: dict[str, dict[str, Any]],
    contextual_accuracy_snapshot: dict[str, Any],
) -> list[str]:
    reference_summary = by_method[STAGE05_V2_BUDGET_REFERENCE_METHOD_NAME]
    stronger_summary = by_method[STAGE05_V2_BUDGET_PUSH_METHOD_NAME]
    contextual_note = contextual_accuracy_snapshot["note"]
    return [
        (
            f"The stronger-budget Stage 05 v2 candidate materially improves configured-step mechanism magnitude over the {int(config.reference_stage05_epochs)}-epoch reference."
            if decision[STAGE05_V2_BUDGET_PUSH_DECISION_NAME]
            else f"The stronger-budget Stage 05 v2 candidate does not yet materially improve configured-step mechanism magnitude over the {int(config.reference_stage05_epochs)}-epoch reference."
        ),
        (
            f"The stronger-budget Stage 05 v2 candidate materially improves report-only accuracy over the {int(config.reference_stage05_epochs)}-epoch reference."
            if decision[STAGE05_V2_BUDGET_PUSH_ACCURACY_DECISION_NAME]
            else f"The stronger-budget Stage 05 v2 candidate does not yet materially improve report-only accuracy over the {int(config.reference_stage05_epochs)}-epoch reference."
        ),
        (
            "The stronger budget still hits the final training boundary on every seed."
            if decision["budget_push_selection_hits_final_training_boundary_on_all_seeds"]
            else "The stronger budget no longer hits the final training boundary on every seed."
        ),
        (
            f"{int(config.reference_stage05_epochs)}-epoch reference configured-step validation energy delta vs identity mean: {reference_summary['configured_step_energy_delta_vs_identity']['mean']:.12f}."
        ),
        (
            f"{int(config.stronger_stage05_epochs)}-epoch candidate configured-step validation energy delta vs identity mean: {stronger_summary['configured_step_energy_delta_vs_identity']['mean']:.12f}."
        ),
        (
            f"{int(config.reference_stage05_epochs)}-epoch reference validation/test accuracy means: {reference_summary['val_accuracy']['mean']:.6f} / {reference_summary['test_accuracy']['mean']:.6f}."
        ),
        (
            f"{int(config.stronger_stage05_epochs)}-epoch candidate validation/test accuracy means: {stronger_summary['val_accuracy']['mean']:.6f} / {stronger_summary['test_accuracy']['mean']:.6f}."
        ),
        (
            f"Diagnostic-only context: Stage 05 v2 stronger-budget validation/test accuracy means remain at "
            f"{contextual_accuracy_snapshot['stage05_v2_stronger_budget']['val_accuracy_mean']:.6f} / "
            f"{contextual_accuracy_snapshot['stage05_v2_stronger_budget']['test_accuracy_mean']:.6f}, "
            f"versus frozen Stage 04 at {contextual_accuracy_snapshot['frozen_stage04_bridge']['val_accuracy']:.6f} / "
            f"{contextual_accuracy_snapshot['frozen_stage04_bridge']['test_accuracy']:.6f}, standalone digits_pc at "
            f"{contextual_accuracy_snapshot['digits_pc']['val_accuracy']:.6f} / "
            f"{contextual_accuracy_snapshot['digits_pc']['test_accuracy']:.6f}, and standalone digits_mlp at "
            f"{contextual_accuracy_snapshot['digits_mlp']['val_accuracy']:.6f} / "
            f"{contextual_accuracy_snapshot['digits_mlp']['test_accuracy']:.6f}."
        ),
        str(contextual_note),
        (
            "Interpretation: the Stage 05 v2 line is still behaving like a boundary-limited mechanism prototype."
            if decision["budget_line_interpretation"] == BOUNDARY_LIMITED_INTERPRETATION
            else "Interpretation: the Stage 05 v2 line now looks structurally inefficient under same-family budget escalation and should stop before a true v3 charter."
        ),
    ]


def _stage05_v2_budget_push_does_not_support_lines(
    *,
    decision: dict[str, Any],
) -> list[str]:
    lines = [
        "This validation does not reopen Stage 04 package-internal work.",
        "This validation does not change the Stage 05 v2 transport family, residual branch structure, corrected residual identity contract, or selection rule.",
        "This validation does not claim that Stage 05 replaces the frozen Stage 04 bridge result on main.",
    ]
    if decision["recommended_next_move"] == "continue_with_budget":
        lines.append(
            "This validation does not yet justify opening a true Stage 05 v3 mechanism charter."
        )
    else:
        lines.append("This validation does not imply any Stage 04 replacement claim.")
    return lines


def _stage05_v2_budget_push_report_markdown(report: dict[str, Any]) -> str:
    protocol = report["comparison_protocol"]
    decision = report["decision"]
    contextual = report["contextual_accuracy_note"]
    lines = [
        "# Stage 05 V2 Budget-Push Validation",
        "",
        "## Protocol",
        f"- dataset: `{protocol['dataset_name']}`",
        f"- seeds: `{protocol['seeds']}`",
        f"- shared batch size: `{protocol['shared_batch_size']}`",
        f"- shared shuffle_batches: `{protocol['shared_shuffle_batches']}`",
        f"- reference budget epochs: `{protocol['reference_budget']['epochs']}`",
        f"- stronger budget epochs: `{protocol['stronger_budget_candidate']['epochs']}`",
        "",
        "## Decision",
        f"- `{STAGE05_V2_BUDGET_PUSH_DECISION_NAME}`: `{decision[STAGE05_V2_BUDGET_PUSH_DECISION_NAME]}`",
        f"- `{STAGE05_V2_BUDGET_PUSH_ACCURACY_DECISION_NAME}`: `{decision[STAGE05_V2_BUDGET_PUSH_ACCURACY_DECISION_NAME]}`",
        f"- configured-step gain fraction vs reference: `{decision['configured_step_gain_fraction_vs_reference']}`",
        f"- report accuracy gain vs reference: `{decision['report_accuracy_gain_vs_reference']}`",
        f"- budget line still looks boundary-limited: `{decision['budget_line_still_looks_boundary_limited']}`",
        f"- budget line should continue: `{decision['budget_line_should_continue']}`",
        f"- budget line should stop and open v3: `{decision['budget_line_should_stop_and_open_v3']}`",
        f"- budget line interpretation: `{decision['budget_line_interpretation']}`",
        f"- stronger budget still hits final training boundary on all seeds: `{decision['budget_push_selection_hits_final_training_boundary_on_all_seeds']}`",
        f"- recommended next move: `{decision['recommended_next_move']}`",
        f"- rationale: `{decision['decision_rationale']}`",
        "",
        "## Contextual Accuracy Note",
        f"- Stage 05 v2 stronger budget validation/test accuracy means: `{contextual['stage05_v2_stronger_budget']['val_accuracy_mean']}` / `{contextual['stage05_v2_stronger_budget']['test_accuracy_mean']}`",
        f"- frozen Stage 04 validation/test accuracy: `{contextual['frozen_stage04_bridge']['val_accuracy']}` / `{contextual['frozen_stage04_bridge']['test_accuracy']}`",
        f"- standalone digits_pc validation/test accuracy: `{contextual['digits_pc']['val_accuracy']}` / `{contextual['digits_pc']['test_accuracy']}`",
        f"- standalone digits_mlp validation/test accuracy: `{contextual['digits_mlp']['val_accuracy']}` / `{contextual['digits_mlp']['test_accuracy']}`",
        f"- note: `{contextual['note']}`",
        "",
        "## Supports",
    ]
    for item in report["supports"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Does Not Support"])
    for item in report["does_not_support"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _stage05_v2_budget_push_suite_config_payload(
    config: Stage05V2BudgetPushValidationConfig,
) -> dict[str, Any]:
    return {
        "phase": "FMPC Stage 05 EF Core Probe",
        "stage": str(config.experiment_name),
        "comparison_protocol": _stage05_v2_budget_push_protocol_payload(config),
        "artifacts": {
            "aggregate_runs_csv": "aggregate_runs.csv",
            "aggregate_summary_json": "aggregate_summary.json",
            "comparison_report_json": "comparison_report.json",
            "comparison_report_md": "comparison_report.md",
        },
    }


def run_stage05_v2_budget_push_validation(
    config: Stage05V2BudgetPushValidationConfig,
) -> FrozenBridgeVsCorrectedCoreComparisonRunResult:
    """Run the next narrow Stage 05 v2 same-family budget-push validation."""

    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, config.resolved_run_id(), config.output_layout)
    )
    _write_json(run_dir / "config.json", _stage05_v2_budget_push_suite_config_payload(config))

    rows: list[dict[str, Any]] = []
    runs_root = run_dir / "runs"
    run_index = 0

    for seed in config.seeds:
        run_index += 1
        reference_config = _stage05_v2_budget_config(
            config,
            seed=seed,
            output_root=runs_root,
            experiment_name=STAGE05_V2_BUDGET_REFERENCE_METHOD_NAME,
            epochs=int(config.reference_stage05_epochs),
        )
        reference_result = run_fmpc_ef_exploratory_probe(reference_config)
        rows.append(
            _stage05_core_row(
                run_index=run_index,
                suite_run_dir=run_dir,
                seed=seed,
                result=reference_result,
                config=reference_config,
                method_name=STAGE05_V2_BUDGET_REFERENCE_METHOD_NAME,
                stage_name=(
                    f"FMPC Stage 05 EF Core Probe v2 {int(config.reference_stage05_epochs)}-Epoch Reference"
                ),
            )
        )

        run_index += 1
        stronger_config = _stage05_v2_budget_config(
            config,
            seed=seed,
            output_root=runs_root,
            experiment_name=STAGE05_V2_BUDGET_PUSH_METHOD_NAME,
            epochs=int(config.stronger_stage05_epochs),
        )
        stronger_result = run_fmpc_ef_exploratory_probe(stronger_config)
        rows.append(
            _stage05_core_row(
                run_index=run_index,
                suite_run_dir=run_dir,
                seed=seed,
                result=stronger_result,
                config=stronger_config,
                method_name=STAGE05_V2_BUDGET_PUSH_METHOD_NAME,
                stage_name=(
                    f"FMPC Stage 05 EF Core Probe v2 {int(config.stronger_stage05_epochs)}-Epoch Budget Push"
                ),
            )
        )

    csv_rows = [
        {
            **row,
            "deterministic_artifact_checks_passed": str(bool(row["deterministic_artifact_checks_passed"])),
            "mechanism_signal_positive": str(bool(row["mechanism_signal_positive"])),
            "config_json_exists": str(bool(row["config_json_exists"])),
            "summary_json_exists": str(bool(row["summary_json_exists"])),
            "epoch_metrics_csv_exists": str(bool(row["epoch_metrics_csv_exists"])),
            "seed_matches": str(bool(row["seed_matches"])),
            "dataset_matches": str(bool(row["dataset_matches"])),
            "batch_protocol_matches": str(bool(row["batch_protocol_matches"])),
            "selection_hits_final_training_boundary": str(
                bool(row["selection_hits_final_training_boundary"])
            ),
        }
        for row in rows
    ]
    _write_csv(run_dir / "aggregate_runs.csv", csv_rows)

    by_method = {
        STAGE05_V2_BUDGET_REFERENCE_METHOD_NAME: _method_summary(
            _method_rows(rows, STAGE05_V2_BUDGET_REFERENCE_METHOD_NAME)
        ),
        STAGE05_V2_BUDGET_PUSH_METHOD_NAME: _method_summary(
            _method_rows(rows, STAGE05_V2_BUDGET_PUSH_METHOD_NAME)
        ),
    }
    pairwise_budget_push_vs_reference = _pairwise_summary(
        rows,
        candidate_method=STAGE05_V2_BUDGET_PUSH_METHOD_NAME,
        reference_method=STAGE05_V2_BUDGET_REFERENCE_METHOD_NAME,
    )
    decision, decision_rationale = _stage05_v2_budget_push_decision(
        rows=rows,
        config=config,
        by_method=by_method,
    )
    contextual_accuracy_snapshot = _load_budget_push_contextual_accuracy_snapshot(
        stronger_summary=by_method[STAGE05_V2_BUDGET_PUSH_METHOD_NAME]
    )

    summary = {
        "phase": "FMPC Stage 05 EF Core Probe",
        "stage": str(config.experiment_name),
        "num_runs": int(len(rows)),
        "comparison_protocol": _stage05_v2_budget_push_protocol_payload(config),
        "by_method": by_method,
        "pairwise_budget_push_vs_reference_budget": pairwise_budget_push_vs_reference,
        "contextual_accuracy_note": contextual_accuracy_snapshot,
        **decision,
        "decision_rationale": decision_rationale,
        "aggregate_runs_csv_path": "aggregate_runs.csv",
        "comparison_report_json_path": "comparison_report.json",
        "comparison_report_md_path": "comparison_report.md",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)

    report = {
        "comparison_protocol": _stage05_v2_budget_push_protocol_payload(config),
        "decision": {
            **decision,
            "decision_rationale": decision_rationale,
        },
        "contextual_accuracy_note": contextual_accuracy_snapshot,
        "supports": _stage05_v2_budget_push_supports_lines(
            decision=decision,
            config=config,
            by_method=by_method,
            contextual_accuracy_snapshot=contextual_accuracy_snapshot,
        ),
        "does_not_support": _stage05_v2_budget_push_does_not_support_lines(
            decision=decision,
        ),
    }
    _write_json(run_dir / "comparison_report.json", report)
    _write_text(run_dir / "comparison_report.md", _stage05_v2_budget_push_report_markdown(report))

    return FrozenBridgeVsCorrectedCoreComparisonRunResult(
        run_dir=run_dir,
        config=_stage05_v2_budget_push_suite_config_payload(config),
        aggregate_rows=rows,
        summary=summary,
        comparison_report=report,
    )


def _stage05_v2_efficiency_protocol_payload(
    config: Stage05V2EfficiencyDiagnosticConfig,
) -> dict[str, Any]:
    return {
        "dataset_name": config.dataset_name,
        "seeds": [int(seed) for seed in config.seeds],
        "train_fraction": float(config.train_fraction),
        "val_fraction": float(config.val_fraction),
        "test_fraction": float(config.test_fraction),
        "shared_batch_size": int(config.batch_size),
        "shared_shuffle_batches": bool(config.shuffle_batches),
        "budget_ceiling_epochs": int(config.reference_stage05_epochs),
        "current_1536_default": {
            "method_name": STAGE05_V2_EFFICIENCY_REFERENCE_METHOD_NAME,
            "transport_family": "two_branch_residual_meanflow_core",
            "residual_branch_structure": "two_branch",
            "feature_aware_state_branch_tangents": True,
            "epochs": int(config.reference_stage05_epochs),
            "source_artifact_root": _repo_relative_posix(
                _resolve_repo_path(config.reference_artifact_root)
            ),
            "configured_transport_steps": int(config.stage05_transport_steps),
            "eval_steps": int(config.stage05_eval_steps),
            "layer_dims": [int(value) for value in config.stage05_layer_dims],
        },
        "optimized_1536_candidate": {
            "method_name": STAGE05_V2_EFFICIENCY_CANDIDATE_METHOD_NAME,
            "transport_family": "two_branch_residual_meanflow_core",
            "residual_branch_structure": "two_branch",
            "feature_aware_state_branch_tangents": True,
            "epochs": int(config.reference_stage05_epochs),
            "configured_transport_steps": int(config.stage05_transport_steps),
            "eval_steps": int(config.stage05_eval_steps),
            "layer_dims": [int(value) for value in config.stage05_layer_dims],
            "optimization_hypothesis": (
                "faster corrected residual identity curriculum at the same 1536-epoch ceiling"
            ),
            "tested_axes": {
                "lambda_id_warmup_epochs": int(config.optimized_lambda_id_warmup_epochs),
                "lambda_id_ramp_epochs": int(config.optimized_lambda_id_ramp_epochs),
            },
        },
        "contextual_3072_reference": {
            "source_summary_path": _repo_relative_posix(
                _resolve_repo_path(config.contextual_reference_summary_path)
            ),
            "method_name": STAGE05_V2_BUDGET_PUSH_METHOD_NAME,
            "epochs": int(config.contextual_reference_stage05_epochs),
        },
        "decision_rule": {
            "primary_split": "validation",
            "task_accuracy_is_report_only": True,
            "configured_step_improvement_fraction_threshold": float(
                config.configured_step_improvement_fraction_threshold
            ),
            "report_accuracy_improvement_threshold": float(
                config.report_accuracy_improvement_threshold
            ),
            "gap_narrowing_fraction_threshold": float(
                config.gap_narrowing_fraction_threshold
            ),
            "selection_rule_unchanged": True,
        },
    }


def _stage05_v2_efficiency_decision(
    *,
    rows: list[dict[str, Any]],
    config: Stage05V2EfficiencyDiagnosticConfig,
    by_method: dict[str, dict[str, Any]],
    contextual_reference: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    reference_rows = _method_rows(rows, STAGE05_V2_EFFICIENCY_REFERENCE_METHOD_NAME)
    candidate_rows = _method_rows(rows, STAGE05_V2_EFFICIENCY_CANDIDATE_METHOD_NAME)
    reference_by_seed = {int(row["seed"]): row for row in reference_rows}
    candidate_by_seed = {int(row["seed"]): row for row in candidate_rows}
    shared_seeds = sorted(set(reference_by_seed).intersection(candidate_by_seed))
    if not shared_seeds:
        raise ValueError("Efficiency diagnostic requires shared seeds.")

    reference_summary = by_method[STAGE05_V2_EFFICIENCY_REFERENCE_METHOD_NAME]
    candidate_summary = by_method[STAGE05_V2_EFFICIENCY_CANDIDATE_METHOD_NAME]
    contextual_energy_mean = float(
        contextual_reference["configured_step_energy_delta_vs_identity"]["mean"]
    )
    contextual_residual_mean = float(
        contextual_reference["configured_step_fixed_point_residual_delta_vs_identity"]["mean"]
    )
    contextual_val_accuracy_mean = float(contextual_reference["val_accuracy"]["mean"])
    contextual_test_accuracy_mean = float(contextual_reference["test_accuracy"]["mean"])

    reference_energy_mean = float(
        reference_summary["configured_step_energy_delta_vs_identity"]["mean"]
    )
    candidate_energy_mean = float(
        candidate_summary["configured_step_energy_delta_vs_identity"]["mean"]
    )
    reference_residual_mean = float(
        reference_summary["configured_step_fixed_point_residual_delta_vs_identity"]["mean"]
    )
    candidate_residual_mean = float(
        candidate_summary["configured_step_fixed_point_residual_delta_vs_identity"]["mean"]
    )
    reference_val_accuracy_mean = float(reference_summary["val_accuracy"]["mean"])
    candidate_val_accuracy_mean = float(candidate_summary["val_accuracy"]["mean"])
    reference_test_accuracy_mean = float(reference_summary["test_accuracy"]["mean"])
    candidate_test_accuracy_mean = float(candidate_summary["test_accuracy"]["mean"])

    configured_energy_gain_fraction = _negative_magnitude_relative_gain(
        current_value=reference_energy_mean,
        candidate_value=candidate_energy_mean,
    )
    configured_residual_gain_fraction = _negative_magnitude_relative_gain(
        current_value=reference_residual_mean,
        candidate_value=candidate_residual_mean,
    )
    energy_seed_improvement_rate = _rate(
        [
            float(candidate_by_seed[seed]["configured_step_energy_delta_vs_identity"])
            < float(reference_by_seed[seed]["configured_step_energy_delta_vs_identity"])
            for seed in shared_seeds
        ]
    )
    residual_seed_improvement_rate = _rate(
        [
            float(candidate_by_seed[seed]["configured_step_fixed_point_residual_delta_vs_identity"])
            < float(reference_by_seed[seed]["configured_step_fixed_point_residual_delta_vs_identity"])
            for seed in shared_seeds
        ]
    )
    configured_step_mechanism_improved_materially = bool(
        configured_energy_gain_fraction
        >= float(config.configured_step_improvement_fraction_threshold)
        and configured_residual_gain_fraction
        >= float(config.configured_step_improvement_fraction_threshold)
        and energy_seed_improvement_rate >= 0.5
        and residual_seed_improvement_rate >= 0.5
    )

    val_accuracy_gain = float(candidate_val_accuracy_mean - reference_val_accuracy_mean)
    test_accuracy_gain = float(candidate_test_accuracy_mean - reference_test_accuracy_mean)
    report_only_accuracy_improved_materially = bool(
        val_accuracy_gain >= float(config.report_accuracy_improvement_threshold)
        and test_accuracy_gain >= float(config.report_accuracy_improvement_threshold)
    )

    configured_energy_gap_closed_fraction = _negative_gap_closed_fraction(
        reference_value=reference_energy_mean,
        candidate_value=candidate_energy_mean,
        target_value=contextual_energy_mean,
    )
    configured_residual_gap_closed_fraction = _negative_gap_closed_fraction(
        reference_value=reference_residual_mean,
        candidate_value=candidate_residual_mean,
        target_value=contextual_residual_mean,
    )
    val_accuracy_gap_closed_fraction = _positive_gap_closed_fraction(
        reference_value=reference_val_accuracy_mean,
        candidate_value=candidate_val_accuracy_mean,
        target_value=contextual_val_accuracy_mean,
    )
    test_accuracy_gap_closed_fraction = _positive_gap_closed_fraction(
        reference_value=reference_test_accuracy_mean,
        candidate_value=candidate_test_accuracy_mean,
        target_value=contextual_test_accuracy_mean,
    )
    same_family_gap_narrowed_materially = bool(
        min(
            configured_energy_gap_closed_fraction,
            configured_residual_gap_closed_fraction,
            val_accuracy_gap_closed_fraction,
            test_accuracy_gap_closed_fraction,
        )
        >= float(config.gap_narrowing_fraction_threshold)
    )
    candidate_boundary_all = bool(
        all(bool(row["selection_hits_final_training_boundary"]) for row in candidate_rows)
    )

    if (
        configured_step_mechanism_improved_materially
        and report_only_accuracy_improved_materially
        and same_family_gap_narrowed_materially
    ):
        recommended_next_move = "adopt_better_same_family_optimization_and_stop_pure_budget_escalation"
        rationale = (
            "The narrow same-family schedule change materially improves configured-step mechanism and "
            "report-only accuracy at the same 1536-epoch ceiling while also closing a material fraction "
            "of the gap to the contextual 3072-epoch reference, so the better same-family setup should "
            "be adopted before any further pure budget escalation."
        )
    elif configured_step_mechanism_improved_materially or report_only_accuracy_improved_materially:
        recommended_next_move = "continue_with_budget"
        rationale = (
            "The narrow same-family schedule change improves the 1536-epoch reference but does not close "
            "enough of the gap to the 3072-epoch reference to justify replacing pure budget escalation yet."
        )
    else:
        recommended_next_move = "open_stage05_v3_charter"
        rationale = (
            "The narrow same-family schedule change does not materially improve the 1536-epoch reference, "
            "so same-family efficiency tuning is not a strong enough next move and a true Stage 05 v3 "
            "mechanism charter is now justified."
        )

    decision = {
        STAGE05_V2_EFFICIENCY_DECISION_NAME: bool(
            configured_step_mechanism_improved_materially
        ),
        STAGE05_V2_EFFICIENCY_ACCURACY_DECISION_NAME: bool(
            report_only_accuracy_improved_materially
        ),
        STAGE05_V2_EFFICIENCY_GAP_DECISION_NAME: bool(same_family_gap_narrowed_materially),
        "configured_step_gain_fraction_vs_reference": float(
            min(configured_energy_gain_fraction, configured_residual_gain_fraction)
        ),
        "report_accuracy_gain_vs_reference": {
            "val_accuracy_delta": float(val_accuracy_gain),
            "test_accuracy_delta": float(test_accuracy_gain),
        },
        "configured_step_gap_closed_fraction_vs_3072_reference": {
            "energy": float(configured_energy_gap_closed_fraction),
            "residual": float(configured_residual_gap_closed_fraction),
        },
        "report_accuracy_gap_closed_fraction_vs_3072_reference": {
            "val_accuracy": float(val_accuracy_gap_closed_fraction),
            "test_accuracy": float(test_accuracy_gap_closed_fraction),
        },
        "optimized_candidate_selection_hits_final_training_boundary_on_all_seeds": bool(
            candidate_boundary_all
        ),
        "recommended_next_move": str(recommended_next_move),
    }
    return decision, rationale


def _stage05_v2_efficiency_supports_lines(
    *,
    config: Stage05V2EfficiencyDiagnosticConfig,
    by_method: dict[str, dict[str, Any]],
    contextual_reference: dict[str, Any],
    decision: dict[str, Any],
) -> list[str]:
    reference_summary = by_method[STAGE05_V2_EFFICIENCY_REFERENCE_METHOD_NAME]
    candidate_summary = by_method[STAGE05_V2_EFFICIENCY_CANDIDATE_METHOD_NAME]
    return [
        (
            f"The optimized 1536-epoch Stage 05 v2 candidate materially improves configured-step mechanism over the current 1536-epoch default."
            if decision[STAGE05_V2_EFFICIENCY_DECISION_NAME]
            else "The optimized 1536-epoch Stage 05 v2 candidate does not materially improve configured-step mechanism over the current 1536-epoch default."
        ),
        (
            f"The optimized 1536-epoch Stage 05 v2 candidate materially improves report-only accuracy over the current 1536-epoch default."
            if decision[STAGE05_V2_EFFICIENCY_ACCURACY_DECISION_NAME]
            else "The optimized 1536-epoch Stage 05 v2 candidate does not materially improve report-only accuracy over the current 1536-epoch default."
        ),
        (
            f"The optimized 1536-epoch Stage 05 v2 candidate materially narrows the gap to the contextual 3072-epoch reference."
            if decision[STAGE05_V2_EFFICIENCY_GAP_DECISION_NAME]
            else "The optimized 1536-epoch Stage 05 v2 candidate does not materially narrow the gap to the contextual 3072-epoch reference."
        ),
        (
            f"Current 1536-epoch default configured-step validation energy delta vs identity mean: {reference_summary['configured_step_energy_delta_vs_identity']['mean']:.12f}."
        ),
        (
            f"Optimized 1536-epoch candidate configured-step validation energy delta vs identity mean: {candidate_summary['configured_step_energy_delta_vs_identity']['mean']:.12f}."
        ),
        (
            f"Contextual 3072-epoch reference configured-step validation energy delta vs identity mean: {contextual_reference['configured_step_energy_delta_vs_identity']['mean']:.12f}."
        ),
        (
            f"Current 1536-epoch default validation/test accuracy means: {reference_summary['val_accuracy']['mean']:.6f} / {reference_summary['test_accuracy']['mean']:.6f}."
        ),
        (
            f"Optimized 1536-epoch candidate validation/test accuracy means: {candidate_summary['val_accuracy']['mean']:.6f} / {candidate_summary['test_accuracy']['mean']:.6f}."
        ),
        (
            f"Contextual 3072-epoch reference validation/test accuracy means: {contextual_reference['val_accuracy']['mean']:.6f} / {contextual_reference['test_accuracy']['mean']:.6f}."
        ),
    ]


def _stage05_v2_efficiency_does_not_support_lines() -> list[str]:
    return [
        "This diagnostic does not reopen Stage 04 package-internal work.",
        "This diagnostic does not change the Stage 05 v2 transport family, residual branch structure, corrected residual identity contract, or selection rule.",
        "This diagnostic does not claim that Stage 05 replaces the frozen Stage 04 bridge result on main.",
    ]


def _stage05_v2_efficiency_report_markdown(report: dict[str, Any]) -> str:
    protocol = report["comparison_protocol"]
    decision = report["decision"]
    contextual = report["contextual_3072_reference"]
    lines = [
        "# Stage 05 V2 Efficiency Diagnostic At 1536",
        "",
        "## Protocol",
        f"- dataset: `{protocol['dataset_name']}`",
        f"- seeds: `{protocol['seeds']}`",
        f"- shared batch size: `{protocol['shared_batch_size']}`",
        f"- shared shuffle_batches: `{protocol['shared_shuffle_batches']}`",
        f"- fixed budget ceiling epochs: `{protocol['budget_ceiling_epochs']}`",
        f"- tested axes: `{protocol['optimized_1536_candidate']['tested_axes']}`",
        f"- contextual reference epochs: `{protocol['contextual_3072_reference']['epochs']}`",
        "",
        "## Decision",
        f"- `{STAGE05_V2_EFFICIENCY_DECISION_NAME}`: `{decision[STAGE05_V2_EFFICIENCY_DECISION_NAME]}`",
        f"- `{STAGE05_V2_EFFICIENCY_ACCURACY_DECISION_NAME}`: `{decision[STAGE05_V2_EFFICIENCY_ACCURACY_DECISION_NAME]}`",
        f"- `{STAGE05_V2_EFFICIENCY_GAP_DECISION_NAME}`: `{decision[STAGE05_V2_EFFICIENCY_GAP_DECISION_NAME]}`",
        f"- configured-step gain fraction vs reference: `{decision['configured_step_gain_fraction_vs_reference']}`",
        f"- report accuracy gain vs reference: `{decision['report_accuracy_gain_vs_reference']}`",
        f"- configured-step gap closed fraction vs 3072 reference: `{decision['configured_step_gap_closed_fraction_vs_3072_reference']}`",
        f"- report accuracy gap closed fraction vs 3072 reference: `{decision['report_accuracy_gap_closed_fraction_vs_3072_reference']}`",
        f"- optimized candidate still hits final training boundary on all seeds: `{decision['optimized_candidate_selection_hits_final_training_boundary_on_all_seeds']}`",
        f"- recommended next move: `{decision['recommended_next_move']}`",
        f"- rationale: `{decision['decision_rationale']}`",
        "",
        "## Contextual 3072 Reference",
        f"- source: `{contextual['source']}`",
        f"- validation/test accuracy means: `{contextual['val_accuracy']['mean']}` / `{contextual['test_accuracy']['mean']}`",
        f"- configured-step validation energy delta vs identity mean: `{contextual['configured_step_energy_delta_vs_identity']['mean']}`",
        f"- configured-step validation fixed-point residual delta vs identity mean: `{contextual['configured_step_fixed_point_residual_delta_vs_identity']['mean']}`",
        "",
        "## Supports",
    ]
    for item in report["supports"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Does Not Support"])
    for item in report["does_not_support"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _stage05_v2_efficiency_suite_config_payload(
    config: Stage05V2EfficiencyDiagnosticConfig,
) -> dict[str, Any]:
    return {
        "phase": "FMPC Stage 05 EF Core Probe",
        "stage": str(config.experiment_name),
        "comparison_protocol": _stage05_v2_efficiency_protocol_payload(config),
        "artifacts": {
            "aggregate_runs_csv": "aggregate_runs.csv",
            "aggregate_summary_json": "aggregate_summary.json",
            "comparison_report_json": "comparison_report.json",
            "comparison_report_md": "comparison_report.md",
        },
    }


def run_stage05_v2_efficiency_diagnostic(
    config: Stage05V2EfficiencyDiagnosticConfig,
) -> FrozenBridgeVsCorrectedCoreComparisonRunResult:
    """Run a narrow Stage 05 v2 same-family efficiency diagnostic at a fixed 1536-epoch ceiling."""

    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, config.resolved_run_id(), config.output_layout)
    )
    _write_json(run_dir / "config.json", _stage05_v2_efficiency_suite_config_payload(config))

    reference_artifact_root = _resolve_repo_path(config.reference_artifact_root)
    if not reference_artifact_root.exists():
        raise FileNotFoundError(
            f"Missing Stage 05 v2 efficiency reference artifacts at '{reference_artifact_root}'."
        )

    rows: list[dict[str, Any]] = []
    runs_root = run_dir / "runs"
    run_index = 0

    for seed in config.seeds:
        reference_run_dir = reference_artifact_root / f"seed_{seed}"
        if not reference_run_dir.exists():
            raise FileNotFoundError(
                f"Missing Stage 05 v2 efficiency reference seed artifact at '{reference_run_dir}'."
            )
        run_index += 1
        rows.append(
            _load_existing_stage05_core_row(
                run_index=run_index,
                existing_run_dir=reference_run_dir,
                seed=seed,
                expected_dataset_name=config.dataset_name,
                expected_batch_size=config.batch_size,
                expected_shuffle_batches=config.shuffle_batches,
                method_name=STAGE05_V2_EFFICIENCY_REFERENCE_METHOD_NAME,
                stage_name=(
                    f"FMPC Stage 05 EF Core Probe v2 {int(config.reference_stage05_epochs)}-Epoch Default"
                ),
            )
        )

        run_index += 1
        optimized_config = _stage05_v2_budget_config(
            config,
            seed=seed,
            output_root=runs_root,
            experiment_name=STAGE05_V2_EFFICIENCY_CANDIDATE_METHOD_NAME,
            epochs=int(config.reference_stage05_epochs),
            lambda_id_warmup_epochs=int(config.optimized_lambda_id_warmup_epochs),
            lambda_id_ramp_epochs=int(config.optimized_lambda_id_ramp_epochs),
        )
        optimized_result = run_fmpc_ef_exploratory_probe(optimized_config)
        rows.append(
            _stage05_core_row(
                run_index=run_index,
                suite_run_dir=run_dir,
                seed=seed,
                result=optimized_result,
                config=optimized_config,
                method_name=STAGE05_V2_EFFICIENCY_CANDIDATE_METHOD_NAME,
                stage_name=(
                    f"FMPC Stage 05 EF Core Probe v2 {int(config.reference_stage05_epochs)}-Epoch Efficiency Candidate"
                ),
            )
        )

    csv_rows = [
        {
            **row,
            "deterministic_artifact_checks_passed": str(bool(row["deterministic_artifact_checks_passed"])),
            "mechanism_signal_positive": str(bool(row["mechanism_signal_positive"])),
            "config_json_exists": str(bool(row["config_json_exists"])),
            "summary_json_exists": str(bool(row["summary_json_exists"])),
            "epoch_metrics_csv_exists": str(bool(row["epoch_metrics_csv_exists"])),
            "seed_matches": str(bool(row["seed_matches"])),
            "dataset_matches": str(bool(row["dataset_matches"])),
            "batch_protocol_matches": str(bool(row["batch_protocol_matches"])),
            "selection_hits_final_training_boundary": str(
                bool(row["selection_hits_final_training_boundary"])
            ),
        }
        for row in rows
    ]
    _write_csv(run_dir / "aggregate_runs.csv", csv_rows)

    by_method = {
        STAGE05_V2_EFFICIENCY_REFERENCE_METHOD_NAME: _method_summary(
            _method_rows(rows, STAGE05_V2_EFFICIENCY_REFERENCE_METHOD_NAME)
        ),
        STAGE05_V2_EFFICIENCY_CANDIDATE_METHOD_NAME: _method_summary(
            _method_rows(rows, STAGE05_V2_EFFICIENCY_CANDIDATE_METHOD_NAME)
        ),
    }
    contextual_reference = _load_stage05_v2_contextual_reference(config)
    pairwise_candidate_vs_default = _pairwise_summary(
        rows,
        candidate_method=STAGE05_V2_EFFICIENCY_CANDIDATE_METHOD_NAME,
        reference_method=STAGE05_V2_EFFICIENCY_REFERENCE_METHOD_NAME,
    )
    decision, decision_rationale = _stage05_v2_efficiency_decision(
        rows=rows,
        config=config,
        by_method=by_method,
        contextual_reference=contextual_reference,
    )

    summary = {
        "phase": "FMPC Stage 05 EF Core Probe",
        "stage": str(config.experiment_name),
        "num_runs": int(len(rows)),
        "comparison_protocol": _stage05_v2_efficiency_protocol_payload(config),
        "by_method": by_method,
        "contextual_3072_reference": contextual_reference,
        "pairwise_best_optimized_1536_vs_current_1536_default": pairwise_candidate_vs_default,
        **decision,
        "decision_rationale": decision_rationale,
        "aggregate_runs_csv_path": "aggregate_runs.csv",
        "comparison_report_json_path": "comparison_report.json",
        "comparison_report_md_path": "comparison_report.md",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)

    report = {
        "comparison_protocol": _stage05_v2_efficiency_protocol_payload(config),
        "decision": {
            **decision,
            "decision_rationale": decision_rationale,
        },
        "contextual_3072_reference": contextual_reference,
        "supports": _stage05_v2_efficiency_supports_lines(
            config=config,
            by_method=by_method,
            contextual_reference=contextual_reference,
            decision=decision,
        ),
        "does_not_support": _stage05_v2_efficiency_does_not_support_lines(),
    }
    _write_json(run_dir / "comparison_report.json", report)
    _write_text(run_dir / "comparison_report.md", _stage05_v2_efficiency_report_markdown(report))

    return FrozenBridgeVsCorrectedCoreComparisonRunResult(
        run_dir=run_dir,
        config=_stage05_v2_efficiency_suite_config_payload(config),
        aggregate_rows=rows,
        summary=summary,
        comparison_report=report,
    )


def run_stage05_v3b_refinement_diagnostic(
    config: Stage05V3BRefinementDiagnosticConfig,
) -> FrozenBridgeVsCorrectedCoreComparisonRunResult:
    """Run a narrow fixed-budget attribution diagnostic around the current Stage 05 v3-B scaffold."""

    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, config.resolved_run_id(), config.output_layout)
    )
    _write_json(run_dir / "config.json", _stage05_v3b_refinement_suite_config_payload(config))

    rows: list[dict[str, Any]] = []
    runs_root = run_dir / "runs"
    run_index = 0
    reuse_v2_root = _resolve_repo_path(config.reference_artifact_root)
    reuse_v3a_root = _resolve_repo_path(config.v3a_reference_artifact_root)
    reuse_v3b_root = _resolve_repo_path(config.v3b_control_artifact_root)

    if config.reuse_stage05_v2_reference_artifacts and not reuse_v2_root.exists():
        raise FileNotFoundError(
            f"Missing Stage 05 v2 reference artifacts at '{reuse_v2_root}'."
        )
    if config.reuse_stage05_v3a_reference_artifacts and not reuse_v3a_root.exists():
        raise FileNotFoundError(
            f"Missing Stage 05 v3-A reference artifacts at '{reuse_v3a_root}'."
        )
    if config.reuse_stage05_v3b_control_artifacts and not reuse_v3b_root.exists():
        raise FileNotFoundError(
            f"Missing Stage 05 v3-B control artifacts at '{reuse_v3b_root}'."
        )

    for seed in config.seeds:
        run_index += 1
        if config.reuse_stage05_v2_reference_artifacts:
            rows.append(
                _load_existing_stage05_core_row(
                    run_index=run_index,
                    existing_run_dir=reuse_v2_root / f"seed_{seed}",
                    seed=seed,
                    expected_dataset_name=config.dataset_name,
                    expected_batch_size=int(config.batch_size),
                    expected_shuffle_batches=bool(config.shuffle_batches),
                    method_name=STAGE05_V2_METHOD_NAME,
                    stage_name="FMPC Stage 05 EF Core Probe v2 Fixed-Budget Context Reference",
                )
            )
        else:
            v2_config = _build_stage05_v2_config(
                output_root=runs_root,
                experiment_name=STAGE05_V2_METHOD_NAME,
                seed=seed,
                train_fraction=float(config.train_fraction),
                val_fraction=float(config.val_fraction),
                test_fraction=float(config.test_fraction),
                batch_size=int(config.batch_size),
                shuffle_batches=bool(config.shuffle_batches),
                epochs=int(config.stage05_epochs),
                eval_steps=int(config.stage05_eval_steps),
                layer_dims=config.stage05_layer_dims,
                transport_steps=int(config.stage05_transport_steps),
            )
            v2_result = run_fmpc_ef_exploratory_probe(v2_config)
            rows.append(
                _stage05_core_row(
                    run_index=run_index,
                    suite_run_dir=run_dir,
                    seed=seed,
                    result=v2_result,
                    config=v2_config,
                    method_name=STAGE05_V2_METHOD_NAME,
                    stage_name="FMPC Stage 05 EF Core Probe v2 Fixed-Budget Context Reference",
                )
            )

        run_index += 1
        if config.reuse_stage05_v3a_reference_artifacts:
            rows.append(
                _load_existing_stage05_core_row(
                    run_index=run_index,
                    existing_run_dir=reuse_v3a_root / f"seed_{seed}",
                    seed=seed,
                    expected_dataset_name=config.dataset_name,
                    expected_batch_size=int(config.batch_size),
                    expected_shuffle_batches=bool(config.shuffle_batches),
                    method_name=STAGE05_V3A_METHOD_NAME,
                    stage_name="FMPC Stage 05 EF Core Probe v3-A Fixed-Budget Reference",
                )
            )
        else:
            v3a_config = _build_stage05_v2_config(
                output_root=runs_root,
                experiment_name=STAGE05_V3A_METHOD_NAME,
                seed=seed,
                train_fraction=float(config.train_fraction),
                val_fraction=float(config.val_fraction),
                test_fraction=float(config.test_fraction),
                batch_size=int(config.batch_size),
                shuffle_batches=bool(config.shuffle_batches),
                epochs=int(config.stage05_epochs),
                eval_steps=int(config.stage05_eval_steps),
                layer_dims=config.stage05_layer_dims,
                transport_steps=int(config.stage05_transport_steps),
                use_explicit_transport_drift_decomposition=True,
                lambda_drift=float(config.lambda_drift),
            )
            v3a_result = run_fmpc_ef_exploratory_probe(v3a_config)
            rows.append(
                _stage05_core_row(
                    run_index=run_index,
                    suite_run_dir=run_dir,
                    seed=seed,
                    result=v3a_result,
                    config=v3a_config,
                    method_name=STAGE05_V3A_METHOD_NAME,
                    stage_name="FMPC Stage 05 EF Core Probe v3-A Fixed-Budget Reference",
                )
            )

        run_index += 1
        if config.reuse_stage05_v3b_control_artifacts:
            rows.append(
                _load_existing_stage05_core_row(
                    run_index=run_index,
                    existing_run_dir=reuse_v3b_root / f"seed_{seed}",
                    seed=seed,
                    expected_dataset_name=config.dataset_name,
                    expected_batch_size=int(config.batch_size),
                    expected_shuffle_batches=bool(config.shuffle_batches),
                    method_name=STAGE05_V3B_METHOD_NAME,
                    stage_name="FMPC Stage 05 EF Core Probe v3-B Control",
                )
            )
        else:
            v3b_control_config = _stage05_v3b_variant_config(
                config,
                seed=seed,
                output_root=runs_root,
                experiment_name=STAGE05_V3B_METHOD_NAME,
                lambda_traj_curr=float(config.control_lambda_traj_curr),
                alpha_floor=float(config.control_alpha_floor),
                alpha_warmup_epochs=int(config.control_alpha_warmup_epochs),
                alpha_ramp_epochs=int(config.control_alpha_ramp_epochs),
            )
            v3b_control_result = run_fmpc_ef_exploratory_probe(v3b_control_config)
            rows.append(
                _stage05_core_row(
                    run_index=run_index,
                    suite_run_dir=run_dir,
                    seed=seed,
                    result=v3b_control_result,
                    config=v3b_control_config,
                    method_name=STAGE05_V3B_METHOD_NAME,
                    stage_name="FMPC Stage 05 EF Core Probe v3-B Control",
                )
            )

        for spec in _stage05_v3b_refinement_variant_specs(config):
            run_index += 1
            variant_config = _stage05_v3b_variant_config(
                config,
                seed=seed,
                output_root=runs_root,
                experiment_name=spec["method_name"],
                lambda_traj_curr=float(spec["lambda_traj_curr"]),
                alpha_floor=float(spec["alpha_floor"]),
                alpha_warmup_epochs=int(spec["alpha_warmup_epochs"]),
                alpha_ramp_epochs=int(spec["alpha_ramp_epochs"]),
            )
            variant_result = run_fmpc_ef_exploratory_probe(variant_config)
            rows.append(
                _stage05_core_row(
                    run_index=run_index,
                    suite_run_dir=run_dir,
                    seed=seed,
                    result=variant_result,
                    config=variant_config,
                    method_name=spec["method_name"],
                    stage_name=str(spec["stage_name"]),
                )
            )

    csv_rows = [
        {
            **row,
            "deterministic_artifact_checks_passed": str(bool(row["deterministic_artifact_checks_passed"])),
            "mechanism_signal_positive": str(bool(row["mechanism_signal_positive"])),
            "config_json_exists": str(bool(row["config_json_exists"])),
            "summary_json_exists": str(bool(row["summary_json_exists"])),
            "epoch_metrics_csv_exists": str(bool(row["epoch_metrics_csv_exists"])),
            "seed_matches": str(bool(row["seed_matches"])),
            "dataset_matches": str(bool(row["dataset_matches"])),
            "batch_protocol_matches": str(bool(row["batch_protocol_matches"])),
            "selection_hits_final_training_boundary": str(
                bool(row["selection_hits_final_training_boundary"])
            ),
        }
        for row in rows
    ]
    _write_csv(run_dir / "aggregate_runs.csv", csv_rows)

    method_names = [
        STAGE05_V2_METHOD_NAME,
        STAGE05_V3A_METHOD_NAME,
        STAGE05_V3B_METHOD_NAME,
        STAGE05_V3B_ALPHA_EARLIER_METHOD_NAME,
        STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME,
    ]
    by_method = {
        method_name: _method_summary(_method_rows(rows, method_name)) for method_name in method_names
    }
    pairwise_vs_control = {
        method_name: _pairwise_summary(
            rows,
            candidate_method=method_name,
            reference_method=STAGE05_V3B_METHOD_NAME,
        )
        for method_name in (
            STAGE05_V3B_ALPHA_EARLIER_METHOD_NAME,
            STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME,
        )
    }
    pairwise_vs_v3a = {
        method_name: _pairwise_summary(
            rows,
            candidate_method=method_name,
            reference_method=STAGE05_V3A_METHOD_NAME,
        )
        for method_name in (
            STAGE05_V3B_ALPHA_EARLIER_METHOD_NAME,
            STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME,
        )
    }
    contextual_reference = (
        None
        if config.comparison_scope == "smoke_only"
        else _load_stage05_v2_contextual_reference(config)
    )
    decision, decision_rationale = _stage05_v3b_refinement_decision(
        rows=rows,
        config=config,
        by_method=by_method,
        pairwise_vs_control=pairwise_vs_control,
        pairwise_vs_v3a=pairwise_vs_v3a,
        contextual_reference=contextual_reference,
    )
    configured_step_mechanism_ranking = [
        {
            "method_name": method_name,
            "configured_step_energy_delta_vs_identity_mean": float(
                by_method[method_name]["configured_step_energy_delta_vs_identity"]["mean"]
            ),
            "configured_step_fixed_point_residual_delta_vs_identity_mean": float(
                by_method[method_name]["configured_step_fixed_point_residual_delta_vs_identity"]["mean"]
            ),
        }
        for method_name in sorted(
            (
                STAGE05_V3A_METHOD_NAME,
                STAGE05_V3B_METHOD_NAME,
                STAGE05_V3B_ALPHA_EARLIER_METHOD_NAME,
                STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME,
            ),
            key=lambda method_name: (
                float(by_method[method_name]["configured_step_energy_delta_vs_identity"]["mean"]),
                float(
                    by_method[method_name]["configured_step_fixed_point_residual_delta_vs_identity"][
                        "mean"
                    ]
                ),
            ),
        )
    ]

    summary = {
        "phase": "FMPC Stage 05 EF Core Probe",
        "stage": str(config.experiment_name),
        "comparison_scope": str(config.comparison_scope),
        "num_runs": int(len(rows)),
        "comparison_protocol": _stage05_v3b_refinement_protocol_payload(config),
        "tested_variant_names": [
            STAGE05_V3B_METHOD_NAME,
            STAGE05_V3B_ALPHA_EARLIER_METHOD_NAME,
            STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME,
        ],
        "by_method": by_method,
        "pairwise_deltas_vs_v3b_control": pairwise_vs_control,
        "pairwise_deltas_vs_v3a_reference": pairwise_vs_v3a,
        "configured_step_mechanism_ranking": configured_step_mechanism_ranking,
        "contextual_3072_reference": contextual_reference,
        **decision,
        "decision_rationale": decision_rationale,
        "aggregate_runs_csv_path": "aggregate_runs.csv",
        "comparison_report_json_path": "comparison_report.json",
        "comparison_report_md_path": "comparison_report.md",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)

    report = {
        "comparison_protocol": _stage05_v3b_refinement_protocol_payload(config),
        "decision": {
            **decision,
            "decision_rationale": decision_rationale,
        },
        "pairwise_deltas_vs_v3b_control": pairwise_vs_control,
        "pairwise_deltas_vs_v3a_reference": pairwise_vs_v3a,
        "contextual_gap_closure_fractions_vs_3072_reference": decision.get(
            "contextual_gap_closure_fractions_vs_3072_reference"
        ),
        "supports": _stage05_v3b_refinement_supports_lines(summary),
        "does_not_support": _stage05_v3b_refinement_does_not_support_lines(),
    }
    _write_json(run_dir / "comparison_report.json", report)
    _write_text(run_dir / "comparison_report.md", _stage05_v3b_refinement_report_markdown(report))

    return FrozenBridgeVsCorrectedCoreComparisonRunResult(
        run_dir=run_dir,
        config=_stage05_v3b_refinement_suite_config_payload(config),
        aggregate_rows=rows,
        summary=summary,
        comparison_report=report,
    )


def _stage05_v3c_refinement_variant_specs(
    config: Stage05V3CRefinementDiagnosticConfig,
) -> list[dict[str, Any]]:
    return [
        {
            "method_name": STAGE05_V3C_STRONGER_SEMIGROUP_METHOD_NAME,
            "stage_name": "FMPC Stage 05 EF Core Probe v3-C Stronger Semigroup-Weight Variant",
            "lambda_sg": float(config.stronger_semigroup_lambda_sg),
        }
    ]


def _stage05_v3c_refinement_variant_payload(
    *,
    method_name: ComparisonMethodName,
    lambda_sg: float,
    config: Stage05V3CRefinementDiagnosticConfig,
) -> dict[str, Any]:
    return {
        "method_name": method_name,
        "candidate_name": method_name,
        "transport_family": "two_branch_residual_meanflow_core",
        "residual_branch_structure": "two_branch",
        "explicit_transport_drift_decomposition_enabled": True,
        "trajectory_curriculum_enabled": True,
        "endpoint_semigroup_consistency_enabled": True,
        "trajectory_curriculum_schedule_identity": "warmup_sigmoid_to_alpha_floor",
        "alpha_floor": float(config.promoted_v3b_alpha_floor),
        "lambda_traj_curr": float(config.promoted_v3b_lambda_traj_curr),
        "lambda_sg": float(lambda_sg),
        "semigroup_split_identity": "s = t + alpha * r; r_s = (1 - alpha) * r",
        "semigroup_target_mode": "single_sided_detached_split_endpoint",
        "semigroup_target_is_single_sided_detached": True,
    }


def _stage05_v3c_refinement_protocol_payload(
    config: Stage05V3CRefinementDiagnosticConfig,
) -> dict[str, Any]:
    if config.comparison_scope == "smoke_only":
        decision_rule = {
            "purpose": "smoke_ready_v3c_refinement_scaffold_check",
            "task_accuracy_is_report_only": True,
            "artifact_checks_required": True,
            "smoke_only": True,
        }
    else:
        decision_rule = {
            "purpose": "fixed_budget_v3c_refinement_attribution_diagnostic",
            "primary_split": "validation",
            "task_accuracy_is_report_only": True,
            "configured_step_improvement_fraction_threshold": float(
                config.configured_step_improvement_fraction_threshold
            ),
            "allowed_accuracy_regression_threshold": float(
                config.allowed_accuracy_regression_threshold
            ),
            "reuse_stage05_v2_reference_artifacts": bool(
                config.reuse_stage05_v2_reference_artifacts
            ),
            "reuse_stage05_v3b_reference_artifacts": bool(
                config.reuse_stage05_v3b_reference_artifacts
            ),
            "reuse_stage05_v3c_control_artifacts": bool(
                config.reuse_stage05_v3c_control_artifacts
            ),
        }
    return {
        "comparison_scope": str(config.comparison_scope),
        "dataset_name": config.dataset_name,
        "seeds": [int(seed) for seed in config.seeds],
        "train_fraction": float(config.train_fraction),
        "val_fraction": float(config.val_fraction),
        "test_fraction": float(config.test_fraction),
        "shared_batch_size": int(config.batch_size),
        "shared_shuffle_batches": bool(config.shuffle_batches),
        "fixed_budget_v2_control": {
            "method_name": STAGE05_V2_METHOD_NAME,
            "reference_reused_from_existing_artifacts": bool(
                config.reuse_stage05_v2_reference_artifacts
            ),
            "source_artifact_root": (
                _repo_relative_posix(_resolve_repo_path(config.reference_artifact_root))
                if config.reuse_stage05_v2_reference_artifacts
                else None
            ),
            "epochs": int(config.stage05_epochs),
            "configured_transport_steps": int(config.stage05_transport_steps),
        },
        "promoted_refined_v3b_reference": {
            **_stage05_v3c_refinement_variant_payload(
                method_name=STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME,
                lambda_sg=0.0,
                config=config,
            ),
            "endpoint_semigroup_consistency_enabled": False,
            "reference_reused_from_existing_artifacts": bool(
                config.reuse_stage05_v3b_reference_artifacts
            ),
            "source_artifact_root": (
                _repo_relative_posix(_resolve_repo_path(config.v3b_reference_artifact_root))
                if config.reuse_stage05_v3b_reference_artifacts
                else None
            ),
            "epochs": int(config.stage05_epochs),
            "configured_transport_steps": int(config.stage05_transport_steps),
        },
        "stage05_v3c_control": {
            **_stage05_v3c_refinement_variant_payload(
                method_name=STAGE05_V3C_METHOD_NAME,
                lambda_sg=float(config.control_lambda_sg),
                config=config,
            ),
            "reference_reused_from_existing_artifacts": bool(
                config.reuse_stage05_v3c_control_artifacts
            ),
            "source_artifact_root": (
                _repo_relative_posix(_resolve_repo_path(config.v3c_control_artifact_root))
                if config.reuse_stage05_v3c_control_artifacts
                else None
            ),
            "epochs": int(config.stage05_epochs),
            "configured_transport_steps": int(config.stage05_transport_steps),
        },
        "refinement_variants": [
            {
                **_stage05_v3c_refinement_variant_payload(
                    method_name=spec["method_name"],
                    lambda_sg=float(spec["lambda_sg"]),
                    config=config,
                ),
                "epochs": int(config.stage05_epochs),
                "configured_transport_steps": int(config.stage05_transport_steps),
            }
            for spec in _stage05_v3c_refinement_variant_specs(config)
        ],
        "contextual_3072_reference": {
            "source_summary_path": _repo_relative_posix(
                _resolve_repo_path(config.contextual_reference_summary_path)
            ),
            "method_name": STAGE05_V2_BUDGET_PUSH_METHOD_NAME,
            "epochs": int(config.contextual_reference_stage05_epochs),
        },
        "decision_rule": decision_rule,
    }


def _stage05_v3c_refinement_suite_config_payload(
    config: Stage05V3CRefinementDiagnosticConfig,
) -> dict[str, Any]:
    return {
        "phase": "FMPC Stage 05 EF Core Probe",
        "stage": str(config.experiment_name),
        "comparison_protocol": _stage05_v3c_refinement_protocol_payload(config),
        "artifacts": {
            "aggregate_runs_csv": "aggregate_runs.csv",
            "aggregate_summary_json": "aggregate_summary.json",
            "comparison_report_json": "comparison_report.json",
            "comparison_report_md": "comparison_report.md",
        },
    }


def _best_stage05_v3c_refinement_method(
    by_method: dict[str, dict[str, Any]],
) -> ComparisonMethodName:
    ranked = sorted(
        (STAGE05_V3C_STRONGER_SEMIGROUP_METHOD_NAME,),
        key=lambda method_name: (
            float(by_method[method_name]["configured_step_energy_delta_vs_identity"]["mean"]),
            float(
                by_method[method_name]["configured_step_fixed_point_residual_delta_vs_identity"][
                    "mean"
                ]
            ),
        ),
    )
    return ranked[0]


def _stage05_v3c_refinement_contextual_gap_payload(
    *,
    by_method: dict[str, dict[str, Any]],
    contextual_reference: dict[str, Any],
) -> dict[str, Any]:
    v2_summary = by_method[STAGE05_V2_METHOD_NAME]
    v2_configured_energy_mean = float(v2_summary["configured_step_energy_delta_vs_identity"]["mean"])
    v2_configured_residual_mean = float(
        v2_summary["configured_step_fixed_point_residual_delta_vs_identity"]["mean"]
    )
    v2_val_accuracy_mean = float(v2_summary["val_accuracy"]["mean"])
    v2_test_accuracy_mean = float(v2_summary["test_accuracy"]["mean"])

    payload: dict[str, Any] = {}
    for method_name in (
        STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME,
        STAGE05_V3C_METHOD_NAME,
        STAGE05_V3C_STRONGER_SEMIGROUP_METHOD_NAME,
    ):
        method_summary = by_method[method_name]
        payload[method_name] = {
            "configured_step_energy": float(
                _negative_gap_closed_fraction(
                    reference_value=v2_configured_energy_mean,
                    candidate_value=float(
                        method_summary["configured_step_energy_delta_vs_identity"]["mean"]
                    ),
                    target_value=float(
                        contextual_reference["configured_step_energy_delta_vs_identity"]["mean"]
                    ),
                )
            ),
            "configured_step_residual": float(
                _negative_gap_closed_fraction(
                    reference_value=v2_configured_residual_mean,
                    candidate_value=float(
                        method_summary["configured_step_fixed_point_residual_delta_vs_identity"][
                            "mean"
                        ]
                    ),
                    target_value=float(
                        contextual_reference[
                            "configured_step_fixed_point_residual_delta_vs_identity"
                        ]["mean"]
                    ),
                )
            ),
            "val_accuracy": float(
                _positive_gap_closed_fraction(
                    reference_value=v2_val_accuracy_mean,
                    candidate_value=float(method_summary["val_accuracy"]["mean"]),
                    target_value=float(contextual_reference["val_accuracy"]["mean"]),
                )
            ),
            "test_accuracy": float(
                _positive_gap_closed_fraction(
                    reference_value=v2_test_accuracy_mean,
                    candidate_value=float(method_summary["test_accuracy"]["mean"]),
                    target_value=float(contextual_reference["test_accuracy"]["mean"]),
                )
            ),
        }
    payload["refined_minus_v3c_control"] = {
        "configured_step_energy": float(
            payload[STAGE05_V3C_STRONGER_SEMIGROUP_METHOD_NAME]["configured_step_energy"]
            - payload[STAGE05_V3C_METHOD_NAME]["configured_step_energy"]
        ),
        "configured_step_residual": float(
            payload[STAGE05_V3C_STRONGER_SEMIGROUP_METHOD_NAME]["configured_step_residual"]
            - payload[STAGE05_V3C_METHOD_NAME]["configured_step_residual"]
        ),
    }
    payload["refined_minus_promoted_v3b"] = {
        "configured_step_energy": float(
            payload[STAGE05_V3C_STRONGER_SEMIGROUP_METHOD_NAME]["configured_step_energy"]
            - payload[STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME]["configured_step_energy"]
        ),
        "configured_step_residual": float(
            payload[STAGE05_V3C_STRONGER_SEMIGROUP_METHOD_NAME]["configured_step_residual"]
            - payload[STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME]["configured_step_residual"]
        ),
    }
    return payload


def _stage05_v3c_refinement_decision(
    *,
    rows: list[dict[str, Any]],
    config: Stage05V3CRefinementDiagnosticConfig,
    by_method: dict[str, dict[str, Any]],
    pairwise_vs_control: dict[str, dict[str, Any]],
    pairwise_vs_v3b: dict[str, dict[str, Any]],
    contextual_reference: dict[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    best_method = _best_stage05_v3c_refinement_method(by_method)
    best_rows = _method_rows(rows, best_method)
    best_vs_control = pairwise_vs_control[best_method]
    best_vs_v3b = pairwise_vs_v3b[best_method]

    artifact_pass = all(bool(row["deterministic_artifact_checks_passed"]) for row in best_rows)
    one_step_positive = all(
        float(row["one_step_energy_delta_vs_identity"]) < 0.0 for row in best_rows
    )
    configured_step_positive = all(
        float(row["configured_step_energy_delta_vs_identity"]) < 0.0
        and float(row["configured_step_fixed_point_residual_delta_vs_identity"]) < 0.0
        for row in best_rows
    )
    mechanism_positive = all(bool(row["mechanism_signal_positive"]) for row in best_rows)

    control_energy_mag = max(
        abs(float(by_method[STAGE05_V3C_METHOD_NAME]["configured_step_energy_delta_vs_identity"]["mean"])),
        1e-12,
    )
    control_residual_mag = max(
        abs(
            float(
                by_method[STAGE05_V3C_METHOD_NAME][
                    "configured_step_fixed_point_residual_delta_vs_identity"
                ]["mean"]
            )
        ),
        1e-12,
    )
    v3b_energy_mag = max(
        abs(
            float(
                by_method[STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME][
                    "configured_step_energy_delta_vs_identity"
                ]["mean"]
            )
        ),
        1e-12,
    )
    v3b_residual_mag = max(
        abs(
            float(
                by_method[STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME][
                    "configured_step_fixed_point_residual_delta_vs_identity"
                ]["mean"]
            )
        ),
        1e-12,
    )

    control_energy_delta = float(
        best_vs_control["configured_step_energy_delta_vs_identity_delta"]["mean"]
    )
    control_residual_delta = float(
        best_vs_control["configured_step_fixed_point_residual_delta_vs_identity_delta"]["mean"]
    )
    v3b_energy_delta = float(best_vs_v3b["configured_step_energy_delta_vs_identity_delta"]["mean"])
    v3b_residual_delta = float(
        best_vs_v3b["configured_step_fixed_point_residual_delta_vs_identity_delta"]["mean"]
    )

    materially_beats_control = bool(
        control_energy_delta < 0.0
        and control_residual_delta < 0.0
        and (
            abs(control_energy_delta)
            >= float(config.configured_step_improvement_fraction_threshold) * control_energy_mag
            or abs(control_residual_delta)
            >= float(config.configured_step_improvement_fraction_threshold) * control_residual_mag
        )
    )
    materially_beats_promoted_v3b = bool(
        v3b_energy_delta < 0.0
        and v3b_residual_delta < 0.0
        and (
            abs(v3b_energy_delta)
            >= float(config.configured_step_improvement_fraction_threshold) * v3b_energy_mag
            or abs(v3b_residual_delta)
            >= float(config.configured_step_improvement_fraction_threshold) * v3b_residual_mag
        )
    )
    avoids_accuracy_regression = bool(
        float(best_vs_v3b["val_accuracy_delta"]["mean"])
        >= -float(config.allowed_accuracy_regression_threshold)
        and float(best_vs_v3b["test_accuracy_delta"]["mean"])
        >= -float(config.allowed_accuracy_regression_threshold)
    )

    contextual_gap_payload: dict[str, Any] | None = None
    positive_gap_closure_vs_control = False
    positive_gap_closure_vs_promoted_v3b = False
    if contextual_reference is not None:
        contextual_gap_payload = _stage05_v3c_refinement_contextual_gap_payload(
            by_method=by_method,
            contextual_reference=contextual_reference,
        )
        refined_gap = contextual_gap_payload[best_method]
        control_gap = contextual_gap_payload[STAGE05_V3C_METHOD_NAME]
        promoted_gap = contextual_gap_payload[STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME]
        positive_gap_closure_vs_control = bool(
            refined_gap["configured_step_energy"] > control_gap["configured_step_energy"]
            and refined_gap["configured_step_residual"] >= control_gap["configured_step_residual"]
        )
        positive_gap_closure_vs_promoted_v3b = bool(
            refined_gap["configured_step_energy"] > promoted_gap["configured_step_energy"]
            and refined_gap["configured_step_residual"] >= promoted_gap["configured_step_residual"]
        )

    if config.comparison_scope == "smoke_only":
        recommended_next_move = "run_real_fixed_budget_v3c_refinement_diagnostic"
        rationale = (
            "The smoke run only verifies that the single-axis v3-C refinement scaffold is wired "
            "and comparable against the promoted refined v3-B reference."
        )
    elif (
        artifact_pass
        and one_step_positive
        and configured_step_positive
        and mechanism_positive
        and materially_beats_promoted_v3b
        and avoids_accuracy_regression
    ):
        recommended_next_move = "promote_refined_v3c_and_recompare"
        rationale = (
            "The stronger semigroup-weight refinement materially improves configured-step mechanism "
            "over the promoted refined v3-B reference while preserving the current mechanism-first gate."
        )
    elif (
        artifact_pass
        and one_step_positive
        and configured_step_positive
        and mechanism_positive
        and avoids_accuracy_regression
        and materially_beats_control
    ):
        recommended_next_move = "keep_v3c_diagnostic_only_and_stop_here"
        rationale = (
            "The stronger semigroup-weight refinement beats the current v3-C control but still "
            "does not materially displace the promoted refined v3-B reference."
        )
    else:
        recommended_next_move = "retain_promoted_v3b_as_active_reference"
        rationale = (
            "The stronger semigroup-weight refinement does not show a strong enough configured-step "
            "gain to justify moving past the promoted refined v3-B reference."
        )

    return (
        {
            "best_variant_name": str(best_method),
            "narrow_v3c_refinement_keeps_one_step_mechanism_positive": bool(one_step_positive),
            "narrow_v3c_refinement_keeps_configured_step_mechanism_positive": bool(
                configured_step_positive
            ),
            "narrow_v3c_refinement_avoids_obvious_report_accuracy_regression": bool(
                avoids_accuracy_regression
            ),
            "narrow_v3c_refinement_materially_beats_v3c_control": bool(materially_beats_control),
            "narrow_v3c_refinement_materially_beats_promoted_v3b_reference": bool(
                materially_beats_promoted_v3b
            ),
            "narrow_v3c_refinement_shows_positive_gap_closure_signal_vs_v3c_control": bool(
                positive_gap_closure_vs_control
            ),
            "narrow_v3c_refinement_shows_positive_gap_closure_signal_vs_promoted_v3b": bool(
                positive_gap_closure_vs_promoted_v3b
            ),
            "contextual_gap_closure_fractions_vs_3072_reference": contextual_gap_payload,
            "recommended_next_move": str(recommended_next_move),
        },
        rationale,
    )


def _stage05_v3c_refinement_supports_lines(summary: dict[str, Any]) -> list[str]:
    best_variant = str(summary["best_variant_name"])
    lines = [
        "The diagnostic keeps the Stage 05 mechanism-first gate unchanged.",
        "The diagnostic isolates one axis only: semigroup-probe weight.",
        f"The strongest tested refinement variant is `{best_variant}`.",
    ]
    if bool(summary["narrow_v3c_refinement_materially_beats_v3c_control"]):
        lines.append("The refined v3-C variant materially improves configured-step mechanism over the current v3-C control.")
    if bool(summary["narrow_v3c_refinement_materially_beats_promoted_v3b_reference"]):
        lines.append("The refined v3-C variant materially improves configured-step mechanism over the promoted refined v3-B reference.")
    return lines


def _stage05_v3c_refinement_does_not_support_lines() -> list[str]:
    return [
        "This diagnostic does not reopen Stage 04.",
        "This diagnostic does not change the Stage 05 family, trajectory scaffold, or semigroup target mode.",
        "This diagnostic does not turn report-only accuracy into the primary gate.",
    ]


def _stage05_v3c_refinement_report_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    pairwise_vs_control = report["pairwise_deltas_vs_v3c_control"][
        STAGE05_V3C_STRONGER_SEMIGROUP_METHOD_NAME
    ]
    pairwise_vs_v3b = report["pairwise_deltas_vs_promoted_refined_v3b_reference"][
        STAGE05_V3C_STRONGER_SEMIGROUP_METHOD_NAME
    ]
    lines = [
        "# Stage 05 v3-C Refinement Diagnostic",
        "",
        "## Protocol",
        f"- comparison scope: `{report['comparison_protocol']['comparison_scope']}`",
        f"- dataset: `{report['comparison_protocol']['dataset_name']}`",
        f"- seeds: `{report['comparison_protocol']['seeds']}`",
        f"- shared batch size: `{report['comparison_protocol']['shared_batch_size']}`",
        "",
        "## Decision",
        f"- best variant: `{decision['best_variant_name']}`",
        f"- `narrow_v3c_refinement_materially_beats_v3c_control`: `{decision['narrow_v3c_refinement_materially_beats_v3c_control']}`",
        f"- `narrow_v3c_refinement_materially_beats_promoted_v3b_reference`: `{decision['narrow_v3c_refinement_materially_beats_promoted_v3b_reference']}`",
        f"- `narrow_v3c_refinement_avoids_obvious_report_accuracy_regression`: `{decision['narrow_v3c_refinement_avoids_obvious_report_accuracy_regression']}`",
        f"- recommended next move: `{decision['recommended_next_move']}`",
        f"- rationale: `{decision['decision_rationale']}`",
        "",
        "## Pairwise Deltas Vs v3-C Control",
        f"- configured-step validation energy delta vs identity delta: `{pairwise_vs_control['configured_step_energy_delta_vs_identity_delta']['mean']}`",
        f"- configured-step validation fixed-point residual delta vs identity delta: `{pairwise_vs_control['configured_step_fixed_point_residual_delta_vs_identity_delta']['mean']}`",
        "",
        "## Pairwise Deltas Vs Promoted v3-B",
        f"- configured-step validation energy delta vs identity delta: `{pairwise_vs_v3b['configured_step_energy_delta_vs_identity_delta']['mean']}`",
        f"- configured-step validation fixed-point residual delta vs identity delta: `{pairwise_vs_v3b['configured_step_fixed_point_residual_delta_vs_identity_delta']['mean']}`",
    ]
    return "\n".join(lines)


def run_stage05_v3c_refinement_diagnostic(
    config: Stage05V3CRefinementDiagnosticConfig,
) -> FrozenBridgeVsCorrectedCoreComparisonRunResult:
    """Run a narrow fixed-budget attribution diagnostic around the current Stage 05 v3-C probe."""

    run_dir = _prepare_run_dir(
        _resolve_run_dir(config.output_root, config.experiment_name, config.resolved_run_id(), config.output_layout)
    )
    _write_json(run_dir / "config.json", _stage05_v3c_refinement_suite_config_payload(config))

    rows: list[dict[str, Any]] = []
    runs_root = run_dir / "runs"
    run_index = 0
    reuse_v2_root = _resolve_repo_path(config.reference_artifact_root)
    reuse_v3b_root = _resolve_repo_path(config.v3b_reference_artifact_root)
    reuse_v3c_root = _resolve_repo_path(config.v3c_control_artifact_root)

    if config.reuse_stage05_v2_reference_artifacts and not reuse_v2_root.exists():
        raise FileNotFoundError(f"Missing Stage 05 v2 reference artifacts at '{reuse_v2_root}'.")
    if config.reuse_stage05_v3b_reference_artifacts and not reuse_v3b_root.exists():
        raise FileNotFoundError(
            f"Missing promoted Stage 05 v3-B reference artifacts at '{reuse_v3b_root}'."
        )
    if config.reuse_stage05_v3c_control_artifacts and not reuse_v3c_root.exists():
        raise FileNotFoundError(f"Missing Stage 05 v3-C control artifacts at '{reuse_v3c_root}'.")

    for seed in config.seeds:
        run_index += 1
        if config.reuse_stage05_v2_reference_artifacts:
            rows.append(
                _load_existing_stage05_core_row(
                    run_index=run_index,
                    existing_run_dir=reuse_v2_root / f"seed_{seed}",
                    seed=seed,
                    expected_dataset_name=config.dataset_name,
                    expected_batch_size=int(config.batch_size),
                    expected_shuffle_batches=bool(config.shuffle_batches),
                    method_name=STAGE05_V2_METHOD_NAME,
                    stage_name="FMPC Stage 05 EF Core Probe v2 Control",
                    expected_total_training_epochs=int(config.stage05_epochs),
                    expected_eval_steps=int(config.stage05_eval_steps),
                    expected_layer_dims=config.stage05_layer_dims,
                    expected_transport_steps=int(config.stage05_transport_steps),
                    expected_transport_family="two_branch_residual_meanflow_core",
                    expected_explicit_transport_drift_decomposition_enabled=False,
                    expected_trajectory_curriculum_enabled=False,
                    expected_endpoint_semigroup_consistency_enabled=False,
                )
            )
        else:
            v2_config = _build_stage05_v2_config(
                output_root=runs_root,
                experiment_name=STAGE05_V2_METHOD_NAME,
                seed=seed,
                train_fraction=float(config.train_fraction),
                val_fraction=float(config.val_fraction),
                test_fraction=float(config.test_fraction),
                batch_size=int(config.batch_size),
                shuffle_batches=bool(config.shuffle_batches),
                epochs=int(config.stage05_epochs),
                eval_steps=int(config.stage05_eval_steps),
                layer_dims=config.stage05_layer_dims,
                transport_steps=int(config.stage05_transport_steps),
            )
            v2_result = run_fmpc_ef_exploratory_probe(v2_config)
            rows.append(
                _stage05_core_row(
                    run_index=run_index,
                    suite_run_dir=run_dir,
                    seed=seed,
                    result=v2_result,
                    config=v2_config,
                    method_name=STAGE05_V2_METHOD_NAME,
                    stage_name="FMPC Stage 05 EF Core Probe v2 Control",
                )
            )

        run_index += 1
        if config.reuse_stage05_v3b_reference_artifacts:
            rows.append(
                _load_existing_stage05_core_row(
                    run_index=run_index,
                    existing_run_dir=reuse_v3b_root / f"seed_{seed}",
                    seed=seed,
                    expected_dataset_name=config.dataset_name,
                    expected_batch_size=int(config.batch_size),
                    expected_shuffle_batches=bool(config.shuffle_batches),
                    method_name=STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME,
                    stage_name="FMPC Stage 05 EF Core Probe Promoted Refined v3-B Reference",
                    expected_total_training_epochs=int(config.stage05_epochs),
                    expected_eval_steps=int(config.stage05_eval_steps),
                    expected_layer_dims=config.stage05_layer_dims,
                    expected_transport_steps=int(config.stage05_transport_steps),
                    expected_transport_family="two_branch_residual_meanflow_core",
                    expected_explicit_transport_drift_decomposition_enabled=True,
                    expected_trajectory_curriculum_enabled=True,
                    expected_endpoint_semigroup_consistency_enabled=False,
                    expected_lambda_traj_curr=float(config.promoted_v3b_lambda_traj_curr),
                    expected_alpha_floor=float(config.promoted_v3b_alpha_floor),
                )
            )
        else:
            promoted_v3b_config = _stage05_promoted_v3b_reference_config(
                config,
                seed=seed,
                output_root=runs_root,
            )
            promoted_v3b_result = run_fmpc_ef_exploratory_probe(promoted_v3b_config)
            rows.append(
                _stage05_core_row(
                    run_index=run_index,
                    suite_run_dir=run_dir,
                    seed=seed,
                    result=promoted_v3b_result,
                    config=promoted_v3b_config,
                    method_name=STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME,
                    stage_name="FMPC Stage 05 EF Core Probe Promoted Refined v3-B Reference",
                )
            )

        run_index += 1
        if config.reuse_stage05_v3c_control_artifacts:
            rows.append(
                _load_existing_stage05_core_row(
                    run_index=run_index,
                    existing_run_dir=reuse_v3c_root / f"seed_{seed}",
                    seed=seed,
                    expected_dataset_name=config.dataset_name,
                    expected_batch_size=int(config.batch_size),
                    expected_shuffle_batches=bool(config.shuffle_batches),
                    method_name=STAGE05_V3C_METHOD_NAME,
                    stage_name="FMPC Stage 05 EF Core Probe v3-C Control",
                    expected_total_training_epochs=int(config.stage05_epochs),
                    expected_eval_steps=int(config.stage05_eval_steps),
                    expected_layer_dims=config.stage05_layer_dims,
                    expected_transport_steps=int(config.stage05_transport_steps),
                    expected_transport_family="two_branch_residual_meanflow_core",
                    expected_explicit_transport_drift_decomposition_enabled=True,
                    expected_trajectory_curriculum_enabled=True,
                    expected_endpoint_semigroup_consistency_enabled=True,
                    expected_lambda_traj_curr=float(config.promoted_v3b_lambda_traj_curr),
                    expected_alpha_floor=float(config.promoted_v3b_alpha_floor),
                    expected_lambda_sg=float(config.control_lambda_sg),
                )
            )
        else:
            v3c_control_config = _stage05_v3c_variant_config(
                config,
                seed=seed,
                output_root=runs_root,
                experiment_name=STAGE05_V3C_METHOD_NAME,
                lambda_sg=float(config.control_lambda_sg),
            )
            v3c_control_result = run_fmpc_ef_exploratory_probe(v3c_control_config)
            rows.append(
                _stage05_core_row(
                    run_index=run_index,
                    suite_run_dir=run_dir,
                    seed=seed,
                    result=v3c_control_result,
                    config=v3c_control_config,
                    method_name=STAGE05_V3C_METHOD_NAME,
                    stage_name="FMPC Stage 05 EF Core Probe v3-C Control",
                )
            )

        for spec in _stage05_v3c_refinement_variant_specs(config):
            run_index += 1
            refined_config = _stage05_v3c_variant_config(
                config,
                seed=seed,
                output_root=runs_root,
                experiment_name=spec["method_name"],
                lambda_sg=float(spec["lambda_sg"]),
            )
            refined_result = run_fmpc_ef_exploratory_probe(refined_config)
            rows.append(
                _stage05_core_row(
                    run_index=run_index,
                    suite_run_dir=run_dir,
                    seed=seed,
                    result=refined_result,
                    config=refined_config,
                    method_name=spec["method_name"],
                    stage_name=str(spec["stage_name"]),
                )
            )

    csv_rows = [
        {
            **row,
            "deterministic_artifact_checks_passed": str(bool(row["deterministic_artifact_checks_passed"])),
            "mechanism_signal_positive": str(bool(row["mechanism_signal_positive"])),
            "config_json_exists": str(bool(row["config_json_exists"])),
            "summary_json_exists": str(bool(row["summary_json_exists"])),
            "epoch_metrics_csv_exists": str(bool(row["epoch_metrics_csv_exists"])),
            "seed_matches": str(bool(row["seed_matches"])),
            "dataset_matches": str(bool(row["dataset_matches"])),
            "batch_protocol_matches": str(bool(row["batch_protocol_matches"])),
            "selection_hits_final_training_boundary": str(
                bool(row["selection_hits_final_training_boundary"])
            ),
        }
        for row in rows
    ]
    _write_csv(run_dir / "aggregate_runs.csv", csv_rows)

    method_names = [
        STAGE05_V2_METHOD_NAME,
        STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME,
        STAGE05_V3C_METHOD_NAME,
        STAGE05_V3C_STRONGER_SEMIGROUP_METHOD_NAME,
    ]
    by_method = {
        method_name: _method_summary(_method_rows(rows, method_name)) for method_name in method_names
    }
    pairwise_vs_control = {
        STAGE05_V3C_STRONGER_SEMIGROUP_METHOD_NAME: _pairwise_summary(
            rows,
            candidate_method=STAGE05_V3C_STRONGER_SEMIGROUP_METHOD_NAME,
            reference_method=STAGE05_V3C_METHOD_NAME,
        )
    }
    pairwise_vs_v3b = {
        STAGE05_V3C_METHOD_NAME: _pairwise_summary(
            rows,
            candidate_method=STAGE05_V3C_METHOD_NAME,
            reference_method=STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME,
        ),
        STAGE05_V3C_STRONGER_SEMIGROUP_METHOD_NAME: _pairwise_summary(
            rows,
            candidate_method=STAGE05_V3C_STRONGER_SEMIGROUP_METHOD_NAME,
            reference_method=STAGE05_V3B_STRONGER_TRAJ_METHOD_NAME,
        ),
    }
    pairwise_vs_v2 = {
        STAGE05_V3C_METHOD_NAME: _pairwise_summary(
            rows,
            candidate_method=STAGE05_V3C_METHOD_NAME,
            reference_method=STAGE05_V2_METHOD_NAME,
        ),
        STAGE05_V3C_STRONGER_SEMIGROUP_METHOD_NAME: _pairwise_summary(
            rows,
            candidate_method=STAGE05_V3C_STRONGER_SEMIGROUP_METHOD_NAME,
            reference_method=STAGE05_V2_METHOD_NAME,
        ),
    }
    contextual_reference = (
        None if config.comparison_scope == "smoke_only" else _load_stage05_v2_contextual_reference(config)
    )
    decision, decision_rationale = _stage05_v3c_refinement_decision(
        rows=rows,
        config=config,
        by_method=by_method,
        pairwise_vs_control=pairwise_vs_control,
        pairwise_vs_v3b=pairwise_vs_v3b,
        contextual_reference=contextual_reference,
    )
    configured_step_mechanism_ranking = [
        {
            "method_name": method_name,
            "configured_step_energy_delta_vs_identity_mean": float(
                by_method[method_name]["configured_step_energy_delta_vs_identity"]["mean"]
            ),
            "configured_step_fixed_point_residual_delta_vs_identity_mean": float(
                by_method[method_name]["configured_step_fixed_point_residual_delta_vs_identity"]["mean"]
            ),
        }
        for method_name in sorted(
            method_names,
            key=lambda method_name: (
                float(by_method[method_name]["configured_step_energy_delta_vs_identity"]["mean"]),
                float(
                    by_method[method_name]["configured_step_fixed_point_residual_delta_vs_identity"][
                        "mean"
                    ]
                ),
            ),
        )
    ]
    variant_settings = {
        STAGE05_V3C_METHOD_NAME: _stage05_v3c_refinement_variant_payload(
            method_name=STAGE05_V3C_METHOD_NAME,
            lambda_sg=float(config.control_lambda_sg),
            config=config,
        ),
        STAGE05_V3C_STRONGER_SEMIGROUP_METHOD_NAME: _stage05_v3c_refinement_variant_payload(
            method_name=STAGE05_V3C_STRONGER_SEMIGROUP_METHOD_NAME,
            lambda_sg=float(config.stronger_semigroup_lambda_sg),
            config=config,
        ),
    }

    summary = {
        "phase": "FMPC Stage 05 EF Core Probe",
        "stage": str(config.experiment_name),
        "comparison_scope": str(config.comparison_scope),
        "num_runs": int(len(rows)),
        "comparison_protocol": _stage05_v3c_refinement_protocol_payload(config),
        "tested_variant_names": [
            STAGE05_V3C_METHOD_NAME,
            STAGE05_V3C_STRONGER_SEMIGROUP_METHOD_NAME,
        ],
        "v3c_variant_settings": variant_settings,
        "by_method": by_method,
        "configured_step_mechanism_ranking": configured_step_mechanism_ranking,
        "pairwise_deltas_vs_v3c_control": pairwise_vs_control,
        "pairwise_deltas_vs_promoted_refined_v3b_reference": pairwise_vs_v3b,
        "pairwise_deltas_vs_stage05_v2_reference": pairwise_vs_v2,
        "contextual_3072_reference": contextual_reference,
        **decision,
        "decision_rationale": decision_rationale,
        "aggregate_runs_csv_path": "aggregate_runs.csv",
        "comparison_report_json_path": "comparison_report.json",
        "comparison_report_md_path": "comparison_report.md",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)

    report = {
        "comparison_protocol": _stage05_v3c_refinement_protocol_payload(config),
        "decision": {**decision, "decision_rationale": decision_rationale},
        "pairwise_deltas_vs_v3c_control": pairwise_vs_control,
        "pairwise_deltas_vs_promoted_refined_v3b_reference": pairwise_vs_v3b,
        "pairwise_deltas_vs_stage05_v2_reference": pairwise_vs_v2,
        "contextual_gap_closure_fractions_vs_3072_reference": decision.get(
            "contextual_gap_closure_fractions_vs_3072_reference"
        ),
        "supports": _stage05_v3c_refinement_supports_lines(summary),
        "does_not_support": _stage05_v3c_refinement_does_not_support_lines(),
    }
    _write_json(run_dir / "comparison_report.json", report)
    _write_text(run_dir / "comparison_report.md", _stage05_v3c_refinement_report_markdown(report))

    return FrozenBridgeVsCorrectedCoreComparisonRunResult(
        run_dir=run_dir,
        config=_stage05_v3c_refinement_suite_config_payload(config),
        aggregate_rows=rows,
        summary=summary,
        comparison_report=report,
    )
