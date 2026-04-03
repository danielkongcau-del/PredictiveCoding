"""Predictive coding baseline, experiment helpers, and Phase 2 analysis tools."""

from .comparison import ComparisonRunResult, run_benchmark_comparison
from .datasets import load_digits_split, load_fashion_mnist_split
from .pc_budget_tradeoff import PCBudgetTradeoffRunResult, run_pc_budget_tradeoff_study
from .experiment import ExperimentConfig, ExperimentRunResult, run_supervised_experiment
from .fmpc_protocol import (
    FMPCPreparationConfig,
    FMPCPreparationRunResult,
    FMPC_PREPARATION_SCHEMA_VERSION,
    PC_TEACHER_CHECKPOINT_FORMAT,
    load_pc_teacher_checkpoint,
    load_prepared_teacher_runtime,
    resolve_fmpc_teacher_manifest_path,
    save_pc_teacher_checkpoint,
    run_fmpc_v0_preparation,
)
from .fmpc_student_data import (
    FMPCStudentDataset,
    FMPCStudentSplit,
    build_fmpc_student_inputs,
    load_fmpc_student_dataset,
)
from .fmpc_student_normalization import (
    FMPCStudentNormalizationStats,
    fit_fmpc_student_normalization,
)
from .fmpc_student_baselines import (
    ClassMeanDeltaStudent,
    RidgeDeltaStudent,
    RidgeDeltaStudentConfig,
    StandardizedMLPStudent,
    StandardizedMLPStudentConfig,
    identity_delta_prediction,
)
from .fmpc_student import (
    FMPCStudentConfig,
    FMPCStudentRunResult,
    FMPCStudentTransporter,
    evaluate_fmpc_delta_predictions,
    evaluate_fmpc_identity_baseline,
    fmpc_split_evaluation_metrics_payload,
    init_fmpc_student_transporter,
    load_fmpc_student_teacher_runtime,
    prepare_fmpc_student_teacher_references,
    run_fmpc_student_experiment,
)
from .fmpc_student_suite import (
    FMPCStudentSuiteConfig,
    FMPCStudentSuiteRunResult,
    run_fmpc_student_suite,
)
from .inference import TeacherInferenceExport, run_teacher_inference_export
from .layers import PCLayerParams, init_mlp_layers
from .metrics import (
    energy_gap_to_teacher,
    hidden_state_l2_distance,
    hidden_state_rms_gap,
    state_update_direction_cosine,
    summarize_teacher_reference_metrics,
    update_direction_cosine,
)
from .mlp_baseline import MLPNetwork, MLPLayerParams, init_mlp_baseline_layers
from .minibatch import iter_minibatches
from .real_mlp import RealMLPConfig, RealMLPRunResult, run_digits_mlp_experiment
from .real_pc_inference_baselines import (
    RealPCInferenceBaselineCandidate,
    RealPCInferenceBaselineStudyConfig,
    RealPCInferenceBaselineStudyResult,
    run_real_pc_inference_baseline_study,
)
from .real_pc import RealPCConfig, RealPCRunResult, run_digits_pc_experiment, run_real_pc_experiment
from .state_io import flatten_hidden_states, unflatten_hidden_states
from .phase2g1_boundary_check import (
    Phase2G1BoundaryCheckRunResult,
    run_phase2g1_boundary_check,
)
from .pc_joint_search import PCJointSearchRunResult, run_pc_joint_search
from .models import PCNetwork
from .pc_diagnostics import PCDiagnosticsRunResult, run_pc_diagnostics_study
from .pc_multiseed import PCMultiSeedRunResult, run_pc_multiseed_study
from .pc_sensitivity import PCSensitivityRunResult, run_pc_sensitivity_study

__all__ = [
    "ComparisonRunResult",
    "ExperimentConfig",
    "ExperimentRunResult",
    "FMPCPreparationConfig",
    "FMPC_PREPARATION_SCHEMA_VERSION",
    "FMPCPreparationRunResult",
    "FMPCStudentConfig",
    "FMPCStudentDataset",
    "FMPCStudentNormalizationStats",
    "FMPCStudentRunResult",
    "FMPCStudentSplit",
    "FMPCStudentSuiteConfig",
    "FMPCStudentSuiteRunResult",
    "FMPCStudentTransporter",
    "ClassMeanDeltaStudent",
    "MLPLayerParams",
    "MLPNetwork",
    "PCBudgetTradeoffRunResult",
    "PC_TEACHER_CHECKPOINT_FORMAT",
    "PCJointSearchRunResult",
    "Phase2G1BoundaryCheckRunResult",
    "PCDiagnosticsRunResult",
    "PCMultiSeedRunResult",
    "PCLayerParams",
    "PCSensitivityRunResult",
    "PCNetwork",
    "RealMLPConfig",
    "RealMLPRunResult",
    "RealPCInferenceBaselineCandidate",
    "RealPCInferenceBaselineStudyConfig",
    "RealPCInferenceBaselineStudyResult",
    "RealPCConfig",
    "RealPCRunResult",
    "TeacherInferenceExport",
    "build_fmpc_student_inputs",
    "energy_gap_to_teacher",
    "evaluate_fmpc_delta_predictions",
    "evaluate_fmpc_identity_baseline",
    "flatten_hidden_states",
    "fit_fmpc_student_normalization",
    "fmpc_split_evaluation_metrics_payload",
    "hidden_state_l2_distance",
    "hidden_state_rms_gap",
    "identity_delta_prediction",
    "init_fmpc_student_transporter",
    "init_mlp_layers",
    "init_mlp_baseline_layers",
    "iter_minibatches",
    "load_fmpc_student_teacher_runtime",
    "load_digits_split",
    "load_pc_teacher_checkpoint",
    "load_prepared_teacher_runtime",
    "load_fmpc_student_dataset",
    "load_fashion_mnist_split",
    "prepare_fmpc_student_teacher_references",
    "resolve_fmpc_teacher_manifest_path",
    "run_benchmark_comparison",
    "run_digits_mlp_experiment",
    "run_digits_pc_experiment",
    "run_fmpc_v0_preparation",
    "run_fmpc_student_experiment",
    "run_fmpc_student_suite",
    "run_real_pc_inference_baseline_study",
    "run_real_pc_experiment",
    "run_teacher_inference_export",
    "run_phase2g1_boundary_check",
    "run_pc_budget_tradeoff_study",
    "run_pc_joint_search",
    "run_pc_diagnostics_study",
    "run_pc_multiseed_study",
    "run_pc_sensitivity_study",
    "run_supervised_experiment",
    "save_pc_teacher_checkpoint",
    "RidgeDeltaStudent",
    "RidgeDeltaStudentConfig",
    "StandardizedMLPStudent",
    "StandardizedMLPStudentConfig",
    "state_update_direction_cosine",
    "summarize_teacher_reference_metrics",
    "unflatten_hidden_states",
    "update_direction_cosine",
]
