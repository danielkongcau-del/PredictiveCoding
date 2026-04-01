"""Predictive coding baseline, experiment helpers, and Phase 2 analysis tools."""

from .comparison import ComparisonRunResult, run_benchmark_comparison
from .pc_budget_tradeoff import PCBudgetTradeoffRunResult, run_pc_budget_tradeoff_study
from .experiment import ExperimentConfig, ExperimentRunResult, run_supervised_experiment
from .layers import PCLayerParams, init_mlp_layers
from .mlp_baseline import MLPNetwork, MLPLayerParams, init_mlp_baseline_layers
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
    "MLPLayerParams",
    "MLPNetwork",
    "PCBudgetTradeoffRunResult",
    "PCJointSearchRunResult",
    "Phase2G1BoundaryCheckRunResult",
    "PCDiagnosticsRunResult",
    "PCMultiSeedRunResult",
    "PCLayerParams",
    "PCSensitivityRunResult",
    "PCNetwork",
    "init_mlp_layers",
    "init_mlp_baseline_layers",
    "run_benchmark_comparison",
    "run_phase2g1_boundary_check",
    "run_pc_budget_tradeoff_study",
    "run_pc_joint_search",
    "run_pc_diagnostics_study",
    "run_pc_multiseed_study",
    "run_pc_sensitivity_study",
    "run_supervised_experiment",
]
