"""Predictive coding baseline, experiment helpers, and Phase 2 analysis tools."""

from .comparison import ComparisonRunResult, run_benchmark_comparison
from .experiment import ExperimentConfig, ExperimentRunResult, run_supervised_experiment
from .layers import PCLayerParams, init_mlp_layers
from .mlp_baseline import MLPNetwork, MLPLayerParams, init_mlp_baseline_layers
from .models import PCNetwork
from .pc_sensitivity import PCSensitivityRunResult, run_pc_sensitivity_study

__all__ = [
    "ComparisonRunResult",
    "ExperimentConfig",
    "ExperimentRunResult",
    "MLPLayerParams",
    "MLPNetwork",
    "PCLayerParams",
    "PCSensitivityRunResult",
    "PCNetwork",
    "init_mlp_layers",
    "init_mlp_baseline_layers",
    "run_benchmark_comparison",
    "run_pc_sensitivity_study",
    "run_supervised_experiment",
]
