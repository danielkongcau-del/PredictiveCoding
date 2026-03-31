"""Predictive coding baseline, experiment helpers, and Phase 2 comparison tools."""

from .comparison import ComparisonRunResult, run_benchmark_comparison
from .experiment import ExperimentConfig, ExperimentRunResult, run_supervised_experiment
from .layers import PCLayerParams, init_mlp_layers
from .mlp_baseline import MLPNetwork, MLPLayerParams, init_mlp_baseline_layers
from .models import PCNetwork

__all__ = [
    "ComparisonRunResult",
    "ExperimentConfig",
    "ExperimentRunResult",
    "MLPLayerParams",
    "MLPNetwork",
    "PCLayerParams",
    "PCNetwork",
    "init_mlp_layers",
    "init_mlp_baseline_layers",
    "run_benchmark_comparison",
    "run_supervised_experiment",
]
