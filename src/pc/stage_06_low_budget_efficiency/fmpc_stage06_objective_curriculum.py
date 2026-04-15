from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np

from ..activations import get_activation
from ..datasets import load_digits_split
from ..minibatch import iter_minibatches
from ..models import PCNetwork
from ..mlp_baseline import MLPNetwork
from ..stage_03_transport_core_v1.fmpc_tf1_flow import (
    FMPCTF1Context,
    bootstrap_average_velocity_target,
    build_tf1_context,
    hidden_local_flow,
    rollout_hidden_transport,
    teacher_free_state_features,
)
from ..stage_05_ef_core_probe.fmpc_ef_exploratory_probe import (
    EndpointSemigroupTargets,
    FMPCEFExploratoryProbeConfig,
    Stage05ResidualCoreNetworks,
    TrajectoryCurriculumTargets,
    _as_batch_first,
    _evaluate_mechanism_metrics,
    _evaluate_slow_pc_metrics,
    _forward_mlp,
    _learned_velocity_fn,
    _make_pc_model,
    _make_psi_network,
    _predict_residual_from_inputs,
    _prepare_run_dir,
    _residual_core_inputs_for_state,
    _resolve_run_dir,
    _restore_pc_parameters,
    _restore_residual_core_parameters,
    _snapshot_pc_parameters,
    _snapshot_residual_core_parameters,
    _theta_update_from_transported_state,
    alpha_for_epoch,
    build_endpoint_semigroup_targets,
    build_stage05_v3c_stronger_semigroup_weight_config,
    build_trajectory_curriculum_targets,
    ensure_finite_array,
    run_fmpc_ef_exploratory_probe,
)
from ..utils import set_seed

OutputLayout = Literal["single_dir", "run_id_subdir"]

STAGE06_V1_CANDIDATE_NAME = "stage06_v1_objective_curriculum_energydrop_default"
STAGE06_V1_METHOD_NAME = STAGE06_V1_CANDIDATE_NAME
STAGE06_V2_CANDIDATE_NAME = (
    "stage06_v2_persistent_overlap_objective_curriculum_energydrop_default"
)
STAGE06_V2_METHOD_NAME = STAGE06_V2_CANDIDATE_NAME
STAGE05_MATCHED_BUDGET_CONTROL_METHOD_NAME = "stage05_v3c_stronger_semigroup_weight"
STAGE06_V1_SINGLE_RUN_STAGE = "stage06_v1_objective_curriculum"
STAGE06_V2_SINGLE_RUN_STAGE = "stage06_v2_objective_curriculum"
STAGE06_OBJECTIVE_CONTRACT_IDENTITY = (
    "objective_curriculum_plus_energydrop_fixed_point_contract"
)
STAGE06_V2_OBJECTIVE_CONTRACT_IDENTITY = (
    "persistent_overlap_objective_curriculum_plus_energydrop_fixed_point_contract"
)
STAGE06_SCAFFOLD_PRESERVATION_IDENTITY = (
    "two_branch_parameterization_plus_stage05_target_reuse_without_branchwise_supervision"
)
STAGE06_SUPERVISION_CONTRACT_IDENTITY = (
    "aggregate_residual_supervision_over_stage05_targets"
)
STAGE06_ROLLOUT_TIME_SEMANTICS_IDENTITY = "remaining_horizon_forward_rollout"
STAGE06_OBJECTIVE_FORMULA = (
    "L_6A(k) = (1 - beta_obj(k)) * L_traj + beta_obj(k) * L_semi "
    "+ lambda_energy_drop * L_drop + lambda_fixed_point * L_fp"
)
STAGE06_BETA_SCHEDULE_IDENTITY = "piecewise_linear_quarter_half_quarter"
STAGE06_V2_BETA_SCHEDULE_IDENTITY = (
    "piecewise_linear_quarter_half_quarter_persistent_overlap"
)
STAGE06_TRAJECTORY_COMPONENT_CONTRACT = "L_traj = mean(||m_hat - m_traj_star||^2)"
STAGE06_SEMIGROUP_COMPONENT_CONTRACT = (
    "L_semi = mean(r^2 * ||m_hat - m_semi_star||^2)"
)
STAGE06_ENERGY_DROP_CONTRACT = (
    "z_roll = z_t + r * u_psi(z_t, r, t; c); "
    "L_drop = mean(relu(E_theta(z_roll; c) - E_theta(z_t; c) + delta_margin))"
)
STAGE06_FIXED_POINT_CONTRACT = "L_fp = mean(||g_roll||_2^2)"
STAGE06_FIXED_POINT_GRADIENT_PROXY = (
    "output-level gradient proxy via centered D_{g_roll} g_roll using g = -grad E symmetry"
)
STAGE06_COMPARISON_STAGE = "stage06_v1_low_budget_comparison"
STAGE06_COMPARISON_EXPERIMENT = "stage06_v1_low_budget_comparison"
STAGE06_V2_COMPARISON_STAGE = "stage06_v2_low_budget_comparison"
STAGE06_V2_COMPARISON_EXPERIMENT = "stage06_v2_low_budget_comparison"
STAGE06_ALLOWED_ACCURACY_REGRESSION = 0.01
STAGE06_IMPROVEMENT_FRACTION_THRESHOLD = 0.05
STAGE06_V2_DEFAULT_BETA_OBJ_FINAL_VALUE = 0.75


@dataclass
class Stage06ObjectiveCurriculumConfig:
    experiment_name: str = "fmpc_stage06_objective_curriculum"
    dataset_name: str = "digits"
    run_seed: int = 0
    data_seed: int = 0
    model_init_seed: int = 0
    psi_init_seed: int = 0
    batch_order_seed: int = 0
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: OutputLayout = "single_dir"
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    layer_dims: tuple[int, ...] = (64, 16, 10)
    hidden_activation: str = "tanh"
    output_activation: str = "identity"
    weight_scale: float = 0.05
    sigma2: float | tuple[float, ...] = 1.0
    eta_x: float = 0.10
    eta_w: float = 0.02
    eta_b: float | None = 0.02
    eval_steps: int = 15
    state_init: str = "forward"
    epochs: int = 128
    batch_size: int = 128
    shuffle_batches: bool = True
    transport_steps: int = 2
    psi_hidden_dims: tuple[int, ...] = (128,)
    psi_weight_scale: float = 0.01
    psi_eta_w: float = 0.01
    psi_eta_b: float | None = 0.01
    tangent_epsilon: float = 1e-3
    bootstrap_integrator: Literal["euler", "rk2"] = "rk2"
    bootstrap_substeps: int = 4
    alpha_floor: float = 0.5
    alpha_warmup_epochs: int = 3
    alpha_ramp_epochs: int = 3
    stage05_scaffold_lambda_traj_curr: float = 0.2
    stage05_scaffold_lambda_sg: float = 0.1
    objective_schedule_variant: Literal["hard_handoff", "persistent_overlap"] = (
        "hard_handoff"
    )
    beta_obj_warmup_fraction: float = 0.25
    beta_obj_ramp_fraction: float = 0.50
    beta_obj_final_value: float = 1.0
    lambda_energy_drop: float = 0.25
    lambda_fixed_point: float = 0.10
    delta_margin: float = 0.0
    fixed_point_tangent_epsilon: float = 1e-3
    candidate_name_override: str | None = None
    selection_metric: Literal["val_configured_transported_final_energy"] = (
        "val_configured_transported_final_energy"
    )

    def __post_init__(self) -> None:
        if self.dataset_name != "digits":
            raise ValueError("Stage 06 v1 currently supports digits only.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.transport_steps <= 0:
            raise ValueError("transport_steps must be positive.")
        if self.bootstrap_substeps <= 0:
            raise ValueError("bootstrap_substeps must be positive.")
        if self.beta_obj_warmup_fraction < 0.0 or self.beta_obj_ramp_fraction < 0.0:
            raise ValueError("beta_obj fractions must be non-negative.")
        if (self.beta_obj_warmup_fraction + self.beta_obj_ramp_fraction) > 1.0 + 1e-12:
            raise ValueError("beta_obj warmup+ramp fractions must not exceed 1.")
        if not (0.0 <= self.beta_obj_final_value <= 1.0):
            raise ValueError("beta_obj_final_value must lie in [0, 1].")
        if self.objective_schedule_variant == "hard_handoff" and not np.isclose(
            self.beta_obj_final_value,
            1.0,
        ):
            raise ValueError("hard_handoff requires beta_obj_final_value == 1.0.")
        if self.objective_schedule_variant == "persistent_overlap" and not (
            0.0 < self.beta_obj_final_value < 1.0
        ):
            raise ValueError(
                "persistent_overlap requires 0 < beta_obj_final_value < 1."
            )
        if self.lambda_energy_drop < 0.0:
            raise ValueError("lambda_energy_drop must be non-negative.")
        if self.lambda_fixed_point < 0.0:
            raise ValueError("lambda_fixed_point must be non-negative.")
        if self.tangent_epsilon <= 0.0 or self.fixed_point_tangent_epsilon <= 0.0:
            raise ValueError("tangent epsilons must be positive.")
        if not (0.0 < self.alpha_floor < 1.0):
            raise ValueError("alpha_floor must satisfy 0 < alpha_floor < 1.")
        if self.alpha_warmup_epochs < 0 or self.alpha_ramp_epochs < 0:
            raise ValueError("alpha warmup/ramp epochs must be non-negative.")
        if self.selection_metric != "val_configured_transported_final_energy":
            raise ValueError("Only val_configured_transported_final_energy is supported.")

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_{self.run_seed}"


@dataclass(frozen=True)
class Stage06SupervisionBatch:
    trajectory_inputs: np.ndarray
    state_inputs: np.ndarray | None
    z_points: np.ndarray
    t_values: np.ndarray
    r_values: np.ndarray
    trajectory_targets: np.ndarray
    semigroup_targets: np.ndarray
    semigroup_loss_weights: np.ndarray


@dataclass(frozen=True)
class Stage06EnergyDropTerms:
    loss: float
    output_delta: np.ndarray
    rollout_state: np.ndarray
    rollout_energy: np.ndarray
    current_energy: np.ndarray
    active_mask: np.ndarray
    rollout_flow: np.ndarray


@dataclass(frozen=True)
class Stage06FixedPointTerms:
    loss: float
    output_delta: np.ndarray
    rollout_flow: np.ndarray
    directional_derivative: np.ndarray


@dataclass(frozen=True)
class Stage06ObjectiveTerms:
    beta_obj: float
    trajectory_objective_weight: float
    semigroup_objective_weight: float
    trajectory_loss: float
    semigroup_loss: float
    energy_drop_loss: float
    fixed_point_loss: float
    total_loss: float
    output_delta: np.ndarray
    rollout_state: np.ndarray
    rollout_flow: np.ndarray
    rollout_energy: np.ndarray
    current_energy: np.ndarray
    active_drop_rate: float
    mean_rollout_flow_norm_sq: float


@dataclass(frozen=True)
class Stage06EpochMetrics:
    epoch: int
    beta_obj: float
    trajectory_objective_weight: float
    semigroup_objective_weight: float
    alpha: float
    train_total_loss: float
    train_traj_loss: float
    train_semi_loss: float
    train_energy_drop_loss: float
    train_fixed_point_loss: float
    train_active_energy_drop_rate: float
    train_mean_rollout_flow_norm_sq: float
    val_one_step_transported_final_energy: float
    val_one_step_energy_delta_vs_identity: float
    val_one_step_fixed_point_residual_delta_vs_identity: float
    val_configured_transported_final_energy: float
    val_configured_energy_delta_vs_identity: float
    val_configured_fixed_point_residual_delta_vs_identity: float
    val_accuracy: float
    val_output_mse: float


@dataclass
class Stage06EpochSnapshot:
    epoch: int
    model_snapshot: list[tuple[np.ndarray, np.ndarray]]
    psi_snapshot: dict[str, list[tuple[np.ndarray, np.ndarray]] | None]


@dataclass
class Stage06ObjectiveCurriculumRunResult:
    run_dir: Path
    config: dict[str, Any]
    epoch_metrics: list[dict[str, Any]]
    summary: dict[str, Any]
    model: PCNetwork | None = None
    psi_network: Stage05ResidualCoreNetworks | None = None


@dataclass
class Stage06LowBudgetComparisonConfig:
    experiment_name: str = STAGE06_COMPARISON_EXPERIMENT
    output_root: str | Path = "outputs/stage_06_low_budget_efficiency"
    run_id: str | None = None
    dataset_name: str = "digits"
    seeds: tuple[int, ...] = (0, 1, 2)
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    batch_size: int = 128
    shuffle_batches: bool = True
    layer_dims: tuple[int, ...] = (64, 16, 10)
    transport_steps: int = 2
    eval_steps: int = 15
    tier1_epochs: int = 128
    tier2_epochs: int = 256
    rescue_epochs: int = 512
    allow_rescue_tier3: bool = True
    improvement_fraction_threshold: float = STAGE06_IMPROVEMENT_FRACTION_THRESHOLD
    allowed_accuracy_regression_threshold: float = STAGE06_ALLOWED_ACCURACY_REGRESSION

    def __post_init__(self) -> None:
        if self.dataset_name != "digits":
            raise ValueError("Stage 06 comparison currently supports digits only.")
        if not self.seeds:
            raise ValueError("At least one seed is required.")
        if self.tier1_epochs <= 0 or self.tier2_epochs <= 0 or self.rescue_epochs <= 0:
            raise ValueError("All tier epoch counts must be positive.")

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_comparison"


@dataclass
class Stage06LowBudgetComparisonRunResult:
    run_dir: Path
    summary: dict[str, Any]
    report: dict[str, Any]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("rows must be non-empty.")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _std(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64)))


def beta_obj_for_epoch(
    total_epochs: int,
    epoch_index: int,
    *,
    warmup_fraction: float = 0.25,
    ramp_fraction: float = 0.50,
    final_value: float = 1.0,
) -> float:
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive.")
    if epoch_index < 0:
        raise ValueError("epoch_index must be non-negative.")
    if warmup_fraction < 0.0 or ramp_fraction < 0.0:
        raise ValueError("fractions must be non-negative.")
    if warmup_fraction + ramp_fraction > 1.0 + 1e-12:
        raise ValueError("warmup_fraction + ramp_fraction must not exceed 1.")
    if final_value < 0.0 or final_value > 1.0:
        raise ValueError("final_value must lie in [0, 1].")
    warmup_end = warmup_fraction * float(total_epochs)
    ramp_end = (warmup_fraction + ramp_fraction) * float(total_epochs)
    if float(epoch_index) < warmup_end:
        return 0.0
    if float(epoch_index) >= ramp_end or ramp_end <= warmup_end:
        return float(final_value)
    progress = (float(epoch_index) - warmup_end) / (ramp_end - warmup_end)
    return float(final_value * np.clip(progress, 0.0, 1.0))


def build_stage06_v1_objective_curriculum_energydrop_default_config(
    **overrides: Any,
) -> Stage06ObjectiveCurriculumConfig:
    payload: dict[str, Any] = {
        "output_root": "outputs/stage_06_low_budget_efficiency",
        "experiment_name": "fmpc_stage06_objective_curriculum",
        "candidate_name_override": STAGE06_V1_CANDIDATE_NAME,
    }
    payload.update(overrides)
    return Stage06ObjectiveCurriculumConfig(**payload)


def build_stage06_v2_persistent_overlap_objective_curriculum_energydrop_default_config(
    **overrides: Any,
) -> Stage06ObjectiveCurriculumConfig:
    payload: dict[str, Any] = {
        "output_root": "outputs/stage_06_low_budget_efficiency",
        "experiment_name": "fmpc_stage06_objective_curriculum",
        "candidate_name_override": STAGE06_V2_CANDIDATE_NAME,
        "objective_schedule_variant": "persistent_overlap",
        "beta_obj_final_value": STAGE06_V2_DEFAULT_BETA_OBJ_FINAL_VALUE,
    }
    payload.update(overrides)
    return Stage06ObjectiveCurriculumConfig(**payload)


def _candidate_name(config: Stage06ObjectiveCurriculumConfig) -> str:
    if config.candidate_name_override is not None:
        return config.candidate_name_override
    if config.objective_schedule_variant == "persistent_overlap":
        return STAGE06_V2_CANDIDATE_NAME
    return STAGE06_V1_CANDIDATE_NAME


def _single_run_stage_name(config: Stage06ObjectiveCurriculumConfig) -> str:
    if config.objective_schedule_variant == "persistent_overlap":
        return STAGE06_V2_SINGLE_RUN_STAGE
    return STAGE06_V1_SINGLE_RUN_STAGE


def _objective_contract_identity(config: Stage06ObjectiveCurriculumConfig) -> str:
    if config.objective_schedule_variant == "persistent_overlap":
        return STAGE06_V2_OBJECTIVE_CONTRACT_IDENTITY
    return STAGE06_OBJECTIVE_CONTRACT_IDENTITY


def _objective_schedule_identity(config: Stage06ObjectiveCurriculumConfig) -> str:
    if config.objective_schedule_variant == "persistent_overlap":
        return STAGE06_V2_BETA_SCHEDULE_IDENTITY
    return STAGE06_BETA_SCHEDULE_IDENTITY


def _hard_late_handoff_enabled(config: Stage06ObjectiveCurriculumConfig) -> bool:
    return bool(config.objective_schedule_variant == "hard_handoff")


def _persistent_overlap_enabled(config: Stage06ObjectiveCurriculumConfig) -> bool:
    return bool(config.objective_schedule_variant == "persistent_overlap")


def _late_phase_trajectory_weight(config: Stage06ObjectiveCurriculumConfig) -> float:
    return float(1.0 - config.beta_obj_final_value)


def _late_phase_semigroup_weight(config: Stage06ObjectiveCurriculumConfig) -> float:
    return float(config.beta_obj_final_value)


def _build_stage05_scaffold_config(
    config: Stage06ObjectiveCurriculumConfig,
) -> FMPCEFExploratoryProbeConfig:
    return build_stage05_v3c_stronger_semigroup_weight_config(
        experiment_name=config.experiment_name,
        dataset_name=config.dataset_name,
        run_seed=config.run_seed,
        data_seed=config.data_seed,
        model_init_seed=config.model_init_seed,
        psi_init_seed=config.psi_init_seed,
        batch_order_seed=config.batch_order_seed,
        output_root=config.output_root,
        run_id=config.run_id,
        output_layout=config.output_layout,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
        layer_dims=config.layer_dims,
        hidden_activation=config.hidden_activation,
        output_activation=config.output_activation,
        weight_scale=config.weight_scale,
        sigma2=config.sigma2,
        eta_x=config.eta_x,
        eta_w=config.eta_w,
        eta_b=config.eta_b,
        eval_steps=config.eval_steps,
        state_init=config.state_init,
        epochs=config.epochs,
        batch_size=config.batch_size,
        shuffle_batches=config.shuffle_batches,
        transport_steps=config.transport_steps,
        psi_hidden_dims=config.psi_hidden_dims,
        psi_weight_scale=config.psi_weight_scale,
        psi_eta_w=config.psi_eta_w,
        psi_eta_b=config.psi_eta_b,
        tangent_epsilon=config.tangent_epsilon,
        bootstrap_integrator=config.bootstrap_integrator,
        bootstrap_substeps=config.bootstrap_substeps,
        alpha_floor=config.alpha_floor,
        alpha_warmup_epochs=config.alpha_warmup_epochs,
        alpha_ramp_epochs=config.alpha_ramp_epochs,
        lambda_traj_curr=config.stage05_scaffold_lambda_traj_curr,
        lambda_sg=config.stage05_scaffold_lambda_sg,
        candidate_name_override=_candidate_name(config),
    )


def _config_payload(config: Stage06ObjectiveCurriculumConfig) -> dict[str, Any]:
    scaffold = _build_stage05_scaffold_config(config)
    return {
        "phase": "FMPC Stage 06 Low-Budget Efficiency",
        "stage": _single_run_stage_name(config),
        "candidate_name": _candidate_name(config),
        "transport": {
            "transport_family": "two_branch_residual_meanflow_core",
            "stage05_scaffold_reference": STAGE05_MATCHED_BUDGET_CONTROL_METHOD_NAME,
            "stage05_scaffold_preservation_identity": STAGE06_SCAFFOLD_PRESERVATION_IDENTITY,
            "stage05_two_branch_parameterization_preserved": True,
            "stage05_target_builder_reuse_enabled": True,
            "stage05_branchwise_supervision_preserved": False,
            "stage06_supervision_contract_identity": STAGE06_SUPERVISION_CONTRACT_IDENTITY,
            "u_psi_family": "u_psi(z_t, r, t; c) = g_t + q_psi + d_psi",
            "explicit_transport_drift_decomposition_enabled": True,
            "trajectory_curriculum_enabled": True,
            "endpoint_semigroup_consistency_enabled": True,
            "midpoint_microfamily_continued": False,
            "alpha_floor": float(scaffold.alpha_floor),
            "alpha_warmup_epochs": int(scaffold.alpha_warmup_epochs),
            "alpha_ramp_epochs": int(scaffold.alpha_ramp_epochs),
            "stage05_scaffold_lambda_traj_curr": float(config.stage05_scaffold_lambda_traj_curr),
            "stage05_scaffold_lambda_sg": float(config.stage05_scaffold_lambda_sg),
            "bootstrap_integrator": config.bootstrap_integrator,
            "bootstrap_substeps": int(config.bootstrap_substeps),
        },
        "objective_contract": {
            "contract_identity": _objective_contract_identity(config),
            "candidate_name": _candidate_name(config),
            "objective_formula": STAGE06_OBJECTIVE_FORMULA,
            "trajectory_component_contract": STAGE06_TRAJECTORY_COMPONENT_CONTRACT,
            "semigroup_component_contract": STAGE06_SEMIGROUP_COMPONENT_CONTRACT,
            "energy_drop_contract": STAGE06_ENERGY_DROP_CONTRACT,
            "fixed_point_contract": STAGE06_FIXED_POINT_CONTRACT,
            "fixed_point_gradient_proxy": STAGE06_FIXED_POINT_GRADIENT_PROXY,
            "rollout_time_semantics_identity": STAGE06_ROLLOUT_TIME_SEMANTICS_IDENTITY,
            "beta_obj_schedule_identity": _objective_schedule_identity(config),
            "objective_schedule_variant": config.objective_schedule_variant,
            "beta_obj_is_distinct_from_alpha": True,
            "beta_obj_warmup_fraction": float(config.beta_obj_warmup_fraction),
            "beta_obj_ramp_fraction": float(config.beta_obj_ramp_fraction),
            "beta_obj_final_value": float(config.beta_obj_final_value),
            "beta_obj_final_plateau_fraction": float(
                1.0 - config.beta_obj_warmup_fraction - config.beta_obj_ramp_fraction
            ),
            "hard_late_handoff_enabled": _hard_late_handoff_enabled(config),
            "persistent_overlap_enabled": _persistent_overlap_enabled(config),
            "late_phase_trajectory_weight": _late_phase_trajectory_weight(config),
            "late_phase_semigroup_weight": _late_phase_semigroup_weight(config),
            "lambda_energy_drop": float(config.lambda_energy_drop),
            "lambda_fixed_point": float(config.lambda_fixed_point),
            "delta_margin": float(config.delta_margin),
            "fixed_point_tangent_epsilon": float(config.fixed_point_tangent_epsilon),
            "energy_drop_penalty_enabled": True,
            "fixed_point_contraction_penalty_enabled": True,
            "loss_breakdown_visible": True,
        },
        "run": {
            "run_seed": int(config.run_seed),
            "data_seed": int(config.data_seed),
            "model_init_seed": int(config.model_init_seed),
            "psi_init_seed": int(config.psi_init_seed),
            "batch_order_seed": int(config.batch_order_seed),
            "epochs": int(config.epochs),
            "batch_size": int(config.batch_size),
            "shuffle_batches": bool(config.shuffle_batches),
            "transport_steps": int(config.transport_steps),
            "eval_steps": int(config.eval_steps),
            "output_layout": config.output_layout,
        },
    }


def _write_epoch_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_csv(path, rows)


def _apply_output_delta_step(
    network: MLPNetwork,
    inputs: np.ndarray,
    output_delta: np.ndarray,
) -> None:
    activations, pre_activations = _forward_mlp(network, inputs)
    delta = _as_batch_first("output_delta", output_delta)
    for layer_index in range(len(network.layers) - 1, -1, -1):
        layer = network.layers[layer_index]
        pre_activation = pre_activations[layer_index + 1]
        if pre_activation is None:
            raise ValueError("Missing pre-activation for output-delta step.")
        _, activation_prime = get_activation(layer.activation_name)
        local_delta = delta * activation_prime(pre_activation)
        grad_w = local_delta.T @ activations[layer_index]
        grad_b = np.sum(local_delta, axis=0)
        next_delta = local_delta @ layer.weight if layer_index > 0 else None
        layer.weight = layer.weight - network.eta_w * grad_w
        eta_b = network.eta_b if network.eta_b is not None else network.eta_w
        layer.bias = layer.bias - eta_b * grad_b
        ensure_finite_array(layer.weight, f"stage06_weight[{layer_index + 1}]")
        ensure_finite_array(layer.bias, f"stage06_bias[{layer_index + 1}]")
        if next_delta is not None:
            delta = next_delta


def _apply_residual_core_output_delta_step(
    residual_core: Stage05ResidualCoreNetworks,
    trajectory_inputs: np.ndarray,
    state_inputs: np.ndarray | None,
    output_delta: np.ndarray,
) -> None:
    _apply_output_delta_step(residual_core.trajectory_network, trajectory_inputs, output_delta)
    if residual_core.state_network is not None:
        if state_inputs is None:
            raise ValueError("state_inputs are required for the two-branch residual core.")
        _apply_output_delta_step(residual_core.state_network, state_inputs, output_delta)


def _trajectory_target_from_stage05_scaffold(
    context: FMPCTF1Context,
    psi_network: Stage05ResidualCoreNetworks,
    scaffold_config: FMPCEFExploratoryProbeConfig,
    z_t: np.ndarray,
    *,
    t: float,
    r: float,
    alpha: float,
) -> np.ndarray:
    if float(alpha) >= 1.0 - 1e-12:
        u_boot = bootstrap_average_velocity_target(
            context,
            z_t,
            t=t,
            r=r,
            integrator=scaffold_config.bootstrap_integrator,
            substeps=scaffold_config.bootstrap_substeps,
        )
        g_t = hidden_local_flow(context, z_t)
        return u_boot - g_t
    targets: TrajectoryCurriculumTargets = build_trajectory_curriculum_targets(
        context,
        psi_network,
        scaffold_config,
        z_t,
        t=t,
        r=r,
        alpha=float(alpha),
    )
    return targets.residual_target


def _collect_stage06_supervision(
    context: FMPCTF1Context,
    psi_network: Stage05ResidualCoreNetworks,
    scaffold_config: FMPCEFExploratoryProbeConfig,
    *,
    z_knots: list[np.ndarray],
    knot_times: np.ndarray,
    trajectory_alpha: float,
) -> Stage06SupervisionBatch:
    trajectory_inputs: list[np.ndarray] = []
    state_inputs: list[np.ndarray] = []
    z_points: list[np.ndarray] = []
    t_values: list[np.ndarray] = []
    r_values: list[np.ndarray] = []
    trajectory_targets: list[np.ndarray] = []
    semigroup_targets: list[np.ndarray] = []
    semigroup_loss_weights: list[np.ndarray] = []

    for knot_index, t_k in enumerate(knot_times[:-1]):
        z_t = z_knots[knot_index]
        t_float = float(t_k)
        r_k = 1.0 - t_float
        trajectory_input, state_input, _ = _residual_core_inputs_for_state(
            context,
            scaffold_config,
            z_t,
            context.targets,
            t=t_float,
            r=r_k,
        )
        trajectory_inputs.append(trajectory_input)
        if state_input is not None:
            state_inputs.append(state_input)
        z_points.append(_as_batch_first("z_t", z_t))
        t_values.append(np.full((z_t.shape[0], 1), t_float, dtype=np.float64))
        r_values.append(np.full((z_t.shape[0], 1), r_k, dtype=np.float64))
        trajectory_targets.append(
            _trajectory_target_from_stage05_scaffold(
                context,
                psi_network,
                scaffold_config,
                z_t,
                t=t_float,
                r=r_k,
                alpha=float(trajectory_alpha),
            )
        )
        if float(trajectory_alpha) < 1.0 - 1e-12:
            semi: EndpointSemigroupTargets = build_endpoint_semigroup_targets(
                context,
                psi_network,
                scaffold_config,
                z_t,
                t=t_float,
                r=r_k,
                alpha=float(trajectory_alpha),
            )
            semigroup_targets.append(semi.residual_target)
            semigroup_loss_weights.append(semi.loss_weights)
        else:
            semigroup_targets.append(np.zeros_like(z_t, dtype=np.float64))
            semigroup_loss_weights.append(np.zeros((z_t.shape[0], 1), dtype=np.float64))

    batch = Stage06SupervisionBatch(
        trajectory_inputs=np.concatenate(trajectory_inputs, axis=0).astype(np.float64, copy=False),
        state_inputs=(
            np.concatenate(state_inputs, axis=0).astype(np.float64, copy=False)
            if state_inputs
            else None
        ),
        z_points=np.concatenate(z_points, axis=0).astype(np.float64, copy=False),
        t_values=np.concatenate(t_values, axis=0).astype(np.float64, copy=False),
        r_values=np.concatenate(r_values, axis=0).astype(np.float64, copy=False),
        trajectory_targets=np.concatenate(trajectory_targets, axis=0).astype(np.float64, copy=False),
        semigroup_targets=np.concatenate(semigroup_targets, axis=0).astype(np.float64, copy=False),
        semigroup_loss_weights=np.concatenate(semigroup_loss_weights, axis=0).astype(
            np.float64, copy=False
        ),
    )
    if batch.state_inputs is not None and batch.state_inputs.shape[0] != batch.trajectory_inputs.shape[0]:
        raise ValueError("state_inputs and trajectory_inputs must share the same batch size.")
    return batch


def _build_stage06_rollout_state(
    z_points: np.ndarray,
    velocity: np.ndarray,
    remaining_horizon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    z_array = _as_batch_first("z_points", z_points)
    velocity_array = _as_batch_first("velocity", velocity)
    horizon_array = _as_batch_first("remaining_horizon", remaining_horizon)
    if z_array.shape != velocity_array.shape:
        raise ValueError("z_points and velocity must share the same shape.")
    if horizon_array.shape != (z_array.shape[0], 1):
        raise ValueError("remaining_horizon must be shaped (batch, 1).")
    z_roll = z_array + (horizon_array * velocity_array)
    ensure_finite_array(z_roll, "stage06_z_roll")
    return z_roll, horizon_array


def compute_energy_drop_penalty_and_output_delta(
    current_energy: np.ndarray,
    rollout_energy: np.ndarray,
    rollout_flow: np.ndarray,
    rollout_coefficient: np.ndarray,
    *,
    delta_margin: float,
) -> Stage06EnergyDropTerms:
    current = _as_batch_first("current_energy", current_energy)
    rolled = _as_batch_first("rollout_energy", rollout_energy)
    g_roll = _as_batch_first("rollout_flow", rollout_flow)
    coeff = _as_batch_first("rollout_coefficient", rollout_coefficient)
    if current.shape != rolled.shape:
        raise ValueError("current_energy and rollout_energy must share the same shape.")
    if coeff.shape != current.shape:
        raise ValueError("rollout_coefficient must be shaped like current_energy.")
    if g_roll.shape[0] != current.shape[0]:
        raise ValueError("rollout_flow must share batch size with current_energy.")
    margin = rolled - current + float(delta_margin)
    active_mask = (margin > 0.0).astype(np.float64)
    loss = float(np.mean(np.maximum(margin, 0.0)))
    output_delta = (-coeff * active_mask / float(current.shape[0])) * g_roll
    return Stage06EnergyDropTerms(
        loss=loss,
        output_delta=output_delta,
        rollout_state=np.zeros_like(g_roll),
        rollout_energy=rolled,
        current_energy=current,
        active_mask=active_mask,
        rollout_flow=g_roll,
    )


def approximate_local_flow_directional_derivative(
    context: FMPCTF1Context,
    z: np.ndarray,
    direction: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    z_array = _as_batch_first("z", z)
    direction_array = _as_batch_first("direction", direction)
    if z_array.shape != direction_array.shape:
        raise ValueError("z and direction must share the same shape.")
    if z_array.shape[1] == 0:
        return np.zeros_like(z_array)
    z_plus = z_array + float(epsilon) * direction_array
    z_minus = z_array - float(epsilon) * direction_array
    g_plus = hidden_local_flow(context, z_plus)
    g_minus = hidden_local_flow(context, z_minus)
    derivative = (g_plus - g_minus) / (2.0 * float(epsilon))
    ensure_finite_array(derivative, "stage06_directional_derivative")
    return derivative


def _chunked_teacher_free_state_features(
    context: FMPCTF1Context,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    z_array = _as_batch_first("z", z)
    batch_size = int(context.batch_size)
    if z_array.shape[0] % batch_size != 0:
        raise ValueError("Chunked Stage 06 state evaluation requires a whole number of batch blocks.")
    g_blocks: list[np.ndarray] = []
    energy_blocks: list[np.ndarray] = []
    for start in range(0, z_array.shape[0], batch_size):
        stop = start + batch_size
        features = teacher_free_state_features(context, z_array[start:stop])
        g_blocks.append(features.g_t)
        energy_blocks.append(features.F_t)
    return (
        np.concatenate(g_blocks, axis=0).astype(np.float64, copy=False),
        np.concatenate(energy_blocks, axis=0).astype(np.float64, copy=False),
    )


def evaluate_fixed_point_contraction_terms(
    context: FMPCTF1Context,
    z_roll: np.ndarray,
    rollout_coefficient: np.ndarray,
    *,
    tangent_epsilon: float,
) -> Stage06FixedPointTerms:
    z_array = _as_batch_first("z_roll", z_roll)
    coeff = _as_batch_first("rollout_coefficient", rollout_coefficient)
    if coeff.shape != (z_array.shape[0], 1):
        raise ValueError("rollout_coefficient must be shaped (batch, 1).")
    g_roll = hidden_local_flow(context, z_array)
    directional = approximate_local_flow_directional_derivative(
        context,
        z_array,
        g_roll,
        epsilon=tangent_epsilon,
    )
    loss = float(np.mean(np.sum(g_roll * g_roll, axis=1, keepdims=True)))
    output_delta = (2.0 * coeff / float(z_array.shape[0])) * directional
    return Stage06FixedPointTerms(
        loss=loss,
        output_delta=output_delta,
        rollout_flow=g_roll,
        directional_derivative=directional,
    )


def _evaluate_chunked_fixed_point_contraction_terms(
    context: FMPCTF1Context,
    z_roll: np.ndarray,
    rollout_coefficient: np.ndarray,
    *,
    tangent_epsilon: float,
) -> Stage06FixedPointTerms:
    z_array = _as_batch_first("z_roll", z_roll)
    coeff_array = _as_batch_first("rollout_coefficient", rollout_coefficient)
    batch_size = int(context.batch_size)
    if z_array.shape[0] % batch_size != 0:
        raise ValueError("Chunked Stage 06 fixed-point evaluation requires whole batch blocks.")
    if coeff_array.shape != (z_array.shape[0], 1):
        raise ValueError("rollout_coefficient must be shaped (batch, 1).")
    terms: list[Stage06FixedPointTerms] = []
    for start in range(0, z_array.shape[0], batch_size):
        stop = start + batch_size
        terms.append(
            evaluate_fixed_point_contraction_terms(
                context,
                z_array[start:stop],
                coeff_array[start:stop],
                tangent_epsilon=tangent_epsilon,
            )
        )
    return Stage06FixedPointTerms(
        loss=float(np.mean([term.loss for term in terms])),
        output_delta=np.concatenate([term.output_delta for term in terms], axis=0).astype(
            np.float64, copy=False
        ),
        rollout_flow=np.concatenate([term.rollout_flow for term in terms], axis=0).astype(
            np.float64, copy=False
        ),
        directional_derivative=np.concatenate(
            [term.directional_derivative for term in terms],
            axis=0,
        ).astype(np.float64, copy=False),
    )


def _compute_stage06_objective_terms(
    context: FMPCTF1Context,
    psi_network: Stage05ResidualCoreNetworks,
    supervision: Stage06SupervisionBatch,
    *,
    beta_obj: float,
    lambda_energy_drop: float,
    lambda_fixed_point: float,
    delta_margin: float,
    fixed_point_tangent_epsilon: float,
) -> Stage06ObjectiveTerms:
    predictions = _predict_residual_from_inputs(
        psi_network,
        supervision.trajectory_inputs,
        state_inputs=supervision.state_inputs,
    )
    m_hat = predictions.total_residual
    output_size = float(m_hat.size)
    semigroup_objective_weight = float(beta_obj)
    trajectory_objective_weight = float(1.0 - semigroup_objective_weight)

    traj_error = m_hat - supervision.trajectory_targets
    trajectory_loss = float(np.mean(traj_error ** 2))
    trajectory_delta = (2.0 * trajectory_objective_weight / output_size) * traj_error

    semi_error = m_hat - supervision.semigroup_targets
    semigroup_loss = float(np.mean(supervision.semigroup_loss_weights * (semi_error ** 2)))
    semigroup_delta = (
        2.0
        * semigroup_objective_weight
        / output_size
        * supervision.semigroup_loss_weights
        * semi_error
    )

    g_t, current_energy = _chunked_teacher_free_state_features(context, supervision.z_points)
    u_hat = g_t + m_hat
    z_roll, rollout_coefficient = _build_stage06_rollout_state(
        supervision.z_points,
        u_hat,
        supervision.r_values,
    )

    rollout_flow, rollout_energy = _chunked_teacher_free_state_features(context, z_roll)
    energy_terms = compute_energy_drop_penalty_and_output_delta(
        current_energy,
        rollout_energy,
        rollout_flow,
        rollout_coefficient,
        delta_margin=delta_margin,
    )
    fixed_point_terms = _evaluate_chunked_fixed_point_contraction_terms(
        context,
        z_roll,
        rollout_coefficient,
        tangent_epsilon=fixed_point_tangent_epsilon,
    )
    total_loss = (
        (trajectory_objective_weight * trajectory_loss)
        + (semigroup_objective_weight * semigroup_loss)
        + (float(lambda_energy_drop) * energy_terms.loss)
        + (float(lambda_fixed_point) * fixed_point_terms.loss)
    )
    total_delta = (
        trajectory_delta
        + semigroup_delta
        + (float(lambda_energy_drop) * energy_terms.output_delta)
        + (float(lambda_fixed_point) * fixed_point_terms.output_delta)
    )
    ensure_finite_array(total_delta, "stage06_total_output_delta")
    mean_rollout_flow_norm_sq = float(
        np.mean(np.sum(fixed_point_terms.rollout_flow * fixed_point_terms.rollout_flow, axis=1))
    )
    return Stage06ObjectiveTerms(
        beta_obj=float(beta_obj),
        trajectory_objective_weight=trajectory_objective_weight,
        semigroup_objective_weight=semigroup_objective_weight,
        trajectory_loss=trajectory_loss,
        semigroup_loss=semigroup_loss,
        energy_drop_loss=float(energy_terms.loss),
        fixed_point_loss=float(fixed_point_terms.loss),
        total_loss=float(total_loss),
        output_delta=total_delta,
        rollout_state=z_roll,
        rollout_flow=fixed_point_terms.rollout_flow,
        rollout_energy=energy_terms.rollout_energy,
        current_energy=energy_terms.current_energy,
        active_drop_rate=float(np.mean(energy_terms.active_mask)),
        mean_rollout_flow_norm_sq=mean_rollout_flow_norm_sq,
    )


def _train_one_batch(
    model: PCNetwork,
    psi_network: Stage05ResidualCoreNetworks,
    config: Stage06ObjectiveCurriculumConfig,
    scaffold_config: FMPCEFExploratoryProbeConfig,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    *,
    beta_obj: float,
    trajectory_alpha: float,
) -> tuple[Stage06ObjectiveTerms, float]:
    context = build_tf1_context(model, x_batch, y_batch)
    source_rollout = rollout_hidden_transport(
        context,
        context.z0,
        transport_steps=config.transport_steps,
        mode="local_field_only",
    )
    supervision = _collect_stage06_supervision(
        context,
        psi_network,
        scaffold_config,
        z_knots=source_rollout.z_knots,
        knot_times=source_rollout.knot_times,
        trajectory_alpha=trajectory_alpha,
    )
    objective_terms = _compute_stage06_objective_terms(
        context,
        psi_network,
        supervision,
        beta_obj=float(beta_obj),
        lambda_energy_drop=float(config.lambda_energy_drop),
        lambda_fixed_point=float(config.lambda_fixed_point),
        delta_margin=float(config.delta_margin),
        fixed_point_tangent_epsilon=float(config.fixed_point_tangent_epsilon),
    )
    _apply_residual_core_output_delta_step(
        psi_network,
        supervision.trajectory_inputs,
        supervision.state_inputs,
        objective_terms.output_delta,
    )
    learned_rollout = rollout_hidden_transport(
        context,
        context.z0,
        transport_steps=config.transport_steps,
        mode="learned",
        velocity_fn=_learned_velocity_fn(context, psi_network, scaffold_config),
    )
    transported_energy = _theta_update_from_transported_state(
        model,
        context,
        learned_rollout.z_knots[-1],
    )
    return objective_terms, float(transported_energy)


def _snapshot_for_epoch(
    snapshots: list[Stage06EpochSnapshot],
    epoch: int,
) -> Stage06EpochSnapshot:
    for snapshot in snapshots:
        if int(snapshot.epoch) == int(epoch):
            return snapshot
    raise ValueError(f"Missing snapshot for epoch {epoch}.")


def run_stage06_objective_curriculum(
    config: Stage06ObjectiveCurriculumConfig,
) -> Stage06ObjectiveCurriculumRunResult:
    set_seed(config.run_seed)
    split = load_digits_split(
        split_seed=config.data_seed,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
    )
    scaffold_config = _build_stage05_scaffold_config(config)
    model = _make_pc_model(scaffold_config)
    psi_network = _make_psi_network(scaffold_config)

    run_dir = _prepare_run_dir(
        _resolve_run_dir(
            config.output_root,
            config.experiment_name,
            config.resolved_run_id(),
            config.output_layout,
        )
    )
    _write_json(run_dir / "config.json", _config_payload(config))

    epoch_rows: list[dict[str, Any]] = []
    snapshots: list[Stage06EpochSnapshot] = []

    train_start = perf_counter()
    for epoch_index in range(config.epochs):
        beta_obj = beta_obj_for_epoch(
            config.epochs,
            epoch_index,
            warmup_fraction=config.beta_obj_warmup_fraction,
            ramp_fraction=config.beta_obj_ramp_fraction,
            final_value=config.beta_obj_final_value,
        )
        trajectory_alpha = alpha_for_epoch(scaffold_config, epoch_index)
        batch_total_losses: list[float] = []
        batch_traj_losses: list[float] = []
        batch_semi_losses: list[float] = []
        batch_energy_drop_losses: list[float] = []
        batch_fixed_point_losses: list[float] = []
        batch_active_drop_rates: list[float] = []
        batch_rollout_flow_norm_sq: list[float] = []
        batch_transported_energies: list[float] = []
        batch_seed = config.batch_order_seed + epoch_index

        for x_batch, y_batch in iter_minibatches(
            split.x_train,
            split.y_train,
            config.batch_size,
            shuffle=config.shuffle_batches,
            seed=batch_seed,
        ):
            objective_terms, transported_energy = _train_one_batch(
                model,
                psi_network,
                config,
                scaffold_config,
                x_batch,
                y_batch,
                beta_obj=float(beta_obj),
                trajectory_alpha=float(trajectory_alpha),
            )
            batch_total_losses.append(objective_terms.total_loss)
            batch_traj_losses.append(objective_terms.trajectory_loss)
            batch_semi_losses.append(objective_terms.semigroup_loss)
            batch_energy_drop_losses.append(objective_terms.energy_drop_loss)
            batch_fixed_point_losses.append(objective_terms.fixed_point_loss)
            batch_active_drop_rates.append(objective_terms.active_drop_rate)
            batch_rollout_flow_norm_sq.append(objective_terms.mean_rollout_flow_norm_sq)
            batch_transported_energies.append(transported_energy)

        val_one_step = _evaluate_mechanism_metrics(
            model,
            psi_network,
            scaffold_config,
            split.x_val,
            split.y_val,
            transport_steps=1,
        )
        val_configured = _evaluate_mechanism_metrics(
            model,
            psi_network,
            scaffold_config,
            split.x_val,
            split.y_val,
            transport_steps=config.transport_steps,
        )
        val_output_mse, val_accuracy = _evaluate_slow_pc_metrics(model, split.x_val, split.y_val)
        row = asdict(
            Stage06EpochMetrics(
                epoch=epoch_index + 1,
                beta_obj=float(beta_obj),
                trajectory_objective_weight=float(
                    objective_terms.trajectory_objective_weight
                ),
                semigroup_objective_weight=float(
                    objective_terms.semigroup_objective_weight
                ),
                alpha=float(trajectory_alpha),
                train_total_loss=float(np.mean(batch_total_losses)),
                train_traj_loss=float(np.mean(batch_traj_losses)),
                train_semi_loss=float(np.mean(batch_semi_losses)),
                train_energy_drop_loss=float(np.mean(batch_energy_drop_losses)),
                train_fixed_point_loss=float(np.mean(batch_fixed_point_losses)),
                train_active_energy_drop_rate=float(np.mean(batch_active_drop_rates)),
                train_mean_rollout_flow_norm_sq=float(np.mean(batch_rollout_flow_norm_sq)),
                val_one_step_transported_final_energy=float(val_one_step.transported_final_energy),
                val_one_step_energy_delta_vs_identity=float(val_one_step.energy_delta_vs_identity),
                val_one_step_fixed_point_residual_delta_vs_identity=float(
                    val_one_step.fixed_point_residual_delta_vs_identity
                ),
                val_configured_transported_final_energy=float(
                    val_configured.transported_final_energy
                ),
                val_configured_energy_delta_vs_identity=float(
                    val_configured.energy_delta_vs_identity
                ),
                val_configured_fixed_point_residual_delta_vs_identity=float(
                    val_configured.fixed_point_residual_delta_vs_identity
                ),
                val_accuracy=float(val_accuracy),
                val_output_mse=float(val_output_mse),
            )
        )
        epoch_rows.append(row)
        snapshots.append(
            Stage06EpochSnapshot(
                epoch=epoch_index + 1,
                model_snapshot=_snapshot_pc_parameters(model),
                psi_snapshot=_snapshot_residual_core_parameters(psi_network),
            )
        )

    train_wall_time_seconds = float(perf_counter() - train_start)

    best_row = min(epoch_rows, key=lambda row: float(row["val_configured_transported_final_energy"]))
    selected_epoch = int(best_row["epoch"])
    selected_snapshot = _snapshot_for_epoch(snapshots, selected_epoch)
    _restore_pc_parameters(model, selected_snapshot.model_snapshot)
    _restore_residual_core_parameters(psi_network, selected_snapshot.psi_snapshot)

    evaluation_start = perf_counter()
    val_one_step = _evaluate_mechanism_metrics(
        model,
        psi_network,
        scaffold_config,
        split.x_val,
        split.y_val,
        transport_steps=1,
    )
    val_configured = _evaluate_mechanism_metrics(
        model,
        psi_network,
        scaffold_config,
        split.x_val,
        split.y_val,
        transport_steps=config.transport_steps,
    )
    test_one_step = _evaluate_mechanism_metrics(
        model,
        psi_network,
        scaffold_config,
        split.x_test,
        split.y_test,
        transport_steps=1,
    )
    test_configured = _evaluate_mechanism_metrics(
        model,
        psi_network,
        scaffold_config,
        split.x_test,
        split.y_test,
        transport_steps=config.transport_steps,
    )
    val_output_mse, val_accuracy = _evaluate_slow_pc_metrics(model, split.x_val, split.y_val)
    test_output_mse, test_accuracy = _evaluate_slow_pc_metrics(model, split.x_test, split.y_test)
    evaluation_wall_time_seconds = float(perf_counter() - evaluation_start)

    mechanism_acceptance = {
        "one_step_mechanism_positive": bool(val_one_step.energy_delta_vs_identity < 0.0),
        "configured_step_mechanism_positive": bool(
            val_configured.energy_delta_vs_identity < 0.0
            and val_configured.fixed_point_residual_delta_vs_identity < 0.0
        ),
        "configured_steps_energy_decrease_vs_identity": bool(
            val_configured.energy_delta_vs_identity < 0.0
        ),
        "configured_steps_fixed_point_residual_decrease_vs_identity": bool(
            val_configured.fixed_point_residual_delta_vs_identity < 0.0
        ),
    }
    runtime_proxy_seconds = float(train_wall_time_seconds + evaluation_wall_time_seconds)
    summary = {
        "phase": "FMPC Stage 06 Low-Budget Efficiency",
        "stage": _single_run_stage_name(config),
        "candidate_name": _candidate_name(config),
        "transport_family": "two_branch_residual_meanflow_core",
        "stage05_scaffold_reference": STAGE05_MATCHED_BUDGET_CONTROL_METHOD_NAME,
        "stage05_scaffold_preservation_identity": STAGE06_SCAFFOLD_PRESERVATION_IDENTITY,
        "stage05_two_branch_parameterization_preserved": True,
        "stage05_target_builder_reuse_enabled": True,
        "stage05_branchwise_supervision_preserved": False,
        "stage06_supervision_contract_identity": STAGE06_SUPERVISION_CONTRACT_IDENTITY,
        "objective_contract_identity": _objective_contract_identity(config),
        "objective_formula": STAGE06_OBJECTIVE_FORMULA,
        "trajectory_component_contract": STAGE06_TRAJECTORY_COMPONENT_CONTRACT,
        "semigroup_component_contract": STAGE06_SEMIGROUP_COMPONENT_CONTRACT,
        "energy_drop_contract": STAGE06_ENERGY_DROP_CONTRACT,
        "fixed_point_contract": STAGE06_FIXED_POINT_CONTRACT,
        "fixed_point_gradient_proxy": STAGE06_FIXED_POINT_GRADIENT_PROXY,
        "rollout_time_semantics_identity": STAGE06_ROLLOUT_TIME_SEMANTICS_IDENTITY,
        "beta_obj_schedule_identity": _objective_schedule_identity(config),
        "objective_schedule_variant": config.objective_schedule_variant,
        "beta_obj_is_distinct_from_alpha": True,
        "beta_obj_warmup_fraction": float(config.beta_obj_warmup_fraction),
        "beta_obj_ramp_fraction": float(config.beta_obj_ramp_fraction),
        "beta_obj_final_value": float(config.beta_obj_final_value),
        "beta_obj_final_plateau_fraction": float(
            1.0 - config.beta_obj_warmup_fraction - config.beta_obj_ramp_fraction
        ),
        "hard_late_handoff_enabled": _hard_late_handoff_enabled(config),
        "persistent_overlap_enabled": _persistent_overlap_enabled(config),
        "late_phase_trajectory_weight": _late_phase_trajectory_weight(config),
        "late_phase_semigroup_weight": _late_phase_semigroup_weight(config),
        "lambda_energy_drop": float(config.lambda_energy_drop),
        "lambda_fixed_point": float(config.lambda_fixed_point),
        "delta_margin": float(config.delta_margin),
        "fixed_point_tangent_epsilon": float(config.fixed_point_tangent_epsilon),
        "energy_drop_penalty_enabled": True,
        "fixed_point_contraction_penalty_enabled": True,
        "loss_breakdown_visible": True,
        "explicit_transport_drift_decomposition_enabled": True,
        "trajectory_curriculum_enabled": True,
        "endpoint_semigroup_consistency_enabled": True,
        "midpoint_microfamily_continued": False,
        "alpha_floor": float(config.alpha_floor),
        "alpha_warmup_epochs": int(config.alpha_warmup_epochs),
        "alpha_ramp_epochs": int(config.alpha_ramp_epochs),
        "stage05_scaffold_lambda_traj_curr": float(config.stage05_scaffold_lambda_traj_curr),
        "stage05_scaffold_lambda_sg": float(config.stage05_scaffold_lambda_sg),
        "selected_epoch": int(selected_epoch),
        "selected_epoch_beta_obj": float(best_row["beta_obj"]),
        "selected_epoch_trajectory_objective_weight": float(
            best_row["trajectory_objective_weight"]
        ),
        "selected_epoch_semigroup_objective_weight": float(
            best_row["semigroup_objective_weight"]
        ),
        "selected_epoch_alpha": float(best_row["alpha"]),
        "selected_epoch_train_total_loss": float(best_row["train_total_loss"]),
        "selected_epoch_train_traj_loss": float(best_row["train_traj_loss"]),
        "selected_epoch_train_semi_loss": float(best_row["train_semi_loss"]),
        "selected_epoch_train_energy_drop_loss": float(best_row["train_energy_drop_loss"]),
        "selected_epoch_train_fixed_point_loss": float(best_row["train_fixed_point_loss"]),
        "selected_epoch_train_active_energy_drop_rate": float(
            best_row["train_active_energy_drop_rate"]
        ),
        "selected_epoch_train_mean_rollout_flow_norm_sq": float(
            best_row["train_mean_rollout_flow_norm_sq"]
        ),
        "selection_metric_source": "val_metric",
        "selection_metric_name": "val_configured_transported_final_energy",
        "selection_metric_value": float(val_configured.transported_final_energy),
        "selection_metric_higher_is_better": False,
        "val_accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy),
        "val_output_mse": float(val_output_mse),
        "test_output_mse": float(test_output_mse),
        "mechanism_acceptance": mechanism_acceptance,
        "mechanism_metrics": {
            "one_step": asdict(val_one_step),
            "configured_steps": asdict(val_configured),
        },
        "test_mechanism_metrics": {
            "one_step": asdict(test_one_step),
            "configured_steps": asdict(test_configured),
        },
        "train_wall_time_seconds": float(train_wall_time_seconds),
        "evaluation_wall_time_seconds": float(evaluation_wall_time_seconds),
        "runtime_proxy_seconds": runtime_proxy_seconds,
        "deterministic_artifacts": True,
        "target_construction_artifact_independent": True,
        "run_artifacts": {
            "config_json": "config.json",
            "epoch_metrics_csv": "epoch_metrics.csv",
            "summary_json": "summary.json",
        },
    }
    _write_epoch_metrics(run_dir / "epoch_metrics.csv", epoch_rows)
    _write_json(run_dir / "summary.json", summary)
    return Stage06ObjectiveCurriculumRunResult(
        run_dir=run_dir,
        config=_config_payload(config),
        epoch_metrics=epoch_rows,
        summary=summary,
        model=model,
        psi_network=psi_network,
    )


def _mechanism_positive_flags(summary: dict[str, Any]) -> tuple[bool, bool]:
    one_step = bool(summary["mechanism_metrics"]["one_step"]["energy_delta_vs_identity"] < 0.0)
    configured = bool(
        summary["mechanism_metrics"]["configured_steps"]["energy_delta_vs_identity"] < 0.0
        and summary["mechanism_metrics"]["configured_steps"]["fixed_point_residual_delta_vs_identity"]
        < 0.0
    )
    return one_step, configured


def _run_row_from_stage06_result(
    result: Stage06ObjectiveCurriculumRunResult,
    *,
    tier_epochs: int,
    seed: int,
) -> dict[str, Any]:
    summary = result.summary
    one_step_positive, configured_positive = _mechanism_positive_flags(summary)
    one_step = summary["mechanism_metrics"]["one_step"]
    configured = summary["mechanism_metrics"]["configured_steps"]
    runtime_proxy = float(summary["runtime_proxy_seconds"])
    return {
        "stage": "FMPC Stage 06 Low-Budget Efficiency",
        "method_name": str(summary["candidate_name"]),
        "candidate_name": str(summary["candidate_name"]),
        "tier_epochs": int(tier_epochs),
        "seed": int(seed),
        "runtime_proxy_seconds": runtime_proxy,
        "selected_epoch": int(summary["selected_epoch"]),
        "one_step_mechanism_positive": bool(one_step_positive),
        "configured_step_mechanism_positive": bool(configured_positive),
        "one_step_energy_delta_vs_identity": float(one_step["energy_delta_vs_identity"]),
        "configured_step_energy_delta_vs_identity": float(
            configured["energy_delta_vs_identity"]
        ),
        "configured_step_fixed_point_residual_delta_vs_identity": float(
            configured["fixed_point_residual_delta_vs_identity"]
        ),
        "val_accuracy": float(summary["val_accuracy"]),
        "test_accuracy": float(summary["test_accuracy"]),
        "val_output_mse": float(summary["val_output_mse"]),
        "test_output_mse": float(summary["test_output_mse"]),
        "mechanism_gain_per_runtime": float(
            abs(configured["energy_delta_vs_identity"]) / max(runtime_proxy, 1e-12)
        ),
        "mechanism_gain_per_epoch": float(
            abs(configured["energy_delta_vs_identity"]) / float(max(int(tier_epochs), 1))
        ),
        "configured_residual_gain_per_runtime": float(
            abs(configured["fixed_point_residual_delta_vs_identity"])
            / max(runtime_proxy, 1e-12)
        ),
        "configured_residual_gain_per_epoch": float(
            abs(configured["fixed_point_residual_delta_vs_identity"])
            / float(max(int(tier_epochs), 1))
        ),
        "run_dir": str(result.run_dir),
    }


def _run_row_from_stage05_control_result(
    result: Any,
    *,
    tier_epochs: int,
    seed: int,
) -> dict[str, Any]:
    summary = result.summary
    one_step_positive, configured_positive = _mechanism_positive_flags(summary)
    one_step = summary["mechanism_metrics"]["one_step"]
    configured = summary["mechanism_metrics"]["configured_steps"]
    runtime_proxy = float(summary["train_wall_time_seconds"]) + float(
        summary["evaluation_wall_time_seconds"]
    )
    return {
        "stage": "FMPC Stage 05 EF Core Probe",
        "method_name": STAGE05_MATCHED_BUDGET_CONTROL_METHOD_NAME,
        "candidate_name": STAGE05_MATCHED_BUDGET_CONTROL_METHOD_NAME,
        "tier_epochs": int(tier_epochs),
        "seed": int(seed),
        "runtime_proxy_seconds": runtime_proxy,
        "selected_epoch": int(summary["selected_epoch"]),
        "one_step_mechanism_positive": bool(one_step_positive),
        "configured_step_mechanism_positive": bool(configured_positive),
        "one_step_energy_delta_vs_identity": float(one_step["energy_delta_vs_identity"]),
        "configured_step_energy_delta_vs_identity": float(
            configured["energy_delta_vs_identity"]
        ),
        "configured_step_fixed_point_residual_delta_vs_identity": float(
            configured["fixed_point_residual_delta_vs_identity"]
        ),
        "val_accuracy": float(summary["val_accuracy"]),
        "test_accuracy": float(summary["test_accuracy"]),
        "val_output_mse": float(summary["val_output_mse"]),
        "test_output_mse": float(summary["test_output_mse"]),
        "mechanism_gain_per_runtime": float(
            abs(configured["energy_delta_vs_identity"]) / max(runtime_proxy, 1e-12)
        ),
        "mechanism_gain_per_epoch": float(
            abs(configured["energy_delta_vs_identity"]) / float(max(int(tier_epochs), 1))
        ),
        "configured_residual_gain_per_runtime": float(
            abs(configured["fixed_point_residual_delta_vs_identity"]) / max(runtime_proxy, 1e-12)
        ),
        "configured_residual_gain_per_epoch": float(
            abs(configured["fixed_point_residual_delta_vs_identity"])
            / float(max(int(tier_epochs), 1))
        ),
        "run_dir": str(result.run_dir),
    }


def _method_rows(
    rows: list[dict[str, Any]],
    *,
    method_name: str,
    tier_epochs: int,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row["method_name"]) == method_name and int(row["tier_epochs"]) == int(tier_epochs)
    ]


def _method_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must be non-empty.")
    scalar_fields = [
        "one_step_energy_delta_vs_identity",
        "configured_step_energy_delta_vs_identity",
        "configured_step_fixed_point_residual_delta_vs_identity",
        "val_accuracy",
        "test_accuracy",
        "val_output_mse",
        "test_output_mse",
        "runtime_proxy_seconds",
        "mechanism_gain_per_runtime",
        "mechanism_gain_per_epoch",
        "configured_residual_gain_per_runtime",
        "configured_residual_gain_per_epoch",
        "selected_epoch",
    ]
    summary: dict[str, Any] = {
        "num_runs": int(len(rows)),
        "one_step_mechanism_positive_rate": float(
            np.mean([1.0 if bool(row["one_step_mechanism_positive"]) else 0.0 for row in rows])
        ),
        "configured_step_mechanism_positive_rate": float(
            np.mean(
                [1.0 if bool(row["configured_step_mechanism_positive"]) else 0.0 for row in rows]
            )
        ),
    }
    for field in scalar_fields:
        values = [float(row[field]) for row in rows]
        summary[field] = {"mean": _mean(values), "std": _std(values)}
    return summary


def _pairwise_summary(
    rows: list[dict[str, Any]],
    *,
    candidate_method: str,
    reference_method: str,
    tier_epochs: int,
) -> dict[str, Any]:
    candidate_rows = sorted(
        _method_rows(rows, method_name=candidate_method, tier_epochs=tier_epochs),
        key=lambda row: int(row["seed"]),
    )
    reference_rows = sorted(
        _method_rows(rows, method_name=reference_method, tier_epochs=tier_epochs),
        key=lambda row: int(row["seed"]),
    )
    if len(candidate_rows) != len(reference_rows):
        raise ValueError("Candidate and reference rows must align by seed count.")

    energy_deltas: list[float] = []
    residual_deltas: list[float] = []
    val_accuracy_deltas: list[float] = []
    test_accuracy_deltas: list[float] = []
    runtime_deltas: list[float] = []
    energy_gain_fractions: list[float] = []
    residual_gain_fractions: list[float] = []
    mechanism_gain_per_runtime_deltas: list[float] = []
    mechanism_gain_per_epoch_deltas: list[float] = []

    for candidate_row, reference_row in zip(candidate_rows, reference_rows, strict=True):
        if int(candidate_row["seed"]) != int(reference_row["seed"]):
            raise ValueError("Candidate/reference rows must align by seed.")
        candidate_energy = float(candidate_row["configured_step_energy_delta_vs_identity"])
        reference_energy = float(reference_row["configured_step_energy_delta_vs_identity"])
        candidate_residual = float(
            candidate_row["configured_step_fixed_point_residual_delta_vs_identity"]
        )
        reference_residual = float(
            reference_row["configured_step_fixed_point_residual_delta_vs_identity"]
        )
        energy_deltas.append(candidate_energy - reference_energy)
        residual_deltas.append(candidate_residual - reference_residual)
        val_accuracy_deltas.append(float(candidate_row["val_accuracy"]) - float(reference_row["val_accuracy"]))
        test_accuracy_deltas.append(
            float(candidate_row["test_accuracy"]) - float(reference_row["test_accuracy"])
        )
        runtime_deltas.append(
            float(candidate_row["runtime_proxy_seconds"]) - float(reference_row["runtime_proxy_seconds"])
        )
        energy_gain_fractions.append((reference_energy - candidate_energy) / max(abs(reference_energy), 1e-12))
        residual_gain_fractions.append(
            (reference_residual - candidate_residual) / max(abs(reference_residual), 1e-12)
        )
        mechanism_gain_per_runtime_deltas.append(
            float(candidate_row["mechanism_gain_per_runtime"])
            - float(reference_row["mechanism_gain_per_runtime"])
        )
        mechanism_gain_per_epoch_deltas.append(
            float(candidate_row["mechanism_gain_per_epoch"])
            - float(reference_row["mechanism_gain_per_epoch"])
        )

    return {
        "configured_step_energy_delta_vs_identity_delta": {
            "mean": _mean(energy_deltas),
            "std": _std(energy_deltas),
        },
        "configured_step_fixed_point_residual_delta_vs_identity_delta": {
            "mean": _mean(residual_deltas),
            "std": _std(residual_deltas),
        },
        "val_accuracy_delta": {"mean": _mean(val_accuracy_deltas), "std": _std(val_accuracy_deltas)},
        "test_accuracy_delta": {
            "mean": _mean(test_accuracy_deltas),
            "std": _std(test_accuracy_deltas),
        },
        "runtime_proxy_seconds_delta": {"mean": _mean(runtime_deltas), "std": _std(runtime_deltas)},
        "configured_step_energy_gain_fraction": {
            "mean": _mean(energy_gain_fractions),
            "std": _std(energy_gain_fractions),
        },
        "configured_step_residual_gain_fraction": {
            "mean": _mean(residual_gain_fractions),
            "std": _std(residual_gain_fractions),
        },
        "mechanism_gain_per_runtime_delta": {
            "mean": _mean(mechanism_gain_per_runtime_deltas),
            "std": _std(mechanism_gain_per_runtime_deltas),
        },
        "mechanism_gain_per_epoch_delta": {
            "mean": _mean(mechanism_gain_per_epoch_deltas),
            "std": _std(mechanism_gain_per_epoch_deltas),
        },
    }


def _passes_main_gate(
    candidate_summary: dict[str, Any],
    pairwise: dict[str, Any],
    *,
    allowed_accuracy_regression_threshold: float,
    improvement_fraction_threshold: float,
) -> bool:
    return bool(
        candidate_summary["one_step_mechanism_positive_rate"] >= 1.0
        and candidate_summary["configured_step_mechanism_positive_rate"] >= 1.0
        and float(pairwise["configured_step_energy_gain_fraction"]["mean"])
        >= float(improvement_fraction_threshold)
        and float(pairwise["configured_step_residual_gain_fraction"]["mean"])
        >= float(improvement_fraction_threshold)
        and float(pairwise["test_accuracy_delta"]["mean"])
        >= -float(allowed_accuracy_regression_threshold)
    )


def _shows_positive_trend_for_rescue(
    candidate_summary: dict[str, Any],
    pairwise: dict[str, Any],
    *,
    allowed_accuracy_regression_threshold: float,
    improvement_fraction_threshold: float,
) -> bool:
    energy_gain = float(pairwise["configured_step_energy_gain_fraction"]["mean"])
    residual_gain = float(pairwise["configured_step_residual_gain_fraction"]["mean"])
    return bool(
        candidate_summary["one_step_mechanism_positive_rate"] >= (2.0 / 3.0)
        and candidate_summary["configured_step_mechanism_positive_rate"] >= (2.0 / 3.0)
        and energy_gain > 0.0
        and residual_gain > 0.0
        and (
            energy_gain < float(improvement_fraction_threshold)
            or residual_gain < float(improvement_fraction_threshold)
        )
        and float(pairwise["test_accuracy_delta"]["mean"])
        >= -float(allowed_accuracy_regression_threshold)
    )


def _shows_better_cost_effectiveness(
    candidate_summary: dict[str, Any],
    reference_summary: dict[str, Any],
) -> bool:
    return bool(
        float(candidate_summary["mechanism_gain_per_runtime"]["mean"])
        > float(reference_summary["mechanism_gain_per_runtime"]["mean"])
        and float(candidate_summary["mechanism_gain_per_epoch"]["mean"])
        > float(reference_summary["mechanism_gain_per_epoch"]["mean"])
    )


def _comparison_config_payload_for_candidate(
    config: Stage06LowBudgetComparisonConfig,
    *,
    comparison_stage: str,
    candidate_config: Stage06ObjectiveCurriculumConfig,
) -> dict[str, Any]:
    return {
        "phase": "FMPC Stage 06 Low-Budget Efficiency",
        "stage": comparison_stage,
        "comparison_protocol": {
            "dataset_name": config.dataset_name,
            "seeds": [int(seed) for seed in config.seeds],
            "batch_size": int(config.batch_size),
            "shuffle_batches": bool(config.shuffle_batches),
            "layer_dims": [int(dim) for dim in config.layer_dims],
            "transport_steps": int(config.transport_steps),
            "eval_steps": int(config.eval_steps),
            "tier1_epochs": int(config.tier1_epochs),
            "tier2_epochs": int(config.tier2_epochs),
            "rescue_epochs": int(config.rescue_epochs),
            "allow_rescue_tier3": bool(config.allow_rescue_tier3),
        },
        "candidate_name": _candidate_name(candidate_config),
        "candidate_stage": _single_run_stage_name(candidate_config),
        "candidate_objective_contract_identity": _objective_contract_identity(
            candidate_config
        ),
        "candidate_beta_obj_schedule_identity": _objective_schedule_identity(
            candidate_config
        ),
        "candidate_objective_schedule_variant": candidate_config.objective_schedule_variant,
        "candidate_hard_late_handoff_enabled": _hard_late_handoff_enabled(candidate_config),
        "candidate_persistent_overlap_enabled": _persistent_overlap_enabled(
            candidate_config
        ),
        "candidate_beta_obj_final_value": float(candidate_config.beta_obj_final_value),
        "candidate_late_phase_trajectory_weight": _late_phase_trajectory_weight(
            candidate_config
        ),
        "candidate_late_phase_semigroup_weight": _late_phase_semigroup_weight(
            candidate_config
        ),
        "stage05_two_branch_parameterization_preserved": True,
        "stage05_target_builder_reuse_enabled": True,
        "stage05_branchwise_supervision_preserved": False,
        "matched_budget_control": STAGE05_MATCHED_BUDGET_CONTROL_METHOD_NAME,
        "improvement_fraction_threshold": float(config.improvement_fraction_threshold),
        "allowed_accuracy_regression_threshold": float(
            config.allowed_accuracy_regression_threshold
        ),
    }


def _comparison_config_payload(config: Stage06LowBudgetComparisonConfig) -> dict[str, Any]:
    return _comparison_config_payload_for_candidate(
        config,
        comparison_stage=STAGE06_COMPARISON_STAGE,
        candidate_config=build_stage06_v1_objective_curriculum_energydrop_default_config(),
    )


def _stage06_report_markdown(report: dict[str, Any], *, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        f"- Tier 1 viable: `{report['decision']['passes_tier1_viability']}`",
        f"- Tier 2 main gate: `{report['decision']['passes_tier2_main_gate']}`",
        f"- Tier 2 rescue trend: `{report['decision']['tier2_positive_trend_for_rescue']}`",
        f"- Rescue 512 warranted: `{report['decision']['rescue_512_warranted']}`",
        f"- Better cost effectiveness than matched-budget Stage 05 control: "
        f"`{report['decision']['shows_better_cost_effectiveness_than_stage05_control']}`",
        f"- Recommended next move: `{report['decision']['recommended_stage06_next_move']}`",
        "",
        "## Tier Summaries",
    ]
    for tier_name, payload in report["tier_summaries"].items():
        lines.extend(
            [
                "",
                f"### {tier_name}",
                f"- Candidate configured-step energy delta vs identity mean: "
                f"`{payload['candidate']['configured_step_energy_delta_vs_identity']['mean']}`",
                f"- Candidate configured-step residual delta vs identity mean: "
                f"`{payload['candidate']['configured_step_fixed_point_residual_delta_vs_identity']['mean']}`",
                f"- Control configured-step energy delta vs identity mean: "
                f"`{payload['control']['configured_step_energy_delta_vs_identity']['mean']}`",
                f"- Control configured-step residual delta vs identity mean: "
                f"`{payload['control']['configured_step_fixed_point_residual_delta_vs_identity']['mean']}`",
                f"- Pairwise energy delta mean: "
                f"`{payload['pairwise']['configured_step_energy_delta_vs_identity_delta']['mean']}`",
                f"- Pairwise residual delta mean: "
                f"`{payload['pairwise']['configured_step_fixed_point_residual_delta_vs_identity_delta']['mean']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def _run_stage06_low_budget_comparison(
    config: Stage06LowBudgetComparisonConfig,
    *,
    comparison_stage: str,
    candidate_builder: Any,
    candidate_method_name: str,
    recommended_keep: str,
    recommended_rescue: str,
    recommended_reject: str,
    report_title: str,
) -> Stage06LowBudgetComparisonRunResult:
    run_dir = _prepare_run_dir(
        Path(config.output_root) / config.experiment_name / config.resolved_run_id()
    )
    candidate_config = candidate_builder()
    comparison_config_payload = _comparison_config_payload_for_candidate(
        config,
        comparison_stage=comparison_stage,
        candidate_config=candidate_config,
    )
    _write_json(run_dir / "config.json", comparison_config_payload)
    runs_root = run_dir / "runs"
    rows: list[dict[str, Any]] = []
    executed_tiers: list[int] = []

    def run_tier(tier_epochs: int) -> None:
        executed_tiers.append(int(tier_epochs))
        tier_label = f"tier_{tier_epochs}"
        for seed in config.seeds:
            seed_int = int(seed)
            stage06_result = run_stage06_objective_curriculum(
                candidate_builder(
                    output_root=runs_root,
                    output_layout="run_id_subdir",
                    run_id=f"{tier_label}_seed_{seed_int}",
                    experiment_name=tier_label,
                    run_seed=seed_int,
                    data_seed=seed_int,
                    model_init_seed=seed_int,
                    psi_init_seed=seed_int,
                    batch_order_seed=seed_int,
                    epochs=int(tier_epochs),
                    batch_size=int(config.batch_size),
                    layer_dims=config.layer_dims,
                    transport_steps=int(config.transport_steps),
                    eval_steps=int(config.eval_steps),
                    shuffle_batches=bool(config.shuffle_batches),
                )
            )
            rows.append(
                _run_row_from_stage06_result(
                    stage06_result,
                    tier_epochs=int(tier_epochs),
                    seed=seed_int,
                )
            )
            stage05_result = run_fmpc_ef_exploratory_probe(
                build_stage05_v3c_stronger_semigroup_weight_config(
                    output_root=runs_root,
                    output_layout="run_id_subdir",
                    run_id=f"{tier_label}_seed_{seed_int}",
                    experiment_name=f"{tier_label}_stage05_control",
                    run_seed=seed_int,
                    data_seed=seed_int,
                    model_init_seed=seed_int,
                    psi_init_seed=seed_int,
                    batch_order_seed=seed_int,
                    epochs=int(tier_epochs),
                    batch_size=int(config.batch_size),
                    layer_dims=config.layer_dims,
                    transport_steps=int(config.transport_steps),
                    eval_steps=int(config.eval_steps),
                    shuffle_batches=bool(config.shuffle_batches),
                )
            )
            rows.append(
                _run_row_from_stage05_control_result(
                    stage05_result,
                    tier_epochs=int(tier_epochs),
                    seed=seed_int,
                )
            )

    run_tier(int(config.tier1_epochs))
    run_tier(int(config.tier2_epochs))
    _write_csv(run_dir / "aggregate_runs.csv", rows)

    tier_summaries: dict[str, Any] = {}
    for tier_epochs in executed_tiers:
        candidate_summary = _method_summary(
            _method_rows(rows, method_name=candidate_method_name, tier_epochs=tier_epochs)
        )
        control_summary = _method_summary(
            _method_rows(
                rows,
                method_name=STAGE05_MATCHED_BUDGET_CONTROL_METHOD_NAME,
                tier_epochs=tier_epochs,
            )
        )
        pairwise = _pairwise_summary(
            rows,
            candidate_method=candidate_method_name,
            reference_method=STAGE05_MATCHED_BUDGET_CONTROL_METHOD_NAME,
            tier_epochs=tier_epochs,
        )
        tier_summaries[f"tier_{tier_epochs}"] = {
            "candidate": candidate_summary,
            "control": control_summary,
            "pairwise": pairwise,
            "better_cost_effectiveness": _shows_better_cost_effectiveness(
                candidate_summary,
                control_summary,
            ),
        }

    tier1_summary = tier_summaries[f"tier_{config.tier1_epochs}"]
    tier2_summary = tier_summaries[f"tier_{config.tier2_epochs}"]
    passes_tier1_viability = bool(
        tier1_summary["candidate"]["one_step_mechanism_positive_rate"] >= (2.0 / 3.0)
        and tier1_summary["candidate"]["configured_step_mechanism_positive_rate"]
        >= (2.0 / 3.0)
        and float(tier1_summary["pairwise"]["test_accuracy_delta"]["mean"])
        >= -float(config.allowed_accuracy_regression_threshold)
    )
    passes_tier2_main_gate = _passes_main_gate(
        tier2_summary["candidate"],
        tier2_summary["pairwise"],
        allowed_accuracy_regression_threshold=float(config.allowed_accuracy_regression_threshold),
        improvement_fraction_threshold=float(config.improvement_fraction_threshold),
    )
    tier2_positive_trend_for_rescue = _shows_positive_trend_for_rescue(
        tier2_summary["candidate"],
        tier2_summary["pairwise"],
        allowed_accuracy_regression_threshold=float(config.allowed_accuracy_regression_threshold),
        improvement_fraction_threshold=float(config.improvement_fraction_threshold),
    )
    rescue_512_warranted = bool(
        config.allow_rescue_tier3
        and not passes_tier2_main_gate
        and tier2_positive_trend_for_rescue
    )

    if rescue_512_warranted:
        run_tier(int(config.rescue_epochs))
        _write_csv(run_dir / "aggregate_runs.csv", rows)
        candidate_summary = _method_summary(
            _method_rows(
                rows,
                method_name=candidate_method_name,
                tier_epochs=int(config.rescue_epochs),
            )
        )
        control_summary = _method_summary(
            _method_rows(
                rows,
                method_name=STAGE05_MATCHED_BUDGET_CONTROL_METHOD_NAME,
                tier_epochs=int(config.rescue_epochs),
            )
        )
        pairwise = _pairwise_summary(
            rows,
            candidate_method=candidate_method_name,
            reference_method=STAGE05_MATCHED_BUDGET_CONTROL_METHOD_NAME,
            tier_epochs=int(config.rescue_epochs),
        )
        tier_summaries[f"tier_{config.rescue_epochs}"] = {
            "candidate": candidate_summary,
            "control": control_summary,
            "pairwise": pairwise,
            "better_cost_effectiveness": _shows_better_cost_effectiveness(
                candidate_summary,
                control_summary,
            ),
        }

    materially_beats_matched_budget_stage05_control = bool(passes_tier2_main_gate)
    shows_better_cost_effectiveness_than_stage05_control = bool(
        tier2_summary["better_cost_effectiveness"]
    )
    if passes_tier2_main_gate:
        recommended_stage06_next_move = recommended_keep
    elif rescue_512_warranted:
        recommended_stage06_next_move = recommended_rescue
    elif tier2_positive_trend_for_rescue:
        recommended_stage06_next_move = "escalate_to_graph_shortcut_follow_up"
    else:
        recommended_stage06_next_move = recommended_reject

    summary = {
        "phase": "FMPC Stage 06 Low-Budget Efficiency",
        "stage": comparison_stage,
        "candidate_name": candidate_method_name,
        "matched_budget_control_name": STAGE05_MATCHED_BUDGET_CONTROL_METHOD_NAME,
        "comparison_protocol": comparison_config_payload["comparison_protocol"],
        "candidate_stage": comparison_config_payload["candidate_stage"],
        "candidate_objective_contract_identity": comparison_config_payload[
            "candidate_objective_contract_identity"
        ],
        "candidate_beta_obj_schedule_identity": comparison_config_payload[
            "candidate_beta_obj_schedule_identity"
        ],
        "candidate_objective_schedule_variant": comparison_config_payload[
            "candidate_objective_schedule_variant"
        ],
        "candidate_hard_late_handoff_enabled": comparison_config_payload[
            "candidate_hard_late_handoff_enabled"
        ],
        "candidate_persistent_overlap_enabled": comparison_config_payload[
            "candidate_persistent_overlap_enabled"
        ],
        "candidate_beta_obj_final_value": comparison_config_payload[
            "candidate_beta_obj_final_value"
        ],
        "candidate_late_phase_trajectory_weight": comparison_config_payload[
            "candidate_late_phase_trajectory_weight"
        ],
        "candidate_late_phase_semigroup_weight": comparison_config_payload[
            "candidate_late_phase_semigroup_weight"
        ],
        "stage05_two_branch_parameterization_preserved": comparison_config_payload[
            "stage05_two_branch_parameterization_preserved"
        ],
        "stage05_target_builder_reuse_enabled": comparison_config_payload[
            "stage05_target_builder_reuse_enabled"
        ],
        "stage05_branchwise_supervision_preserved": comparison_config_payload[
            "stage05_branchwise_supervision_preserved"
        ],
        "tier_summaries": tier_summaries,
        "passes_tier1_viability": passes_tier1_viability,
        "passes_tier2_main_gate": passes_tier2_main_gate,
        "tier2_positive_trend_for_rescue": tier2_positive_trend_for_rescue,
        "rescue_512_warranted": rescue_512_warranted,
        "materially_beats_matched_budget_stage05_control": materially_beats_matched_budget_stage05_control,
        "shows_better_cost_effectiveness_than_stage05_control": shows_better_cost_effectiveness_than_stage05_control,
        "recommended_stage06_next_move": recommended_stage06_next_move,
        "aggregate_runs_csv_path": "aggregate_runs.csv",
        "aggregate_summary_json_path": "aggregate_summary.json",
        "comparison_report_json_path": "comparison_report.json",
        "comparison_report_md_path": "comparison_report.md",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)

    report = {
        "comparison_protocol": summary["comparison_protocol"],
        "decision": {
            "passes_tier1_viability": passes_tier1_viability,
            "passes_tier2_main_gate": passes_tier2_main_gate,
            "tier2_positive_trend_for_rescue": tier2_positive_trend_for_rescue,
            "rescue_512_warranted": rescue_512_warranted,
            "materially_beats_matched_budget_stage05_control": materially_beats_matched_budget_stage05_control,
            "shows_better_cost_effectiveness_than_stage05_control": shows_better_cost_effectiveness_than_stage05_control,
            "recommended_stage06_next_move": recommended_stage06_next_move,
        },
        "tier_summaries": tier_summaries,
    }
    _write_json(run_dir / "comparison_report.json", report)
    _write_text(
        run_dir / "comparison_report.md",
        _stage06_report_markdown(report, title=report_title),
    )
    return Stage06LowBudgetComparisonRunResult(
        run_dir=run_dir,
        summary=summary,
        report=report,
    )


def run_stage06_v1_low_budget_comparison(
    config: Stage06LowBudgetComparisonConfig,
) -> Stage06LowBudgetComparisonRunResult:
    return _run_stage06_low_budget_comparison(
        config,
        comparison_stage=STAGE06_COMPARISON_STAGE,
        candidate_builder=build_stage06_v1_objective_curriculum_energydrop_default_config,
        candidate_method_name=STAGE06_V1_CANDIDATE_NAME,
        recommended_keep="keep_stage06_v1_objective_curriculum_energydrop_default_direction",
        recommended_rescue="run_stage06_v1_objective_curriculum_energydrop_default_512_rescue",
        recommended_reject="reject_stage06_v1_objective_curriculum_energydrop_default",
        report_title="Stage 06 v1 Low-Budget Comparison",
    )


def run_stage06_v2_low_budget_comparison(
    config: Stage06LowBudgetComparisonConfig,
) -> Stage06LowBudgetComparisonRunResult:
    return _run_stage06_low_budget_comparison(
        config,
        comparison_stage=STAGE06_V2_COMPARISON_STAGE,
        candidate_builder=build_stage06_v2_persistent_overlap_objective_curriculum_energydrop_default_config,
        candidate_method_name=STAGE06_V2_CANDIDATE_NAME,
        recommended_keep="keep_stage06_v2_persistent_overlap_objective_curriculum_energydrop_default_direction",
        recommended_rescue="run_stage06_v2_persistent_overlap_objective_curriculum_energydrop_default_512_rescue",
        recommended_reject="reject_stage06_v2_persistent_overlap_objective_curriculum_energydrop_default",
        report_title="Stage 06 v2 Low-Budget Comparison",
    )
