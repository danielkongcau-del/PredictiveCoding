from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np

from ..activations import get_activation
from ..datasets import load_digits_split
from .fmpc_tf2 import (
    FMPCTF2Config,
    _extract_detached_local_flow_anchor,
    _identity_tangent_mode,
    _make_pc_model,
    _make_psi_network,
    _target_dim,
    _teacher_free_feature_dim,
    _train_one_batch_tf2,
    build_tf2_corrective_transport_default_config,
    run_fmpc_tf2_experiment,
)
from ..metrics import classification_accuracy
from ..minibatch import iter_minibatches
from ..mlp_baseline import MLPNetwork

PsiExpressivityStage = Literal["family_raw", "encoding"]


@dataclass(frozen=True)
class FMPCTF2PsiExpressivityCandidate:
    key: str
    stage: PsiExpressivityStage
    psi_family: Literal["baseline_plain", "residualized_local_field"]
    psi_hidden_dims: tuple[int, ...]
    time_encoding_variant: Literal["raw", "poly_rt2"]
    notes: str


@dataclass
class FMPCTF2PsiExpressivitySuiteConfig:
    """Offline-first psi expressivity study for the TF2 corrective transport default."""

    experiment_name: str = "fmpc_tf2_psi_expressivity_suite"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: str = "single_dir"
    offline_train_seeds: tuple[int, ...] = (0, 1)
    offline_val_seeds: tuple[int, ...] = (2,)
    end_to_end_seeds: tuple[int, ...] = (0, 1, 2)
    epochs: int = 60
    batch_size: int = 128
    eval_steps: int = 15
    layer_dims: tuple[int, ...] = (64, 64, 10)
    offline_probe_epoch_indices: tuple[int, ...] = (0, 4, 14, 29, 59)
    sample_batches_per_probe_epoch: int = 1
    offline_fit_epochs: int = 150
    primary_hybrid_improvement_fraction: float = 0.05
    fallback_max_runtime_ratio: float = 2.0
    fallback_max_parameter_ratio: float = 2.5
    material_test_gain: float = 0.005
    material_val_gain: float = 0.0
    larger_plain_hidden_dims: tuple[int, ...] = (256,)
    slow_pc_reference_runs_path: str | Path = "outputs/tf2/fmpc_tf2_default_adoption_suite/aggregate_runs.csv"
    slow_pc_reference_name: str = "canonical_slow_pc_digits_baseline"

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_0"


@dataclass
class FMPCTF2PsiExpressivitySuiteRunResult:
    run_dir: Path
    config: dict[str, Any]
    offline_family_rows: list[dict[str, Any]]
    offline_encoding_rows: list[dict[str, Any]]
    end_to_end_rows: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass
class _OfflineSampleSplit:
    psi_input_raw: np.ndarray
    boot_target: np.ndarray
    identity_target: np.ndarray
    lambda_id: np.ndarray
    t_k: np.ndarray
    r_k: np.ndarray
    identity_tangent_mode: np.ndarray
    source_count_local_field_only: np.ndarray
    source_count_learned_on_policy: np.ndarray
    seed: np.ndarray
    epoch_index: np.ndarray
    step_index: np.ndarray
    preset_name: np.ndarray


def _resolve_run_dir(output_root: str | Path, experiment_name: str, run_id: str, output_layout: str) -> Path:
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("rows must contain at least one row.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Required psi-expressivity reference artifact is missing: {path_obj}")
    with path_obj.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _cosine_mean(a: np.ndarray, b: np.ndarray) -> float:
    a_array = np.asarray(a, dtype=np.float64)
    b_array = np.asarray(b, dtype=np.float64)
    numerator = np.sum(a_array * b_array, axis=1)
    denom = np.linalg.norm(a_array, axis=1) * np.linalg.norm(b_array, axis=1)
    denom = np.maximum(denom, 1e-12)
    return float(np.mean(numerator / denom))


def _count_parameters(network: MLPNetwork) -> int:
    return int(sum(layer.weight.size + layer.bias.size for layer in network.layers))


def _collectors_template() -> dict[str, list[Any]]:
    return {
        "psi_input_raw": [],
        "boot_target": [],
        "identity_target": [],
        "lambda_id": [],
        "t_k": [],
        "r_k": [],
        "identity_tangent_mode": [],
        "source_count_local_field_only": [],
        "source_count_learned_on_policy": [],
        "seed": [],
        "epoch_index": [],
        "step_index": [],
        "preset_name": [],
    }


def _append_samples(
    records: dict[str, list[Any]],
    payload: dict[str, Any],
    *,
    seed: int,
    preset_name: str,
) -> None:
    psi_inputs = np.asarray(payload["psi_inputs"], dtype=np.float64)
    boot_targets = np.asarray(payload["boot_targets"], dtype=np.float64)
    identity_targets = np.asarray(payload["identity_targets"], dtype=np.float64)
    if psi_inputs.ndim != 2 or boot_targets.ndim != 2 or identity_targets.ndim != 2:
        raise ValueError("Collected psi inputs and targets must be batch-first.")
    batch_size = int(psi_inputs.shape[0])
    for row_index in range(batch_size):
        records["psi_input_raw"].append(psi_inputs[row_index].copy())
        records["boot_target"].append(boot_targets[row_index].copy())
        records["identity_target"].append(identity_targets[row_index].copy())
        records["lambda_id"].append(float(payload["lambda_id"]))
        records["t_k"].append(float(payload["t_k"]))
        records["r_k"].append(float(payload["r_k"]))
        records["identity_tangent_mode"].append(str(payload["identity_tangent_mode"]))
        records["source_count_local_field_only"].append(
            int(dict(payload["source_counts"]).get("local_field_only", 0))
        )
        records["source_count_learned_on_policy"].append(
            int(dict(payload["source_counts"]).get("learned_on_policy", 0))
        )
        records["seed"].append(int(seed))
        records["epoch_index"].append(int(payload["epoch_index"]))
        records["step_index"].append(int(payload["step_index"]))
        records["preset_name"].append(str(preset_name))


def _finalize_split(records: dict[str, list[Any]]) -> _OfflineSampleSplit:
    if not records["psi_input_raw"]:
        raise ValueError("Offline sample collector did not record any examples.")
    return _OfflineSampleSplit(
        psi_input_raw=np.asarray(records["psi_input_raw"], dtype=np.float64),
        boot_target=np.asarray(records["boot_target"], dtype=np.float64),
        identity_target=np.asarray(records["identity_target"], dtype=np.float64),
        lambda_id=np.asarray(records["lambda_id"], dtype=np.float64),
        t_k=np.asarray(records["t_k"], dtype=np.float64),
        r_k=np.asarray(records["r_k"], dtype=np.float64),
        identity_tangent_mode=np.asarray(records["identity_tangent_mode"], dtype=str),
        source_count_local_field_only=np.asarray(records["source_count_local_field_only"], dtype=np.int64),
        source_count_learned_on_policy=np.asarray(
            records["source_count_learned_on_policy"], dtype=np.int64
        ),
        seed=np.asarray(records["seed"], dtype=np.int64),
        epoch_index=np.asarray(records["epoch_index"], dtype=np.int64),
        step_index=np.asarray(records["step_index"], dtype=np.int64),
        preset_name=np.asarray(records["preset_name"], dtype=str),
    )


def _save_offline_split(path: Path, split: _OfflineSampleSplit) -> None:
    np.savez_compressed(
        path,
        psi_input_raw=split.psi_input_raw,
        boot_target=split.boot_target,
        identity_target=split.identity_target,
        lambda_id=split.lambda_id,
        t_k=split.t_k,
        r_k=split.r_k,
        identity_tangent_mode=split.identity_tangent_mode,
        source_count_local_field_only=split.source_count_local_field_only,
        source_count_learned_on_policy=split.source_count_learned_on_policy,
        seed=split.seed,
        epoch_index=split.epoch_index,
        step_index=split.step_index,
        preset_name=split.preset_name,
    )


def _base_corrective_config(**overrides: Any) -> FMPCTF2Config:
    return build_tf2_corrective_transport_default_config(**overrides)


def _family_stage_candidates(config: FMPCTF2PsiExpressivitySuiteConfig) -> list[FMPCTF2PsiExpressivityCandidate]:
    baseline_hidden_dims = _base_corrective_config().psi_hidden_dims
    return [
        FMPCTF2PsiExpressivityCandidate(
            key="baseline_plain_raw",
            stage="family_raw",
            psi_family="baseline_plain",
            psi_hidden_dims=baseline_hidden_dims,
            time_encoding_variant="raw",
            notes="current corrective-default psi family and raw time concat",
        ),
        FMPCTF2PsiExpressivityCandidate(
            key="wider_plain_raw",
            stage="family_raw",
            psi_family="baseline_plain",
            psi_hidden_dims=config.larger_plain_hidden_dims,
            time_encoding_variant="raw",
            notes="single-axis capacity increase by widening the plain psi MLP",
        ),
        FMPCTF2PsiExpressivityCandidate(
            key="residualized_local_field_raw",
            stage="family_raw",
            psi_family="residualized_local_field",
            psi_hidden_dims=baseline_hidden_dims,
            time_encoding_variant="raw",
            notes="u_hat = g_t_detached + delta_u_psi with baseline hidden dims",
        ),
    ]


def _encoding_challenger(stage1_winner: FMPCTF2PsiExpressivityCandidate) -> FMPCTF2PsiExpressivityCandidate:
    return FMPCTF2PsiExpressivityCandidate(
        key=f"{stage1_winner.key.replace('_raw', '')}_poly_rt2",
        stage="encoding",
        psi_family=stage1_winner.psi_family,
        psi_hidden_dims=stage1_winner.psi_hidden_dims,
        time_encoding_variant="poly_rt2",
        notes="same family as stage-1 winner, but with low-order polynomial time encoding",
    )


def _suite_config_payload(
    config: FMPCTF2PsiExpressivitySuiteConfig,
    family_candidates: list[FMPCTF2PsiExpressivityCandidate],
) -> dict[str, Any]:
    return {
        "phase": "Phase TF2",
        "stage": "corrective_psi_expressivity_suite",
        "offline_train_seeds": [int(seed) for seed in config.offline_train_seeds],
        "offline_val_seeds": [int(seed) for seed in config.offline_val_seeds],
        "end_to_end_seeds": [int(seed) for seed in config.end_to_end_seeds],
        "offline_probe_epoch_indices": [int(value) for value in config.offline_probe_epoch_indices],
        "sample_batches_per_probe_epoch": int(config.sample_batches_per_probe_epoch),
        "offline_fit_epochs": int(config.offline_fit_epochs),
        "family_candidates": [
            {
                "key": candidate.key,
                "psi_family": candidate.psi_family,
                "psi_hidden_dims": [int(value) for value in candidate.psi_hidden_dims],
                "time_encoding_variant": candidate.time_encoding_variant,
                "notes": candidate.notes,
            }
            for candidate in family_candidates
        ],
        "fixed": {
            "preset_name": "tf2_corrective_transport_default",
            "use_teacher_free_features": True,
            "feature_aware_tangents": False,
            "micro_steps": 4,
            "incremental_weight_updates": False,
            "supervision_policy": "local_only",
            "theta_update_cadence": "terminal_only",
            "theta_update_budget": "matched",
            "identity_loss_weight": 0.2,
            "warmup_epochs": 5,
            "hybrid_ramp_epochs": 10,
            "bootstrap_integrator": "rk2",
            "bootstrap_substeps": 4,
            "checkpoint_selector": "gate_constrained_accuracy_then_val_accuracy",
        },
        "offline_primary_gate": {
            "hybrid_loss_improvement_fraction": float(config.primary_hybrid_improvement_fraction),
            "bootstrap_mse_must_not_worsen": True,
            "identity_residual_must_not_worsen": True,
        },
        "offline_fallback_gate": {
            "hybrid_loss_must_improve": True,
            "identity_residual_must_improve": True,
            "max_runtime_ratio": float(config.fallback_max_runtime_ratio),
            "max_parameter_ratio": float(config.fallback_max_parameter_ratio),
        },
        "slow_pc_reference_runs_path": str(config.slow_pc_reference_runs_path),
        "slow_pc_reference_name": str(config.slow_pc_reference_name),
    }


def _load_slow_pc_reference_by_seed(
    rows: list[dict[str, str]],
    *,
    reference_name: str,
    seeds: tuple[int, ...],
) -> dict[int, dict[str, float]]:
    filtered = [row for row in rows if str(row.get("preset_name", "")) == reference_name]
    if not filtered:
        raise ValueError(f"No slow-PC reference rows found for '{reference_name}'.")
    by_seed = {
        int(row["seed"]): {
            "val_accuracy": float(row["val_accuracy"]),
            "test_accuracy": float(row["test_accuracy"]),
        }
        for row in filtered
    }
    missing = [seed for seed in seeds if seed not in by_seed]
    if missing:
        raise ValueError(f"Slow-PC reference rows are missing seeds: {missing}.")
    return {int(seed): by_seed[int(seed)] for seed in seeds}


def _collect_offline_samples(
    config: FMPCTF2PsiExpressivitySuiteConfig,
) -> tuple[_OfflineSampleSplit, _OfflineSampleSplit]:
    probe_epoch_indices = set(int(value) for value in config.offline_probe_epoch_indices)
    train_records = _collectors_template()
    val_records = _collectors_template()
    split_by_seed = {int(seed): "train" for seed in config.offline_train_seeds} | {
        int(seed): "val" for seed in config.offline_val_seeds
    }
    for seed in (*config.offline_train_seeds, *config.offline_val_seeds):
        split_name = split_by_seed[int(seed)]
        run_config = _base_corrective_config(
            run_seed=seed,
            data_seed=seed,
            model_init_seed=seed,
            psi_init_seed=seed,
            batch_order_seed=seed,
            epochs=config.epochs,
            batch_size=config.batch_size,
            eval_steps=config.eval_steps,
            layer_dims=config.layer_dims,
        )
        split = load_digits_split(
            split_seed=run_config.data_seed,
            train_fraction=run_config.train_fraction,
            val_fraction=run_config.val_fraction,
            test_fraction=run_config.test_fraction,
        )
        model = _make_pc_model(run_config)
        psi_network = _make_psi_network(run_config)
        for epoch_index in range(run_config.epochs):
            if epoch_index < run_config.warmup_epochs:
                lambda_id = 0.0
            elif run_config.hybrid_ramp_epochs > 0:
                progress = (epoch_index - run_config.warmup_epochs + 1) / float(
                    run_config.hybrid_ramp_epochs
                )
                lambda_id = float(run_config.identity_loss_weight) * float(np.clip(progress, 0.0, 1.0))
            else:
                lambda_id = float(run_config.identity_loss_weight)
            batch_seed = run_config.batch_order_seed + epoch_index
            for batch_index, (x_batch, y_batch) in enumerate(
                iter_minibatches(
                    split.x_train,
                    split.y_train,
                    run_config.batch_size,
                    shuffle=run_config.shuffle_batches,
                    seed=batch_seed,
                )
            ):
                should_probe = (
                    epoch_index in probe_epoch_indices
                    and batch_index < int(config.sample_batches_per_probe_epoch)
                )
                collector = None
                if should_probe:
                    target_records = train_records if split_name == "train" else val_records

                    def _collector(
                        payload: dict[str, Any],
                        *,
                        target_records: dict[str, list[Any]] = target_records,
                        current_seed: int = int(seed),
                        preset_name: str = str(run_config.preset_name),
                    ) -> None:
                        _append_samples(
                            target_records,
                            payload,
                            seed=current_seed,
                            preset_name=preset_name,
                        )

                    collector = _collector
                _train_one_batch_tf2(
                    model,
                    psi_network,
                    run_config,
                    x_batch,
                    y_batch,
                    lambda_id=float(lambda_id),
                    epoch_index=epoch_index,
                    sample_collector=collector,
                )
    return _finalize_split(train_records), _finalize_split(val_records)


def _raw_input_layout(base_config: FMPCTF2Config) -> tuple[int, int, int]:
    hidden_dim = int(sum(base_config.layer_dims[1:-1]))
    target_dim = _target_dim(base_config)
    feature_dim = _teacher_free_feature_dim(base_config)
    return hidden_dim, target_dim, feature_dim


def _build_time_encoding_from_vectors(
    t_values: np.ndarray,
    r_values: np.ndarray,
    *,
    variant: Literal["raw", "poly_rt2"],
) -> np.ndarray:
    t_array = np.asarray(t_values, dtype=np.float64).reshape(-1, 1)
    r_array = np.asarray(r_values, dtype=np.float64).reshape(-1, 1)
    if variant == "raw":
        return np.concatenate([t_array, r_array], axis=1).astype(np.float64, copy=False)
    if variant == "poly_rt2":
        return np.concatenate(
            [t_array, r_array, t_array * r_array, t_array * t_array, r_array * r_array],
            axis=1,
        ).astype(np.float64, copy=False)
    raise ValueError(f"Unsupported time_encoding_variant '{variant}'.")


def _candidate_inputs_from_raw(
    split: _OfflineSampleSplit,
    *,
    base_config: FMPCTF2Config,
    candidate: FMPCTF2PsiExpressivityCandidate,
) -> np.ndarray:
    raw_inputs = np.asarray(split.psi_input_raw, dtype=np.float64)
    hidden_dim, target_dim, feature_dim = _raw_input_layout(base_config)
    z_block = raw_inputs[:, :hidden_dim]
    target_block = raw_inputs[:, hidden_dim : hidden_dim + target_dim]
    feature_offset = hidden_dim + target_dim + 2
    feature_block = raw_inputs[:, feature_offset : feature_offset + feature_dim]
    time_block = _build_time_encoding_from_vectors(
        split.t_k,
        split.r_k,
        variant=candidate.time_encoding_variant,
    )
    return np.concatenate([z_block, target_block, time_block, feature_block], axis=1).astype(
        np.float64,
        copy=False,
    )


def _candidate_config(
    candidate: FMPCTF2PsiExpressivityCandidate,
    *,
    seed: int = 0,
    layer_dims: tuple[int, ...] = (64, 64, 10),
) -> FMPCTF2Config:
    return _base_corrective_config(
        model_init_seed=seed,
        psi_init_seed=seed,
        layer_dims=layer_dims,
        psi_family=candidate.psi_family,
        psi_hidden_dims=candidate.psi_hidden_dims,
        time_encoding_variant=candidate.time_encoding_variant,
    )


def _candidate_anchor(
    inputs: np.ndarray,
    candidate_config: FMPCTF2Config,
) -> np.ndarray | None:
    if candidate_config.psi_family != "residualized_local_field":
        return None
    return _extract_detached_local_flow_anchor(inputs, candidate_config)


def _candidate_predict(
    network: MLPNetwork,
    inputs: np.ndarray,
    candidate_config: FMPCTF2Config,
) -> np.ndarray:
    base_output = network.predict(inputs)
    anchor = _candidate_anchor(inputs, candidate_config)
    if anchor is None:
        return base_output
    return anchor + base_output


def _weighted_offline_step(
    network: MLPNetwork,
    inputs: np.ndarray,
    targets: np.ndarray,
    sample_weights: np.ndarray,
) -> None:
    current = np.asarray(inputs, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    weight_array = np.asarray(sample_weights, dtype=np.float64).reshape(-1, 1)
    activations: list[np.ndarray] = [current]
    pre_activations: list[np.ndarray | None] = [None]
    for layer in network.layers:
        activation_fn, _ = get_activation(layer.activation_name)
        pre_activation = current @ layer.weight.T + layer.bias
        current = activation_fn(pre_activation)
        pre_activations.append(pre_activation)
        activations.append(current)
    predictions = activations[-1]
    batch_size = float(predictions.shape[0])
    output_dim = float(predictions.shape[1])
    delta = (2.0 / (batch_size * output_dim)) * weight_array * (predictions - target_array)
    for layer_index in range(len(network.layers) - 1, -1, -1):
        layer = network.layers[layer_index]
        pre_activation = pre_activations[layer_index + 1]
        if pre_activation is None:
            raise ValueError("pre_activation must be present for every layer.")
        _, activation_prime = get_activation(layer.activation_name)
        local_delta = delta * activation_prime(pre_activation)
        grad_w = local_delta.T @ activations[layer_index]
        grad_b = np.sum(local_delta, axis=0)
        next_delta = local_delta @ layer.weight if layer_index > 0 else None
        layer.weight = layer.weight - network.eta_w * grad_w
        layer.bias = layer.bias - network.eta_b * grad_b
        if next_delta is not None:
            delta = next_delta


def _combined_targets(
    split: _OfflineSampleSplit,
    *,
    anchor: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    lambda_block = split.lambda_id.reshape(-1, 1)
    combined = np.where(
        lambda_block > 0.0,
        (split.boot_target + lambda_block * split.identity_target) / (1.0 + lambda_block),
        split.boot_target,
    )
    if anchor is not None:
        combined = combined - anchor
    weights = (1.0 + lambda_block).reshape(-1)
    return combined.astype(np.float64, copy=False), weights.astype(np.float64, copy=False)


def _fit_offline_candidate(
    candidate: FMPCTF2PsiExpressivityCandidate,
    *,
    train_split: _OfflineSampleSplit,
    val_split: _OfflineSampleSplit,
    base_config: FMPCTF2Config,
    fit_epochs: int,
) -> dict[str, Any]:
    candidate_config = _candidate_config(candidate, seed=0, layer_dims=base_config.layer_dims)
    network = _make_psi_network(candidate_config)
    train_inputs = _candidate_inputs_from_raw(train_split, base_config=base_config, candidate=candidate)
    val_inputs = _candidate_inputs_from_raw(val_split, base_config=base_config, candidate=candidate)
    train_anchor = _candidate_anchor(train_inputs, candidate_config)
    train_targets, train_weights = _combined_targets(train_split, anchor=train_anchor)
    train_start = perf_counter()
    for _ in range(int(fit_epochs)):
        _weighted_offline_step(network, train_inputs, train_targets, train_weights)
    train_wall_time_seconds = float(perf_counter() - train_start)
    runtime_per_batch = float(train_wall_time_seconds / float(max(1, int(fit_epochs))))

    val_predictions = _candidate_predict(network, val_inputs, candidate_config)
    boot_mse = float(np.mean((val_predictions - val_split.boot_target) ** 2))
    relative_boot_mse = float(
        boot_mse / max(float(np.mean(val_split.boot_target**2)), 1e-12)
    )
    cosine = _cosine_mean(val_predictions, val_split.boot_target)
    identity_mse = float(np.mean((val_predictions - val_split.identity_target) ** 2))
    identity_residual = float(
        np.mean(np.linalg.norm(val_predictions - val_split.identity_target, axis=1))
    )
    boot_per_sample = np.mean((val_predictions - val_split.boot_target) ** 2, axis=1)
    identity_per_sample = np.mean((val_predictions - val_split.identity_target) ** 2, axis=1)
    hybrid_loss = float(np.mean(boot_per_sample + val_split.lambda_id * identity_per_sample))
    return {
        "candidate_key": candidate.key,
        "stage": candidate.stage,
        "psi_family": candidate.psi_family,
        "psi_hidden_dims": json.dumps([int(value) for value in candidate.psi_hidden_dims]),
        "time_encoding_variant": candidate.time_encoding_variant,
        "notes": candidate.notes,
        "parameter_count": int(_count_parameters(network)),
        "val_bootstrap_target_mse": float(boot_mse),
        "val_relative_bootstrap_target_mse": float(relative_boot_mse),
        "val_bootstrap_cosine_similarity": float(cosine),
        "val_identity_target_mse": float(identity_mse),
        "val_identity_residual_error": float(identity_residual),
        "val_hybrid_loss": float(hybrid_loss),
        "runtime_per_batch_seconds": float(runtime_per_batch),
    }


def _primary_gate_passes(
    candidate_row: dict[str, Any],
    baseline_row: dict[str, Any],
    config: FMPCTF2PsiExpressivitySuiteConfig,
) -> bool:
    return bool(
        float(candidate_row["val_hybrid_loss"])
        <= float(baseline_row["val_hybrid_loss"]) * (1.0 - float(config.primary_hybrid_improvement_fraction))
        and float(candidate_row["val_bootstrap_target_mse"]) <= float(baseline_row["val_bootstrap_target_mse"])
        and float(candidate_row["val_identity_residual_error"])
        <= float(baseline_row["val_identity_residual_error"])
    )


def _fallback_candidate(
    rows: list[dict[str, Any]],
    baseline_row: dict[str, Any],
    config: FMPCTF2PsiExpressivitySuiteConfig,
) -> dict[str, Any] | None:
    baseline_runtime = float(baseline_row["runtime_per_batch_seconds"])
    baseline_params = float(baseline_row["parameter_count"])
    eligible: list[dict[str, Any]] = []
    for row in rows:
        if row["candidate_key"] == baseline_row["candidate_key"]:
            continue
        if float(row["val_hybrid_loss"]) >= float(baseline_row["val_hybrid_loss"]):
            continue
        if float(row["val_identity_residual_error"]) >= float(baseline_row["val_identity_residual_error"]):
            continue
        runtime_ratio = float(row["runtime_per_batch_seconds"]) / max(baseline_runtime, 1e-12)
        parameter_ratio = float(row["parameter_count"]) / max(baseline_params, 1e-12)
        if runtime_ratio > float(config.fallback_max_runtime_ratio):
            continue
        if parameter_ratio > float(config.fallback_max_parameter_ratio):
            continue
        eligible.append(row)
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda payload: (
            float(payload["val_hybrid_loss"]),
            float(payload["val_identity_residual_error"]),
            float(payload["runtime_per_batch_seconds"]),
        ),
    )


def _success_end_to_end_row(
    *,
    run_index: int,
    candidate: FMPCTF2PsiExpressivityCandidate,
    seed: int,
    result: Any,
    run_dir: Path,
) -> dict[str, Any]:
    summary = result.summary
    timing = dict(summary.get("timing", {}))
    return {
        "run_index": int(run_index),
        "candidate_key": candidate.key,
        "stage": candidate.stage,
        "psi_family": candidate.psi_family,
        "psi_hidden_dims": json.dumps([int(value) for value in candidate.psi_hidden_dims]),
        "time_encoding_variant": candidate.time_encoding_variant,
        "seed": int(seed),
        "val_accuracy": float(summary["val_accuracy"]),
        "test_accuracy": float(summary["test_accuracy"]),
        "gate_passing_epoch_count": int(summary["gate_passing_epoch_count"]),
        "val_transported_final_energy": float(summary["val_transported_final_energy"]),
        "selected_epoch": int(summary["best_epoch"]),
        "selected_epoch_passes_gate": bool(summary["selected_epoch_passes_gate"]),
        "selector_fallback_used": bool(summary["selector_fallback_used"]),
        "total_wall_time_seconds": float(
            timing.get("train_wall_time_seconds", 0.0)
            + timing.get("final_evaluation_wall_time_seconds", 0.0)
        ),
        "run_status": "success",
        "run_summary_path": _relative_posix(run_dir, result.run_dir / "summary.json"),
    }


def _failure_end_to_end_row(
    *,
    run_index: int,
    candidate: FMPCTF2PsiExpressivityCandidate,
    seed: int,
    error: Exception,
) -> dict[str, Any]:
    return {
        "run_index": int(run_index),
        "candidate_key": candidate.key,
        "stage": candidate.stage,
        "psi_family": candidate.psi_family,
        "psi_hidden_dims": json.dumps([int(value) for value in candidate.psi_hidden_dims]),
        "time_encoding_variant": candidate.time_encoding_variant,
        "seed": int(seed),
        "val_accuracy": "",
        "test_accuracy": "",
        "gate_passing_epoch_count": "",
        "val_transported_final_energy": "",
        "selected_epoch": "",
        "selected_epoch_passes_gate": "",
        "selector_fallback_used": "",
        "total_wall_time_seconds": "",
        "run_status": f"failure:{error.__class__.__name__}",
        "run_summary_path": "",
    }


def _end_to_end_summary(
    rows: list[dict[str, Any]],
    candidate: FMPCTF2PsiExpressivityCandidate,
    *,
    slow_pc_ref: dict[int, dict[str, float]],
) -> dict[str, Any]:
    relevant = [row for row in rows if row["candidate_key"] == candidate.key]
    successes = [row for row in relevant if row["run_status"] == "success"]
    payload: dict[str, Any] = {
        "candidate_key": candidate.key,
        "stage": candidate.stage,
        "psi_family": candidate.psi_family,
        "time_encoding_variant": candidate.time_encoding_variant,
        "success_count": int(len(successes)),
        "failure_count": int(len(relevant) - len(successes)),
    }
    if not successes:
        payload.update(
            {
                "mean_val_accuracy": None,
                "std_val_accuracy": None,
                "mean_test_accuracy": None,
                "std_test_accuracy": None,
                "mean_gate_passing_epoch_count": None,
                "mean_val_transported_final_energy": None,
                "mean_total_wall_time_seconds": None,
                "std_total_wall_time_seconds": None,
                "mean_test_accuracy_gap_to_slow_pc": None,
            }
        )
        return payload
    val_values = [float(row["val_accuracy"]) for row in successes]
    test_values = [float(row["test_accuracy"]) for row in successes]
    gate_values = [float(row["gate_passing_epoch_count"]) for row in successes]
    energy_values = [float(row["val_transported_final_energy"]) for row in successes]
    wall_values = [float(row["total_wall_time_seconds"]) for row in successes]
    gap_values = [
        float(row["test_accuracy"]) - float(slow_pc_ref[int(row["seed"])]["test_accuracy"])
        for row in successes
    ]
    payload.update(
        {
            "mean_val_accuracy": _mean(val_values),
            "std_val_accuracy": _std(val_values),
            "mean_test_accuracy": _mean(test_values),
            "std_test_accuracy": _std(test_values),
            "mean_gate_passing_epoch_count": _mean(gate_values),
            "mean_val_transported_final_energy": _mean(energy_values),
            "mean_total_wall_time_seconds": _mean(wall_values),
            "std_total_wall_time_seconds": _std(wall_values),
            "mean_test_accuracy_gap_to_slow_pc": _mean(gap_values),
        }
    )
    return payload


def _pairwise_delta(
    challenger_summary: dict[str, Any],
    default_summary: dict[str, Any],
) -> dict[str, Any]:
    if challenger_summary["mean_test_accuracy"] is None or default_summary["mean_test_accuracy"] is None:
        return {
            "mean_val_accuracy_delta": None,
            "mean_test_accuracy_delta": None,
            "mean_gate_passing_epoch_count_delta": None,
            "mean_val_transported_final_energy_delta": None,
            "mean_total_wall_time_seconds_delta": None,
            "mean_test_gap_to_slow_pc_delta": None,
        }
    return {
        "mean_val_accuracy_delta": float(challenger_summary["mean_val_accuracy"] - default_summary["mean_val_accuracy"]),
        "mean_test_accuracy_delta": float(challenger_summary["mean_test_accuracy"] - default_summary["mean_test_accuracy"]),
        "mean_gate_passing_epoch_count_delta": float(
            challenger_summary["mean_gate_passing_epoch_count"] - default_summary["mean_gate_passing_epoch_count"]
        ),
        "mean_val_transported_final_energy_delta": float(
            challenger_summary["mean_val_transported_final_energy"] - default_summary["mean_val_transported_final_energy"]
        ),
        "mean_total_wall_time_seconds_delta": float(
            challenger_summary["mean_total_wall_time_seconds"] - default_summary["mean_total_wall_time_seconds"]
        ),
        "mean_test_gap_to_slow_pc_delta": float(
            challenger_summary["mean_test_accuracy_gap_to_slow_pc"] - default_summary["mean_test_accuracy_gap_to_slow_pc"]
        ),
    }


def _run_end_to_end_candidates(
    config: FMPCTF2PsiExpressivitySuiteConfig,
    *,
    candidates: list[FMPCTF2PsiExpressivityCandidate],
    run_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_index = 0
    for candidate in candidates:
        for seed in config.end_to_end_seeds:
            run_index += 1
            run_config = _base_corrective_config(
                run_seed=seed,
                data_seed=seed,
                model_init_seed=seed,
                psi_init_seed=seed,
                batch_order_seed=seed,
                epochs=config.epochs,
                batch_size=config.batch_size,
                eval_steps=config.eval_steps,
                layer_dims=config.layer_dims,
                psi_family=candidate.psi_family,
                psi_hidden_dims=candidate.psi_hidden_dims,
                time_encoding_variant=candidate.time_encoding_variant,
            )
            run_config.experiment_name = f"{config.experiment_name}/runs/{candidate.key}"
            run_config.output_root = str(config.output_root)
            run_config.output_layout = "run_id_subdir"
            run_config.run_id = f"seed_{int(seed)}"
            try:
                result = run_fmpc_tf2_experiment(run_config)
                rows.append(
                    _success_end_to_end_row(
                        run_index=run_index,
                        candidate=candidate,
                        seed=seed,
                        result=result,
                        run_dir=run_dir,
                    )
                )
            except Exception as error:
                rows.append(
                    _failure_end_to_end_row(
                        run_index=run_index,
                        candidate=candidate,
                        seed=seed,
                        error=error,
                    )
                )
    return rows


def run_fmpc_tf2_psi_expressivity_suite(
    config: FMPCTF2PsiExpressivitySuiteConfig,
) -> FMPCTF2PsiExpressivitySuiteRunResult:
    base_config = _base_corrective_config(layer_dims=config.layer_dims)
    family_candidates = _family_stage_candidates(config)
    run_dir = _prepare_run_dir(
        _resolve_run_dir(
            config.output_root,
            config.experiment_name,
            config.resolved_run_id(),
            config.output_layout,
        )
    )
    config_payload = _suite_config_payload(config, family_candidates)
    _write_json(run_dir / "config.json", config_payload)

    train_split, val_split = _collect_offline_samples(config)
    _save_offline_split(run_dir / "offline_train_samples.npz", train_split)
    _save_offline_split(run_dir / "offline_val_samples.npz", val_split)

    offline_family_rows = [
        _fit_offline_candidate(
            candidate,
            train_split=train_split,
            val_split=val_split,
            base_config=base_config,
            fit_epochs=config.offline_fit_epochs,
        )
        for candidate in family_candidates
    ]
    _write_csv(run_dir / "offline_family_metrics.csv", offline_family_rows)

    baseline_family_row = next(row for row in offline_family_rows if row["candidate_key"] == "baseline_plain_raw")
    stage1_winner_row = min(offline_family_rows, key=lambda row: float(row["val_hybrid_loss"]))
    stage1_winner_candidate = next(
        candidate for candidate in family_candidates if candidate.key == stage1_winner_row["candidate_key"]
    )
    encoding_candidate = _encoding_challenger(stage1_winner_candidate)
    offline_encoding_rows = [
        stage1_winner_row,
        _fit_offline_candidate(
            encoding_candidate,
            train_split=train_split,
            val_split=val_split,
            base_config=base_config,
            fit_epochs=config.offline_fit_epochs,
        ),
    ]
    _write_csv(run_dir / "offline_encoding_metrics.csv", offline_encoding_rows)

    candidate_lookup = {candidate.key: candidate for candidate in family_candidates}
    candidate_lookup[encoding_candidate.key] = encoding_candidate
    offline_all_rows = [
        *offline_family_rows,
        next(row for row in offline_encoding_rows if row["candidate_key"] == encoding_candidate.key),
    ]

    primary_pass_rows = [
        row
        for row in offline_all_rows
        if row["candidate_key"] != baseline_family_row["candidate_key"]
        and _primary_gate_passes(row, baseline_family_row, config)
    ]
    primary_pass_rows = sorted(
        primary_pass_rows,
        key=lambda row: (
            float(row["val_hybrid_loss"]),
            float(row["runtime_per_batch_seconds"]),
            int(row["parameter_count"]),
        ),
    )[:2]
    fallback_row = None if primary_pass_rows else _fallback_candidate(offline_all_rows, baseline_family_row, config)
    selected_challenger_rows = primary_pass_rows or ([] if fallback_row is None else [fallback_row])

    end_to_end_candidates = [candidate_lookup["baseline_plain_raw"]]
    end_to_end_candidates.extend(candidate_lookup[row["candidate_key"]] for row in selected_challenger_rows)
    end_to_end_rows = _run_end_to_end_candidates(config, candidates=end_to_end_candidates, run_dir=run_dir)
    _write_csv(run_dir / "end_to_end_runs.csv", end_to_end_rows)

    slow_pc_rows = _read_csv(config.slow_pc_reference_runs_path)
    slow_pc_ref = _load_slow_pc_reference_by_seed(
        slow_pc_rows,
        reference_name=config.slow_pc_reference_name,
        seeds=config.end_to_end_seeds,
    )
    end_to_end_summaries = {
        candidate.key: _end_to_end_summary(end_to_end_rows, candidate, slow_pc_ref=slow_pc_ref)
        for candidate in end_to_end_candidates
    }
    default_summary = end_to_end_summaries["baseline_plain_raw"]
    pairwise_delta = {
        key: _pairwise_delta(summary, default_summary)
        for key, summary in end_to_end_summaries.items()
        if key != "baseline_plain_raw"
    }

    material_end_to_end_winner: dict[str, Any] | None = None
    for key, summary in end_to_end_summaries.items():
        if key == "baseline_plain_raw" or summary["mean_test_accuracy"] is None:
            continue
        test_delta = float(summary["mean_test_accuracy"] - default_summary["mean_test_accuracy"])
        val_delta = float(summary["mean_val_accuracy"] - default_summary["mean_val_accuracy"])
        if test_delta >= float(config.material_test_gain) and val_delta >= float(config.material_val_gain):
            if material_end_to_end_winner is None or test_delta > float(
                material_end_to_end_winner["mean_test_accuracy"] - default_summary["mean_test_accuracy"]
            ):
                material_end_to_end_winner = summary

    wider_row = next(row for row in offline_family_rows if row["candidate_key"] == "wider_plain_raw")
    residual_row = next(row for row in offline_family_rows if row["candidate_key"] == "residualized_local_field_raw")
    encoding_row = next(row for row in offline_encoding_rows if row["candidate_key"] == encoding_candidate.key)
    capacity_signal = bool(
        float(wider_row["val_hybrid_loss"]) < float(baseline_family_row["val_hybrid_loss"])
        and float(wider_row["val_identity_residual_error"]) < float(baseline_family_row["val_identity_residual_error"])
    )
    output_parameterization_signal = bool(
        float(residual_row["val_hybrid_loss"]) < float(baseline_family_row["val_hybrid_loss"])
        and float(residual_row["val_identity_residual_error"]) < float(baseline_family_row["val_identity_residual_error"])
    )
    input_representation_signal = bool(
        float(encoding_row["val_hybrid_loss"]) < float(stage1_winner_row["val_hybrid_loss"])
        and float(encoding_row["val_identity_residual_error"]) < float(stage1_winner_row["val_identity_residual_error"])
    )
    offline_improvement_exists = any(
        float(row["val_hybrid_loss"]) < float(baseline_family_row["val_hybrid_loss"])
        for row in offline_all_rows
        if row["candidate_key"] != baseline_family_row["candidate_key"]
    )
    if material_end_to_end_winner is not None:
        decision = "psi_expressivity_likely_real_limiter"
    elif offline_improvement_exists:
        decision = "offline_fit_improves_but_end_to_end_does_not_so_downstream_coupling_more_likely"
    else:
        decision = "current_limiter_probably_not_psi_expressivity_or_input_representation"

    summary = {
        "phase": "Phase TF2",
        "stage": "corrective_psi_expressivity_suite",
        "offline_dataset_artifacts": {
            "train_npz_path": "offline_train_samples.npz",
            "val_npz_path": "offline_val_samples.npz",
            "identity_tangent_mode": _identity_tangent_mode(base_config),
        },
        "family_stage_offline_metrics": {
            row["candidate_key"]: {
                "psi_family": row["psi_family"],
                "psi_hidden_dims": json.loads(row["psi_hidden_dims"]),
                "time_encoding_variant": row["time_encoding_variant"],
                "parameter_count": row["parameter_count"],
                "val_bootstrap_target_mse": row["val_bootstrap_target_mse"],
                "val_relative_bootstrap_target_mse": row["val_relative_bootstrap_target_mse"],
                "val_bootstrap_cosine_similarity": row["val_bootstrap_cosine_similarity"],
                "val_identity_target_mse": row["val_identity_target_mse"],
                "val_identity_residual_error": row["val_identity_residual_error"],
                "val_hybrid_loss": row["val_hybrid_loss"],
                "runtime_per_batch_seconds": row["runtime_per_batch_seconds"],
            }
            for row in offline_family_rows
        },
        "encoding_stage_offline_metrics": {
            row["candidate_key"]: {
                "psi_family": row["psi_family"],
                "psi_hidden_dims": json.loads(row["psi_hidden_dims"]),
                "time_encoding_variant": row["time_encoding_variant"],
                "parameter_count": row["parameter_count"],
                "val_bootstrap_target_mse": row["val_bootstrap_target_mse"],
                "val_relative_bootstrap_target_mse": row["val_relative_bootstrap_target_mse"],
                "val_bootstrap_cosine_similarity": row["val_bootstrap_cosine_similarity"],
                "val_identity_target_mse": row["val_identity_target_mse"],
                "val_identity_residual_error": row["val_identity_residual_error"],
                "val_hybrid_loss": row["val_hybrid_loss"],
                "runtime_per_batch_seconds": row["runtime_per_batch_seconds"],
            }
            for row in offline_encoding_rows
        },
        "stage1_winner": stage1_winner_row["candidate_key"],
        "selected_end_to_end_challengers": [row["candidate_key"] for row in selected_challenger_rows],
        "primary_gate_passed_candidates": [row["candidate_key"] for row in primary_pass_rows],
        "fallback_challenger": None if fallback_row is None else fallback_row["candidate_key"],
        "capacity_bottleneck_signal": capacity_signal,
        "input_representation_bottleneck_signal": input_representation_signal,
        "output_parameterization_bottleneck_signal": output_parameterization_signal,
        "mean_std_val_accuracy_by_candidate": {
            key: {"mean": value["mean_val_accuracy"], "std": value["std_val_accuracy"]}
            for key, value in end_to_end_summaries.items()
        },
        "mean_std_test_accuracy_by_candidate": {
            key: {"mean": value["mean_test_accuracy"], "std": value["std_test_accuracy"]}
            for key, value in end_to_end_summaries.items()
        },
        "mean_gate_passing_epoch_count_by_candidate": {
            key: value["mean_gate_passing_epoch_count"]
            for key, value in end_to_end_summaries.items()
        },
        "mean_val_transported_final_energy_by_candidate": {
            key: value["mean_val_transported_final_energy"]
            for key, value in end_to_end_summaries.items()
        },
        "mean_wall_clock_runtime_by_candidate": {
            key: {
                "mean_total_wall_time_seconds": value["mean_total_wall_time_seconds"],
                "std_total_wall_time_seconds": value["std_total_wall_time_seconds"],
            }
            for key, value in end_to_end_summaries.items()
        },
        "gap_to_canonical_slow_pc_by_candidate": {
            key: {"mean_test_accuracy_gap": value["mean_test_accuracy_gap_to_slow_pc"]}
            for key, value in end_to_end_summaries.items()
        },
        "pairwise_delta_vs_current_corrective_default": pairwise_delta,
        "decision_logic_outcome": decision,
        "is_psi_expressivity_a_real_limiter": material_end_to_end_winner is not None,
        "best_material_end_to_end_candidate": material_end_to_end_winner,
        "next_single_narrow_research_move": (
            "if no offline fit gain appears, move away from psi expressivity; "
            "if offline fit gains fail to transfer, target downstream coupling rather than target construction again"
        ),
        "offline_family_csv_path": "offline_family_metrics.csv",
        "offline_encoding_csv_path": "offline_encoding_metrics.csv",
        "end_to_end_csv_path": "end_to_end_runs.csv",
    }
    _write_json(run_dir / "aggregate_summary.json", summary)
    return FMPCTF2PsiExpressivitySuiteRunResult(
        run_dir=run_dir,
        config=config_payload,
        offline_family_rows=offline_family_rows,
        offline_encoding_rows=offline_encoding_rows,
        end_to_end_rows=end_to_end_rows,
        summary=summary,
    )
