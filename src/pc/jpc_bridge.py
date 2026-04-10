from __future__ import annotations

import importlib.metadata
import importlib.machinery
import importlib.util
import json
import shutil
import sys
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .datasets import load_digits_split
from .energy import compute_cache, total_energy
from .inference import build_clamped_mask, compute_state_gradients, initialize_states, run_inference
from .layers import init_mlp_layers
from .models import PCNetwork

OutputLayout = Literal["single_dir", "run_id_subdir"]


@dataclass
class TF2JPCProbeConfig:
    """Configuration for the minimal TF2 JPC bridge probe."""

    experiment_name: str = "tf2_jpc_probe"
    output_root: str | Path = "outputs"
    run_id: str | None = None
    output_layout: OutputLayout = "single_dir"
    data_seed: int = 0
    model_seed: int = 0
    batch_size: int = 8
    layer_dims: tuple[int, ...] = (64, 64, 10)
    hidden_activation: str = "tanh"
    output_activation: str = "identity"
    weight_scale: float = 0.05
    sigma2: float = 1.0
    eta_x: float = 0.10
    inference_steps_horizon: int = 8
    jpc_act_fn: str = "relu"
    jpc_use_bias: bool = False

    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed_{self.data_seed}"


@dataclass
class TF2JPCProbeRunResult:
    run_dir: Path
    config: dict[str, Any]
    summary: dict[str, Any]


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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _jpc_submodule_root() -> Path:
    return _repo_root() / "third_party" / "jpc"


def _import_optional_jpc_stack() -> dict[str, Any]:
    """Import the optional JPC/JAX stack with local compatibility shims.

    The JPC checkout is kept read-only. Compatibility fixes for the local probe
    environment are applied only in this process via `sys.modules` and a
    temporary `importlib.metadata.version` patch.
    """

    jpc_root = _jpc_submodule_root()
    if not jpc_root.exists():
        return {
            "available": False,
            "reason": f"JPC submodule not found at '{jpc_root}'.",
            "submodule_root": str(jpc_root),
        }

    required_modules = ("jax", "diffrax", "equinox", "optax", "optimistix", "jaxtyping")
    missing_modules = [module_name for module_name in required_modules if importlib.util.find_spec(module_name) is None]
    if missing_modules:
        return {
            "available": False,
            "reason": (
                "Optional JPC probe dependencies are unavailable. "
                f"Missing modules: {', '.join(missing_modules)}"
            ),
            "submodule_root": str(jpc_root),
        }

    if str(jpc_root) not in sys.path:
        sys.path.insert(0, str(jpc_root))

    try:
        if importlib.util.find_spec("jaxlib.xla_extension") is None and importlib.util.find_spec("jaxlib._jax") is not None:
            import jaxlib._jax as jaxlib_internal  # type: ignore

            shim = types.ModuleType("jaxlib.xla_extension")
            shim.PjitFunction = jaxlib_internal.PjitFunction
            shim.__spec__ = importlib.machinery.ModuleSpec("jaxlib.xla_extension", loader=None)
            sys.modules.setdefault("jaxlib.xla_extension", shim)

        original_version = importlib.metadata.version

        def _patched_version(package_name: str) -> str:
            if package_name == "jpc":
                try:
                    return original_version(package_name)
                except importlib.metadata.PackageNotFoundError:
                    return "submodule"
            return original_version(package_name)

        importlib.metadata.version = _patched_version
        try:
            import jpc  # type: ignore
        finally:
            importlib.metadata.version = original_version

        import jax  # type: ignore
        import diffrax  # type: ignore
        import equinox  # type: ignore
        import optax  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised indirectly in smoke tests
        return {
            "available": False,
            "reason": (
                "Optional JPC probe dependencies are unavailable. "
                f"Import failed with {type(exc).__name__}: {exc}"
            ),
            "submodule_root": str(jpc_root),
        }

    return {
        "available": True,
        "reason": None,
        "submodule_root": str(jpc_root),
        "jpc_version": getattr(jpc, "__version__", "unknown"),
        "jax_version": getattr(jax, "__version__", "unknown"),
        "diffrax_version": getattr(diffrax, "__version__", "unknown"),
        "equinox_version": getattr(equinox, "__version__", "unknown"),
        "optax_version": getattr(optax, "__version__", "unknown"),
        "modules": {
            "jpc": jpc,
            "jax": jax,
            "diffrax": diffrax,
            "equinox": equinox,
            "optax": optax,
        },
    }


def probe_jpc_availability() -> dict[str, Any]:
    """Probe whether the optional JPC/JAX stack is importable in the current environment."""

    status = _import_optional_jpc_stack()
    if "modules" in status:
        status = {key: value for key, value in status.items() if key != "modules"}
    return status


def _rms(array: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(array, dtype=np.float64) ** 2)))


def _mean_l2(array: np.ndarray) -> float:
    batch = np.asarray(array, dtype=np.float64)
    return float(np.mean(np.linalg.norm(batch, axis=1)))


def _activity_stats_by_layer(activities: list[np.ndarray], *, layer_index_offset: int = 0) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    for local_index, activity in enumerate(activities, start=1):
        layer_index = local_index + layer_index_offset
        stats.append(
            {
                "layer_index": int(layer_index),
                "feature_dim": int(activity.shape[1]),
                "rms": _rms(activity),
                "mean_l2_norm": _mean_l2(activity),
            }
        )
    return stats


def _free_gradient_stats(
    gradients: list[np.ndarray | None],
    *,
    include_output: bool = False,
) -> dict[str, float | None]:
    chunks: list[np.ndarray] = []
    upper = len(gradients) if include_output else len(gradients) - 1
    for gradient in gradients[1:upper]:
        if gradient is not None:
            chunks.append(np.asarray(gradient, dtype=np.float64))
    if not chunks:
        return {
            "residual_rms": None,
            "residual_mean_l2_norm": None,
        }
    combined = np.concatenate(chunks, axis=1)
    return {
        "residual_rms": _rms(combined),
        "residual_mean_l2_norm": _mean_l2(combined),
    }


def _sample_energy_trace(energy_trace: list[float], horizon_steps: int) -> dict[str, Any]:
    full_trace = [float(value) for value in energy_trace]
    candidate_steps = [0, 1, 2, 4, int(horizon_steps)]
    sampled_steps = sorted({step for step in candidate_steps if 0 <= step < len(full_trace)})
    return {
        "sampled_steps": [int(step) for step in sampled_steps],
        "sampled_energies": [float(full_trace[step]) for step in sampled_steps],
        "full_energy_trace": full_trace,
        "step_to_energy": {str(step): float(full_trace[step]) for step in sampled_steps},
    }


def _many_step_materially_beats_one_step(energy_trace: list[float]) -> bool | None:
    if len(energy_trace) <= 2:
        return None
    one_step = float(energy_trace[1])
    many_step = float(energy_trace[-1])
    threshold = max(1e-8, 0.01 * max(abs(one_step), 1e-8))
    return bool((one_step - many_step) > threshold)


def probe_current_repo_forward_init(
    model: PCNetwork,
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    """Probe forward-initialized hidden activity statistics for current repo PC.

    Shape contract:
    - `x`: `(batch, input_dim)`
    - `y`: `(batch, target_dim)`
    """

    states = initialize_states(model.layers, x, y=y, init=model.state_init, mode="train")
    clamped_mask = build_clamped_mask(len(model.layers) + 1, mode="train")
    cache = compute_cache(states, model.layers)
    gradients = compute_state_gradients(states, cache, model.layers, clamped_mask)
    hidden_states = [np.asarray(state, dtype=np.float64) for state in states[1:-1]]
    return {
        "batch_size": int(x.shape[0]),
        "layer_dims": [int(states[0].shape[1])] + [int(state.shape[1]) for state in states[1:]],
        "state_init": str(model.state_init),
        "hidden_layer_stats": _activity_stats_by_layer(hidden_states, layer_index_offset=0),
        "initial_target_clamped_energy": float(total_energy(cache, model.layers, x.shape[0])),
        "includes_output_layer": False,
        **_free_gradient_stats(gradients, include_output=False),
    }


def probe_current_repo_inference_energy(
    model: PCNetwork,
    x: np.ndarray,
    y: np.ndarray,
    *,
    horizon_steps: int,
) -> dict[str, Any]:
    """Probe current repo target-clamped energy trajectory.

    Shape contract:
    - `x`: `(batch, input_dim)`
    - `y`: `(batch, target_dim)`
    """

    states = initialize_states(model.layers, x, y=y, init=model.state_init, mode="train")
    clamped_mask = build_clamped_mask(len(model.layers) + 1, mode="train")
    result = run_inference(
        states,
        model.layers,
        clamped_mask,
        eta_x=model.eta_x,
        steps=horizon_steps,
        backend=model.inference_backend,
        record_trace=True,
    )
    final_gradients = compute_state_gradients(result.states, result.cache, model.layers, clamped_mask)
    return {
        **_sample_energy_trace(result.energy_trace, horizon_steps),
        "horizon_steps": int(horizon_steps),
        "final_energy": float(result.final_energy),
        "many_step_materially_beats_one_step": _many_step_materially_beats_one_step(result.energy_trace),
        **_free_gradient_stats(final_gradients, include_output=False),
    }


def _jpc_width_and_depth(layer_dims: tuple[int, ...]) -> tuple[int, int]:
    if len(layer_dims) < 2:
        raise ValueError("layer_dims must include at least input and output dimensions.")
    if len(layer_dims) == 2:
        return int(max(layer_dims[0], layer_dims[1])), 1
    return int(layer_dims[1]), int(len(layer_dims) - 1)


def _extract_jpc_activity_arrays(activities: Any) -> list[np.ndarray]:
    return [np.asarray(activity, dtype=np.float64) for activity in list(activities)]


def _forward_init_stability_proxy(stats: dict[str, Any] | None) -> float | None:
    if stats is None:
        return None
    layer_stats = stats.get("activity_layer_stats", stats.get("hidden_layer_stats"))
    if not isinstance(layer_stats, list) or len(layer_stats) == 0:
        return None
    rms_values = [float(layer["rms"]) for layer in layer_stats]
    min_rms = max(min(rms_values), 1e-12)
    return float(max(rms_values) / min_rms)


def _run_jpc_variant_probe(
    *,
    param_type: str,
    x: np.ndarray,
    y: np.ndarray,
    layer_dims: tuple[int, ...],
    act_fn: str,
    use_bias: bool,
    seed: int,
    horizon_steps: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    jpc_stack = _import_optional_jpc_stack()
    if not bool(jpc_stack["available"]):
        raise RuntimeError(str(jpc_stack["reason"]))

    jpc = jpc_stack["modules"]["jpc"]
    import jax.numpy as jnp  # type: ignore
    import jax.random as jr  # type: ignore
    from diffrax import Heun, PIDController  # type: ignore

    width, depth = _jpc_width_and_depth(layer_dims)
    key = jr.PRNGKey(seed)
    model = jpc.make_mlp(
        key=key,
        input_dim=int(layer_dims[0]),
        width=width,
        depth=depth,
        output_dim=int(layer_dims[-1]),
        act_fn=act_fn,
        use_bias=use_bias,
        param_type=param_type,
    )
    skip_model = jpc.make_skip_model(depth) if param_type == "mupc" and depth > 1 else None
    x_jnp = jnp.asarray(x)
    y_jnp = jnp.asarray(y)
    activities = jpc.init_activities_with_ffwd(
        model=model,
        input=x_jnp,
        skip_model=skip_model,
        param_type=param_type,
    )
    activity_arrays = _extract_jpc_activity_arrays(activities)
    activity_norms = np.asarray(jpc.compute_activity_norms(activities), dtype=np.float64)
    rms_ratio_proxy = _forward_init_stability_proxy(
        {"activity_layer_stats": _activity_stats_by_layer(activity_arrays, layer_index_offset=0)}
    )
    forward_stats = {
        "available": True,
        "param_type": str(param_type),
        "uses_skip_model": bool(skip_model is not None),
        "includes_output_layer": True,
        "activity_layer_stats": _activity_stats_by_layer(activity_arrays, layer_index_offset=0),
        "activity_mean_l2_norms_by_layer": [float(value) for value in activity_norms.tolist()],
        "conditioning_proxy": (
            None
            if rms_ratio_proxy is None
            else {
                "metric": "max_layer_rms_over_min_layer_rms",
                "value": float(rms_ratio_proxy),
            }
        ),
    }

    solution = jpc.solve_inference(
        params=(model, skip_model),
        activities=activities,
        output=y_jnp,
        input=x_jnp,
        loss_id="mse",
        param_type=param_type,
        solver=Heun(),
        max_t1=int(horizon_steps),
        stepsize_controller=PIDController(rtol=1e-3, atol=1e-3),
        record_every=1,
    )
    if np.asarray(solution[0]).ndim != 3:
        raise RuntimeError("JPC probe expected recorded activity trajectories with a time axis.")
    recorded_steps = int(np.asarray(solution[0]).shape[0])
    t_max = min(recorded_steps, int(horizon_steps) + 1, 500)
    infer_energies = jpc.compute_infer_energies(
        params=(model, skip_model),
        activities_iters=solution,
        t_max=t_max,
        y=y_jnp,
        x=x_jnp,
        loss="mse",
        param_type=param_type,
    )
    layer_energy_trace = np.asarray(infer_energies, dtype=np.float64)
    total_energy_trace = np.sum(layer_energy_trace[:, :t_max], axis=0).astype(np.float64).tolist()
    energy_summary = {
        **_sample_energy_trace([float(value) for value in total_energy_trace], min(int(horizon_steps), t_max - 1)),
        "recorded_points": int(t_max),
        "many_step_materially_beats_one_step": _many_step_materially_beats_one_step(
            [float(value) for value in total_energy_trace]
        ),
    }
    return forward_stats, energy_summary


def probe_jpc_forward_init(
    *,
    param_type: Literal["sp", "mupc"],
    x: np.ndarray,
    y: np.ndarray,
    layer_dims: tuple[int, ...],
    act_fn: str,
    use_bias: bool,
    seed: int,
) -> dict[str, Any] | None:
    """Probe JPC feedforward activity statistics for standard or μPC parameterisations.

    Shape contract:
    - `x`: `(batch, input_dim)`
    - `y`: `(batch, target_dim)`
    """

    availability = probe_jpc_availability()
    if not bool(availability["available"]):
        return None
    stats, _ = _run_jpc_variant_probe(
        param_type=param_type,
        x=x,
        y=y,
        layer_dims=layer_dims,
        act_fn=act_fn,
        use_bias=use_bias,
        seed=seed,
        horizon_steps=1,
    )
    return stats


def probe_jpc_inference_energy(
    *,
    param_type: Literal["sp", "mupc"],
    x: np.ndarray,
    y: np.ndarray,
    layer_dims: tuple[int, ...],
    act_fn: str,
    use_bias: bool,
    seed: int,
    horizon_steps: int,
) -> dict[str, Any] | None:
    """Probe JPC inference energy trajectory for standard or μPC parameterisations.

    Shape contract:
    - `x`: `(batch, input_dim)`
    - `y`: `(batch, target_dim)`
    """

    availability = probe_jpc_availability()
    if not bool(availability["available"]):
        return None
    _, energy = _run_jpc_variant_probe(
        param_type=param_type,
        x=x,
        y=y,
        layer_dims=layer_dims,
        act_fn=act_fn,
        use_bias=use_bias,
        seed=seed,
        horizon_steps=horizon_steps,
    )
    return energy


def _config_payload(config: TF2JPCProbeConfig) -> dict[str, Any]:
    return {
        "phase": "Phase Incremental Bridge",
        "stage": "ifmpc_bridge_jpc_probe",
        "probe_type": "reference_only",
        "benchmark_equivalence": False,
        "output_layout": str(config.output_layout),
        "data_seed": int(config.data_seed),
        "model_seed": int(config.model_seed),
        "batch_size": int(config.batch_size),
        "layer_dims": [int(dim) for dim in config.layer_dims],
        "hidden_activation": str(config.hidden_activation),
        "output_activation": str(config.output_activation),
        "weight_scale": float(config.weight_scale),
        "sigma2": float(config.sigma2),
        "eta_x": float(config.eta_x),
        "inference_steps_horizon": int(config.inference_steps_horizon),
        "jpc_act_fn": str(config.jpc_act_fn),
        "jpc_use_bias": bool(config.jpc_use_bias),
        "jpc_submodule_root": str(_jpc_submodule_root()),
    }


def _recommended_tf2_emphasis(
    *,
    mupc_improves_init_stability: bool | None,
    jpc_standard_many_step_beats_one_step: bool | None,
    jpc_mupc_many_step_beats_one_step: bool | None,
) -> str:
    if (
        mupc_improves_init_stability is True
        and (jpc_standard_many_step_beats_one_step is True or jpc_mupc_many_step_beats_one_step is True)
    ):
        return "both"
    if mupc_improves_init_stability is True:
        return "substrate scaling"
    return "incremental scheduling"


def run_tf2_jpc_probe(config: TF2JPCProbeConfig) -> TF2JPCProbeRunResult:
    """Run the minimal TF2 JPC bridge probe and materialize JSON artifacts."""

    run_dir = _prepare_run_dir(
        _resolve_run_dir(
            config.output_root,
            config.experiment_name,
            config.resolved_run_id(),
            config.output_layout,
        )
    )
    config_payload = _config_payload(config)
    _write_json(run_dir / "config.json", config_payload)

    split = load_digits_split(split_seed=config.data_seed)
    x_batch = np.asarray(split.x_train[: config.batch_size], dtype=np.float64)
    y_batch = np.asarray(split.y_train[: config.batch_size], dtype=np.float64)

    layers = init_mlp_layers(
        config.layer_dims,
        hidden_activation=config.hidden_activation,
        output_activation=config.output_activation,
        weight_scale=config.weight_scale,
        sigma2=config.sigma2,
        seed=config.model_seed,
    )
    model = PCNetwork(
        layers=layers,
        eta_x=config.eta_x,
        eta_w=0.02,
        eta_b=0.02,
        train_steps=config.inference_steps_horizon,
        eval_steps=config.inference_steps_horizon,
        state_init="forward",
    )

    current_forward_stats = probe_current_repo_forward_init(model, x_batch, y_batch)
    current_energy_trajectory = probe_current_repo_inference_energy(
        model,
        x_batch,
        y_batch,
        horizon_steps=config.inference_steps_horizon,
    )

    jpc_status = probe_jpc_availability()
    jpc_standard_forward_init_stats: dict[str, Any] | None = None
    jpc_mupc_forward_init_stats: dict[str, Any] | None = None
    jpc_standard_energy_trajectory: dict[str, Any] | None = None
    jpc_mupc_energy_trajectory: dict[str, Any] | None = None
    mupc_improves_init_stability: bool | None = None
    jpc_standard_many_step_beats_one_step: bool | None = None
    jpc_mupc_many_step_beats_one_step: bool | None = None
    forward_init_stability_proxy: dict[str, Any] | None = None

    if bool(jpc_status["available"]):
        jpc_standard_forward_init_stats, jpc_standard_energy_trajectory = _run_jpc_variant_probe(
            param_type="sp",
            x=x_batch,
            y=y_batch,
            layer_dims=config.layer_dims,
            act_fn=config.jpc_act_fn,
            use_bias=config.jpc_use_bias,
            seed=config.model_seed,
            horizon_steps=config.inference_steps_horizon,
        )
        jpc_mupc_forward_init_stats, jpc_mupc_energy_trajectory = _run_jpc_variant_probe(
            param_type="mupc",
            x=x_batch,
            y=y_batch,
            layer_dims=config.layer_dims,
            act_fn=config.jpc_act_fn,
            use_bias=config.jpc_use_bias,
            seed=config.model_seed,
            horizon_steps=config.inference_steps_horizon,
        )
        standard_proxy = _forward_init_stability_proxy(jpc_standard_forward_init_stats)
        mupc_proxy = _forward_init_stability_proxy(jpc_mupc_forward_init_stats)
        if standard_proxy is not None and mupc_proxy is not None:
            mupc_improves_init_stability = bool(mupc_proxy < standard_proxy)
            forward_init_stability_proxy = {
                "metric": "max_layer_rms_over_min_layer_rms",
                "jpc_standard": float(standard_proxy),
                "jpc_mupc": float(mupc_proxy),
            }
        jpc_standard_many_step_beats_one_step = (
            None
            if jpc_standard_energy_trajectory is None
            else bool(jpc_standard_energy_trajectory["many_step_materially_beats_one_step"])
        )
        jpc_mupc_many_step_beats_one_step = (
            None
            if jpc_mupc_energy_trajectory is None
            else bool(jpc_mupc_energy_trajectory["many_step_materially_beats_one_step"])
        )

    summary = {
        "phase": "Phase Incremental Bridge",
        "stage": "ifmpc_bridge_jpc_probe",
        "probe_type": "reference_only",
        "benchmark_equivalence": False,
        "comparability_note": (
            "This is a diagnostic probe, not an apples-to-apples benchmark. "
            "The current repo uses the existing NumPy layered PC substrate, "
            "while JPC is an optional JAX reference implementation with its own model parameterisation."
        ),
        "current_repo_forward_init_stats": current_forward_stats,
        "jpc_standard_forward_init_stats": jpc_standard_forward_init_stats,
        "jpc_mupc_forward_init_stats": jpc_mupc_forward_init_stats,
        "current_repo_energy_trajectory": current_energy_trajectory,
        "jpc_standard_energy_trajectory": jpc_standard_energy_trajectory,
        "jpc_mupc_energy_trajectory": jpc_mupc_energy_trajectory,
        "jpc_probe_status": jpc_status,
        "forward_init_stability_proxy": forward_init_stability_proxy,
        "whether_mupc_like_scaling_appears_to_improve_forward_init_stability": mupc_improves_init_stability,
        "whether_many_step_inference_still_materially_outperforms_1_step_in_jpc_standard_pc": (
            jpc_standard_many_step_beats_one_step
        ),
        "whether_many_step_inference_still_materially_outperforms_1_step_in_jpc_mupc": (
            jpc_mupc_many_step_beats_one_step
        ),
        "whether_many_step_inference_still_materially_outperforms_1_step_in_the_jpc_probe": (
            None
            if jpc_standard_many_step_beats_one_step is None and jpc_mupc_many_step_beats_one_step is None
            else bool(
                (jpc_standard_many_step_beats_one_step is True)
                or (jpc_mupc_many_step_beats_one_step is True)
            )
        ),
        "recommended_tf2_emphasis": _recommended_tf2_emphasis(
            mupc_improves_init_stability=mupc_improves_init_stability,
            jpc_standard_many_step_beats_one_step=jpc_standard_many_step_beats_one_step,
            jpc_mupc_many_step_beats_one_step=jpc_mupc_many_step_beats_one_step,
        ),
    }
    _write_json(run_dir / "summary.json", summary)
    return TF2JPCProbeRunResult(
        run_dir=run_dir,
        config=config_payload,
        summary=summary,
    )
