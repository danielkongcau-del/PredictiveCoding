from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.tf1.fmpc_tf1 import (
    FMPCTF1RunResult,
    build_tf1_preset_config,
    run_fmpc_tf1_experiment,
)


def run(
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    preset_name: str = "mechanism_smoke",
    model_variant: str | None = None,
    use_teacher_free_features: bool | None = None,
    feature_aware_tangents: bool | None = None,
    transport_steps: int | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    warmup_epochs: int | None = None,
    hybrid_ramp_epochs: int | None = None,
    identity_loss_weight: float | None = None,
    checkpoint_selector: str | None = None,
    eval_steps: int | None = None,
    psi_hidden_dims: tuple[int, ...] | None = None,
    layer_dims: tuple[int, ...] | None = None,
) -> FMPCTF1RunResult:
    """Run one named teacher-free FMPC v1 preset on digits.

    `mechanism_smoke` remains the preserved smoke/default preset.
    `baseline_working_default` is the current evidence-driven but provisional
    working preset for the gate-feasible TF1 family.
    """

    override_kwargs = {
        "output_root": output_root,
        "run_id": run_id,
    }
    if model_variant is not None:
        override_kwargs["model_variant"] = model_variant
    if use_teacher_free_features is not None:
        override_kwargs["use_teacher_free_features"] = use_teacher_free_features
    if feature_aware_tangents is not None:
        override_kwargs["feature_aware_tangents"] = feature_aware_tangents
    if transport_steps is not None:
        override_kwargs["transport_steps"] = transport_steps
    if epochs is not None:
        override_kwargs["epochs"] = epochs
    if batch_size is not None:
        override_kwargs["batch_size"] = batch_size
    if warmup_epochs is not None:
        override_kwargs["warmup_epochs"] = warmup_epochs
    if hybrid_ramp_epochs is not None:
        override_kwargs["hybrid_ramp_epochs"] = hybrid_ramp_epochs
    if identity_loss_weight is not None:
        override_kwargs["identity_loss_weight"] = identity_loss_weight
    if checkpoint_selector is not None:
        override_kwargs["checkpoint_selector"] = checkpoint_selector
    if eval_steps is not None:
        override_kwargs["eval_steps"] = eval_steps
    if psi_hidden_dims is not None:
        override_kwargs["psi_hidden_dims"] = psi_hidden_dims
    if layer_dims is not None:
        override_kwargs["layer_dims"] = layer_dims
    config = build_tf1_preset_config(
        preset_name,  # type: ignore[arg-type]
        **override_kwargs,
    )
    return run_fmpc_tf1_experiment(config)


def main() -> None:
    result = run()
    print("Phase TF1 teacher-free FMPC v1 run completed.")
    print(f"Run directory: {result.run_dir}")
    print(f"Summary: {result.run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
