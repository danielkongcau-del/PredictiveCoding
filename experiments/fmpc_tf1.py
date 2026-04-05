from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.fmpc_tf1 import (
    FMPCTF1RunResult,
    build_tf1_preset_config,
    run_fmpc_tf1_experiment,
)


def run(
    output_root: str | Path = "outputs",
    run_id: str | None = None,
    preset_name: str = "mechanism_smoke",
    model_variant: str = "tf1_mlp_core",
    use_teacher_free_features: bool = False,
    feature_aware_tangents: bool = False,
    transport_steps: int = 2,
    epochs: int = 60,
    batch_size: int = 128,
    warmup_epochs: int = 5,
    hybrid_ramp_epochs: int = 10,
    identity_loss_weight: float = 0.1,
    eval_steps: int = 15,
    psi_hidden_dims: tuple[int, ...] = (128,),
    layer_dims: tuple[int, ...] | None = None,
) -> FMPCTF1RunResult:
    """Run one named teacher-free FMPC v1 preset on digits."""

    override_kwargs = {
        "output_root": output_root,
        "run_id": run_id,
        "model_variant": model_variant,  # type: ignore[dict-item]
        "use_teacher_free_features": use_teacher_free_features,
        "feature_aware_tangents": feature_aware_tangents,
        "transport_steps": transport_steps,
        "epochs": epochs,
        "batch_size": batch_size,
        "warmup_epochs": warmup_epochs,
        "hybrid_ramp_epochs": hybrid_ramp_epochs,
        "identity_loss_weight": identity_loss_weight,
        "eval_steps": eval_steps,
        "psi_hidden_dims": psi_hidden_dims,
    }
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
