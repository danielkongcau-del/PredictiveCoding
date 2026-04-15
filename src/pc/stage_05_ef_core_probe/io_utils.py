
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from ..mlp_baseline import MLPNetwork
from ..models import PCNetwork
from .contracts import OutputLayout
from .residual_core import Stage05ResidualCoreNetworks

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

def _write_epoch_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("epoch metrics must contain at least one row.")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def _sigma2_payload(sigma2: float | tuple[float, ...]) -> float | list[float]:
    if isinstance(sigma2, tuple):
        return [float(value) for value in sigma2]
    return float(sigma2)

def _snapshot_pc_parameters(model: PCNetwork) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(layer.weight.copy(), layer.bias.copy()) for layer in model.layers]

def _restore_pc_parameters(model: PCNetwork, snapshot: list[tuple[np.ndarray, np.ndarray]]) -> None:
    if len(snapshot) != len(model.layers):
        raise ValueError("PC parameter snapshot must align with model layers.")
    for layer, (weight, bias) in zip(model.layers, snapshot, strict=True):
        layer.weight = weight.copy()
        layer.bias = bias.copy()

def _snapshot_mlp_parameters(network: MLPNetwork) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(layer.weight.copy(), layer.bias.copy()) for layer in network.layers]

def _restore_mlp_parameters(
    network: MLPNetwork,
    snapshot: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    if len(snapshot) != len(network.layers):
        raise ValueError("MLP parameter snapshot must align with network layers.")
    for layer, (weight, bias) in zip(network.layers, snapshot, strict=True):
        layer.weight = weight.copy()
        layer.bias = bias.copy()

def _snapshot_residual_core_parameters(
    residual_core: Stage05ResidualCoreNetworks,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]] | None]:
    return {
        "trajectory": _snapshot_mlp_parameters(residual_core.trajectory_network),
        "state": (
            _snapshot_mlp_parameters(residual_core.state_network)
            if residual_core.state_network is not None
            else None
        ),
    }

def _restore_residual_core_parameters(
    residual_core: Stage05ResidualCoreNetworks,
    snapshot: dict[str, list[tuple[np.ndarray, np.ndarray]] | None],
) -> None:
    trajectory_snapshot = snapshot.get("trajectory")
    if trajectory_snapshot is None:
        raise ValueError("Residual-core trajectory snapshot is required.")
    _restore_mlp_parameters(residual_core.trajectory_network, trajectory_snapshot)
    if residual_core.state_network is None:
        return
    state_snapshot = snapshot.get("state")
    if state_snapshot is None:
        raise ValueError("Residual-core state snapshot is required for two-branch mode.")
    _restore_mlp_parameters(residual_core.state_network, state_snapshot)
