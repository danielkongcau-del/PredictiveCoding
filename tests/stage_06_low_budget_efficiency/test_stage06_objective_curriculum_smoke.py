from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.datasets import load_digits_split
from pc.stage_03_transport_core_v1.fmpc_tf1_flow import build_tf1_context
from pc.stage_05_ef_core_probe.fmpc_ef_exploratory_probe import _make_pc_model
from pc.stage_06_low_budget_efficiency.fmpc_stage06_objective_curriculum import (
    _build_stage05_scaffold_config,
    _make_psi_network,
    beta_obj_for_epoch,
    build_stage06_v1_objective_curriculum_energydrop_default_config,
    compute_energy_drop_penalty_and_output_delta,
    evaluate_fixed_point_contraction_terms,
)


def load_run():
    module = runpy.run_path(
        str(ROOT / "experiments" / "stage_06_low_budget_efficiency" / "fmpc_stage06_objective_curriculum.py")
    )
    return module["run"]


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_beta_obj_schedule_is_monotone_and_distinct_from_alpha_schedule() -> None:
    values = [beta_obj_for_epoch(8, epoch) for epoch in range(8)]
    assert values[0] == 0.0
    assert values[1] == 0.0
    assert values[2] == 0.0
    assert values[6] == 1.0
    assert values[7] == 1.0
    assert values == sorted(values)

    config = build_stage06_v1_objective_curriculum_energydrop_default_config(epochs=8)
    scaffold_config = _build_stage05_scaffold_config(config)
    alpha_start = scaffold_config.alpha_floor if scaffold_config.alpha_warmup_epochs == 0 else 1.0
    assert values[0] != alpha_start


def test_energy_drop_penalty_is_zero_when_rollout_energy_drops() -> None:
    terms = compute_energy_drop_penalty_and_output_delta(
        current_energy=np.array([[1.0], [1.0]], dtype=np.float64),
        rollout_energy=np.array([[0.8], [0.9]], dtype=np.float64),
        rollout_flow=np.ones((2, 3), dtype=np.float64),
        rollout_coefficient=np.full((2, 1), 0.5, dtype=np.float64),
        delta_margin=0.0,
    )
    assert terms.loss == 0.0
    np.testing.assert_allclose(terms.output_delta, 0.0, atol=1e-12)


def test_energy_drop_penalty_is_positive_when_rollout_energy_rises() -> None:
    terms = compute_energy_drop_penalty_and_output_delta(
        current_energy=np.array([[1.0], [1.0]], dtype=np.float64),
        rollout_energy=np.array([[1.2], [1.1]], dtype=np.float64),
        rollout_flow=np.ones((2, 3), dtype=np.float64),
        rollout_coefficient=np.full((2, 1), 0.5, dtype=np.float64),
        delta_margin=0.0,
    )
    assert terms.loss > 0.0
    assert np.all(terms.output_delta > 0.0)


def test_fixed_point_terms_are_deterministic_on_rollout_state() -> None:
    config = build_stage06_v1_objective_curriculum_energydrop_default_config()
    scaffold_config = _build_stage05_scaffold_config(config)
    split = load_digits_split(split_seed=0)
    x_batch = split.x_train[:6]
    y_batch = split.y_train[:6]
    model = _make_pc_model(scaffold_config)
    _ = _make_psi_network(scaffold_config)
    context = build_tf1_context(model, x_batch, y_batch)

    z_roll = context.z0.copy()
    coeff = np.full((z_roll.shape[0], 1), 0.25, dtype=np.float64)
    first = evaluate_fixed_point_contraction_terms(
        context,
        z_roll,
        coeff,
        tangent_epsilon=config.fixed_point_tangent_epsilon,
    )
    second = evaluate_fixed_point_contraction_terms(
        context,
        z_roll,
        coeff,
        tangent_epsilon=config.fixed_point_tangent_epsilon,
    )
    assert first.rollout_flow.shape == z_roll.shape
    assert first.output_delta.shape == z_roll.shape
    np.testing.assert_allclose(first.rollout_flow, second.rollout_flow, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(first.output_delta, second.output_delta, atol=1e-12, rtol=1e-12)


def test_stage06_probe_writes_expected_artifacts(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="stage06_smoke",
        output_layout="run_id_subdir",
        epochs=3,
        batch_size=128,
        eval_steps=5,
        transport_steps=2,
        layer_dims=(64, 16, 10),
    )

    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "epoch_metrics.csv").exists()
    assert (run_dir / "summary.json").exists()

    config = _read_json(run_dir / "config.json")
    summary = _read_json(run_dir / "summary.json")
    epoch_rows = _read_csv(run_dir / "epoch_metrics.csv")

    assert summary["phase"] == "FMPC Stage 06 Low-Budget Efficiency"
    assert summary["stage"] == "stage06_v1_objective_curriculum"
    assert summary["candidate_name"] == "stage06_v1_objective_curriculum_energydrop_default"
    assert summary["objective_contract_identity"] == (
        "objective_curriculum_plus_energydrop_fixed_point_contract"
    )
    assert summary["beta_obj_schedule_identity"] == "piecewise_linear_quarter_half_quarter"
    assert summary["beta_obj_is_distinct_from_alpha"] is True
    assert summary["energy_drop_penalty_enabled"] is True
    assert summary["fixed_point_contraction_penalty_enabled"] is True
    assert summary["midpoint_microfamily_continued"] is False
    assert summary["loss_breakdown_visible"] is True
    assert summary["runtime_proxy_seconds"] >= 0.0

    assert config["objective_contract"]["contract_identity"] == (
        "objective_curriculum_plus_energydrop_fixed_point_contract"
    )
    assert config["objective_contract"]["beta_obj_schedule_identity"] == (
        "piecewise_linear_quarter_half_quarter"
    )
    assert config["objective_contract"]["energy_drop_penalty_enabled"] is True
    assert config["objective_contract"]["fixed_point_contraction_penalty_enabled"] is True

    assert len(epoch_rows) == 3
    assert "beta_obj" in epoch_rows[0]
    assert "alpha" in epoch_rows[0]
    assert "train_traj_loss" in epoch_rows[0]
    assert "train_semi_loss" in epoch_rows[0]
    assert "train_energy_drop_loss" in epoch_rows[0]
    assert "train_fixed_point_loss" in epoch_rows[0]
