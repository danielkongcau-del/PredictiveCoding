from __future__ import annotations

import csv
import json
import runpy
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pc.datasets import load_digits_split
from pc.stage_03_transport_core_v1.fmpc_tf1_flow import build_tf1_context
from pc.stage_05_ef_core_probe.fmpc_ef_exploratory_probe import (
    _make_pc_model,
    _make_psi_network,
    build_corrected_residual_identity_target,
    build_fmpc_ef_exploratory_probe_config,
    lambda_id_for_epoch,
)


def load_run():
    module = runpy.run_path(str(ROOT / "experiments" / "stage_05_ef_core_probe" / "fmpc_ef_exploratory_probe.py"))
    return module["run"]


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_exploratory_probe_writes_expected_teacher_free_artifacts(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="exploratory_probe_smoke",
        epochs=6,
        warmup_epochs=2,
        batch_size=128,
        eval_steps=10,
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

    assert summary["phase"] == "post_incremental_bridge_exploratory"
    assert summary["stage"] == "ef_core_probe"
    assert summary["teacher_free"] is True
    assert summary["uses_teacher_artifacts"] is False
    assert summary["transport_family"] == "residual_meanflow_core"
    assert summary["residual_identity_mode"] == "residual_corrected_meanflow"
    assert summary["energy_substrate"] == "baseline_pc_energy"
    assert summary["local_flow_definition"] == "exact_negative_hidden_state_gradient"
    assert summary["direct_anchor_source"] == "self_bootstrap_local_field"
    assert summary["transport_scope"] == "train_only"
    assert summary["transport_steps"] == 2
    assert summary["u_psi_input_contract"] == "concat([z_t, target_onehot, t, r])"
    assert summary["bootstrap_target_contract"] == "m_boot = ((Phi_LF_r(z_t; c) - z_t) / r) - g_t"
    assert summary["residual_identity_target_contract"] == "m_id = r * D_T g_t + r * D_T m_psi"
    assert summary["selection_metric_source"] == "val_metric"
    assert summary["report_metric_source"] == "test_metric"
    assert summary["acceptance_contract"] == "mechanism_first"
    assert summary["task_accuracy_is_gate"] is False
    assert summary["no_teacher_dependency_in_target_construction"] is True
    assert summary["deterministic_artifacts"] is True
    assert config["transport"]["transport_family"] == "residual_meanflow_core"
    assert config["transport"]["residual_identity_mode"] == "residual_corrected_meanflow"
    assert config["transport"]["u_psi_input_contract"] == "concat([z_t, target_onehot, t, r])"
    assert config["transport"]["bootstrap_target_contract"] == "m_boot = ((Phi_LF_r(z_t; c) - z_t) / r) - g_t"
    assert config["transport"]["residual_identity_target_contract"] == "m_id = r * D_T g_t + r * D_T m_psi"
    assert config["transport"]["no_teacher_dependency_in_target_construction"] is True
    assert config["transport"]["use_teacher_free_features"] is False
    assert len(epoch_rows) == 6
    assert "lambda_id" in epoch_rows[0]
    assert "train_total_loss" in epoch_rows[0]
    assert "train_identity_loss" in epoch_rows[0]
    assert "val_one_step_energy_delta_vs_identity" in epoch_rows[0]
    assert "val_configured_fixed_point_residual_delta_vs_identity" in epoch_rows[0]


def test_corrected_residual_identity_target_includes_anchor_derivative_term() -> None:
    config = build_fmpc_ef_exploratory_probe_config(
        run_seed=0,
        data_seed=0,
        model_init_seed=0,
        psi_init_seed=0,
        batch_order_seed=0,
        tangent_epsilon=1e-3,
    )
    split = load_digits_split(
        split_seed=config.data_seed,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
    )
    x_batch = split.x_train[:8]
    y_batch = split.y_train[:8]
    model = _make_pc_model(config)
    psi_network = _make_psi_network(config)
    context = build_tf1_context(model, x_batch, y_batch)

    corrected = build_corrected_residual_identity_target(
        context,
        psi_network,
        context.z0,
        context.targets,
        t=0.25,
        r=0.75,
        tangent_epsilon=config.tangent_epsilon,
    )

    np.testing.assert_allclose(
        corrected.target,
        corrected.anchor_term + corrected.residual_term,
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        corrected.target - corrected.residual_term,
        corrected.anchor_term,
        atol=1e-12,
        rtol=1e-12,
    )
    assert np.linalg.norm(corrected.anchor_term) > 1e-12
    assert not np.allclose(corrected.target, corrected.residual_term)


def test_lambda_id_schedule_has_zero_warmup_and_positive_ramp() -> None:
    config = build_fmpc_ef_exploratory_probe_config(
        identity_loss_weight=0.2,
        lambda_id_warmup_epochs=2,
        lambda_id_ramp_epochs=3,
    )

    lambda_0 = lambda_id_for_epoch(config, 0)
    lambda_1 = lambda_id_for_epoch(config, 1)
    lambda_2 = lambda_id_for_epoch(config, 2)
    lambda_3 = lambda_id_for_epoch(config, 3)
    lambda_4 = lambda_id_for_epoch(config, 4)
    lambda_5 = lambda_id_for_epoch(config, 5)

    assert lambda_0 == pytest.approx(0.0)
    assert lambda_1 == pytest.approx(0.0)
    assert 0.0 < lambda_2 < lambda_3 < lambda_4 <= 0.2
    assert lambda_5 == pytest.approx(0.2)


def test_exploratory_probe_is_deterministic_under_fixed_seeds(tmp_path: Path) -> None:
    run = load_run()
    result_a = run(
        output_root=tmp_path / "a",
        run_id="deterministic_a",
        epochs=4,
        warmup_epochs=2,
        batch_size=128,
        eval_steps=8,
        transport_steps=2,
        run_seed=7,
        data_seed=7,
        model_init_seed=7,
        psi_init_seed=7,
        batch_order_seed=7,
    )
    result_b = run(
        output_root=tmp_path / "b",
        run_id="deterministic_b",
        epochs=4,
        warmup_epochs=2,
        batch_size=128,
        eval_steps=8,
        transport_steps=2,
        run_seed=7,
        data_seed=7,
        model_init_seed=7,
        psi_init_seed=7,
        batch_order_seed=7,
    )

    summary_a = _read_json(result_a.run_dir / "summary.json")
    summary_b = _read_json(result_b.run_dir / "summary.json")

    assert summary_a["selected_epoch"] == summary_b["selected_epoch"]
    assert summary_a["val_accuracy"] == pytest.approx(summary_b["val_accuracy"])
    assert summary_a["test_accuracy"] == pytest.approx(summary_b["test_accuracy"])
    assert summary_a["val_output_mse"] == pytest.approx(summary_b["val_output_mse"])
    assert summary_a["test_output_mse"] == pytest.approx(summary_b["test_output_mse"])
    for step_name in ("one_step", "configured_steps"):
        metrics_a = summary_a["mechanism_metrics"][step_name]
        metrics_b = summary_b["mechanism_metrics"][step_name]
        assert metrics_a["transport_steps"] == metrics_b["transport_steps"]
        assert metrics_a["transported_final_energy"] == pytest.approx(
            metrics_b["transported_final_energy"]
        )
        assert metrics_a["energy_delta_vs_identity"] == pytest.approx(
            metrics_b["energy_delta_vs_identity"]
        )
        assert metrics_a["transported_final_fixed_point_residual_rms"] == pytest.approx(
            metrics_b["transported_final_fixed_point_residual_rms"]
        )
    assert summary_a["selected_epoch_lambda_id"] == pytest.approx(summary_b["selected_epoch_lambda_id"])
    assert summary_a["transport_family"] == summary_b["transport_family"]
    assert summary_a["residual_identity_mode"] == summary_b["residual_identity_mode"]


def test_exploratory_probe_smoke_improves_mechanism_metrics_vs_identity(tmp_path: Path) -> None:
    result = load_run()(
        output_root=tmp_path,
        run_id="exploratory_probe_mechanism_smoke",
        epochs=8,
        warmup_epochs=3,
        batch_size=128,
        eval_steps=12,
        transport_steps=2,
        layer_dims=(64, 16, 10),
    )

    summary = _read_json(result.run_dir / "summary.json")
    acceptance = summary["mechanism_acceptance"]
    mechanism = summary["mechanism_metrics"]

    assert acceptance["one_step_energy_decrease_vs_identity"] is True
    assert acceptance["configured_steps_fixed_point_residual_decrease_vs_identity"] is True
    assert mechanism["one_step"]["transported_final_energy"] < mechanism["one_step"]["identity_final_energy"]
    assert (
        mechanism["configured_steps"]["transported_final_fixed_point_residual_rms"]
        < mechanism["configured_steps"]["identity_final_fixed_point_residual_rms"]
    )
