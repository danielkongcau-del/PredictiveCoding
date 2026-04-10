from __future__ import annotations

import json
import importlib.util
import runpy
from pathlib import Path

from pc.jpc_bridge import probe_jpc_availability


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_jpc_bridge_probe_writes_expected_artifacts(tmp_path: Path) -> None:
    module = runpy.run_path(str(ROOT / "experiments" / "incremental_bridge" / "tf2_jpc_probe.py"))
    run = module["run"]

    result = run(output_root=tmp_path, run_id="tf2_jpc_probe_smoke", batch_size=6, inference_steps_horizon=4)
    run_dir = result.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "summary.json").exists()

    config = _read_json(run_dir / "config.json")
    summary = _read_json(run_dir / "summary.json")

    assert config["phase"] == "Phase Incremental Bridge"
    assert config["stage"] == "ifmpc_bridge_jpc_probe"
    assert summary["phase"] == "Phase Incremental Bridge"
    assert summary["stage"] == "ifmpc_bridge_jpc_probe"
    assert summary["probe_type"] == "reference_only"
    assert summary["benchmark_equivalence"] is False
    assert "current_repo_forward_init_stats" in summary
    assert "current_repo_energy_trajectory" in summary
    assert summary["current_repo_forward_init_stats"]["initial_target_clamped_energy"] >= 0.0
    assert len(summary["current_repo_energy_trajectory"]["full_energy_trace"]) == 5
    assert summary["current_repo_energy_trajectory"]["sampled_steps"] == [0, 1, 2, 4]
    assert summary["recommended_tf2_emphasis"] in {"substrate scaling", "incremental scheduling", "both"}
    assert "whether_many_step_inference_still_materially_outperforms_1_step_in_jpc_standard_pc" in summary
    assert "whether_many_step_inference_still_materially_outperforms_1_step_in_jpc_mupc" in summary

    jpc_status = summary["jpc_probe_status"]
    assert "available" in jpc_status
    if jpc_status["available"]:
        assert summary["jpc_standard_forward_init_stats"] is not None
        assert summary["jpc_mupc_forward_init_stats"] is not None
        assert summary["jpc_standard_energy_trajectory"] is not None
        assert summary["jpc_mupc_energy_trajectory"] is not None
        assert summary["whether_many_step_inference_still_materially_outperforms_1_step_in_jpc_standard_pc"] is not None
        assert summary["whether_many_step_inference_still_materially_outperforms_1_step_in_jpc_mupc"] is not None
    else:
        assert isinstance(jpc_status["reason"], str)
        assert len(jpc_status["reason"]) > 0
        assert summary["jpc_standard_forward_init_stats"] is None
        assert summary["jpc_mupc_forward_init_stats"] is None
        assert summary["jpc_standard_energy_trajectory"] is None
        assert summary["jpc_mupc_energy_trajectory"] is None


def test_probe_jpc_availability_fails_gracefully_when_optional_stack_is_missing(monkeypatch) -> None:
    original_find_spec = importlib.util.find_spec

    def _fake_find_spec(name: str, package: str | None = None):
        if name in {"jax", "diffrax", "equinox", "optax"}:
            return None
        return original_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)
    status = probe_jpc_availability()
    assert status["available"] is False
    assert isinstance(status["reason"], str)
    assert len(status["reason"]) > 0
