from __future__ import annotations

import numpy as np
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_run(script_name: str):
    module = runpy.run_path(str(ROOT / "experiments" / script_name))
    return module["run"]


def test_reproducible_outputs_with_explicit_run_id(tmp_path: Path) -> None:
    run = load_run("toy_regression.py")
    first = run(output_root=tmp_path / "run_a", run_id="repro_fixed", plot_energy=False)
    second = run(output_root=tmp_path / "run_b", run_id="repro_fixed", plot_energy=False)

    assert first.summary == second.summary
    assert first.epoch_metrics == second.epoch_metrics
    assert first.trace_manifest == second.trace_manifest
    assert sorted(first.trace_arrays.keys()) == sorted(second.trace_arrays.keys())

    for key in first.trace_arrays:
        np.testing.assert_allclose(first.trace_arrays[key], second.trace_arrays[key])
