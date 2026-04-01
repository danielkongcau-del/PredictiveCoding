from __future__ import annotations

import json
from pathlib import Path

from pc.benchmark_specs import BENCHMARK_NAMES, get_benchmark_spec
from pc.comparison import run_benchmark_comparison


def test_comparison_runs_write_expected_artifacts(tmp_path: Path) -> None:
    for benchmark_name in BENCHMARK_NAMES:
        result = run_benchmark_comparison(
            benchmark_name,
            output_root=tmp_path,
            run_id=f"{benchmark_name}_compare",
            plot_energy=False,
        )
        spec = get_benchmark_spec(benchmark_name)
        comparison_root = tmp_path / f"compare_{benchmark_name}"

        assert result.run_dir == comparison_root
        assert (comparison_root / "pc" / "summary.json").exists()
        assert (comparison_root / "mlp" / "summary.json").exists()
        assert (comparison_root / "comparison_summary.json").exists()

        with (comparison_root / "comparison_summary.json").open("r", encoding="utf-8") as handle:
            comparison_summary = json.load(handle)
        with (comparison_root / "mlp" / "config.json").open("r", encoding="utf-8") as handle:
            mlp_config = json.load(handle)
        with (comparison_root / "mlp" / "summary.json").open("r", encoding="utf-8") as handle:
            mlp_summary = json.load(handle)

        assert comparison_summary["benchmark_name"] == benchmark_name
        assert comparison_summary["metric_name"] == spec.primary_metric_name
        assert comparison_summary["metric_higher_is_better"] == spec.primary_metric_higher_is_better
        assert comparison_summary["primary_metric_name"] == spec.primary_metric_name
        assert comparison_summary["baseline_metric_name"] == spec.baseline_metric_name
        assert "pc_train_metric_value" in comparison_summary
        assert "pc_val_metric_value" in comparison_summary
        assert "pc_test_metric_value" in comparison_summary
        assert "mlp_train_metric_value" in comparison_summary
        assert "mlp_val_metric_value" in comparison_summary
        assert "mlp_test_metric_value" in comparison_summary
        assert comparison_summary["winner_tolerance_rtol"] == 1.0e-12
        assert comparison_summary["winner_tolerance_atol"] == 1.0e-12
        assert comparison_summary["pc_summary_path"] == "pc/summary.json"
        assert comparison_summary["mlp_summary_path"] == "mlp/summary.json"
        assert comparison_summary["winner"] in {"pc", "mlp", "tie"}
        assert mlp_config["run_seed"] == spec.run_seed
        assert mlp_config["data_seed"] == spec.data_seed
        assert mlp_config["model_init_seed"] == spec.model_init_seed
        assert mlp_config["logging"]["output_layout"] == "single_dir"
        assert mlp_config["seeds"] == {
            "run_seed": spec.run_seed,
            "data_seed": spec.data_seed,
            "model_init_seed": spec.model_init_seed,
        }
        assert mlp_summary["run_seed"] == spec.run_seed
        assert mlp_summary["data_seed"] == spec.data_seed
        assert mlp_summary["model_init_seed"] == spec.model_init_seed
        assert "train_metric" in mlp_summary
        assert "val_metric" in mlp_summary
        assert "test_metric" in mlp_summary
        assert mlp_summary["metric_name"] == spec.primary_metric_name
        assert mlp_summary["seeds"] == {
            "run_seed": spec.run_seed,
            "data_seed": spec.data_seed,
            "model_init_seed": spec.model_init_seed,
        }


def test_mlp_config_records_actual_output_layout_for_archival_runs(tmp_path: Path) -> None:
    result = run_benchmark_comparison(
        "toy_regression",
        output_root=tmp_path,
        run_id="archival_compare",
        output_layout="run_id_subdir",
        plot_energy=False,
    )

    with (result.mlp_run_dir / "config.json").open("r", encoding="utf-8") as handle:
        mlp_config = json.load(handle)

    assert result.run_dir == tmp_path / "compare_toy_regression" / "archival_compare"
    assert mlp_config["logging"]["output_layout"] == "run_id_subdir"
