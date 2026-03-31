from __future__ import annotations

from pc.comparison import select_comparison_winner
from pc.metrics import metric_higher_is_better


def test_metric_higher_is_better_matches_current_metrics() -> None:
    assert metric_higher_is_better("accuracy") is True
    assert metric_higher_is_better("mse") is False


def test_accuracy_winner_and_reason_are_explicit() -> None:
    winner, reason = select_comparison_winner("accuracy", pc_value=0.70, mlp_value=0.80)
    assert winner == "mlp"
    assert reason == "higher_is_better: mlp_primary_metric_value > pc_primary_metric_value"

    winner, reason = select_comparison_winner("accuracy", pc_value=0.90, mlp_value=0.80)
    assert winner == "pc"
    assert reason == "higher_is_better: pc_primary_metric_value > mlp_primary_metric_value"


def test_mse_winner_and_reason_are_explicit() -> None:
    winner, reason = select_comparison_winner("mse", pc_value=0.20, mlp_value=0.10)
    assert winner == "mlp"
    assert reason == "lower_is_better: mlp_primary_metric_value < pc_primary_metric_value"

    winner, reason = select_comparison_winner("mse", pc_value=0.10, mlp_value=0.20)
    assert winner == "pc"
    assert reason == "lower_is_better: pc_primary_metric_value < mlp_primary_metric_value"


def test_metric_winner_tie_uses_tolerance() -> None:
    winner, reason = select_comparison_winner("mse", pc_value=0.1, mlp_value=0.1 + 1.0e-13)
    assert winner == "tie"
    assert reason == "tie: primary metric values are equal within tolerance"
