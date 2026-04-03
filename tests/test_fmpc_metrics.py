from __future__ import annotations

import numpy as np
import pytest

from pc.metrics import (
    energy_gap_to_teacher,
    hidden_state_l2_distance,
    hidden_state_rms_gap,
    state_update_direction_cosine,
    summarize_teacher_reference_metrics,
    update_direction_cosine,
)


def test_hidden_state_gap_helpers_match_manual_values() -> None:
    candidate = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    teacher = np.array([[0.0, 2.0], [1.0, 1.0]], dtype=np.float64)

    diff = candidate - teacher
    expected_l2 = float(np.linalg.norm(diff))
    expected_rms = float(np.sqrt(np.mean(diff**2)))

    assert np.isclose(hidden_state_l2_distance(candidate, teacher), expected_l2)
    assert np.isclose(hidden_state_rms_gap(candidate, teacher), expected_rms)


def test_energy_gap_to_teacher_is_candidate_minus_teacher() -> None:
    assert np.isclose(energy_gap_to_teacher(1.25, 0.75), 0.5)


def test_update_direction_cosine_matches_simple_geometry() -> None:
    candidate_direction = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    teacher_direction = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64)

    cosine = update_direction_cosine(candidate_direction, teacher_direction)
    assert cosine is not None
    assert np.isclose(cosine, 1.0)


def test_state_update_direction_cosine_returns_none_for_zero_norm_direction() -> None:
    z0 = np.array([[1.0, 2.0]], dtype=np.float64)
    z_terminal = z0.copy()
    teacher_z0 = np.array([[1.0, 2.0]], dtype=np.float64)
    teacher_z_terminal = np.array([[2.0, 3.0]], dtype=np.float64)

    assert state_update_direction_cosine(z0, z_terminal, teacher_z0, teacher_z_terminal) is None


def test_teacher_reference_metric_summary_serializes_expected_fields() -> None:
    candidate_z0 = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    candidate_z_terminal = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
    teacher_z0 = candidate_z0.copy()
    teacher_z_terminal = candidate_z_terminal.copy()

    summary = summarize_teacher_reference_metrics(
        candidate_z0=candidate_z0,
        candidate_z_terminal=candidate_z_terminal,
        candidate_final_energy=0.25,
        teacher_z0=teacher_z0,
        teacher_z_terminal=teacher_z_terminal,
        teacher_final_energy=0.25,
    )

    assert set(summary) == {
        "terminal_state_l2_gap",
        "terminal_state_rms_gap",
        "candidate_final_energy",
        "teacher_final_energy",
        "energy_gap_to_teacher",
        "update_direction_cosine",
    }
    assert np.isclose(summary["terminal_state_l2_gap"], 0.0)
    assert np.isclose(summary["terminal_state_rms_gap"], 0.0)
    assert np.isclose(summary["energy_gap_to_teacher"], 0.0)
    assert np.isclose(summary["update_direction_cosine"], 1.0)


def test_hidden_state_gap_helpers_validate_shape_mismatch() -> None:
    candidate = np.zeros((2, 3), dtype=np.float64)
    teacher = np.zeros((2, 4), dtype=np.float64)

    with pytest.raises(ValueError):
        hidden_state_l2_distance(candidate, teacher)
