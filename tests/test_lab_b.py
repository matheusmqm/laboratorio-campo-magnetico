"""Testes de lab_b.py."""
import numpy as np
import pytest

import lab_b
import physics as ph

E = ph.ELEMENTARY_CHARGE


def test_random_field_within_limits():
    rng = np.random.default_rng(0)
    for _ in range(200):
        mag = np.linalg.norm(lab_b.random_field(rng))
        assert lab_b.FIELD_MIN_TESLA <= mag <= lab_b.FIELD_MAX_TESLA


def test_manual_field_rejects_non_positive():
    with pytest.raises(ValueError):
        lab_b.manual_field(0.0, 90, 0)


def test_launch_along_b_gives_zero_force():
    b = lab_b.manual_field(0.1, 30, 45)
    attempt = lab_b.launch_probe(b, E, 1e6, 30, 45, 1)
    assert attempt.force_magnitude == pytest.approx(0.0, abs=1e-25)


def test_launch_perpendicular_gives_max_force():
    b = lab_b.manual_field(0.1, 0, 0)  # B em +z
    attempt = lab_b.launch_probe(b, E, 1e6, 90, 0, 1)  # v em +x
    assert attempt.force_magnitude == pytest.approx(E * 1e6 * 0.1)
    assert ph.field_from_force(attempt.force_magnitude, E, 1e6, 90) == pytest.approx(0.1)


def test_launch_rejects_negative_speed():
    with pytest.raises(ValueError):
        lab_b.launch_probe(np.array([0, 0, 1.0]), E, -1.0, 0, 0, 1)


def test_evaluate_exact_estimate_is_success():
    b = lab_b.manual_field(0.02, 70, 120)
    ev = lab_b.evaluate_estimate(b, 0.02, 70, 120)
    assert ev.is_success and not ev.right_axis_wrong_sense


def test_evaluate_opposite_sense():
    b = lab_b.manual_field(0.02, 70, 120)
    ev = lab_b.evaluate_estimate(b, 0.02, 110, 300)  # −B
    assert not ev.is_success
    assert ev.right_axis_wrong_sense
    assert ev.angle_error_deg == pytest.approx(180.0)
