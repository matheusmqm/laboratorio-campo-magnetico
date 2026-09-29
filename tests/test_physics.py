"""Testes de physics.py."""
import math

import numpy as np
import pytest

import physics as ph

E = ph.ELEMENTARY_CHARGE


def test_force_is_q_v_cross_b():
    # Próton com v em +x e B em +z: F = e v B (x × z) = −e v B ŷ
    force = ph.magnetic_force(E, [1e6, 0, 0], [0, 0, 1e-3])
    assert np.allclose(force, [0, -E * 1e6 * 1e-3, 0])


def test_negative_charge_inverts_force():
    v, b = np.array([1e6, 2e5, 0]), np.array([0, 1e-3, 5e-4])
    assert np.allclose(ph.magnetic_force(-E, v, b), -ph.magnetic_force(E, v, b))


def test_vector_and_scalar_forms_agree():
    v = ph.spherical_to_cartesian(1e6, 60, 30)
    b = ph.spherical_to_cartesian(2e-3, 10, 200)
    phi = ph.angle_between_deg(v, b)
    f_cross = np.linalg.norm(ph.magnetic_force(E, v, b))
    assert f_cross == pytest.approx(ph.force_magnitude(E, 1e6, 2e-3, phi), rel=1e-12)


def test_force_perpendicular_to_v_and_b():
    v, b = np.array([3e5, -1e5, 2e5]), np.array([1e-3, 2e-3, -5e-4])
    force = ph.magnetic_force(E, v, b)
    assert ph.angle_between_deg(force, v) == pytest.approx(90.0)
    assert ph.angle_between_deg(force, b) == pytest.approx(90.0)
    scale = np.linalg.norm(force) * np.linalg.norm(v)
    assert abs(ph.magnetic_power(force, v)) <= 1e-12 * scale


def test_field_from_force_recovers_b():
    b_true, v, phi = 0.25, 1e5, 37.0
    f = ph.force_magnitude(E, v, b_true, phi)
    assert ph.field_from_force(f, E, v, phi) == pytest.approx(b_true)


@pytest.mark.parametrize("charge, v_mag, phi", [(0.0, 1e6, 90), (E, 0.0, 90), (E, 1e6, 0), (E, 1e6, 180)])
def test_field_from_force_rejects_undefined_cases(charge, v_mag, phi):
    with pytest.raises(ValueError):
        ph.field_from_force(1e-16, charge, v_mag, phi)


def test_angle_precise_near_parallel():
    assert ph.angle_between_deg([1, 0, 0], [1, 1e-9, 0]) == pytest.approx(math.degrees(1e-9), rel=1e-6)
    assert ph.angle_between_deg([1, 0, 0], [-2, 0, 0]) == pytest.approx(180.0)
    assert math.isnan(ph.angle_between_deg([0, 0, 0], [1, 0, 0]))


def test_spherical_roundtrip():
    vec = ph.spherical_to_cartesian(2.5, 123.0, 311.0)
    assert ph.cartesian_to_spherical(vec) == pytest.approx((2.5, 123.0, 311.0))


def test_force_zero_reasons():
    assert ph.force_zero_reasons(E, [1, 0, 0], [0, 1, 0]) == []
    assert len(ph.force_zero_reasons(0.0, [1, 0, 0], [0, 1, 0])) == 1
    # θ = 180° gera sen(π) ≈ 1e-16, que ainda deve contar como paralelo
    v = ph.spherical_to_cartesian(1e6, 180.0, 0.0)
    assert any("∥" in r for r in ph.force_zero_reasons(E, v, [0, 0, 1e-3]))


def test_unit_conversion():
    assert ph.to_tesla(5.0, "G") == pytest.approx(5e-4)
    assert ph.from_tesla(1e-3, "G") == pytest.approx(10.0)
    with pytest.raises(ValueError):
        ph.to_tesla(1.0, "mT")


@pytest.mark.parametrize("text, expected", [("1,5", 1.5), (" 3e-4 ", 3e-4), ("2×10^-3", 2e-3), ("-7x10^2", -700.0)])
def test_parse_number_accepts(text, expected):
    assert ph.parse_number(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", "abc", "inf", "nan", "1e400"])
def test_parse_number_rejects(text):
    with pytest.raises(ValueError):
        ph.parse_number(text)
