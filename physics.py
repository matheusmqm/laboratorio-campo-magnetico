"""Física da força magnética sobre uma carga em movimento (Halliday, seção 28-1).

Todas as grandezas são tratadas em unidades do SI. As conversões gauss <-> tesla
só devem ser usadas na entrada e na exibição de dados.

Este módulo NÃO importa Streamlit nem Plotly.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- #
# Constantes (CODATA 2018)
# --------------------------------------------------------------------------- #
ELEMENTARY_CHARGE = 1.602176634e-19  # C (exata)

GAUSS_PER_TESLA = 1.0e4
FIELD_UNITS = ("T", "G")

# Dois vetores são considerados paralelos quando |v×B| <= tol·|v||B|.
PARALLEL_TOLERANCE = 1.0e-9

Vector = np.ndarray


# --------------------------------------------------------------------------- #
# Partículas e campos típicos
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Particle:
    """Partícula de prova: nome e carga (C)."""

    name: str
    charge: float


PARTICLES: dict[str, Particle] = {
    p.name: p
    for p in (
        Particle("Elétron", -ELEMENTARY_CHARGE),
        Particle("Próton", +ELEMENTARY_CHARGE),
        Particle("Partícula alfa", +2.0 * ELEMENTARY_CHARGE),
        Particle("Nêutron", 0.0),
    )
}


@dataclass(frozen=True)
class FieldReference:
    """Ordem de grandeza de um campo magnético real."""

    name: str
    value_tesla: float
    note: str


# Tabela 28-1 do Halliday: ordem de grandeza de alguns campos magnéticos
TYPICAL_FIELDS: tuple[FieldReference, ...] = (
    FieldReference("Na superfície de uma estrela de nêutrons", 1.0e8, "10⁸ T"),
    FieldReference("Perto de um grande eletroímã", 1.5, "1,5 T"),
    FieldReference("Perto de um ímã pequeno", 1.0e-2, "10⁻² T"),
    FieldReference("Na superfície da Terra", 1.0e-4, "10⁻⁴ T (100 μT ou 1 G)"),
    FieldReference("No espaço sideral", 1.0e-10, "10⁻¹⁰ T"),
    FieldReference("Em uma sala magneticamente blindada", 1.0e-14, "10⁻¹⁴ T"),
)


# --------------------------------------------------------------------------- #
# Vetores e geometria
# --------------------------------------------------------------------------- #
def unit_vector(vec: Vector) -> Vector:
    """Retorna o vetor normalizado (ou o vetor nulo, se |vec| = 0)."""
    vec = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0.0 else np.zeros(3)


def spherical_to_cartesian(magnitude: float, theta_deg: float, azimuth_deg: float) -> Vector:
    """Converte (módulo, θ polar a partir de +z, azimute a partir de +x) em (x, y, z)."""
    theta, azimuth = math.radians(theta_deg), math.radians(azimuth_deg)
    return magnitude * np.array(
        [
            math.sin(theta) * math.cos(azimuth),
            math.sin(theta) * math.sin(azimuth),
            math.cos(theta),
        ]
    )


def cartesian_to_spherical(vec: Vector) -> tuple[float, float, float]:
    """Converte (x, y, z) em (módulo, θ em graus, azimute em graus [0, 360))."""
    vec = np.asarray(vec, dtype=float)
    magnitude = float(np.linalg.norm(vec))
    if magnitude == 0.0:
        return 0.0, 0.0, 0.0
    theta = math.degrees(math.acos(max(-1.0, min(1.0, vec[2] / magnitude))))
    azimuth = math.degrees(math.atan2(vec[1], vec[0])) % 360.0
    return magnitude, theta, azimuth


def angle_between_deg(a: Vector, b: Vector) -> float:
    """Ângulo (graus, 0 a 180) entre dois vetores; NaN se algum for nulo.

    Usa atan2(|a×b|, a·b), que é preciso também perto de 0° e 180°.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if np.linalg.norm(a) == 0.0 or np.linalg.norm(b) == 0.0:
        return float("nan")
    return math.degrees(math.atan2(float(np.linalg.norm(np.cross(a, b))), float(np.dot(a, b))))


def is_parallel(a: Vector, b: Vector, tolerance: float = PARALLEL_TOLERANCE) -> bool:
    """True se a e b são paralelos ou antiparalelos (ambos não nulos)."""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return False
    return float(np.linalg.norm(np.cross(a, b))) <= tolerance * na * nb


# --------------------------------------------------------------------------- #
# Força magnética
# --------------------------------------------------------------------------- #
def magnetic_force(charge: float, v: Vector, b_field: Vector) -> Vector:
    """Força magnética F = q v × B, em newtons."""
    return charge * np.cross(np.asarray(v, dtype=float), np.asarray(b_field, dtype=float))


def force_magnitude(charge: float, v_mag: float, b_mag: float, phi_deg: float) -> float:
    """Módulo da força: F = |q| v B sen φ."""
    return abs(charge) * v_mag * b_mag * math.sin(math.radians(phi_deg))


def magnetic_power(force: Vector, v: Vector) -> float:
    """Potência da força magnética, P = F·v (watts). É sempre nula: F ⊥ v."""
    return float(np.dot(np.asarray(force, dtype=float), np.asarray(v, dtype=float)))


def field_from_force(force_mag: float, charge: float, v_mag: float, phi_deg: float) -> float:
    """Definição operacional de B: B = F / (|q| v sen φ).

    Raises:
        ValueError: se q = 0, v = 0 ou sen φ = 0 (a definição não se aplica).
    """
    denominator = abs(charge) * v_mag * math.sin(math.radians(phi_deg))
    if denominator <= 0.0:
        raise ValueError("B não pode ser medido com q = 0, v = 0 ou v paralelo a B (sen φ = 0).")
    return force_mag / denominator


def force_zero_reasons(charge: float, v: Vector, b_field: Vector) -> list[str]:
    """Lista os motivos (em texto) pelos quais F = 0; lista vazia se F ≠ 0."""
    reasons: list[str] = []
    if charge == 0.0:
        reasons.append("q = 0: uma partícula neutra não sofre força magnética (ex.: nêutron).")
    if float(np.linalg.norm(v)) == 0.0:
        reasons.append("v = 0: a força magnética só age sobre cargas em movimento.")
    if float(np.linalg.norm(b_field)) == 0.0:
        reasons.append("B = 0: não há campo magnético, logo não há força magnética.")
    if is_parallel(v, b_field):
        reasons.append("v ∥ B (φ = 0° ou 180°): sen φ = 0, então v × B = 0.")
    return reasons


# --------------------------------------------------------------------------- #
# Unidades
# --------------------------------------------------------------------------- #
def to_tesla(value: float, unit: str) -> float:
    """Converte um campo de 'T' ou 'G' para tesla."""
    if unit == "T":
        return value
    if unit == "G":
        return value / GAUSS_PER_TESLA
    raise ValueError(f"Unidade desconhecida: {unit!r} (use 'T' ou 'G').")


def from_tesla(value_tesla: float, unit: str) -> float:
    """Converte um campo em tesla para 'T' ou 'G'."""
    if unit == "T":
        return value_tesla
    if unit == "G":
        return value_tesla * GAUSS_PER_TESLA
    raise ValueError(f"Unidade desconhecida: {unit!r} (use 'T' ou 'G').")


# --------------------------------------------------------------------------- #
# Validação de entradas de texto
# --------------------------------------------------------------------------- #
_TIMES_TEN = re.compile(r"[x×*]10\^")


def parse_number(text: str) -> float:
    """Converte texto em número finito; aceita vírgula decimal e '1e-3' ou '1×10^-3'.

    Raises:
        ValueError: com mensagem em português se o texto não for um número finito.
    """
    cleaned = str(text).strip().replace(" ", "").replace(",", ".")
    if not cleaned:
        raise ValueError("campo vazio; digite um número (ex.: 1.5 ou 3e-4).")
    cleaned = _TIMES_TEN.sub("e", cleaned)
    try:
        value = float(cleaned)
    except ValueError:
        raise ValueError(f"'{text}' não é um número válido (ex.: 1.5 ou 3e-4).") from None
    if not math.isfinite(value):
        raise ValueError("o valor precisa ser finito.")
    return value
