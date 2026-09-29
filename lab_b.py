"""Lógica do módulo "Descobrindo B": campo oculto, lançamentos e avaliação da estimativa.

Não importa Streamlit nem Plotly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import physics as ph

FIELD_MIN_TESLA = 1.0e-3  # limites do sorteio (módulo log-uniforme)
FIELD_MAX_TESLA = 1.0
SUCCESS_MAGNITUDE_ERROR_PCT = 5.0
SUCCESS_ANGLE_ERROR_DEG = 5.0


@dataclass(frozen=True)
class Attempt:
    """Um lançamento da carga de prova e a força medida (B não é guardado aqui)."""

    index: int
    charge: float
    speed: float
    theta_deg: float
    azimuth_deg: float
    direction: np.ndarray  # vetor unitário de lançamento
    force: np.ndarray  # N
    force_magnitude: float  # N


@dataclass(frozen=True)
class Evaluation:
    """Comparação entre a estimativa do usuário e o campo verdadeiro."""

    true_magnitude: float  # T
    true_theta_deg: float
    true_azimuth_deg: float
    magnitude_error_pct: float
    angle_error_deg: float  # entre B_verdadeiro e a direção estimada (0 a 180)
    axis_error_deg: float  # ignorando o sentido (0 a 90)

    @property
    def is_success(self) -> bool:
        """Estimativa considerada boa (erro de módulo e de ângulo pequenos)."""
        return (
            self.magnitude_error_pct <= SUCCESS_MAGNITUDE_ERROR_PCT
            and self.angle_error_deg <= SUCCESS_ANGLE_ERROR_DEG
        )

    @property
    def right_axis_wrong_sense(self) -> bool:
        """Acertou a reta de B, mas com o sentido invertido."""
        return self.axis_error_deg <= SUCCESS_ANGLE_ERROR_DEG < self.angle_error_deg


def random_field(rng: np.random.Generator) -> np.ndarray:
    """Sorteia B (tesla): direção uniforme na esfera, módulo log-uniforme."""
    magnitude = 10.0 ** rng.uniform(math.log10(FIELD_MIN_TESLA), math.log10(FIELD_MAX_TESLA))
    cos_theta = rng.uniform(-1.0, 1.0)
    azimuth = rng.uniform(0.0, 360.0)
    return ph.spherical_to_cartesian(magnitude, math.degrees(math.acos(cos_theta)), azimuth)


def manual_field(magnitude_tesla: float, theta_deg: float, azimuth_deg: float) -> np.ndarray:
    """Campo definido pelo professor (módulo em tesla, ângulos em graus).

    Raises:
        ValueError: se o módulo não for positivo.
    """
    if magnitude_tesla <= 0.0:
        raise ValueError("O módulo do campo oculto deve ser maior que zero.")
    return ph.spherical_to_cartesian(magnitude_tesla, theta_deg, azimuth_deg)


def launch_probe(
    b_hidden: np.ndarray,
    charge: float,
    speed: float,
    theta_deg: float,
    azimuth_deg: float,
    index: int,
) -> Attempt:
    """Lança a carga de prova e devolve a força medida F = q v × B.

    Raises:
        ValueError: se a velocidade for negativa.
    """
    if speed < 0.0:
        raise ValueError("A velocidade deve ser maior ou igual a zero.")
    direction = ph.spherical_to_cartesian(1.0, theta_deg, azimuth_deg)
    force = ph.magnetic_force(charge, speed * direction, b_hidden)
    return Attempt(
        index=index,
        charge=charge,
        speed=speed,
        theta_deg=theta_deg,
        azimuth_deg=azimuth_deg,
        direction=direction,
        force=force,
        force_magnitude=float(np.linalg.norm(force)),
    )


def attempts_to_rows(attempts: list[Attempt]) -> list[dict[str, str | float | int]]:
    """Converte as tentativas em linhas de tabela (só o que o usuário mede)."""
    return [
        {
            "#": a.index,
            "θ lançamento (°)": a.theta_deg,
            "azimute (°)": a.azimuth_deg,
            "q (C)": f"{a.charge:.4e}",
            "v (m/s)": f"{a.speed:.3e}",
            "|F| (N)": f"{a.force_magnitude:.4e}",
            "Fx (N)": f"{a.force[0]:.3e}",
            "Fy (N)": f"{a.force[1]:.3e}",
            "Fz (N)": f"{a.force[2]:.3e}",
        }
        for a in attempts
    ]


def evaluate_estimate(
    b_true: np.ndarray, magnitude_tesla: float, theta_deg: float, azimuth_deg: float
) -> Evaluation:
    """Compara a estimativa (módulo em T e direção por ângulos) com o campo real."""
    true_mag, true_theta, true_az = ph.cartesian_to_spherical(b_true)
    estimate_dir = ph.spherical_to_cartesian(1.0, theta_deg, azimuth_deg)
    angle = ph.angle_between_deg(b_true, estimate_dir)
    return Evaluation(
        true_magnitude=true_mag,
        true_theta_deg=true_theta,
        true_azimuth_deg=true_az,
        magnitude_error_pct=abs(magnitude_tesla - true_mag) / true_mag * 100.0,
        angle_error_deg=angle,
        axis_error_deg=min(angle, 180.0 - angle),
    )
