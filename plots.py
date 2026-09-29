"""Todas as figuras Plotly do laboratório. Recebe dados e devolve `go.Figure`.

Não importa Streamlit. Fundos transparentes, para funcionar nos temas claro e escuro.
"""
from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

import physics as ph

# Cores legíveis nos dois temas
COLOR_V = "#1f77b4"
COLOR_B = "#2ca02c"
COLOR_F = "#d62728"
COLOR_ARC = "#ff7f0e"
COLOR_NEUTRAL = "#8c8c8c"
GRID_COLOR = "rgba(128,128,128,0.30)"
TRANSPARENT = "rgba(0,0,0,0)"

# Parâmetros gráficos nomeados
ARROW_TIP_SIZE = 0.35  # tamanho do cone (vetor de comprimento 1)
ARC_RADIUS = 0.45
ARC_POINTS = 40
SCENE_HALF_RANGE = 1.3
ANGLE_CURVE_POINTS = 361


def _base_layout(fig: go.Figure, height: int, title: str | None = None) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        paper_bgcolor=TRANSPARENT,
        plot_bgcolor=TRANSPARENT,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0.0),
    )
    return fig


def _scene(half_range: float | tuple[float, float, float], titles: tuple[str, str, str] = ("x", "y", "z"),
           centers: tuple[float, float, float] = (0.0, 0.0, 0.0), exponent: bool = False) -> dict:
    halves = (half_range,) * 3 if isinstance(half_range, (int, float)) else half_range
    axes = {}
    for name, title, half, center in zip(("xaxis", "yaxis", "zaxis"), titles, halves, centers):
        axes[name] = dict(
            title=title,
            range=[center - half, center + half],
            backgroundcolor=TRANSPARENT,
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            showbackground=False,
            exponentformat="e" if exponent else "none",
        )
    return dict(aspectmode="cube", **axes)


def _colorscale(color: str) -> list[list]:
    return [[0.0, color], [1.0, color]]


def _add_arrow3d(fig: go.Figure, direction: np.ndarray, color: str, name: str) -> None:
    """Haste (Scatter3d) + ponta (Cone) partindo da origem, comprimento unitário."""
    tip = direction
    fig.add_trace(go.Scatter3d(
        x=[0, tip[0]], y=[0, tip[1]], z=[0, tip[2]], mode="lines",
        line=dict(color=color, width=7), name=name, legendgroup=name,
        hovertemplate=f"{name}<extra></extra>",
    ))
    fig.add_trace(go.Cone(
        x=[tip[0]], y=[tip[1]], z=[tip[2]], u=[direction[0]], v=[direction[1]], w=[direction[2]],
        anchor="tip", sizemode="absolute", sizeref=ARROW_TIP_SIZE,
        colorscale=_colorscale(color), showscale=False, showlegend=False,
        legendgroup=name, name=name, hoverinfo="name",
    ))


# --------------------------------------------------------------------------- #
# Aba B: vetores v, B e F em 3D
# --------------------------------------------------------------------------- #
def vectors_figure(v: np.ndarray, b_field: np.ndarray, force: np.ndarray, charge: float,
                   force_is_zero: bool) -> go.Figure:
    """Vetores v, B e F (comprimentos normalizados), com o arco do ângulo φ."""
    fig = go.Figure()
    v_hat, b_hat = ph.unit_vector(v), ph.unit_vector(b_field)
    if np.any(v_hat):
        _add_arrow3d(fig, v_hat, COLOR_V, "v (velocidade)")
    if np.any(b_hat):
        _add_arrow3d(fig, b_hat, COLOR_B, "B (campo magnético)")
    sign = "+" if charge > 0 else "−"
    if force_is_zero:
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", name="F = 0",
                                   marker=dict(symbol="x", size=6, color=COLOR_F)))
    else:
        _add_arrow3d(fig, ph.unit_vector(force), COLOR_F, f"F (q {sign})")

    phi = ph.angle_between_deg(v_hat, b_hat)
    if np.any(v_hat) and np.any(b_hat) and math.isfinite(phi) and not ph.is_parallel(v_hat, b_hat):
        phi_rad = math.radians(phi)
        s = np.linspace(0.0, 1.0, ARC_POINTS)[:, None]
        arc = ARC_RADIUS * (np.sin((1 - s) * phi_rad) * v_hat + np.sin(s * phi_rad) * b_hat) / math.sin(phi_rad)
        fig.add_trace(go.Scatter3d(x=arc[:, 0], y=arc[:, 1], z=arc[:, 2], mode="lines",
                                   line=dict(color=COLOR_ARC, width=5), name=f"φ = {phi:.1f}°"))
        mid = 1.35 * arc[ARC_POINTS // 2]
        fig.add_trace(go.Scatter3d(x=[mid[0]], y=[mid[1]], z=[mid[2]], mode="text", text=["φ"],
                                   textfont=dict(color=COLOR_ARC, size=16), showlegend=False,
                                   hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", marker=dict(size=3, color=COLOR_NEUTRAL),
                               showlegend=False, hoverinfo="skip"))
    _base_layout(fig, 560)
    fig.update_layout(scene=_scene(SCENE_HALF_RANGE), scene_camera=dict(eye=dict(x=1.5, y=1.4, z=1.0)))
    return fig


# --------------------------------------------------------------------------- #
# Aba C: |F| × φ
# --------------------------------------------------------------------------- #
def force_vs_angle_figure(charge: float, v_mag: float, b_mag: float, phi_marker_deg: float,
                          phi_current_deg: float | None = None) -> go.Figure:
    """Curva |F| = |q| v B sen φ (0° a 180°) com marcador móvel e pontos notáveis."""
    phi = np.linspace(0.0, 180.0, ANGLE_CURVE_POINTS)
    f_max = abs(charge) * v_mag * b_mag
    force = f_max * np.sin(np.radians(phi))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=phi, y=force, mode="lines", line=dict(color=COLOR_V, width=3),
                             name="|F| = |q| v B sen φ", hovertemplate="φ = %{x:.1f}°<br>|F| = %{y:.3e} N<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 180], y=[0, 0], mode="markers+text", text=["F = 0", "F = 0"],
                             textposition="top center", marker=dict(size=12, color=COLOR_F, symbol="x"),
                             name="F = 0 (v ∥ B)"))
    fig.add_trace(go.Scatter(x=[90], y=[f_max], mode="markers+text", text=["F máx"], textposition="top center",
                             marker=dict(size=14, color=COLOR_B, symbol="star"), name="F máxima (φ = 90°)"))
    marker_f = f_max * math.sin(math.radians(phi_marker_deg))
    fig.add_trace(go.Scatter(x=[phi_marker_deg], y=[marker_f], mode="markers",
                             marker=dict(size=16, color=COLOR_ARC, line=dict(width=2, color="white")),
                             name=f"φ do controle = {phi_marker_deg:.1f}°"))
    if phi_current_deg is not None and math.isfinite(phi_current_deg):
        current_f = f_max * math.sin(math.radians(phi_current_deg))
        fig.add_trace(go.Scatter(x=[phi_current_deg], y=[current_f], mode="markers",
                                 marker=dict(size=11, color=COLOR_NEUTRAL, symbol="diamond"),
                                 name=f"φ atual (sidebar) = {phi_current_deg:.1f}°"))
    _base_layout(fig, 480)
    fig.update_xaxes(title="φ (graus)", tickvals=list(range(0, 181, 30)), range=[-5, 185], gridcolor=GRID_COLOR)
    fig.update_yaxes(title="|F| (N)", gridcolor=GRID_COLOR, exponentformat="e",
                     range=[-0.05 * f_max, 1.25 * f_max] if f_max > 0 else [-0.05, 1.0])
    return fig


# --------------------------------------------------------------------------- #
# Aba D: histórico de lançamentos
# --------------------------------------------------------------------------- #
def probe_history_figure(directions: np.ndarray, forces: np.ndarray) -> go.Figure:
    """Direções de lançamento na esfera unitária, coloridas por |F|."""
    u, w = np.meshgrid(np.linspace(0, 2 * math.pi, 40), np.linspace(0, math.pi, 20))
    fig = go.Figure(go.Surface(
        x=np.cos(u) * np.sin(w), y=np.sin(u) * np.sin(w), z=np.cos(w), opacity=0.12, showscale=False,
        colorscale=_colorscale(COLOR_NEUTRAL), hoverinfo="skip", name="esfera de direções"))
    axes_len = 1.35
    for i, (label, color) in enumerate((("x", "#888"), ("y", "#888"), ("z", "#888"))):
        end = np.zeros(3)
        end[i] = axes_len
        fig.add_trace(go.Scatter3d(x=[0, end[0]], y=[0, end[1]], z=[0, end[2]], mode="lines+text",
                                   text=["", label], line=dict(color=color, width=2), showlegend=False,
                                   hoverinfo="skip"))
    if len(forces):
        directions, forces = np.asarray(directions), np.asarray(forces)
        fig.add_trace(go.Scatter3d(
            x=directions[:, 0], y=directions[:, 1], z=directions[:, 2], mode="markers+text",
            text=[str(i + 1) for i in range(len(forces))], textposition="top center",
            marker=dict(size=7, color=forces, colorscale="Viridis", cmin=0.0,
                        colorbar=dict(title="|F| (N)", exponentformat="e", len=0.7),
                        line=dict(width=1, color="white")),
            customdata=forces, name="lançamentos",
            hovertemplate="tentativa %{text}<br>|F| = %{customdata:.4e} N<extra></extra>"))
    _base_layout(fig, 540)
    fig.update_layout(scene=_scene(1.4), showlegend=False)
    return fig

