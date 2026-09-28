"""Laboratório Virtual de Campo Magnético: medindo B a partir da força magnética.

Interface Streamlit (sidebar + abas A a F). Execução: `streamlit run app.py`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import streamlit as st

import lab_b
import physics as ph
import plots

TITLE = "Laboratório Virtual de Campo Magnético: medindo B a partir da força magnética"
CUSTOM_PARTICLE = "Personalizada"
MODE_POLAR = "Módulo + ângulos"
MODE_CART = "Componentes"

# Valores padrão (o caso inicial é um próton com φ = 60°)
DEFAULT_V_MAG = "1e6"
DEFAULT_V_THETA = 60.0
DEFAULT_V_COMPONENTS = ("8.66e5", "0", "5e5")
DEFAULT_B_TESLA = 1.0e-3
DEFAULT_PHI_MARKER = 60.0
DEFAULT_PROBE_SPEED = "1e6"
DEFAULT_PARTICLE = "Próton"


# --------------------------------------------------------------------------- #
# Utilitários de interface
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Params:
    """Parâmetros globais (SI) lidos da sidebar."""

    particle: str
    q: float
    v: np.ndarray
    b: np.ndarray
    unit: str

    @property
    def v_mag(self) -> float:
        return float(np.linalg.norm(self.v))

    @property
    def b_mag(self) -> float:
        return float(np.linalg.norm(self.b))

    @property
    def phi(self) -> float:
        return ph.angle_between_deg(self.v, self.b)

    @property
    def force(self) -> np.ndarray:
        return ph.magnetic_force(self.q, self.v, self.b)

    @property
    def zero_reasons(self) -> list[str]:
        return ph.force_zero_reasons(self.q, self.v, self.b)


def init_state(key: str, default: object) -> None:
    """Cria a chave em st.session_state apenas se ainda não existir."""
    if key not in st.session_state:
        st.session_state[key] = default


def show_figure(fig, key: str) -> None:
    """Exibe uma figura Plotly ocupando toda a largura."""
    st.plotly_chart(fig, width="stretch", key=key)


def sci(value: float, unit: str, digits: int = 4) -> str:
    """Formata em notação científica com unidade, ex.: '1.602e-16 N'."""
    if not math.isfinite(value):
        return "—"
    return f"0 {unit}" if value == 0 else f"{value:.{digits}e} {unit}"


def fmt_pct(value: float) -> str:
    """Formata erro percentual ('—' se não definido)."""
    return "—" if not math.isfinite(value) else f"{value:.3g} %"


def read_number(label: str, key: str, default: str, *, strictly_positive: bool = False,
                minimum: float | None = None, help_text: str | None = None) -> float:
    """Campo de texto numérico validado; em caso de erro, mostra a mensagem e usa o último valor válido."""
    init_state(key, default)
    text = st.text_input(label, key=key, help=help_text)
    try:
        value = ph.parse_number(text)
        if strictly_positive and value <= 0.0:
            raise ValueError("o valor deve ser maior que zero.")
        if minimum is not None and value < minimum:
            raise ValueError(f"o valor deve ser maior ou igual a {minimum:g}.")
    except ValueError as err:
        st.error(f"**{label}**: {err}")
        return float(st.session_state.get(f"_ok_{key}", ph.parse_number(default)))
    st.session_state[f"_ok_{key}"] = value
    return value


FIELD_TEXT_KEYS = ("B_mag", "B_x", "B_y", "B_z", "est_mag")  # campos de texto expressos na unidade do campo


def convert_field_texts() -> None:
    """Callback da troca T ↔ G: converte os textos de campo já digitados, preservando o valor físico."""
    old = st.session_state.get("_unit_prev", "T")
    new = st.session_state["unit"]
    for key in FIELD_TEXT_KEYS:
        for state_key in (key, f"_ok_{key}"):
            if state_key not in st.session_state:
                continue
            raw = st.session_state[state_key]
            try:
                value = ph.parse_number(raw) if isinstance(raw, str) else float(raw)
            except ValueError:
                continue  # texto inválido: mantém como está
            converted = ph.from_tesla(ph.to_tesla(value, old), new)
            st.session_state[state_key] = f"{converted:.6g}" if isinstance(raw, str) else converted
    st.session_state["_unit_prev"] = new


def apply_field_preset(value_tesla: float) -> None:
    """Callback: aplica um campo típico ao B da sidebar (módulo na unidade escolhida)."""
    unit = st.session_state.get("unit", "T")
    st.session_state["vec_mode"] = MODE_POLAR
    st.session_state["B_mag"] = f"{ph.from_tesla(value_tesla, unit):.6g}"


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def _vector_input(tag: str, title: str, unit: str, scale_to_si: float, mode: str, mag_default: str,
                  comp_defaults: tuple[str, str, str], theta_default: float) -> np.ndarray:
    """Lê um vetor (v ou B) por módulo + ângulos ou por componentes; devolve em SI."""
    st.markdown(f"**{title}**")
    if mode == MODE_POLAR:
        magnitude = read_number(f"|{tag}| ({unit})", f"{tag}_mag", mag_default, minimum=0.0)
        init_state(f"{tag}_theta", theta_default)
        init_state(f"{tag}_az", 0.0)
        theta = st.slider(f"θ de {tag}: ângulo com +z (°)", 0.0, 180.0, step=0.5, key=f"{tag}_theta")
        azimuth = st.slider(f"Azimute de {tag}: ângulo no plano xy (°)", 0.0, 360.0, step=0.5, key=f"{tag}_az")
        return ph.spherical_to_cartesian(magnitude * scale_to_si, theta, azimuth)
    columns = st.columns(3)
    components = []
    for col, axis, default in zip(columns, "xyz", comp_defaults):
        with col:
            components.append(read_number(f"{tag}{axis}", f"{tag}_{axis}", default))
    st.caption(f"componentes em {unit}")
    return np.array(components) * scale_to_si


def sidebar_parameters() -> Params:
    """Desenha a sidebar e devolve os parâmetros globais em SI."""
    with st.sidebar:
        st.header("Parâmetros globais")
        st.subheader("Partícula")
        name = st.selectbox("Preset", [*ph.PARTICLES, CUSTOM_PARTICLE],
                            index=list(ph.PARTICLES).index(DEFAULT_PARTICLE), key="particle")
        if name == CUSTOM_PARTICLE:
            q_in_e = st.number_input("Carga q (em múltiplos de e)", value=1.0, step=1.0, format="%.4g",
                                     key="custom_q_e", help="e = 1,602×10⁻¹⁹ C. Use valores negativos para carga negativa.")
            charge = q_in_e * ph.ELEMENTARY_CHARGE
        else:
            charge = ph.PARTICLES[name].charge
            st.caption(f"q = {charge:+.4e} C")
        if st.checkbox("Inverter o sinal de q", key="flip_q", help="Troca q por −q (útil para ver F inverter)."):
            charge = -charge

        st.subheader("Vetores")
        mode = st.radio("Entrada dos vetores", [MODE_POLAR, MODE_CART], horizontal=True, key="vec_mode")
        init_state("_unit_prev", "T")
        unit = st.radio("Unidade do campo B", list(ph.FIELD_UNITS), horizontal=True, key="unit",
                        format_func=lambda u: "tesla (T)" if u == "T" else "gauss (G)",
                        on_change=convert_field_texts)
        v = _vector_input("v", "Velocidade", "m/s", 1.0, mode, DEFAULT_V_MAG, DEFAULT_V_COMPONENTS,
                          DEFAULT_V_THETA)
        b_default = f"{DEFAULT_B_TESLA:g}"  # o padrão é definido em tesla; a unidade só afeta a exibição
        b = _vector_input("B", "Campo magnético", unit, ph.to_tesla(1.0, unit), mode, b_default,
                          ("0", "0", b_default), 0.0)
    return Params(name, charge, v, b, unit)


# --------------------------------------------------------------------------- #
# Aba A: calculadora vetorial
# --------------------------------------------------------------------------- #
def render_tab_a(p: Params) -> None:
    st.subheader("Calculadora vetorial da força")
    st.latex(r"\vec F = q\,\vec v \times \vec B \qquad\qquad F = |q|\,v\,B\,\sin\varphi")
    st.write("A força magnética é o produto vetorial de **v** por **B**, multiplicado pela carga. "
             "Ela é perpendicular a ambos; se q < 0, o sentido é invertido. "
             "Os vetores são editados na barra lateral (no modo *Componentes* ou *Módulo + ângulos*).")

    force = p.force
    phi = p.phi
    f_cross = float(np.linalg.norm(force))
    f_formula = abs(p.q) * p.v_mag * p.b_mag * math.sin(math.radians(phi)) if math.isfinite(phi) else 0.0

    cols = st.columns(4)
    for col, label, value in zip(cols, ("Fx", "Fy", "Fz", "|F|"), (*force, f_cross)):
        col.metric(label, sci(float(value), "N"))
    cols = st.columns(4)
    cols[0].metric("Ângulo φ entre v e B", "—" if not math.isfinite(phi) else f"{phi:.3f} °")
    cols[1].metric("|v|", sci(p.v_mag, "m/s"))
    cols[2].metric("|B|", sci(p.b_mag, "T"), help=f"= {ph.from_tesla(p.b_mag, 'G'):.4e} G")
    cols[3].metric("q", sci(p.q, "C"))

    for reason in p.zero_reasons:
        st.warning(f"F = 0. {reason}")

    st.markdown("##### Conferência automática")
    if math.isfinite(phi):
        st.latex(rf"|q\,\vec v\times\vec B| = |q|\,v\,B\,\sin\varphi \;\Rightarrow\; "
                 rf"{f_cross:.4e}\ \mathrm{{N}} \;\overset{{?}}{{=}}\; {abs(p.q):.3e}\cdot{p.v_mag:.3e}\cdot"
                 rf"{p.b_mag:.3e}\cdot\sin({phi:.3f}^\circ) = {f_formula:.4e}\ \mathrm{{N}}")
    diff = abs(f_cross - f_formula)
    rel = diff / f_cross if f_cross > 0 else 0.0
    cols = st.columns(3)
    cols[0].metric("|q v × B| (np.cross)", sci(f_cross, "N"))
    cols[1].metric("|q| v B sen φ", sci(f_formula, "N"))
    cols[2].metric("Diferença", sci(diff, "N"), delta=f"relativa {rel:.1e}", delta_color="off")

    st.markdown("##### A força magnética não altera |v| nem a energia cinética")
    power = ph.magnetic_power(force, p.v)
    cols = st.columns(2)
    cols[0].metric("Potência da força, F · v", sci(power, "W"), help="Sempre 0: F é perpendicular a v.")
    cols[1].metric("Componente de F ao longo de v", sci(power / p.v_mag if p.v_mag > 0 else 0.0, "N"))
    st.write("Como F ⊥ v, a componente de F na direção de v é nula: a força **não pode mudar a velocidade escalar** "
             "(nem a energia cinética) da partícula, só a **direção** de v.")

    st.markdown("##### Direção e sentido")
    if p.zero_reasons:
        st.info("Como F = 0, não há direção nem sentido definidos.")
    else:
        f_hat = ph.unit_vector(force)
        sense = ("**q > 0**: o sentido é o de **v × B** (regra da mão direita)." if p.q > 0
                 else "**q < 0**: o sentido é **oposto** ao de v × B (regra da mão direita invertida).")
        st.markdown(
            f"- F é **perpendicular a v e a B** (perpendicular ao plano dos dois).\n"
            f"- {sense}\n"
            f"- Direção unitária: F̂ = ({f_hat[0]:+.4f}, {f_hat[1]:+.4f}, {f_hat[2]:+.4f}).\n"
            f"- Conferência: ângulo(F, v) = {ph.angle_between_deg(force, p.v):.3f}° · "
            f"ângulo(F, B) = {ph.angle_between_deg(force, p.b):.3f}°.")


# --------------------------------------------------------------------------- #
# Aba B: vetores em 3D
# --------------------------------------------------------------------------- #
def right_hand_rule_text(p: Params) -> str:
    """Legenda textual da regra da mão direita para o caso atual."""
    if p.zero_reasons:
        return "Neste caso F = 0, então não há sentido a determinar. " + p.zero_reasons[0]
    base = ("Estenda os dedos da mão direita ao longo de **v** e feche-os em direção a **B** (pelo menor ângulo φ). "
            "O polegar aponta o sentido de **v × B**.")
    if p.q > 0:
        return base + " Como **q > 0**, **F** aponta no sentido do polegar."
    return base + " Como **q < 0**, **F** aponta no sentido **contrário** ao do polegar."


def render_tab_b(p: Params) -> None:
    st.subheader("Visualização 3D dos vetores")
    st.latex(r"\vec F = q\,\vec v\times\vec B \qquad \vec F\perp\vec v,\ \ \vec F\perp\vec B")
    force, reasons = p.force, p.zero_reasons
    cols = st.columns(3)
    cols[0].metric("|F|", sci(float(np.linalg.norm(force)), "N"))
    cols[1].metric("φ", "—" if not math.isfinite(p.phi) else f"{p.phi:.2f} °")
    cols[2].metric("Sinal de q", "positivo (+)" if p.q > 0 else ("negativo (−)" if p.q < 0 else "nulo (0)"))
    show_figure(plots.vectors_figure(p.v, p.b, force, p.q, bool(reasons)), "fig_vectors")
    st.caption("Os vetores são desenhados com comprimento unitário para mostrar só a orientação; "
               "os módulos reais estão nas métricas. Gire a cena com o mouse. "
               "Troque o sinal de q na barra lateral e veja F inverter.")
    st.info(right_hand_rule_text(p))


# --------------------------------------------------------------------------- #
# Aba C: F × φ
# --------------------------------------------------------------------------- #
def render_tab_c(p: Params) -> None:
    st.subheader("Gráfico F × φ")
    st.latex(r"F = |q|\,v\,B\,\sin\varphi \qquad F_{\max} = |q|\,v\,B\ \ (\varphi = 90^\circ)")
    init_state("phi_marker", DEFAULT_PHI_MARKER)
    phi = st.slider("Ângulo φ entre v e B (°)", 0.0, 180.0, step=0.5, key="phi_marker")
    f_max = abs(p.q) * p.v_mag * p.b_mag
    f_phi = f_max * math.sin(math.radians(phi))
    cols = st.columns(3)
    cols[0].metric("|F| em φ do controle", sci(f_phi, "N"))
    cols[1].metric("|F| máxima (φ = 90°)", sci(f_max, "N"))
    cols[2].metric("sen φ", f"{math.sin(math.radians(phi)):.4f}")
    if f_max == 0.0:
        st.warning("F é nula para qualquer φ, pois q = 0, v = 0 ou B = 0.")
    show_figure(plots.force_vs_angle_figure(p.q, p.v_mag, p.b_mag, phi, p.phi), "fig_f_phi")
    st.write("A força é nula quando **v ∥ B** (φ = 0° ou 180°) e máxima quando **v ⊥ B** (φ = 90°). "
             "Só a componente de v perpendicular a B, v sen φ, contribui. "
             "O losango cinza marca o φ real dos vetores definidos na barra lateral.")


# --------------------------------------------------------------------------- #
# Aba D: Descobrindo B
# --------------------------------------------------------------------------- #
def reset_lab() -> None:
    """Callback: sorteia novo campo oculto e limpa o histórico."""
    st.session_state["lab_field"] = lab_b.random_field(np.random.default_rng())
    st.session_state["lab_attempts"] = []
    st.session_state["lab_revealed"] = False


def render_tab_d(p: Params) -> None:
    st.subheader("Descobrindo B")
    st.latex(r"B = \frac{F}{|q|\,v\,\sin\varphi}")
    st.write("Um campo **B oculto** foi sorteado. Lance uma carga de prova em várias direções e meça só a força. "
             "Ache a direção em que **F = 0** (é a reta de B) e a direção de **|F| máxima**, onde |B| = F/(|q| v). "
             "O sentido de B sai da regra da mão direita aplicada ao sentido de F. "
             "Esta aba usa sua própria carga de prova; os vetores da barra lateral não entram aqui.")
    if "lab_field" not in st.session_state:
        reset_lab()
    b_hidden: np.ndarray = st.session_state["lab_field"]
    attempts: list[lab_b.Attempt] = st.session_state["lab_attempts"]

    top = st.columns([1, 2])
    top[0].button("Novo campo", on_click=reset_lab, type="primary")
    with top[1].expander("Modo professor: definir o campo oculto manualmente"):
        with st.form("teacher_form", clear_on_submit=True):
            mag_text = st.text_input("|B| (T)", type="password", help="Oculto na tela.")
            t_theta = st.slider("θ de B (°)", 0.0, 180.0, 90.0, step=0.5)
            t_az = st.slider("Azimute de B (°)", 0.0, 360.0, 0.0, step=0.5)
            submitted = st.form_submit_button("Definir campo oculto")
        if submitted:
            try:
                st.session_state["lab_field"] = lab_b.manual_field(ph.parse_number(mag_text), t_theta, t_az)
                st.session_state["lab_attempts"] = []
                st.session_state["lab_revealed"] = False
                st.rerun()
            except ValueError as err:
                st.error(str(err))

    st.markdown("##### 1. Carga de prova e lançamento")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        q_in_e = st.number_input("Carga de prova q (× e)", value=1.0, step=1.0, format="%.4g", key="probe_q")
    with c2:
        speed = read_number("Velocidade v (m/s)", "probe_v", DEFAULT_PROBE_SPEED, minimum=0.0)
    with c3:
        theta = st.slider("θ do lançamento (°)", 0.0, 180.0, 90.0, step=0.5, key="probe_theta")
    with c4:
        azimuth = st.slider("Azimute do lançamento (°)", 0.0, 360.0, 0.0, step=0.5, key="probe_az")
    charge = q_in_e * ph.ELEMENTARY_CHARGE
    if charge == 0.0 or speed == 0.0:
        st.warning("Com q = 0 ou v = 0 a força é sempre nula: nenhuma medição é possível.")
    if st.button("Lançar carga de prova"):
        attempts.append(lab_b.launch_probe(b_hidden, charge, speed, theta, azimuth, len(attempts) + 1))

    if attempts:
        last = attempts[-1]
        st.markdown(f"##### Força medida no lançamento nº {last.index}")
        cols = st.columns(4)
        for col, label, value in zip(cols, ("Fx", "Fy", "Fz", "|F|"), (*last.force, last.force_magnitude)):
            col.metric(label, sci(float(value), "N"))
        magnitudes = [a.force_magnitude for a in attempts]
        cols = st.columns(3)
        cols[0].metric("Tentativas", len(attempts))
        cols[1].metric("Maior |F| medido", sci(max(magnitudes), "N"))
        cols[2].metric("Menor |F| medido", sci(min(magnitudes), "N"))
    else:
        st.info("Nenhum lançamento ainda. Ajuste a direção e clique em **Lançar carga de prova**.")

    st.markdown("##### 2. Histórico de tentativas")
    left, right = st.columns([1, 1.2])
    with left:
        if attempts:
            st.dataframe(lab_b.attempts_to_rows(attempts), hide_index=True)
        else:
            st.caption("A tabela aparece após o primeiro lançamento.")
    with right:
        directions = np.array([a.direction for a in attempts]).reshape(-1, 3)
        forces = np.array([a.force_magnitude for a in attempts])
        show_figure(plots.probe_history_figure(directions, forces), "fig_probe_history")
    st.caption("Cada ponto é uma direção de lançamento na esfera unitária, colorida por |F|. "
               "Onde a cor chega perto de zero, a direção está perto de B (ou de −B).")

    st.markdown("##### 3. Sua estimativa de B")
    unit = p.unit
    e1, e2, e3 = st.columns(3)
    with e1:
        est_mag = read_number(f"|B| estimado ({unit})", "est_mag", f"{ph.from_tesla(1.0, unit):g}",
                              strictly_positive=True)
    with e2:
        est_theta = st.slider("θ estimado (°)", 0.0, 180.0, 90.0, step=0.5, key="est_theta")
    with e3:
        est_az = st.slider("Azimute estimado (°)", 0.0, 360.0, 0.0, step=0.5, key="est_az")
    if st.button("Revelar"):
        st.session_state["lab_revealed"] = True
        st.session_state["lab_estimate"] = (ph.to_tesla(est_mag, unit), est_theta, est_az)
    if st.session_state.get("lab_revealed"):
        mag_t, th_e, az_e = st.session_state["lab_estimate"]
        ev = lab_b.evaluate_estimate(b_hidden, mag_t, th_e, az_e)
        cols = st.columns(4)
        cols[0].metric("|B| real", sci(ev.true_magnitude, "T"), help=f"= {ph.from_tesla(ev.true_magnitude, 'G'):.4e} G")
        cols[1].metric("Direção real (θ, azimute)", f"{ev.true_theta_deg:.1f}°, {ev.true_azimuth_deg:.1f}°")
        cols[2].metric("Erro no módulo", fmt_pct(ev.magnitude_error_pct))
        cols[3].metric("Erro angular", f"{ev.angle_error_deg:.2f} °")
        st.caption(f"Estimativa avaliada: |B| = {sci(mag_t, 'T')}, θ = {th_e:.1f}°, azimute = {az_e:.1f}°.")
        if ev.is_success:
            st.success("Ótima medição! Módulo e direção estão muito próximos do campo real.")
        elif ev.right_axis_wrong_sense:
            st.warning("Você acertou a **reta** de B, mas com o **sentido invertido**. "
                       "Use o sentido de F e a regra da mão direita (lembrando do sinal de q).")
        else:
            st.info("Ainda dá para melhorar. Procure a direção de F = 0 e depois a de |F| máxima, e tente de novo.")


# --------------------------------------------------------------------------- #
# Aba E: linhas de campo e ordens de grandeza
# --------------------------------------------------------------------------- #
def render_tab_e(p: Params) -> None:
    st.subheader("Linhas de campo e ordens de grandeza")
    st.latex(r"\text{Dipolo:}\quad \vec B(\vec r)=\frac{\mu_0}{4\pi r^3}\left[3(\vec m\cdot\hat r)\hat r-\vec m\right]")
    kind = st.selectbox("Configuração de campo", ["Campo uniforme", "Ímã de barra (dipolo magnético)"], key="field_kind")
    if kind == "Campo uniforme":
        show_figure(plots.uniform_field_figure(p.b), "fig_field_uniform")
        st.write("Linhas paralelas e igualmente espaçadas: **B é o mesmo em todo o espaço**. "
                 "A figura mostra a projeção do B da barra lateral no plano x–z.")
    else:
        show_figure(plots.dipole_field_figure(), "fig_field_dipole")
        st.write("As linhas **saem pelo polo norte e entram pelo polo sul**. A tangente à linha dá a direção de B e "
                 "as linhas ficam mais densas onde B é mais intenso (perto dos polos). "
                 "O ímã de barra é representado por um dipolo magnético.")

    st.markdown("##### Ordem de grandeza de alguns campos magnéticos (Tabela 28-1)")
    st.caption("Clique em **Aplicar** para usar o valor como |B| na barra lateral (a direção é mantida).")
    header = st.columns([2.2, 1.6, 1.6, 1.2])
    for col, text in zip(header, ("Fonte", "B (T)", "B (G)", "")):
        col.markdown(f"**{text}**")
    for i, ref in enumerate(ph.TYPICAL_FIELDS):
        row = st.columns([2.2, 1.6, 1.6, 1.2])
        row[0].write(f"{ref.name}  \n:gray[{ref.note}]")
        row[1].write(f"{ref.value_tesla:.3g}")
        row[2].write(f"{ph.from_tesla(ref.value_tesla, 'G'):.3g}")
        row[3].button("Aplicar", key=f"apply_field_{i}", on_click=apply_field_preset, args=(ref.value_tesla,))
    st.metric("B atual (barra lateral)", sci(p.b_mag, "T"), delta=f"= {ph.from_tesla(p.b_mag, 'G'):.4e} G",
              delta_color="off")
    st.caption("1 T = 1 N/(C·m/s) = 1 N/(A·m) = 10⁴ G.")


# --------------------------------------------------------------------------- #
# Programa principal
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Laboratório de Campo Magnético", page_icon="🧲", layout="wide")
    st.title(TITLE)
    st.caption("Halliday, seção 28-1: o campo magnético **B** é definido pela força que ele exerce sobre uma carga de prova em movimento.")
    params = sidebar_parameters()
    tabs = st.tabs(["A · Força vetorial", "B · Vetores 3D", "C · F × φ", "D · Descobrindo B",
                    "E · Linhas de campo"])
    for tab, render in zip(tabs, (render_tab_a, render_tab_b, render_tab_c, render_tab_d, render_tab_e)):
        with tab:
            render(params)


if __name__ == "__main__":
    main()
