# Laboratório Virtual de Campo Magnético

**Medindo B a partir da força magnética** (Halliday, Cap. 28, módulo 28-1: *Campos magnéticos e a definição de B*).

Aplicação web interativa em Python (**Streamlit** + **Plotly**) que mostra como o campo magnético **B** é
definido operacionalmente: pela força magnética que ele exerce sobre uma carga de prova em movimento,

```
F = q v × B        F = |q| v B sen φ        B = F / (|q| v sen φ)
```

O conteúdo se limita à teoria do módulo 28-1: definição de B, força sobre carga em movimento, regra da mão
direita, F ⊥ v e F ⊥ B, casos em que F = 0, conservação de |v| e da energia cinética, unidades (tesla e gauss).

## Instalação

Requer Python 3.10 ou superior.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Como rodar

```bash
streamlit run app.py
```

Com [uv](https://docs.astral.sh/uv/): `uv run streamlit run app.py`.

## Testes

```bash
pip install pytest      # ou: uv sync
pytest                  # ou: uv run pytest
```

## Interface

A **barra lateral** guarda os parâmetros globais: partícula (elétron, próton, nêutron ou
personalizada, com a carga digitada em coulombs), carga q (com botão para inverter o sinal), velocidade v e campo B.
Cada vetor pode ser dado por **módulo + ângulos** (θ a partir de +z e azimute no plano xy) ou por **componentes**.
Todo ângulo tem uma barra e uma caixa de texto sincronizadas: arraste ou digite o valor exato. O campo aceita **tesla (T)**
ou **gauss (G)**; a troca converte o valor digitado, e internamente tudo é calculado em SI. Números aceitam
notação científica (`1e-3`) e vírgula decimal; entradas inválidas geram mensagens claras.

| Aba | Conteúdo (objetivos 28.04 a 28.11) |
|---|---|
| **A · Força vetorial** | F = q v × B por componentes, \|F\|, ângulo φ; conferência `np.cross` × `\|q\| v B sen φ`; F · v = 0 (a força não muda \|v\| nem a energia cinética); direção e sentido; avisos quando F = 0. |
| **B · Vetores 3D** | v, B e F em 3D com o arco de φ; F inverte ao trocar o sinal de q; legenda da regra da mão direita. |
| **C · F × φ** | Curva \|F\| × φ (0° a 180°) com marcador móvel; F = 0 em 0° e 180°, máxima em 90°. |
| **D · Descobrindo B** | Reproduz a definição operacional: um B oculto é sorteado (ou definido pelo professor); lance cargas de prova, meça só F, ache a direção de F = 0 (direção de B) e a de \|F\| máxima (\|B\| = F/(\|q\| v)); estime e clique em "Revelar". |

## Estrutura

```
campo_magnetico/
├── app.py              # interface Streamlit (sidebar + abas)
├── physics.py          # F = q v×B, ângulo, unidades, validação de texto
├── plots.py            # todas as figuras Plotly
├── lab_b.py            # lógica do módulo "Descobrindo B"
├── tests/              # testes pytest de physics.py e lab_b.py
├── pyproject.toml      # metadados e dependências (uv)
├── requirements.txt
└── README.md
```

`physics.py` e `lab_b.py` não importam Streamlit nem Plotly; só `app.py` e `plots.py` tocam a interface e os gráficos.

A explicação detalhada de cada módulo e função (o que faz, como e por quê) está em [DOCUMENTACAO.md](DOCUMENTACAO.md).

## Limitações do modelo

- Campo magnético **uniforme** em todas as abas.
- Só a força magnética é considerada (sem campo elétrico) e não se calcula a trajetória da partícula
  (movimento circular e helicoidal pertencem a outros módulos do capítulo).

## Capturas de tela

Adicione suas capturas em `docs/` e referencie aqui, por exemplo: `![Aba A](docs/aba_a.png)`.
