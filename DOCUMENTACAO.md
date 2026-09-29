# Documentação técnica

Este documento explica, arquivo por arquivo e função por função, **o que** cada parte do projeto faz,
**como** ela faz e **por que** ela existe. Para instalação e uso, veja o [README](README.md).

---

## 1. Visão geral

O projeto é um laboratório virtual (Streamlit + Plotly) sobre a **definição operacional do campo magnético**
(Halliday, seção 28-1): B é definido pela força que exerce sobre uma carga de prova em movimento,

```
F = q v × B        |F| = |q| v B sen φ        B = F / (|q| v sen φ)
```

### Arquitetura em camadas

```
          ┌────────────┐
          │   app.py   │  interface: sidebar, abas, estado da sessão
          └─────┬──────┘
        ┌───────┼─────────┐
        ▼       ▼         ▼
   plots.py  lab_b.py  physics.py
   (Plotly)  (jogo D)  (física pura)
        │       │         ▲
        └───────┴─────────┘  ambos usam physics.py
```

| Camada | Arquivo | Pode importar Streamlit? | Pode importar Plotly? |
|---|---|---|---|
| Física | `physics.py` | não | não |
| Lógica da aba D | `lab_b.py` | não | não |
| Gráficos | `plots.py` | não | sim |
| Interface | `app.py` | sim | (via `plots`) |

**Por que separar assim:** a física e a lógica do experimento ficam testáveis sem abrir navegador
(basta `import physics` num teste com `pytest`), podem ser reaproveitadas em outra interface, e erros de
física não se misturam com erros de layout. `plots.py` só recebe números e devolve figuras, então pode
ser trocado sem mexer na física.

### Convenção de unidades

Tudo é calculado **em SI** (C, m/s, T, N). Gauss só aparece na **entrada** e na **exibição**.
Motivo: misturar unidades dentro dos cálculos é a fonte de erro mais comum; convertendo na borda,
o núcleo nunca precisa saber qual unidade o usuário escolheu.

### Convenção de ângulos

Vetores podem ser dados em coordenadas esféricas:

- **θ (polar)**: ângulo a partir do eixo **+z**, de 0° a 180°;
- **azimute**: ângulo no plano xy a partir de **+x**, de 0° a 360°.

```
x = r sen θ cos(az)     y = r sen θ sen(az)     z = r cos θ
```

---

## 2. `physics.py` — física pura

Não depende de interface. Tudo o que envolve fórmulas mora aqui.

### Constantes

| Nome | Valor | Por quê |
|---|---|---|
| `ELEMENTARY_CHARGE` | 1,602176634×10⁻¹⁹ C | Carga elementar (valor exato no SI desde 2019). Base de todas as cargas. |
| `GAUSS_PER_TESLA` | 10⁴ | 1 T = 10⁴ G. Única fonte do fator de conversão. |
| `FIELD_UNITS` | `("T", "G")` | Lista das unidades aceitas; a interface monta o seletor a partir dela. |
| `PARALLEL_TOLERANCE` | 10⁻⁹ | Tolerância relativa para decidir se v ∥ B (ver `is_parallel`). |
| `Vector` | `np.ndarray` | Apelido de tipo, só para deixar as assinaturas legíveis. |

### `Particle` e `PARTICLES`

- **O que:** `Particle` é um registro imutável (`name`, `charge`); `PARTICLES` é um dicionário
  nome → partícula com elétron (−e), próton (+e) e nêutron (0).
- **Como:** `@dataclass(frozen=True)`; o dicionário é montado por compreensão a partir de uma tupla.
- **Por quê:** a sidebar oferece esses presets; o nêutron existe de propósito para mostrar que
  **q = 0 ⇒ F = 0**. `frozen=True` impede alterar a carga de uma partícula "oficial" por engano.

### Geometria vetorial

#### `unit_vector(vec)`
- **O que:** devolve o vetor normalizado (módulo 1), ou o vetor nulo se `|vec| = 0`.
- **Como:** divide pelo `np.linalg.norm`; testa o zero antes.
- **Por quê:** os gráficos 3D desenham só a **direção** dos vetores (v e B têm ordens de grandeza
  muito diferentes, 10⁶ m/s e 10⁻³ T). O caso nulo evita divisão por zero e `NaN` na figura.

#### `spherical_to_cartesian(magnitude, theta_deg, azimuth_deg)`
- **O que:** converte (módulo, θ, azimute) em (x, y, z).
- **Como:** fórmulas esféricas padrão, com ângulos convertidos para radianos.
- **Por quê:** pensar em "direção" (ângulos) é mais natural para o aluno do que digitar componentes;
  a aba D inteira funciona por ângulos de lançamento.

#### `cartesian_to_spherical(vec)`
- **O que:** o inverso: (x, y, z) → (módulo, θ, azimute ∈ [0°, 360°)).
- **Como:** `acos(z/r)` para θ, com o argumento **limitado a [−1, 1]** (erros de arredondamento
  podem gerar 1,0000000002, que faria `acos` falhar); `atan2(y, x) % 360` para o azimute.
- **Por quê:** usado para mostrar a direção real do campo oculto ao "Revelar" na aba D.

#### `angle_between_deg(a, b)`
- **O que:** ângulo φ entre dois vetores, de 0° a 180°; `NaN` se algum for nulo.
- **Como:** `atan2(|a × b|, a · b)` em vez do tradicional `acos(a·b / |a||b|)`.
- **Por quê:** `acos` perde precisão perto de 0° e 180° (a derivada explode), justamente os casos
  fisicamente importantes (v ∥ B, F = 0). `atan2` é preciso em toda a faixa. Retornar `NaN`
  (e não 0) sinaliza que φ **não está definido**, e a interface mostra "—".

#### `is_parallel(a, b, tolerance)`
- **O que:** `True` se os vetores são paralelos ou antiparalelos (ambos não nulos).
- **Como:** testa `|a × b| ≤ tol · |a| · |b|`, ou seja, `sen φ ≤ 10⁻⁹`.
- **Por quê:** com ponto flutuante, `v × B` quase nunca dá **exatamente** zero (ex.: θ = 180° gera
  `sen(π) ≈ 1,2×10⁻¹⁶`). A tolerância **relativa** funciona para qualquer escala de v e B.

### Força magnética

#### `magnetic_force(charge, v, b_field)`
- **O que:** `F = q v × B`, em newtons.
- **Como:** `charge * np.cross(v, b)`.
- **Por quê:** é a equação central do módulo; forma vetorial dá direção, sentido e módulo de uma vez.

#### `force_magnitude(charge, v_mag, b_mag, phi_deg)`
- **O que:** `|F| = |q| v B sen φ`.
- **Como:** fórmula direta.
- **Por quê:** forma escalar do livro. A aba A a compara com `|q v × B|` para o aluno ver que coincidem,
  e a aba C a usa para |F| no φ do controle e para F máx (φ = 90°).

#### `magnetic_power(force, v)`
- **O que:** potência `P = F · v`.
- **Como:** produto escalar.
- **Por quê:** demonstra numericamente que **P = 0** sempre (F ⊥ v), logo a força magnética
  **não muda |v| nem a energia cinética**, só a direção do movimento.

#### `field_from_force(force_mag, charge, v_mag, phi_deg)`
- **O que:** a definição operacional `B = F / (|q| v sen φ)`.
- **Como:** lança `ValueError` se q = 0, v = 0 ou `sen φ ≤ PARALLEL_TOLERANCE`; senão divide.
- **Por quê:** é exatamente o que a aba D pede ao aluno para fazer "à mão" (os testes usam esta
  função para conferir que a medição da aba D recupera o B verdadeiro). O erro explícito documenta
  que a definição **não se aplica** com q = 0, v = 0 ou v ∥ B. A tolerância é necessária porque
  `sen(180°)` em ponto flutuante vale ~1,2×10⁻¹⁶, não 0; sem ela, φ = 180° devolveria um B absurdo.

#### `force_zero_reasons(charge, v, b_field)`
- **O que:** lista, em português, **todos** os motivos pelos quais F = 0 (vazia se F ≠ 0).
- **Como:** verifica em sequência q = 0, v = 0, B = 0 e `is_parallel(v, B)`.
- **Por quê:** a interface não diz só "F = 0", explica **por que**. Isso transforma um caso-limite
  em conteúdo didático. Também serve de teste robusto de "F é nula?", melhor que comparar
  `|F| == 0` com ponto flutuante.

### Unidades

#### `to_tesla(value, unit)` / `from_tesla(value_tesla, unit)`
- **O que:** convertem entre T/G e tesla.
- **Como:** dividem ou multiplicam por `GAUSS_PER_TESLA`; unidade desconhecida gera `ValueError`.
- **Por quê:** o livro usa as duas unidades (campo da Terra ≈ 0,5 G). O erro explícito evita que uma
  unidade digitada errada seja tratada silenciosamente como tesla.

### Validação de texto

#### `parse_number(text)`
- **O que:** converte texto digitado em `float` finito.
- **Como:**
  1. remove espaços e troca vírgula por ponto (`"1,5"` → `"1.5"`);
  2. troca `x10^`, `×10^` ou `*10^` por `e` via a regex `_TIMES_TEN` (`"3×10^-4"` → `"3e-4"`);
  3. `float(...)`; rejeita vazio, texto inválido, `inf` e `nan`, sempre com mensagem em português.
- **Por quê:** grandezas físicas vão de 10⁻¹⁹ a 10⁶; o `st.number_input` do Streamlit lida mal com
  essas escalas, então os campos são **texto**. Alunos brasileiros digitam vírgula decimal e
  "×10^", e a função aceita as duas formas. `from None` esconde o traceback interno do Python.

---

## 3. `lab_b.py` — lógica da aba "Descobrindo B"

Implementa o experimento mental do livro: há um campo desconhecido; lançamos cargas de prova,
medimos **apenas F** e deduzimos B.

### Constantes

| Nome | Valor | Por quê |
|---|---|---|
| `FIELD_MIN_TESLA`, `FIELD_MAX_TESLA` | 10⁻³ T a 1 T | Faixa realista de campos de laboratório. |
| `SUCCESS_MAGNITUDE_ERROR_PCT` | 5 % | Erro máximo no módulo para considerar a medição boa. |
| `SUCCESS_ANGLE_ERROR_DEG` | 5° | Erro máximo na direção. |

### `Attempt`
- **O que:** registro imutável de um lançamento: índice, carga, velocidade, ângulos, direção
  unitária, vetor força e seu módulo.
- **Por quê:** o histórico precisa guardar **só o que o aluno mediria**. Propositalmente **não guarda B**,
  para que nada na tabela entregue a resposta.

### `Evaluation`
- **O que:** resultado da comparação estimativa × campo real (módulo, ângulos reais, erro percentual,
  erro angular e erro de **eixo**).
- **Propriedades:**
  - `is_success`: erro de módulo ≤ 5 % **e** erro angular ≤ 5°.
  - `right_axis_wrong_sense`: acertou a **reta** de B (erro de eixo ≤ 5°) mas não o **sentido**.
- **Por quê do erro de eixo:** `axis_error = min(φ, 180° − φ)` ignora o sentido. A direção de F = 0
  revela só a **reta** de B (B e −B dão F = 0 na mesma direção); o sentido exige a regra da mão
  direita. Distinguir os dois erros permite dar uma dica específica em vez de só "errou".

### `random_field(rng)`
- **O que:** sorteia o campo oculto.
- **Como:**
  - **módulo log-uniforme**: sorteia o expoente uniforme entre log₁₀(10⁻³) e log₁₀(1). Assim
    campos de 1 mT, 10 mT e 100 mT são igualmente prováveis. Um sorteio uniforme linear daria
    quase sempre valores entre 0,1 T e 1 T;
  - **direção uniforme na esfera**: sorteia **cos θ** uniforme em [−1, 1], não θ. Sortear θ
    uniforme concentraria pontos nos polos.
- **Por quê:** o desafio precisa ser imprevisível e sem direções "favoritas".
- **Recebe `rng` como parâmetro** para que testes possam passar um gerador com semente fixa.

### `manual_field(magnitude_tesla, theta_deg, azimuth_deg)`
- **O que:** campo definido pelo professor ("Modo professor").
- **Como:** valida módulo > 0 e converte de esféricas.
- **Por quê:** o professor pode preparar um desafio específico para a turma. Módulo zero é recusado
  porque aí não há campo a descobrir.

### `launch_probe(b_hidden, charge, speed, theta_deg, azimuth_deg, index)`
- **O que:** simula um lançamento e devolve um `Attempt` com a força medida.
- **Como:** direção unitária a partir dos ângulos → `v = speed · direção` → `F = q v × B`.
  Velocidade negativa gera `ValueError` (sentido negativo já se obtém pelos ângulos).
- **Por quê:** é o "instrumento de medida" do laboratório virtual.

### `attempts_to_rows(attempts)`
- **O que:** converte a lista de `Attempt` em linhas de tabela (dicionários).
- **Como:** formata carga (C), força e velocidade em notação científica.
- **Por quê:** mantém a formatação da tabela fora de `app.py` e mostra só colunas mensuráveis.

### `evaluate_estimate(b_true, magnitude_tesla, theta_deg, azimuth_deg)`
- **O que:** compara a estimativa com o campo real e devolve um `Evaluation`.
- **Como:** converte o campo real para esféricas, calcula o ângulo entre as direções com
  `angle_between_deg` e o erro relativo do módulo.
- **Por quê:** transforma o "Revelar" em retorno quantitativo, não só "certo/errado".

---

## 4. `plots.py` — figuras Plotly

Cada função recebe números e devolve um `go.Figure`. Não conhece Streamlit.

### Constantes de estilo

- **Cores** (`COLOR_V` azul, `COLOR_B` verde, `COLOR_F` vermelho, `COLOR_ARC` laranja, `COLOR_NEUTRAL` cinza):
  cada grandeza tem **sempre a mesma cor** em todas as abas, para o aluno associar cor ↔ vetor.
- **`TRANSPARENT` e `GRID_COLOR` semitransparente:** as figuras funcionam nos temas claro e escuro do
  Streamlit sem precisar detectar o tema.
- **Parâmetros geométricos nomeados** (`ARROW_TIP_SIZE`, `ARC_RADIUS`, `ARC_POINTS`, `SCENE_HALF_RANGE`,
  `ANGLE_CURVE_POINTS`): evitam "números mágicos" espalhados e facilitam ajustes.

### Funções auxiliares (privadas, prefixo `_`)

| Função | O que faz | Por quê |
|---|---|---|
| `_base_layout(fig, height, title)` | Aplica altura, fundos transparentes, margens e legenda horizontal no topo. | Visual uniforme entre as figuras, num só lugar. |
| `_scene(half_range, titles, centers, exponent)` | Monta a configuração da cena 3D: faixas dos eixos, grade, `aspectmode="cube"`. | `cube` garante que ângulos de 90° **pareçam** 90°; sem isso, eixos esticados distorcem a perpendicularidade que se quer mostrar. |
| `_colorscale(color)` | Escala de cor de uma cor só. | `go.Cone` e `go.Surface` exigem escala de cores; esta força cor única. |
| `_add_arrow3d(fig, direction, color, name)` | Desenha uma seta 3D: haste (`Scatter3d`) + ponta (`Cone` ancorado na ponta). | O Plotly não tem seta 3D nativa. `legendgroup` faz haste e ponta ligarem/desligarem juntas na legenda. |

### `vectors_figure(v, b_field, force, charge, force_is_zero)` — aba B
- **O que:** v, B e F em 3D, mais o arco do ângulo φ entre v e B.
- **Como:**
  1. normaliza v e B e desenha as setas (se não nulas);
  2. F é desenhada normalizada, com o sinal de q na legenda; se F = 0, desenha um "×" na origem;
  3. o arco de φ é feito por **interpolação esférica (slerp)**:
     `arco(s) = R · [sen((1−s)φ) v̂ + sen(sφ) B̂] / sen φ`, com s de 0 a 1. Isso gera um arco
     circular exato no plano de v e B. Só é desenhado se v e B não forem paralelos
     (senão `sen φ = 0` e haveria divisão por zero);
  4. rótulo "φ" no meio do arco, levemente afastado (fator 1,35).
- **Por quê:** comprimentos normalizados mostram a **orientação** (o conteúdo do módulo), já que
  os módulos reais diferem por muitas ordens de grandeza. O arco torna φ visível.

### `force_vs_angle_figure(charge, v_mag, b_mag, phi_marker_deg, phi_current_deg)` — aba C
- **O que:** curva `|F| × φ` de 0° a 180°.
- **Como:** 361 pontos (passo de 0,5°); marcadores fixos em F = 0 (0° e 180°) e F máx (90°);
  marcador laranja do controle deslizante; losango cinza no φ real da sidebar (se definido).
  O eixo y tem 25 % de folga acima do máximo para caber os rótulos; se `f_max = 0`, usa uma
  faixa fixa para não gerar eixo degenerado.
- **Por quê:** mostra de uma vez a dependência senoidal, os zeros e o máximo.

### `probe_history_figure(directions, forces)` — aba D
- **O que:** esfera unitária translúcida com um ponto para cada lançamento, colorido por |F|.
- **Como:** `Surface` paramétrica da esfera (opacidade 0,12), eixos x/y/z rotulados, pontos numerados
  com escala `Viridis` a partir de 0.
- **Por quê:** o aluno **vê** onde |F| cai a zero (perto da reta de B) e onde é máximo (no "equador"
  perpendicular a B). Isso guia a busca sem entregar a resposta.

---

## 5. `app.py` — interface Streamlit

### Modelo de execução do Streamlit (importante para entender o código)

O Streamlit **reexecuta o script inteiro** a cada interação. Por isso:

- valores que precisam sobreviver entre interações (campo oculto, histórico, último valor válido)
  ficam em `st.session_state`;
- widgets com `key=` guardam seu valor no `session_state` sob essa chave;
- *callbacks* (`on_change`, `on_click`) rodam **antes** da reexecução, o que permite alterar estado
  que um widget vai ler.

### Constantes

- `TITLE`, `CUSTOM_PARTICLE`, `MODE_POLAR`, `MODE_CART`: textos usados em mais de um lugar.
- `DEFAULT_*`: caso inicial = **próton, v = 10⁶ m/s a 60° de B, B = 1 mT em +z**. Escolhido para que
  a primeira tela já mostre uma força não nula com φ "genérico" (nem 0°, nem 90°).
  As componentes padrão (8,66×10⁵; 0; 5×10⁵) são o mesmo vetor v em cartesianas, assim os dois
  modos de entrada começam coerentes.
- `FIELD_TEXT_KEYS`: chaves dos campos de texto **expressos na unidade de B** (precisam ser
  convertidas quando a unidade muda).

### `Params`
- **O que:** registro imutável com os parâmetros globais em SI (partícula, q, v, B, unidade) e
  propriedades derivadas: `v_mag`, `b_mag`, `phi`, `force`, `zero_reasons`.
- **Por quê:** a sidebar é lida **uma vez** e o mesmo objeto é passado a todas as abas; as
  propriedades evitam repetir cálculos e garantem que todas as abas usem os mesmos números.

### Utilitários

| Função | O que faz | Por quê |
|---|---|---|
| `init_state(key, default)` | Cria a chave no `session_state` só se não existir. | Definir valor padrão **e** `key` num widget gera aviso do Streamlit; iniciar o estado antes evita isso e não sobrescreve o que o usuário digitou. |
| `show_figure(fig, key)` | `st.plotly_chart` em largura total. | Padroniza a exibição; `key` fixa evita recriar o gráfico (e perder a rotação 3D) a cada rerun. |
| `sci(value, unit, digits)` | Formata `1.6022e-16 N`; `0 N` para zero; `—` para `NaN`. | Grandezas de 10⁻¹⁹ a 10⁶ só são legíveis em notação científica. |
| `fmt_pct(value)` | Formata percentual ou `—`. | Mesmo motivo, para erros relativos. |

### `read_number(label, key, default, *, strictly_positive, minimum, help_text)`
- **O que:** campo de texto numérico validado.
- **Como:**
  1. mostra `st.text_input`;
  2. valida com `ph.parse_number` e com as restrições opcionais (> 0, ≥ mínimo);
  3. se válido, guarda em `_ok_<key>` e devolve;
  4. se inválido, mostra `st.error` com a mensagem e devolve o **último valor válido**
     (`_ok_<key>`) ou o padrão.
- **Por quê:** enquanto o aluno digita (ex.: `"1e-"`), o texto fica temporariamente inválido. Sem o
  "último valor válido", o app quebraria ou os gráficos pulariam para zero a cada tecla.

### `angle_input(label, key, lo, hi, default)` — ângulo por barra **e** por texto
- **O que:** todo ângulo do app (θ e azimute de v, B, lançamento, estimativa e modo professor; φ da aba C)
  aparece como uma **barra** com uma **caixa de texto** ao lado. Mexer em um atualiza o outro.
- **Como:**
  - o valor oficial fica em `st.session_state[key]`; a barra (`<key>_sl`) e o texto (`<key>_txt`)
    são só duas visões dele;
  - `_angle_from_slider` (callback da barra): copia o valor para o oficial e reescreve o texto;
  - `_angle_from_text` (callback do texto): aceita vírgula decimal e o símbolo `°`, valida a faixa
    (0–180° para θ e φ, 0–360° para o azimute). Se válido, atualiza o oficial e move a barra
    (`_snap_to_slider` arredonda ao passo de 0,5°). Se inválido, guarda a mensagem em `<key>_err`,
    que é exibida com `st.error`, e o valor oficial **não muda**;
  - na criação, barra e texto são iniciados **a partir do valor oficial**, porque o Streamlit apaga
    o estado de widgets que não foram desenhados numa execução (ex.: trocar para o modo
    *Componentes* e voltar). Sem isso, o ângulo voltaria ao padrão.
- **Por quê:** a barra é rápida para explorar; o texto permite um valor exato (ex.: 37,25°, que a
  barra de passo 0,5° não alcança). O cálculo usa o valor digitado; a barra mostra o passo mais próximo.
  Precisa de callbacks porque um widget não pode ter seu estado alterado depois de desenhado
  na mesma execução.

### `convert_field_texts()` — callback da troca T ↔ G
- **O que:** quando a unidade muda, reescreve os textos de campo já digitados na nova unidade,
  preservando o **valor físico**.
- **Como:** para cada chave em `FIELD_TEXT_KEYS` (e sua cópia `_ok_`), lê o valor, converte
  `antiga → tesla → nova` e grava de volta (6 algarismos significativos para texto). A unidade
  anterior fica em `_unit_prev`. Textos inválidos são mantidos como estão.
- **Por quê:** sem isso, trocar de T para G com "0.001" digitado transformaria 1 mT em 0,001 G,
  um campo 10⁴ vezes menor, só por mudar a unidade de exibição. Precisa ser **callback** porque
  deve rodar antes que os `text_input` leiam o estado.

### Sidebar

#### `_vector_input(tag, title, unit, scale_to_si, mode, mag_default, comp_defaults, theta_default)`
- **O que:** lê um vetor (v ou B) no modo escolhido e devolve em SI.
- **Como:** modo polar: módulo (texto, ≥ 0) + θ e azimute por `angle_input` → `spherical_to_cartesian`.
  Modo componentes: três campos de texto lado a lado. Multiplica por `scale_to_si`
  (1 para v; fator T/G para B).
- **Por quê:** uma função só para os dois vetores evita duplicar código; os dois modos atendem
  tanto quem pensa em ângulos quanto quem tem as componentes do enunciado do livro.

#### `sidebar_parameters()`
- **O que:** desenha toda a sidebar e devolve um `Params`.
- **Como:** preset de partícula (ou carga personalizada digitada **em coulombs**, via `read_number`,
  aceitando `1.6e-19` ou `1,6×10^-19`); caixa "Inverter o sinal
  de q"; modo de entrada; unidade de B (com `on_change=convert_field_texts`); lê v e B.
  O padrão de B é escrito em tesla porque, no primeiro carregamento, a unidade é sempre T.
- **Por quê:** parâmetros compartilhados pelas abas A, B e C ficam num lugar fixo e visível.
  O botão de inverter q existe para o aluno ver **F inverter** sem redigitar nada.

### Aba A — `render_tab_a(p)`: calculadora vetorial
- **O que mostra:**
  1. componentes Fx, Fy, Fz e |F|; φ, |v|, |B| (com tooltip em gauss) e q;
  2. avisos com cada motivo de F = 0;
  3. **conferência automática**: `|q v × B|` (via `np.cross`) × `|q| v B sen φ`, com a diferença
     absoluta e relativa;
  4. **potência F · v** (sempre 0) e componente de F ao longo de v;
  5. **direção e sentido**: F̂, regra para q > 0 e q < 0, e os ângulos F–v e F–B (ambos 90°).
- **Por quê:** cobre os objetivos do módulo: calcular F pelas duas formas, ver que são a mesma coisa
  (a diferença é só arredondamento, ~10⁻¹⁶), e verificar numericamente F ⊥ v, F ⊥ B e a
  conservação da energia cinética.

### Aba B — `render_tab_b(p)` e `right_hand_rule_text(p)`
- **O que:** métricas (|F|, φ, sinal de q), figura 3D e legenda da regra da mão direita adaptada
  ao sinal de q (ou o motivo de F = 0).
- **Por quê:** a regra da mão direita é espacial; um texto sozinho não basta. A figura girável
  com a explicação ao lado liga a regra ao desenho.

### Aba C — `render_tab_c(p)`: F × φ
- **O que:** φ **independente** da sidebar (barra + texto, via `angle_input`), métricas (|F| no φ escolhido, F máx, sen φ)
  e o gráfico.
- **Por quê:** permite explorar a dependência em sen φ sem desmontar os vetores da sidebar; o losango
  cinza liga o gráfico ao caso atual. Se q, v ou B forem zero, avisa que F = 0 para qualquer φ.

### Aba D — `reset_lab()` e `render_tab_d(p)`: Descobrindo B

#### `reset_lab()`
- **O que:** callback do botão "Novo campo": sorteia campo, limpa histórico e esconde a resposta.
- **Por quê:** como callback, o estado novo já vale na reexecução seguinte.

#### `render_tab_d(p)`
Fluxo da tela:

1. **Campo oculto**: criado na primeira visita (`reset_lab`). O **Modo professor** tem o módulo
   em campo de **senha** (não aparece para a turma no projetor) e θ/azimute por `angle_input`.
   O botão chama o callback `set_teacher_field`, que valida, define o campo, limpa o histórico e
   **apaga o módulo digitado**. Não se usa `st.form` porque widgets dentro de um formulário não
   aceitam callbacks, e a sincronização barra ↔ texto depende deles.
2. **Carga de prova e lançamento**: carga **em coulombs** (texto), velocidade, θ e azimute próprios da aba (a sidebar
   não entra, para não "vazar" B). Avisa se q = 0 ou v = 0. O botão "Lançar" adiciona um
   `Attempt` à lista guardada no `session_state` (a lista é mutada no lugar, então persiste).
3. **Resultado do último lançamento** e resumo (nº de tentativas, maior e menor |F|).
4. **Histórico**: tabela (`attempts_to_rows`) e esfera de direções (`probe_history_figure`).
   O `reshape(-1, 3)` garante formato correto mesmo com a lista vazia.
5. **Estimativa**: |B| na unidade da sidebar, θ e azimute. "Revelar" guarda a estimativa **convertida
   para tesla** em `lab_estimate`; a avaliação é refeita a cada rerun a partir desse valor salvo.
   Mensagens: sucesso, "reta certa, sentido invertido" ou dica geral.

- **Por quê:** é o núcleo conceitual do módulo 28-1. B **não é medido diretamente**; é
  **definido** pelo procedimento: achar a direção de F = 0 (reta de B), achar |F| máximo
  (|B| = F/(|q| v)) e usar a regra da mão direita para o sentido. Guardar a estimativa no clique
  evita que mexer nos sliders depois altere a avaliação já revelada.

### `main()`
- **O que:** configura a página (título, ícone, layout largo), lê a sidebar uma vez e renderiza as
  quatro abas.
- **Como:** `zip` entre as abas e as funções `render_tab_*`, todas com a mesma assinatura `(Params)`.
- **Por quê:** mantém o ponto de entrada curto; adicionar uma aba é acrescentar um nome e uma função.

> Observação: o Streamlit executa **todas** as abas a cada rerun (só a exibição é por aba). Por isso
> nenhuma aba faz cálculo pesado.

---

## 6. Arquivos de projeto

| Arquivo | O que é | Por que existe |
|---|---|---|
| `requirements.txt` | Dependências para `pip` (streamlit, plotly, numpy). | Instalação simples descrita no README. |
| `pyproject.toml` | Metadados e dependências no formato padrão (usado pelo `uv`). | Permite `uv sync` / `uv run`. |
| `uv.lock` | Versões exatas resolvidas pelo `uv`. | Instalações reprodutíveis. Não editar à mão. |
| `.python-version` | Versão do Python para o `uv`/pyenv. | Fixa o interpretador do projeto. |
| `.gitignore` | Ignora `__pycache__`, builds e `.venv`. | Mantém o repositório só com código-fonte. |

O `pyproject.toml` declara `requires-python = ">=3.10"` (o código usa `from __future__ import annotations`
e `X | Y`, que exigem 3.10+). O `.python-version` escolhe 3.14 só para o ambiente local do `uv`.
O `pytest` fica no grupo `dev` (`[dependency-groups]`): não é necessário para rodar o app, e o `uv`
o instala por padrão em `uv sync`/`uv run`.

---

## 7. Testes (`tests/`)

Rodar com `uv run pytest` (ou `pytest`, com o pytest instalado). A configuração em
`[tool.pytest.ini_options]` põe a raiz do projeto no `sys.path`, para os testes importarem
`physics` e `lab_b` diretamente.

| Arquivo | O que verifica | Por quê |
|---|---|---|
| `test_physics.py` | F = q v × B num caso conhecido; F inverte com o sinal de q; forma vetorial = forma escalar; F ⊥ v, F ⊥ B e F · v ≈ 0; `field_from_force` recupera B e recusa q = 0, v = 0, φ = 0° e 180°; precisão de φ perto de 0° e 180°; ida e volta esférica ↔ cartesiana; motivos de F = 0 (incluindo θ = 180° com erro de arredondamento); conversão T ↔ G; `parse_number` aceitando vírgula e `×10^` e recusando vazio, texto, `inf`, `nan`. | Garante as leis do módulo 28-1 e os casos-limite em ponto flutuante, que são os mais fáceis de quebrar. |
| `test_lab_b.py` | Sorteio dentro da faixa; campo manual recusa módulo ≤ 0; lançar ao longo de B dá F = 0 e perpendicular dá F máx (e `field_from_force` recupera |B|); velocidade negativa é recusada; estimativa exata é sucesso; estimativa −B é "reta certa, sentido invertido". | Garante que o experimento da aba D é solucionável pelo método ensinado e que a avaliação dá a dica certa. |

A física usa tolerâncias **relativas** nos testes (ex.: `|F · v| ≤ 10⁻¹² |F||v|`), porque as grandezas
vão de 10⁻¹⁹ a 10⁶ e uma tolerância absoluta fixa não serve para todas.

---

## 8. Fluxo de dados (resumo)

```
usuário digita/move widget
        │
        ▼
Streamlit reexecuta app.py
        │
        ├─ sidebar_parameters() ── read_number / parse_number ──► Params (SI)
        │
        ├─ Aba A ── physics.magnetic_force / angle_between_deg / magnetic_power
        ├─ Aba B ── plots.vectors_figure
        ├─ Aba C ── plots.force_vs_angle_figure
        └─ Aba D ── lab_b.launch_probe / evaluate_estimate ── session_state (campo, histórico)
                    plots.probe_history_figure
```
