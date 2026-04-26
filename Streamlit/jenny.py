import streamlit as st

st.set_page_config(page_title="Acertijo de Jenny", page_icon="🧩")

st.title("Acertijo de los cuadrados amarillos")

st.markdown(
    r"""
Completa los **cuadrados amarillos** y pulsa **“Verificar respuesta”**.
"""
)

# --- CSS: que los NumberInput parezcan cuadros amarillos ---
st.markdown(
    """
    <style>
    div[data-testid="stNumberInput"] > label {
        display: none;
    }
    div[data-testid="stNumberInput"] input {
        background-color: #ffe866;
        border: 2px solid #e0c000;
        border-radius: 8px;
        text-align: center;
        font-size: 1.5rem;
        height: 3rem;
        width: 4rem;
        margin: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Disposición del acertijo ---

# Fila 1: a · b = 15
fila1 = st.columns([1, 0.25, 1, 0.25, 1])
with fila1[0]:
    a = st.number_input("", key="a", value=0, step=1)
with fila1[1]:
    st.markdown("### $\\cdot$")
with fila1[2]:
    b = st.number_input("", key="b", value=0, step=1)
with fila1[3]:
    st.markdown("### $=$")
with fila1[4]:
    st.markdown("### $15$")

# Fila 2: signos de +
fila2 = st.columns([1, 0.25, 1, 0.25, 1])
with fila2[0]:
    st.markdown("### $+$")
with fila2[2]:
    st.markdown("### $+$")

# Fila 3: c − d = 5
fila3 = st.columns([1, 0.25, 1, 0.25, 1])
with fila3[0]:
    c = st.number_input("", key="c", value=0, step=1)
with fila3[1]:
    st.markdown("### $-$")
with fila3[2]:
    d = st.number_input("", key="d", value=0, step=1)
with fila3[3]:
    st.markdown("### $=$")
with fila3[4]:
    st.markdown("### $5$")

# Fila 4: igualdades de columna
fila4 = st.columns([1, 0.25, 1, 0.25, 1])
with fila4[0]:
    st.markdown("### $=\\;3$")
with fila4[2]:
    st.markdown("### $=\\;12$")

st.write("---")

# --- Verificar respuesta ---

if st.button("Verificar respuesta"):
    eq1 = abs(a * b - 15) < 1e-9
    eq2 = abs(a + c - 3) < 1e-9
    eq3 = abs(b + d - 12) < 1e-9
    eq4 = abs(c - d - 5) < 1e-9

    if eq1 and eq2 and eq3 and eq4:
        st.success("✅ ¡Correcto! Tus números satisfacen todas las ecuaciones.")
    else:
        st.error("❌ Aún no es correcto. Alguna ecuación no se cumple.")
        st.markdown("### Valores que tienes actualmente:")

        prod = a * b
        suma1 = a + c
        suma2 = b + d
        resta = c - d

        # LaTeX limpio, con \text{} y sin decimales forzados
        st.latex(rf"{a}\cdot {b} = {prod} \quad \text{{debería ser }} 15")
        st.latex(rf"{a} + {c} = {suma1} \quad \text{{debería ser }} 3")
        st.latex(rf"{b} + {d} = {suma2} \quad \text{{debería ser }} 12")
        st.latex(rf"{c} - {d} = {resta} \quad \text{{debería ser }} 5")

# --- Solución algebraica en LaTeX ---

with st.expander("Mostrar solución"):
    st.markdown(
        r"""
Llamemos \(a,b,c,d\) a los cuatro números, dispuestos así:

$$
\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
$$

El acertijo se traduce en el sistema

$$
\begin{aligned}
a\cdot b &= 15,\\
a + c &= 3,\\
b + d &= 12,\\
c - d &= 5.
\end{aligned}
$$

De la segunda ecuación:

$$
a + c = 3 \quad\Rightarrow\quad c = 3 - a.
$$

De la cuarta ecuación:

$$
c - d = 5 
\ \Rightarrow\ 
(3 - a) - d = 5 
\ \Rightarrow\ 
d = -2 - a.
$$

De la tercera ecuación:

$$
b + d = 12 
\ \Rightarrow\ 
b + (-2 - a) = 12 
\ \Rightarrow\ 
b = 14 + a.
$$

Sustituimos en la primera ecuación \(a\cdot b = 15\):

$$
a(14 + a) = 15
\ \Rightarrow\ 
a^2 + 14a - 15 = 0.
$$

Resolvemos la ecuación cuadrática:

$$
\Delta = 14^2 - 4\cdot 1\cdot(-15) = 196 + 60 = 256,
\qquad
\sqrt{\Delta} = 16,
$$

$$
a = \frac{-14 \pm 16}{2}
= \frac{-14 + 16}{2},\ \frac{-14 - 16}{2}
= 1,\ -15.
$$

**Caso 1:** $a = 1$.

$$
b = 14 + 1 = 15,\qquad
c = 3 - 1 = 2,\qquad
d = -2 - 1 = -3.
$$

**Caso 2:** $a = -15$.

$$
b = 14 - 15 = -1,\qquad
c = 3 - (-15) = 18,\qquad
d = -2 - (-15) = 13.
$$

Por lo tanto, las soluciones del sistema son

$$
(1,15,2,-3)
\quad\text{y}\quad
(-15,-1,18,13).
$$

Cualquiera de estos dos conjuntos de números hace verdaderas todas las ecuaciones del acertijo.
"""
    )



