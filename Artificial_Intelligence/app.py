import streamlit as st
import sympy as sp
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Solución Examen Álgebra Lineal",
    layout="wide",
    page_icon="📘"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&display=swap');

html, body, [class*="css"] {
    font-family: 'Libre Baskerville', Georgia, serif;
}

.section-header {
    background: linear-gradient(90deg, #18294a 0%, #274c7c 100%);
    color: white !important;
    padding: 0.65em 1em;
    border-radius: 8px;
    margin: 1.5em 0 0.8em 0;
    font-size: 1.15em;
    font-weight: bold;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
}

.data-box {
    background: #f6f8fc;
    border-left: 6px solid #18294a;
    padding: 1em 1.1em;
    border-radius: 8px;
    margin: 0.8em 0 1em 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.solution-box {
    background: #f3faf5;
    border-left: 6px solid #3a8258;
    padding: 1em 1.1em;
    border-radius: 8px;
    margin: 0.7em 0 1em 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.concept-box {
    background: #fffbea;
    border-left: 6px solid #c6983a;
    padding: 0.9em 1em;
    border-radius: 8px;
    margin: 0.7em 0 1em 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.stDownloadButton > button {
    background: #18294a !important;
    color: white !important;
    border: none !important;
    padding: 0.65em 1.4em !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

.stButton > button {
    background: #18294a !important;
    color: white !important;
    border: none !important;
    padding: 0.7em 2em !important;
    border-radius: 8px !important;
    width: 100%;
    font-weight: 600 !important;
}

hr {
    margin-top: 1.2em !important;
    margin-bottom: 1.2em !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("# Solucionario Interactivo: Álgebra Lineal")
st.markdown(
    "Ingresa tu número de cuenta de **7 dígitos**. "
    "Se genera la solución completa con gráficas y el código LaTeX listo para Overleaf."
)

cuenta = st.text_input(
    "Número de cuenta (exactamente 7 dígitos):",
    "4221286",
    max_chars=7
)


# =============================================================================
# HELPERS — GRÁFICAS 3D
# =============================================================================

def _build_plane_surface_from_normal(normal, lim=4, resolution=30):
    """
    Construye una parametrización del plano:
        n1*x + n2*y + n3*z = 0
    eligiendo automáticamente la variable a despejar para evitar
    errores cuando alguno de los coeficientes es cero.
    """
    n1, n2, n3 = [float(v) for v in normal]
    coeffs = np.array([abs(n1), abs(n2), abs(n3)])

    # Elegimos despejar la variable cuyo coeficiente tenga mayor magnitud
    # para evitar divisiones por números muy pequeños.
    idx = int(np.argmax(coeffs))

    s = np.linspace(-lim, lim, resolution)
    t = np.linspace(-lim, lim, resolution)
    S, T = np.meshgrid(s, t)

    if coeffs[idx] < 1e-12:
        # Caso degenerado: normal ~ 0, no define plano.
        # Devolvemos plano horizontal por seguridad.
        X, Y = S, T
        Z = np.zeros_like(S)
        return X, Y, Z

    if idx == 2:
        # despejar z
        X = S
        Y = T
        Z = (-n1 * X - n2 * Y) / n3
    elif idx == 1:
        # despejar y
        X = S
        Z = T
        Y = (-n1 * X - n3 * Z) / n2
    else:
        # despejar x
        Y = S
        Z = T
        X = (-n2 * Y - n3 * Z) / n1

    return X, Y, Z


def plot_plane_with_vectors(normal, vecs, labels, title):
    fig = go.Figure()
    colors = ['#7852aa', '#de7e2c', '#3a8258']

    X, Y, Z = _build_plane_surface_from_normal(normal, lim=4, resolution=35)

    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        opacity=0.35,
        colorscale=[[0, '#d8e5fb'], [1, '#244a78']],
        showscale=False,
        name="Plano W"
    ))

    for i, v in enumerate(vecs):
        vf = [float(x) for x in v]
        mx = max((abs(x) for x in vf if abs(x) > 0.01), default=1)
        scale = 3.0 / mx
        vs = [x * scale for x in vf]

        fig.add_trace(go.Scatter3d(
            x=[0, vs[0]],
            y=[0, vs[1]],
            z=[0, vs[2]],
            mode='lines+markers+text',
            line=dict(color=colors[i % 3], width=7),
            marker=dict(size=[2, 6], color=colors[i % 3]),
            text=['', labels[i]],
            textposition='top center',
            name=labels[i]
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z',
            xaxis=dict(backgroundcolor='rgba(240,244,250,0.25)'),
            yaxis=dict(backgroundcolor='rgba(240,244,250,0.25)'),
            zaxis=dict(backgroundcolor='rgba(240,244,250,0.25)')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        height=340
    )
    return fig


def plot_vectors_3d(vecs, labels, title, plane_vecs=None):
    fig = go.Figure()
    colors = ['#7852aa', '#de7e2c', '#3a8258', '#18294a']

    if plane_vecs and len(plane_vecs) >= 2:
        v1 = np.array([float(x) for x in plane_vecs[0]])
        v2 = np.array([float(x) for x in plane_vecs[1]])
        T2, S2 = np.meshgrid(np.linspace(-1.5, 1.5, 10), np.linspace(-1.5, 1.5, 10))

        fig.add_trace(go.Surface(
            x=T2 * v1[0] + S2 * v2[0],
            y=T2 * v1[1] + S2 * v2[1],
            z=T2 * v1[2] + S2 * v2[2],
            opacity=0.25,
            colorscale=[[0, '#d8e5fb'], [1, '#244a78']],
            showscale=False,
            name="Span"
        ))

    for i, v in enumerate(vecs):
        vf = [float(x) for x in v]
        mx = max((abs(x) for x in vf if abs(x) > 0.01), default=1)
        scale = min(3.0 / mx, 1.5)
        vs = [x * scale for x in vf]

        fig.add_trace(go.Scatter3d(
            x=[0, vs[0]],
            y=[0, vs[1]],
            z=[0, vs[2]],
            mode='lines+markers+text',
            line=dict(color=colors[i % 4], width=7),
            marker=dict(size=[2, 6], color=colors[i % 4]),
            text=['', labels[i]],
            textposition='top center',
            name=labels[i]
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z',
            xaxis=dict(backgroundcolor='rgba(240,244,250,0.25)'),
            yaxis=dict(backgroundcolor='rgba(240,244,250,0.25)'),
            zaxis=dict(backgroundcolor='rgba(240,244,250,0.25)')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        height=340
    )
    return fig


def plot_kernel_line(ker_vec, title):
    fig = go.Figure()
    v = [float(x) for x in ker_vec]
    mx = max((abs(x) for x in v if abs(x) > 0.01), default=1)
    scale = 3.0 / mx
    t = np.linspace(-1, 1, 50)

    fig.add_trace(go.Scatter3d(
        x=t * v[0] * scale,
        y=t * v[1] * scale,
        z=t * v[2] * scale,
        mode='lines',
        line=dict(color='#7852aa', width=7),
        name="Ker(T)"
    ))

    label = "(" + str(int(v[0])) + "," + str(int(v[1])) + "," + str(int(v[2])) + ")"
    fig.add_trace(go.Scatter3d(
        x=[0, v[0] * scale * 0.9],
        y=[0, v[1] * scale * 0.9],
        z=[0, v[2] * scale * 0.9],
        mode='lines+markers+text',
        line=dict(color='#7852aa', width=4),
        marker=dict(size=[2, 7], color='#7852aa'),
        text=['', label],
        textposition='top center',
        name="base ker"
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z',
            xaxis=dict(backgroundcolor='rgba(240,244,250,0.25)'),
            yaxis=dict(backgroundcolor='rgba(240,244,250,0.25)'),
            zaxis=dict(backgroundcolor='rgba(240,244,250,0.25)')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        height=340
    )
    return fig


# =============================================================================
# HELPER — CONSTRUCCIÓN DEL LATEX
# =============================================================================

def build_latex(cuenta, d1, d2, d3, d4, d5, d6, d7,
                A, u, v, w,
                A11, A12, A13,
                base_W_1, base_W_2,
                M_uv, M_uv_rref, pivotes_uv, son_indep, w_en_u,
                A_rref, pivotes_A, rango_A, base_col_A, base_row_A,
                base_ker, nulidad_A,
                A1_sys, bI, bI_1, bI_2, bI_3,
                x_val, y_val, z_val,
                b_II, bII_1, bII_2, bII_3, sol_II,
                bIII_1, bIII_2, bIII_3, bIII_sum,
                eq_W):

    def m(expr):
        return sp.latex(expr)

    def cols(lst):
        return r",\; ".join([m(c) for c in lst])

    def rows_l(lst):
        return r",\; ".join([m(r) for r in lst])

    n_piv = str(len(pivotes_uv))
    indep_txt = "linealmente independientes (LI)" if son_indep else "linealmente dependientes (LD)"
    w_txt = "tiene solución: $w\\in U$." if w_en_u else "es incompatible: $w\\notin U$."
    w_box = r"w\in U." if w_en_u else r"w\notin U."
    ker_geom = (
        "Aquí el kernel es una recta." if nulidad_A == 1
        else "Aquí el kernel es solo el origen." if nulidad_A == 0
        else "Aquí el kernel es un plano."
    )
    pivotes_txt = ", ".join([str(p + 1) for p in pivotes_A])
    det_val = str(A.det())

    if nulidad_A == 1:
        tikz_ker = "\n".join([
            r"\begin{center}",
            r"\tdplotsetmaincoords{68}{118}",
            r"\begin{tikzpicture}[tdplot_main_coords,scale=0.95,line cap=round,line join=round]",
            r"  \draw[eje](0,0,0)--(4,0,0)node[below left]{$x$};",
            r"  \draw[eje](0,0,0)--(0,4,0)node[right]{$y$};",
            r"  \draw[eje](0,0,0)--(0,0,3.5)node[above]{$z$};",
            r"  \draw[color=morado,line width=1.3pt](-1.2,4.2,-1.5)--(1.2,-4.2,1.5);",
            r"  \draw[vecA](0,0,0)--(0.7,-2.7,0.9)node[right]{\(\," + m(base_ker) + r"\)};",
            r"  \fill(0,0,0)circle(1.2pt);",
            r"\end{tikzpicture}",
            r"\end{center}",
        ])
    else:
        tikz_ker = ""

    L = []

    def a(s=""):
        L.append(s)

    a(r"\documentclass[12pt,letterpaper]{article}")
    a(r"\usepackage[utf8]{inputenc}")
    a(r"\usepackage[T1]{fontenc}")
    a(r"\usepackage[spanish,es-nodecimaldot]{babel}")
    a(r"\usepackage{amsmath,amssymb,amsthm,mathtools}")
    a(r"\usepackage[expansion=false]{microtype}")
    a(r"\usepackage{geometry}")
    a(r"\usepackage{array,booktabs,multirow,enumitem,multicol,xcolor}")
    a(r"\usepackage[most]{tcolorbox}")
    a(r"\usepackage{fancyhdr,lastpage,hyperref,ifthen}")
    a(r"\usepackage{tikz,tikz-3dplot,tabularx}")
    a(r"\usetikzlibrary{arrows.meta,calc,positioning,decorations.pathreplacing,")
    a(r"  backgrounds,fit,shadows.blur,shadows,shapes.geometric,matrix,3d,babel}")
    a(r"\geometry{margin=2.35cm}")
    a(r"\setlength{\parskip}{0.5em}")
    a(r"\setlength{\parindent}{0pt}")
    a(r"\newif\ifprofesor\profesortrue")
    a(r"\definecolor{azul}{RGB}{24,74,140}")
    a(r"\definecolor{azulclaro}{RGB}{230,239,252}")
    a(r"\definecolor{grisclaro}{RGB}{246,247,249}")
    a(r"\definecolor{verdesuave}{RGB}{240,248,243}")
    a(r"\definecolor{rojoclaro}{RGB}{252,243,243}")
    a(r"\definecolor{morado}{RGB}{120,82,170}")
    a(r"\definecolor{naranja}{RGB}{222,126,44}")
    a(r"\definecolor{verdeoscuro}{RGB}{58,130,88}")
    a(r"\definecolor{dorado}{RGB}{198,152,58}")
    a(r"\definecolor{amarillosuave}{RGB}{255,251,230}")
    a(r"\hypersetup{colorlinks=true,linkcolor=azul,urlcolor=azul,citecolor=azul}")
    a(r"\pagestyle{fancy}\fancyhf{}")
    a(r"\fancyhead[L]{\textsc{Álgebra Lineal}}")
    a(r"\fancyhead[R]{\textsc{Versión profesor}}")
    a(r"\fancyfoot[C]{\thepage\ de \pageref{LastPage}}")
    a(r"\setlength{\headheight}{15pt}")
    a(r"\newtheorem{exercise}{Ejercicio}")
    a(r"\tcbset{enhanced,breakable,boxrule=0.9pt,arc=2mm,outer arc=2mm,")
    a(r"  left=3mm,right=3mm,top=2mm,bottom=2mm,fonttitle=\bfseries,coltitle=black}")
    a(r"\newtcolorbox{databox}{colback=grisclaro,colframe=azul,")
    a(r"  title=\textbf{Datos construidos a partir del número de cuenta},drop fuzzy shadow}")
    a(r"\newtcolorbox{solutionbox}{colback=verdesuave,colframe=verdeoscuro,")
    a(r"  title=\textbf{Solución},drop fuzzy shadow}")
    a(r"\newtcolorbox{obsbox}{colback=rojoclaro,colframe=naranja!80!black,")
    a(r"  title=\textbf{Observación},drop fuzzy shadow}")
    a(r"\newtcolorbox{conceptbox}{colback=amarillosuave,colframe=dorado!80!black,")
    a(r"  title=\textbf{Concepto clave},drop fuzzy shadow,")
    a(r"  before={\par\smallskip\noindent},after={\par\smallskip}}")
    a(r"\newcommand{\R}{\mathbb{R}}")
    a(r"\newcommand{\Ker}{\operatorname{Ker}}")
    a(r"\newcommand{\Ima}{\operatorname{Im}}")
    a(r"\newcommand{\Span}{\operatorname{span}}")
    a(r"\newcommand{\rank}{\operatorname{rank}}")
    a(r"\newcommand{\nul}{\operatorname{nul}}")
    a(r"\newcommand{\RREF}{\operatorname{RREF}}")
    a(r"\tikzset{")
    a(r"  eje/.style={->, thick, black},")
    a(r"  vecA/.style={->, line width=1.4pt, morado},")
    a(r"  vecB/.style={->, line width=1.4pt, naranja},")
    a(r"  vecC/.style={->, line width=1.4pt, verdeoscuro},")
    a(r"  planito/.style={fill=azul!14,draw=azul,line width=0.9pt,fill opacity=0.55},")
    a(r"  elegantbox/.style={draw=azul,rounded corners=2mm,fill=azulclaro,inner sep=6pt,drop shadow},")
    a(r"  flowbox/.style={draw=azul,rounded corners=2mm,fill=azulclaro,inner sep=5pt,")
    a(r"    minimum height=8mm,text centered,drop shadow},")
    a(r"  flowboxW/.style={draw=azul,rounded corners=2mm,fill=azulclaro,inner sep=5pt,")
    a(r"    minimum height=8mm,text width=#1,text centered,drop shadow},")
    a(r"  myarrow/.style={->, thick, >=Latex, azul}")
    a(r"}")
    a()
    a(r"\begin{document}")
    a(r"\begin{center}")
    a(r"  {\Large \textbf{Universidad / Facultad}}\\[0.4em]")
    a(r"  {\large \textbf{Solucionario de Álgebra Lineal}}\\[0.4em]")
    a(r"  {\large Espacios vectoriales, subespacios, kernel, imagen, bases, rango y nulidad}\\[0.8em]")
    a(r"  {\large \textbf{Versión profesor con soluciones}}")
    a(r"\end{center}")
    a(r"\vspace{0.7em}")
    a(r"\begin{tabularx}{\textwidth}{|X|X|}")
    a(r"\hline")
    a(r"\textbf{Nombre del Profesor:}~\hrulefill & \textbf{Grupo:}~\hrulefill \\[0.8em]")
    a(r"\hline")
    a(r"\textbf{Semestre:}~\hrulefill & \textbf{Fecha:}~\hrulefill \\[0.8em]")
    a(r"\hline")
    a(r"\end{tabularx}")
    a(r"\vspace{1em}")
    a(r"\begin{databox}")
    a("A partir del número de cuenta $" + cuenta + "$, definimos:")
    a(r"\[ A=" + m(A) + r",\quad u=" + m(u) + r",\quad v=" + m(v) + r",\quad w=" + m(w) + r" \]")
    a(r"\end{databox}")
    a(r"\hrule\vspace{1em}")
    a(r"\section*{Soluciones}")

    a(r"\subsection*{Ejercicio 1}")
    a(r"\[ W=\left\{(x,y,z)\in\R^3:\; " + m(eq_W) + r"=0\right\}. \]")

    a(r"\begin{solutionbox}")
    a(r"\textbf{a) $W$ es subespacio vectorial de $\R^3$.}")
    a(r"\begin{conceptbox}")
    a(r"$W\subseteq V$ es subespacio si contiene al origen, es cerrado bajo suma y bajo multiplicación por escalares.")
    a(r"\end{conceptbox}")
    a(r"\begin{enumerate}[label=\textbf{(\roman*)}]")
    a("  \\item \\textbf{Origen:} $" + str(A11) + "(0)+" + str(A12) + "(0)+" + str(A13) + r"(0)=0$, luego $(0,0,0)\in W$.")
    a("  \\item \\textbf{Suma:} $" + str(A11) + r"(x_1+x_2)+" + str(A12) + r"(y_1+y_2)+" + str(A13) + r"(z_1+z_2)=0$.")
    a("  \\item \\textbf{Escalar:} $" + str(A11) + r"(\lambda x)+" + str(A12) + r"(\lambda y)+" + str(A13) + r"(\lambda z)=\lambda(0)=0$.")
    a(r"\end{enumerate}")
    a(r"Por tanto, $W$ es subespacio de $\R^3$.")
    a(r"\end{solutionbox}")

    a(r"\begin{solutionbox}")
    a(r"\textbf{b) Base de $W$.}")
    a("Resolvemos $" + str(A11) + "x+" + str(A12) + "y+" + str(A13) + "z=0$. Despejando:")
    a(r"\[ x = -\frac{" + str(A12) + "}{" + str(A11) + r"}y - \frac{" + str(A13) + "}{" + str(A11) + r"}z. \]")
    a(r"\[ \boxed{\mathcal{B}_W = \left\{ " + m(base_W_1) + r",\; " + m(base_W_2) + r" \right\}.} \]")
    a(r"\textbf{c) Dimensión.}")
    a(r"\[ \boxed{\dim(W)=2.} \]")
    a(r"\end{solutionbox}")

    a(r"\subsection*{Ejercicio 2}")
    a(r"\[ u=" + m(u) + r",\quad v=" + m(v) + r",\quad w=" + m(w) + r". \]")
    a(r"\begin{solutionbox}")
    a(r"\textbf{a) Independencia lineal de $u$ y $v$.}")
    a(r"\[ " + m(M_uv) + r" \xrightarrow{\text{RREF}} " + m(M_uv_rref) + r". \]")
    a("Hay " + n_piv + " pivotes; $u$ y $v$ son \\textbf{" + indep_txt + "}.")
    a(r"\textbf{b) Base de $U$.}")
    a(r"\[ \boxed{\mathcal{B}_U = \left\{ " + m(u) + r",\; " + m(v) + r" \right\}.} \]")
    a(r"\textbf{c) Dimensión.} $\dim(U)=" + n_piv + "$.")
    a(r"\textbf{d) ¿$w\in U$?} El sistema " + w_txt)
    a(r"\[ \boxed{" + w_box + r"} \]")
    a(r"\end{solutionbox}")

    a(r"\subsection*{Ejercicio 3}")
    a(r"\[ A=" + m(A) + r". \]")
    a(r"\begin{solutionbox}")
    a(r"\[ A \xrightarrow{\text{Gauss-Jordan}} \RREF(A)=" + m(A_rref) + r". \]")
    a(r"\textbf{a) Rango:} $\rank(A)=" + str(rango_A) + r".$")
    a(r"\textbf{b) Columnas pivote:} columnas " + pivotes_txt + ".")
    a(r"\textbf{c) Base del espacio columna:}")
    a(r"\[ \mathcal{B}_{\Ima} = \left\{ " + cols(base_col_A) + r" \right\}. \]")
    a(r"\textbf{d) Base del espacio fila:}")
    a(r"\[ \mathcal{B}_{\text{Row}} = \left\{ " + rows_l(base_row_A) + r" \right\}. \]")
    a(r"\end{solutionbox}")

    a(r"\subsection*{Ejercicio 4}")
    a(r"\begin{solutionbox}")
    a(r"\[ \RREF(A)=" + m(A_rref) + r". \]")
    a(r"\[ \boxed{\mathcal{B}_{\Ker(T)}=\left\{ " + m(base_ker) + r" \right\},\qquad \nul(T)=" + str(nulidad_A) + r".} \]")
    a(r"\begin{conceptbox}")
    a(ker_geom)
    a(r"\end{conceptbox}")
    a(tikz_ker)
    a(r"\end{solutionbox}")

    a(r"\subsection*{Ejercicio 5}")
    a(r"\begin{solutionbox}")
    a(r"\[ \boxed{\mathcal{B}_{\Ima(T)}=\left\{ " + cols(base_col_A) + r" \right\},\qquad \rank(T)=" + str(rango_A) + r".} \]")
    a(r"\[ \rank(T)+\nul(T)=" + str(rango_A) + "+" + str(nulidad_A) + r"=3=\dim(\R^3). \]")
    a(r"\end{solutionbox}")

    a(r"\subsection*{Ejercicio 6}")
    a(r"\begin{solutionbox}")
    a(r"\textbf{Sistema I (compatible determinado).}")
    a(r"\[ A_1=" + m(A1_sys) + r",\qquad b=" + m(bI) + r". \]")
    a(r"\begin{alignat*}{3}")
    a(str(d6+1) + r"\,z &= " + str(bI_3) + r" &\quad\Rightarrow\quad z &&= " + m(z_val) + r",\\[4pt]")
    a(str(d5+1) + r"\,y + " + str(d6) + r"\,z &= " + str(bI_2) + r" &\quad\Rightarrow\quad y &&= " + m(y_val) + r",\\[4pt]")
    a(str(A11) + r"\,x + " + str(A12) + r"\,y + " + str(A13) + r"\,z &= " + str(bI_1) + r" &\quad\Rightarrow\quad x &&= " + m(x_val) + ".")
    a(r"\end{alignat*}")
    a(r"\[ \boxed{(x,y,z)=\left(" + m(x_val) + r",\;" + m(y_val) + r",\;" + m(z_val) + r"\right).} \]")
    a(r"\end{solutionbox}")

    a(r"\begin{solutionbox}")
    a(r"\textbf{Sistema II (compatible indeterminado).}")
    a(r"\[ b_{II}=" + m(b_II) + r". \]")
    a(r"\[ \boxed{(x,y,z)=" + m(sol_II) + r".} \]")
    a(r"\end{solutionbox}")

    a(r"\begin{solutionbox}")
    a(r"\textbf{Sistema III (incompatible).}")
    a(r"\[ b_{III}=" + m(sp.Matrix([bIII_1, bIII_2, bIII_3])) + r". \]")
    a("Se requeriría $b_3=b_1+b_2=" + str(bIII_sum) + "$, pero $b_3=" + str(bIII_3) + "$.")
    a(r"\[ \boxed{\text{No tiene solución.}} \]")
    a(r"\end{solutionbox}")

    a(r"\subsection*{Ejercicio 7}")
    a(r"\begin{solutionbox}")
    a(r"Las filas de $A$ satisfacen $R_3=R_1+R_2$. Verificación: $\det(A)=" + det_val + "$.")
    a(r"\begin{enumerate}[label=\alph*),leftmargin=*]")
    a(r"  \item $\det(A)=0$.")
    a(r"  \item $A$ no es invertible.")
    a(r"  \item $\Ker(T)\neq\{\mathbf{0}\}$.")
    a(r"  \item $Ax=b$ no siempre tiene solución única.")
    a(r"\end{enumerate}")
    a(r"\begin{conceptbox}")
    a(r"$\det(A)\neq0 \iff A \text{ es invertible } \iff \Ker(T)=\{\mathbf{0}\} \iff Ax=b \text{ tiene solución única para todo } b.$")
    a(r"\end{conceptbox}")
    a(r"\end{solutionbox}")

    a(r"\end{document}")
    return "\n".join(L)


# =============================================================================
# BOTÓN PRINCIPAL
# =============================================================================

if st.button("Generar Solución Completa, Gráficas y LaTeX"):
    if not (len(cuenta) == 7 and cuenta.isdigit()):
        st.error("Por favor ingresa exactamente 7 dígitos numéricos.")
        st.stop()

    d1, d2, d3, d4, d5, d6, d7 = [int(x) for x in cuenta]

    A11, A12, A13 = d1 + 1, d2, d3
    A21, A22, A23 = d4, d5 + 1, d6

    A = sp.Matrix([
        [A11, A12, A13],
        [A21, A22, A23],
        [A11 + A21, A12 + A22, A13 + A23]
    ])

    u = sp.Matrix([d7, d1, d2])
    v = sp.Matrix([d3, d4, d5])
    w = sp.Matrix([d6, d7, d1])

    x_s, y_s, z_s = sp.symbols('x y z')
    eq_W = A11 * x_s + A12 * y_s + A13 * z_s
    base_W_1 = sp.Matrix([-A12, A11, 0])
    base_W_2 = sp.Matrix([-A13, 0, A11])

    M_uv = u.row_join(v)
    M_uv_rref, pivotes_uv = M_uv.rref()
    son_indep = len(pivotes_uv) == 2
    w_en_u = M_uv.row_join(w).rank() == M_uv.rank()

    A_rref, pivotes_A = A.rref()
    rango_A = len(pivotes_A)
    base_col_A = [A[:, p] for p in pivotes_A]
    base_row_A = [A_rref[i, :] for i in range(rango_A)]

    kernel_A = A.nullspace()
    if kernel_A:
        raw = kernel_A[0]
        denoms = [sp.denom(x) for x in raw]
        lcm_d = denoms[0]
        for dd in denoms[1:]:
            lcm_d = sp.lcm(lcm_d, dd)
        base_ker = sp.simplify(raw * lcm_d)
        nulidad_A = len(kernel_A)
    else:
        base_ker = sp.Matrix([0, 0, 0])
        nulidad_A = 0

    A1_sys = sp.Matrix([
        [d1 + 1, d2, d3],
        [0, d5 + 1, d6],
        [0, 0, d6 + 1]
    ])

    bI = sp.Matrix([d7, d1 + d2, d3 + d4])
    bI_1, bI_2, bI_3 = int(bI[0]), int(bI[1]), int(bI[2])

    z_val = sp.Rational(bI_3, d6 + 1)
    y_val = sp.Rational(bI_2 - d6 * z_val, d5 + 1)
    x_val = sp.Rational(bI_1 - d2 * y_val - d3 * z_val, d1 + 1)

    bII_1, bII_2 = d7, d1 + d2
    bII_3 = bII_1 + bII_2
    b_II = sp.Matrix([bII_1, bII_2, bII_3])

    sol_II_set = list(sp.linsolve((A, b_II)))
    sol_II = sol_II_set[0] if sol_II_set else sp.Matrix([0, 0, 0])

    bIII_1, bIII_2, bIII_3 = d7, d1 + d2, d3 + d4
    bIII_sum = bIII_1 + bIII_2

    st.success("Solucionario generado para número de cuenta: **" + cuenta + "**")

    st.markdown('<div class="data-box">', unsafe_allow_html=True)
    st.markdown("**Datos construidos a partir del número de cuenta:**")
    st.latex(r"d_1 d_2 d_3 d_4 d_5 d_6 d_7 = " + cuenta)
    st.latex(
        r"A = " + sp.latex(A) +
        r",\quad u = " + sp.latex(u) +
        r",\quad v = " + sp.latex(v) +
        r",\quad w = " + sp.latex(w)
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Ejercicio 1
    st.markdown('<div class="section-header">Ejercicio 1</div>', unsafe_allow_html=True)
    st.latex(r"W = \left\{(x,y,z)\in\mathbb{R}^3:\; " + sp.latex(eq_W) + r" = 0\right\}")

    st.markdown("#### a) $W$ es subespacio vectorial")
    st.markdown("#### Criterio de subespacio")
    st.latex(
        r"W\subseteq V \text{ es subespacio } \iff \mathbf{0}\in W,\ "
        r"\text{cerrado bajo suma, y cerrado bajo escalar}"
    )

    st.markdown('<div class="solution-box">', unsafe_allow_html=True)
    st.markdown("**(i)** $" + str(A11) + "(0)+" + str(A12) + "(0)+" + str(A13) + r"(0)=0 \Rightarrow (0,0,0)\in W$")
    st.markdown("**(ii)** $" + str(A11) + r"(x_1+x_2)+" + str(A12) + r"(y_1+y_2)+" + str(A13) + r"(z_1+z_2)=0$")
    st.markdown("**(iii)** $" + str(A11) + r"(\lambda x)+" + str(A12) + r"(\lambda y)+" + str(A13) + r"(\lambda z)=\lambda(0)=0$")
    st.markdown("Los tres axiomas se cumplen: **$W$ es subespacio de $\\mathbb{R}^3$**.")
    st.markdown('</div>', unsafe_allow_html=True)

    col_t, col_g = st.columns([1, 1])

    with col_t:
        st.markdown("#### b) Base de $W$")
        st.markdown('<div class="solution-box">', unsafe_allow_html=True)
        st.latex(r"x = -\frac{" + str(A12) + "}{" + str(A11) + r"}y - \frac{" + str(A13) + "}{" + str(A11) + r"}z")
        st.latex(r"\mathcal{B}_W = \left\{ " + sp.latex(base_W_1) + r",\; " + sp.latex(base_W_2) + r" \right\}")
        st.markdown("#### c) Dimensión")
        st.latex(r"\dim(W) = 3 - 1 = 2")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_g:
        st.plotly_chart(
            plot_plane_with_vectors(
                (A11, A12, A13),
                [base_W_1, base_W_2],
                ['b₁', 'b₂'],
                "Plano W: " + str(A11) + "x+" + str(A12) + "y+" + str(A13) + "z=0"
            ),
            use_container_width=True
        )

    st.markdown("---")

    # Ejercicio 2
    st.markdown('<div class="section-header">Ejercicio 2</div>', unsafe_allow_html=True)
    st.latex(r"u=" + sp.latex(u) + r",\quad v=" + sp.latex(v) + r",\quad w=" + sp.latex(w))

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("#### a) Independencia lineal")
        st.markdown('<div class="solution-box">', unsafe_allow_html=True)
        st.latex(sp.latex(M_uv) + r" \xrightarrow{\text{RREF}} " + sp.latex(M_uv_rref))
        res = "**LI**" if son_indep else "**LD**"
        st.markdown(str(len(pivotes_uv)) + " pivotes → $u$ y $v$ son " + res + ".")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("#### b) Base de $U$   c) Dimensión")
        st.markdown('<div class="solution-box">', unsafe_allow_html=True)
        st.latex(
            r"\mathcal{B}_U = \left\{ " + sp.latex(u) + r",\; " + sp.latex(v) +
            r" \right\},\quad \dim(U)=" + str(len(pivotes_uv))
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown("#### d) ¿$w \\in U$?")
        st.markdown('<div class="solution-box">', unsafe_allow_html=True)
        for i in range(3):
            st.latex(str(u[i]) + r"\,\alpha + " + str(v[i]) + r"\,\beta = " + str(w[i]))
        if w_en_u:
            st.markdown("Sistema **compatible** → $w \\in U$.")
            st.latex(r"\boxed{w \in U}")
        else:
            st.markdown("Sistema **incompatible** → $w \\notin U$.")
            st.latex(r"\boxed{w \notin U}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.plotly_chart(
            plot_vectors_3d(
                [u, v, w],
                ['u', 'v', 'w'],
                "Vectores u, v, w",
                plane_vecs=[u, v] if son_indep else None
            ),
            use_container_width=True
        )

    st.markdown("---")

    # Ejercicio 3
    st.markdown('<div class="section-header">Ejercicio 3: Reducción Gauss-Jordan de A</div>', unsafe_allow_html=True)
    st.markdown('<div class="solution-box">', unsafe_allow_html=True)
    st.latex(r"A = " + sp.latex(A) + r" \xrightarrow{\text{Gauss-Jordan}} \operatorname{RREF}(A) = " + sp.latex(A_rref))

    col3a, col3b = st.columns(2)
    with col3a:
        st.markdown("**a) Rango:** $\\operatorname{rank}(A) = " + str(rango_A) + "$")
        st.markdown("**b) Columnas pivote:** " + ", ".join([str(p + 1) for p in pivotes_A]))
    with col3b:
        st.markdown("**c) Base espacio columna:**")
        st.latex(r"\mathcal{B}_{\operatorname{Im}} = \left\{ " + r",\; ".join([sp.latex(c) for c in base_col_A]) + r" \right\}")
        st.markdown("**d) Base espacio fila:**")
        st.latex(r"\mathcal{B}_{\text{Row}} = \left\{ " + r",\; ".join([sp.latex(r) for r in base_row_A]) + r" \right\}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Ejercicio 4
    st.markdown('<div class="section-header">Ejercicio 4: Kernel de T</div>', unsafe_allow_html=True)
    col4t, col4g = st.columns([1, 1])

    with col4t:
        st.markdown('<div class="solution-box">', unsafe_allow_html=True)
        st.latex(r"\operatorname{RREF}(A) = " + sp.latex(A_rref))
        st.latex(
            r"\mathcal{B}_{\operatorname{Ker}(T)} = \left\{ " + sp.latex(base_ker) +
            r" \right\},\qquad \operatorname{nul}(T) = " + str(nulidad_A)
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="concept-box">', unsafe_allow_html=True)
        st.markdown("- nul = 0: kernel trivial")
        st.markdown("- nul = 1: recta por el origen")
        st.markdown("- nul = 2: plano por el origen")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4g:
        if nulidad_A == 1:
            st.plotly_chart(plot_kernel_line(base_ker, "Recta del Kernel"), use_container_width=True)
        else:
            st.info("Kernel trivial." if nulidad_A == 0 else "Kernel de dimensión 2.")

    st.markdown("---")

    # Ejercicio 5
    st.markdown('<div class="section-header">Ejercicio 5: Imagen y Teorema Rango-Nulidad</div>', unsafe_allow_html=True)
    st.markdown('<div class="solution-box">', unsafe_allow_html=True)
    st.latex(
        r"\mathcal{B}_{\operatorname{Im}(T)} = \left\{ " +
        r",\; ".join([sp.latex(c) for c in base_col_A]) +
        r" \right\},\qquad \operatorname{rank}(T) = " + str(rango_A)
    )
    st.latex(
        r"\operatorname{rank}(T) + \operatorname{nul}(T) = " +
        str(rango_A) + " + " + str(nulidad_A) + r" = 3 = \dim(\mathbb{R}^3)"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Ejercicio 6
    st.markdown('<div class="section-header">Ejercicio 6: Sistemas Lineales</div>', unsafe_allow_html=True)

    st.markdown("### Sistema I — Compatible Determinado")
    st.markdown('<div class="solution-box">', unsafe_allow_html=True)
    st.latex(r"A_1 = " + sp.latex(A1_sys) + r",\quad b = " + sp.latex(bI))
    st.latex(
        r"\begin{aligned}" +
        str(d6 + 1) + r"\,z &= " + str(bI_3) + r" \Rightarrow z = " + sp.latex(z_val) + r"\\" +
        str(d5 + 1) + r"\,y + " + str(d6) + r"\,z &= " + str(bI_2) + r" \Rightarrow y = " + sp.latex(y_val) + r"\\" +
        str(A11) + r"\,x + " + str(A12) + r"\,y + " + str(A13) + r"\,z &= " + str(bI_1) + r" \Rightarrow x = " + sp.latex(x_val) +
        r"\end{aligned}"
    )
    st.latex(r"\boxed{(x,y,z)=\left(" + sp.latex(x_val) + r",\;" + sp.latex(y_val) + r",\;" + sp.latex(z_val) + r"\right)}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Sistema II — Compatible Indeterminado")
    st.markdown('<div class="solution-box">', unsafe_allow_html=True)
    st.latex(r"b_{II} = " + sp.latex(b_II))
    st.markdown("$R_3=R_1+R_2$ y $b_3=" + str(bII_3) + "=b_1+b_2$ → ecuación redundante:")
    st.latex(r"\boxed{(x,y,z) = " + sp.latex(sol_II) + "}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Sistema III — Incompatible")
    st.markdown('<div class="solution-box">', unsafe_allow_html=True)
    st.latex(r"b_{III} = " + sp.latex(sp.Matrix([bIII_1, bIII_2, bIII_3])))
    st.markdown(
        "Se requeriría $b_3=" + str(bIII_sum) +
        "$, pero $b_3=" + str(bIII_3) + "\\neq " + str(bIII_sum) + "$ → sin solución."
    )
    st.latex(r"\boxed{\text{No tiene solución. (Sistema incompatible)}}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Ejercicio 7
    st.markdown('<div class="section-header">Ejercicio 7: Consecuencias de filas linealmente dependientes</div>', unsafe_allow_html=True)
    st.markdown('<div class="solution-box">', unsafe_allow_html=True)
    st.latex(r"\det(A) = " + sp.latex(A.det()))
    st.markdown("""
- **a)** $\\det(A)=0$
- **b)** $A$ **no es invertible**
- **c)** $\\operatorname{Ker}(T)\\neq\\{\\mathbf{0}\\}$
- **d)** la unicidad de soluciones falla
""")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("#### Teorema fundamental")
    st.latex(
        r"\det(A)\neq0 \iff A \text{ es invertible } \iff \operatorname{Ker}(T)=\{\mathbf{0}\} \iff Ax=b \text{ tiene solución única para todo } b"
    )

    st.markdown("---")

    latex_code = build_latex(
        cuenta, d1, d2, d3, d4, d5, d6, d7,
        A, u, v, w,
        A11, A12, A13,
        base_W_1, base_W_2,
        M_uv, M_uv_rref, pivotes_uv, son_indep, w_en_u,
        A_rref, pivotes_A, rango_A, base_col_A, base_row_A,
        base_ker, nulidad_A,
        A1_sys, bI, bI_1, bI_2, bI_3,
        x_val, y_val, z_val,
        b_II, bII_1, bII_2, bII_3, sol_II,
        bIII_1, bIII_2, bIII_3, bIII_sum,
        eq_W
    )

    st.subheader("Descargar archivo LaTeX para Overleaf")
    st.download_button(
        label="Descargar .tex para Overleaf",
        data=latex_code,
        file_name="Solucionario_" + cuenta + ".tex",
        mime="text/plain"
    )

    with st.expander("Ver código LaTeX completo"):
        st.code(latex_code, language='latex')
