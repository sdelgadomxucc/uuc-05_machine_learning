import streamlit as st

# -------------------------------------------------
# Bloc théorique (LaTeX) — version corrigée
# -------------------------------------------------

with st.expander("📘 Théorie, et formules", expanded=False):

    # --- PPH de base
    st.markdown("### Processus de Poisson homogène (PPH)")
    st.latex(r"""
        N(0)=0, \qquad N(t)-N(s)\sim \mathrm{Poisson}\!\big(\lambda (t-s)\big), \qquad 0\le s<t.
    """)
    st.latex(r"""
        \mathbb{P}\!\big[N(t)=k\big] \;=\; e^{-\lambda t}\,\frac{(\lambda t)^k}{k!}, \qquad k\in\mathbb{N}.
    """)
    st.latex(r"""
        W_i \sim \mathrm{Exp}(\lambda), \qquad
        f_W(w)=\lambda e^{-\lambda w}\ \ (w>0), \qquad
        \mathbb{E}[W_i]=\frac{1}{\lambda}, \quad \mathrm{Var}(W_i)=\frac{1}{\lambda^2}.
    """)
    st.latex(r"""
        \mathbb{E}[N(t)] = \lambda t, \qquad \mathrm{Var}(N(t)) = \lambda t.
    """)

    # --- Estimateur et IC
    st.markdown("### Estimateur du taux et intervalle de confiance exact")
    st.latex(r"""
        \ell(\lambda) \;=\; K\log\lambda \;-\; \lambda T \;-\; \log(K!) \quad\Longrightarrow\quad
        \frac{\partial \ell}{\partial \lambda}=\frac{K}{\lambda}-T=0 \ \Rightarrow\ 
        \widehat{\lambda}=\frac{K}{T}.
    """)
    st.latex(r"""
        \text{IC}_{95\%}(\lambda) \;=\;
        \left[\; \frac{1}{2T}\,\chi^2_{0.025,\,2K}\; ,\; \frac{1}{2T}\,\chi^2_{0.975,\,2K+2}\; \right].
    """)

    # --- PGF / MGF
    st.markdown("### Fonctions génératrices")
    st.latex(r"""
        G_{N(t)}(z)=\exp\!\{\lambda t (z-1)\}, \qquad
        M_{N(t)}(\theta)=\exp\!\{\lambda t (e^{\theta}-1)\}.
    """)

    # --- PPNH
    st.markdown("### Processus de Poisson non homogène (PPNH)")
    st.latex(r"""
        \Lambda(t) \;=\; \int_0^t \lambda(u)\,du, \qquad
        N(t)-N(s)\sim \mathrm{Poisson}\!\Big(\int_s^t \lambda(u)\,du\Big).
    """)
    st.latex(r"""
        \mathbb{P}\!\big[N(t)=k\big] = e^{-\Lambda(t)}\,\frac{\Lambda(t)^k}{k!},
        \qquad \mathbb{E}[N(t)] = \Lambda(t).
    """)




with st.expander("🧪 Exemples concrets"):
    st.markdown(r"""
    ### Exemples de PPH (homogènes)
    1. **Photons détectés** sous illumination stable (λ constant).
    2. **Appels entrants** dans un centre d’appel (flux stationnaire).
    3. **Défauts** sur une fibre optique par km.

    ### Exemples de PPNH (non homogènes)
    1. **Trafic web diurne** : $\lambda(t)=\lambda_0 + A\max(0,\sin(2\pi t/24))$.
    2. **Arrivées aux urgences** : $λ(t)$ plus forte la nuit/week-end.
    3. **Après-chocs sismiques** : $\lambda(t)=\frac{K}{(c+t)^p}$.
    """)

# -------------------------------------------------
# -------------------------------------------------
# Introduction et exemples
# -------------------------------------------------
# Cette application Streamlit illustre la théorie et la simulation des processus de Poisson homogènes (PPH) et non homogènes (PPNH).
# Elle combine la rigueur mathématique avec une visualisation interactive.
#
# Exemples réels de PPH :
#   1. Comptage de photons détectés sous illumination stable : le flux lumineux est constant, donc λ est fixe.
#   2. Appels entrants dans un centre d’appel pendant une heure creuse : les arrivées sont aléatoires mais stationnaires.
#   3. Défauts sur une fibre optique par kilomètre : λ constant par unité de longueur.
#
# Exemples réels de PPNH :
#   1. Trafic web au cours de la journée : λ(t) périodique avec pics aux heures de pointe.
#   2. Arrivées aux urgences hospitalières : intensité plus forte la nuit et le week-end.
#   3. Activité sismique post-choc : λ(t) décroissante selon la loi d’Omori.
#
# Application Streamlit : Processus de Poisson (homogène et non homogène)
# -------------------------------------------------
# Détails mathématiques précis et simulation interactive
# -------------------------------------------------
# 1. Processus de Poisson homogène (PPH)
#    Un processus de Poisson homogène \( \{N(t), t \ge 0\} \) de taux \( \lambda > 0 \) satisfait :
#       (i) \( N(0) = 0 \)
#       (ii) Incréments indépendants : pour tout \( 0 \le s < t \), \( N(t) - N(s) \) est indépendant du passé.
#       (iii) Incréments stationnaires : \( N(t) - N(s) \sim \text{Poisson}(\lambda (t - s)) \).
#    Ainsi, le nombre d'événements pendant un intervalle de longueur \( t \) suit une loi de Poisson de paramètre \( \lambda t \).
#    De plus, les intertemps \( W_i = T_i - T_{i-1} \) sont i.i.d. \( \text{Exp}(\lambda) \), c’est-à-dire :
#        \[ f_W(w) = \lambda e^{-\lambda w}, \quad w > 0. \]
#    L'espérance et la variance des intertemps sont toutes deux égales à \( 1/\lambda \).
#
# 2. Processus de Poisson non homogène (PPNH)
#    Pour une intensité variable \( \lambda(t) \ge 0 \), on définit le processus \( N(t) \) tel que :
#        \[ \mathbb{E}[N(t)] = \int_0^t \lambda(u) \, du, \]
#    et les incréments sur des intervalles disjoints sont indépendants.
#    La distribution de \( N(t) - N(s) \) est Poissonienne avec paramètre \( \int_s^t \lambda(u) du. \)
#    Une méthode de simulation efficace est **l’amincissement** :
#       - on choisit une borne supérieure \( \lambda_{\max} \ge \lambda(t) \);
#       - on génère un PPH de taux \( \lambda_{\max} \);
#       - chaque événement à l’instant \( T_i \) est conservé avec probabilité \( \lambda(T_i)/\lambda_{\max} \).
#    Ce procédé produit une réalisation du PPNH désiré.
#
# 3. Estimateur du taux \( \lambda \)
#    Pour un PPH observé sur \([0, T]\) avec \( K = N(T) \) événements :
#        \[ \widehat{\lambda}_{EMV} = \frac{K}{T}. \]
#    Un intervalle de confiance exact à 95% pour \( \lambda \) est :
#        \[ \left[ \frac{1}{2T}\chi^2_{0.025, 2K}, \; \frac{1}{2T}\chi^2_{0.975, 2K + 2} \right]. \]
#
# 4. Propriétés :
#       - **Superposition :** la somme de deux PPH indépendants de taux \( \lambda_1 \) et \( \lambda_2 \) est un PPH de taux \( \lambda_1 + \lambda_2 \).
#       - **Amincissement :** en conservant chaque événement d’un PPH de taux \( \lambda \) avec probabilité \( p \), on obtient un PPH de taux \( p\lambda. \)
#       - **Loi de Little (files M/M/1) :** \( L = \lambda W, \; L_q = \lambda W_q, \; \rho = \lambda/\mu. \)
#
# -------------------------------------------------
# Démonstrations détaillées : superposition et amincissement
# -------------------------------------------------
# A. Superposition de PPH
#   Soient N1(t) et N2(t) deux PPH indépendants de taux lambda1 et lambda2. Posons N(t) = N1(t) + N2(t).
#   1) Lois des incréments. Pour 0 <= s < t : N(t)-N(s) = [N1(t)-N1(s)] + [N2(t)-N2(s)].
#      Or N1(t)-N1(s) ~ Poisson(lambda1*(t-s)) et N2(t)-N2(s) ~ Poisson(lambda2*(t-s)), indépendants.
#      La somme indépendante de lois de Poisson est Poisson de paramètre somme, donc N(t)-N(s) ~ Poisson((lambda1+lambda2)*(t-s)).
#   2) Incréments indépendants. Sur des intervalles disjoints, les vecteurs d'incréments des deux processus sont indépendants; la somme coordonnée par coordonnée conserve cette indépendance.
#   Conclusion : N est un PPH de taux lambda1+lambda2.
#   (Preuve alternative par fonctions génératrices : E[z^{N(t)}] = exp{ (lambda1+lambda2) t (z-1) }.)
#
# B. Amincissement d'un PPH
#   Soit N(t) un PPH de taux lambda. Chaque événement est conservé indépendamment avec probabilité p (marquage Bernoulli). On note N~(t) le nombre d'événements conservés.
#   1) Loi marginale. Conditionnellement à N(t)=k, N~(t) suit Binomiale(k,p). Par composition Poisson–Binomiale, en déconditionnant : N~(t) ~ Poisson(p*lambda*t).
#   2) Incréments indépendants. Le marquage Bernoulli indépendant point par point sur des intervalles disjoints préserve l'indépendance des sous-comptes.
#   3) Intertemps. Les instants conservés forment un PPH de taux p*lambda; les intertemps sont donc i.i.d. Exp(p*lambda).
#   Conclusion : N~ est un PPH de taux p*lambda.
#
# C. Versions non homogènes (PPNH)
#   1) Superposition : si les intensités sont lambda_i(t), la somme de processus indépendants est un PPNH d'intensité lambda(t) = sum_i lambda_i(t).
#   2) Amincissement : si la probabilité de conservation dépend du temps p(t), le processus aminci est un PPNH d'intensité p(t)*lambda(t).
#      Esquisse de preuve : conditionner sur le processus sous-jacent puis utiliser la propriété de marquage des PPNH.
#
# -------------------------------------------------
# Implémentation Streamlit (simulation interactive)
# -------------------------------------------------

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import poisson, expon, chi2

# ------------------------------
# Configuration de la page
# ------------------------------
st.set_page_config(page_title="Explorateur du processus de Poisson", page_icon="🟣", layout="wide")

st.markdown(
    """
    <style>
      .subtitle {font-size:0.95rem; color:#666;}
      .small {font-size:0.85rem; color:#666;}
      .stTabs [data-baseweb="tab-list"] { gap: 8px; }
      .stTabs [data-baseweb="tab"] { padding: 8px 14px; }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# Fonctions mathématiques
# ------------------------------

def simuler_pph(lam: float, T: float, seed: int | None = None):
    if lam <= 0 or T <= 0:
        return np.array([]), np.array([])
    rng = np.random.default_rng(seed)
    t, temps = 0.0, []
    while True:
        w = rng.exponential(1.0 / lam)
        t += w
        if t > T:
            break
        temps.append(t)
    temps = np.array(temps)
    inter = np.diff(np.insert(temps, 0, 0.0)) if temps.size else np.array([])
    return temps, inter


def simuler_ppnh(lam_max: float, T: float, lam_func, seed: int | None = None):
    if lam_max <= 0 or T <= 0:
        return np.array([]), np.array([])
    rng = np.random.default_rng(seed)
    t, temps = 0.0, []
    while t < T:
        w = rng.exponential(1.0 / lam_max)
        t += w
        if t > T:
            break
        if rng.uniform() <= lam_func(t) / lam_max:
            temps.append(t)
    temps = np.array(temps)
    inter = np.diff(np.insert(temps, 0, 0.0)) if temps.size else np.array([])
    return temps, inter


def escalier_evenements(temps: np.ndarray, T: float):
    if T <= 0:
        return np.array([0.0]), np.array([0])
    t_vals, n_vals, n = [0.0], [0], 0
    for s in temps:
        t_vals += [s, s]
        n_vals += [n, n + 1]
        n += 1
    t_vals.append(T)
    n_vals.append(n)
    return np.array(t_vals), np.array(n_vals)


def intervalle_confiance_poisson(k: int, T: float, alpha: float = 0.05):
    if T <= 0:
        return np.nan, np.nan
    lo = 0.0 if k == 0 else 0.5 * chi2.ppf(alpha / 2, 2 * k) / T
    hi = 0.5 * chi2.ppf(1 - alpha / 2, 2 * k + 2) / T
    return lo, hi

# ------------------------------
# Interface utilisateur
# ------------------------------
with st.sidebar:
    st.header("⚙️ Paramètres du modèle")
    lam = st.number_input("Taux λ", min_value=0.01, value=2.0, step=0.1)
    T = st.number_input("Horizon T", min_value=0.1, value=10.0, step=0.5)
    n_paths = st.slider("Nombre de trajectoires", 1, 20, 3)
    seed = st.number_input("Graine aléatoire", min_value=0, value=42, step=1)

    st.divider()
    st.caption("Processus non homogène")
    use_nhpp = st.checkbox("Activer le PPNH (amincissement)", value=False)
    base = st.number_input("λ₀ (base)", min_value=0.0, value=1.0, step=0.1)
    amp = st.number_input("Amplitude", min_value=0.0, value=1.0, step=0.1)
    freq = st.number_input("Fréquence (cycles / T)", min_value=0.0, value=1.0, step=0.1)
    lam_max = base + amp

lam_func = lambda t: base + amp * np.maximum(0.0, np.sin(2 * np.pi * freq * t / T))

# ------------------------------
# Graphiques et résultats
# ------------------------------
st.title("🟣 Explorateur du processus de Poisson")
st.markdown("<div class='subtitle'>Visualisation mathématique et statistique du processus de Poisson homogène et non homogène.</div>", unsafe_allow_html=True)

fig = go.Figure()
for i in range(n_paths):
    temps, _ = simuler_pph(lam, T, seed + i)
    t_vals, n_vals = escalier_evenements(temps, T)
    fig.add_trace(go.Scatter(x=t_vals, y=n_vals, mode="lines", line_shape="hv", name=f"Trajectoire {i+1}"))
fig.update_layout(title=f"Trajectoires simulées : PPH (λ={lam:.2f}, T={T:.2f})", xaxis_title="t", yaxis_title="N(t)")
st.plotly_chart(fig, use_container_width=True)

_, inter = simuler_pph(lam, T, seed)
if inter.size:
    fig2 = px.histogram(pd.DataFrame({'w': inter}), x='w', histnorm='probability density')
    x_line = np.linspace(0, inter.max() * 1.2, 300)
    fig2.add_trace(go.Scatter(x=x_line, y=lam * np.exp(-lam * x_line), mode='lines', name='Exp(λ)'))
    fig2.update_layout(title="Distribution des intertemps : comparaison à Exp(λ)", xaxis_title="w", yaxis_title="densité")
    st.plotly_chart(fig2, use_container_width=True)

M = 2000
rng = np.random.default_rng(seed)
k_samp = rng.poisson(lam * T, size=M)
vc = pd.DataFrame({'K': k_samp}).value_counts().reset_index(name='freq').sort_values('K')
fig3 = go.Figure()
fig3.add_bar(x=vc['K'], y=vc['freq']/M, name='Empirique')
fig3.add_scatter(x=np.arange(vc['K'].max()+1), y=poisson.pmf(np.arange(vc['K'].max()+1), lam*T), mode='lines+markers', name='Poisson(λT)')
fig3.update_layout(title=f"Distribution du nombre d’événements N(T)", xaxis_title="k", yaxis_title="P(N(T)=k)")
st.plotly_chart(fig3, use_container_width=True)

if use_nhpp:
    temps_n, inter_n = simuler_ppnh(lam_max, T, lam_func, seed)
    t_vals, n_vals = escalier_evenements(temps_n, T)
    fig4 = go.Figure(go.Scatter(x=t_vals, y=n_vals, mode='lines', line_shape='hv'))
    fig4.update_layout(title="Processus de Poisson non homogène (méthode d’amincissement)", xaxis_title="t", yaxis_title="N(t)")
    st.plotly_chart(fig4, use_container_width=True)
    st.caption(f"{temps_n.size} événements simulés, λ_max = {lam_max:.2f}.")

st.markdown("""---\n**Notes mathématiques :** les trajectoires simulées vérifient les propriétés fondamentales du processus de Poisson. L’intervalle de confiance est basé sur la distribution du chi-deux et les intertemps suivent une loi exponentielle Exp(λ).""")

