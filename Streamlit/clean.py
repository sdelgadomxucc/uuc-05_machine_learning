import numpy as np
import pandas as pd
import yfinance as yf
from scipy.linalg import eigh

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# 0. Configuración de la página + tema oscuro
# =====================================================

st.set_page_config(
    page_title="Clean Markowitz – IPC & Dow",
    layout="wide",
)

# CSS simple para fondo oscuro y texto claro
st.markdown(
    """
    <style>
    .stApp {
        background-color: #111111;
        color: #f5f5f5;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Markowitz con IPC y Dow")
st.markdown(
    r"""
Esta app ilustra:

- Portafolio tangente de Markowitz (plug-in) vs portafolio con **covarianza limpiada** (RIE).  
- Comparación **in-sample vs out-of-sample**.  
- Uso de **muchos activos** (IPC .MX + varios del Dow) en régimen de matrices aleatorias.  

Siguiendo el marco de Bun–Bouchaud–Potters (2017)
"""
)

# =====================================================
# 1. Universo de activos: IPC (.MX) + Dow
# =====================================================

TICKERS_IPC = [
    "AMXB.MX",      # América Móvil
    "BIMBOA.MX",    # Grupo Bimbo
    "ASURB.MX",     # ASUR
    "AC.MX",        # Arca Continental
    "ALFAA.MX",     # Alfa
    "ALSEA.MX",     # Alsea
    "GAPB.MX",      # GAP
    "VOLARA.MX",    # Volaris
    "CEMEXCPO.MX",  # Cemex
    "GMEXICOB.MX",  # Grupo México
    "WALMEX.MX",    # Walmart de México
    "KIMBERA.MX",   # Kimberly-Clark México
    "FEMSAUBD.MX",  # FEMSA
    "GFNORTEO.MX",  # Banorte
    "BBAJIOO.MX",   # Banco del Bajío
    "GENTERA.MX",   # Gentera
    "TLEVISACPO.MX",# Televisa
    "GFINBURO.MX",  # Inbursa
    "GRUMAB.MX",    # Gruma
    "MEGACPO.MX",   # Megacable
    "OMAB.MX",      # OMA
    "PE&OLES.MX",   # Peñoles
]

TICKERS_DOW = [
    "AAPL", "MSFT", "GS", "CAT", "JPM", "V", "KO", "MCD",
    "UNH", "PG", "DIS", "HD", "NKE", "IBM", "CSCO",
    "WMT", "TRV", "AMZN", "NVDA"
]

ALL_TICKERS = TICKERS_IPC + TICKERS_DOW

# =====================================================
# 2. Sidebar: controles de usuario
# =====================================================

st.sidebar.header("Parámetros del experimento")

start_date = st.sidebar.date_input("Fecha de inicio", value=pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("Fecha de fin", value=pd.to_datetime("2025-01-01"))

split_date = st.sidebar.date_input(
    "Fecha de corte (in-sample / out-of-sample)",
    value=pd.to_datetime("2018-01-01"),
)

rf_annual = st.sidebar.slider("Tasa libre de riesgo anual (%)", 0.0, 10.0, 3.0, 0.1)
RF_ANNUAL = rf_annual / 100.0
RF_DAILY = (1 + RF_ANNUAL) ** (1 / 252) - 1

eta = st.sidebar.slider("ETA (imaginario para g_E)", 1e-4, 5e-2, 1e-3, format="%.4f")

st.sidebar.markdown("---")

st.sidebar.markdown("### Selección de activos")

universe_size = st.sidebar.slider(
    "Número de activos (tomados de la lista IPC + Dow)",
    min_value=4,
    max_value=len(ALL_TICKERS),
    value=20,
    step=1,
)

st.sidebar.caption("Se toman los primeros *n* tickers de la lista predefinida.")
run_button = st.sidebar.button("Ejecutar experimento")

# =====================================================
# 3. Funciones auxiliares
# =====================================================

@st.cache_data(show_spinner=True)
def download_prices_and_returns(tickers, start, end):
    """
    Descarga precios ajustados de Yahoo Finance y construye rendimientos simples.
    - Elimina columnas con todo NaN (tickers que no bajan).
    - Considera sólo fechas en las que todos los tickers restantes tienen dato.
    """
    data = yf.download(tickers, start=start, end=end)["Close"]

    if isinstance(data, pd.Series):
        data = data.to_frame()

    # Eliminar columnas completamente vacías
    data = data.dropna(axis=1, how="all")

    # Mantener sólo tickers que realmente tienen datos
    cols = [c for c in tickers if c in data.columns]
    data = data[cols]

    # Rendimientos simples diarios y sincronización
    returns = data.pct_change().dropna()

    return data, returns


def tangency_portfolio(mu, Sigma, rf):
    excess = mu - rf
    inv_Sigma = np.linalg.inv(Sigma)
    num = inv_Sigma @ excess
    den = np.ones_like(excess) @ num
    w = num / den
    return w


def portfolio_performance(w, mu, Sigma, rf):
    mu_p = float(w @ mu)
    var_p = float(w @ Sigma @ w)
    sigma_p = np.sqrt(var_p)
    sr = (mu_p - rf) / sigma_p if sigma_p > 0 else np.nan
    return mu_p, sigma_p, sr


def realized_sharpe(w, rets, rf):
    rp = rets.values @ w
    rp_mean = rp.mean()
    rp_std = rp.std(ddof=1)
    sr = (rp_mean - rf) / rp_std if rp_std > 0 else np.nan
    return rp_mean, rp_std, sr


def rie_clean_covariance(Sigma, q, eta=1e-3):
    """
    Limpia la covarianza Sigma (NxN) usando la fórmula tipo Bun–Bouchaud–Potters:
        tilde_lambda_i = lambda_i / |1 - q + q lambda_i g_E(lambda_i - i eta)|^2
    con
        g_E(z) = (1/N) sum_j 1/(z - lambda_j).
    """
    lambdas, U = eigh(Sigma)
    N = Sigma.shape[0]

    lambdas = lambdas.real
    cleaned_lambdas = np.zeros_like(lambdas)

    for i, lam in enumerate(lambdas):
        z = lam - 1j * eta
        g = np.mean(1.0 / (z - lambdas))
        denom = 1.0 - q + q * lam * g
        cleaned_lambdas[i] = lam / (np.abs(denom) ** 2)

    Lambda_clean = np.diag(cleaned_lambdas)
    Sigma_clean = U @ Lambda_clean @ U.T
    return Sigma_clean, lambdas, cleaned_lambdas


# =============== Markowitz Efficient Frontier ==================

def efficient_frontier(mu, Sigma, n_points=50):
    """
    Frontera eficiente (sin restricciones de no short-selling),
    fully invested: 1^T w = 1, objetivo: variar retorno objetivo r.
    Devuelve arrays (sigma, mu) para la frontera.
    """
    mu = np.asarray(mu)
    Sigma_inv = np.linalg.inv(Sigma)
    ones = np.ones_like(mu)

    A = ones @ (Sigma_inv @ ones)
    B = ones @ (Sigma_inv @ mu)
    C = mu @ (Sigma_inv @ mu)

    # Rango de retornos objetivo: un poco más allá de [min(mu), max(mu)]
    r_min = float(mu.min())
    r_max = float(mu.max())
    r_vals = np.linspace(r_min, r_max, n_points)

    sigmas = []
    mus = []

    for r in r_vals:
        # Resolver [A B; B C] [α; β] = [1; r]
        M = np.array([[A, B], [B, C]])
        b = np.array([1.0, r])
        alpha, beta = np.linalg.solve(M, b)
        w = alpha * (Sigma_inv @ ones) + beta * (Sigma_inv @ mu)
        mu_p = float(w @ mu)
        var_p = float(w @ Sigma @ w)
        sigma_p = np.sqrt(var_p)
        sigmas.append(sigma_p)
        mus.append(mu_p)

    return np.array(sigmas), np.array(mus)


def plot_efficient_frontier(mu, Sigma_in, Sigma_rie, rf, w_tan_sample, w_tan_rie):
    """
    Gráfico interactivo de la frontera de Markowitz para:
    - Σ_in muestral (plug-in),
    - Σ_in limpiada (RIE),
    resaltando los portafolios tangentes.
    """
    # Fronteras
    sig_sample, mu_sample = efficient_frontier(mu, Sigma_in, n_points=80)
    sig_rie, mu_rie = efficient_frontier(mu, Sigma_rie, n_points=80)

    # Puntos tangentes
    mu_tan_sample, sig_tan_sample, sr_tan_sample = portfolio_performance(
        w_tan_sample, mu, Sigma_in, rf
    )
    mu_tan_rie, sig_tan_rie, sr_tan_rie = portfolio_performance(
        w_tan_rie, mu, Sigma_rie, rf
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=sig_sample,
            y=mu_sample,
            mode="lines",
            name="Frontera plug-in (Σ_in)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sig_rie,
            y=mu_rie,
            mode="lines",
            name="Frontera RIE (Σ_in limpia)",
        )
    )

    # Tangency points
    fig.add_trace(
        go.Scatter(
            x=[sig_tan_sample],
            y=[mu_tan_sample],
            mode="markers",
            name="Tangente plug-in",
            marker=dict(size=10, symbol="star"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[sig_tan_rie],
            y=[mu_tan_rie],
            mode="markers",
            name="Tangente RIE",
            marker=dict(size=10, symbol="star"),
        )
    )

    fig.update_layout(
        title="Frontera eficiente de Markowitz: plug-in vs RIE",
        xaxis_title="Riesgo (σ)",
        yaxis_title="Retorno esperado (μ)",
        template="plotly_dark",
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
    )
    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# 4. Gráficas (con template oscuro)
# =====================================================

def plot_price_series(prices):
    fig = px.line(
        prices,
        x=prices.index,
        y=prices.columns,
        title="Precios ajustados (IPC .MX + componentes del Dow)",
        labels={"value": "Precio", "variable": "Ticker", "index": "Fecha"},
        template="plotly_dark",
    )
    fig.update_layout(plot_bgcolor="#111111", paper_bgcolor="#111111")
    st.plotly_chart(fig, use_container_width=True)


def plot_returns_series(returns_in, returns_out):
    # Para no saturar, mostrar primeros 6 tickers
    subset_cols = returns_in.columns[:6]

    df_in = returns_in[subset_cols].copy()
    df_in["window"] = "in-sample"
    df_out = returns_out[subset_cols].copy()
    df_out["window"] = "out-of-sample"
    df = pd.concat([df_in, df_out], axis=0)

    for col in subset_cols:
        fig = px.line(
            df,
            x=df.index,
            y=col,
            color="window",
            title=f"Rendimientos diarios {col}: in-sample vs out-of-sample",
            labels={"value": "Rendimiento diario", "index": "Fecha"},
            template="plotly_dark",
        )
        fig.update_layout(plot_bgcolor="#111111", paper_bgcolor="#111111")
        st.plotly_chart(fig, use_container_width=True)


def plot_eigenvalues_comparison(lambdas_raw, lambdas_clean):
    idx = np.arange(len(lambdas_raw))
    df = pd.DataFrame({
        "index": np.concatenate([idx, idx]),
        "eigenvalue": np.concatenate([lambdas_raw, lambdas_clean]),
        "type": ["Muestral"] * len(lambdas_raw) + ["RIE"] * len(lambdas_clean),
    })

    fig = px.bar(
        df,
        x="index",
        y="eigenvalue",
        color="type",
        barmode="group",
        title="Eigenvalores de Σ_in: muestral vs RIE",
        labels={"index": "Índice de eigenvalor", "eigenvalue": "Valor propio"},
        template="plotly_dark",
    )
    fig.update_layout(plot_bgcolor="#111111", paper_bgcolor="#111111")
    st.plotly_chart(fig, use_container_width=True)


def plot_weights_comparison(tickers, w_sample, w_rie, top_k=20):
    order = np.argsort(-np.abs(w_rie))[:top_k]
    tickers_sel = [tickers[i] for i in order]
    w_sample_sel = w_sample[order]
    w_rie_sel = w_rie[order]

    df = pd.DataFrame({
        "ticker": np.tile(tickers_sel, 2),
        "weight": np.concatenate([w_sample_sel, w_rie_sel]),
        "type": ["Plug-in"] * len(tickers_sel) + ["RIE"] * len(tickers_sel),
    })

    fig = px.bar(
        df,
        x="ticker",
        y="weight",
        color="type",
        barmode="group",
        title=f"Pesos del portafolio tangente (top {top_k} por |w_RIE|)",
        labels={"weight": "Peso", "ticker": "Ticker"},
        template="plotly_dark",
    )
    fig.update_layout(plot_bgcolor="#111111", paper_bgcolor="#111111")
    st.plotly_chart(fig, use_container_width=True)


def plot_sharpe_comparison(sr_in_sample, sr_out_sample, sr_in_rie, sr_out_rie):
    methods = ["Plug-in", "Plug-in", "RIE", "RIE"]
    window = ["in-sample", "out-of-sample", "in-sample", "out-of-sample"]
    sr_vals = [sr_in_sample, sr_out_sample, sr_in_rie, sr_out_rie]

    df = pd.DataFrame({
        "Método": methods,
        "Ventana": window,
        "Sharpe": sr_vals,
    })

    fig = px.bar(
        df,
        x="Método",
        y="Sharpe",
        color="Ventana",
        barmode="group",
        title="Comparación de Sharpe: plug-in vs RIE (in/out-of-sample)",
        template="plotly_dark",
    )
    fig.update_layout(plot_bgcolor="#111111", paper_bgcolor="#111111")
    st.plotly_chart(fig, use_container_width=True)


def plot_risk_gap(
    R_in2_emp_plug, R_out2_emp_plug, R_in2_theo_plug, R_out2_theo_plug,
    R_in2_emp_rie, R_out2_emp_rie, R_in2_theo_rie, R_out2_theo_rie
):
    """
    Gráfico de barras comparando:
    - R_in^2 y R_out^2 empíricos vs teóricos (BBP)
    - Para ambos portafolios: plug-in y RIE
    """
    portfolios = []
    which_list = []
    kind_list = []
    values = []

    # Plug-in
    portfolios += ["Plug-in", "Plug-in", "Plug-in", "Plug-in"]
    which_list += ["R_in^2", "R_out^2", "R_in^2", "R_out^2"]
    kind_list += ["Empírico", "Empírico", "Teórico (BBP)", "Teórico (BBP)"]
    values += [R_in2_emp_plug, R_out2_emp_plug, R_in2_theo_plug, R_out2_theo_plug]

    # RIE
    portfolios += ["RIE", "RIE", "RIE", "RIE"]
    which_list += ["R_in^2", "R_out^2", "R_in^2", "R_out^2"]
    kind_list += ["Empírico", "Empírico", "Teórico (BBP)", "Teórico (BBP)"]
    values += [R_in2_emp_rie, R_out2_emp_rie, R_in2_theo_rie, R_out2_theo_rie]

    df = pd.DataFrame({
        "Portafolio": portfolios,
        "Magnitud": which_list,
        "Tipo": kind_list,
        "Valor": values,
    })

    fig = px.bar(
        df,
        x="Portafolio",
        y="Valor",
        color="Tipo",
        barmode="group",
        facet_col="Magnitud",
        title="Gap ",
        labels={"Valor": "R^2 (varianza del portafolio)"},
        template="plotly_dark",
    )
    fig.update_layout(plot_bgcolor="#111111", paper_bgcolor="#111111")
    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# 5. Lógica principal
# =====================================================

if run_button:
    st.subheader("Resultados del experimento")

    # 1) Selección de universo
    universe_tickers = ALL_TICKERS[:universe_size]
    st.markdown(f"**Tickers en el universo ({len(universe_tickers)}):**")
    st.write(universe_tickers)

    # 2) Descarga de datos
    with st.spinner("Descargando precios y calculando rendimientos..."):
        prices, returns = download_prices_and_returns(
            universe_tickers,
            start=pd.to_datetime(start_date),
            end=pd.to_datetime(end_date),
        )

    if returns.empty:
        st.error("No se obtuvieron rendimientos (quizá el rango de fechas es muy corto o sin datos).")
        st.stop()

    # 3) Split in-sample / out-of-sample
    split_ts = pd.to_datetime(split_date)
    returns_in = returns[returns.index < split_ts].copy()
    returns_out = returns[returns.index >= split_ts].copy()

    if returns_in.empty or returns_out.empty:
        st.error("La ventana in-sample o out-of-sample quedó vacía. Ajusta la fecha de corte.")
        st.stop()

    N = returns_in.shape[1]
    T_in = returns_in.shape[0]
    T_out = returns_out.shape[0]
    q = N / T_in

    st.markdown(
        f"""
- Número de activos $N$: **{N}**  
- Observaciones in-sample $T_{{in}}$: **{T_in}**  
- Observaciones out-of-sample $T_{{out}}$: **{T_out}**  
- Aspect ratio $q = N/T_{{in}}$: **{q:.4f}**
"""
    )

    # 4) Estimadores in-sample y out-sample
    mu_in = returns_in.mean().values
    Sigma_in = np.cov(returns_in.values, rowvar=False, ddof=1)

    mu_out = returns_out.mean().values
    Sigma_out = np.cov(returns_out.values, rowvar=False, ddof=1)

    # 5) Covarianza limpiada RIE
    Sigma_rie, lambdas_raw, lambdas_clean = rie_clean_covariance(Sigma_in, q, eta=eta)

    # 6) Portafolios tangentes
    try:
        w_tan_sample = tangency_portfolio(mu_in, Sigma_in, RF_DAILY)
        w_tan_rie = tangency_portfolio(mu_in, Sigma_rie, RF_DAILY)
    except np.linalg.LinAlgError:
        st.error("Matriz de covarianza singular: intenta reducir N o mover la fecha de corte.")
        st.stop()

    tickers_final = list(returns_in.columns)

    with st.expander("Pesos del portafolio tangente (tabla completa)"):
        df_w = pd.DataFrame({
            "Ticker": tickers_final,
            "w_plug_in": w_tan_sample,
            "w_RIE": w_tan_rie,
        })
        st.dataframe(df_w.set_index("Ticker"))

    # 7) Desempeño in-sample (teórico)
    mu_p_in_sample, sigma_p_in_sample, sr_in_sample = portfolio_performance(
        w_tan_sample, mu_in, Sigma_in, RF_DAILY
    )
    mu_p_in_rie, sigma_p_in_rie, sr_in_rie = portfolio_performance(
        w_tan_rie, mu_in, Sigma_rie, RF_DAILY
    )

    # 8) Desempeño out-of-sample (realizado)
    mu_out_sample, sigma_out_sample, sr_out_sample = realized_sharpe(
        w_tan_sample, returns_out, RF_DAILY
    )
    mu_out_rie, sigma_out_rie, sr_out_rie = realized_sharpe(
        w_tan_rie, returns_out, RF_DAILY
    )

    st.markdown("### Sharpe ratios (diarios)")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Plug-in (muestral)**")
        st.write(f"- In-sample  SR ≈ `{sr_in_sample:.4f}`")
        st.write(f"- Out-of-sample SR ≈ `{sr_out_sample:.4f}`")

    with col2:
        st.write("**RIE (covarianza limpiada)**")
        st.write(f"- In-sample  SR ≈ `{sr_in_rie:.4f}`")
        st.write(f"- Out-of-sample SR ≈ `{sr_out_rie:.4f}`")

    # =====================================================
    # 9) Sección estilo Bun–Bouchaud–Potters: R_in^2, R_out^2, q
    # =====================================================

    st.markdown("### Gap $R_{\\text{in}}^2$ vs $R_{\\text{out}}^2$")

    # --- Plug-in ---
    R_in2_emp_plug = sigma_p_in_sample**2
    R_out2_emp_plug = sigma_out_sample**2
    R_true2_est_plug = float(w_tan_sample @ Sigma_out @ w_tan_sample)
    R_in2_theo_plug = (1 - q) * R_true2_est_plug
    R_out2_theo_plug = R_true2_est_plug / (1 - q) if (1 - q) != 0 else np.nan

    # --- RIE ---
    R_in2_emp_rie = sigma_p_in_rie**2
    R_out2_emp_rie = sigma_out_rie**2
    R_true2_est_rie = float(w_tan_rie @ Sigma_out @ w_tan_rie)
    R_in2_theo_rie = (1 - q) * R_true2_est_rie
    R_out2_theo_rie = R_true2_est_rie / (1 - q) if (1 - q) != 0 else np.nan

    st.markdown(
        r"""
Suponiendo que la covarianza verdadera $C$ se aproxima por $\Sigma_{\text{out}}$,  
para un portafolio $w$ (plug-in o RIE) definimos:

- $R_{\text{true}}^2 \approx w^{\top}\,\Sigma_{\text{out}}\,w$  
- $R_{\text{in}}^2 = w^{\top}\,\Sigma_{\text{in}}\,w$  
- $R_{\text{out}}^2 \approx \operatorname{Var}_{\text{out}}\big(w^{\top}R_t\big)$  

La fórmula asintótica de Bun–Bouchaud–Potters da la relación:
$$
\frac{R_{\text{in}}^2}{1-q} = R_{\text{true}}^2 = (1-q)R_{\text{out}}^2.
$$

A partir de $R_{\text{true}}^2$ estimado, los valores **teóricos** serían:

- $R_{\text{in}}^2(\text{teo}) = (1-q)\,R_{\text{true}}^2$  
- $R_{\text{out}}^2(\text{teo}) = \dfrac{R_{\text{true}}^2}{1-q}$.
"""
    )

    # Tabla numérica para ambos portafolios
    df_risks = pd.DataFrame(
        {
            "Portafolio": [
                "Plug-in", "Plug-in", "Plug-in", "Plug-in", "Plug-in",
                "RIE", "RIE", "RIE", "RIE", "RIE"
            ],
            "Cantidad": [
                "R_in^2 empírico",
                "R_out^2 empírico",
                "R_true^2 estimado",
                "R_in^2 teórico (BBP)",
                "R_out^2 teórico (BBP)",
                "R_in^2 empírico",
                "R_out^2 empírico",
                "R_true^2 estimado",
                "R_in^2 teórico (BBP)",
                "R_out^2 teórico (BBP)",
            ],
            "Valor": [
                R_in2_emp_plug,
                R_out2_emp_plug,
                R_true2_est_plug,
                R_in2_theo_plug,
                R_out2_theo_plug,
                R_in2_emp_rie,
                R_out2_emp_rie,
                R_true2_est_rie,
                R_in2_theo_rie,
                R_out2_theo_rie,
            ],
        }
    )

    with st.expander("Tabla numérica de riesgos (varianzas) – Plug-in y RIE"):
        st.dataframe(
            df_risks.pivot(index="Cantidad", columns="Portafolio", values="Valor")
            .style.format("{:.4e}")
        )

    # Gráfico del gap para ambos portafolios
    plot_risk_gap(
        R_in2_emp_plug, R_out2_emp_plug, R_in2_theo_plug, R_out2_theo_plug,
        R_in2_emp_rie, R_out2_emp_rie, R_in2_theo_rie, R_out2_theo_rie
    )

    # =====================================================
    # 10) Frontera de Markowitz plug-in vs RIE
    # =====================================================

    st.markdown("## Frontera eficiente de Markowitz (plug-in vs RIE)")
    plot_efficient_frontier(mu_in, Sigma_in, Sigma_rie, RF_DAILY, w_tan_sample, w_tan_rie)

    # =====================================================
    # 11) Gráficas interactivas estándar
    # =====================================================

    st.markdown("## Gráficas interactivas adicionales")

    st.markdown("#### Precios ajustados")
    plot_price_series(prices[tickers_final])

    st.markdown("#### Rendimientos diarios: in-sample vs out-of-sample (subconjunto de activos)")
    plot_returns_series(returns_in[tickers_final], returns_out[tickers_final])

    st.markdown("#### Espectro de eigenvalores: Σ_in muestral vs Σ_in limpiada (RIE)")
    plot_eigenvalues_comparison(lambdas_raw, lambdas_clean)

    st.markdown("#### Pesos del portafolio tangente: plug-in vs RIE (top por |w_RIE|)")
    plot_weights_comparison(tickers_final, w_tan_sample, w_tan_rie, top_k=20)

    st.markdown("#### Comparación de Sharpe: plug-in vs RIE")
    plot_sharpe_comparison(sr_in_sample, sr_out_sample, sr_in_rie, sr_out_rie)

else:
    st.info("Configura los parámetros en la barra lateral y haz clic en **Ejecutar experimento**.")
