"""
edhec_risk_kit_oop.py
=====================
Consolidación OOP de todas las versiones de edhec_risk_kit (104–129).
Organizado en tres clases principales:

    DataLoader      – carga y preprocesamiento de datos
    RiskAnalytics   – métricas de riesgo y estadísticas
    PortfolioEngine – optimización, fronteras eficientes y backtests
"""

import math
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import norm
from scipy.optimize import minimize


# ──────────────────────────────────────────────────────────────────────────────
# 1. DataLoader
# ──────────────────────────────────────────────────────────────────────────────

class DataLoader:
    """
    Carga y formatea los datasets usados en el curso EDHEC.

    Todos los métodos son estáticos y asumen que los archivos CSV
    viven en el subdirectorio ``data/``.
    """

    @staticmethod
    def get_ffme_returns() -> pd.DataFrame:
        """
        Carga el dataset Fama-French: retornos mensuales del decil
        inferior (SmallCap) y superior (LargeCap) por capitalización.
        """
        me_m = pd.read_csv(
            "data/Portfolios_Formed_on_ME_monthly_EW.csv",
            header=0, index_col=0, na_values=-99.99
        )
        rets = me_m[["Lo 10", "Hi 10"]].copy()
        rets.columns = ["SmallCap", "LargeCap"]
        rets /= 100
        rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period("M")
        return rets

    @staticmethod
    def get_hfi_returns() -> pd.DataFrame:
        """Carga los retornos del EDHEC Hedge Fund Index."""
        hfi = pd.read_csv(
            "data/edhec-hedgefundindices.csv",
            header=0, index_col=0, parse_dates=True
        )
        hfi /= 100
        hfi.index = hfi.index.to_period("M")
        return hfi

    @staticmethod
    def get_ind_file(filetype: str) -> pd.DataFrame:
        """
        Carga uno de los tres archivos de las 30 industrias Ken French.

        Parameters
        ----------
        filetype : {"returns", "nfirms", "size"}
        """
        known = {"returns": ("vw_rets", 100),
                 "nfirms":  ("nfirms",  1),
                 "size":    ("size",    1)}
        if filetype not in known:
            raise ValueError(f"filetype debe ser uno de: {', '.join(known)}")
        name, divisor = known[filetype]
        ind = pd.read_csv(
            f"data/ind30_m_{name}.csv", header=0, index_col=0
        ) / divisor
        ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period("M")
        ind.columns = ind.columns.str.strip()
        return ind

    @staticmethod
    def get_ind_returns() -> pd.DataFrame:
        """Retornos mensuales value-weighted de las 30 industrias."""
        return DataLoader.get_ind_file("returns")

    @staticmethod
    def get_ind_nfirms() -> pd.DataFrame:
        """Número promedio de empresas por industria."""
        return DataLoader.get_ind_file("nfirms")

    @staticmethod
    def get_ind_size() -> pd.DataFrame:
        """Capitalización promedio por industria."""
        return DataLoader.get_ind_file("size")

    @staticmethod
    def get_total_market_index_returns() -> pd.Series:
        """
        Construye un índice de mercado total cap-weighted
        a partir de las 30 industrias Ken French.
        """
        ind_nfirms  = DataLoader.get_ind_nfirms()
        ind_size    = DataLoader.get_ind_size()
        ind_return  = DataLoader.get_ind_returns()
        ind_mktcap  = ind_nfirms * ind_size
        total_mktcap = ind_mktcap.sum(axis=1)
        ind_capweight = ind_mktcap.divide(total_mktcap, axis="rows")
        return (ind_capweight * ind_return).sum(axis="columns")


# ──────────────────────────────────────────────────────────────────────────────
# 2. RiskAnalytics
# ──────────────────────────────────────────────────────────────────────────────

class RiskAnalytics:
    """
    Métricas de riesgo, estadísticas descriptivas y pruebas de normalidad.
    Todos los métodos son estáticos para facilitar el uso como utilidades.
    """

    # ── Estadísticas descriptivas ─────────────────────────────────────────────

    @staticmethod
    def skewness(r) -> float | pd.Series:
        """Asimetría muestral (alternativa a scipy.stats.skew)."""
        demeaned = r - r.mean()
        sigma    = r.std(ddof=0)
        return (demeaned ** 3).mean() / sigma ** 3

    @staticmethod
    def kurtosis(r) -> float | pd.Series:
        """Curtosis muestral (alternativa a scipy.stats.kurtosis)."""
        demeaned = r - r.mean()
        sigma    = r.std(ddof=0)
        return (demeaned ** 4).mean() / sigma ** 4

    @staticmethod
    def is_normal(r, level: float = 0.01) -> bool | pd.Series:
        """
        Test Jarque-Bera de normalidad al nivel especificado.
        Devuelve True si no se rechaza la hipótesis de normalidad.
        """
        if isinstance(r, pd.DataFrame):
            return r.aggregate(RiskAnalytics.is_normal)
        _, p_value = scipy.stats.jarque_bera(r)
        return p_value > level

    @staticmethod
    def compound(r) -> float:
        """Retorno compuesto de una serie de retornos."""
        return np.expm1(np.log1p(r).sum())

    @staticmethod
    def annualize_rets(r, periods_per_year: int) -> float | pd.Series:
        """Anualiza retornos dados como serie de períodos."""
        compounded = (1 + r).prod()
        n_periods  = r.shape[0]
        return compounded ** (periods_per_year / n_periods) - 1

    @staticmethod
    def annualize_vol(r, periods_per_year: int) -> float | pd.Series:
        """Anualiza la volatilidad."""
        return r.std() * (periods_per_year ** 0.5)

    @staticmethod
    def sharpe_ratio(r, riskfree_rate: float, periods_per_year: int) -> float | pd.Series:
        """Ratio de Sharpe anualizado."""
        rf_per_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
        excess        = r - rf_per_period
        ann_ex_ret    = RiskAnalytics.annualize_rets(excess, periods_per_year)
        ann_vol       = RiskAnalytics.annualize_vol(r, periods_per_year)
        return ann_ex_ret / ann_vol

    # ── Drawdown ──────────────────────────────────────────────────────────────

    @staticmethod
    def drawdown(return_series: pd.Series) -> pd.DataFrame:
        """
        Calcula el drawdown de una serie de retornos.

        Returns
        -------
        DataFrame con columnas: Wealth, Previous Peak, Drawdown.
        """
        wealth         = 1000 * (1 + return_series).cumprod()
        previous_peaks = wealth.cummax()
        drawdowns      = (wealth - previous_peaks) / previous_peaks
        return pd.DataFrame({
            "Wealth":        wealth,
            "Previous Peak": previous_peaks,
            "Drawdown":      drawdowns,
        })

    # ── Semidesvación ─────────────────────────────────────────────────────────

    @staticmethod
    def semideviation(r) -> float | pd.Series:
        """Semidesvación negativa (downside deviation)."""
        if isinstance(r, pd.Series):
            return r[r < 0].std(ddof=0)
        elif isinstance(r, pd.DataFrame):
            return r.aggregate(RiskAnalytics.semideviation)
        raise TypeError("r debe ser Series o DataFrame")

    # ── Value at Risk ─────────────────────────────────────────────────────────

    @staticmethod
    def var_historic(r, level: int = 5) -> float | pd.Series:
        """VaR histórico al nivel dado (en %)."""
        if isinstance(r, pd.DataFrame):
            return r.aggregate(RiskAnalytics.var_historic, level=level)
        elif isinstance(r, pd.Series):
            return -np.percentile(r, level)
        raise TypeError("r debe ser Series o DataFrame")

    @staticmethod
    def cvar_historic(r, level: int = 5) -> float | pd.Series:
        """CVaR (Expected Shortfall) histórico."""
        if isinstance(r, pd.Series):
            is_beyond = r <= -RiskAnalytics.var_historic(r, level=level)
            return -r[is_beyond].mean()
        elif isinstance(r, pd.DataFrame):
            return r.aggregate(RiskAnalytics.cvar_historic, level=level)
        raise TypeError("r debe ser Series o DataFrame")

    @staticmethod
    def var_gaussian(r, level: int = 5, modified: bool = False) -> float | pd.Series:
        """
        VaR paramétrico Gaussiano.
        Si ``modified=True`` aplica la corrección Cornish-Fisher.
        """
        z = norm.ppf(level / 100)
        if modified:
            s = RiskAnalytics.skewness(r)
            k = RiskAnalytics.kurtosis(r)
            z = (z
                 + (z**2 - 1) * s / 6
                 + (z**3 - 3*z) * (k - 3) / 24
                 - (2*z**3 - 5*z) * (s**2) / 36)
        return -(r.mean() + z * r.std(ddof=0))

    # ── Summary ───────────────────────────────────────────────────────────────

    @staticmethod
    def summary_stats(r: pd.DataFrame, riskfree_rate: float = 0.03) -> pd.DataFrame:
        """
        Tabla resumen de estadísticas para cada columna de retornos.
        Asume datos mensuales (12 períodos por año).
        """
        ann_r    = r.aggregate(RiskAnalytics.annualize_rets, periods_per_year=12)
        ann_vol  = r.aggregate(RiskAnalytics.annualize_vol,  periods_per_year=12)
        ann_sr   = r.aggregate(RiskAnalytics.sharpe_ratio,
                               riskfree_rate=riskfree_rate, periods_per_year=12)
        dd       = r.aggregate(lambda s: RiskAnalytics.drawdown(s).Drawdown.min())
        skew     = r.aggregate(RiskAnalytics.skewness)
        kurt     = r.aggregate(RiskAnalytics.kurtosis)
        cf_var5  = r.aggregate(RiskAnalytics.var_gaussian, modified=True)
        hist_cvar5 = r.aggregate(RiskAnalytics.cvar_historic)
        return pd.DataFrame({
            "Annualized Return":      ann_r,
            "Annualized Vol":         ann_vol,
            "Skewness":               skew,
            "Kurtosis":               kurt,
            "Cornish-Fisher VaR (5%)": cf_var5,
            "Historic CVaR (5%)":     hist_cvar5,
            "Sharpe Ratio":           ann_sr,
            "Max Drawdown":           dd,
        })


# ──────────────────────────────────────────────────────────────────────────────
# 3. PortfolioEngine
# ──────────────────────────────────────────────────────────────────────────────

class PortfolioEngine:
    """
    Optimización de portafolios, frontera eficiente, simulaciones y backtests.
    """

    # ── Retorno y volatilidad de portafolio ───────────────────────────────────

    @staticmethod
    def portfolio_return(weights: np.ndarray, returns: np.ndarray) -> float:
        """Retorno esperado de un portafolio."""
        return weights.T @ returns

    @staticmethod
    def portfolio_vol(weights: np.ndarray, covmat: np.ndarray) -> float:
        """Volatilidad de un portafolio."""
        return (weights.T @ covmat @ weights) ** 0.5

    # ── Optimización ─────────────────────────────────────────────────────────

    @staticmethod
    def minimize_vol(target_return: float, er: np.ndarray,
                     cov: np.ndarray) -> np.ndarray:
        """
        Pesos que minimizan la volatilidad para un retorno objetivo dado.
        """
        n = er.shape[0]
        init_guess = np.repeat(1 / n, n)
        bounds = ((0.0, 1.0),) * n
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "args": (er,),
             "fun": lambda w, er: target_return - PortfolioEngine.portfolio_return(w, er)},
        )
        result = minimize(
            PortfolioEngine.portfolio_vol, init_guess,
            args=(cov,), method="SLSQP",
            options={"disp": False},
            constraints=constraints, bounds=bounds,
        )
        return result.x

    @staticmethod
    def msr(riskfree_rate: float, er: np.ndarray,
            cov: np.ndarray) -> np.ndarray:
        """Pesos del portafolio de máximo Sharpe Ratio (MSR)."""
        n = er.shape[0]
        init_guess = np.repeat(1 / n, n)
        bounds = ((0.0, 1.0),) * n

        def neg_sharpe(weights, riskfree_rate, er, cov):
            r   = PortfolioEngine.portfolio_return(weights, er)
            vol = PortfolioEngine.portfolio_vol(weights, cov)
            return -(r - riskfree_rate) / vol

        result = minimize(
            neg_sharpe, init_guess,
            args=(riskfree_rate, er, cov), method="SLSQP",
            options={"disp": False},
            constraints=({"type": "eq", "fun": lambda w: np.sum(w) - 1},),
            bounds=bounds,
        )
        return result.x

    @staticmethod
    def gmv(cov: np.ndarray) -> np.ndarray:
        """Pesos del portafolio de mínima varianza global (GMV)."""
        n = cov.shape[0]
        return PortfolioEngine.msr(0, np.repeat(1, n), cov)

    @staticmethod
    def optimal_weights(n_points: int, er: np.ndarray,
                        cov: np.ndarray) -> list[np.ndarray]:
        """Lista de pesos óptimos para n_points sobre la frontera eficiente."""
        target_rs = np.linspace(er.min(), er.max(), n_points)
        return [PortfolioEngine.minimize_vol(t, er, cov) for t in target_rs]

    # ── Gráficas de frontera eficiente ────────────────────────────────────────

    @staticmethod
    def plot_ef2(n_points: int, er, cov):
        """Frontera eficiente de 2 activos."""
        if er.shape[0] != 2:
            raise ValueError("plot_ef2 solo acepta 2 activos")
        weights = [np.array([w, 1 - w]) for w in np.linspace(0, 1, n_points)]
        rets = [PortfolioEngine.portfolio_return(w, er) for w in weights]
        vols = [PortfolioEngine.portfolio_vol(w, cov)   for w in weights]
        ef = pd.DataFrame({"Returns": rets, "Volatility": vols})
        return ef.plot.line(x="Volatility", y="Returns", style=".-")

    @staticmethod
    def plot_ef(n_points: int, er, cov,
                style: str = ".-", legend: bool = False,
                show_cml: bool = False, riskfree_rate: float = 0.0,
                show_ew: bool = False, show_gmv: bool = False):
        """
        Frontera eficiente multi-activo con opciones para mostrar:
        - CML (Capital Market Line)
        - portafolio equal-weight (EW)
        - portafolio de mínima varianza global (GMV)
        """
        weights = PortfolioEngine.optimal_weights(n_points, er, cov)
        rets = [PortfolioEngine.portfolio_return(w, er) for w in weights]
        vols = [PortfolioEngine.portfolio_vol(w, cov)   for w in weights]
        ef = pd.DataFrame({"Returns": rets, "Volatility": vols})
        ax = ef.plot.line(x="Volatility", y="Returns", style=style, legend=legend)

        if show_cml:
            ax.set_xlim(left=0)
            w_msr   = PortfolioEngine.msr(riskfree_rate, er, cov)
            r_msr   = PortfolioEngine.portfolio_return(w_msr, er)
            vol_msr = PortfolioEngine.portfolio_vol(w_msr, cov)
            ax.plot([0, vol_msr], [riskfree_rate, r_msr],
                    color="green", marker="o", linestyle="dashed",
                    linewidth=2, markersize=10)

        if show_ew:
            n     = er.shape[0]
            w_ew  = np.repeat(1 / n, n)
            r_ew  = PortfolioEngine.portfolio_return(w_ew, er)
            v_ew  = PortfolioEngine.portfolio_vol(w_ew, cov)
            ax.plot([v_ew], [r_ew], color="goldenrod", marker="o", markersize=10)

        if show_gmv:
            w_gmv  = PortfolioEngine.gmv(cov)
            r_gmv  = PortfolioEngine.portfolio_return(w_gmv, er)
            vol_gmv = PortfolioEngine.portfolio_vol(w_gmv, cov)
            ax.plot([vol_gmv], [r_gmv], color="midnightblue",
                    marker="o", markersize=10)

        return ax

    # ── CPPI ──────────────────────────────────────────────────────────────────

    @staticmethod
    def run_cppi(risky_r, safe_r=None, m: int = 3, start: float = 1000,
                 floor: float = 0.8, riskfree_rate: float = 0.03,
                 drawdown: float = None) -> dict:
        """
        Backtest de la estrategia CPPI.

        Returns
        -------
        dict con claves: Wealth, Risky Wealth, Risk Budget,
                         Risky Allocation, m, start, floor,
                         risky_r, safe_r, drawdown, peak.
        """
        dates         = risky_r.index
        n_steps       = len(dates)
        account_value = start
        floor_value   = start * floor
        peak          = account_value

        if isinstance(risky_r, pd.Series):
            risky_r = pd.DataFrame(risky_r, columns=["R"])

        if safe_r is None:
            safe_r = pd.DataFrame().reindex_like(risky_r)
            safe_r.values[:] = riskfree_rate / 12

        account_history  = pd.DataFrame().reindex_like(risky_r)
        risky_w_history  = pd.DataFrame().reindex_like(risky_r)
        cushion_history  = pd.DataFrame().reindex_like(risky_r)
        floorval_history = pd.DataFrame().reindex_like(risky_r)
        peak_history     = pd.DataFrame().reindex_like(risky_r)

        for step in range(n_steps):
            if drawdown is not None:
                peak        = np.maximum(peak, account_value)
                floor_value = peak * (1 - drawdown)
            cushion = (account_value - floor_value) / account_value
            risky_w = np.clip(m * cushion, 0, 1)
            safe_w  = 1 - risky_w
            risky_alloc = account_value * risky_w
            safe_alloc  = account_value * safe_w
            account_value = (risky_alloc * (1 + risky_r.iloc[step])
                             + safe_alloc  * (1 + safe_r.iloc[step]))
            cushion_history.iloc[step]  = cushion
            risky_w_history.iloc[step]  = risky_w
            account_history.iloc[step]  = account_value
            floorval_history.iloc[step] = floor_value
            peak_history.iloc[step]     = peak

        risky_wealth = start * (1 + risky_r).cumprod()
        return {
            "Wealth":           account_history,
            "Risky Wealth":     risky_wealth,
            "Risk Budget":      cushion_history,
            "Risky Allocation": risky_w_history,
            "m":                m,
            "start":            start,
            "floor":            floorval_history,
            "risky_r":          risky_r,
            "safe_r":           safe_r,
            "drawdown":         drawdown,
            "peak":             peak_history,
        }

    # ── Simulaciones ─────────────────────────────────────────────────────────

    @staticmethod
    def gbm(n_years: int = 10, n_scenarios: int = 1000,
            mu: float = 0.07, sigma: float = 0.15,
            steps_per_year: int = 12, s_0: float = 100.0,
            prices: bool = True) -> pd.DataFrame:
        """
        Simulación de Movimiento Browniano Geométrico (GBM) por Monte Carlo.
        """
        dt      = 1 / steps_per_year
        n_steps = int(n_years * steps_per_year) + 1
        rets_plus_1 = np.random.normal(
            loc=(1 + mu) ** dt,
            scale=sigma * np.sqrt(dt),
            size=(n_steps, n_scenarios),
        )
        rets_plus_1[0] = 1
        ret_val = s_0 * pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1 - 1
        return ret_val

    # ── Tasas e interés ───────────────────────────────────────────────────────

    @staticmethod
    def inst_to_ann(r) -> float:
        """Convierte tasa instantánea a tasa anual."""
        return np.expm1(r)

    @staticmethod
    def ann_to_inst(r) -> float:
        """Convierte tasa anual a tasa instantánea."""
        return np.log1p(r)

    @staticmethod
    def cir(n_years: int = 10, n_scenarios: int = 1,
            a: float = 0.05, b: float = 0.03, sigma: float = 0.05,
            steps_per_year: int = 12, r_0: float = None):
        """
        Genera trayectorias de tasas de interés mediante el modelo CIR.

        Returns
        -------
        rates : DataFrame  – tasas anualizadas
        prices : DataFrame – precios de bonos cero-cupón
        """
        if r_0 is None:
            r_0 = b
        r_0       = PortfolioEngine.ann_to_inst(r_0)
        dt        = 1 / steps_per_year
        num_steps = int(n_years * steps_per_year) + 1

        shock  = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
        rates  = np.empty_like(shock)
        rates[0] = r_0
        prices = np.empty_like(shock)

        h = math.sqrt(a ** 2 + 2 * sigma ** 2)

        def price(ttm, r):
            _A = (
                (2 * h * math.exp((h + a) * ttm / 2))
                / (2 * h + (h + a) * (math.exp(h * ttm) - 1))
            ) ** (2 * a * b / sigma ** 2)
            _B = (2 * (math.exp(h * ttm) - 1)) / (
                2 * h + (h + a) * (math.exp(h * ttm) - 1)
            )
            return _A * np.exp(-_B * r)

        prices[0] = price(n_years, r_0)

        for step in range(1, num_steps):
            r_t   = rates[step - 1]
            d_r_t = a * (b - r_t) * dt + sigma * np.sqrt(r_t) * shock[step]
            rates[step]  = abs(r_t + d_r_t)
            prices[step] = price(n_years - step * dt, rates[step])

        rates  = pd.DataFrame(data=PortfolioEngine.inst_to_ann(rates),
                              index=range(num_steps))
        prices = pd.DataFrame(data=prices, index=range(num_steps))
        return rates, prices

    # ── Bonos ─────────────────────────────────────────────────────────────────

    @staticmethod
    def discount(t, r) -> pd.DataFrame:
        """
        Precio de un bono cero-cupón que paga $1 en el período t,
        con tasa periódica r.
        """
        discounts = pd.DataFrame([(r + 1) ** -i for i in t])
        discounts.index = t
        return discounts

    @staticmethod
    def pv(flows: pd.Series, r) -> pd.Series:
        """Valor presente de flujos de caja dados."""
        dates     = flows.index
        discounts = PortfolioEngine.discount(dates, r)
        return discounts.multiply(flows, axis="rows").sum()

    @staticmethod
    def funding_ratio(assets, liabilities, r) -> float:
        """Ratio de financiamiento de un pasivo."""
        return float(
            PortfolioEngine.pv(assets, r) / PortfolioEngine.pv(liabilities, r)
        )

    @staticmethod
    def bond_cash_flows(maturity: float, principal: float = 100,
                        coupon_rate: float = 0.03,
                        coupons_per_year: int = 12) -> pd.Series:
        """Flujos de caja de un bono con cupones regulares."""
        n_coupons  = round(maturity * coupons_per_year)
        coupon_amt = principal * coupon_rate / coupons_per_year
        coupon_times = np.arange(1, n_coupons + 1)
        cash_flows = pd.Series(data=coupon_amt, index=coupon_times)
        cash_flows.iloc[-1] += principal
        return cash_flows

    @staticmethod
    def bond_price(maturity: float, principal: float = 100,
                   coupon_rate: float = 0.03, coupons_per_year: int = 12,
                   discount_rate=0.03):
        """
        Precio de un bono.
        Si ``discount_rate`` es un DataFrame se asume que es la tasa
        para cada fecha de cupón y se devuelve una serie de precios.
        """
        if isinstance(discount_rate, pd.DataFrame):
            pricing_dates = discount_rate.index
            prices = pd.DataFrame(index=pricing_dates, columns=discount_rate.columns)
            for t in pricing_dates:
                prices.loc[t] = PortfolioEngine.bond_price(
                    maturity - t / coupons_per_year,
                    principal, coupon_rate, coupons_per_year,
                    discount_rate.loc[t],
                )
            return prices
        if maturity <= 0:
            return principal + principal * coupon_rate / coupons_per_year
        cash_flows = PortfolioEngine.bond_cash_flows(
            maturity, principal, coupon_rate, coupons_per_year
        )
        return PortfolioEngine.pv(cash_flows, discount_rate / coupons_per_year)

    @staticmethod
    def macaulay_duration(flows: pd.Series, discount_rate: float) -> float:
        """Duración de Macaulay de una secuencia de flujos de caja."""
        discounted = PortfolioEngine.discount(flows.index, discount_rate) * pd.DataFrame(flows)
        weights    = discounted / discounted.sum()
        return np.average(flows.index, weights=weights.iloc[:, 0])

    @staticmethod
    def match_durations(cf_t, cf_s, cf_l, discount_rate: float) -> float:
        """
        Peso W en el bono corto (cf_s) tal que la combinación W·cf_s + (1-W)·cf_l
        tenga la misma duración que el pasivo objetivo cf_t.
        """
        d_t = PortfolioEngine.macaulay_duration(cf_t, discount_rate)
        d_s = PortfolioEngine.macaulay_duration(cf_s, discount_rate)
        d_l = PortfolioEngine.macaulay_duration(cf_l, discount_rate)
        return (d_l - d_t) / (d_l - d_s)

    @staticmethod
    def bond_total_return(monthly_prices: pd.DataFrame, principal: float,
                          coupon_rate: float, coupons_per_year: int) -> pd.DataFrame:
        """Retorno total de un bono incluyendo cupones reinvertidos."""
        coupons  = pd.DataFrame(data=0,
                                index=monthly_prices.index,
                                columns=monthly_prices.columns)
        t_max    = monthly_prices.index.max()
        pay_date = np.linspace(12 / coupons_per_year, t_max,
                               int(coupons_per_year * t_max / 12), dtype=int)
        coupons.iloc[pay_date] = principal * coupon_rate / coupons_per_year
        total_returns = (monthly_prices + coupons) / monthly_prices.shift() - 1
        return total_returns.dropna()

    # ── Allocators / backtests mixtos ─────────────────────────────────────────

    @staticmethod
    def bt_mix(r1: pd.DataFrame, r2: pd.DataFrame, allocator, **kwargs) -> pd.DataFrame:
        """
        Backtest de una estrategia de mezcla entre dos activos.
        ``allocator`` es una función que retorna pesos hacia r1.
        """
        if r1.shape != r2.shape:
            raise ValueError("r1 y r2 deben tener la misma forma")
        weights = allocator(r1, r2, **kwargs)
        if weights.shape != r1.shape:
            raise ValueError("El allocator devolvió pesos con forma incorrecta")
        return weights * r1 + (1 - weights) * r2

    @staticmethod
    def fixedmix_allocator(r1: pd.DataFrame, r2: pd.DataFrame,
                           w1: float, **kwargs) -> pd.DataFrame:
        """Asignación fija w1 al primer activo."""
        return pd.DataFrame(data=w1, index=r1.index, columns=r1.columns)

    @staticmethod
    def glidepath_allocator(r1: pd.DataFrame, r2: pd.DataFrame,
                            start_glide: float = 1.0,
                            end_glide: float = 0.0) -> pd.DataFrame:
        """Asignación deslizante lineal de start_glide a end_glide."""
        n_points = r1.shape[0]
        n_col    = r1.shape[1]
        path     = pd.Series(np.linspace(start_glide, end_glide, n_points))
        paths    = pd.concat([path] * n_col, axis=1)
        paths.index   = r1.index
        paths.columns = r1.columns
        return paths

    @staticmethod
    def floor_allocator(psp_r: pd.DataFrame, ghp_r: pd.DataFrame,
                        floor: float, zc_prices: pd.DataFrame,
                        m: int = 3) -> pd.DataFrame:
        """
        CPPI-style allocator con piso basado en precios de bonos cero-cupón.
        """
        if zc_prices.shape != psp_r.shape:
            raise ValueError("psp_r y zc_prices deben tener la misma forma")
        n_steps, n_scenarios = psp_r.shape
        account_value = np.repeat(1, n_scenarios)
        floor_value   = np.repeat(1, n_scenarios)
        w_history     = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)

        for step in range(n_steps):
            floor_value   = floor * zc_prices.iloc[step]
            cushion       = (account_value - floor_value) / account_value
            psp_w         = np.clip(m * cushion, 0, 1)
            ghp_w         = 1 - psp_w
            account_value = (account_value * psp_w * (1 + psp_r.iloc[step])
                             + account_value * ghp_w * (1 + ghp_r.iloc[step]))
            w_history.iloc[step] = psp_w

        return w_history

    @staticmethod
    def drawdown_allocator(psp_r: pd.DataFrame, ghp_r: pd.DataFrame,
                           maxdd: float, m: int = 3) -> pd.DataFrame:
        """
        CPPI-style allocator con piso basado en drawdown máximo permitido.
        """
        n_steps, n_scenarios = psp_r.shape
        account_value = np.repeat(1, n_scenarios)
        peak_value    = np.repeat(1, n_scenarios)
        w_history     = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)

        for step in range(n_steps):
            floor_value   = (1 - maxdd) * peak_value
            cushion       = (account_value - floor_value) / account_value
            psp_w         = np.clip(m * cushion, 0, 1)
            ghp_w         = 1 - psp_w
            account_value = (account_value * psp_w * (1 + psp_r.iloc[step])
                             + account_value * ghp_w * (1 + ghp_r.iloc[step]))
            peak_value    = np.maximum(peak_value, account_value)
            w_history.iloc[step] = psp_w

        return w_history

    # ── Terminal wealth stats ─────────────────────────────────────────────────

    @staticmethod
    def terminal_values(rets: pd.DataFrame) -> pd.Series:
        """Valor terminal de cada escenario (dólar invertido)."""
        return (rets + 1).prod()

    @staticmethod
    def terminal_stats(rets: pd.DataFrame, floor: float = 0.8,
                       cap: float = np.inf, name: str = "Stats") -> pd.DataFrame:
        """Estadísticas resumidas sobre los valores terminales."""
        tw      = (rets + 1).prod()
        breach  = tw < floor
        reach   = tw >= cap
        p_breach  = breach.mean()         if breach.sum() > 0 else np.nan
        p_reach   = reach.mean()          if reach.sum()  > 0 else np.nan
        e_short   = (floor - tw[breach]).mean() if breach.sum() > 0 else np.nan
        e_surplus = (cap   - tw[reach]).mean()  if reach.sum()  > 0 else np.nan
        return pd.DataFrame.from_dict({
            "mean":      tw.mean(),
            "std":       tw.std(),
            "p_breach":  p_breach,
            "e_short":   e_short,
            "p_reach":   p_reach,
            "e_surplus": e_surplus,
        }, orient="index", columns=[name])
