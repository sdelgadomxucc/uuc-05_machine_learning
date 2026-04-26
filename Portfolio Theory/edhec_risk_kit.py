"""
edhec_risk_kit.py
=================
Versión consolidada y completa (todas las funciones de los notebooks).
Funciones disponibles:
  get_ffme_returns, get_hfi_returns, get_ind_file, get_ind_returns,
  get_ind_nfirms, get_ind_size, get_total_market_index_returns,
  drawdown, skewness, kurtosis, is_normal, compound,
  annualize_rets, annualize_vol, sharpe_ratio, semideviation,
  var_historic, cvar_historic, var_gaussian, summary_stats,
  portfolio_return, portfolio_vol, minimize_vol, msr, gmv,
  optimal_weights, plot_ef2, plot_ef,
  run_cppi, gbm,
  discount_simple, pv_simple, funding_ratio_simple,
  discount, pv, funding_ratio,
  inst_to_ann, ann_to_inst, cir,
  bond_cash_flows, bond_price, macaulay_duration, match_durations,
  bond_total_return,
  bt_mix, fixedmix_allocator, glidepath_allocator,
  floor_allocator, drawdown_allocator,
  terminal_values, terminal_stats
"""

import math
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import norm
from scipy.optimize import minimize


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom
    Deciles by MarketCap (SmallCap / LargeCap).
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


def get_hfi_returns():
    """Load and format the EDHEC Hedge Fund Index Returns."""
    hfi = pd.read_csv(
        "data/edhec-hedgefundindices.csv",
        header=0, index_col=0, parse_dates=True
    )
    hfi /= 100
    hfi.index = hfi.index.to_period("M")
    return hfi


def get_ind_file(filetype):
    """
    Load and format the Ken French 30 Industry Portfolios files.
    filetype: 'returns' | 'nfirms' | 'size'
    """
    known = {"returns": ("vw_rets", 100), "nfirms": ("nfirms", 1), "size": ("size", 1)}
    if filetype not in known:
        raise ValueError(f"filetype must be one of: {', '.join(known)}")
    name, divisor = known[filetype]
    ind = pd.read_csv(f"data/ind30_m_{name}.csv", header=0, index_col=0) / divisor
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period("M")
    ind.columns = ind.columns.str.strip()
    return ind


def get_ind_returns():
    """Value Weighted Monthly Returns for 30 Industry Portfolios."""
    return get_ind_file("returns")


def get_ind_nfirms():
    """Average number of firms for 30 Industry Portfolios."""
    return get_ind_file("nfirms")


def get_ind_size():
    """Average size (market cap) for 30 Industry Portfolios."""
    return get_ind_file("size")


def get_total_market_index_returns():
    """
    Derive the cap-weighted total market index returns
    from the 30 Ken French Industry Portfolios.
    """
    ind_nfirms  = get_ind_nfirms()
    ind_size    = get_ind_size()
    ind_return  = get_ind_returns()
    ind_mktcap  = ind_nfirms * ind_size
    total_mktcap = ind_mktcap.sum(axis=1)
    ind_capweight = ind_mktcap.divide(total_mktcap, axis="rows")
    return (ind_capweight * ind_return).sum(axis="columns")


# ──────────────────────────────────────────────────────────────────────────────
# DESCRIPTIVE STATISTICS & RISK METRICS
# ──────────────────────────────────────────────────────────────────────────────

def drawdown(return_series: pd.Series):
    """
    Takes a time series of asset returns.
    Returns a DataFrame with: Wealth, Previous Peak, Drawdown.
    """
    wealth         = 1000 * (1 + return_series).cumprod()
    previous_peaks = wealth.cummax()
    drawdowns      = (wealth - previous_peaks) / previous_peaks
    return pd.DataFrame({
        "Wealth":        wealth,
        "Previous Peak": previous_peaks,
        "Drawdown":      drawdowns,
    })


def skewness(r):
    """
    Computes the skewness of the supplied Series or DataFrame.
    Alternative to scipy.stats.skew().
    """
    demeaned = r - r.mean()
    sigma    = r.std(ddof=0)
    return (demeaned ** 3).mean() / sigma ** 3


def kurtosis(r):
    """
    Computes the kurtosis of the supplied Series or DataFrame.
    Alternative to scipy.stats.kurtosis().
    """
    demeaned = r - r.mean()
    sigma    = r.std(ddof=0)
    return (demeaned ** 4).mean() / sigma ** 4


def compound(r):
    """Returns the result of compounding the set of returns in r."""
    return np.expm1(np.log1p(r).sum())


def annualize_rets(r, periods_per_year):
    """Annualizes a set of returns."""
    compounded = (1 + r).prod()
    n_periods  = r.shape[0]
    return compounded ** (periods_per_year / n_periods) - 1


def annualize_vol(r, periods_per_year):
    """Annualizes the volatility of a set of returns."""
    return r.std() * (periods_per_year ** 0.5)


def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """Computes the annualized Sharpe Ratio of a set of returns."""
    rf_per_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
    excess        = r - rf_per_period
    ann_ex_ret    = annualize_rets(excess, periods_per_year)
    ann_vol       = annualize_vol(r, periods_per_year)
    return ann_ex_ret / ann_vol


def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal.
    Returns True if the hypothesis of normality is accepted.
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(is_normal)
    _, p_value = scipy.stats.jarque_bera(r)
    return p_value > level


def semideviation(r):
    """Returns the negative semideviation of r (Series or DataFrame)."""
    if isinstance(r, pd.Series):
        return r[r < 0].std(ddof=0)
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(semideviation)
    raise TypeError("Expected r to be a Series or DataFrame")


def var_historic(r, level=5):
    """
    Historic Value at Risk at a specified level (%).
    Returns the number such that `level`% of returns fall below it.
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    raise TypeError("Expected r to be a Series or DataFrame")


def cvar_historic(r, level=5):
    """Computes the Conditional VaR (Expected Shortfall) of Series or DataFrame."""
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    raise TypeError("Expected r to be a Series or DataFrame")


def var_gaussian(r, level=5, modified=False):
    """
    Parametric Gaussian VaR of a Series or DataFrame.
    If modified=True, applies the Cornish-Fisher modification.
    """
    z = norm.ppf(level / 100)
    if modified:
        s = skewness(r)
        k = kurtosis(r)
        z = (z
             + (z**2 - 1) * s / 6
             + (z**3 - 3*z) * (k - 3) / 24
             - (2*z**3 - 5*z) * (s**2) / 36)
    return -(r.mean() + z * r.std(ddof=0))


def summary_stats(r, riskfree_rate=0.03):
    """
    Returns a DataFrame of aggregated summary statistics for each column of r.
    Assumes monthly returns (12 periods per year).
    """
    ann_r      = r.aggregate(annualize_rets, periods_per_year=12)
    ann_vol    = r.aggregate(annualize_vol,  periods_per_year=12)
    ann_sr     = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=12)
    dd         = r.aggregate(lambda s: drawdown(s).Drawdown.min())
    skew       = r.aggregate(skewness)
    kurt       = r.aggregate(kurtosis)
    cf_var5    = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Return":       ann_r,
        "Annualized Vol":          ann_vol,
        "Skewness":                skew,
        "Kurtosis":                kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)":      hist_cvar5,
        "Sharpe Ratio":            ann_sr,
        "Max Drawdown":            dd,
    })


# ──────────────────────────────────────────────────────────────────────────────
# PORTFOLIO CONSTRUCTION & OPTIMIZATION
# ──────────────────────────────────────────────────────────────────────────────

def portfolio_return(weights, returns):
    """Computes the return of a portfolio given weights and expected returns."""
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    """Computes the volatility of a portfolio given weights and covariance matrix."""
    return (weights.T @ covmat @ weights) ** 0.5


def minimize_vol(target_return, er, cov):
    """
    Returns weights that achieve target_return with minimum volatility.
    """
    n = er.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n
    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "args": (er,),
         "fun": lambda w, er: target_return - portfolio_return(w, er)},
    )
    result = minimize(portfolio_vol, init_guess, args=(cov,),
                      method="SLSQP", options={"disp": False},
                      constraints=constraints, bounds=bounds)
    return result.x


def msr(riskfree_rate, er, cov):
    """Returns the weights of the Maximum Sharpe Ratio portfolio."""
    n = er.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n

    def neg_sharpe(weights, riskfree_rate, er, cov):
        r   = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate) / vol

    result = minimize(neg_sharpe, init_guess,
                      args=(riskfree_rate, er, cov), method="SLSQP",
                      options={"disp": False},
                      constraints=({"type": "eq", "fun": lambda w: np.sum(w) - 1},),
                      bounds=bounds)
    return result.x


def gmv(cov):
    """Returns the weights of the Global Minimum Variance portfolio."""
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)


def optimal_weights(n_points, er, cov):
    """Returns a list of optimal weights for n_points on the efficient frontier."""
    target_rs = np.linspace(er.min(), er.max(), n_points)
    return [minimize_vol(t, er, cov) for t in target_rs]


def plot_ef2(n_points, er, cov):
    """Plots the 2-asset efficient frontier."""
    if er.shape[0] != 2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")
    weights = [np.array([w, 1 - w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov)   for w in weights]
    ef = pd.DataFrame({"Returns": rets, "Volatility": vols})
    return ef.plot.line(x="Volatility", y="Returns", style=".-")


def plot_ef(n_points, er, cov, style=".-", legend=False,
            show_cml=False, riskfree_rate=0.0,
            show_ew=False, show_gmv=False):
    """
    Plots the multi-asset efficient frontier.
    Options: show_cml, show_ew (equal-weight), show_gmv (global min variance).
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov)   for w in weights]
    ef = pd.DataFrame({"Returns": rets, "Volatility": vols})
    ax = ef.plot.line(x="Volatility", y="Returns", style=style, legend=legend)

    if show_cml:
        ax.set_xlim(left=0)
        w_msr   = msr(riskfree_rate, er, cov)
        r_msr   = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        ax.plot([0, vol_msr], [riskfree_rate, r_msr],
                color="green", marker="o", linestyle="dashed",
                linewidth=2, markersize=10)
    if show_ew:
        n    = er.shape[0]
        w_ew = np.repeat(1 / n, n)
        ax.plot([portfolio_vol(w_ew, cov)], [portfolio_return(w_ew, er)],
                color="goldenrod", marker="o", markersize=10)
    if show_gmv:
        w_gmv = gmv(cov)
        ax.plot([portfolio_vol(w_gmv, cov)], [portfolio_return(w_gmv, er)],
                color="midnightblue", marker="o", markersize=10)
    return ax


# ──────────────────────────────────────────────────────────────────────────────
# CPPI
# ──────────────────────────────────────────────────────────────────────────────

def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8,
             riskfree_rate=0.03, drawdown=None):
    """
    Runs a backtest of the CPPI strategy.
    Returns a dict: Wealth, Risky Wealth, Risk Budget, Risky Allocation, etc.
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
        account_value = (account_value * risky_w * (1 + risky_r.iloc[step])
                         + account_value * safe_w  * (1 + safe_r.iloc[step]))
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


# ──────────────────────────────────────────────────────────────────────────────
# MONTE CARLO – GBM
# ──────────────────────────────────────────────────────────────────────────────

def gbm(n_years=10, n_scenarios=1000, mu=0.07, sigma=0.15,
        steps_per_year=12, s_0=100.0, prices=True):
    """
    Geometric Brownian Motion Monte Carlo simulation.
    Returns a DataFrame of price (or return) paths.
    """
    dt      = 1 / steps_per_year
    n_steps = int(n_years * steps_per_year) + 1
    rets_plus_1 = np.random.normal(
        loc=(1 + mu) ** dt,
        scale=sigma * np.sqrt(dt),
        size=(n_steps, n_scenarios),
    )
    rets_plus_1[0] = 1
    if prices:
        return s_0 * pd.DataFrame(rets_plus_1).cumprod()
    else:
        return pd.DataFrame(rets_plus_1 - 1)


# ──────────────────────────────────────────────────────────────────────────────
# INTEREST RATES & DISCOUNTING
# ──────────────────────────────────────────────────────────────────────────────

def inst_to_ann(r):
    """Convert instantaneous interest rate to annual rate."""
    return np.expm1(r)


def ann_to_inst(r):
    """Convert annual interest rate to instantaneous rate."""
    return np.log1p(r)


def discount_simple(t, r):
    """
    Price of a pure discount bond paying $1 at time t (years),
    with annual interest rate r.
    """
    return (1 + r) ** (-t)


def pv_simple(l, r):
    """
    Present value of a Series of liabilities (indexed by time in years)
    using a flat annual rate r.
    """
    dates     = l.index
    discounts = discount_simple(dates, r)
    return (discounts * l).sum()


def funding_ratio_simple(assets, liabilities, r):
    """Funding ratio using a flat discount rate."""
    return assets / pv_simple(liabilities, r)


def discount(t, r):
    """
    Price of a pure discount bond paying $1 at period t,
    with per-period rate r.
    r can be a float, Series, or DataFrame.
    Returns a DataFrame indexed by t.
    """
    discounts = pd.DataFrame([(r + 1) ** -i for i in t])
    discounts.index = t
    return discounts


def pv(flows, r):
    """
    Present value of cash flows given by a Series (indexed by time),
    discounted at per-period rate r.
    """
    dates     = flows.index
    discounts = discount(dates, r)
    return discounts.multiply(flows, axis="rows").sum()


def funding_ratio(assets, liabilities, r):
    """Funding ratio: PV(assets) / PV(liabilities)."""
    return float(pv(assets, r) / pv(liabilities, r))


# ──────────────────────────────────────────────────────────────────────────────
# CIR MODEL
# ──────────────────────────────────────────────────────────────────────────────

def cir(n_years=10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05,
        steps_per_year=12, r_0=None):
    """
    Generate random interest rate paths using the Cox-Ingersoll-Ross model.
    Returns (rates DataFrame, prices DataFrame) — both annualized.
    """
    if r_0 is None:
        r_0 = b
    r_0       = ann_to_inst(r_0)
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

    rates  = pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
    prices = pd.DataFrame(data=prices,             index=range(num_steps))
    return rates, prices


# ──────────────────────────────────────────────────────────────────────────────
# BONDS
# ──────────────────────────────────────────────────────────────────────────────

def bond_cash_flows(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12):
    """Returns the Series of cash flows for a coupon-bearing bond."""
    n_coupons    = round(maturity * coupons_per_year)
    coupon_amt   = principal * coupon_rate / coupons_per_year
    coupon_times = np.arange(1, n_coupons + 1)
    cash_flows   = pd.Series(data=coupon_amt, index=coupon_times)
    cash_flows.iloc[-1] += principal
    return cash_flows


def bond_price(maturity, principal=100, coupon_rate=0.03,
               coupons_per_year=12, discount_rate=0.03):
    """
    Computes the price of a coupon-bearing bond.
    If discount_rate is a DataFrame, returns a time series of prices.
    """
    if isinstance(discount_rate, pd.DataFrame):
        pricing_dates = discount_rate.index
        prices = pd.DataFrame(index=pricing_dates, columns=discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t] = bond_price(
                maturity - t / coupons_per_year,
                principal, coupon_rate, coupons_per_year,
                discount_rate.loc[t],
            )
        return prices
    if maturity <= 0:
        return principal + principal * coupon_rate / coupons_per_year
    cash_flows = bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
    return pv(cash_flows, discount_rate / coupons_per_year)


def macaulay_duration(flows, discount_rate):
    """Computes the Macaulay Duration of a sequence of cash flows."""
    discounted = discount(flows.index, discount_rate) * pd.DataFrame(flows)
    weights    = discounted / discounted.sum()
    return np.average(flows.index, weights=weights.iloc[:, 0])


def match_durations(cf_t, cf_s, cf_l, discount_rate):
    """
    Returns weight W in cf_s such that W*cf_s + (1-W)*cf_l
    has the same duration as cf_t.
    """
    d_t = macaulay_duration(cf_t, discount_rate)
    d_s = macaulay_duration(cf_s, discount_rate)
    d_l = macaulay_duration(cf_l, discount_rate)
    return (d_l - d_t) / (d_l - d_s)


def bond_total_return(monthly_prices, principal, coupon_rate, coupons_per_year):
    """
    Computes total return of a bond from monthly prices + reinvested coupons.
    """
    coupons = pd.DataFrame(data=0,
                           index=monthly_prices.index,
                           columns=monthly_prices.columns)
    t_max    = monthly_prices.index.max()
    pay_date = np.linspace(12 / coupons_per_year, t_max,
                           int(coupons_per_year * t_max / 12), dtype=int)
    coupons.iloc[pay_date] = principal * coupon_rate / coupons_per_year
    total_returns = (monthly_prices + coupons) / monthly_prices.shift() - 1
    return total_returns.dropna()


# ──────────────────────────────────────────────────────────────────────────────
# ALLOCATORS & BACKTESTS
# ──────────────────────────────────────────────────────────────────────────────

def bt_mix(r1, r2, allocator, **kwargs):
    """
    Backtest of a strategy that mixes two return streams r1 and r2.
    allocator is a function returning weights for r1 (T x N DataFrame).
    """
    if r1.shape != r2.shape:
        raise ValueError("r1 and r2 must have the same shape")
    weights = allocator(r1, r2, **kwargs)
    if weights.shape != r1.shape:
        raise ValueError("Allocator returned weights with wrong shape")
    return weights * r1 + (1 - weights) * r2


def fixedmix_allocator(r1, r2, w1, **kwargs):
    """Fixed allocation of w1 to r1 at every time step."""
    return pd.DataFrame(data=w1, index=r1.index, columns=r1.columns)


def glidepath_allocator(r1, r2, start_glide=1.0, end_glide=0.0):
    """Linearly decreasing allocation from start_glide to end_glide."""
    n_points = r1.shape[0]
    n_col    = r1.shape[1]
    path     = pd.Series(np.linspace(start_glide, end_glide, n_points))
    paths    = pd.concat([path] * n_col, axis=1)
    paths.index   = r1.index
    paths.columns = r1.columns
    return paths


def floor_allocator(psp_r, ghp_r, floor, zc_prices, m=3):
    """
    CPPI-style allocator with a floor based on zero-coupon bond prices.
    Returns a DataFrame of PSP weights (T x N).
    """
    if zc_prices.shape != psp_r.shape:
        raise ValueError("psp_r and zc_prices must have the same shape")
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
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


def drawdown_allocator(psp_r, ghp_r, maxdd, m=3):
    """
    CPPI-style allocator with a floor based on maximum drawdown constraint.
    Returns a DataFrame of PSP weights (T x N).
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


# ──────────────────────────────────────────────────────────────────────────────
# TERMINAL WEALTH STATISTICS
# ──────────────────────────────────────────────────────────────────────────────

def terminal_values(rets):
    """Terminal wealth per dollar invested across N scenarios."""
    return (rets + 1).prod()


def terminal_stats(rets, floor=0.8, cap=np.inf, name="Stats"):
    """
    Summary statistics on terminal wealth values.
    rets: T x N DataFrame of returns.
    """
    tw        = (rets + 1).prod()
    breach    = tw < floor
    reach     = tw >= cap
    p_breach  = breach.mean()               if breach.sum() > 0 else np.nan
    p_reach   = reach.mean()                if reach.sum()  > 0 else np.nan
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
