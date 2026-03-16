import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# Helpers

def _dropna_pair(y: pd.Series, x: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Align two series on a common index and drop any rows with NaNs.
    Returns aligned (y, x).
    """
    if y is None or x is None:
        raise ValueError("y and x must not be None")
    y, x = y.align(x, join="inner")
    df = pd.concat([y, x], axis=1).dropna()
    if df.shape[0] == 0:
        raise ValueError("No overlapping non-NaN data between y and x.")
    return df.iloc[:, 0], df.iloc[:, 1]

# Core regression + spread

def estimate_hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """
    Estimate alpha and beta in:
        y = alpha + beta * x
    Returns (alpha, beta).
    """
    y, x = _dropna_pair(y, x)
    if len(y) < 5:
        raise ValueError("Not enough overlapping data points to regress.")

    x_df = x.to_frame("x")
    x_with_const = sm.add_constant(x_df, has_constant="add")

    model = sm.OLS(y.values, x_with_const.values)
    results = model.fit()

    alpha = float(results.params[0])  # const
    beta = float(results.params[1])   # x
    return alpha, beta


def compute_spread(y: pd.Series, x: pd.Series, alpha: float, beta: float) -> pd.Series:
    
    # spread = y - (alpha + beta * x)
    
    y, x = y.align(x, join="inner")
    return y - (alpha + beta * x)


# Rolling estimation utilities

def rolling_hedge_ratio(y: pd.Series, x: pd.Series, lookback: int = 252) -> pd.DataFrame:
    """
    Walk-forward rolling estimation:
    For each date t, estimate alpha/beta using prior 'lookback' points (t-lookback ... t-1),
    then output alpha_t and beta_t at time t.

    Returns DataFrame with columns ['alpha', 'beta'] indexed by date.
    """
    if lookback < 20:
        raise ValueError("lookback is too small; use at least ~20+.")

    y, x = y.align(x, join="inner")
    alphas = pd.Series(index=y.index, dtype=float)
    betas = pd.Series(index=y.index, dtype=float)

    for i in range(lookback, len(y)):
        y_train = y.iloc[i - lookback:i]
        x_train = x.iloc[i - lookback:i]
        # keep training window NaN-free
        try:
            a, b = estimate_hedge_ratio(y_train, x_train)
            alphas.iloc[i] = a
            betas.iloc[i] = b
        except Exception:
            alphas.iloc[i] = np.nan
            betas.iloc[i] = np.nan

    return pd.DataFrame({"alpha": alphas, "beta": betas})


def rolling_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    """
    Rolling z-score with no lookahead.
    """
    if window < 5:
        raise ValueError("window too small")
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    return (series - mean) / std


def generate_spread_position(z: pd.Series, entry: float = 2.0, exit: float = 0.5) -> pd.Series:
    """
    Stateful entry/exit for spread position:
      +1 => long spread (long y, short x-hedge)
      -1 => short spread (short y, long x-hedge)
       0 => flat
    """
    pos = pd.Series(0.0, index=z.index)
    current = 0.0

    for i in range(len(z)):
        zi = z.iloc[i]
        if np.isnan(zi):
            pos.iloc[i] = current
            continue

        if current == 0.0:
            if zi > entry:
                current = -1.0
            elif zi < -entry:
                current = 1.0
        else:
            if abs(zi) < exit:
                current = 0.0

        pos.iloc[i] = current

    return pos


# Cointegration testing - Engle-Granger on residuals (ADF test)

def engle_granger_adf_pvalue(
    y: pd.Series,
    x: pd.Series,
    lookback: int = 252,
) -> pd.Series:
    """
    ROLLING Engle–Granger cointegration diagnostic (slow):
    For each t, fit y ~ alpha + beta*x on prior lookback window,
    then run ADF on residuals and store the p-value at t.
    """
    if lookback < 60:
        raise ValueError("lookback too small for cointegration testing; use 126+ or 252.")

    y, x = y.align(x, join="inner")
    pvals = pd.Series(index=y.index, dtype=float)

    for i in range(lookback, len(y)):
        y_train = y.iloc[i - lookback:i]
        x_train = x.iloc[i - lookback:i]

        try:
            y_train, x_train = _dropna_pair(y_train, x_train)
            a, b = estimate_hedge_ratio(y_train, x_train)
            resid = compute_spread(y_train, x_train, a, b).dropna()

            # degenerate residual guard
            if np.std(resid.values) < 1e-8:
                pvals.iloc[i] = np.nan
                continue

            pvals.iloc[i] = float(adfuller(resid.values, autolag="AIC", regression="c")[1])
        except Exception:
            pvals.iloc[i] = np.nan

    return pvals


def is_cointegrated_recent(
    y: pd.Series,
    x: pd.Series,
    lookback: int = 252,
    significance: float = 0.05,
    confirm_periods: int = 5,
) -> tuple[bool, float]:
    """
    Uses the rolling p-values and requires the most recent 'confirm_periods' to all be < significance.
    Returns (passed, latest_pvalue).
    """
    pvals = engle_granger_adf_pvalue(y, x, lookback=lookback)
    recent = pvals.dropna().tail(confirm_periods)

    if len(recent) < confirm_periods:
        return False, float("nan")

    latest_p = float(recent.iloc[-1])
    passed = bool((recent < significance).all())
    return passed, latest_p


def engle_granger_pvalue_static(y: pd.Series, x: pd.Series) -> float:
    """
    FAST Engle–Granger test on a single sample:
    1) Fit y ~ alpha + beta*x on the provided sample
    2) ADF test on residuals
    Returns ADF p-value (lower => more evidence of cointegration)
    """
    try:
        y, x = _dropna_pair(y, x)
    except Exception:
        return float("nan")

    if len(y) < 100:
        return float("nan")

    try:
        a, b = estimate_hedge_ratio(y, x)
        resid = compute_spread(y, x, a, b).dropna()

        if np.std(resid.values) < 1e-8:
            return float("nan")

        return float(adfuller(resid.values, autolag="AIC", regression="c")[1])
    except Exception:
        return float("nan")

# Backtest engine (two assets)

def backtest_pairs(
    log_prices: pd.DataFrame,
    y_ticker: str,
    x_ticker: str,
    lookback_beta: int = 252,
    z_window: int = 60,
    entry: float = 2.0,
    exit: float = 0.5,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Walk-forward pairs backtest using log prices:
    - rolling alpha/beta (lookback_beta)
    - spread + rolling z-score (z_window)
    - stateful entry/exit signals
    - normalized gross exposure
    - positions lagged by 1 day (no lookahead)
    - simple turnover-based transaction costs

    Returns DataFrame including strat_ret_net.
    """
    if y_ticker not in log_prices.columns or x_ticker not in log_prices.columns:
        raise ValueError("tickers not found in log_prices columns")

    y = log_prices[y_ticker].copy()
    x = log_prices[x_ticker].copy()
    y, x = y.align(x, join="inner")

    hr = rolling_hedge_ratio(y, x, lookback=lookback_beta)
    alpha = hr["alpha"]
    beta = hr["beta"]

    spread = y - (alpha + beta * x)
    z = rolling_zscore(spread, window=z_window)
    spread_pos = generate_spread_position(z, entry=entry, exit=exit)

    # Leg weights (normalized gross exposure)
    w_y = pd.Series(0.0, index=spread_pos.index)
    w_x = pd.Series(0.0, index=spread_pos.index)

    for i in range(len(spread_pos)):
        p = float(spread_pos.iloc[i])
        b = beta.iloc[i]

        if np.isnan(p) or np.isnan(b):
            continue

        wy_raw = p * 1.0
        wx_raw = -p * float(b)

        gross = abs(wy_raw) + abs(wx_raw)
        if gross == 0:
            continue

        scale = 2.0 / gross
        w_y.iloc[i] = wy_raw * scale
        w_x.iloc[i] = wx_raw * scale

    # Convert log prices to simple returns
    ret_y = np.exp(y.diff()) - 1.0
    ret_x = np.exp(x.diff()) - 1.0

    # No lookahead: yesterday's weights applied to today's returns
    strat_ret = w_y.shift(1) * ret_y + w_x.shift(1) * ret_x

    # Transaction costs: bps * turnover
    turnover = (w_y.diff().abs() + w_x.diff().abs()).fillna(0.0)
    cost = (cost_bps / 10000.0) * turnover
    strat_ret_net = strat_ret - cost

    out = pd.DataFrame(
        {
            "alpha": alpha,
            "beta": beta,
            "spread": spread,
            "z": z,
            "spread_pos": spread_pos,
            "w_y": w_y,
            "w_x": w_x,
            "ret_y": ret_y,
            "ret_x": ret_x,
            "strat_ret": strat_ret,
            "turnover": turnover,
            "cost": cost,
            "strat_ret_net": strat_ret_net,
        }
    )

    return out.dropna(subset=["strat_ret_net"])
