import yfinance as yf
import pandas as pd
import numpy as np


def load_adj_close(tickers, start, end) -> pd.DataFrame:
    
    # Download Adjusted Close prices for tickers between start and end,
    # align dates, drop missing values, and return prices (NOT logs)
    
    if tickers is None or len(tickers) == 0:
        raise ValueError("tickers must be a non-empty list")
    if start is None or end is None:
        raise ValueError("start and end must not be None")

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
    )

    if raw is None or len(raw) == 0:
        raise ValueError("No data returned from yfinance. Check tickers/date range.")

    # Handle MultiIndex vs single ticker formats
    if isinstance(raw.columns, pd.MultiIndex):
        adj = raw.loc[:, (slice(None), "Adj Close")]
        adj.columns = adj.columns.droplevel(1)
    else:
        # Single ticker case
        if "Adj Close" not in raw.columns:
            raise ValueError("Expected 'Adj Close' column in yfinance output.")
        adj = raw[["Adj Close"]].copy()
        # name the column as ticker
        adj.columns = [tickers[0]]

    adj = adj.sort_index()
    adj = adj.dropna(how="any")

    # prices must be positive for log transforms later
    if (adj <= 0).any().any():
        raise ValueError("Found non-positive adjusted close prices; cannot safely log.")

    return adj


def load_log_prices(tickers, start, end) -> pd.DataFrame:
    
    # Convenience wrapper: return aligned log prices
    
    prices = load_adj_close(tickers, start, end)
    return np.log(prices)


def daily_returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    
    # Simple close-to-close returns from price series
    
    if prices is None or prices.empty:
        raise ValueError("prices must be a non-empty DataFrame")
    rets = prices.pct_change()
    return rets
