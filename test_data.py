import numpy as np
import pandas as pd
from itertools import combinations

from src.data import load_log_prices
from src.hedge import backtest_pairs, engle_granger_pvalue_static


def performance_stats(returns: pd.Series, market: pd.Series = None):
    returns = returns.dropna()
    if len(returns) < 50:
        raise ValueError("Not enough return observations to compute meaningful stats.")

    equity = (1.0 + returns).cumprod()
    total_return = equity.iloc[-1] - 1.0
    ann_return = (1.0 + total_return) ** (252.0 / len(returns)) - 1.0
    ann_vol = returns.std(ddof=0) * np.sqrt(252.0)
    sharpe = np.nan if ann_vol == 0 else ann_return / ann_vol

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = drawdown.min()

    stats = {
        "Total Return": total_return,
        "Annualized Return": ann_return,
        "Annualized Vol": ann_vol,
        "Sharpe (no rf)": sharpe,
        "Max Drawdown": max_dd,
        "Days": len(returns),
    }

    if market is not None:
        market = market.reindex(returns.index).dropna()
        r, m = returns.align(market, join="inner")
        if len(r) > 50 and m.var(ddof=0) > 0:
            beta = np.cov(r, m, ddof=0)[0, 1] / m.var(ddof=0)
            stats["Beta vs Market"] = beta

    return stats


def main():
    # split data into train and test windows
    train_start = "2017-01-01"
    train_end = "2021-01-01"   # training is [train_start, train_end)
    test_start = "2021-01-01"
    test_end = "2024-01-01"

    # Ticker universe - this can be changed to look at various tickers
    universe = [
        "XOM","CVX","KO","PEP","V","MA","JPM","BAC","HD","LOW",
        "MSFT","AAPL","GOOGL","META","AMZN","NVDA","TSLA","UNH","PG","JNJ",
        "WMT","COST","DIS","NFLX","ORCL","CSCO","ADBE","CRM","INTC","AMD"
    ]

    # Include SPY for beta measurement in TEST
    tickers = sorted(set(universe + ["SPY"]))

    # Download once for whole span (train + test)
    prices_all = load_log_prices(tickers, train_start, test_end)

    # Split into train/test subsets
    prices_train = prices_all.loc[(prices_all.index >= train_start) & (prices_all.index < train_end), universe]
    prices_test = prices_all.loc[(prices_all.index >= test_start) & (prices_all.index < test_end), universe]

    # Fast cointegration scan on train window only
    significance = 0.05
    pairs = list(combinations(universe, 2))

    print("\n=== TRAIN: Cointegration scan (FAST Engle–Granger ADF) ===")
    results = []

    for i, (y_ticker, x_ticker) in enumerate(pairs, start=1):
        if i % 25 == 0:
            print(f"Checked {i}/{len(pairs)} pairs...")

        pval = engle_granger_pvalue_static(prices_train[y_ticker], prices_train[x_ticker])
        if (not np.isnan(pval)) and (pval < significance):
            results.append((pval, y_ticker, x_ticker))

    if not results:
        print("\nNo pairs passed cointegration on the training window.")
        print("Next move: expand universe, extend training window, or relax significance (e.g., 0.10).")
        return

    results.sort(key=lambda t: t[0])
    top_k = 5
    top_pairs = results[:top_k]

    print(f"\nTop {top_k} cointegrated pairs on TRAIN (lowest p-values):")
    for pval, y_t, x_t in top_pairs:
        print(f"{y_t:5s}-{x_t:5s}  p={pval:.4f}")

    # Backtest top pairs on test window
    print("\n=== TEST: Backtesting selected pairs (net of costs) ===")

    # Market returns for beta measurement on test
    spy_logret = prices_all["SPY"].loc[prices_test.index].diff()
    spy_ret = (np.exp(spy_logret) - 1.0)

    for pval, y_ticker, x_ticker in top_pairs:
        bt = backtest_pairs(
            log_prices=prices_test[[y_ticker, x_ticker]],
            y_ticker=y_ticker,
            x_ticker=x_ticker,
            lookback_beta=252,
            z_window=60,
            entry=2.0,
            exit=0.5,
            cost_bps=5.0,
        )

        strat = bt["strat_ret_net"]
        stats = performance_stats(strat, market=spy_ret)

        print(f"\nPair {y_ticker}-{x_ticker}  (train p={pval:.4f})")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"{k:18s}: {v: .4f}")
            else:
                print(f"{k:18s}: {v}")


if __name__ == "__main__":
    main()
