"""
backtester.py — a small, honest stock/ETF backtesting framework.

WHAT THIS DOES
--------------
You give it daily price data and pick a strategy. It simulates trading the
strategy over the whole history and reports REALISTIC performance metrics so
you can see what the strategy actually would have done — including the losing
stretches — before risking any real money.

WHAT IT WILL NOT DO
-------------------
It will not promise "10% per day" or "consistent daily profit." No real
strategy does that. This tool exists to show you the truth: total return,
how bumpy the ride is (drawdown), and whether the edge is real (Sharpe ratio).

DATA INPUT
----------
Option A — real data (recommended):
    Download a CSV from Yahoo Finance (visit a ticker page, e.g. SPY, click
    "Historical Data" -> "Download"). It has columns: Date, Open, High, Low,
    Close, Adj Close, Volume. Then run:
        python3 backtester.py --csv SPY.csv --strategy sma
Option B — demo mode (no data needed, runs immediately):
        python3 backtester.py --demo --strategy sma
    This generates synthetic random-walk prices just so you can see the
    machinery work. Demo numbers are meaningless for real decisions.

STRATEGIES INCLUDED
-------------------
    buyhold : buy on day 1, hold to the end (the benchmark to beat)
    sma     : simple moving-average crossover (long when fast SMA > slow SMA)
    momentum: long when the trailing N-day return is positive

This is a teaching/research tool. It is NOT financial advice.
"""

import argparse
import sys
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_csv(path):
    """Load an OHLCV CSV (Yahoo Finance format) and return a clean price series."""
    df = pd.read_csv(path)
    # Be flexible about column names.
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date")
    # Prefer adjusted close (accounts for dividends/splits); fall back to close.
    price_col = cols.get("adj close") or cols.get("close")
    if date_col is None or price_col is None:
        sys.exit("ERROR: CSV must have a Date column and a Close/Adj Close column.")
    df = df[[date_col, price_col]].copy()
    df.columns = ["date", "price"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna().sort_values("date").reset_index(drop=True)
    if len(df) < 50:
        sys.exit("ERROR: need at least ~50 rows of data to backtest meaningfully.")
    return df


def make_demo_data(days=1000, seed=42, annual_drift=0.07, annual_vol=0.20):
    """Synthetic daily prices via geometric Brownian motion. For demo only."""
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    mu, sigma = annual_drift, annual_vol
    shocks = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), days)
    price = 100 * np.exp(np.cumsum(shocks))
    dates = pd.bdate_range("2021-01-01", periods=days)
    return pd.DataFrame({"date": dates, "price": price})


# ----------------------------------------------------------------------------
# Strategies — each returns a "position" series: 1.0 = fully long, 0.0 = in cash.
# Positions are SHIFTED by one day at execution time so we only act on
# information available yesterday (no look-ahead cheating).
# ----------------------------------------------------------------------------
def strat_buyhold(price):
    return pd.Series(1.0, index=price.index)


def strat_sma(price, fast=20, slow=100):
    fast_ma = price.rolling(fast).mean()
    slow_ma = price.rolling(slow).mean()
    return (fast_ma > slow_ma).astype(float)


def strat_momentum(price, lookback=60):
    trailing_return = price / price.shift(lookback) - 1
    return (trailing_return > 0).astype(float)


STRATEGIES = {
    "buyhold": strat_buyhold,
    "sma": strat_sma,
    "momentum": strat_momentum,
}


# ----------------------------------------------------------------------------
# Backtest engine
# ----------------------------------------------------------------------------
def backtest(df, strategy_fn, cost_per_trade=0.0005, starting_cash=1000.0):
    """
    Simulate the strategy.

    cost_per_trade: round-trip-ish friction applied whenever the position
                    changes, expressed as a fraction of traded value
                    (0.0005 = 5 basis points). Models commission + spread + slippage.
    """
    price = df["price"].reset_index(drop=True)
    daily_ret = price.pct_change().fillna(0.0)

    # Raw target position from the strategy.
    target = strategy_fn(price).fillna(0.0)
    # Act on yesterday's signal: shift positions forward one day.
    position = target.shift(1).fillna(0.0)

    # Strategy daily return = position held * that day's asset return.
    strat_ret = position * daily_ret

    # Subtract trading costs whenever the position changes.
    turnover = position.diff().abs().fillna(0.0)
    strat_ret = strat_ret - turnover * cost_per_trade

    # Build equity curves.
    equity = starting_cash * (1 + strat_ret).cumprod()
    bench_equity = starting_cash * (1 + daily_ret).cumprod()

    out = df.copy().reset_index(drop=True)
    out["position"] = position
    out["strat_ret"] = strat_ret
    out["equity"] = equity
    out["bench_equity"] = bench_equity
    n_trades = int((turnover > 0).sum())
    return out, n_trades


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def metrics(out, n_trades, starting_cash=1000.0):
    ret = out["strat_ret"]
    equity = out["equity"]
    days = len(out)
    years = days / 252

    total_return = equity.iloc[-1] / starting_cash - 1
    cagr = (equity.iloc[-1] / starting_cash) ** (1 / years) - 1 if years > 0 else np.nan

    # Sharpe (annualized, risk-free assumed 0 for simplicity).
    daily_mean, daily_std = ret.mean(), ret.std()
    sharpe = (daily_mean / daily_std) * np.sqrt(252) if daily_std > 0 else np.nan

    ann_vol = daily_std * np.sqrt(252)

    # Max drawdown — the worst peak-to-trough drop. This is the "pain" metric.
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()

    # Day-level win rate, counting only days we held a position.
    active = ret[out["position"] > 0]
    win_rate = (active > 0).mean() if len(active) > 0 else np.nan

    bench_total = out["bench_equity"].iloc[-1] / starting_cash - 1

    return {
        "Days tested": days,
        "Years": round(years, 2),
        "Final equity ($)": round(equity.iloc[-1], 2),
        "Total return": f"{total_return*100:.1f}%",
        "Annualized (CAGR)": f"{cagr*100:.1f}%",
        "Buy & hold total return": f"{bench_total*100:.1f}%",
        "Annualized volatility": f"{ann_vol*100:.1f}%",
        "Sharpe ratio": round(sharpe, 2),
        "Max drawdown": f"{max_dd*100:.1f}%",
        "Daily win rate (when in market)": f"{win_rate*100:.1f}%",
        "Number of trades": n_trades,
    }


def print_report(name, m, starting_cash):
    print("\n" + "=" * 60)
    print(f"  BACKTEST REPORT  —  strategy: {name}")
    print("=" * 60)
    print(f"  Starting cash: ${starting_cash:,.2f}")
    for k, v in m.items():
        print(f"  {k:<34}: {v}")
    print("=" * 60)
    print("  Reading the numbers:")
    print("   - CAGR is your realistic annual growth rate, not a daily one.")
    print("   - Max drawdown is how much you'd have lost at the worst point.")
    print("     A small account must be able to stomach this without panic.")
    print("   - Sharpe > 1 is good, > 2 is excellent and rare. Below ~0.5")
    print("     the 'edge' is probably just noise.")
    print("   - Beating buy & hold is the real bar. Many strategies don't.")
    print("=" * 60 + "\n")


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Honest stock/ETF backtester.")
    p.add_argument("--csv", help="Path to OHLCV CSV (Yahoo Finance format).")
    p.add_argument("--demo", action="store_true", help="Use synthetic demo data.")
    p.add_argument("--strategy", default="sma", choices=list(STRATEGIES.keys()))
    p.add_argument("--cash", type=float, default=1000.0, help="Starting capital.")
    p.add_argument("--cost", type=float, default=0.0005,
                   help="Per-trade cost as a fraction (0.0005 = 5 bps).")
    # Strategy params
    p.add_argument("--fast", type=int, default=20)
    p.add_argument("--slow", type=int, default=100)
    p.add_argument("--lookback", type=int, default=60)
    args = p.parse_args()

    if args.csv:
        df = load_csv(args.csv)
        print(f"Loaded {len(df)} rows from {args.csv} "
              f"({df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()})")
    elif args.demo:
        df = make_demo_data()
        print("Using SYNTHETIC demo data — results are illustrative only, "
              "not a real market.")
    else:
        sys.exit("Provide --csv <file> for real data, or --demo to try it out.")

    fn_base = STRATEGIES[args.strategy]
    if args.strategy == "sma":
        strategy_fn = lambda pr: strat_sma(pr, args.fast, args.slow)
        label = f"sma({args.fast}/{args.slow})"
    elif args.strategy == "momentum":
        strategy_fn = lambda pr: strat_momentum(pr, args.lookback)
        label = f"momentum({args.lookback})"
    else:
        strategy_fn = fn_base
        label = args.strategy

    out, n_trades = backtest(df, strategy_fn, args.cost, args.cash)
    m = metrics(out, n_trades, args.cash)
    print_report(label, m, args.cash)


if __name__ == "__main__":
    main()
