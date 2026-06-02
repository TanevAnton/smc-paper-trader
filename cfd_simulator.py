"""
cfd_simulator.py — leveraged CFD simulator for a small EU retail account.

PURPOSE
-------
Show, concretely, what trading Nasdaq-100 CFDs with a EUR4,000 account looks
like under realistic EU/ESMA rules — including leverage, stop-based position
sizing, overnight financing, margin close-out, and (most importantly) the
PROBABILITY OF RUIN at different risk-per-trade settings.

This is a research/education tool, NOT financial advice. Synthetic demo data
is used when you don't supply a real CSV; demo P&L is illustrative only, but
the risk-of-ruin comparison is driven by win-rate/payoff assumptions and is
the part you should pay attention to.

DATA
----
Real data:  python3 cfd_simulator.py --csv NDX.csv
Demo data:  python3 cfd_simulator.py --demo
CSV format: Yahoo Finance export (Date + Close/Adj Close).

KEY EU/ESMA DEFAULTS (you can override)
---------------------------------------
  --max-leverage 20      major stock index cap is 1:20
  --account 4000         EUR4,000 starting capital
  --margin-close 0.50    broker force-closes at 50% of required margin
  --financing 0.065      ~6.5% annualized overnight financing on exposure
"""

import argparse
import sys
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_csv(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date")
    price_col = cols.get("adj close") or cols.get("close")
    if not date_col or not price_col:
        sys.exit("ERROR: CSV needs a Date column and a Close/Adj Close column.")
    df = df[[date_col, price_col]].copy()
    df.columns = ["date", "price"]
    df["date"] = pd.to_datetime(df["date"])
    return df.dropna().sort_values("date").reset_index(drop=True)


def make_nasdaq_like(days=1250, seed=11, annual_drift=0.10, annual_vol=0.24):
    """Synthetic series with Nasdaq-ish drift/volatility. Demo only."""
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    shocks = rng.normal((annual_drift - 0.5 * annual_vol**2) * dt,
                        annual_vol * np.sqrt(dt), days)
    price = 15000 * np.exp(np.cumsum(shocks))
    dates = pd.bdate_range("2021-01-01", periods=days)
    return pd.DataFrame({"date": dates, "price": price})


# ----------------------------------------------------------------------------
# Strategy signal: trend-following long/short via SMA crossover.
# +1 = long, -1 = short, 0 = flat. Acted on next day (no look-ahead).
# ----------------------------------------------------------------------------
def signal_sma(price, fast=20, slow=100):
    f = price.rolling(fast).mean()
    s = price.rolling(slow).mean()
    sig = pd.Series(0.0, index=price.index)
    sig[f > s] = 1.0
    sig[f < s] = -1.0
    return sig.shift(1).fillna(0.0)  # trade on yesterday's signal


# ----------------------------------------------------------------------------
# Leveraged CFD backtest with stop-based position sizing
# ----------------------------------------------------------------------------
def run_cfd(df, risk_pct, max_leverage=20, account=4000.0,
            stop_atr_mult=2.0, atr_window=14, financing=0.065,
            margin_close=0.50, fast=20, slow=100):
    """
    Position sizing: risk `risk_pct` of CURRENT equity per trade. The stop
    distance (in price %) is set from recent volatility (ATR-like). Exposure =
    risk_amount / stop_distance, capped by leverage. P&L accrues daily on
    exposure; overnight financing is charged daily; if equity falls below the
    margin-close threshold the account is closed (ruin).
    """
    price = df["price"].reset_index(drop=True)
    ret = price.pct_change().fillna(0.0)
    # Simple volatility proxy for stop distance (% of price).
    vol_pct = ret.rolling(atr_window).std().bfill().clip(lower=0.003)
    stop_dist = (stop_atr_mult * vol_pct).clip(lower=0.005)  # min 0.5% stop

    sig = signal_sma(price, fast, slow)

    equity = account
    exposure = 0.0          # signed EUR notional currently held
    direction = 0.0
    peak = account
    max_dd = 0.0
    n_trades = 0
    ruined = False
    eq_curve = []

    for i in range(len(price)):
        # Daily P&L on existing exposure.
        equity += exposure * ret.iloc[i]
        # Overnight financing cost on absolute exposure (charged when holding).
        equity -= abs(exposure) * (financing / 252)

        peak = max(peak, equity)
        dd = equity / peak - 1
        max_dd = min(max_dd, dd)

        # Ruin: negative balance protection floors equity at 0, and once
        # ~90% of the account is gone it's effectively unrecoverable, so we
        # close it out.
        if equity <= account * 0.10:
            equity = max(equity, 0.0)
            ruined = True
            eq_curve.append(equity)
            exposure = 0.0
            break

        target_dir = sig.iloc[i]
        # Rebalance / open / close when direction changes.
        if target_dir != direction:
            direction = target_dir
            if direction == 0.0:
                exposure = 0.0
            else:
                risk_amount = equity * risk_pct
                notional = risk_amount / stop_dist.iloc[i]
                cap = equity * max_leverage
                notional = min(notional, cap)
                exposure = direction * notional
                n_trades += 1
        eq_curve.append(equity)

    eq = pd.Series(eq_curve)
    total_return = equity / account - 1
    years = len(price) / 252
    cagr = (max(equity, 1e-9) / account) ** (1 / years) - 1 if years > 0 else np.nan
    return {
        "risk_pct": risk_pct,
        "final_equity": equity,
        "total_return": total_return,
        "cagr": cagr,
        "max_dd": max_dd,
        "n_trades": n_trades,
        "ruined": ruined,
        "equity_curve": eq,
    }


# ----------------------------------------------------------------------------
# Risk-of-ruin Monte Carlo (trade-level)
# ----------------------------------------------------------------------------
def monte_carlo_ruin(risk_pct, win_rate=0.50, payoff=1.5, n_trades=200,
                     n_paths=20000, ruin_dd=0.50, seed=1):
    """
    Simulate many sequences of `n_trades` trades. Each trade risks `risk_pct`
    of current equity: a win gains risk_pct*payoff, a loss drops risk_pct.
    Returns probability of hitting `ruin_dd` drawdown, plus median outcome.

    payoff = average win / average loss (reward-to-risk). win_rate and payoff
    are the two assumptions that drive everything — change them to your edge.
    """
    rng = np.random.default_rng(seed)
    wins = rng.random((n_paths, n_trades)) < win_rate
    step = np.where(wins, risk_pct * payoff, -risk_pct)
    equity = np.ones(n_paths)
    peak = np.ones(n_paths)
    ruined = np.zeros(n_paths, dtype=bool)
    for t in range(n_trades):
        equity *= (1 + step[:, t])
        equity = np.maximum(equity, 0.0)
        peak = np.maximum(peak, equity)
        dd = equity / np.where(peak == 0, 1, peak) - 1
        ruined |= dd <= -ruin_dd
    return {
        "risk_pct": risk_pct,
        "p_ruin": ruined.mean(),
        "median_final_mult": float(np.median(equity)),
        "p_lose_money": float((equity < 1.0).mean()),
    }


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Leveraged Nasdaq CFD simulator.")
    p.add_argument("--csv")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--account", type=float, default=4000.0)
    p.add_argument("--max-leverage", type=float, default=20.0)
    p.add_argument("--financing", type=float, default=0.065)
    p.add_argument("--fast", type=int, default=20)
    p.add_argument("--slow", type=int, default=100)
    # Monte Carlo edge assumptions
    p.add_argument("--win-rate", type=float, default=0.50)
    p.add_argument("--payoff", type=float, default=1.5)
    p.add_argument("--mc-trades", type=int, default=200)
    args = p.parse_args()

    if args.csv:
        df = load_csv(args.csv)
        print(f"Loaded {len(df)} rows from {args.csv}.")
    elif args.demo:
        df = make_nasdaq_like()
        print("Using SYNTHETIC Nasdaq-like data — equity P&L is illustrative; "
              "focus on the risk-of-ruin table.")
    else:
        sys.exit("Provide --csv <file> or --demo.")

    risk_levels = [0.01, 0.02, 0.05, 0.10]

    print("\n" + "=" * 72)
    print(f"  LEVERAGED BACKTEST — Nasdaq CFD, EUR{args.account:,.0f}, "
          f"max leverage 1:{args.max_leverage:.0f}")
    print("=" * 72)
    print(f"  {'risk/trade':>10} | {'final EUR':>10} | {'total':>8} | "
          f"{'CAGR':>8} | {'max DD':>8} | {'trades':>6} | ruined?")
    print("  " + "-" * 68)
    for r in risk_levels:
        res = run_cfd(df, r, args.max_leverage, args.account,
                      financing=args.financing, fast=args.fast, slow=args.slow)
        print(f"  {r*100:>9.0f}% | {res['final_equity']:>10,.0f} | "
              f"{res['total_return']*100:>7.0f}% | {res['cagr']*100:>7.1f}% | "
              f"{res['max_dd']*100:>7.1f}% | {res['n_trades']:>6} | "
              f"{'YES' if res['ruined'] else 'no'}")

    print("\n" + "=" * 72)
    print(f"  RISK OF RUIN (Monte Carlo, {args.mc_trades} trades, "
          f"win rate {args.win_rate*100:.0f}%, payoff {args.payoff}:1)")
    print("  'Ruin' = hitting a 50% drawdown. 20,000 simulated careers each.")
    print("=" * 72)
    print(f"  {'risk/trade':>10} | {'P(ruin)':>9} | {'P(lose money)':>14} | "
          f"{'median outcome':>15}")
    print("  " + "-" * 64)
    for r in risk_levels:
        mc = monte_carlo_ruin(r, args.win_rate, args.payoff, args.mc_trades)
        print(f"  {r*100:>9.0f}% | {mc['p_ruin']*100:>8.1f}% | "
              f"{mc['p_lose_money']*100:>13.1f}% | "
              f"{mc['median_final_mult']:>14.2f}x")
    print("=" * 72)
    print("  How to read this:")
    print("   - P(ruin) is the share of simulated trading 'careers' that lost")
    print("     half the account. At high risk-per-trade it climbs fast even")
    print("     with a genuine edge.")
    print("   - 'median outcome' is the typical ending account multiple. Below")
    print("     1.00x means you most likely end with less than you started.")
    print("   - These assume you actually HAVE an edge (win rate & payoff). If")
    print("     you don't, every column gets worse. Most retail traders don't.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
