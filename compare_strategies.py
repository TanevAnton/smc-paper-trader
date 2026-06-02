"""
compare_strategies.py — head-to-head of every strategy variant on the same
daily NDX series, printed as a single decision table.

Variants compared:
  - Buy & Hold (NDX)
  - SMC v1 (original)
  - SMC v2 — conservative (default filters, risk 0.75%)
  - SMC v2 — tuned       (sweep-best params, risk 0.75%)
  - SMC v2 — aggressive  (sweep-best params, risk 2.00%)
  - SMC v2 — long-only   (no shorts, tuned)

The comparison uses the same starting cash, costs, and date range. All numbers
come from the same equity-curve metric functions so they're directly
comparable.

Usage:
    python compare_strategies.py
    python compare_strategies.py --csv NDX.csv --cash 1000 --from-date 2010-01-01
"""

import argparse
import numpy as np
import pandas as pd

from smc_strategy import backtest as backtest_v1, load_ohlc as load_v1
from smc_strategy_v2 import backtest_v2, load_ohlc as load_v2


def metrics_from_curve(eq, df, account):
    L = len(eq)
    years = L / 252
    final = eq[-1]
    daily = pd.Series(eq).pct_change().fillna(0.0)
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak) - 1
    sharpe = (daily.mean() / daily.std()) * np.sqrt(252) if daily.std() > 0 else 0.0
    sortino_d = daily[daily < 0].std()
    sortino = (daily.mean() / sortino_d) * np.sqrt(252) if sortino_d > 0 else 0.0
    cagr = (max(final, 1e-9) / account) ** (1 / years) - 1 if years > 0 else 0.0
    avg_daily = daily.mean()
    pct_up = (daily > 0).mean()
    days_in_market = (daily.abs() > 1e-12).mean()
    return {
        "Final $": round(final, 0),
        "Total %": f"{(final/account - 1)*100:.1f}%",
        "CAGR %": f"{cagr*100:.2f}%",
        "Sharpe": round(sharpe, 2),
        "Sortino": round(sortino, 2),
        "MaxDD %": f"{dd.min()*100:.1f}%",
        "Vol %": f"{daily.std()*np.sqrt(252)*100:.1f}%",
        "Avg daily %": f"{avg_daily*100:.4f}%",
        "% up days": f"{pct_up*100:.1f}%",
        "% in mkt": f"{days_in_market*100:.0f}%",
    }


def trade_stats(trades):
    if trades is None or len(trades) == 0:
        return {"Trades": 0, "Win %": "n/a", "PF": "n/a"}
    wins = (trades["pnl_$"] > 0).sum()
    wr = wins / len(trades)
    wpnl = trades.loc[trades["pnl_$"] > 0, "pnl_$"].sum()
    lpnl = -trades.loc[trades["pnl_$"] <= 0, "pnl_$"].sum()
    pf = wpnl / lpnl if lpnl > 0 else float("inf")
    return {
        "Trades": len(trades),
        "Win %": f"{wr*100:.1f}%",
        "PF": round(pf, 2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="NDX.csv")
    p.add_argument("--cash", type=float, default=1000.0)
    p.add_argument("--from-date")
    p.add_argument("--to-date")
    args = p.parse_args()

    df = load_v2(args.csv)
    if args.from_date:
        df = df[df["date"] >= pd.Timestamp(args.from_date)].reset_index(drop=True)
    if args.to_date:
        df = df[df["date"] <= pd.Timestamp(args.to_date)].reset_index(drop=True)
    print(f"Loaded {len(df)} rows  ({df['date'].iloc[0].date()} -> "
          f"{df['date'].iloc[-1].date()})")

    rows = {}

    # 1) Buy & hold
    bh = df["close"].pct_change().fillna(0.0)
    bh_eq = ((1 + bh).cumprod() * args.cash).values
    rows["Buy & Hold NDX"] = {**metrics_from_curve(bh_eq, df, args.cash),
                              "Trades": 1, "Win %": "n/a", "PF": "n/a"}

    # 2) v1 SMC
    v1_trades, v1_eq = backtest_v1(df, swing_n=3, min_fvg_pct=0.0015, rr=2.0,
                                   stop_buf=0.5, atr_n=14, max_hold=20,
                                   risk_per_trade=0.01, cost_bps=5.0,
                                   account=args.cash, allow_short=True)
    rows["SMC v1"] = {**metrics_from_curve(v1_eq, df, args.cash),
                      **trade_stats(v1_trades)}

    # 3) v2 — conservative (defaults)
    t2c, e2c = backtest_v2(df, account=args.cash)
    rows["SMC v2 conservative"] = {**metrics_from_curve(e2c, df, args.cash),
                                    **trade_stats(t2c)}

    # 4) v2 — tuned (sweep best, risk 0.75%)
    t2t, e2t = backtest_v2(
        df, swing_n=3, rr_runner=5.0, max_bars_after_bos=20,
        min_fvg_pct=0.001, risk_per_trade=0.0075, account=args.cash,
    )
    rows["SMC v2 tuned"] = {**metrics_from_curve(e2t, df, args.cash),
                            **trade_stats(t2t)}

    # 5) v2 — aggressive (tuned + risk 2%)
    t2a, e2a = backtest_v2(
        df, swing_n=3, rr_runner=5.0, max_bars_after_bos=20,
        min_fvg_pct=0.001, risk_per_trade=0.02, account=args.cash,
    )
    rows["SMC v2 aggressive"] = {**metrics_from_curve(e2a, df, args.cash),
                                  **trade_stats(t2a)}

    # 6) v2 — long-only (tuned)
    t2l, e2l = backtest_v2(
        df, swing_n=3, rr_runner=5.0, max_bars_after_bos=20,
        min_fvg_pct=0.001, risk_per_trade=0.02, account=args.cash,
        allow_short=False,
    )
    rows["SMC v2 long-only +2%"] = {**metrics_from_curve(e2l, df, args.cash),
                                     **trade_stats(t2l)}

    # Pretty-print
    df_out = pd.DataFrame(rows).T
    cols = ["Final $", "Total %", "CAGR %", "Sharpe", "Sortino",
            "MaxDD %", "Vol %", "Avg daily %", "% up days", "% in mkt",
            "Trades", "Win %", "PF"]
    df_out = df_out[cols]
    print("\n" + "=" * 110)
    print("  STRATEGY COMPARISON — daily NDX, identical costs/sizing assumptions")
    print("=" * 110)
    with pd.option_context("display.width", 200,
                           "display.max_colwidth", 24):
        print(df_out.to_string())
    print("=" * 110)
    print("  How to read this:")
    print("   - CAGR  = annualized return.  Sharpe  = return per unit total vol.")
    print("   - MaxDD = worst peak-to-trough loss along the way.")
    print("   - 'Avg daily %' is your average DAILY equity move — the closest")
    print("     thing to 'daily profitability'. Higher is better, but on its own")
    print("     it's meaningless: pair it with MaxDD and Sharpe.")
    print("   - SMC variants spend most days flat, so 'avg daily' looks small;")
    print("     they earn in bursts. The right comparison is RISK-ADJUSTED return")
    print("     (Sharpe, Sortino), not raw CAGR vs a 100%-invested benchmark.")
    print("=" * 110)


if __name__ == "__main__":
    main()
