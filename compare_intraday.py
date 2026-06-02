"""
compare_intraday.py — side-by-side report of intraday SMC variants on the same
5-minute QQQ data.

Each variant uses the same underlying engine (smc_intraday.backtest_intraday);
only the filter switches and a couple of param dials change. This isolates
what each layer is actually contributing.
"""

import argparse
import numpy as np
import pandas as pd

from smc_intraday import backtest_intraday, load_intraday, bars_per_year


def stats_from_curve(eq, df, account):
    L = len(eq)
    bpy = bars_per_year(df)
    years = L / bpy
    final = eq[-1]
    cagr = (max(final, 1e-9) / account) ** (1 / years) - 1 if years > 0 else 0.0
    bar_ret = pd.Series(eq).pct_change().fillna(0.0)
    sharpe = (bar_ret.mean() / bar_ret.std()) * np.sqrt(bpy) if bar_ret.std() > 0 else 0.0
    sortino_d = bar_ret[bar_ret < 0].std()
    sortino = (bar_ret.mean() / sortino_d) * np.sqrt(bpy) if sortino_d > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    mdd = (eq / peak - 1).min()
    # Daily aggregation.
    daily_eq = pd.Series(eq, index=pd.to_datetime(df["datetime"]).dt.date.values)
    end_of_day = daily_eq.groupby(daily_eq.index).last()
    daily_pnl = end_of_day.diff().fillna(end_of_day.iloc[0] - account)
    pct_green = (daily_pnl > 0).mean()
    return {
        "Final $": round(final, 0),
        "Tot %": f"{(final / account - 1) * 100:.2f}%",
        "CAGR %": f"{cagr * 100:.1f}%",
        "Sharpe": round(sharpe, 2),
        "Sortino": round(sortino, 2),
        "MaxDD %": f"{mdd * 100:.2f}%",
        "Vol %": f"{bar_ret.std() * np.sqrt(bpy) * 100:.1f}%",
        "Avg daily $": round(daily_pnl.mean(), 2),
        "Med daily $": round(daily_pnl.median(), 2),
        "% green dy": f"{pct_green * 100:.0f}%",
        "Best dy $": round(daily_pnl.max(), 2),
        "Worst dy $": round(daily_pnl.min(), 2),
    }


def trade_stats(trades):
    if trades is None or len(trades) == 0:
        return {"Trades": 0, "Win %": "n/a", "PF": "n/a", "Exp $": 0}
    wins = (trades["pnl_$"] > 0).sum()
    wpnl = trades.loc[trades["pnl_$"] > 0, "pnl_$"].sum()
    lpnl = -trades.loc[trades["pnl_$"] <= 0, "pnl_$"].sum()
    pf = wpnl / lpnl if lpnl > 0 else float("inf")
    return {
        "Trades": len(trades),
        "Win %": f"{wins / len(trades) * 100:.1f}%",
        "PF": round(pf, 2),
        "Exp $": round(trades["pnl_$"].mean(), 2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="QQQ_5m.csv")
    p.add_argument("--cash", type=float, default=10000.0)
    args = p.parse_args()

    df = load_intraday(args.csv)
    print(f"Loaded {len(df)} RTH bars  "
          f"({df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]})  "
          f"{df['date'].nunique()} trading days")

    # Buy & hold benchmark.
    bh = df["close"].pct_change().fillna(0.0)
    bh_eq = ((1 + bh).cumprod() * args.cash).values

    variants = [
        ("Buy & Hold QQQ", None),

        ("v3 strict (all filters on)",
         dict()),  # defaults
        ("v3 no session-confluence",
         dict(use_session_confluence=False)),
        ("v3 no PD + no session",
         dict(use_session_confluence=False, use_pd_filter=False)),
        ("v3 wide + no PD/session  [PRIMARY]",
         dict(use_session_confluence=False, use_pd_filter=False,
              min_fvg_pct=0.0005, max_bars_after_bos=24)),
        ("v3 wide L+S, EOD flat",
         dict(use_session_confluence=False, use_pd_filter=False,
              min_fvg_pct=0.0005, max_bars_after_bos=24,
              flatten_at_session_close=True)),
        ("v3 wide long-only",
         dict(use_session_confluence=False, use_pd_filter=False,
              min_fvg_pct=0.0005, max_bars_after_bos=24,
              allow_short=False)),
        ("v3 wide @ 1.5% risk",
         dict(use_session_confluence=False, use_pd_filter=False,
              min_fvg_pct=0.0005, max_bars_after_bos=24,
              risk_per_trade=0.015)),
        ("v3 no trend filter (BOS-only)",
         dict(use_session_confluence=False, use_pd_filter=False,
              use_trend_filter=False, min_fvg_pct=0.0005,
              max_bars_after_bos=24)),
    ]

    rows = {}
    for label, kw in variants:
        if label == "Buy & Hold QQQ":
            rows[label] = {**stats_from_curve(bh_eq, df, args.cash),
                           "Trades": 1, "Win %": "n/a", "PF": "n/a", "Exp $": 0}
            continue
        trades, eq = backtest_intraday(df, account=args.cash, **kw)
        rows[label] = {**stats_from_curve(eq, df, args.cash),
                       **trade_stats(trades)}

    out = pd.DataFrame(rows).T
    cols = ["Final $", "Tot %", "CAGR %", "Sharpe", "Sortino",
            "MaxDD %", "Vol %", "Avg daily $", "Med daily $",
            "% green dy", "Best dy $", "Worst dy $",
            "Trades", "Win %", "PF", "Exp $"]
    out = out[cols]
    print("\n" + "=" * 160)
    print("  INTRADAY SMC — variant comparison on QQQ 5m, 60 trading days")
    print("=" * 160)
    with pd.option_context("display.width", 220, "display.max_colwidth", 36):
        print(out.to_string())
    print("=" * 160)
    print("  Sample size caveat: 60 trading days is a SHORT backtest. Use these")
    print("  numbers to compare variants against each other, not to forecast.")
    print("  The Feb-May 2026 window was a strong bull leg (QQQ +16%), so")
    print("  Buy & Hold sets a high bar on absolute return — but a strategy that")
    print("  matches it with 1/4 the drawdown is much more leverageable.")
    print("=" * 160)


if __name__ == "__main__":
    main()
