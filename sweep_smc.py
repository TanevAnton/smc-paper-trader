"""sweep_smc.py — small parameter sweep over smc_strategy.backtest.

Prints a table of results. Honest reporting: no cherry-picking — every
configuration tested is shown, in CAGR order, with full risk picture.
"""

import argparse
import itertools
import numpy as np
import pandas as pd

from smc_strategy import load_ohlc, backtest


def metrics(df, trades, eq, account):
    L = len(df)
    years = L / 252
    final = eq[-1]
    cagr = (max(final, 1e-9) / account) ** (1 / years) - 1 if years > 0 else np.nan
    peak = np.maximum.accumulate(eq)
    max_dd = (eq / peak - 1).min()
    daily = pd.Series(eq).pct_change().fillna(0.0)
    sharpe = (daily.mean() / daily.std()) * np.sqrt(252) if daily.std() > 0 else np.nan
    if len(trades):
        wr = (trades["pnl_$"] > 0).mean()
        pf = (trades.loc[trades["pnl_$"] > 0, "pnl_$"].sum() /
              max(-trades.loc[trades["pnl_$"] <= 0, "pnl_$"].sum(), 1e-9))
        n = len(trades)
    else:
        wr, pf, n = float("nan"), float("nan"), 0
    return cagr, max_dd, sharpe, wr, pf, n, final


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--cash", type=float, default=1000.0)
    args = p.parse_args()

    df = load_ohlc(args.csv)
    years = len(df) / 252
    bh = df["close"].iloc[-1] / df["close"].iloc[0] - 1
    bh_cagr = (1 + bh) ** (1 / years) - 1

    grid = {
        "swing_n":      [2, 3, 5],
        "rr":           [1.5, 2.0, 3.0],
        "risk":         [0.01, 0.02, 0.03],
        "min_fvg_pct":  [0.001, 0.0015, 0.0030],
        "allow_short":  [True, False],
    }
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"Sweeping {len(combos)} configurations on {len(df)} bars "
          f"({years:.1f} yr). Buy & hold CAGR = {bh_cagr*100:.1f}%.\n")

    rows = []
    for vals in combos:
        params = dict(zip(keys, vals))
        trades, eq = backtest(
            df,
            swing_n=params["swing_n"], rr=params["rr"],
            risk_per_trade=params["risk"],
            min_fvg_pct=params["min_fvg_pct"],
            allow_short=params["allow_short"],
            account=args.cash,
        )
        cagr, dd, sh, wr, pf, n, final = metrics(df, trades, eq, args.cash)
        rows.append({
            **params,
            "cagr": cagr, "max_dd": dd, "sharpe": sh,
            "win_rate": wr, "profit_factor": pf, "n_trades": n,
            "final": final,
        })

    out = pd.DataFrame(rows).sort_values("cagr", ascending=False)

    # Print top 12 and bottom 4 by CAGR.
    def fmt(r):
        return (f"  sn={r['swing_n']} rr={r['rr']:.1f} risk={r['risk']*100:.0f}% "
                f"fvg>={r['min_fvg_pct']*1000:.1f}‰ "
                f"{'L+S' if r['allow_short'] else 'L  '}  "
                f"| CAGR {r['cagr']*100:6.2f}% | DD {r['max_dd']*100:6.1f}% "
                f"| Sh {r['sharpe']:.2f} | WR {r['win_rate']*100:4.1f}% "
                f"| PF {r['profit_factor']:.2f} | n={int(r['n_trades']):4d} "
                f"| ${r['final']:>8,.0f}")

    print("TOP 12 BY CAGR")
    print("-" * 110)
    for _, r in out.head(12).iterrows():
        print(fmt(r))
    print("\nBOTTOM 4 BY CAGR (sanity check — strategy isn't magic)")
    print("-" * 110)
    for _, r in out.tail(4).iterrows():
        print(fmt(r))

    # Also: best by Sharpe and best by lowest DD with positive CAGR.
    print("\nTOP 5 BY SHARPE (risk-adjusted)")
    print("-" * 110)
    for _, r in out.sort_values("sharpe", ascending=False).head(5).iterrows():
        print(fmt(r))

    print("\nLOWEST DRAWDOWN AMONG POSITIVE-CAGR CONFIGS")
    print("-" * 110)
    pos = out[out["cagr"] > 0].sort_values("max_dd", ascending=False)
    for _, r in pos.head(5).iterrows():
        print(fmt(r))

    out.to_csv("sweep_results.csv", index=False)
    print(f"\nFull sweep written to sweep_results.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
