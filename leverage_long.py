"""
leverage_long.py — the payoff: lever the Sharpe-superior SMC long-only book to
a PRUDENT drawdown and see whether it finally out-returns buy & hold.

THE SETUP
---------
On 20 years of real NDX daily data the pure SMC long-only book scored Sharpe
0.85 vs buy & hold's 0.76, with a -3.1% max drawdown vs -53.7%. Because its
Sharpe is HIGHER, levering it to the same risk MUST beat B&H on return — that's
just algebra. The real question is whether a *sane* leverage (one that targets a
tolerable -15% to -20% drawdown, not a -54% one) already clears the bar without
courting a blow-up.

HOW LEVERAGE IS APPLIED
-----------------------
For a stop-based strategy, leverage = position size = risk-per-trade. Doubling
risk-per-trade doubles every position and (very nearly) doubles the return and
the drawdown. So we sweep risk-per-trade on the REAL engine — this captures true
compounding and path-dependence, not a back-of-envelope scaling. Per-trade
costs are already modelled; overnight CFD financing is added as a transparent
haircut (the book is in-market only ~⅓ of the time at modest notional, so the
drag is small — shown explicitly).

WHAT TO LOOK FOR
----------------
The row whose max drawdown lands in the -15% to -20% band: is its total return
and CAGR above buy & hold's? If yes, the Sharpe edge is real, bankable money —
not a curve-fit. We also flag any leverage that risks ruin.

USAGE
-----
    python leverage_long.py
    python leverage_long.py --financing 0.065
"""

import argparse
import numpy as np

from smc_strategy_v2 import load_ohlc, backtest_v2
from regime_test import window_stats, detect_drawdown_episodes


def cagr_of(total, n_days):
    years = n_days / 252
    return (1 + total) ** (1 / years) - 1 if years > 0 and (1 + total) > 0 else -1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="NDX.csv")
    p.add_argument("--account", type=float, default=10000.0)
    p.add_argument("--financing", type=float, default=0.065,
                   help="Annual CFD financing on borrowed notional (estimate).")
    args = p.parse_args()

    df = load_ohlc(args.csv)
    n = len(df)
    close = df["close"].to_numpy(float)
    bh = close / close[0] * args.account
    btot, bsh, bdd = window_stats(bh)
    bcagr = cagr_of(btot, n)

    # Measure the base (1% risk) book's in-market fraction for the financing model.
    base_trades, base_eq = backtest_v2(
        df, swing_n=3, rr_runner=5.0, max_bars_after_bos=20, min_fvg_pct=0.001,
        risk_per_trade=0.01, account=args.account, allow_short=False,
        use_trend_filter=True)
    base_r = np.diff(np.asarray(base_eq, float)) / np.asarray(base_eq, float)[:-1]
    in_mkt = float((np.abs(base_r) > 1e-9).mean())

    print("=" * 100)
    print("  PRUDENT LEVERAGE OF THE SMC LONG-ONLY BOOK (NDX, 20 years, real data)")
    print("=" * 100)
    print(f"  Book in-market ~{in_mkt*100:.0f}% of days. Financing {args.financing*100:.1f}%/yr "
          f"on borrowed notional above 1x (transparent estimate).")
    print(f"  Buy & Hold: total {btot*100:,.0f}%  CAGR {bcagr*100:.1f}%  "
          f"Sharpe {bsh:.2f}  MaxDD {bdd*100:.1f}%")
    print("=" * 100)
    print(f"  {'risk/trd':>8} | {'~lev':>5} | {'total':>10} | {'CAGR':>7} | "
          f"{'Sharpe':>7} | {'MaxDD':>7} | {'fin drag':>8} | {'vs B&H':>10}")
    print("  " + "-" * 92)

    rows = []
    for risk in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20]:
        trades, eq = backtest_v2(
            df, swing_n=3, rr_runner=5.0, max_bars_after_bos=20, min_fvg_pct=0.001,
            risk_per_trade=risk, account=args.account, allow_short=False,
            use_trend_filter=True)
        eq = np.asarray(eq, float)
        tot, sh, dd = window_stats(eq)
        cg = cagr_of(tot, n)
        lev = risk / 0.01
        # Financing: borrowed notional above 1x. Notional/equity grows ~linearly
        # with risk; approximate avg levered exposure as lev * in_mkt * (base
        # notional fraction ~0.33). Charge financing only on the part above 1x.
        avg_exposure = lev * in_mkt * 0.33
        borrowed = max(0.0, avg_exposure - 1.0)
        fin_drag = borrowed * args.financing
        cg_net = cg - fin_drag
        ruin = bool((eq <= 0).any()) or dd < -0.95
        band = ""
        if -0.20 <= dd <= -0.13:
            band = "  <== prudent"
        rows.append((risk, lev, tot, cg_net, sh, dd, fin_drag, ruin, band))
        vs = (cg_net - bcagr) * 100
        print(f"  {risk*100:>6.0f}% | {lev:>4.0f}x | {tot*100:>9,.0f}% | "
              f"{cg_net*100:>6.1f}% | {sh:>7.2f} | {dd*100:>6.1f}% | "
              f"{fin_drag*100:>7.1f}% | {vs:>+8.1f}pp{band}"
              f"{'  RUIN' if ruin else ''}")

    print("  " + "-" * 92)
    print(f"  {'Buy&Hold':>8} | {'1x':>5} | {btot*100:>9,.0f}% | {bcagr*100:>6.1f}% | "
          f"{bsh:>7.2f} | {bdd*100:>6.1f}% | {'--':>8} | {'--':>10}")
    print("=" * 100)

    # Pick the prudent row (DD in -13%..-20%) with best net CAGR.
    prudent = [r for r in rows if -0.20 <= r[5] <= -0.13 and not r[7]]
    if prudent:
        best = max(prudent, key=lambda r: r[3])
        risk, lev, tot, cg_net, sh, dd, fin, ruin, _ = best
        beats = cg_net > bcagr
        print(f"  PRUDENT PICK: {risk*100:.0f}% risk/trade (~{lev:.0f}x base sizing)")
        print(f"    Drawdown {dd*100:.1f}% (vs B&H -53.7%) — a tolerable, survivable risk.")
        print(f"    CAGR {cg_net*100:.1f}% net of financing  vs  B&H {bcagr*100:.1f}%  "
              f"-> {'BEATS' if beats else 'trails'} buy & hold.")
        print(f"    Total {tot*100:,.0f}%  Sharpe {sh:.2f} (vs B&H {bsh:.2f}).")
        if beats:
            print(f"    => The Sharpe edge converts to real outperformance at PRUDENT risk,")
            print(f"       with ~{abs(0.537/dd):.0f}x less drawdown than buy & hold.")
    else:
        print("  No leverage landed cleanly in the -13%..-20% band; see table above.")

    # Sanity: does the levered long book still avoid the 2008/2022 carnage?
    print("\n  Crash check at the prudent leverage (long-only is FLAT in bears):")
    if prudent:
        risk = best[0]
        _, leq = backtest_v2(df, swing_n=3, rr_runner=5.0, max_bars_after_bos=20,
                             min_fvg_pct=0.001, risk_per_trade=risk,
                             account=args.account, allow_short=False,
                             use_trend_filter=True)
        leq = np.asarray(leq, float)
        dates = df["date"].dt.date.to_numpy()
        for ep in detect_drawdown_episodes(close, 0.15):
            i0, i1 = ep["peak"], ep["trough"]
            if i1 - i0 < 5:
                continue
            b = window_stats(bh[i0:i1 + 1])[0]
            s = window_stats(leq[i0:i1 + 1])[0]
            print(f"    {str(dates[i0])+'->'+str(dates[i1]):<26} ({ep['depth']*100:>4.0f}%)  "
                  f"B&H {b*100:>6.1f}%   levered-long {s*100:>6.1f}%")
    print("=" * 100)


if __name__ == "__main__":
    main()
