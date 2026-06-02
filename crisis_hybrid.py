"""
crisis_hybrid.py — combine the GOOD low-drawdown SMC longs with the trend-
following crisis-short sleeve, so the book profits in bears without bleeding
itself to death in bulls.

THE TWO SLEEVES
---------------
  LONG sleeve  : SMC v2 long-only (FVG-retest longs, EMA200 regime). Modestly
                 positive in bulls, FLAT in bears (no longs below trend), tiny
                 drawdown. This is the bull participation.
  SHORT sleeve : trend-following shorts (Donchian breakdown + chandelier trail).
                 Profits in sustained declines (6/7 real crashes), bleeds slowly
                 in bulls. This is the crisis insurance.

Because the long sleeve is flat exactly when the short sleeve pays (and vice
versa), blending them should net to a curve that is up-trending in bulls AND
green in crashes. We sweep the short-sleeve weight to find the blend that turns
"flat in crashes" into "PROFITABLE in crashes" while keeping the full-cycle
result positive and low-drawdown.

USAGE
-----
    python crisis_hybrid.py
    python crisis_hybrid.py --long-risk 0.01 --short-risk 0.01
"""

import argparse
import numpy as np
import pandas as pd

from smc_strategy_v2 import load_ohlc, backtest_v2
from smc_crisis import backtest_trend
from regime_test import detect_drawdown_episodes, window_stats


def rets(eq):
    eq = np.asarray(eq, float)
    return np.concatenate([[0.0], np.diff(eq) / eq[:-1]])


def full_stats(eq):
    return window_stats(eq)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="NDX.csv")
    p.add_argument("--account", type=float, default=10000.0)
    p.add_argument("--long-risk", type=float, default=0.01)
    p.add_argument("--short-risk", type=float, default=0.01)
    p.add_argument("--threshold", type=float, default=0.15)
    args = p.parse_args()

    df = load_ohlc(args.csv)
    close = df["close"].to_numpy(float)
    dates = df["date"].dt.date.to_numpy()
    bh = close / close[0] * args.account

    # LONG sleeve — SMC v2 long-only (tuned, low drawdown).
    lt, leq = backtest_v2(df, swing_n=3, rr_runner=5.0, max_bars_after_bos=20,
                          min_fvg_pct=0.001, risk_per_trade=args.long_risk,
                          account=args.account, allow_short=False,
                          use_trend_filter=True)
    # SHORT sleeve — trend-following crisis shorts.
    st, seq = backtest_trend(df, ema_slow=100, ema_fast=20, donchian=20,
                             atr_init=2.0, atr_trail=3.5, risk=args.short_risk,
                             account=args.account, allow_long=False,
                             allow_short=True)

    lr, sr = rets(leq), rets(seq)
    episodes = detect_drawdown_episodes(close, args.threshold)

    def regime_breakdown(eq):
        bhd, sd = [], []
        for ep in episodes:
            i0, i1 = ep["peak"], ep["trough"]
            if i1 - i0 < 5:
                continue
            bhd.append(window_stats(bh[i0:i1 + 1])[0])
            sd.append(window_stats(eq[i0:i1 + 1])[0])
        return np.mean(bhd), np.mean(sd), sum(1 for x in sd if x > 0), len(sd)

    print("=" * 96)
    print("  CRISIS HYBRID — SMC longs + trend-following crisis shorts (NDX, 20y)")
    print("=" * 96)
    print(f"  {'short wt':>8} | {'total':>9} | {'Sharpe':>7} | {'MaxDD':>7} | "
          f"{'avg crash':>9} | {'crashes green':>13}")
    print("  " + "-" * 84)

    results = {}
    for w in [0.0, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 1.0]:
        cr = (1 - w) * lr + w * sr
        ceq = args.account * np.cumprod(1 + cr)
        tot, sh, dd = full_stats(ceq)
        _, avg_crash, green, ncr = regime_breakdown(ceq)
        results[w] = (ceq, tot, sh, dd, avg_crash, green, ncr)
        print(f"  {w:>7.0%} | {tot*100:>8.1f}% | {sh:>7.2f} | {dd*100:>6.1f}% | "
              f"{avg_crash*100:>+8.1f}% | {green:>6}/{ncr:<6}")

    # Buy & hold reference.
    btot, bsh, bdd = full_stats(bh)
    bhd_avg, _, _, _ = regime_breakdown(bh)
    print("  " + "-" * 84)
    print(f"  {'Buy&Hold':>8} | {btot*100:>8.1f}% | {bsh:>7.2f} | {bdd*100:>6.1f}% | "
          f"{bhd_avg*100:>+8.1f}% | {'0':>6}/{'7':<6}")
    print("=" * 96)

    # Pick the blend that is crash-green on >=5/7 and best full Sharpe.
    good = {w: r for w, r in results.items() if r[5] >= 5}
    pick = max(good or results, key=lambda w: results[w][2])
    ceq, tot, sh, dd, avg_crash, green, ncr = results[pick]

    print(f"\n  RECOMMENDED BLEND: {pick:.0%} crisis-short / {1-pick:.0%} SMC-long")
    print("  " + "-" * 84)
    print(f"  {'crash decline':<30} | {'B&H':>8} | {'HYBRID':>8} | {'edge':>8}")
    print("  " + "-" * 64)
    for ep in episodes:
        i0, i1 = ep["peak"], ep["trough"]
        if i1 - i0 < 5:
            continue
        b = window_stats(bh[i0:i1 + 1])[0]
        s = window_stats(ceq[i0:i1 + 1])[0]
        lab = f"{dates[i0]}->{dates[i1]} ({ep['depth']*100:.0f}%)"
        print(f"  {lab:<30} | {b*100:>7.1f}% | {s*100:>7.1f}% | {(s-b)*100:>+7.1f}%")
    print("  " + "-" * 64)
    print(f"  HYBRID: total {tot*100:.1f}%  Sharpe {sh:.2f}  MaxDD {dd*100:.1f}%  "
          f"| green in {green}/{ncr} crashes")
    print(f"  B&Hold: total {btot*100:.1f}%  Sharpe {bsh:.2f}  MaxDD {bdd*100:.1f}%")
    print("=" * 96)
    print("  READ: a blend that is GREEN in most crashes while staying net-positive")
    print("  full-cycle is genuine crisis alpha — an uncorrelated hedge. It still")
    print("  won't out-TOTAL a 20-year bull, but its return is POSITIVE exactly when")
    print("  B&H bleeds, which is what makes it worth holding alongside the index.")
    print("=" * 96)


if __name__ == "__main__":
    main()
