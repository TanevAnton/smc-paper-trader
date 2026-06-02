"""
regime_test.py — the missing test: does the SMC edge become a RETURN edge when
the market actually FALLS?

WHY THIS EXISTS
---------------
Every previous result was measured on data dominated by an uptrend, where
buy & hold is nearly impossible to beat on risk-adjusted return. The leverage
test showed the SMC book only out-returns B&H at matched risk when its Sharpe
exceeds B&H's — which didn't happen in a clean bull. The open question was:
what about a BEAR?

No intraday data exists for 2008/2018/2020/2022, so the evolved 1h ensemble
can't be tested there. But the SAME SMC primitives (swings, BOS, FVG, with
shorts + an EMA200 regime filter) run on DAILY bars, and we have 20 years of
real Nasdaq-100 daily data — every real crash included. This script:

  1. Runs the daily SMC long/short strategy once over the full 20 years.
  2. Auto-detects every real drawdown episode where the index fell >= a
     threshold (default 15%) peak-to-trough.
  3. For each crash's DECLINE phase (peak -> trough, i.e. exactly when B&H is
     bleeding) reports SMC return vs B&H return, plus Sharpe and drawdown.
  4. Aggregates: in real bear declines, does SMC protect capital / make money
     while B&H loses? And the mirror — does it give back the edge in calm bull
     stretches?

This is the honest stress test the intraday data could not provide. It tests
the APPROACH, not the exact evolved genomes — but it answers the regime
question with 100% real crash data.

USAGE
-----
    python regime_test.py                       # NDX.csv, 15% threshold
    python regime_test.py --threshold 0.12 --risk 0.01
"""

import argparse
import numpy as np
import pandas as pd

from smc_strategy_v2 import load_ohlc, backtest_v2


def detect_drawdown_episodes(close, threshold=0.15):
    """Return non-overlapping peak->trough->recovery episodes whose peak-to-trough
    decline reached at least `threshold`."""
    n = len(close)
    peak = np.maximum.accumulate(close)
    dd = close / peak - 1.0
    episodes = []
    i = 0
    while i < n:
        if dd[i] <= -threshold:
            # Peak = last new-high bar before this decline.
            j = i
            while j > 0 and dd[j] < 0:
                j -= 1
            peak_idx = j
            # March forward to recovery (dd back to ~0 = new high), tracking trough.
            k = i
            trough_idx = i
            while k < n and dd[k] < 0:
                if close[k] < close[trough_idx]:
                    trough_idx = k
                k += 1
            recovery_idx = k - 1 if k < n else n - 1
            episodes.append({
                "peak": peak_idx, "trough": trough_idx, "recovery": recovery_idx,
                "depth": float(close[trough_idx] / close[peak_idx] - 1.0),
                "recovered": k < n,
            })
            i = k
        else:
            i += 1
    return episodes


def window_stats(eq_slice):
    """Total return, annualized Sharpe, and max drawdown over an equity slice."""
    eq_slice = np.asarray(eq_slice, dtype=float)
    if eq_slice.size < 2:
        return 0.0, 0.0, 0.0
    r = np.diff(eq_slice) / eq_slice[:-1]
    ret = float(eq_slice[-1] / eq_slice[0] - 1.0)
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0
    peak = np.maximum.accumulate(eq_slice)
    dd = float((eq_slice / peak - 1.0).min())
    return ret, sharpe, dd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="NDX.csv")
    p.add_argument("--threshold", type=float, default=0.15,
                   help="Min peak-to-trough decline to count as a bear episode.")
    p.add_argument("--risk", type=float, default=0.0075)
    p.add_argument("--rr-runner", type=float, default=5.0)
    p.add_argument("--account", type=float, default=10000.0)
    args = p.parse_args()

    df = load_ohlc(args.csv)
    close = df["close"].to_numpy(dtype=float)
    dates = df["date"].dt.date.to_numpy()
    n = len(df)

    # One full-history daily SMC long/short backtest (shorts + EMA200 on).
    trades, eq = backtest_v2(
        df, swing_n=3, rr_runner=args.rr_runner, max_bars_after_bos=20,
        min_fvg_pct=0.001, risk_per_trade=args.risk, account=args.account,
        allow_short=True, use_trend_filter=True,
    )
    eq = np.asarray(eq, dtype=float)
    # Buy & hold "equity" is just the index, scaled to the same start capital.
    bh = close / close[0] * args.account

    episodes = detect_drawdown_episodes(close, args.threshold)
    print("=" * 100)
    print(f"  REGIME TEST — daily SMC (long/short) vs Buy & Hold across real "
          f"Nasdaq-100 crashes")
    print(f"  {dates[0]} -> {dates[-1]}   |   {len(episodes)} drawdown episodes "
          f">= {args.threshold*100:.0f}%   |   SMC risk {args.risk*100:.2f}%/trade")
    print("=" * 100)
    print(f"  {'crash decline (peak->trough)':<34} | {'B&H ret':>8} | "
          f"{'SMC ret':>8} | {'edge':>8} | {'SMC DD':>7} | {'shorts':>6}")
    print("  " + "-" * 96)

    bh_declines, smc_declines, wins = [], [], 0
    for ep in episodes:
        i0, i1 = ep["peak"], ep["trough"]
        if i1 - i0 < 5:
            continue
        bh_ret, _, _ = window_stats(bh[i0:i1 + 1])
        smc_ret, _, smc_dd = window_stats(eq[i0:i1 + 1])
        edge = smc_ret - bh_ret
        # Count shorts taken during the decline window.
        win_dates = set(dates[i0:i1 + 1].tolist())
        n_short = 0
        if len(trades):
            for _, t in trades.iterrows():
                if t["direction"] == "short" and t["entry_date"] in win_dates:
                    n_short += 1
        label = f"{dates[i0]}->{dates[i1]} ({ep['depth']*100:.0f}%)"
        print(f"  {label:<34} | {bh_ret*100:>7.1f}% | {smc_ret*100:>7.1f}% | "
              f"{edge*100:>+7.1f}% | {smc_dd*100:>6.1f}% | {n_short:>6}")
        bh_declines.append(bh_ret)
        smc_declines.append(smc_ret)
        if smc_ret > bh_ret:
            wins += 1

    print("  " + "-" * 96)
    if bh_declines:
        print(f"  {'AVERAGE across crash declines':<34} | "
              f"{np.mean(bh_declines)*100:>7.1f}% | {np.mean(smc_declines)*100:>7.1f}% | "
              f"{(np.mean(smc_declines)-np.mean(bh_declines))*100:>+7.1f}% |")
        print(f"  SMC beat B&H in {wins}/{len(bh_declines)} crash declines.")
    print("=" * 100)

    # ---- Mirror: the calm/bull stretches between crashes ----
    bear_mask = np.zeros(n, dtype=bool)
    for ep in episodes:
        bear_mask[ep["peak"]:ep["recovery"] + 1] = True
    # Build contiguous bull spans (not in any drawdown episode).
    bull_spans = []
    i = 0
    while i < n:
        if not bear_mask[i]:
            j = i
            while j < n and not bear_mask[j]:
                j += 1
            if j - i > 20:
                bull_spans.append((i, j - 1))
            i = j
        else:
            i += 1

    print(f"\n  Mirror — CALM / BULL stretches between crashes "
          f"({len(bull_spans)} spans):")
    print(f"  {'bull stretch':<34} | {'B&H ret':>8} | {'SMC ret':>8} | {'edge':>8}")
    print("  " + "-" * 70)
    bh_bull, smc_bull = [], []
    for (i0, i1) in bull_spans:
        bh_ret, _, _ = window_stats(bh[i0:i1 + 1])
        smc_ret, _, _ = window_stats(eq[i0:i1 + 1])
        label = f"{dates[i0]}->{dates[i1]}"
        print(f"  {label:<34} | {bh_ret*100:>7.1f}% | {smc_ret*100:>7.1f}% | "
              f"{(smc_ret-bh_ret)*100:>+7.1f}%")
        bh_bull.append(bh_ret)
        smc_bull.append(smc_ret)
    print("  " + "-" * 70)
    if bh_bull:
        print(f"  {'AVERAGE across bull stretches':<34} | "
              f"{np.mean(bh_bull)*100:>7.1f}% | {np.mean(smc_bull)*100:>7.1f}% | "
              f"{(np.mean(smc_bull)-np.mean(bh_bull))*100:>+7.1f}%")

    # ---- Full-period summary ----
    full_bh = window_stats(bh)
    full_smc = window_stats(eq)
    print("\n" + "=" * 100)
    print("  FULL 20-YEAR SUMMARY")
    print("=" * 100)
    print(f"    Buy & Hold : total {full_bh[0]*100:>8.1f}%   Sharpe {full_bh[1]:.2f}   "
          f"MaxDD {full_bh[2]*100:.1f}%")
    print(f"    SMC L/S    : total {full_smc[0]*100:>8.1f}%   Sharpe {full_smc[1]:.2f}   "
          f"MaxDD {full_smc[2]*100:.1f}%")
    print("=" * 100)
    print("  THE VERDICT ON REGIME-DEPENDENCE:")
    if bh_declines and bh_bull:
        d_edge = (np.mean(smc_declines) - np.mean(bh_declines)) * 100
        b_edge = (np.mean(smc_bull) - np.mean(bh_bull)) * 100
        print(f"   - In real crash DECLINES: SMC edge over B&H = {d_edge:+.1f}% per episode.")
        print(f"   - In calm BULL stretches: SMC edge over B&H = {b_edge:+.1f}% per stretch.")
        print(f"   - If the decline edge is strongly positive and the bull edge negative,")
        print(f"     the thesis holds: this is a DRAWDOWN/BEAR strategy, not a bull-beater.")
    print("=" * 100)


if __name__ == "__main__":
    main()
