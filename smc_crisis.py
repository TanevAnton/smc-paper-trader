"""
smc_crisis.py — add CRISIS ALPHA: a trend-following long/short overlay that
actively RIDES sustained declines instead of sitting in cash.

THE GOAL
--------
The daily SMC strategy sidesteps crashes (goes flat) but barely profits from
them — it made only +0.8% in 2008 and +2.5% in 2022 while the index lost 50%+.
To turn that "stay flat" into "make money," we need to be SHORT and STAY short
through the decline. The classic, proven tool for this is trend following
(Donchian breakout + moving-average regime + a chandelier trailing stop) — the
same mechanism that made managed-futures funds money in every major bear.

THE ENGINE
----------
  * Regime  : EMA(slow). Long only above it, short only below it.
  * Trigger : Donchian breakout — go long on a new `donchian`-day high (in an
              uptrend), short on a new `donchian`-day low (in a downtrend).
  * Ride    : chandelier trailing stop (extreme-since-entry -/+ atr_trail*ATR)
              that only ratchets in your favour, so a winner is held for the
              WHOLE trend and only exits on a real reversal.
  * Exit    : trail hit, or close crossing back through EMA(fast).
  * Sizing  : fixed-fractional risk on the initial ATR stop.

No look-ahead: Donchian/EMA/ATR are computed on prior bars (shifted), the
decision is made at today's close, and the fill is next day's open.

This is deliberately NOT the FVG/BOS entry logic — it's a momentum overlay
whose job is the one thing pure SMC retests do poorly: capture a waterfall.

USAGE
-----
    python smc_crisis.py                         # textbook params, regime report
    python smc_crisis.py --search 400            # random-search params, then report
    python smc_crisis.py --shorts-only           # isolate the bear contribution
"""

import argparse
import numpy as np
import pandas as pd

from smc_strategy_v2 import load_ohlc, compute_atr, compute_ema
from regime_test import detect_drawdown_episodes, window_stats


# ----------------------------------------------------------------------------
# Trend-following long/short engine
# ----------------------------------------------------------------------------
def backtest_trend(df, ema_slow=100, ema_fast=20, donchian=20, atr_n=14,
                   atr_init=2.0, atr_trail=3.5, risk=0.01, account=10000.0,
                   cost_bps=5.0, allow_long=True, allow_short=True):
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    L = len(df)

    atr = compute_atr(h, l, c, atr_n)            # shifted: uses bars < t
    ef = compute_ema(c, ema_fast)                # shifted
    es = compute_ema(c, ema_slow)                # shifted
    roll_hi = pd.Series(h).rolling(donchian).max().shift(1).to_numpy()
    roll_lo = pd.Series(l).rolling(donchian).min().shift(1).to_numpy()

    equity = account
    eq = np.full(L, account, dtype=float)
    pos = 0
    entry = stop = shares = 0.0
    ext = 0.0                 # extreme price since entry (high for long, low for short)
    entry_bar = -1
    trades = []

    for t in range(L):
        # ---- manage open position (intrabar fills) ----
        exitp, reason = None, None
        if pos == 1:
            ext = max(ext, h[t])
            stop = max(stop, ext - atr_trail * atr[t])
            if l[t] <= stop:
                exitp, reason = stop, "trail"
            elif not np.isnan(ef[t]) and c[t] < ef[t]:
                exitp, reason = c[t], "ema-exit"
        elif pos == -1:
            ext = min(ext, l[t])
            stop = min(stop, ext + atr_trail * atr[t])
            if h[t] >= stop:
                exitp, reason = stop, "trail"
            elif not np.isnan(ef[t]) and c[t] > ef[t]:
                exitp, reason = c[t], "ema-exit"

        if exitp is not None:
            pnl = shares * (exitp - entry) * pos
            pnl -= shares * exitp * (cost_bps / 1e4)
            equity += pnl
            trades.append({
                "entry_date": df["date"].iloc[entry_bar].date(),
                "exit_date": df["date"].iloc[t].date(),
                "direction": "long" if pos == 1 else "short",
                "pnl_$": round(pnl, 2), "reason": reason,
            })
            pos = 0

        # ---- entry decision at close t, fill at open t+1 ----
        if pos == 0 and t + 1 < L and not np.isnan(es[t]) \
                and not np.isnan(roll_hi[t]) and atr[t] > 0:
            go = 0
            if allow_long and c[t] > es[t] and c[t] >= roll_hi[t]:
                go = 1
            elif allow_short and c[t] < es[t] and c[t] <= roll_lo[t]:
                go = -1
            if go != 0:
                entry = o[t + 1]
                if go == 1:
                    stop = entry - atr_init * atr[t]
                    sd = entry - stop
                else:
                    stop = entry + atr_init * atr[t]
                    sd = stop - entry
                if sd > 0:
                    shares = (equity * risk) / sd
                    equity -= shares * entry * (cost_bps / 1e4)
                    pos = go
                    entry_bar = t + 1
                    ext = entry

        eq[t] = equity + (shares * (c[t] - entry) * pos if pos != 0 else 0.0)

    return pd.DataFrame(trades), eq


# ----------------------------------------------------------------------------
# Regime attribution (reuses the real-crash detector)
# ----------------------------------------------------------------------------
def regime_report(df, eq, trades, account, threshold=0.15, label="trend L/S"):
    close = df["close"].to_numpy(float)
    dates = df["date"].dt.date.to_numpy()
    eq = np.asarray(eq, float)
    bh = close / close[0] * account
    episodes = detect_drawdown_episodes(close, threshold)

    print("=" * 92)
    print(f"  CRISIS-ALPHA REGIME TEST — {label} vs Buy & Hold "
          f"({dates[0]} -> {dates[-1]})")
    print("=" * 92)
    print(f"  {'crash decline (peak->trough)':<32} | {'B&H':>8} | {'STRAT':>8} | "
          f"{'edge':>8} | {'shorts':>6}")
    print("  " + "-" * 80)
    bhd, sd, wins = [], [], 0
    for ep in episodes:
        i0, i1 = ep["peak"], ep["trough"]
        if i1 - i0 < 5:
            continue
        bh_ret, _, _ = window_stats(bh[i0:i1 + 1])
        s_ret, _, _ = window_stats(eq[i0:i1 + 1])
        win_dates = set(dates[i0:i1 + 1].tolist())
        n_short = sum(1 for _, t in trades.iterrows()
                      if t["direction"] == "short" and t["entry_date"] in win_dates) \
            if len(trades) else 0
        label_ep = f"{dates[i0]}->{dates[i1]} ({ep['depth']*100:.0f}%)"
        print(f"  {label_ep:<32} | {bh_ret*100:>7.1f}% | {s_ret*100:>7.1f}% | "
              f"{(s_ret-bh_ret)*100:>+7.1f}% | {n_short:>6}")
        bhd.append(bh_ret); sd.append(s_ret); wins += (s_ret > bh_ret)
    print("  " + "-" * 80)
    if bhd:
        print(f"  {'AVERAGE crash decline':<32} | {np.mean(bhd)*100:>7.1f}% | "
              f"{np.mean(sd)*100:>7.1f}% | {(np.mean(sd)-np.mean(bhd))*100:>+7.1f}% |")
        print(f"  STRAT made ABSOLUTE money in "
              f"{sum(1 for x in sd if x>0)}/{len(sd)} crash declines; "
              f"beat B&H in {wins}/{len(sd)}.")
    full_bh = window_stats(bh)
    full_s = window_stats(eq)
    print("=" * 92)
    print(f"  FULL PERIOD:  Buy&Hold total {full_bh[0]*100:>8.1f}%  Sharpe {full_bh[1]:.2f}  MaxDD {full_bh[2]*100:.1f}%")
    print(f"                {label:<11} total {full_s[0]*100:>8.1f}%  Sharpe {full_s[1]:.2f}  MaxDD {full_s[2]*100:.1f}%")
    print("=" * 92)
    return full_s, np.mean(sd) if sd else 0.0


def sortino(eq):
    eq = np.asarray(eq, float)
    r = np.diff(eq) / eq[:-1]
    dn = r[r < 0].std()
    return float(r.mean() / dn * np.sqrt(252)) if dn > 0 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="NDX.csv")
    p.add_argument("--account", type=float, default=10000.0)
    p.add_argument("--risk", type=float, default=0.01)
    p.add_argument("--ema-slow", type=int, default=100)
    p.add_argument("--ema-fast", type=int, default=20)
    p.add_argument("--donchian", type=int, default=20)
    p.add_argument("--atr-init", type=float, default=2.0)
    p.add_argument("--atr-trail", type=float, default=3.5)
    p.add_argument("--shorts-only", action="store_true")
    p.add_argument("--longs-only", action="store_true")
    p.add_argument("--threshold", type=float, default=0.15)
    p.add_argument("--search", type=int, default=0,
                   help="Random-search N param sets on full history (by Sortino), "
                        "then report the best.")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    df = load_ohlc(args.csv)
    allow_long = not args.shorts_only
    allow_short = not args.longs_only

    if args.search > 0:
        import random
        rng = random.Random(args.seed)
        best = None
        # Time-split guard: optimize on first 70%, then report full + regime.
        n = len(df)
        cut = int(n * 0.7)
        df_tr = df.iloc[:cut].reset_index(drop=True)
        for i in range(args.search):
            params = dict(
                ema_slow=rng.choice([50, 80, 100, 150, 200]),
                ema_fast=rng.choice([10, 15, 20, 30, 40]),
                donchian=rng.choice([10, 15, 20, 30, 40, 55]),
                atr_init=rng.choice([1.5, 2.0, 2.5, 3.0]),
                atr_trail=rng.choice([2.5, 3.0, 3.5, 4.0, 5.0]),
            )
            if params["ema_fast"] >= params["ema_slow"]:
                continue
            _, eqtr = backtest_trend(df_tr, risk=args.risk, account=args.account,
                                     allow_long=allow_long, allow_short=allow_short,
                                     **params)
            sc = sortino(eqtr)
            if best is None or sc > best[0]:
                best = (sc, params)
        print(f"Random search of {args.search} configs (optimized on first 70%, "
              f"by Sortino). Best train Sortino={best[0]:.2f}")
        print(f"Best params: {best[1]}\n")
        trades, eq = backtest_trend(df, risk=args.risk, account=args.account,
                                    allow_long=allow_long, allow_short=allow_short,
                                    **best[1])
        lbl = f"trend({'S' if not allow_long else 'L+S' if allow_short else 'L'})"
        regime_report(df, eq, trades, args.account, args.threshold, lbl)
        return

    trades, eq = backtest_trend(
        df, ema_slow=args.ema_slow, ema_fast=args.ema_fast, donchian=args.donchian,
        atr_init=args.atr_init, atr_trail=args.atr_trail, risk=args.risk,
        account=args.account, allow_long=allow_long, allow_short=allow_short)
    mode = "S-only" if args.shorts_only else ("L-only" if args.longs_only else "L+S")
    regime_report(df, eq, trades, args.account, args.threshold, f"trend {mode}")


if __name__ == "__main__":
    main()
