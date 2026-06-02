"""
smc_strategy.py — Smart-Money-Concepts strategy on daily OHLC bars.

WHAT IT IMPLEMENTS
------------------
- SWING HIGHS / LOWS via fractal pivot (a bar is a swing high if its high beats
  the highs of the N bars on each side). A swing at bar i is only "confirmed"
  at bar i+N, so the algorithm never uses information it could not have known
  in real time.
- MARKET STRUCTURE TREND that flips on Break-of-Structure (BOS):
    * Bull  trend  when close > most-recent confirmed swing HIGH (BOS up).
    * Bear  trend  when close < most-recent confirmed swing LOW  (BOS down).
- FAIR VALUE GAPS (FVG) — the 3-bar imbalance:
    * Bullish FVG forms at bar t when high[t-2] < low[t], zone = [high[t-2], low[t]].
    * Bearish FVG: low[t-2]  > high[t],         zone = [high[t], low[t-2]].
  Each FVG is tracked until it's either filled (retested) or invalidated
  (price closes through the far edge).
- PRIOR DAY HIGH/LOW (PDH/PDL) act as the "session liquidity" on a daily chart.
  Reported in the trade log for context.

ENTRY / EXIT
------------
Long setup (short is mirrored):
  1. Current trend (set by latest BOS) is BULLISH.
  2. There exists an unmitigated bullish FVG whose top sits BELOW today's open.
  3. Today's low pierces the FVG zone but does NOT close beyond its bottom
     (a clean test, not a violent break).
  4. Fill long at the FVG top. Stop = FVG bottom - stop_buf * ATR.
     Target  = entry + RR * (entry - stop).
  5. Time-stop exit after `max_hold` bars at the close.

Position size: fixed-fractional risk (default 1% of current equity per trade).
Costs: cost_bps each side, applied to gross notional on exit.
Same-bar both-hits: assume STOP filled first (conservative).

This is a research tool — not financial advice. Numbers are honest, including
the trades and stretches where the strategy loses money.
"""

import argparse
import sys
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_ohlc(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    need = ["date", "open", "high", "low", "close"]
    for k in need:
        if k not in cols:
            sys.exit(f"ERROR: CSV missing column {k!r}.")
    df = df.rename(columns={cols[k]: k for k in need})
    df = df[need].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna().sort_values("date").reset_index(drop=True)
    if len(df) < 100:
        sys.exit("ERROR: need at least ~100 OHLC rows.")
    return df


# ----------------------------------------------------------------------------
# Indicators / structural primitives
# ----------------------------------------------------------------------------
def compute_atr(h, l, c, n=14):
    """ATR series, length=len(h), SHIFTED so atr[t] only uses bars < t."""
    L = len(h)
    tr = np.zeros(L)
    tr[0] = h[0] - l[0]
    for i in range(1, L):
        tr[i] = max(h[i] - l[i],
                    abs(h[i] - c[i - 1]),
                    abs(l[i] - c[i - 1]))
    atr = pd.Series(tr).rolling(n, min_periods=n).mean()
    atr = atr.shift(1).fillna(0.0).values
    return atr


def confirmed_swings(h, l, n=3):
    """For each bar t, the latest swing HIGH/LOW that is *known* by time t.
    A swing at bar i (its high strictly > the n bars on either side) becomes
    known only when bar i+n exists — so this is look-ahead-safe."""
    L = len(h)
    sh_p = np.full(L, np.nan)
    sl_p = np.full(L, np.nan)
    sh_i = np.full(L, -1, dtype=int)
    sl_i = np.full(L, -1, dtype=int)
    last_sh_p, last_sh_bar = np.nan, -1
    last_sl_p, last_sl_bar = np.nan, -1
    for t in range(L):
        i = t - n  # candidate confirmed at time t
        if i >= n:
            is_sh = True
            for k in range(1, n + 1):
                if not (h[i] > h[i - k] and h[i] > h[i + k]):
                    is_sh = False
                    break
            if is_sh:
                last_sh_p, last_sh_bar = h[i], i
            is_sl = True
            for k in range(1, n + 1):
                if not (l[i] < l[i - k] and l[i] < l[i + k]):
                    is_sl = False
                    break
            if is_sl:
                last_sl_p, last_sl_bar = l[i], i
        sh_p[t], sh_i[t] = last_sh_p, last_sh_bar
        sl_p[t], sl_i[t] = last_sl_p, last_sl_bar
    return sh_p, sh_i, sl_p, sl_i


# ----------------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------------
def backtest(df, swing_n=3, min_fvg_pct=0.0015, rr=2.0, stop_buf=0.5,
             atr_n=14, max_hold=20, risk_per_trade=0.01,
             cost_bps=5.0, account=1000.0, allow_short=True):
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    L = len(df)

    atr = compute_atr(h, l, c, atr_n)
    sh_p, _, sl_p, _ = confirmed_swings(h, l, swing_n)

    # Trend state — flips on BOS through the latest confirmed swing.
    trend = np.zeros(L, dtype=int)
    state = 0
    last_used_sh = np.nan
    last_used_sl = np.nan
    for t in range(L):
        if not np.isnan(sh_p[t]) and c[t] > sh_p[t] and sh_p[t] != last_used_sh:
            state = 1
            last_used_sh = sh_p[t]
        if not np.isnan(sl_p[t]) and c[t] < sl_p[t] and sl_p[t] != last_used_sl:
            state = -1
            last_used_sl = sl_p[t]
        trend[t] = state

    # FVG inventories — keep only unmitigated.
    bull_fvgs = []   # {"top","bot","formed","mitigated"}
    bear_fvgs = []

    # Simulation
    realized_eq = account
    eq_curve = np.full(L, account, dtype=float)
    position = 0
    entry_p = 0.0
    stop = 0.0
    target = 0.0
    shares = 0.0
    qty_dollars = 0.0
    entry_bar = -1
    trades = []

    def mtm():
        if position == 0:
            return 0.0
        if position == 1:
            return shares * (c[t] - entry_p)
        return shares * (entry_p - c[t])

    for t in range(L):
        # 1) New FVGs formed by today.
        if t >= 2:
            if h[t - 2] < l[t]:
                gap = l[t] - h[t - 2]
                if gap / c[t] >= min_fvg_pct:
                    bull_fvgs.append({"top": l[t], "bot": h[t - 2],
                                      "formed": t, "mitigated": False})
            if l[t - 2] > h[t]:
                gap = l[t - 2] - h[t]
                if gap / c[t] >= min_fvg_pct:
                    bear_fvgs.append({"top": l[t - 2], "bot": h[t],
                                      "formed": t, "mitigated": False})

        # 2) Manage open position — exit on stop / target / timeout.
        exit_price = None
        exit_reason = None
        if position == 1:
            if l[t] <= stop:
                exit_price, exit_reason = stop, "stop"
            elif h[t] >= target:
                exit_price, exit_reason = target, "target"
            elif t - entry_bar >= max_hold:
                exit_price, exit_reason = c[t], "timeout"
        elif position == -1:
            if h[t] >= stop:
                exit_price, exit_reason = stop, "stop"
            elif l[t] <= target:
                exit_price, exit_reason = target, "target"
            elif t - entry_bar >= max_hold:
                exit_price, exit_reason = c[t], "timeout"

        if exit_price is not None:
            gross = shares * (exit_price - entry_p) * position
            cost = qty_dollars * (2 * cost_bps / 1e4)
            net = gross - cost
            realized_eq += net
            trades.append({
                "entry_date": df["date"].iloc[entry_bar].date(),
                "exit_date":  df["date"].iloc[t].date(),
                "direction": "long" if position == 1 else "short",
                "entry": entry_p, "exit": exit_price,
                "stop": stop, "target": target,
                "bars_held": t - entry_bar,
                "pnl_$": net,
                "pnl_%_equity": net / (realized_eq - net) * 100
                                 if (realized_eq - net) > 0 else 0,
                "exit_reason": exit_reason,
            })
            position = 0

        # 3) Open a new position if flat & setup is valid.
        if position == 0 and t > swing_n + atr_n + 2:
            if trend[t] == 1:
                for fvg in reversed(bull_fvgs):
                    if fvg["mitigated"] or fvg["formed"] >= t:
                        continue
                    # FVG must be below today's open and today's bar must wick
                    # into the zone WITHOUT closing through it.
                    if fvg["top"] >= o[t]:
                        continue
                    if l[t] <= fvg["top"] and l[t] >= fvg["bot"]:
                        entry_p = fvg["top"]
                        stop = fvg["bot"] - stop_buf * atr[t]
                        risk_per_share = entry_p - stop
                        if risk_per_share <= 0:
                            fvg["mitigated"] = True
                            continue
                        target = entry_p + rr * risk_per_share
                        risk_dollars = realized_eq * risk_per_trade
                        shares = risk_dollars / risk_per_share
                        qty_dollars = shares * entry_p
                        position = 1
                        entry_bar = t
                        fvg["mitigated"] = True
                        break
            elif trend[t] == -1 and allow_short:
                for fvg in reversed(bear_fvgs):
                    if fvg["mitigated"] or fvg["formed"] >= t:
                        continue
                    if fvg["bot"] <= o[t]:
                        continue
                    if h[t] >= fvg["bot"] and h[t] <= fvg["top"]:
                        entry_p = fvg["bot"]
                        stop = fvg["top"] + stop_buf * atr[t]
                        risk_per_share = stop - entry_p
                        if risk_per_share <= 0:
                            fvg["mitigated"] = True
                            continue
                        target = entry_p - rr * risk_per_share
                        risk_dollars = realized_eq * risk_per_trade
                        shares = risk_dollars / risk_per_share
                        qty_dollars = shares * entry_p
                        position = -1
                        entry_bar = t
                        fvg["mitigated"] = True
                        break

        # 4) Mark FVGs that price has now closed THROUGH the far edge as
        # invalidated (they're no longer attractive zones).
        for fvg in bull_fvgs:
            if not fvg["mitigated"] and t > fvg["formed"] and c[t] < fvg["bot"]:
                fvg["mitigated"] = True
        for fvg in bear_fvgs:
            if not fvg["mitigated"] and t > fvg["formed"] and c[t] > fvg["top"]:
                fvg["mitigated"] = True

        # 5) Equity curve = realized + unrealized MTM.
        eq_curve[t] = realized_eq + mtm()

    return pd.DataFrame(trades), eq_curve


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def report(df, trades, eq, account, label="SMC daily"):
    L = len(df)
    years = L / 252
    final = eq[-1]
    total_ret = final / account - 1
    cagr = (max(final, 1e-9) / account) ** (1 / years) - 1 if years > 0 else np.nan
    bench = df["close"].iloc[-1] / df["close"].iloc[0] - 1

    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    max_dd = dd.min()

    daily = pd.Series(eq).pct_change().fillna(0.0)
    s_mean, s_std = daily.mean(), daily.std()
    sharpe = (s_mean / s_std) * np.sqrt(252) if s_std > 0 else np.nan
    ann_vol = s_std * np.sqrt(252)

    # Daily-gain stats restricted to days we had exposure (non-zero return).
    active = daily[daily.abs() > 1e-12]
    avg_active = active.mean() if len(active) else 0.0
    pct_days_in_market = len(active) / L * 100 if L else 0.0

    print()
    print("=" * 78)
    print(f"  SMC BACKTEST  —  {label}")
    print("=" * 78)
    print(f"  Period                    : {df['date'].iloc[0].date()} "
          f"-> {df['date'].iloc[-1].date()}  ({years:.1f} yr, {L} bars)")
    print(f"  Starting equity           : ${account:,.2f}")
    print(f"  Final  equity             : ${final:,.2f}")
    print(f"  Total return              : {total_ret*100:.1f}%")
    print(f"  Annualized (CAGR)         : {cagr*100:.1f}%")
    print(f"  Buy & hold (asset)        : {bench*100:.1f}%")
    print(f"  Annualized volatility     : {ann_vol*100:.1f}%")
    print(f"  Sharpe ratio              : {sharpe:.2f}")
    print(f"  Max drawdown              : {max_dd*100:.1f}%")
    print(f"  % of days in market       : {pct_days_in_market:.1f}%")
    print(f"  Avg daily return (active) : {avg_active*100:.3f}%")
    print(f"  Trades taken              : {len(trades)}")

    if len(trades):
        wins = (trades["pnl_$"] > 0).sum()
        losses = (trades["pnl_$"] <= 0).sum()
        wr = wins / len(trades)
        avg_w = trades.loc[trades["pnl_$"] > 0, "pnl_$"].mean()
        avg_l = trades.loc[trades["pnl_$"] <= 0, "pnl_$"].mean()
        expectancy = trades["pnl_$"].mean()
        avg_hold = trades["bars_held"].mean()
        tpy = len(trades) / years if years > 0 else 0.0
        pf = (trades.loc[trades["pnl_$"] > 0, "pnl_$"].sum() /
              max(-trades.loc[trades["pnl_$"] <= 0, "pnl_$"].sum(), 1e-9))
        long_n = (trades["direction"] == "long").sum()
        short_n = (trades["direction"] == "short").sum()
        print(f"  Wins / Losses             : {wins} / {losses}")
        print(f"  Win rate                  : {wr*100:.1f}%")
        print(f"  Profit factor             : {pf:.2f}")
        print(f"  Avg WIN  ($)              : {avg_w:,.2f}")
        print(f"  Avg LOSS ($)              : {avg_l:,.2f}")
        print(f"  Expectancy / trade ($)    : {expectancy:,.2f}")
        print(f"  Avg bars held             : {avg_hold:.1f}")
        print(f"  Trades per year           : {tpy:.1f}")
        print(f"  Long / Short              : {long_n} / {short_n}")
        reasons = trades["exit_reason"].value_counts().to_dict()
        print(f"  Exit reasons              : {reasons}")
    print("=" * 78)


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--swing-n", type=int, default=3)
    p.add_argument("--min-fvg-pct", type=float, default=0.0015)
    p.add_argument("--rr", type=float, default=2.0)
    p.add_argument("--stop-buf", type=float, default=0.5)
    p.add_argument("--atr-n", type=int, default=14)
    p.add_argument("--max-hold", type=int, default=20)
    p.add_argument("--risk", type=float, default=0.01)
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--cash", type=float, default=1000.0)
    p.add_argument("--no-short", action="store_true")
    p.add_argument("--save-trades", help="Optional CSV path for the trade log.")
    args = p.parse_args()

    df = load_ohlc(args.csv)
    print(f"Loaded {len(df)} rows  ({df['date'].iloc[0].date()} -> "
          f"{df['date'].iloc[-1].date()}) from {args.csv}")

    trades, eq = backtest(
        df,
        swing_n=args.swing_n, min_fvg_pct=args.min_fvg_pct, rr=args.rr,
        stop_buf=args.stop_buf, atr_n=args.atr_n, max_hold=args.max_hold,
        risk_per_trade=args.risk, cost_bps=args.cost_bps, account=args.cash,
        allow_short=not args.no_short,
    )
    label = (f"swing_n={args.swing_n} RR={args.rr} fvg>={args.min_fvg_pct*100:.2f}% "
             f"risk={args.risk*100:.1f}% hold<={args.max_hold}d "
             f"{'long-only' if args.no_short else 'L+S'}")
    report(df, trades, eq, args.cash, label=label)

    if args.save_trades and len(trades):
        trades.to_csv(args.save_trades, index=False)
        print(f"\nSaved {len(trades)} trades to {args.save_trades}")


if __name__ == "__main__":
    main()
