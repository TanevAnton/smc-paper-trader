"""
smc_strategy_v2.py — refined Smart-Money-Concepts strategy on daily OHLC bars.

WHAT'S NEW vs v1
----------------
v1 had a real edge (PF 1.43, Sharpe 0.77 on 20yr NDX) but gave up most of the
trend by exiting at a fixed 2R and traded too many low-quality FVGs. v2 keeps
the same primitives (swings, BOS, FVG, ATR) and adds:

  1. TREND REGIME FILTER (EMA200)
       Longs only when close > EMA200.  Shorts only when close < EMA200.
       Cuts countertrend chop where the FVG concept is weakest.

  2. PREMIUM / DISCOUNT FILTER
       Within the last confirmed swing leg [swing_low, swing_high]:
         * Long FVGs must sit in the DISCOUNT half (price <= midpoint).
         * Short FVGs must sit in the PREMIUM half (price >= midpoint).
       This is the SMC dealing-range idea — buy cheap, sell rich.

  3. FRESHNESS GATE
       Only trade an FVG within `max_bars_after_bos` bars of the aligning BOS.
       Stale setups after the impulse has run rarely follow through.

  4. PARTIAL TP + BREAK-EVEN + SWING TRAIL
       At +1R, scale out HALF the position and move stop to entry.
       Beyond that, ratchet the stop to each NEW confirmed swing low (longs)
       or swing high (shorts). This captures the right-tail trend moves v1
       was capping at 2R.

  5. POST-LOSS COOLDOWN
       After a stopped-out trade in a given direction, require a NEW BOS in
       that direction before considering more entries. Stops overtrading
       inside an exhausted leg.

  6. SESSION-LEVEL CONFLUENCE
       Bonus entry condition: the day price tests the FVG, it also reclaims
       (long) or rejects (short) the prior-day high/low — a daily proxy for
       intraday liquidity sweeps.

Everything is computed with strict no-look-ahead: signals at close of bar t,
fills at open of bar t+1.

DAILY PROFITABILITY NOTE
------------------------
"Better daily profitability" is best read as better PER-DAY MEAN return at
comparable or lower risk. No discretionary or algorithmic strategy reliably
makes money every day; what improves is the SHAPE of the daily-return
distribution (higher mean, smaller drawdowns, fatter right tail).
"""

import argparse
import sys
import itertools
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
    if len(df) < 250:
        sys.exit("ERROR: need at least ~250 OHLC rows for EMA200 + warmup.")
    return df


# ----------------------------------------------------------------------------
# Indicators
# ----------------------------------------------------------------------------
def compute_atr(h, l, c, n=14):
    """ATR series, length=len(h), SHIFTED so atr[t] only uses bars < t."""
    L = len(h)
    tr = np.zeros(L)
    tr[0] = h[0] - l[0]
    for i in range(1, L):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr = pd.Series(tr).rolling(n, min_periods=n).mean().shift(1).fillna(0.0).values
    return atr


def compute_ema(arr, n):
    """EMA series, SHIFTED by 1 so ema[t] is the EMA as of the prior bar's close."""
    s = pd.Series(arr).ewm(span=n, adjust=False).mean()
    return s.shift(1).bfill().values


def confirmed_swings(h, l, n=3):
    """Most recent CONFIRMED swing high/low known by time t (look-ahead-safe).
    A pivot at bar i is confirmed only at bar i+n."""
    L = len(h)
    sh_p = np.full(L, np.nan); sh_i = np.full(L, -1, dtype=int)
    sl_p = np.full(L, np.nan); sl_i = np.full(L, -1, dtype=int)
    last_sh_p, last_sh_bar = np.nan, -1
    last_sl_p, last_sl_bar = np.nan, -1
    for t in range(L):
        i = t - n
        if i >= n:
            ok_sh = all(h[i] > h[i - k] and h[i] > h[i + k] for k in range(1, n + 1))
            if ok_sh:
                last_sh_p, last_sh_bar = h[i], i
            ok_sl = all(l[i] < l[i - k] and l[i] < l[i + k] for k in range(1, n + 1))
            if ok_sl:
                last_sl_p, last_sl_bar = l[i], i
        sh_p[t], sh_i[t] = last_sh_p, last_sh_bar
        sl_p[t], sl_i[t] = last_sl_p, last_sl_bar
    return sh_p, sh_i, sl_p, sl_i


# ----------------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------------
def backtest_v2(
    df,
    swing_n=3,
    min_fvg_pct=0.0015,
    rr_partial=1.0,         # take half off at +rr_partial * R
    rr_runner=3.0,          # max R for the runner (also acts as final TP)
    stop_buf=0.5,           # ATR buffer beyond FVG bottom/top
    atr_n=14,
    ema_n=200,
    max_hold=30,
    max_bars_after_bos=10,  # freshness gate
    risk_per_trade=0.0075,  # 0.75% per trade
    cost_bps=5.0,
    account=1000.0,
    allow_short=True,
    use_pd_filter=True,     # premium/discount filter
    use_session_confluence=False,  # prior-day H/L reclaim
    use_trend_filter=True,  # EMA200 regime
    use_cooldown=True,      # require fresh BOS after a stop
    return_open_state=False,  # also return the live open position (for paper trading)
):
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    L = len(df)

    atr = compute_atr(h, l, c, atr_n)
    ema = compute_ema(c, ema_n)
    sh_p, sh_i, sl_p, sl_i = confirmed_swings(h, l, swing_n)

    # Trend state — flips on BOS through the latest confirmed swing.
    # Also remember the bar at which the latest BOS happened, so we can
    # gate FVG entries on freshness.
    trend = np.zeros(L, dtype=int)
    bos_bar = np.full(L, -1, dtype=int)
    state, last_bos = 0, -1
    last_used_sh = np.nan; last_used_sl = np.nan
    for t in range(L):
        if not np.isnan(sh_p[t]) and c[t] > sh_p[t] and sh_p[t] != last_used_sh:
            state, last_bos = 1, t
            last_used_sh = sh_p[t]
        if not np.isnan(sl_p[t]) and c[t] < sl_p[t] and sl_p[t] != last_used_sl:
            state, last_bos = -1, t
            last_used_sl = sl_p[t]
        trend[t] = state
        bos_bar[t] = last_bos

    bull_fvgs, bear_fvgs = [], []

    realized_eq = account
    eq_curve = np.full(L, account, dtype=float)
    position = 0
    entry_p = 0.0
    stop = 0.0
    init_stop = 0.0
    target = 0.0
    rr_partial_price = 0.0
    shares = 0.0
    shares_open = 0.0
    partial_taken = False
    entry_bar = -1
    cooldown_long_until_new_bos = False
    cooldown_short_until_new_bos = False
    last_loss_bos_bar = -1
    trade_partial_pnl = 0.0     # accumulates partial-leg PnL on the open trade
    trades = []

    def mtm(t):
        if position == 0:
            return 0.0
        return shares_open * (c[t] - entry_p) * position

    def in_discount(price, t):
        if np.isnan(sh_p[t]) or np.isnan(sl_p[t]):
            return False
        if sh_p[t] <= sl_p[t]:
            return False
        mid = 0.5 * (sh_p[t] + sl_p[t])
        return price <= mid

    def in_premium(price, t):
        if np.isnan(sh_p[t]) or np.isnan(sl_p[t]):
            return False
        if sh_p[t] <= sl_p[t]:
            return False
        mid = 0.5 * (sh_p[t] + sl_p[t])
        return price >= mid

    for t in range(L):
        # ---- 1) Register new FVGs formed at bar t ----
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

        # ---- 2) Manage open trade (intrabar fills) ----
        exit_price, exit_reason = None, None
        if position != 0:
            # Trail logic first — only ratchets, never loosens.
            if position == 1:
                # If price has gone +rr_partial*R, ensure stop >= break-even.
                if not partial_taken and h[t] >= rr_partial_price:
                    # Scale out half at the partial level.
                    fill = rr_partial_price
                    qty_out = shares_open * 0.5
                    gross = qty_out * (fill - entry_p)
                    cost = qty_out * fill * (cost_bps / 1e4)
                    net = gross - cost
                    realized_eq += net
                    trade_partial_pnl += net
                    shares_open -= qty_out
                    partial_taken = True
                    # Move stop to break-even (entry).
                    stop = max(stop, entry_p)
                # After partial, ratchet stop to latest confirmed swing low.
                if partial_taken and not np.isnan(sl_p[t]):
                    stop = max(stop, sl_p[t])
                # Check stop / final TP / time stop.
                if l[t] <= stop:
                    exit_price, exit_reason = stop, "stop/trail"
                elif h[t] >= target:
                    exit_price, exit_reason = target, "target"
                elif t - entry_bar >= max_hold:
                    exit_price, exit_reason = c[t], "timeout"
            else:  # short
                if not partial_taken and l[t] <= rr_partial_price:
                    fill = rr_partial_price
                    qty_out = shares_open * 0.5
                    gross = qty_out * (entry_p - fill)
                    cost = qty_out * fill * (cost_bps / 1e4)
                    net = gross - cost
                    realized_eq += net
                    trade_partial_pnl += net
                    shares_open -= qty_out
                    partial_taken = True
                    stop = min(stop, entry_p)
                if partial_taken and not np.isnan(sh_p[t]):
                    stop = min(stop, sh_p[t])
                if h[t] >= stop:
                    exit_price, exit_reason = stop, "stop/trail"
                elif l[t] <= target:
                    exit_price, exit_reason = target, "target"
                elif t - entry_bar >= max_hold:
                    exit_price, exit_reason = c[t], "timeout"

            if exit_price is not None:
                gross = shares_open * (exit_price - entry_p) * position
                cost = shares_open * exit_price * (cost_bps / 1e4)
                net_runner = gross - cost
                realized_eq += net_runner
                total_pnl = trade_partial_pnl + net_runner
                # Whole-trade R-multiple: total $ / risk_per_share / original_shares.
                r_per_share = abs(entry_p - init_stop)
                original_risk_usd = shares * r_per_share
                trade_r = total_pnl / original_risk_usd if original_risk_usd > 0 else 0.0
                is_loss = total_pnl <= 0
                trades.append({
                    "entry_date": df["date"].iloc[entry_bar].date(),
                    "exit_date":  df["date"].iloc[t].date(),
                    "direction": "long" if position == 1 else "short",
                    "entry": round(entry_p, 4),
                    "exit": round(exit_price, 4),
                    "init_stop": round(init_stop, 4),
                    "target": round(target, 4),
                    "bars_held": t - entry_bar,
                    "partial": partial_taken,
                    "pnl_$": round(total_pnl, 2),
                    "r_total": round(trade_r, 2),
                    "exit_reason": exit_reason,
                })
                if use_cooldown and is_loss:
                    if position == 1:
                        cooldown_long_until_new_bos = True
                        last_loss_bos_bar = bos_bar[t]
                    else:
                        cooldown_short_until_new_bos = True
                        last_loss_bos_bar = bos_bar[t]
                position = 0
                shares_open = 0.0
                partial_taken = False
                trade_partial_pnl = 0.0

        # ---- 3) Cooldown lift: a fresh BOS clears it ----
        if cooldown_long_until_new_bos and bos_bar[t] != last_loss_bos_bar and trend[t] == 1:
            cooldown_long_until_new_bos = False
        if cooldown_short_until_new_bos and bos_bar[t] != last_loss_bos_bar and trend[t] == -1:
            cooldown_short_until_new_bos = False

        # ---- 4) Look for a new entry ----
        if position == 0 and t > max(swing_n + atr_n, ema_n) + 2:
            # Direction by trend (BOS). Add EMA200 regime filter on top.
            cand_dir = trend[t]
            if cand_dir == 0:
                pass
            elif use_trend_filter:
                if cand_dir == 1 and c[t] < ema[t]:
                    cand_dir = 0
                if cand_dir == -1 and c[t] > ema[t]:
                    cand_dir = 0

            # Freshness gate.
            if cand_dir != 0 and (t - bos_bar[t]) > max_bars_after_bos:
                cand_dir = 0

            # Cooldown.
            if cand_dir == 1 and cooldown_long_until_new_bos:
                cand_dir = 0
            if cand_dir == -1 and cooldown_short_until_new_bos:
                cand_dir = 0

            if cand_dir == 1:
                # Iterate newest FVG first.
                for fvg in reversed(bull_fvgs):
                    if fvg["mitigated"] or fvg["formed"] >= t:
                        continue
                    if fvg["top"] >= o[t]:
                        continue
                    # Wick into zone but not closed beyond it.
                    if not (l[t] <= fvg["top"] and l[t] >= fvg["bot"]):
                        continue
                    # Premium/discount filter on the FVG TOP (entry price).
                    if use_pd_filter and not in_discount(fvg["top"], t):
                        continue
                    # Session confluence: prior-day low reclaim.
                    if use_session_confluence and t >= 1:
                        pdl = l[t - 1]
                        if not (l[t] <= pdl and c[t] > pdl):
                            continue
                    # Build the trade.
                    entry_p = fvg["top"]
                    init_stop = fvg["bot"] - stop_buf * atr[t]
                    r_per_share = entry_p - init_stop
                    if r_per_share <= 0:
                        fvg["mitigated"] = True
                        continue
                    target = entry_p + rr_runner * r_per_share
                    rr_partial_price = entry_p + rr_partial * r_per_share
                    risk_dollars = realized_eq * risk_per_trade
                    shares = risk_dollars / r_per_share
                    shares_open = shares
                    stop = init_stop
                    position = 1
                    partial_taken = False
                    entry_bar = t
                    fvg["mitigated"] = True
                    # Pay entry cost.
                    realized_eq -= shares * entry_p * (cost_bps / 1e4)
                    break
            elif cand_dir == -1 and allow_short:
                for fvg in reversed(bear_fvgs):
                    if fvg["mitigated"] or fvg["formed"] >= t:
                        continue
                    if fvg["bot"] <= o[t]:
                        continue
                    if not (h[t] >= fvg["bot"] and h[t] <= fvg["top"]):
                        continue
                    if use_pd_filter and not in_premium(fvg["bot"], t):
                        continue
                    if use_session_confluence and t >= 1:
                        pdh = h[t - 1]
                        if not (h[t] >= pdh and c[t] < pdh):
                            continue
                    entry_p = fvg["bot"]
                    init_stop = fvg["top"] + stop_buf * atr[t]
                    r_per_share = init_stop - entry_p
                    if r_per_share <= 0:
                        fvg["mitigated"] = True
                        continue
                    target = entry_p - rr_runner * r_per_share
                    rr_partial_price = entry_p - rr_partial * r_per_share
                    risk_dollars = realized_eq * risk_per_trade
                    shares = risk_dollars / r_per_share
                    shares_open = shares
                    stop = init_stop
                    position = -1
                    partial_taken = False
                    entry_bar = t
                    fvg["mitigated"] = True
                    realized_eq -= shares * entry_p * (cost_bps / 1e4)
                    break

        # ---- 5) Invalidate FVGs that price has closed THROUGH ----
        for fvg in bull_fvgs:
            if not fvg["mitigated"] and t > fvg["formed"] and c[t] < fvg["bot"]:
                fvg["mitigated"] = True
        for fvg in bear_fvgs:
            if not fvg["mitigated"] and t > fvg["formed"] and c[t] > fvg["top"]:
                fvg["mitigated"] = True

        # ---- 6) Mark equity ----
        eq_curve[t] = realized_eq + mtm(t)

    if return_open_state:
        open_state = None
        if position != 0:
            open_state = {
                "direction": "long" if position == 1 else "short",
                "entry_date": str(df["date"].iloc[entry_bar].date()),
                "entry_price": float(entry_p),
                "stop": float(stop),
                "init_stop": float(init_stop),
                "target": float(target),
                "shares_open": float(shares_open),
                "partial_taken": bool(partial_taken),
                "bars_held": int((L - 1) - entry_bar),
                "last_close": float(c[-1]),
                "unrealized_$": float(shares_open * (c[-1] - entry_p) * position),
            }
        return pd.DataFrame(trades), eq_curve, open_state
    return pd.DataFrame(trades), eq_curve


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def report(df, trades, eq, account, label="SMC v2"):
    L = len(df)
    years = L / 252
    final = eq[-1]
    total_ret = final / account - 1
    cagr = (max(final, 1e-9) / account) ** (1 / years) - 1 if years > 0 else np.nan
    bench = df["close"].iloc[-1] / df["close"].iloc[0] - 1
    bh_cagr = (df["close"].iloc[-1] / df["close"].iloc[0]) ** (1 / years) - 1

    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    max_dd = dd.min()

    daily = pd.Series(eq).pct_change().fillna(0.0)
    sharpe = (daily.mean() / daily.std()) * np.sqrt(252) if daily.std() > 0 else np.nan
    ann_vol = daily.std() * np.sqrt(252)
    sortino_denom = daily[daily < 0].std()
    sortino = (daily.mean() / sortino_denom) * np.sqrt(252) if sortino_denom > 0 else np.nan

    # Buy & hold daily stats (for fair comparison).
    bh = df["close"].pct_change().fillna(0.0)
    bh_sharpe = (bh.mean() / bh.std()) * np.sqrt(252) if bh.std() > 0 else np.nan
    bh_eq = (1 + bh).cumprod() * account
    bh_peak = np.maximum.accumulate(bh_eq.values)
    bh_dd = (bh_eq.values / bh_peak - 1).min()

    active = daily[daily.abs() > 1e-12]
    avg_active = active.mean() if len(active) else 0.0
    pct_in_market = len(active) / L * 100 if L else 0.0

    print()
    print("=" * 80)
    print(f"  SMC BACKTEST  —  {label}")
    print("=" * 80)
    print(f"  Period                    : {df['date'].iloc[0].date()} -> "
          f"{df['date'].iloc[-1].date()}  ({years:.1f} yr, {L} bars)")
    print(f"  Starting equity           : ${account:,.2f}")
    print(f"  Final  equity             : ${final:,.2f}")
    print(f"  Total return              : {total_ret*100:.1f}%   "
          f"(Buy&Hold: {bench*100:.1f}%)")
    print(f"  CAGR                      : {cagr*100:.2f}%   "
          f"(Buy&Hold: {bh_cagr*100:.2f}%)")
    print(f"  Annualized volatility     : {ann_vol*100:.1f}%")
    print(f"  Sharpe ratio              : {sharpe:.2f}   "
          f"(Buy&Hold: {bh_sharpe:.2f})")
    print(f"  Sortino ratio             : {sortino:.2f}")
    print(f"  Max drawdown              : {max_dd*100:.1f}%   "
          f"(Buy&Hold: {bh_dd*100:.1f}%)")
    print(f"  % of days in market       : {pct_in_market:.1f}%")
    print(f"  Avg daily return (active) : {avg_active*100:.3f}%")
    print(f"  Avg daily return (all)    : {daily.mean()*100:.4f}%")

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
        reasons = trades["exit_reason"].value_counts().to_dict()
        print(f"  Trades taken              : {len(trades)}  "
              f"(Long {long_n} / Short {short_n})")
        print(f"  Wins / Losses             : {wins} / {losses}  ({wr*100:.1f}%)")
        print(f"  Profit factor             : {pf:.2f}")
        print(f"  Avg WIN  ($)              : {avg_w:,.2f}")
        print(f"  Avg LOSS ($)              : {avg_l:,.2f}")
        print(f"  Expectancy / trade ($)    : {expectancy:,.2f}")
        print(f"  Avg bars held             : {avg_hold:.1f}")
        print(f"  Trades per year           : {tpy:.1f}")
        print(f"  Exit reasons              : {reasons}")
    print("=" * 80)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="NDX.csv")
    p.add_argument("--swing-n", type=int, default=3)
    p.add_argument("--min-fvg-pct", type=float, default=0.0015)
    p.add_argument("--rr-partial", type=float, default=1.0)
    p.add_argument("--rr-runner", type=float, default=3.0)
    p.add_argument("--stop-buf", type=float, default=0.5)
    p.add_argument("--atr-n", type=int, default=14)
    p.add_argument("--ema-n", type=int, default=200)
    p.add_argument("--max-hold", type=int, default=30)
    p.add_argument("--max-bars-after-bos", type=int, default=10)
    p.add_argument("--risk", type=float, default=0.0075)
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--cash", type=float, default=1000.0)
    p.add_argument("--no-short", action="store_true")
    p.add_argument("--no-pd-filter", action="store_true")
    p.add_argument("--no-trend-filter", action="store_true")
    p.add_argument("--no-cooldown", action="store_true")
    p.add_argument("--session-confluence", action="store_true")
    p.add_argument("--from-date")
    p.add_argument("--to-date")
    p.add_argument("--sweep", action="store_true",
                   help="Grid search a small set of params and print a sorted table.")
    p.add_argument("--save-trades")
    args = p.parse_args()

    df = load_ohlc(args.csv)
    if args.from_date:
        df = df[df["date"] >= pd.Timestamp(args.from_date)].reset_index(drop=True)
    if args.to_date:
        df = df[df["date"] <= pd.Timestamp(args.to_date)].reset_index(drop=True)
    print(f"Loaded {len(df)} rows  ({df['date'].iloc[0].date()} -> "
          f"{df['date'].iloc[-1].date()}) from {args.csv}")

    if args.sweep:
        # Small disciplined grid (≈80 combos) — keeps overfitting in check.
        grid = {
            "swing_n": [2, 3, 5],
            "rr_runner": [2.0, 3.0, 5.0],
            "max_bars_after_bos": [5, 10, 20],
            "min_fvg_pct": [0.001, 0.002, 0.004],
        }
        rows = []
        for sn, rrr, mb, mfp in itertools.product(
                grid["swing_n"], grid["rr_runner"],
                grid["max_bars_after_bos"], grid["min_fvg_pct"]):
            trades, eq = backtest_v2(
                df, swing_n=sn, rr_runner=rrr,
                max_bars_after_bos=mb, min_fvg_pct=mfp,
                rr_partial=args.rr_partial, stop_buf=args.stop_buf,
                atr_n=args.atr_n, ema_n=args.ema_n, max_hold=args.max_hold,
                risk_per_trade=args.risk, cost_bps=args.cost_bps,
                account=args.cash, allow_short=not args.no_short,
                use_pd_filter=not args.no_pd_filter,
                use_trend_filter=not args.no_trend_filter,
                use_cooldown=not args.no_cooldown,
                use_session_confluence=args.session_confluence,
            )
            L = len(df); years = L / 252
            final = eq[-1]
            cagr = (max(final, 1e-9) / args.cash) ** (1 / years) - 1
            daily = pd.Series(eq).pct_change().fillna(0.0)
            sh = (daily.mean() / daily.std()) * np.sqrt(252) if daily.std() > 0 else 0.0
            peak = np.maximum.accumulate(eq); mdd = (eq / peak - 1).min()
            pf = 0.0
            if len(trades):
                wpnl = trades.loc[trades["pnl_$"] > 0, "pnl_$"].sum()
                lpnl = -trades.loc[trades["pnl_$"] <= 0, "pnl_$"].sum()
                pf = wpnl / lpnl if lpnl > 0 else float("inf")
            rows.append({
                "swing_n": sn, "rr_runner": rrr, "mb_bos": mb,
                "min_fvg_pct": mfp, "trades": len(trades),
                "CAGR%": round(cagr * 100, 2),
                "Sharpe": round(sh, 2),
                "MaxDD%": round(mdd * 100, 1),
                "PF": round(pf, 2),
                "Final$": round(final, 0),
            })
        res = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
        print("\nParameter sweep (sorted by Sharpe):")
        with pd.option_context("display.max_rows", None, "display.width", 130):
            print(res.to_string(index=False))
        return

    trades, eq = backtest_v2(
        df,
        swing_n=args.swing_n, min_fvg_pct=args.min_fvg_pct,
        rr_partial=args.rr_partial, rr_runner=args.rr_runner,
        stop_buf=args.stop_buf, atr_n=args.atr_n, ema_n=args.ema_n,
        max_hold=args.max_hold, max_bars_after_bos=args.max_bars_after_bos,
        risk_per_trade=args.risk, cost_bps=args.cost_bps,
        account=args.cash, allow_short=not args.no_short,
        use_pd_filter=not args.no_pd_filter,
        use_trend_filter=not args.no_trend_filter,
        use_cooldown=not args.no_cooldown,
        use_session_confluence=args.session_confluence,
    )
    label = (f"v2 swing_n={args.swing_n} rr={args.rr_partial}/{args.rr_runner} "
             f"fvg>={args.min_fvg_pct*100:.2f}% risk={args.risk*100:.2f}% "
             f"hold<={args.max_hold}d bos_age<={args.max_bars_after_bos} "
             f"{'L+S' if not args.no_short else 'L-only'}"
             f"{' +trend' if not args.no_trend_filter else ''}"
             f"{' +PD' if not args.no_pd_filter else ''}"
             f"{' +cool' if not args.no_cooldown else ''}"
             f"{' +sess' if args.session_confluence else ''}")
    report(df, trades, eq, args.cash, label=label)

    if args.save_trades and len(trades):
        trades.to_csv(args.save_trades, index=False)
        print(f"\nSaved {len(trades)} trades to {args.save_trades}")


if __name__ == "__main__":
    main()
