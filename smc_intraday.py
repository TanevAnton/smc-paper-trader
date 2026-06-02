"""
smc_intraday.py — intraday Smart-Money-Concepts strategy for 5-minute bars.

WHAT IT TRADES
--------------
QQQ (or any single-ticker 5-min OHLCV file) using the same SMC primitives that
worked on daily bars, now adapted to a real intraday session:

  * Prior-Day High / Low                (PDH / PDL — true session liquidity)
  * Opening Range High / Low            (first 30 min after NY open)
  * Swing pivots on 5-min bars          (fractal lookback k each side)
  * Break of Structure (BOS)            (close beyond last confirmed swing)
  * Fair Value Gaps (FVG)               (3-bar imbalance on 5m)
  * Premium / Discount of recent leg
  * Partial TP + Break-even + Swing trail
  * Time-of-day filter                  (skip first 30m noise, last 30m for new entries)

EXECUTION REALISM
-----------------
- Signals decided at close of 5m bar t, fills at open of bar t+1.
- Intrabar stop / TP fills: assume STOP fires first if both hit on the same bar
  (conservative).
- Costs: cost_bps per side on traded notional (default 2 bps each side — QQQ
  is tight: $0.01 spread on a $700 instrument is ~1.4 bps).
- The algo can hold across the overnight gap (per your request); positions
  carry next-day open-gap risk just like a real swing trade.

POSITION SIZING
---------------
Fixed-fractional risk: `risk_per_trade` of CURRENT equity divided by the stop
distance gives share count. With a $1k account and 0.75% risk, a single loss
costs ~$7.50. With 1m or 5m timeframes you get many shots, so keep risk small.

USAGE
-----
    python smc_intraday.py --csv QQQ_5m.csv
    python smc_intraday.py --csv QQQ_5m.csv --risk 0.01 --rr-runner 4.0
    python smc_intraday.py --csv QQQ_5m.csv --sweep
"""

import argparse
import itertools
import sys

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_intraday(path: str, tz: str = "America/New_York") -> pd.DataFrame:
    """Load intraday OHLCV CSV (Datetime/Open/High/Low/Close/Volume).

    yfinance writes Datetime in UTC. We convert to NY time so session filters
    map directly to 09:30 / 10:00 / 15:30 / 16:00 etc.
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    need = ["datetime", "open", "high", "low", "close"]
    for k in need:
        if k not in cols:
            sys.exit(f"ERROR: CSV missing column {k!r}. Have: {list(df.columns)}")
    df = df.rename(columns={cols[k]: k for k in need})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    df["datetime"] = df["datetime"].dt.tz_convert(tz)

    # Derive session calendar columns.
    df["date"] = df["datetime"].dt.date
    df["time"] = df["datetime"].dt.strftime("%H:%M")
    df["minute_of_day"] = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute

    # Regular trading hours filter (QQQ): 09:30 - 16:00 ET.
    rth = (df["minute_of_day"] >= 9 * 60 + 30) & (df["minute_of_day"] < 16 * 60)
    df = df[rth].reset_index(drop=True)

    if len(df) < 200:
        sys.exit(f"ERROR: only {len(df)} RTH bars — need at least ~200.")
    return df


# ----------------------------------------------------------------------------
# Timeframe helper — annualization must match the bar size (5m vs 1h vs ...).
# ----------------------------------------------------------------------------
def bars_per_year(df):
    """Detect bars-per-trading-day from the data and return the annualization
    bar count (252 trading days * bars/day). Works for 5m (~78), 15m (~26),
    1h (~7), etc., so Sharpe/Sortino are correct for any intraday timeframe."""
    n_days = max(1, df["date"].nunique())
    bpd = len(df) / n_days
    return 252.0 * bpd


# ----------------------------------------------------------------------------
# Indicators
# ----------------------------------------------------------------------------
def compute_atr(h, l, c, n=14):
    """ATR, length=len(h), SHIFTED so atr[t] only uses bars < t."""
    L = len(h)
    tr = np.zeros(L)
    tr[0] = h[0] - l[0]
    for i in range(1, L):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    return pd.Series(tr).rolling(n, min_periods=n).mean().shift(1).fillna(0.0).values


def compute_ema(arr, n):
    s = pd.Series(arr).ewm(span=n, adjust=False).mean()
    return s.shift(1).bfill().values


def confirmed_swings(h, l, n=3):
    """Most recent CONFIRMED swing high/low known by time t (no look-ahead)."""
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
# Session levels: prior-day H/L and opening-range H/L, per bar.
# ----------------------------------------------------------------------------
def session_levels(df):
    """For each bar return (pdh, pdl, orh, orl):
        pdh/pdl  : high/low of the PRIOR trading day (NaN on first day)
        orh/orl  : high/low of TODAY'S first 30 minutes (NaN until 10:00 ET)
    All look-ahead safe.
    """
    L = len(df)
    pdh = np.full(L, np.nan)
    pdl = np.full(L, np.nan)
    orh = np.full(L, np.nan)
    orl = np.full(L, np.nan)

    # Day boundaries.
    days = df["date"].values
    minute = df["minute_of_day"].values
    h = df["high"].values
    l = df["low"].values

    # Group indices by day.
    day_to_indices = {}
    for i, d in enumerate(days):
        day_to_indices.setdefault(d, []).append(i)

    sorted_days = sorted(day_to_indices.keys())
    prev_day_high = np.nan
    prev_day_low = np.nan
    for d in sorted_days:
        idxs = day_to_indices[d]
        # Opening range = bars with minute_of_day in [9:30, 10:00) = 6 bars.
        or_idxs = [i for i in idxs if minute[i] < 10 * 60]
        cur_orh = max(h[i] for i in or_idxs) if or_idxs else np.nan
        cur_orl = min(l[i] for i in or_idxs) if or_idxs else np.nan

        # Within the day: PDH/PDL set from prior day (constant within day).
        # ORH/ORL: NaN until the opening range completes (last OR bar's close
        # is the earliest a trader could know them). i.e. orh/orl known from
        # the bar AFTER the last OR bar.
        last_or_bar = max(or_idxs) if or_idxs else -1
        for i in idxs:
            pdh[i] = prev_day_high
            pdl[i] = prev_day_low
            if i > last_or_bar:
                orh[i] = cur_orh
                orl[i] = cur_orl

        prev_day_high = max(h[i] for i in idxs)
        prev_day_low = min(l[i] for i in idxs)

    return pdh, pdl, orh, orl


# ----------------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------------
def backtest_intraday(
    df,
    swing_n=3,
    min_fvg_pct=0.0008,        # 8 bps — smaller because 5m bars are tighter
    rr_partial=1.0,
    rr_runner=3.0,
    stop_buf=0.5,              # ATR buffer
    atr_n=14,
    ema_n=200,                 # EMA200 on 5m ~= regime over ~17 hours
    max_hold_bars=300,         # 300 5m bars = ~4 days of RTH
    max_bars_after_bos=12,     # 12 * 5m = 1 hour of freshness
    risk_per_trade=0.0075,
    cost_bps=2.0,              # per side
    account=10000.0,
    allow_short=True,
    use_pd_filter=True,
    use_trend_filter=True,
    use_cooldown=True,
    use_session_confluence=True,    # require PDH/PDL or ORH/ORL interaction
    no_new_entries_after_min=15 * 60 + 30,  # 15:30 ET cutoff (last 30m)
    no_entries_before_min=10 * 60,           # 10:00 ET cutoff (skip first 30m)
    flatten_at_session_close=False,
):
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    minute = df["minute_of_day"].values
    days = df["date"].values
    L = len(df)

    atr = compute_atr(h, l, c, atr_n)
    ema = compute_ema(c, ema_n)
    sh_p, sh_i, sl_p, sl_i = confirmed_swings(h, l, swing_n)
    pdh, pdl, orh, orl = session_levels(df)

    # Trend state — flips on BOS through the latest confirmed swing.
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
    trade_partial_pnl = 0.0
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
        return price <= 0.5 * (sh_p[t] + sl_p[t])

    def in_premium(price, t):
        if np.isnan(sh_p[t]) or np.isnan(sl_p[t]):
            return False
        if sh_p[t] <= sl_p[t]:
            return False
        return price >= 0.5 * (sh_p[t] + sl_p[t])

    def session_confluence_long(t):
        """At bar t, did price reclaim PDL or ORL (sweep + recover)?"""
        if not use_session_confluence:
            return True
        pdl_t, orl_t = pdl[t], orl[t]
        # Wicked through liquidity below and closed back above counts as a sweep.
        if not np.isnan(pdl_t) and l[t] <= pdl_t and c[t] > pdl_t:
            return True
        if not np.isnan(orl_t) and l[t] <= orl_t and c[t] > orl_t:
            return True
        return False

    def session_confluence_short(t):
        if not use_session_confluence:
            return True
        pdh_t, orh_t = pdh[t], orh[t]
        if not np.isnan(pdh_t) and h[t] >= pdh_t and c[t] < pdh_t:
            return True
        if not np.isnan(orh_t) and h[t] >= orh_t and c[t] < orh_t:
            return True
        return False

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

        # ---- 2) Manage open trade ----
        exit_price, exit_reason = None, None
        force_flat_now = False
        if position != 0 and flatten_at_session_close and minute[t] >= 15 * 60 + 55:
            # Force flat in the very last 5m bar of the day.
            force_flat_now = True

        if position == 1:
            if not partial_taken and h[t] >= rr_partial_price:
                fill = rr_partial_price
                qty_out = shares_open * 0.5
                gross = qty_out * (fill - entry_p)
                cost = qty_out * fill * (cost_bps / 1e4)
                net = gross - cost
                realized_eq += net
                trade_partial_pnl += net
                shares_open -= qty_out
                partial_taken = True
                stop = max(stop, entry_p)
            if partial_taken and not np.isnan(sl_p[t]):
                stop = max(stop, sl_p[t])
            if l[t] <= stop:
                exit_price, exit_reason = stop, "stop/trail"
            elif h[t] >= target:
                exit_price, exit_reason = target, "target"
            elif t - entry_bar >= max_hold_bars:
                exit_price, exit_reason = c[t], "timeout"
            elif force_flat_now:
                exit_price, exit_reason = c[t], "session-flat"
        elif position == -1:
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
            elif t - entry_bar >= max_hold_bars:
                exit_price, exit_reason = c[t], "timeout"
            elif force_flat_now:
                exit_price, exit_reason = c[t], "session-flat"

        if exit_price is not None:
            gross = shares_open * (exit_price - entry_p) * position
            cost = shares_open * exit_price * (cost_bps / 1e4)
            net_runner = gross - cost
            realized_eq += net_runner
            total_pnl = trade_partial_pnl + net_runner
            r_per_share = abs(entry_p - init_stop)
            original_risk_usd = shares * r_per_share
            trade_r = total_pnl / original_risk_usd if original_risk_usd > 0 else 0.0
            is_loss = total_pnl <= 0
            trades.append({
                "entry_dt": df["datetime"].iloc[entry_bar],
                "exit_dt":  df["datetime"].iloc[t],
                "dir": "L" if position == 1 else "S",
                "entry": round(entry_p, 4),
                "exit": round(exit_price, 4),
                "init_stop": round(init_stop, 4),
                "target": round(target, 4),
                "bars_held": t - entry_bar,
                "partial": partial_taken,
                "pnl_$": round(total_pnl, 2),
                "r_total": round(trade_r, 2),
                "reason": exit_reason,
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

        # ---- 3) Cooldown lift on fresh BOS ----
        if cooldown_long_until_new_bos and bos_bar[t] != last_loss_bos_bar and trend[t] == 1:
            cooldown_long_until_new_bos = False
        if cooldown_short_until_new_bos and bos_bar[t] != last_loss_bos_bar and trend[t] == -1:
            cooldown_short_until_new_bos = False

        # ---- 4) Look for a new entry ----
        in_entry_window = (no_entries_before_min <= minute[t] < no_new_entries_after_min)
        if position == 0 and in_entry_window and t > max(swing_n + atr_n, ema_n) + 2:
            cand_dir = trend[t]
            if cand_dir != 0 and use_trend_filter:
                if cand_dir == 1 and c[t] < ema[t]:
                    cand_dir = 0
                if cand_dir == -1 and c[t] > ema[t]:
                    cand_dir = 0
            if cand_dir != 0 and (t - bos_bar[t]) > max_bars_after_bos:
                cand_dir = 0
            if cand_dir == 1 and cooldown_long_until_new_bos:
                cand_dir = 0
            if cand_dir == -1 and cooldown_short_until_new_bos:
                cand_dir = 0
            if cand_dir == 1 and not session_confluence_long(t):
                cand_dir = 0
            if cand_dir == -1 and not session_confluence_short(t):
                cand_dir = 0

            if cand_dir == 1 and t + 1 < L:
                for fvg in reversed(bull_fvgs):
                    if fvg["mitigated"] or fvg["formed"] >= t:
                        continue
                    if fvg["top"] >= o[t]:
                        continue
                    if not (l[t] <= fvg["top"] and l[t] >= fvg["bot"]):
                        continue
                    if use_pd_filter and not in_discount(fvg["top"], t):
                        continue
                    entry_p = fvg["top"]
                    init_stop = fvg["bot"] - stop_buf * atr[t]
                    r_per_share = entry_p - init_stop
                    if r_per_share <= 0:
                        fvg["mitigated"] = True
                        continue
                    target = entry_p + rr_runner * r_per_share
                    rr_partial_price = entry_p + rr_partial * r_per_share
                    shares = (realized_eq * risk_per_trade) / r_per_share
                    shares_open = shares
                    stop = init_stop
                    position = 1
                    partial_taken = False
                    entry_bar = t
                    fvg["mitigated"] = True
                    realized_eq -= shares * entry_p * (cost_bps / 1e4)
                    break
            elif cand_dir == -1 and allow_short and t + 1 < L:
                for fvg in reversed(bear_fvgs):
                    if fvg["mitigated"] or fvg["formed"] >= t:
                        continue
                    if fvg["bot"] <= o[t]:
                        continue
                    if not (h[t] >= fvg["bot"] and h[t] <= fvg["top"]):
                        continue
                    if use_pd_filter and not in_premium(fvg["bot"], t):
                        continue
                    entry_p = fvg["bot"]
                    init_stop = fvg["top"] + stop_buf * atr[t]
                    r_per_share = init_stop - entry_p
                    if r_per_share <= 0:
                        fvg["mitigated"] = True
                        continue
                    target = entry_p - rr_runner * r_per_share
                    rr_partial_price = entry_p - rr_partial * r_per_share
                    shares = (realized_eq * risk_per_trade) / r_per_share
                    shares_open = shares
                    stop = init_stop
                    position = -1
                    partial_taken = False
                    entry_bar = t
                    fvg["mitigated"] = True
                    realized_eq -= shares * entry_p * (cost_bps / 1e4)
                    break

        # ---- 5) Invalidate FVGs price has closed through ----
        for fvg in bull_fvgs:
            if not fvg["mitigated"] and t > fvg["formed"] and c[t] < fvg["bot"]:
                fvg["mitigated"] = True
        for fvg in bear_fvgs:
            if not fvg["mitigated"] and t > fvg["formed"] and c[t] > fvg["top"]:
                fvg["mitigated"] = True

        eq_curve[t] = realized_eq + mtm(t)

    return pd.DataFrame(trades), eq_curve


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def report(df, trades, eq, account, label):
    L = len(df)
    # Trading days in sample.
    n_days = df["date"].nunique()
    bpy = bars_per_year(df)     # auto-detect: 5m~19656, 1h~1764, etc.
    years = L / bpy

    final = eq[-1]
    total_ret = final / account - 1
    cagr = (max(final, 1e-9) / account) ** (1 / years) - 1 if years > 0 else 0.0
    bench = df["close"].iloc[-1] / df["close"].iloc[0] - 1
    bh_cagr = (df["close"].iloc[-1] / df["close"].iloc[0]) ** (1 / years) - 1

    peak = np.maximum.accumulate(eq)
    max_dd = (eq / peak - 1).min()

    bar_ret = pd.Series(eq).pct_change().fillna(0.0)
    # Sharpe annualized using the detected bar size.
    sharpe = (bar_ret.mean() / bar_ret.std()) * np.sqrt(bpy) if bar_ret.std() > 0 else 0.0
    ann_vol = bar_ret.std() * np.sqrt(bpy)
    sortino_d = bar_ret[bar_ret < 0].std()
    sortino = (bar_ret.mean() / sortino_d) * np.sqrt(bpy) if sortino_d > 0 else 0.0

    # Buy & hold benchmark.
    bh_bar = df["close"].pct_change().fillna(0.0)
    bh_sharpe = (bh_bar.mean() / bh_bar.std()) * np.sqrt(bpy) if bh_bar.std() > 0 else 0.0
    bh_eq = (1 + bh_bar).cumprod() * account
    bh_dd = (bh_eq.values / np.maximum.accumulate(bh_eq.values) - 1).min()

    # Per-day P&L.
    eq_series = pd.Series(eq, index=df["datetime"].values)
    daily_eq = pd.Series(eq, index=pd.to_datetime(df["datetime"]).dt.date.values)
    end_of_day_eq = daily_eq.groupby(daily_eq.index).last()
    daily_pnl = end_of_day_eq.diff().fillna(end_of_day_eq.iloc[0] - account)
    pct_green_days = (daily_pnl > 0).mean()
    avg_daily = daily_pnl.mean()
    median_daily = daily_pnl.median()
    best_day, worst_day = daily_pnl.max(), daily_pnl.min()

    print()
    print("=" * 86)
    print(f"  INTRADAY SMC BACKTEST  —  {label}")
    print("=" * 86)
    print(f"  Period               : {df['datetime'].iloc[0]} -> "
          f"{df['datetime'].iloc[-1]}")
    print(f"                         {n_days} trading days, {L} bars "
          f"(~{L/max(n_days,1):.0f}/day)")
    print(f"  Starting equity      : ${account:,.2f}")
    print(f"  Final  equity        : ${final:,.2f}")
    print(f"  Total return         : {total_ret*100:.2f}%   "
          f"(Buy&Hold: {bench*100:.2f}%)")
    print(f"  Annualized (CAGR)    : {cagr*100:.1f}%   "
          f"(Buy&Hold: {bh_cagr*100:.1f}%)")
    print(f"  Annualized vol       : {ann_vol*100:.1f}%")
    print(f"  Sharpe               : {sharpe:.2f}   "
          f"(Buy&Hold: {bh_sharpe:.2f})")
    print(f"  Sortino              : {sortino:.2f}")
    print(f"  Max drawdown         : {max_dd*100:.2f}%   "
          f"(Buy&Hold: {bh_dd*100:.2f}%)")

    print()
    print(f"  Trading days         : {n_days}")
    print(f"  Avg daily P&L        : ${avg_daily:,.2f}")
    print(f"  Median daily P&L     : ${median_daily:,.2f}")
    print(f"  Best day             : ${best_day:,.2f}")
    print(f"  Worst day            : ${worst_day:,.2f}")
    print(f"  % green days         : {pct_green_days*100:.1f}%")

    if len(trades):
        wins = (trades["pnl_$"] > 0).sum()
        losses = (trades["pnl_$"] <= 0).sum()
        wr = wins / len(trades)
        wpnl = trades.loc[trades["pnl_$"] > 0, "pnl_$"].sum()
        lpnl = -trades.loc[trades["pnl_$"] <= 0, "pnl_$"].sum()
        pf = wpnl / lpnl if lpnl > 0 else float("inf")
        avg_w = trades.loc[trades["pnl_$"] > 0, "pnl_$"].mean() if wins else 0.0
        avg_l = trades.loc[trades["pnl_$"] <= 0, "pnl_$"].mean() if losses else 0.0
        expectancy = trades["pnl_$"].mean()
        avg_hold_bars = trades["bars_held"].mean()
        avg_hold_min = avg_hold_bars * 5
        long_n = (trades["dir"] == "L").sum()
        short_n = (trades["dir"] == "S").sum()
        reasons = trades["reason"].value_counts().to_dict()
        print()
        print(f"  Trades               : {len(trades)}  (L {long_n} / S {short_n})")
        print(f"  Trades / day         : {len(trades) / n_days:.2f}")
        print(f"  Wins / Losses        : {wins} / {losses}  ({wr*100:.1f}%)")
        print(f"  Profit factor        : {pf:.2f}")
        print(f"  Avg WIN  ($)         : {avg_w:,.2f}")
        print(f"  Avg LOSS ($)         : {avg_l:,.2f}")
        print(f"  Expectancy / trade   : ${expectancy:,.2f}")
        print(f"  Avg hold             : {avg_hold_bars:.1f} bars ({avg_hold_min:.0f} min)")
        print(f"  Exit reasons         : {reasons}")
    print("=" * 86)

    return daily_pnl


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="QQQ_5m.csv")
    p.add_argument("--swing-n", type=int, default=3)
    p.add_argument("--min-fvg-pct", type=float, default=0.0008)
    p.add_argument("--rr-partial", type=float, default=1.0)
    p.add_argument("--rr-runner", type=float, default=3.0)
    p.add_argument("--stop-buf", type=float, default=0.5)
    p.add_argument("--atr-n", type=int, default=14)
    p.add_argument("--ema-n", type=int, default=200)
    p.add_argument("--max-hold-bars", type=int, default=300)
    p.add_argument("--max-bars-after-bos", type=int, default=12)
    p.add_argument("--risk", type=float, default=0.0075)
    p.add_argument("--cost-bps", type=float, default=2.0)
    p.add_argument("--cash", type=float, default=10000.0)
    p.add_argument("--no-short", action="store_true")
    p.add_argument("--no-pd-filter", action="store_true")
    p.add_argument("--no-trend-filter", action="store_true")
    p.add_argument("--no-cooldown", action="store_true")
    p.add_argument("--no-session-confluence", action="store_true")
    p.add_argument("--flatten-eod", action="store_true",
                   help="Force-flat at session close every day.")
    p.add_argument("--no-entries-before-min", type=int, default=10 * 60,
                   help="Minute-of-day before which NO new entries are taken "
                        "(default 600 = 10:00 ET, skips the opening 30m).")
    p.add_argument("--no-new-entries-after-min", type=int, default=15 * 60 + 30,
                   help="Minute-of-day after which NO new entries are taken "
                        "(default 930 = 15:30 ET, skips the last 30m).")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--save-trades")
    args = p.parse_args()

    df = load_intraday(args.csv)
    print(f"Loaded {len(df)} RTH bars  "
          f"({df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]})  "
          f"{df['date'].nunique()} trading days")

    if args.sweep:
        grid = {
            "swing_n":    [2, 3, 5],
            "rr_runner":  [2.0, 3.0, 5.0],
            "max_bars_after_bos": [6, 12, 24],
            "min_fvg_pct": [0.0005, 0.0010, 0.0020],
        }
        rows = []
        for sn, rrr, mb, mfp in itertools.product(
                grid["swing_n"], grid["rr_runner"],
                grid["max_bars_after_bos"], grid["min_fvg_pct"]):
            trades, eq = backtest_intraday(
                df, swing_n=sn, rr_runner=rrr,
                max_bars_after_bos=mb, min_fvg_pct=mfp,
                rr_partial=args.rr_partial, stop_buf=args.stop_buf,
                atr_n=args.atr_n, ema_n=args.ema_n,
                max_hold_bars=args.max_hold_bars,
                risk_per_trade=args.risk, cost_bps=args.cost_bps,
                account=args.cash, allow_short=not args.no_short,
                use_pd_filter=not args.no_pd_filter,
                use_trend_filter=not args.no_trend_filter,
                use_cooldown=not args.no_cooldown,
                use_session_confluence=not args.no_session_confluence,
                flatten_at_session_close=args.flatten_eod,
            )
            L = len(df); bpy = bars_per_year(df)
            years = L / bpy
            final = eq[-1]
            cagr = (max(final, 1e-9) / args.cash) ** (1 / years) - 1 if years > 0 else 0.0
            bar_ret = pd.Series(eq).pct_change().fillna(0.0)
            sh = (bar_ret.mean() / bar_ret.std()) * np.sqrt(bpy) if bar_ret.std() > 0 else 0.0
            peak = np.maximum.accumulate(eq); mdd = (eq / peak - 1).min()
            pf = 0.0
            if len(trades):
                wp = trades.loc[trades["pnl_$"] > 0, "pnl_$"].sum()
                lp = -trades.loc[trades["pnl_$"] <= 0, "pnl_$"].sum()
                pf = wp / lp if lp > 0 else float("inf")
            rows.append({
                "swing_n": sn, "rr_runner": rrr, "mb_bos": mb,
                "min_fvg_pct": mfp, "trades": len(trades),
                "CAGR%": round(cagr * 100, 1),
                "Sharpe": round(sh, 2),
                "MaxDD%": round(mdd * 100, 1),
                "PF": round(pf, 2),
                "Final$": round(final, 0),
            })
        res = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
        print("\nIntraday parameter sweep (sorted by Sharpe):")
        with pd.option_context("display.max_rows", None, "display.width", 130):
            print(res.to_string(index=False))
        return

    trades, eq = backtest_intraday(
        df,
        swing_n=args.swing_n, min_fvg_pct=args.min_fvg_pct,
        rr_partial=args.rr_partial, rr_runner=args.rr_runner,
        stop_buf=args.stop_buf, atr_n=args.atr_n, ema_n=args.ema_n,
        max_hold_bars=args.max_hold_bars, max_bars_after_bos=args.max_bars_after_bos,
        risk_per_trade=args.risk, cost_bps=args.cost_bps,
        account=args.cash, allow_short=not args.no_short,
        use_pd_filter=not args.no_pd_filter,
        use_trend_filter=not args.no_trend_filter,
        use_cooldown=not args.no_cooldown,
        use_session_confluence=not args.no_session_confluence,
        flatten_at_session_close=args.flatten_eod,
        no_entries_before_min=args.no_entries_before_min,
        no_new_entries_after_min=args.no_new_entries_after_min,
    )
    label = (f"5m swing_n={args.swing_n} rr={args.rr_partial}/{args.rr_runner} "
             f"fvg>={args.min_fvg_pct*100:.2f}% risk={args.risk*100:.2f}% "
             f"bos_age<={args.max_bars_after_bos} "
             f"{'L+S' if not args.no_short else 'L-only'}"
             f"{' +trend' if not args.no_trend_filter else ''}"
             f"{' +PD' if not args.no_pd_filter else ''}"
             f"{' +sess' if not args.no_session_confluence else ''}"
             f"{' EOD-flat' if args.flatten_eod else ' overnight-OK'}")
    daily_pnl = report(df, trades, eq, args.cash, label)

    if args.save_trades and len(trades):
        trades.to_csv(args.save_trades, index=False)
        print(f"\nSaved {len(trades)} trades to {args.save_trades}")


if __name__ == "__main__":
    main()
