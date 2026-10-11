"""
smc_v3.py — the rebuilt strategy, from everything the research found.

THE SHORT VERSION
  The edge that survived every honest test is not the SMC entry timing — it is
  being in the market when the trend is up and in T-bills when it is not, sized
  to a steady level of risk. v3 is built around that, keeps the (now honest)
  SMC engine as an optional sleeve, and uses SMC structure as live context.

WHAT THE RESEARCH FOUND (and how v3 responds)
  1. LOOK-AHEAD BUG in v1/v2 and the intraday engine: entries were decided with
     bar t's own close and required bar t's low to have held the FVG zone —
     knowledge you only have after the bar. It deleted the losing fills and
     roughly doubled the Sharpe (0.88 -> 0.38 for the same settings, win rate
     80% -> 58%). Every downstream result (GA champions, ensembles, the leverage
     study, the live tracker) inherited it.
     -> v3 decides at the close of bar t and acts at the open of bar t+1.
        Stops gapped through fill at the open. Fill-then-stop is a loss.
  2. Across 72 reasonable settings the honest FVG-retest strategy averages a
     Sharpe of ~0.1. Its one durable property was sitting out crashes.
     -> The SMC sleeve is kept (honest engine) but weighted 0 by default.
  3. Crash protection comes from the trend REGIME, not the entry pattern.
     -> Core = ensemble of 8 standard trend rules (SMA/EMA 100/150/200/250).
        Exposure = share of rules that are bullish. No parameter is "picked".
  4. Volatility targeting smooths the ride without leverage.
     -> Exposure scaled toward 20% annualised vol, capped at 1.0x by default.
  5. Leverage at CFD financing (~6.5%/yr) lowered Sharpe in every test.
     -> Default cap 1.0x: tradable with a plain ETF, no margin, no CFD.
  6. Idle cash is real money: it earns the 13-week T-bill rate (^IRX).

COMMANDS
  python smc_v3.py report     # full honest validation (default)
  python smc_v3.py live       # today's target exposure + SMC structure context
  python smc_v3.py track      # update the forward paper account (used by CI)
  python smc_v3.py smc        # the honest SMC sleeve on its own, with trades

Backtests simulate past data; they are not predictions. Research/education
only — not financial advice.
"""

import argparse
import itertools
import json
import math
import os
import sys
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


# =============================================================================
# Data
# =============================================================================
def _normalize(raw):
    if hasattr(raw.columns, "nlevels") and raw.columns.nlevels > 1:
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    cols = {c.lower(): c for c in raw.columns}
    return pd.DataFrame({
        "date": pd.to_datetime(raw[cols["date"]]),
        "open": raw[cols["open"]].astype(float), "high": raw[cols["high"]].astype(float),
        "low": raw[cols["low"]].astype(float), "close": raw[cols["close"]].astype(float),
    }).dropna().sort_values("date").reset_index(drop=True)


def load_daily(path="NDX.csv"):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if any(k not in cols for k in ("date", "open", "high", "low", "close")):
        sys.exit(f"ERROR: {path} needs Date/Open/High/Low/Close columns")
    return _normalize(df.set_index(cols["date"]))


def fetch_daily(ticker="^NDX", start="2006-05-22"):
    import yfinance as yf
    raw = yf.download(ticker, start=start, auto_adjust=False, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"no data for {ticker}")
    return _normalize(raw)


def get_prices(csv, ticker, online):
    if online:
        try:
            return fetch_daily(ticker), f"yfinance {ticker}"
        except Exception as e:
            note = f" (download failed: {e})"
    else:
        note = ""
    return load_daily(csv), f"local {csv}{note}"


def cash_rate_for(df, path="IRX.csv", online=False):
    """Daily T-bill rate (decimal/yr) aligned to df dates; zeros if unavailable."""
    irx = None
    if online:
        try:
            irx = fetch_daily("^IRX")
        except Exception:
            irx = None
    if irx is None and os.path.exists(path):
        irx = load_daily(path)
    if irx is None:
        return np.zeros(len(df))
    s = (irx.set_index("date")["close"] / 100.0).reindex(df["date"]).ffill().bfill()
    return s.fillna(0.0).clip(lower=0.0).to_numpy()


# =============================================================================
# Regime core — trend-rule ensemble + volatility targeting
# All signals use data up to the close of bar t; positions change at open t+1.
# =============================================================================
@dataclass(frozen=True)
class CoreConfig:
    ma_lengths: tuple = (100, 150, 200, 250)
    ma_types: tuple = ("sma", "ema")
    vol_target: float = 0.20          # annualised; None/0 disables scaling
    vol_window: int = 20
    max_expo: float = 1.0             # 1.0 = no leverage
    cost_bps: float = 5.0             # per unit of exposure traded
    fin_rate: float = 0.065           # yearly cost of exposure above 1.0x (CFD-like)
    fin_over_rf: float = None         # if set: financing = T-bill + this (futures-like)
    cash_yield: bool = True           # idle cash earns the T-bill rate
    smc_weight: float = 0.0           # blend in the SMC sleeve (0 = off)


def trend_votes(c, cfg):
    """Per-bar fraction of trend rules that are bullish (close above its MA)."""
    s = pd.Series(c)
    votes = []
    for n in cfg.ma_lengths:
        for kind in cfg.ma_types:
            ma = (s.rolling(n, min_periods=n).mean() if kind == "sma"
                  else s.ewm(span=n, adjust=False).mean())
            ma = ma.to_numpy()
            v = (c > ma).astype(float)
            v[: n] = 0.0                              # no vote before warm-up
            votes.append(v)
    return np.mean(votes, axis=0), len(votes)


def realized_vol(c, n):
    r = np.concatenate([[0.0], np.diff(c) / c[:-1]])
    return pd.Series(r).rolling(n, min_periods=5).std().to_numpy() * math.sqrt(252)


def core_target(df, cfg):
    """Target exposure decided at the close of each bar."""
    c = df["close"].to_numpy(float)
    votes, n_rules = trend_votes(c, cfg)
    if cfg.vol_target:
        rv = realized_vol(c, cfg.vol_window)
        scale = np.nan_to_num(cfg.vol_target / np.where(rv > 0, rv, np.nan), nan=0.0)
    else:
        scale = np.ones(len(c))
    target = np.clip(votes * scale, 0.0, cfg.max_expo)
    return target, votes, scale, n_rules


def simulate_exposure(df, target, rf, cfg):
    """Daily returns when target[t] (known at close t) is held from open t+1.
    The overnight gap is earned at the OLD exposure; trading costs apply to the
    change; idle cash earns T-bills; exposure above 1x pays financing."""
    o = df["open"].to_numpy(float)
    c = df["close"].to_numpy(float)
    L = len(c)
    cost = cfg.cost_bps / 1e4
    r = np.zeros(L)
    held = 0.0
    for t in range(1, L):
        new = target[t - 1]
        day = held * (o[t] / c[t - 1] - 1) + new * (c[t] / o[t] - 1)
        day -= cost * abs(new - held)
        idle = 1.0 - new
        if idle > 0 and cfg.cash_yield:
            day += idle * rf[t] / 252
        elif idle < 0:
            fin = (rf[t] + cfg.fin_over_rf) if cfg.fin_over_rf is not None else cfg.fin_rate
            day += idle * fin / 252
        r[t] = day
        held = new
    return r


def strategy_returns(df, rf, core=CoreConfig(), smc_cfgs=None):
    target, votes, scale, _ = core_target(df, core)
    r = simulate_exposure(df, target, rf, core)
    if core.smc_weight > 0:
        smc_r, _, _ = run_ensemble(
            df, smc_cfgs or [SmcConfig(min_fvg_pct=0.002, cash_yield=core.cash_yield)], rf)
        r = (1 - core.smc_weight) * r + core.smc_weight * smc_r
    return r, target


# =============================================================================
# SMC — market structure (swings, BOS, FVGs), all known at the close of bar t
# =============================================================================
def atr_at_close(h, l, c, n):
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    return pd.Series(tr).rolling(n, min_periods=1).mean().to_numpy()


def confirmed_swings(h, l, n):
    L = len(h)
    sh, sl = np.full(L, np.nan), np.full(L, np.nan)
    last_h = last_l = np.nan
    for t in range(L):
        i = t - n
        if i >= n:
            if all(h[i] > h[i - k] and h[i] > h[i + k] for k in range(1, n + 1)):
                last_h = h[i]
            if all(l[i] < l[i - k] and l[i] < l[i + k] for k in range(1, n + 1)):
                last_l = l[i]
        sh[t], sl[t] = last_h, last_l
    return sh, sl


def structure(c, sh, sl):
    L = len(c)
    trend, bos_bar = np.zeros(L, dtype=int), np.full(L, -1, dtype=int)
    state, last_bos, used_h, used_l = 0, -1, np.nan, np.nan
    for t in range(L):
        if not np.isnan(sh[t]) and c[t] > sh[t] and sh[t] != used_h:
            state, last_bos, used_h = 1, t, sh[t]
        if not np.isnan(sl[t]) and c[t] < sl[t] and sl[t] != used_l:
            state, last_bos, used_l = -1, t, sl[t]
        trend[t], bos_bar[t] = state, last_bos
    return trend, bos_bar


_IND = {}


def smc_indicators(df, swing_n, atr_n=14, ema_n=200):
    key = (id(df), len(df), swing_n, atr_n, ema_n)
    if key not in _IND:
        o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        sh, sl = confirmed_swings(h, l, swing_n)
        trend, bos_bar = structure(c, sh, sl)
        gap = np.full(len(c), np.nan)
        gap[2:] = (l[2:] - h[:-2]) / c[2:]
        _IND[key] = dict(o=o, h=h, l=l, c=c, sh=sh, sl=sl, trend=trend,
                         bos_bar=bos_bar, atr=atr_at_close(h, l, c, atr_n),
                         ema=pd.Series(c).ewm(span=ema_n, adjust=False).mean().to_numpy(),
                         gap=gap)
    return _IND[key]


# =============================================================================
# SMC sleeve — honest FVG-retest engine (optional; weight 0 by default)
# =============================================================================
@dataclass(frozen=True)
class SmcConfig:
    swing_n: int = 3
    min_fvg_pct: float = 0.001
    stop_buf: float = 0.5
    rr_partial: float = 1.0
    rr_runner: float = 5.0
    max_hold: int = 30
    max_bars_after_bos: int = 20
    ema_n: int = 200
    atr_n: int = 14
    use_pd_filter: bool = True
    use_cooldown: bool = True
    regime_exit: bool = False
    risk: float = 0.01
    max_lev: float = 2.0
    cost_bps: float = 5.0
    slip_bps: float = 2.0
    fin_rate: float = 0.065
    cash_yield: bool = False


SMC_GRID = dict(swing_n=[2, 3, 5], min_fvg_pct=[0.0005, 0.001, 0.002],
                rr_runner=[3.0, 5.0], max_hold=[30, 60], regime_exit=[False, True])


def smc_grid(base=SmcConfig()):
    keys = list(SMC_GRID)
    return [replace(base, **dict(zip(keys, v)))
            for v in itertools.product(*(SMC_GRID[k] for k in keys))]


def run_smc(df, cfg, rf=None, return_state=False):
    """Decide at close t, rest a limit order for bar t+1; gap-aware exits."""
    ind = smc_indicators(df, cfg.swing_n, cfg.atr_n, cfg.ema_n)
    o, h, l, c = ind["o"], ind["h"], ind["l"], ind["c"]
    sh, sl, trend, bos_bar = ind["sh"], ind["sl"], ind["trend"], ind["bos_bar"]
    atr, ema, gap = ind["atr"], ind["ema"], ind["gap"]
    L, dates = len(c), df["date"].to_numpy()
    rf = rf if (rf is not None and cfg.cash_yield) else np.zeros(L)
    warm = max(cfg.ema_n, cfg.atr_n, 2 * cfg.swing_n + 2)
    cost, slip = cfg.cost_bps / 1e4, cfg.slip_bps / 1e4

    cash, eq = 10000.0, np.empty(L)
    sh_qty = entry = stop = target = partial_px = R = init_qty = 0.0
    partial_done, entry_bar, trade_pnl, entry_cost = False, -1, 0.0, 0.0
    exit_next_open, pending, fvgs = None, None, []
    cooldown, loss_bos, trades = False, -2, []

    def sell(qty, px, t, reason, final):
        nonlocal cash, sh_qty, trade_pnl, cooldown, loss_bos
        cash += qty * px * (1 - cost)
        trade_pnl += qty * (px - entry) - qty * px * cost
        sh_qty -= qty
        if final or sh_qty <= 1e-12:
            sh_qty = 0.0
            pnl = trade_pnl - entry_cost
            trades.append({
                "entry_date": pd.Timestamp(dates[entry_bar]).date(),
                "exit_date": pd.Timestamp(dates[t]).date(),
                "entry": round(entry, 2), "exit": round(px, 2),
                "bars": t - entry_bar, "partial": partial_done,
                "pnl_$": round(pnl, 2),
                "r_mult": round(pnl / (init_qty * R), 2) if init_qty * R > 0 else 0.0,
                "reason": reason})
            if cfg.use_cooldown and pnl <= 0:
                cooldown, loss_bos = True, bos_bar[t]

    for t in range(L):
        # --- execute at bar t ---
        if sh_qty > 0 and t > entry_bar:
            if exit_next_open:
                sell(sh_qty, o[t], t, exit_next_open, True)
            elif o[t] <= stop:
                sell(sh_qty, o[t] * (1 - slip), t, "stop-gap", True)
            else:
                if not partial_done and o[t] >= partial_px:
                    sell(sh_qty * 0.5, o[t], t, "partial", False)
                    partial_done = True
                if o[t] >= target:
                    sell(sh_qty, o[t], t, "target-gap", True)
                elif l[t] <= stop:
                    sell(sh_qty, stop * (1 - slip), t, "stop", True)
                else:
                    if not partial_done and h[t] >= partial_px:
                        sell(sh_qty * 0.5, partial_px, t, "partial", False)
                        partial_done = True
                    if h[t] >= target:
                        sell(sh_qty, target, t, "target", True)
        elif sh_qty == 0 and pending is not None:
            top, stp, qty = pending["top"], pending["stop"], pending["qty"]
            fill = None if o[t] <= stp else (o[t] if o[t] <= top else (top if l[t] <= top else None))
            if fill is not None:
                pending["fvg"][3] = True
                entry, stop, entry_bar = fill, stp, t
                R = entry - stop
                partial_px, target = entry + cfg.rr_partial * R, entry + cfg.rr_runner * R
                sh_qty = init_qty = qty
                partial_done, trade_pnl, entry_cost = False, 0.0, qty * entry * cost
                cash -= qty * entry * (1 + cost)
                if l[t] <= stop:
                    sell(sh_qty, stop * (1 - slip), t, "stop-same-bar", True)
        pending, exit_next_open = None, None

        # --- close of bar t: update structure ---
        if t >= 2 and gap[t] >= cfg.min_fvg_pct:
            fvgs.append([l[t], h[t - 2], t, False])          # top, bottom, formed, used
        fvgs = [f for f in fvgs if not f[3] and c[t] >= f[1]]
        if sh_qty > 0:
            if partial_done:
                stop = max(stop, entry, sl[t] if not np.isnan(sl[t]) else stop)
            if t - entry_bar >= cfg.max_hold:
                exit_next_open = "timeout"
            elif cfg.regime_exit and c[t] < ema[t]:
                exit_next_open = "regime"

        # --- close of bar t: tomorrow's order ---
        if cooldown and trend[t] == 1 and bos_bar[t] != loss_bos:
            cooldown = False
        if (sh_qty == 0 and t >= warm and (t + 1 < L or return_state)
                and trend[t] == 1 and c[t] > ema[t] and not cooldown
                and t - bos_bar[t] <= cfg.max_bars_after_bos):
            mid = (sh[t] + sl[t]) / 2
            for f in reversed(fvgs):
                top, bot = f[0], f[1]
                if top >= c[t] or (cfg.use_pd_filter and not (sh[t] > sl[t] and top <= mid)):
                    continue
                stp = bot - cfg.stop_buf * atr[t]
                if top - stp <= 0:
                    continue
                eq_now = cash + sh_qty * c[t]
                qty = min(eq_now * cfg.risk / (top - stp), eq_now * cfg.max_lev / top)
                pending = {"top": top, "stop": stp, "qty": qty, "fvg": f}
                break

        # --- carry + mark ---
        cash += cash * (rf[t] if cash > 0 else cfg.fin_rate) / 252 if cash != 0 else 0.0
        eq[t] = cash + sh_qty * c[t]

    tr = pd.DataFrame(trades)
    if not return_state:
        return tr, eq
    return tr, eq, {"in_position": sh_qty > 0,
                    "pending": None if pending is None else
                    {"limit": pending["top"], "stop": pending["stop"]}}


def run_ensemble(df, cfgs, rf=None):
    rets, trs = [], []
    for cfg in cfgs:
        tr, eq = run_smc(df, cfg, rf)
        rets.append(np.concatenate([[0.0], np.diff(eq) / eq[:-1]]))
        if len(tr):
            trs.append(tr)
    rets = np.vstack(rets)
    return rets.mean(axis=0), rets, (pd.concat(trs, ignore_index=True) if trs else pd.DataFrame())


# =============================================================================
# Metrics
# =============================================================================
def stats(r, mask=None):
    r = np.asarray(r, float)
    if mask is not None:
        r = r[mask]
    if r.size < 2:
        return dict(total=0.0, cagr=0.0, sharpe=0.0, sortino=0.0, maxdd=0.0)
    eq = np.cumprod(1 + r)
    years = r.size / 252
    sd, dn = r.std(), r[r < 0].std()
    return dict(total=eq[-1] - 1,
                cagr=eq[-1] ** (1 / years) - 1 if eq[-1] > 0 else -1.0,
                sharpe=r.mean() / sd * math.sqrt(252) if sd > 0 else 0.0,
                sortino=r.mean() / dn * math.sqrt(252) if dn > 0 else 0.0,
                maxdd=float((eq / np.maximum.accumulate(eq) - 1).min()))


def trade_stats(tr):
    if tr is None or not len(tr):
        return dict(n=0, win=0.0, pf=0.0, avg_r=0.0)
    w = tr["pnl_$"] > 0
    gl = -tr.loc[~w, "pnl_$"].sum()
    rcol = "r_mult" if "r_mult" in tr else "r_total"
    return dict(n=len(tr), win=w.mean(),
                pf=tr.loc[w, "pnl_$"].sum() / gl if gl > 0 else float("inf"),
                avg_r=tr[rcol].mean())


HEADER = (f"  {'':<38} {'total':>9} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8}")


def row(name, s, note=""):
    return (f"  {name:<38} {s['total']*100:>8.0f}% {s['cagr']*100:>6.1f}% "
            f"{s['sharpe']:>7.2f} {s['sortino']:>8.2f} {s['maxdd']*100:>7.1f}%  {note}")


def bh_returns(df):
    c = df["close"].to_numpy(float)
    return np.concatenate([[0.0], np.diff(c) / c[:-1]])


# =============================================================================
# Commands
# =============================================================================
def cmd_report(args):
    df = load_daily(args.csv)
    rf = cash_rate_for(df)
    dates = pd.to_datetime(df["date"])
    core = CoreConfig()
    warm = max(core.ma_lengths) + 10
    m = np.arange(len(df)) >= warm
    live = dates.dt.date.to_numpy() >= pd.Timestamp("2026-06-03").date()
    bh = bh_returns(df)
    v3, target = strategy_returns(df, rf, core)

    print("=" * 100)
    print(f"  SMC v3 — honest rebuild | ^NDX daily {dates.iloc[warm].date()} -> "
          f"{dates.iloc[-1].date()} | decide at close, trade next open, 5 bps costs")
    print("=" * 100)

    # 1) What changed
    from smc_strategy_v2 import backtest_v2
    _, v2_eq = backtest_v2(df, swing_n=3, rr_runner=5.0, max_bars_after_bos=20,
                           min_fvg_pct=0.001, risk_per_trade=0.01, account=10000.0,
                           allow_short=False, use_trend_filter=True)
    v2 = np.concatenate([[0.0], np.diff(v2_eq) / v2_eq[:-1]])
    smc_r, smc_sleeves, smc_tr = run_ensemble(df, smc_grid())
    smc_cash_r, _, _ = run_ensemble(df, smc_grid(SmcConfig(cash_yield=True)), rf)
    sleeve_sh = [stats(x, m)["sharpe"] for x in smc_sleeves]
    print("\n  1) EVERY GENERATION, MEASURED THE SAME WAY")
    print(HEADER)
    print(row("Buy & hold ^NDX (price only)", stats(bh, m)))
    print(row("v2 SMC long-only (HAS look-ahead)", stats(v2, m), "<- old headline, not real"))
    print(row("SMC honest, 72-setting average", stats(smc_r, m),
              f"sleeve Sharpe {min(sleeve_sh):.2f}..{max(sleeve_sh):.2f}"))
    print(row("  same, + T-bill interest on cash", stats(smc_cash_r, m), "mostly the T-bills"))
    v3_nocash = strategy_returns(df, rf, replace(core, cash_yield=False))[0]
    print(row("v3 regime core, no cash interest", stats(v3_nocash, m)))
    print(row("v3 REGIME CORE (recommended)", stats(v3, m),
              f"avg exposure {target[m].mean()*100:.0f}%"))

    # 2) Robustness of the core
    print("\n  2) ROBUSTNESS — nearby settings and harsher assumptions")
    print(HEADER)
    variants = [
        ("SMA/EMA 150-200 only", replace(core, ma_lengths=(150, 200))),
        ("SMA/EMA 50-300 (6 lengths)", replace(core, ma_lengths=(50, 100, 150, 200, 250, 300))),
        ("vol target 15%", replace(core, vol_target=0.15)),
        ("vol target 25%", replace(core, vol_target=0.25)),
        ("no vol targeting", replace(core, vol_target=0)),
        ("costs doubled (10 bps)", replace(core, cost_bps=10.0)),
        ("no T-bill interest on cash", replace(core, cash_yield=False)),
        ("with 20% SMC sleeve", replace(core, smc_weight=0.2)),
    ]
    for name, cfg in variants:
        print(row(name, stats(strategy_returns(df, rf, cfg)[0], m)))

    # 3) Leverage — depends entirely on the financing you can get
    print("\n  3) LEVERAGE (vol target 25%) — only worth it with cheap financing")
    print(HEADER)
    for cap in (1.0, 1.5, 2.0):
        cfd = replace(core, vol_target=0.25, max_expo=cap)
        fut = replace(cfd, fin_over_rf=0.005)
        print(row(f"cap {cap:.1f}x, CFD financing 6.5%/yr", stats(strategy_returns(df, rf, cfd)[0], m)))
        if cap > 1:
            print(row(f"cap {cap:.1f}x, futures-like T-bill+0.5%", stats(strategy_returns(df, rf, fut)[0], m)))

    # 4) Crashes and calendar years
    from regime_test import detect_drawdown_episodes
    c = df["close"].to_numpy(float)
    eq_v3, eq_bh = np.cumprod(1 + v3), np.cumprod(1 + bh)
    print("\n  4) EVERY REAL >=15% NASDAQ DECLINE (peak -> trough)")
    worse = []
    for ep in detect_drawdown_episodes(c, 0.15):
        i0, i1 = max(ep["peak"], warm), ep["trough"]
        if i1 - i0 < 5:
            continue
        b, s = eq_bh[i1] / eq_bh[i0] - 1, eq_v3[i1] / eq_v3[i0] - 1
        worse.append((b, s))
        print(f"     {dates.iloc[i0].date()} -> {dates.iloc[i1].date()}   "
              f"B&H {b*100:6.1f}%   v3 {s*100:6.1f}%")
    print(f"     average                    B&H {np.mean([w[0] for w in worse])*100:6.1f}%   "
          f"v3 {np.mean([w[1] for w in worse])*100:6.1f}%")
    yr = dates.dt.year.to_numpy()
    print("\n     year:   " + " ".join(f"{y % 100:>5}" for y in range(dates.iloc[warm].year + 1, yr.max() + 1)))
    for name, r in (("B&H", bh), ("v3", v3)):
        cells = [f"{(np.prod(1 + r[yr == y]) - 1) * 100:5.0f}"
                 for y in range(dates.iloc[warm].year + 1, yr.max() + 1)]
        print(f"     {name:<6}  " + " ".join(cells))

    # 5) Live forward window
    if live.sum() > 5:
        print(f"\n  5) LIVE WINDOW 2026-06-03 -> {dates.iloc[-1].date()} "
              f"({live.sum()} days the research never touched)")
        print(HEADER)
        print(row("Buy & hold", stats(bh, live)))
        print(row("v2 tracker logic (look-ahead)", stats(v2, live)))
        print(row("SMC honest, 72-setting average", stats(smc_r, live)))
        print(row("v3 regime core", stats(v3, live)))
    print("=" * 100)
    print("  Simulations of past data — not predictions or guarantees. Not financial advice.")
    print("=" * 100)


def live_snapshot(df, rf, core):
    target, votes, scale, n_rules = core_target(df, core)
    c = df["close"].to_numpy(float)
    ind = smc_indicators(df, 3)
    t = len(df) - 1
    fv = []
    for i in range(max(2, t - 120), t + 1):                # recent unfilled bullish FVGs
        if ind["gap"][i] >= 0.001 and ind["l"][i] < c[t]:
            bot = ind["h"][i - 2]
            if (c[i:t + 1] >= bot).all():
                fv.append((pd.Timestamp(df["date"].iloc[i]).date(), bot, ind["l"][i]))
    return dict(date=str(pd.Timestamp(df["date"].iloc[t]).date()), close=c[t],
                target=float(target[t]), prev=float(target[t - 1]),
                votes=float(votes[t]), n_rules=n_rules, scale=float(scale[t]),
                rv=float(realized_vol(c, core.vol_window)[t]),
                bos="bullish" if ind["trend"][t] == 1 else ("bearish" if ind["trend"][t] == -1 else "none"),
                swing_high=float(ind["sh"][t]), swing_low=float(ind["sl"][t]),
                fvgs=fv[-3:])


def cmd_live(args):
    df, src = get_prices(args.csv, args.ticker, online=True)
    rf = cash_rate_for(df, online=True)
    s = live_snapshot(df, rf, CoreConfig())
    change = s["target"] - s["prev"]
    print("=" * 74)
    print(f"  SMC v3 LIVE | {src}")
    print(f"  Latest close {s['date']}: {s['close']:,.2f}")
    print("=" * 74)
    print(f"  Trend rules bullish : {round(s['votes']*s['n_rules'])}/{s['n_rules']}")
    print(f"  Realized vol (20d)  : {s['rv']*100:.1f}%  -> vol scale {min(s['scale'], 9.99):.2f}")
    print(f"  TARGET EXPOSURE     : {s['target']*100:.0f}% of equity long, rest in T-bills")
    act = ("hold" if abs(change) < 0.005 else
           f"{'BUY' if change > 0 else 'SELL'} {abs(change)*100:.0f}% of equity at next open")
    print(f"  Action              : {act}  (was {s['prev']*100:.0f}%)")
    print("  " + "-" * 70)
    print(f"  SMC context: structure {s['bos']}; last swing high {s['swing_high']:,.2f}, "
          f"swing low {s['swing_low']:,.2f}")
    if s["fvgs"]:
        for d, bot, top in s["fvgs"]:
            print(f"    unfilled bullish FVG from {d}: {bot:,.2f} - {top:,.2f}")
    print("=" * 74)
    print("  Paper/research signal only — not financial advice.")


def cmd_track(args):
    """Forward paper account for v3 (deterministic: re-derived from data)."""
    state_file, log_file = "paper_state_v3.json", "PAPER_LOG_V3.md"
    df, src = get_prices(args.csv, args.ticker, online=not args.offline)
    rf = cash_rate_for(df, online=not args.offline)
    core = CoreConfig()
    r, target = strategy_returns(df, rf, core)
    bh = bh_returns(df)
    state = json.load(open(state_file)) if os.path.exists(state_file) else None
    if state is None or args.reset:
        state = {"ticker": args.ticker, "go_live_date": str(pd.Timestamp.today().date()),
                 "strategy": "v3 regime core: 8 trend rules + 20% vol target, cap 1.0x",
                 "journal": []}
    go = pd.Timestamp(state["go_live_date"]).date()
    d = pd.to_datetime(df["date"]).dt.date.to_numpy()
    mask = d >= go
    equity = 10000.0 * float(np.prod(1 + r[mask])) if mask.any() else 10000.0
    bh_eq = 10000.0 * float(np.prod(1 + bh[mask])) if mask.any() else 10000.0
    s = live_snapshot(df, rf, core)
    change = s["target"] - s["prev"]
    action = ("hold" if abs(change) < 0.005 else
              f"{'buy' if change > 0 else 'sell'} {abs(change)*100:.0f}% at next open")
    today = str(pd.Timestamp.today().date())
    state["journal"] = [j for j in state["journal"] if j["date"] != today] + [{
        "date": today, "last_bar": s["date"], "exposure_next": round(s["target"], 3),
        "rules_up": f"{round(s['votes']*s['n_rules'])}/{s['n_rules']}",
        "action": action, "paper_equity": round(equity, 2), "buyhold_equity": round(bh_eq, 2)}]
    json.dump(state, open(state_file, "w"), indent=2)
    lines = [f"# v3 paper journal — {state['ticker']}", "",
             f"- **Strategy:** {state['strategy']}",
             f"- **Go-live:** {state['go_live_date']}  (decide at close, act at next open)",
             f"- **Latest:** {s['date']} close {s['close']:,.2f} — paper ${equity:,.2f} "
             f"vs buy & hold ${bh_eq:,.2f}", "",
             "| Date | Last bar | Rules up | Exposure next | Action | Paper equity | Buy & hold |",
             "|---|---|---|---|---|---|---|"]
    for j in state["journal"]:
        lines.append(f"| {j['date']} | {j['last_bar']} | {j['rules_up']} | "
                     f"{j['exposure_next']*100:.0f}% | {j['action']} | "
                     f"${j['paper_equity']:,.2f} | ${j['buyhold_equity']:,.2f} |")
    lines += ["", "_Paper trading only — not financial advice._"]
    open(log_file, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"[{src}] {s['date']}: next exposure {s['target']*100:.0f}% ({action}); "
          f"paper ${equity:,.2f} vs B&H ${bh_eq:,.2f} since {state['go_live_date']}")


def cmd_smc(args):
    df = load_daily(args.csv)
    tr, eq = run_smc(df, SmcConfig())
    r = np.concatenate([[0.0], np.diff(eq) / eq[:-1]])
    s, ts = stats(r, np.arange(len(df)) >= 200), trade_stats(tr)
    print(HEADER)
    print(row("SMC sleeve, center setting", s,
              f"{ts['n']} trades, win {ts['win']*100:.0f}%, PF {ts['pf']:.2f}, avg {ts['avg_r']:+.2f}R"))
    if len(tr):
        print("\n  exits:", tr["reason"].value_counts().to_dict())
        print(tr.tail(10).to_string(index=False))


def main():
    p = argparse.ArgumentParser(description="SMC v3 — honest rebuilt strategy")
    p.add_argument("command", nargs="?", default="report",
                   choices=["report", "live", "track", "smc"])
    p.add_argument("--csv", default="NDX.csv")
    p.add_argument("--ticker", default="^NDX")
    p.add_argument("--offline", action="store_true", help="track: use local CSVs only")
    p.add_argument("--reset", action="store_true", help="track: restart paper account today")
    args = p.parse_args()
    {"report": cmd_report, "live": cmd_live, "track": cmd_track, "smc": cmd_smc}[args.command](args)


if __name__ == "__main__":
    main()
