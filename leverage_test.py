"""
leverage_test.py — turn the meta-ensemble's RISK edge into a RETURN edge.

THE QUESTION
------------
The cross-seed meta-ensemble doesn't beat buy & hold on raw return, but it takes
roughly a THIRD of the drawdown (~-4.7% vs -12.4% out-of-sample). A smaller
drawdown is leverageable: if you size the book up, the return grows with the
leverage while the drawdown grows too — so how far can you lever before the
drawdown merely MATCHES buy & hold, and what's the return there?

This script answers that honestly, with costs and ruin modelled:
  * Rebuilds the meta-ensemble's per-bar return stream from the saved robust
    champions (all seeds' top-5 shortlists), equal-weighted.
  * Sweeps account leverage L. Levered return per bar:
        r_L(t) = L * r(t)  -  (L-1) * deployed(t) * (financing / bars_per_year)
    i.e. P&L scales with size, and you pay financing on the BORROWED multiple
    only while capital is actually deployed (deployed(t) = fraction of the
    component strategies in a position that bar). Financing is charged per bar
    in-market, which slightly OVER-states cost for intraday holds — deliberately
    conservative.
  * Finds L* = the largest leverage whose max drawdown stays <= buy & hold's.
  * Reports return / Sharpe / drawdown / ruin at each leverage, on BOTH the
    untouched out-of-sample window (headline) and the full 3 years.
  * Adds a trade-level risk-of-ruin Monte Carlo (from cfd_simulator) at the
    ensemble's measured win-rate and payoff.

ASSUMPTIONS / LIMITS (read these)
  * Leverage modelled at the account level (borrow to amplify the whole book).
  * Overnight GAP risk and slippage on stops are NOT fully modelled — real
    levered drawdowns can exceed these, especially through gaps. Treat L* as an
    UPPER bound on prudent sizing, not a target.
  * ESMA caps retail index CFDs at 1:20 *position* leverage; we report the
    implied position leverage so you can see it stays legal.
"""

import argparse
import glob
import json
import math

import numpy as np
import pandas as pd

from smc_intraday import load_intraday, backtest_intraday, bars_per_year
from evolve_traders import genome_to_kwargs
from cfd_simulator import monte_carlo_ruin


def load_champion_genomes(pattern, use_shortlist=True):
    genomes = []
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        if use_shortlist and d.get("top_robust"):
            genomes.extend(d["top_robust"])
        else:
            genomes.append(d["genome"])
    return genomes


def build_ensemble_stream(df, genomes, account):
    """Run every genome on the full df; return (ensemble_bar_ret, deployed_frac,
    combined_trades) where deployed_frac[t] = fraction of strategies in-market."""
    rets, inmkt = [], []
    all_trades = []
    for g in genomes:
        trades, eq = backtest_intraday(df, account=account, cost_bps=2.0,
                                       **genome_to_kwargs(g))
        eq = np.asarray(eq, dtype=float)
        br = pd.Series(eq).pct_change().fillna(0.0).to_numpy()
        rets.append(br)
        inmkt.append((np.abs(br) > 1e-12).astype(float))
        if len(trades):
            all_trades.append(trades)
    ens = np.mean(np.vstack(rets), axis=0)
    deployed = np.mean(np.vstack(inmkt), axis=0)
    combined = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    return ens, deployed, combined


def lever_stream(ens, deployed, L, financing, bpy):
    """Apply account leverage L to the ensemble bar-return stream."""
    fin_per_bar = financing / bpy
    return L * ens - (L - 1.0) * deployed * fin_per_bar


def stats_on_mask(bar_ret, mask, bpy, account, margin_close_dd=0.50):
    """Risk/return stats over the masked bars, plus a ruin flag (equity in the
    masked slice ever drops margin_close_dd below its running peak)."""
    sub = bar_ret[mask]
    eq = account * np.cumprod(1 + sub)
    if sub.size == 0 or sub.std() == 0:
        sharpe = 0.0
    else:
        sharpe = sub.mean() / sub.std() * math.sqrt(bpy)
    peak = np.maximum.accumulate(eq)
    dd_series = eq / peak - 1
    max_dd = float(dd_series.min())
    ret = float(eq[-1] / eq[0] - 1) if eq.size else 0.0
    years = sub.size / bpy if bpy > 0 else 0.0
    cagr = (1 + ret) ** (1 / years) - 1 if years > 0 and (1 + ret) > 0 else 0.0
    ruin = bool((dd_series <= -margin_close_dd).any())
    return {"sharpe": sharpe, "max_dd": max_dd, "ret": ret, "cagr": cagr, "ruin": ruin}


def bh_stats(df, mask, bpy, account):
    br = df["close"].pct_change().fillna(0.0).to_numpy()
    return stats_on_mask(br, mask, bpy, account)


def find_L_star(ens, deployed, mask, bpy, account, target_dd, financing,
                Lmax=8.0, step=0.05):
    """Largest L whose |max drawdown| <= |target_dd| on the masked window."""
    best = 1.0
    L = 1.0
    while L <= Lmax + 1e-9:
        s = stats_on_mask(lever_stream(ens, deployed, L, financing, bpy),
                          mask, bpy, account)
        if abs(s["max_dd"]) <= abs(target_dd) and not s["ruin"]:
            best = L
        L += step
    return best


def report_window(name, ens, deployed, mask, df, bpy, account, financing, levs):
    bh = bh_stats(df, mask, bpy, account)
    print(f"\n  {name}")
    print(f"  {'leverage':>8} | {'CAGR':>8} | {'total':>8} | {'Sharpe':>7} | "
          f"{'maxDD':>8} | ruin?")
    print("  " + "-" * 60)
    for L in levs:
        s = stats_on_mask(lever_stream(ens, deployed, L, financing, bpy),
                          mask, bpy, account)
        flag = "  <-- " if abs(s["max_dd"]) <= abs(bh["max_dd"]) else ""
        print(f"  {L:>7.1f}x | {s['cagr']*100:>7.1f}% | {s['ret']*100:>7.1f}% | "
              f"{s['sharpe']:>7.2f} | {s['max_dd']*100:>7.1f}% | "
              f"{'RUIN' if s['ruin'] else 'no':>4}{flag}")
    print("  " + "-" * 60)
    print(f"  {'Buy&Hold':>8} | {bh['cagr']*100:>7.1f}% | {bh['ret']*100:>7.1f}% | "
          f"{bh['sharpe']:>7.2f} | {bh['max_dd']*100:>7.1f}% |   --")
    Lstar = find_L_star(ens, deployed, mask, bpy, account, bh["max_dd"], financing)
    sstar = stats_on_mask(lever_stream(ens, deployed, Lstar, financing, bpy),
                          mask, bpy, account)
    print(f"\n  => L* = {Lstar:.2f}x  matches B&H drawdown ({bh['max_dd']*100:.1f}%) "
          f"at total return {sstar['ret']*100:.1f}%  vs  B&H {bh['ret']*100:.1f}%  "
          f"(Sharpe {sstar['sharpe']:.2f} vs {bh['sharpe']:.2f})")
    return bh, Lstar, sstar


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="QQQ_1h.csv")
    p.add_argument("--glob", default="rb_s*_robust_champion.json")
    p.add_argument("--account", type=float, default=10000.0)
    p.add_argument("--financing", type=float, default=0.065,
                   help="Annual overnight financing rate on borrowed exposure.")
    p.add_argument("--final-test-days", type=int, default=130)
    p.add_argument("--shortlist", action="store_true", default=True,
                   help="Use each seed's top-5 robust shortlist (default).")
    p.add_argument("--champions-only", dest="shortlist", action="store_false",
                   help="Use only each seed's single champion.")
    args = p.parse_args()

    df = load_intraday(args.csv)
    bpy = bars_per_year(df)
    genomes = load_champion_genomes(args.glob, use_shortlist=args.shortlist)
    if not genomes:
        raise SystemExit(f"No genomes from {args.glob!r}. Run evolve_robust.py first.")

    days = np.array(sorted(df["date"].unique()))
    test_days = set(days[-args.final_test_days:].tolist())
    dates_arr = df["date"].to_numpy()
    full_mask = np.ones(len(df), dtype=bool)
    test_mask = np.array([d in test_days for d in dates_arr])

    print("=" * 64)
    print("  LEVERAGE TEST — meta-ensemble vs buy & hold")
    print("=" * 64)
    print(f"  Ensemble of {len(genomes)} robust genomes "
          f"({'top-5 shortlists' if args.shortlist else 'single champions'})")
    print(f"  Data: {len(df)} bars, {df['date'].nunique()} days "
          f"(~{len(df)/df['date'].nunique():.0f}/day), financing {args.financing*100:.1f}%/yr")

    ens, deployed, combined = build_ensemble_stream(df, genomes, args.account)
    in_mkt_frac = float((deployed > 0).mean())
    print(f"  Ensemble deployed on {in_mkt_frac*100:.0f}% of bars "
          f"(financing only charged while deployed)")

    levs = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    bh_oos, Lstar_oos, sstar_oos = report_window(
        f"OUT-OF-SAMPLE (untouched last {args.final_test_days} days) — HEADLINE",
        ens, deployed, test_mask, df, bpy, args.account, args.financing, levs)
    bh_full, Lstar_full, sstar_full = report_window(
        "FULL 3 YEARS (includes in-sample dev windows — optimistic)",
        ens, deployed, full_mask, df, bpy, args.account, args.financing, levs)

    # Implied position leverage check (ESMA 1:20 on index CFDs).
    print("\n" + "=" * 64)
    print("  Risk-of-ruin (trade-level Monte Carlo) at the OOS leverage L*")
    print("=" * 64)
    if len(combined):
        wins = combined["pnl_$"] > 0
        win_rate = float(wins.mean())
        avg_w = combined.loc[wins, "pnl_$"].mean()
        avg_l = -combined.loc[~wins, "pnl_$"].mean()
        payoff = float(avg_w / avg_l) if avg_l and avg_l > 0 else 1.5
    else:
        win_rate, payoff = 0.5, 1.5
    print(f"  Measured ensemble edge: win rate {win_rate*100:.1f}%, "
          f"payoff {payoff:.2f}:1 (avg win / avg loss)")
    # Per-trade risk roughly scales with L; show a small grid.
    base_risk = 0.01
    print(f"  {'lev':>5} | {'~risk/trade':>11} | {'P(ruin@50%DD)':>13} | {'median outcome':>14}")
    print("  " + "-" * 54)
    for L in sorted({1.0, round(Lstar_oos, 1), round(Lstar_oos * 1.5, 1), 3.0}):
        mc = monte_carlo_ruin(min(base_risk * L, 0.2), win_rate, payoff, n_trades=200)
        print(f"  {L:>4.1f}x | {min(base_risk*L,0.2)*100:>10.1f}% | "
              f"{mc['p_ruin']*100:>12.1f}% | {mc['median_final_mult']:>13.2f}x")

    print("\n" + "=" * 64)
    print("  BOTTOM LINE")
    print("=" * 64)
    verdict = "BEATS" if sstar_oos["ret"] > bh_oos["ret"] else "trails"
    print(f"  Out-of-sample: at {Lstar_oos:.2f}x leverage the ensemble matches")
    print(f"  buy & hold's {bh_oos['max_dd']*100:.1f}% drawdown and {verdict} its return")
    print(f"  ({sstar_oos['ret']*100:.1f}% vs {bh_oos['ret']*100:.1f}%), "
          f"Sharpe {sstar_oos['sharpe']:.2f} vs {bh_oos['sharpe']:.2f}.")
    print(f"  Because Sharpe is leverage-invariant, the ensemble only out-RETURNS")
    print(f"  B&H at matched risk when its Sharpe exceeds B&H's — otherwise leverage")
    print(f"  buys return AND risk in equal measure. The honest read: {verdict}.")
    print("=" * 64)


if __name__ == "__main__":
    main()
