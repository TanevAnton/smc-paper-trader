"""
ensemble.py — trade ALL evolved champions together as one diversified book.

WHY AN ENSEMBLE
---------------
Each champion from evolve_traders.py is the best genome of its random seed — and
each is partly overfit in its own way. Picking one means betting on one set of
quirks. Running all of them with equal capital diversifies that risk: they enter
at different times, in different directions, with different hold lengths, so
their individual errors partly cancel. When the strategies aren't perfectly
correlated, the combined equity curve is smoother than any single one — higher
Sharpe and (the real prize) lower drawdown for the same style of edge.

HOW IT'S COMBINED
-----------------
Equal-weight, continuously-rebalanced portfolio:
    * Give each of the N champions 1/N of the capital.
    * Each bar, the ensemble return is the simple average of the N champions'
      bar returns. (This is the standard "equal-weight portfolio of strategies"
      and implicitly rebalances each bar back to equal weight.)
No further optimization of the weights — that would just add another layer of
overfitting. Equal weight is the honest default.

WHAT IT REPORTS
---------------
    * Each champion standalone vs the ensemble vs buy & hold,
      on the TRAIN and the untouched HOLDOUT window.
    * The daily-return CORRELATION matrix of the champions — the thing that
      actually decides whether diversification helps.
    * A one-line verdict: ensemble Sharpe/▽DD vs the AVERAGE individual, so the
      diversification benefit is explicit and not cherry-picked.

USAGE
-----
    python ensemble.py
    python ensemble.py --csv QQQ_1h.csv --champs h1_seed1_champion_genome.json,...
    python ensemble.py --train-frac 0.7 --account 10000
"""

import argparse
import glob
import json
import math

import numpy as np
import pandas as pd

from smc_intraday import load_intraday, backtest_intraday, bars_per_year
from evolve_traders import genome_to_kwargs


# ----------------------------------------------------------------------------
# Metrics on a return series over a day-mask
# ----------------------------------------------------------------------------
def slice_stats(bar_ret, full_eq, day_mask, bpy):
    """Risk stats for the bars selected by day_mask.
    bar_ret  : per-bar returns of the strategy (np.array)
    full_eq  : the full equity curve (np.array), used for slice drawdown/return
    """
    sel = bar_ret[day_mask]
    if sel.size == 0 or sel.std() == 0:
        sharpe = sortino = 0.0
    else:
        sharpe = sel.mean() / sel.std() * math.sqrt(bpy)
        dn = sel[sel < 0].std()
        sortino = sel.mean() / dn * math.sqrt(bpy) if dn > 0 else 0.0
    eq_slice = full_eq[day_mask]
    if eq_slice.size:
        peak = np.maximum.accumulate(eq_slice)
        max_dd = float((eq_slice / peak - 1).min())
        ret = float(eq_slice[-1] / eq_slice[0] - 1) if eq_slice[0] > 0 else 0.0
    else:
        max_dd = ret = 0.0
    # Annualized return over the slice.
    n_bars = int(day_mask.sum())
    years = n_bars / bpy if bpy > 0 else 0.0
    cagr = (1 + ret) ** (1 / years) - 1 if years > 0 and (1 + ret) > 0 else 0.0
    return {"sharpe": sharpe, "sortino": sortino, "max_dd": max_dd,
            "ret": ret, "cagr": cagr}


def run_champion(df, genome, account):
    """Run one champion; return its per-bar return series and equity curve."""
    trades, eq = backtest_intraday(df, account=account, cost_bps=2.0,
                                   **genome_to_kwargs(genome))
    eq = np.asarray(eq, dtype=float)
    bar_ret = pd.Series(eq).pct_change().fillna(0.0).to_numpy()
    return bar_ret, eq, trades


def daily_returns(df, bar_ret):
    """Aggregate per-bar returns into per-DAY returns (for honest correlation —
    bar-level is mostly shared zeros while flat, which biases correlation)."""
    s = pd.Series(bar_ret, index=pd.to_datetime(df["datetime"]).dt.date.values)
    # Compound within each day.
    return s.groupby(s.index).apply(lambda x: (1 + x).prod() - 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="QQQ_1h.csv")
    p.add_argument("--champs", default="",
                   help="Comma-separated champion JSON paths. "
                        "Default: all h1_seed*_champion_genome.json")
    p.add_argument("--account", type=float, default=10000.0)
    p.add_argument("--train-frac", type=float, default=0.7)
    args = p.parse_args()

    df = load_intraday(args.csv)
    bpy = bars_per_year(df)
    days = np.array(sorted(df["date"].unique()))
    n_train = max(1, int(len(days) * args.train_frac))
    train_days = set(days[:n_train].tolist())
    dates_arr = df["date"].to_numpy()
    train_mask = np.array([d in train_days for d in dates_arr])
    holdout_mask = ~train_mask

    # Locate champion files.
    if args.champs:
        files = [f.strip() for f in args.champs.split(",") if f.strip()]
    else:
        files = sorted(glob.glob("h1_seed*_champion_genome.json"))
    if not files:
        raise SystemExit("No champion JSON files found. Run evolve_traders.py first.")

    print(f"Loaded {len(df)} bars ({df['datetime'].iloc[0].date()} -> "
          f"{df['datetime'].iloc[-1].date()}), {df['date'].nunique()} days "
          f"(~{len(df)/df['date'].nunique():.0f}/day)")
    print(f"Train/holdout split @ {args.train_frac:.0%}  "
          f"(train {len(train_days)}d / holdout {len(days)-len(train_days)}d)")
    print(f"Champions: {', '.join(files)}\n")

    # Run each champion.
    names, bar_rets, eqs, trade_counts = [], [], [], []
    for f in files:
        d = json.load(open(f))
        g = d["genome"]
        name = f.replace("_champion_genome.json", "").replace("h1_", "")
        br, eq, trades = run_champion(df, g, args.account)
        names.append(name)
        bar_rets.append(br)
        eqs.append(eq)
        trade_counts.append(len(trades))

    # Equal-weight, continuously-rebalanced ensemble.
    ens_bar = np.mean(np.vstack(bar_rets), axis=0)
    ens_eq = args.account * np.cumprod(1 + ens_bar)

    # Buy & hold benchmark.
    bh_bar = df["close"].pct_change().fillna(0.0).to_numpy()
    bh_eq = args.account * np.cumprod(1 + bh_bar)

    # ---- Build comparison table ----
    def row(label, br, eq, n_trades):
        full = slice_stats(br, eq, np.ones(len(df), dtype=bool), bpy)
        tr = slice_stats(br, eq, train_mask, bpy)
        ho = slice_stats(br, eq, holdout_mask, bpy)
        return {
            "Strategy": label,
            "Trades": n_trades,
            "Full Ret%": f"{full['ret']*100:.1f}",
            "Full Shp": f"{full['sharpe']:.2f}",
            "Full DD%": f"{full['max_dd']*100:.1f}",
            "HO Ret%": f"{ho['ret']*100:.1f}",
            "HO Shp": f"{ho['sharpe']:.2f}",
            "HO Sortino": f"{ho['sortino']:.2f}",
            "HO DD%": f"{ho['max_dd']*100:.1f}",
            "_ho_sharpe": ho["sharpe"],
            "_ho_dd": ho["max_dd"],
            "_full_sharpe": full["sharpe"],
            "_full_dd": full["max_dd"],
        }

    rows = []
    for nm, br, eq, nt in zip(names, bar_rets, eqs, trade_counts):
        rows.append(row(nm, br, eq, nt))
    ens_row = row("ENSEMBLE (eq-wt)", ens_bar, ens_eq, sum(trade_counts))
    bh_row = row("Buy & Hold", bh_bar, bh_eq, 1)
    rows.append(ens_row)
    rows.append(bh_row)

    table = pd.DataFrame(rows)
    show_cols = ["Strategy", "Trades", "Full Ret%", "Full Shp", "Full DD%",
                 "HO Ret%", "HO Shp", "HO Sortino", "HO DD%"]
    print("=" * 96)
    print("  ENSEMBLE vs INDIVIDUAL CHAMPIONS vs BUY & HOLD  (HO = holdout, untouched OOS)")
    print("=" * 96)
    with pd.option_context("display.width", 200):
        print(table[show_cols].to_string(index=False))
    print("=" * 96)

    # ---- Diversification verdict (vs the AVERAGE individual) ----
    ind = [r for r in rows if r["Strategy"] not in ("ENSEMBLE (eq-wt)", "Buy & Hold")]
    avg_ho_sharpe = np.mean([r["_ho_sharpe"] for r in ind])
    avg_ho_dd = np.mean([r["_ho_dd"] for r in ind])
    avg_full_sharpe = np.mean([r["_full_sharpe"] for r in ind])
    avg_full_dd = np.mean([r["_full_dd"] for r in ind])
    print("\n  Diversification benefit (ensemble vs the AVERAGE single champion):")
    print(f"    HOLDOUT Sharpe : {ens_row['_ho_sharpe']:.2f}  vs avg {avg_ho_sharpe:.2f}"
          f"   ({ens_row['_ho_sharpe']-avg_ho_sharpe:+.2f})")
    print(f"    HOLDOUT MaxDD  : {ens_row['_ho_dd']*100:.1f}% vs avg {avg_ho_dd*100:.1f}%"
          f"   ({(ens_row['_ho_dd']-avg_ho_dd)*100:+.1f} pts)")
    print(f"    FULL    Sharpe : {ens_row['_full_sharpe']:.2f}  vs avg {avg_full_sharpe:.2f}"
          f"   ({ens_row['_full_sharpe']-avg_full_sharpe:+.2f})")
    print(f"    FULL    MaxDD  : {ens_row['_full_dd']*100:.1f}% vs avg {avg_full_dd*100:.1f}%"
          f"   ({(ens_row['_full_dd']-avg_full_dd)*100:+.1f} pts)")

    # ---- Correlation matrix (daily returns, full period) ----
    dr = pd.DataFrame({nm: daily_returns(df, br) for nm, br in zip(names, bar_rets)})
    corr = dr.corr()
    print("\n  Daily-return correlation between champions (lower = more diversification):")
    print(corr.round(2).to_string())
    avg_corr = corr.values[np.triu_indices(len(names), k=1)].mean()
    print(f"\n    Average pairwise correlation: {avg_corr:.2f}")
    if avg_corr < 0.4:
        print("    -> Low/moderate: genuine diversification, the ensemble should smooth returns.")
    elif avg_corr < 0.7:
        print("    -> Moderate-high: some diversification, but the champions overlap.")
    else:
        print("    -> High: champions are near-duplicates; ensemble adds little.")

    # ---- Save ensemble equity curve ----
    out = pd.DataFrame({
        "datetime": df["datetime"].values,
        "ensemble_equity": ens_eq,
        "buyhold_equity": bh_eq,
    })
    out.to_csv("ensemble_equity.csv", index=False)
    print(f"\n  Saved ensemble equity curve -> ensemble_equity.csv")
    print("=" * 96)


if __name__ == "__main__":
    main()
