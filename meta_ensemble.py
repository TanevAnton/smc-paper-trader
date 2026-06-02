"""
meta_ensemble.py — the culmination: diversify across SEEDS, not just genomes.

THE LOGIC
---------
The robust (rotating-data) evolution proved that a single evolved champion is
still seed-dependent out-of-sample: different random seeds produce champions
whose final-test Sharpe ranges from ~0 to ~2.8, even though every one passes the
12-window cross-validation. You cannot know in advance which seed got lucky.

The cure for "which one do I pick?" is: don't pick. Run them ALL as one book.
This script loads the robust champions from every seed and trades them equally
weighted, then judges the combined book on the untouched final-test window. If
the seeds' errors are partly independent, the meta-ensemble lands near the TOP
of the seed range with a smoother curve — turning "pick the lucky seed" into
"hold the diversified portfolio."

Two ensembles are built:
  * SEED CHAMPIONS  — the single best genome from each seed (N seeds).
  * ALL ROBUST      — every seed's top-5 robust shortlist (N x 5 genomes).

USAGE
-----
    python meta_ensemble.py                       # auto-loads rb_s*_robust_champion.json
    python meta_ensemble.py --final-test-days 130 --warmup-days 60
"""

import argparse
import glob
import json
import math

import numpy as np
import pandas as pd

from smc_intraday import load_intraday, backtest_intraday, bars_per_year
from evolve_traders import genome_to_kwargs


def run_genome_returns(sub, genome, account):
    _, eq = backtest_intraday(sub, account=account, cost_bps=2.0,
                              **genome_to_kwargs(genome))
    eq = np.asarray(eq, dtype=float)
    return pd.Series(eq).pct_change().fillna(0.0).to_numpy()


def stats(bar_ret, mask, bpy, account):
    full_eq = account * np.cumprod(1 + bar_ret)
    sel = bar_ret[mask]
    if sel.size == 0 or sel.std() == 0:
        sharpe = sortino = 0.0
    else:
        sharpe = sel.mean() / sel.std() * math.sqrt(bpy)
        dn = sel[sel < 0].std()
        sortino = sel.mean() / dn * math.sqrt(bpy) if dn > 0 else 0.0
    eqs = full_eq[mask]
    peak = np.maximum.accumulate(eqs)
    dd = float((eqs / peak - 1).min())
    ret = float(eqs[-1] / eqs[0] - 1) if eqs[0] > 0 else 0.0
    return {"sharpe": sharpe, "sortino": sortino, "max_dd": dd, "ret": ret}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="QQQ_1h.csv")
    p.add_argument("--glob", default="rb_s*_robust_champion.json")
    p.add_argument("--final-test-days", type=int, default=130)
    p.add_argument("--warmup-days", type=int, default=60)
    p.add_argument("--account", type=float, default=10000.0)
    args = p.parse_args()

    df = load_intraday(args.csv)
    bpy = bars_per_year(df)
    days = np.array(sorted(df["date"].unique()))
    test_days = set(days[-args.final_test_days:].tolist())
    warm_day = days[-(args.final_test_days + args.warmup_days)]
    start_row = int(np.argmax(df["date"].to_numpy() == warm_day))
    sub = df.iloc[start_row:].reset_index(drop=True)
    sdates = sub["date"].to_numpy()
    test_mask = np.array([d in test_days for d in sdates])

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"No champion files match {args.glob!r}. Run evolve_robust.py with --out-prefix first.")

    seed_champ_returns = []      # one per seed (the single champion)
    all_robust_returns = []      # every seed's top-5
    seed_rows = []
    for f in files:
        d = json.load(open(f))
        seed = f.split("robust_champion")[0]
        champ_ret = run_genome_returns(sub, d["genome"], args.account)
        seed_champ_returns.append(champ_ret)
        m = stats(champ_ret, test_mask, bpy, args.account)
        seed_rows.append({"name": seed, **m})
        for gen in d.get("top_robust", [d["genome"]]):
            all_robust_returns.append(run_genome_returns(sub, gen, args.account))

    # Benchmarks / ensembles.
    bh_ret = df["close"].pct_change().fillna(0.0)
    bh_ret = sub["close"].pct_change().fillna(0.0).to_numpy()
    bh_m = stats(bh_ret, test_mask, bpy, args.account)

    seed_ens = np.mean(np.vstack(seed_champ_returns), axis=0)
    seed_ens_m = stats(seed_ens, test_mask, bpy, args.account)
    all_ens = np.mean(np.vstack(all_robust_returns), axis=0)
    all_ens_m = stats(all_ens, test_mask, bpy, args.account)

    def line(name, m, n=""):
        return (f"  {name:<26} Sharpe={m['sharpe']:5.2f}  Sortino={m['sortino']:5.2f}  "
                f"ret={m['ret']*100:6.2f}%  MaxDD={m['max_dd']*100:6.2f}%  {n}")

    print("=" * 92)
    print(f"  CROSS-SEED META-ENSEMBLE on the untouched final {args.final_test_days}-day test")
    print(f"  ({sub[test_mask]['datetime'].iloc[0].date()} -> "
          f"{sub[test_mask]['datetime'].iloc[-1].date()})")
    print("=" * 92)
    print("  Individual seed champions:")
    sharpes = []
    for r in seed_rows:
        print(line(r["name"], r))
        sharpes.append(r["sharpe"])
    print("  " + "-" * 88)
    print(line("SEED-CHAMPIONS ensemble", seed_ens_m, f"({len(seed_champ_returns)} genomes)"))
    print(line("ALL-ROBUST ensemble", all_ens_m, f"({len(all_robust_returns)} genomes)"))
    print(line("Buy & Hold", bh_m))
    print("=" * 92)
    print(f"  Individual seed-champion Sharpe range: "
          f"{min(sharpes):.2f} .. {max(sharpes):.2f}  (mean {np.mean(sharpes):.2f})")
    print(f"  Meta-ensemble Sharpe: {all_ens_m['sharpe']:.2f}  — diversifying across")
    print(f"  seeds removes the 'which lucky seed?' gamble.")
    print(f"  Drawdown: meta-ensemble {all_ens_m['max_dd']*100:.1f}% vs "
          f"buy & hold {bh_m['max_dd']*100:.1f}%.")
    print("=" * 92)


if __name__ == "__main__":
    main()
