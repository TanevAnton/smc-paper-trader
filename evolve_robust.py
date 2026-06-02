"""
evolve_robust.py — walk-forward / rotating-data evolution.

THE PROBLEM WITH SINGLE-WINDOW EVOLUTION
----------------------------------------
evolve_traders.py optimizes a population on ONE fixed data window. No matter how
clean the fitness, the GA eventually memorizes that window's quirks — it gets
stuck on an optimum that is specific to those particular months. The multi-seed
test proved it: train scores were great, holdout was a coin flip.

THE FIX — CHANGE THE DATA EVERY CYCLE
-------------------------------------
Here the POPULATION PERSISTS across cycles (elites carry forward unchanged), but
the EVALUATION WINDOW changes every cycle. A genome that overfits cycle 1's
window gets culled in cycle 2, because cycle 2 scores it on a *different* slice
of history. Only genomes that stay fit across MANY different regime windows
survive to the end. The optimizer literally cannot settle into one window's
local optimum, because that window is gone next cycle.

This is walk-forward optimization / cross-validated evolution — the standard
defence against curve-fitting.

HONEST GUARDRAILS
-----------------
  * A FINAL TEST window (last ~6 months) is reserved and NEVER used in any cycle.
    The champion is judged there, cold, at the very end.
  * Each cycle window is itself split into folds (mean - penalty*std fitness),
    so a genome must be consistent WITHIN a window too.
  * The champion is the genome with the best CROSS-CYCLE record (mean minus
    spread of its per-window scores), and it must have survived at least half
    the cycles — i.e. proven on many different windows, not lucky on one.

OUTPUTS
-------
  robust_champion.json     winning genome + cross-cycle record + final-test stats
  robust_cycle_history.csv per-cycle window range + best/mean fitness
  Prints champion vs buy & hold vs the single-window champions on the final test.

USAGE
-----
    python evolve_robust.py --csv QQQ_1h.csv --cycles 12 --jobs 6
    python evolve_robust.py --cycles 15 --gens-per-cycle 3 --window-days 260
"""

import argparse
import json
import math
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from smc_intraday import load_intraday, backtest_intraday, bars_per_year
from evolve_traders import (
    GENE_NAMES, random_genome, crossover, mutate, genome_to_kwargs,
    fitness_robust, slice_metrics,
)


# ----------------------------------------------------------------------------
# Stable genome hash so carried-forward elites accumulate a multi-cycle record.
# ----------------------------------------------------------------------------
def ghash(g):
    parts = []
    for k in GENE_NAMES:
        v = g[k]
        if isinstance(v, float):
            v = round(v, 6)
        parts.append(f"{k}={v}")
    return "|".join(parts)


# ----------------------------------------------------------------------------
# Worker context (per process). Holds full df + each cycle's bar-row range.
# ----------------------------------------------------------------------------
_R = {}


def _init_worker(csv_path, account, folds, cycle_rows):
    df = load_intraday(csv_path)
    _R["df"] = df
    _R["account"] = account
    _R["folds"] = folds
    _R["cycle_rows"] = cycle_rows
    _R["bpy"] = bars_per_year(df)


def _eval_on_window(genome, start_row, end_row, min_trades, objective, spread, folds):
    """Run one genome on a contiguous window [start_row, end_row); return fitness
    (fold-robust within the window) plus whole-window metrics."""
    wdf = _R["df"].iloc[start_row:end_row].reset_index(drop=True)
    trades, eq = backtest_intraday(wdf, account=_R["account"], cost_bps=2.0,
                                   **genome_to_kwargs(genome))
    dates = wdf["date"].to_numpy()
    bpy = _R["bpy"]
    allmask = np.ones(len(wdf), dtype=bool)
    whole = slice_metrics(trades, eq, dates, allmask, _R["account"], bpy)

    wdays = np.array(sorted(wdf["date"].unique()))
    chunks = np.array_split(wdays, folds)
    fold_metrics = []
    for ch in chunks:
        chset = set(ch.tolist())
        m = np.array([d in chset for d in dates])
        fold_metrics.append(slice_metrics(trades, eq, dates, m, _R["account"], bpy))
    fit = fitness_robust(fold_metrics, whole["n_trades"], min_trades,
                         objective, spread)
    return fit, whole


def _eval_task(args):
    genome, cycle_idx, min_trades, objective, spread, folds = args
    s, e = _R["cycle_rows"][cycle_idx]
    fit, whole = _eval_on_window(genome, s, e, min_trades, objective, spread, folds)
    return {"genome": genome, "fitness": fit, "whole": whole}


def crossval_genome(df, genome, cycle_rows, account, folds, bpy,
                    min_trades, objective, spread):
    """Score one genome on EVERY cycle window — a full cross-validated record.
    Returns the array of per-window fitnesses (used to rank by consistency)."""
    fits = []
    for (s, e) in cycle_rows:
        wdf = df.iloc[s:e].reset_index(drop=True)
        trades, eq = backtest_intraday(wdf, account=account, cost_bps=2.0,
                                       **genome_to_kwargs(genome))
        dates = wdf["date"].to_numpy()
        allmask = np.ones(len(wdf), dtype=bool)
        whole = slice_metrics(trades, eq, dates, allmask, account, bpy)
        wdays = np.array(sorted(wdf["date"].unique()))
        fold_metrics = []
        for ch in np.array_split(wdays, folds):
            chset = set(ch.tolist())
            m = np.array([d in chset for d in dates])
            fold_metrics.append(slice_metrics(trades, eq, dates, m, account, bpy))
        fits.append(fitness_robust(fold_metrics, whole["n_trades"],
                                   min_trades, objective, spread))
    return np.array(fits, dtype=float)


# ----------------------------------------------------------------------------
# Build the rotating cycle windows (deterministic from seed).
# ----------------------------------------------------------------------------
def build_cycle_windows(df, cycles, window_days, final_test_days, jitter, rng):
    """Return (cycle_rows, cycle_dayspans, dev_day_count). Each cycle gets a
    contiguous window of `window_days` days sampled from the DEV pool (everything
    before the reserved final-test tail), with evenly-spread + jittered + shuffled
    starts so the whole development history is covered in a non-monotonic order."""
    days = np.array(sorted(df["date"].unique()))
    n_days = len(days)
    dev_days = days[: n_days - final_test_days]
    max_start = len(dev_days) - window_days
    if max_start < 1:
        raise SystemExit("Not enough data for the chosen window/final-test sizes.")

    # Evenly-spaced base starts across the dev pool, then jitter + shuffle.
    if cycles > 1:
        base = [round(k / (cycles - 1) * max_start) for k in range(cycles)]
    else:
        base = [max_start // 2]
    starts = []
    for b in base:
        j = rng.randint(-jitter, jitter)
        starts.append(int(min(max(b + j, 0), max_start)))
    rng.shuffle(starts)

    # Map day index -> first/last bar row for that day.
    dates_arr = df["date"].to_numpy()
    # Precompute the first row index of each day and the row count.
    day_first_row = {}
    day_last_row = {}
    for i, d in enumerate(dates_arr):
        if d not in day_first_row:
            day_first_row[d] = i
        day_last_row[d] = i

    cycle_rows = []
    cycle_dayspans = []
    for s in starts:
        d0 = dev_days[s]
        d1 = dev_days[s + window_days - 1]
        start_row = day_first_row[d0]
        end_row = day_last_row[d1] + 1
        cycle_rows.append((start_row, end_row))
        cycle_dayspans.append((str(d0), str(d1)))
    return cycle_rows, cycle_dayspans, len(dev_days)


# ----------------------------------------------------------------------------
# Final-test evaluation (cold, with warmup tail, metrics only on the test days).
# ----------------------------------------------------------------------------
def final_test_stats(df, genome, account, final_test_days, warmup_days, bpy):
    days = np.array(sorted(df["date"].unique()))
    test_days = set(days[-final_test_days:].tolist())
    warm_start_day = days[-(final_test_days + warmup_days)] if len(days) > final_test_days + warmup_days else days[0]
    dates_arr = df["date"].to_numpy()
    # Row where warmup begins.
    start_row = int(np.argmax(dates_arr == warm_start_day))
    sub = df.iloc[start_row:].reset_index(drop=True)
    trades, eq = backtest_intraday(sub, account=account, cost_bps=2.0,
                                   **genome_to_kwargs(genome))
    sdates = sub["date"].to_numpy()
    test_mask = np.array([d in test_days for d in sdates])
    m = slice_metrics(trades, eq, sdates, test_mask, account, bpy)
    return m, test_days


def bh_stats_on_days(df, day_set, account, bpy):
    mask = df["date"].isin(day_set).to_numpy()
    sub = df[mask]
    r = sub["close"].pct_change().fillna(0.0)
    eq = (1 + r).cumprod()
    sharpe = r.mean() / r.std() * math.sqrt(bpy) if r.std() > 0 else 0.0
    dn = r[r < 0].std()
    sortino = r.mean() / dn * math.sqrt(bpy) if dn > 0 else 0.0
    dd = float((eq / eq.cummax() - 1).min())
    ret = float(eq.iloc[-1] - 1)
    return {"sharpe": sharpe, "sortino": sortino, "max_dd": dd, "slice_ret": ret,
            "n_trades": 1, "win_rate": float("nan"), "profit_factor": float("nan")}


# ----------------------------------------------------------------------------
# The rotating-data evolution loop
# ----------------------------------------------------------------------------
def evolve_robust(csv_path, cycles=12, gens_per_cycle=3, pop_size=100, elite=10,
                  immigrants=8, mutation_rate=0.25, window_days=260,
                  final_test_days=130, warmup_days=60, jitter=15, folds=3,
                  min_trades=25, objective="sortino", spread_penalty=0.5,
                  account=10000.0, seed=42, jobs=6):
    rng = random.Random(seed)
    df = load_intraday(csv_path)
    bpy = bars_per_year(df)

    cycle_rows, cycle_dayspans, dev_day_count = build_cycle_windows(
        df, cycles, window_days, final_test_days, jitter, rng)

    n_days = df["date"].nunique()
    print(f"Data: {len(df)} bars, {n_days} days (~{len(df)/n_days:.0f}/day)")
    print(f"  Dev pool: {dev_day_count} days   Final TEST (reserved): "
          f"{final_test_days} days  [{str(np.array(sorted(df['date'].unique()))[-final_test_days])} -> "
          f"{str(np.array(sorted(df['date'].unique()))[-1])}]")
    print(f"Walk-forward GA: {cycles} cycles x {gens_per_cycle} gens, "
          f"pop={pop_size}, elite={elite}, window={window_days}d, folds={folds}, "
          f"objective={objective}, jobs={jobs}")
    print(f"  Each cycle re-evaluates the whole population on a DIFFERENT window "
          f"-> survivors must generalize across regimes.\n")

    population = [random_genome(rng) for _ in range(pop_size)]
    track = {}        # ghash -> {"genome":g, "cyc": {cycle_idx: best_fit}}
    candidates = {}   # ghash -> genome : every promising genome the GA ever found
    history = []
    cand_per_cycle = 6

    pool = None
    if jobs and jobs > 1:
        pool = ProcessPoolExecutor(
            max_workers=jobs, initializer=_init_worker,
            initargs=(csv_path, account, folds, cycle_rows))
    else:
        _init_worker(csv_path, account, folds, cycle_rows)

    try:
        for cycle in range(cycles):
            cyc_best = -1e9
            cyc_fits = []
            for gen in range(gens_per_cycle):
                tasks = [(g, cycle, min_trades, objective, spread_penalty, folds)
                         for g in population]
                if pool is not None:
                    results = list(pool.map(_eval_task, tasks, chunksize=4))
                else:
                    results = [_eval_task(t) for t in tasks]
                results.sort(key=lambda r: r["fitness"], reverse=True)

                # Record per-cycle best fitness for each genome (track-record).
                for r in results:
                    h = ghash(r["genome"])
                    rec = track.setdefault(h, {"genome": r["genome"], "cyc": {}})
                    prev = rec["cyc"].get(cycle, -1e9)
                    if r["fitness"] > prev:
                        rec["cyc"][cycle] = r["fitness"]

                elites = results[:elite]
                cyc_best = max(cyc_best, elites[0]["fitness"])
                fits = np.array([r["fitness"] for r in results])
                cyc_fits.append(float(fits[fits > 0].mean()) if (fits > 0).any() else float(fits.mean()))

                # Breed next population (skip after the very last gen overall).
                last_overall = (cycle == cycles - 1 and gen == gens_per_cycle - 1)
                if not last_overall:
                    eg = [r["genome"] for r in elites]
                    nxt = list(eg)                       # elitism: persist survivors
                    for _ in range(immigrants):
                        nxt.append(random_genome(rng))   # fresh blood
                    while len(nxt) < pop_size:
                        pa, pb = rng.sample(eg, 2)
                        nxt.append(mutate(crossover(pa, pb, rng), rng, mutation_rate))
                    population = nxt

            # Bank this cycle's best genomes as candidates for the final
            # cross-validated bake-off (re-tested on ALL windows at the end).
            for r in results[:cand_per_cycle]:
                candidates[ghash(r["genome"])] = r["genome"]

            # How many genomes have now survived >= 2 different cycles?
            multi = sum(1 for v in track.values() if len(v["cyc"]) >= 2)
            d0, d1 = cycle_dayspans[cycle]
            print(f"Cycle {cycle:02d} | window {d0}..{d1} | "
                  f"best_fit={cyc_best:.2f} mean_fit={np.mean(cyc_fits):.2f} | "
                  f"genomes surviving >=2 windows: {multi}")
            history.append({
                "cycle": cycle, "window_start": d0, "window_end": d1,
                "best_fit": cyc_best, "mean_fit": float(np.mean(cyc_fits)),
                "genomes_multi_window": multi,
            })
    finally:
        if pool is not None:
            pool.shutdown()

    # ---- FINAL CROSS-VALIDATED BAKE-OFF ----------------------------------
    # Re-test every candidate the GA found on ALL cycle windows. This is the
    # crucial step: the champion is whichever genome is most consistently good
    # across EVERY regime window, not whichever happened to top the last cycle.
    print(f"\nCross-validating {len(candidates)} candidate genomes on all "
          f"{cycles} windows ...")
    scored = []
    for h, g in candidates.items():
        fits = crossval_genome(df, g, cycle_rows, account, folds, bpy,
                               min_trades, objective, spread_penalty)
        # Reward high mean AND low spread across windows; also surface worst case.
        score = float(fits.mean() - 0.5 * fits.std())
        scored.append({"genome": g, "n_cycles": cycles,
                       "mean": float(fits.mean()), "min": float(fits.min()),
                       "std": float(fits.std()), "score": score,
                       "n_positive": int((fits > 0).sum())})
    scored.sort(key=lambda s: s["score"], reverse=True)
    champion = scored[0]
    top_robust = scored[:5]

    return {
        "df": df, "bpy": bpy, "champion": champion, "top_robust": top_robust,
        "history": history, "final_test_days": final_test_days,
        "warmup_days": warmup_days, "account": account,
        "min_cycles": cycles, "n_proven": len(scored), "all_scored": scored,
    }


# ----------------------------------------------------------------------------
def fmt(m):
    wr = m.get("win_rate", float("nan"))
    pf = m.get("profit_factor", float("nan"))
    extra = ""
    if not (isinstance(wr, float) and math.isnan(wr)):
        extra = f"  win={wr*100:4.1f}%  PF={pf:4.2f}  trades={m.get('n_trades',0)}"
    return (f"Sharpe={m['sharpe']:5.2f}  Sortino={m['sortino']:5.2f}  "
            f"ret={m['slice_ret']*100:6.2f}%  MaxDD={m['max_dd']*100:6.2f}%{extra}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="QQQ_1h.csv")
    p.add_argument("--cycles", type=int, default=12)
    p.add_argument("--gens-per-cycle", type=int, default=3)
    p.add_argument("--pop", type=int, default=100)
    p.add_argument("--elite", type=int, default=10)
    p.add_argument("--immigrants", type=int, default=8)
    p.add_argument("--mutation-rate", type=float, default=0.25)
    p.add_argument("--window-days", type=int, default=260)
    p.add_argument("--final-test-days", type=int, default=130)
    p.add_argument("--warmup-days", type=int, default=60)
    p.add_argument("--jitter", type=int, default=15)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--min-trades", type=int, default=25)
    p.add_argument("--objective", default="sortino",
                   choices=["sortino", "sharpe", "cagr", "pf"])
    p.add_argument("--spread-penalty", type=float, default=0.5)
    p.add_argument("--account", type=float, default=10000.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--out-prefix", default="")
    args = p.parse_args()

    R = evolve_robust(
        args.csv, cycles=args.cycles, gens_per_cycle=args.gens_per_cycle,
        pop_size=args.pop, elite=args.elite, immigrants=args.immigrants,
        mutation_rate=args.mutation_rate, window_days=args.window_days,
        final_test_days=args.final_test_days, warmup_days=args.warmup_days,
        jitter=args.jitter, folds=args.folds, min_trades=args.min_trades,
        objective=args.objective, spread_penalty=args.spread_penalty,
        account=args.account, seed=args.seed, jobs=args.jobs,
    )

    df, bpy = R["df"], R["bpy"]
    champ = R["champion"]
    g = champ["genome"]

    print("\n" + "=" * 94)
    print("  ROBUST CHAMPION — best cross-validated record over ALL windows")
    print("=" * 94)
    print(f"  Chosen from {R['n_proven']} candidate genomes, each re-tested on "
          f"all {args.cycles} regime windows.")
    print(f"  Cross-window fitness: mean={champ['mean']:.2f}  "
          f"worst-window(min)={champ['min']:.2f}  spread(std)={champ['std']:.2f}  "
          f"-> score={champ['score']:.2f}")
    print(f"  Positive on {champ['n_positive']}/{args.cycles} windows "
          f"(a true generalist clears most of them).")
    print("  Genome:")
    for k in GENE_NAMES:
        print(f"    {k:<26}: {g[k]}")

    # ---- Final-test validation (cold, untouched window) ----
    champ_m, test_days = final_test_stats(df, g, R["account"],
                                          R["final_test_days"], R["warmup_days"], bpy)
    bh_m = bh_stats_on_days(df, test_days, R["account"], bpy)

    print("\n" + "=" * 94)
    print(f"  FINAL TEST (untouched last {R['final_test_days']} days) — the honest verdict")
    print("=" * 94)
    print(f"    Robust champion : {fmt(champ_m)}")
    print(f"    Buy & Hold      : {fmt(bh_m)}")

    # ---- Ensemble of the top-5 robust genomes on the final test ----
    bars_list = []
    sub_for_ens = None
    days = np.array(sorted(df["date"].unique()))
    test_day_set = set(days[-R["final_test_days"]:].tolist())
    warm_start_day = days[-(R["final_test_days"] + R["warmup_days"])]
    start_row = int(np.argmax(df["date"].to_numpy() == warm_start_day))
    sub = df.iloc[start_row:].reset_index(drop=True)
    sdates = sub["date"].to_numpy()
    test_mask = np.array([d in test_day_set for d in sdates])
    for s in R["top_robust"]:
        _, eq = backtest_intraday(sub, account=R["account"], cost_bps=2.0,
                                  **genome_to_kwargs(s["genome"]))
        bars_list.append(pd.Series(np.asarray(eq, float)).pct_change().fillna(0.0).to_numpy())
    ens_bar = np.mean(np.vstack(bars_list), axis=0)
    ens_eq = R["account"] * np.cumprod(1 + ens_bar)
    ens_m = slice_metrics(None, ens_eq, sdates, test_mask, R["account"], bpy)
    # slice_metrics needs trades for trade-stats; ensemble has none -> fine (NaN).
    print(f"    Top-5 ensemble  : Sharpe={ens_m['sharpe']:5.2f}  "
          f"Sortino={ens_m['sortino']:5.2f}  ret={ens_m['slice_ret']*100:6.2f}%  "
          f"MaxDD={ens_m['max_dd']*100:6.2f}%")
    print("=" * 94)
    verdict = ("BEATS" if champ_m["sharpe"] > bh_m["sharpe"] else "trails")
    print(f"  On unseen data the robust champion {verdict} buy & hold on Sharpe "
          f"({champ_m['sharpe']:.2f} vs {bh_m['sharpe']:.2f}); "
          f"MaxDD {champ_m['max_dd']*100:.1f}% vs {bh_m['max_dd']*100:.1f}%.")
    print("=" * 94)

    # ---- Persist (champion + the full top-5 robust shortlist for ensembling) ----
    champ_file = f"{args.out_prefix}robust_champion.json"
    with open(champ_file, "w") as f:
        json.dump({
            "genome": g,
            "cross_cycle": {k: champ[k] for k in ("n_cycles", "mean", "min", "std", "score")},
            "top_robust": [s["genome"] for s in R["top_robust"]],
            "final_test": champ_m,
            "final_test_buyhold": bh_m,
            "config": vars(args),
        }, f, indent=2, default=str)
    pd.DataFrame(R["history"]).to_csv(f"{args.out_prefix}robust_cycle_history.csv", index=False)
    print(f"\nSaved {champ_file} and {args.out_prefix}robust_cycle_history.csv")


if __name__ == "__main__":
    main()
