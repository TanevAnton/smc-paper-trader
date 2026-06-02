"""
evolve_traders.py — a genetic algorithm that evolves a population of 100
intraday "trader" agents and breeds the next generation from the best survivors.

THE IDEA (your spec)
--------------------
  * Each AGENT is a genome = one full set of SMC strategy parameters.
  * Spawn a POPULATION of 100 agents with random genomes.
  * Score every agent by trading the QQQ 5m data.
  * Keep the TOP 10 (truncation selection / elitism).
  * Breed a NEW population of 100 from those 10 via crossover + mutation
    (+ a few fresh random "immigrants" so the gene pool never stagnates).
  * Repeat for N generations. Fitness climbs each generation.

WHY A TRAIN / HOLDOUT SPLIT (the part that keeps this honest)
-------------------------------------------------------------
A GA is an *overfitting machine*. Given enough generations it WILL find a
genome that looks spectacular on the data it was scored on — and falls apart
live. To measure that, we:

  * Evolve using fitness on the TRAIN window only (first `train_frac` of days).
  * Never let the GA see the HOLDOUT window.
  * After evolution, re-score the champion on the untouched holdout and print
    the train -> holdout decay side by side. The gap IS the overfitting.

We also GATE fitness on a minimum trade count, so the GA can't "win" with two
lucky trades — a genome must trade enough to be statistically real before its
risk-adjusted score counts.

OUTPUTS
-------
  champion_genome.json    best genome + its train/holdout metrics
  evolution_history.csv   per-generation best/mean fitness (the learning curve)
  Prints the exact `smc_intraday.py` command to reproduce the champion.

USAGE
-----
    python evolve_traders.py
    python evolve_traders.py --pop 100 --elite 10 --gens 20 --jobs 8
    python evolve_traders.py --objective sharpe --train-frac 0.7 --seed 7
"""

import argparse
import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from smc_intraday import load_intraday, backtest_intraday, bars_per_year


# ----------------------------------------------------------------------------
# Gene space — the search domain for each strategy parameter.
#   ("int",   lo, hi)  integer gene in [lo, hi]
#   ("float", lo, hi)  float gene in [lo, hi]
#   ("bool",)          boolean gene
# ----------------------------------------------------------------------------
GENE_SPACE = {
    "swing_n":                 ("int", 2, 8),
    "min_fvg_pct":             ("float", 0.0002, 0.0040),
    "rr_partial":              ("float", 0.5, 2.5),
    "rr_runner":               ("float", 1.5, 8.0),
    "stop_buf":                ("float", 0.0, 2.0),
    "atr_n":                   ("int", 7, 30),
    "ema_n":                   ("int", 50, 400),
    "max_hold_bars":           ("int", 40, 468),    # up to ~6 RTH days
    "max_bars_after_bos":      ("int", 4, 48),       # 20 min .. 4 hours
    "risk_per_trade":          ("float", 0.0025, 0.0200),
    "allow_short":             ("bool",),
    "use_pd_filter":           ("bool",),
    "use_trend_filter":        ("bool",),
    "use_cooldown":            ("bool",),
    "use_session_confluence":  ("bool",),
    "flatten_at_session_close":("bool",),
    "no_entries_before_min":   ("int", 9 * 60 + 30, 11 * 60),   # 09:30 .. 11:00
    "no_new_entries_after_min":("int", 14 * 60, 16 * 60),       # 14:00 .. 16:00
}
GENE_NAMES = list(GENE_SPACE.keys())


# ----------------------------------------------------------------------------
# Genome operations
# ----------------------------------------------------------------------------
def random_gene(name, rng):
    spec = GENE_SPACE[name]
    if spec[0] == "int":
        return rng.randint(spec[1], spec[2])
    if spec[0] == "float":
        return rng.uniform(spec[1], spec[2])
    return bool(rng.getrandbits(1))


def random_genome(rng):
    return repair({n: random_gene(n, rng) for n in GENE_NAMES})


def repair(g):
    """Clamp genes to bounds and enforce sane cross-gene constraints."""
    out = dict(g)
    for n in GENE_NAMES:
        spec = GENE_SPACE[n]
        if spec[0] == "int":
            out[n] = int(round(max(spec[1], min(spec[2], out[n]))))
        elif spec[0] == "float":
            out[n] = float(max(spec[1], min(spec[2], out[n])))
        else:
            out[n] = bool(out[n])
    # Partial target must sit inside the runner target.
    if out["rr_partial"] >= out["rr_runner"]:
        out["rr_partial"] = max(0.5, out["rr_runner"] * 0.5)
    # Entry window must be a positive interval with >= 30 min of room.
    if out["no_entries_before_min"] >= out["no_new_entries_after_min"] - 30:
        out["no_entries_before_min"] = 10 * 60
        out["no_new_entries_after_min"] = 15 * 60 + 30
    return out


def crossover(a, b, rng):
    """Uniform crossover: each gene taken from parent a or b with equal odds."""
    return repair({n: (a[n] if rng.random() < 0.5 else b[n]) for n in GENE_NAMES})


def mutate(g, rng, rate):
    """Mutate each gene with probability `rate`. Floats/ints get a Gaussian
    step ~15% of their range; bools flip."""
    out = dict(g)
    for n in GENE_NAMES:
        if rng.random() >= rate:
            continue
        spec = GENE_SPACE[n]
        if spec[0] == "int":
            span = spec[2] - spec[1]
            out[n] = out[n] + rng.gauss(0, 0.15 * span)
        elif spec[0] == "float":
            span = spec[2] - spec[1]
            out[n] = out[n] + rng.gauss(0, 0.15 * span)
        else:
            out[n] = not out[n]
    return repair(out)


def genome_to_kwargs(g):
    """Map a genome straight onto backtest_intraday's keyword args."""
    return {
        "swing_n": g["swing_n"],
        "min_fvg_pct": g["min_fvg_pct"],
        "rr_partial": g["rr_partial"],
        "rr_runner": g["rr_runner"],
        "stop_buf": g["stop_buf"],
        "atr_n": g["atr_n"],
        "ema_n": g["ema_n"],
        "max_hold_bars": g["max_hold_bars"],
        "max_bars_after_bos": g["max_bars_after_bos"],
        "risk_per_trade": g["risk_per_trade"],
        "allow_short": g["allow_short"],
        "use_pd_filter": g["use_pd_filter"],
        "use_trend_filter": g["use_trend_filter"],
        "use_cooldown": g["use_cooldown"],
        "use_session_confluence": g["use_session_confluence"],
        "flatten_at_session_close": g["flatten_at_session_close"],
        "no_entries_before_min": g["no_entries_before_min"],
        "no_new_entries_after_min": g["no_new_entries_after_min"],
    }


# ----------------------------------------------------------------------------
# Metrics on a slice (train or holdout) of one full backtest result
# ----------------------------------------------------------------------------
def slice_metrics(trades, eq, dates_arr, day_mask, account, bpy):
    """Compute risk metrics over the bars selected by `day_mask`.

    Ratio metrics (Sharpe/Sortino) use bar pct-returns, so they're independent
    of the absolute equity level carried in from earlier bars — fair to compare
    across genomes. Trade stats use trades whose ENTRY falls in the masked days.
    `bpy` = bars-per-year for this timeframe (annualization constant).
    """
    eq = np.asarray(eq, dtype=float)
    bar_ret = pd.Series(eq).pct_change().fillna(0.0).to_numpy()
    sel = bar_ret[day_mask]
    if sel.size == 0 or sel.std() == 0:
        sharpe = sortino = 0.0
    else:
        sharpe = (sel.mean() / sel.std()) * math.sqrt(bpy)
        downside = sel[sel < 0].std()
        sortino = (sel.mean() / downside) * math.sqrt(bpy) if downside > 0 else 0.0

    eq_slice = eq[day_mask]
    if eq_slice.size:
        peak = np.maximum.accumulate(eq_slice)
        max_dd = float((eq_slice / peak - 1).min())
        # Total return contribution of the slice (start->end of slice).
        slice_ret = float(eq_slice[-1] / eq_slice[0] - 1) if eq_slice[0] > 0 else 0.0
    else:
        max_dd = 0.0
        slice_ret = 0.0

    # Trades whose entry date falls in the slice.
    if trades is not None and len(trades):
        ent_dates = pd.to_datetime(trades["entry_dt"]).dt.date.to_numpy()
        day_set = set(dates_arr[day_mask].tolist())
        in_slice = np.array([d in day_set for d in ent_dates])
        ts = trades[in_slice]
    else:
        ts = trades.iloc[0:0] if trades is not None else None

    n = 0 if ts is None else len(ts)
    if n:
        wins = int((ts["pnl_$"] > 0).sum())
        wpnl = float(ts.loc[ts["pnl_$"] > 0, "pnl_$"].sum())
        lpnl = float(-ts.loc[ts["pnl_$"] <= 0, "pnl_$"].sum())
        pf = wpnl / lpnl if lpnl > 0 else (float("inf") if wpnl > 0 else 0.0)
        win_rate = wins / n
        expectancy = float(ts["pnl_$"].mean())
        net_pnl = float(ts["pnl_$"].sum())
    else:
        pf = 0.0
        win_rate = 0.0
        expectancy = 0.0
        net_pnl = 0.0

    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_dd": max_dd,
        "slice_ret": slice_ret,
        "n_trades": int(n),
        "win_rate": float(win_rate),
        "profit_factor": float(pf) if np.isfinite(pf) else 999.0,
        "expectancy": expectancy,
        "net_pnl": net_pnl,
    }


def fitness_of(train_m, min_trades, objective, dd_penalty=4.0):
    """Single fitness number the GA maximizes — computed on TRAIN only.

    Gated by trade count so flukey low-sample genomes can't win, with a light
    drawdown penalty to favor durable equity curves over fragile ones.
    """
    n = train_m["n_trades"]
    if n < min_trades:
        # Monotonic in n: still rewards evolving toward enough activity.
        return -10.0 + (n / max(min_trades, 1))
    base = {
        "sortino": train_m["sortino"],
        "sharpe": train_m["sharpe"],
        "cagr": train_m["slice_ret"],         # slice total return as growth proxy
        "pf": min(train_m["profit_factor"], 5.0),
    }[objective]
    if not np.isfinite(base):
        base = 0.0
    return base / (1.0 + dd_penalty * abs(train_m["max_dd"]))


# ----------------------------------------------------------------------------
# Worker side (supports multiprocessing). Globals are populated per-process.
# ----------------------------------------------------------------------------
_W = {}


def _init_worker(csv_path, account, train_frac, folds=1):
    df = load_intraday(csv_path)
    days = np.array(sorted(df["date"].unique()))
    n_train = max(1, int(len(days) * train_frac))
    train_day_list = days[:n_train]
    train_days = set(train_day_list.tolist())
    dates_arr = df["date"].to_numpy()
    train_mask = np.array([d in train_days for d in dates_arr])
    holdout_mask = ~train_mask

    # Split the TRAIN days into `folds` contiguous chunks for walk-forward
    # robustness scoring. Each fold mask selects that chunk's bars.
    fold_masks = []
    if folds > 1:
        chunks = np.array_split(train_day_list, folds)
        for ch in chunks:
            ch_set = set(ch.tolist())
            fold_masks.append(np.array([d in ch_set for d in dates_arr]))
    else:
        fold_masks = [train_mask]

    _W["df"] = df
    _W["dates_arr"] = dates_arr
    _W["train_mask"] = train_mask
    _W["holdout_mask"] = holdout_mask
    _W["fold_masks"] = fold_masks
    _W["account"] = account
    _W["bpy"] = bars_per_year(df)


def fitness_robust(fold_metrics, total_train_trades, min_trades, objective,
                   spread_penalty=0.5):
    """Walk-forward fitness: reward genomes that score well in EVERY train fold,
    not just on average. fitness = mean(fold score) - spread_penalty * std.

    A genome that's brilliant in one sub-period and flat in another gets
    punished by the std term — which is exactly the fragility that fails on
    holdout. Gated by TOTAL train trade count so low-sample flukes can't win.
    """
    if total_train_trades < min_trades:
        return -10.0 + (total_train_trades / max(min_trades, 1))
    key = {"sortino": "sortino", "sharpe": "sharpe",
           "cagr": "slice_ret", "pf": "profit_factor"}[objective]
    vals = []
    for fm in fold_metrics:
        v = fm[key]
        if objective == "pf":
            v = min(v, 5.0)
        if not np.isfinite(v):
            v = 0.0
        vals.append(v)
    vals = np.array(vals, dtype=float)
    return float(vals.mean() - spread_penalty * vals.std())


def evaluate(genome, min_trades, objective, spread_penalty=0.5):
    """Run one genome on the shared data; return fitness + train/holdout metrics.

    If the worker was initialized with folds>1, fitness is the walk-forward
    robust score across train folds; otherwise it's the single-window score.
    """
    df = _W["df"]
    trades, eq = backtest_intraday(df, account=_W["account"], cost_bps=2.0,
                                   **genome_to_kwargs(genome))
    dates_arr, account, bpy = _W["dates_arr"], _W["account"], _W["bpy"]
    train_m = slice_metrics(trades, eq, dates_arr, _W["train_mask"], account, bpy)
    hold_m = slice_metrics(trades, eq, dates_arr, _W["holdout_mask"], account, bpy)

    fold_masks = _W["fold_masks"]
    if len(fold_masks) > 1:
        fold_metrics = [slice_metrics(trades, eq, dates_arr, fm, account, bpy)
                        for fm in fold_masks]
        fit = fitness_robust(fold_metrics, train_m["n_trades"], min_trades,
                             objective, spread_penalty)
    else:
        fit = fitness_of(train_m, min_trades, objective)

    return {"genome": genome, "fitness": fit, "train": train_m, "holdout": hold_m}


def _eval_task(args):
    genome, min_trades, objective, spread_penalty = args
    return evaluate(genome, min_trades, objective, spread_penalty)


# ----------------------------------------------------------------------------
# Evolution loop
# ----------------------------------------------------------------------------
def evolve(csv_path, pop_size=100, elite=10, gens=15, immigrants=5,
           mutation_rate=0.2, train_frac=0.7, min_trades=20,
           objective="sortino", account=10000.0, seed=42, jobs=1,
           folds=1, spread_penalty=0.5):
    rng = random.Random(seed)

    # Build the data context once in THIS process too (for serial mode + repair).
    _init_worker(csv_path, account, train_frac, folds)
    df = _W["df"]
    n_days = df["date"].nunique()
    n_train_days = len(set(df["date"].to_numpy()[_W["train_mask"]]))
    n_hold_days = n_days - n_train_days
    fit_mode = (f"walk-forward {folds} folds (mean - {spread_penalty}*std)"
                if folds > 1 else "single-window")
    print(f"Data: {len(df)} bars, {n_days} days "
          f"(train {n_train_days}d / holdout {n_hold_days}d, "
          f"split @ {train_frac:.0%})")
    print(f"GA: pop={pop_size}, elite={elite}, immigrants={immigrants}, "
          f"gens={gens}, mutation={mutation_rate}, objective={objective}, "
          f"min_trades={min_trades}, jobs={jobs}")
    print(f"Fitness: {fit_mode}\n")

    population = [random_genome(rng) for _ in range(pop_size)]
    history = []
    hall_of_fame = None  # best-on-train genome ever seen

    pool = None
    if jobs and jobs > 1:
        pool = ProcessPoolExecutor(
            max_workers=jobs,
            initializer=_init_worker,
            initargs=(csv_path, account, train_frac, folds),
        )

    try:
        for gen in range(gens):
            tasks = [(g, min_trades, objective, spread_penalty) for g in population]
            if pool is not None:
                results = list(pool.map(_eval_task, tasks, chunksize=4))
            else:
                results = [evaluate(g, min_trades, objective, spread_penalty)
                           for g in population]

            results.sort(key=lambda r: r["fitness"], reverse=True)
            elites = results[:elite]

            best = elites[0]
            fits = np.array([r["fitness"] for r in results])
            viable = fits[fits > 0]
            mean_fit = float(viable.mean()) if viable.size else float(fits.mean())

            # Track champion across all generations (by TRAIN fitness).
            if hall_of_fame is None or best["fitness"] > hall_of_fame["fitness"]:
                hall_of_fame = best

            print(f"Gen {gen:02d} | best_fit={best['fitness']:.3f} "
                  f"mean_fit={mean_fit:.3f} | "
                  f"champ train: Sortino={best['train']['sortino']:.2f} "
                  f"trades={best['train']['n_trades']} "
                  f"PF={best['train']['profit_factor']:.2f} "
                  f"DD={best['train']['max_dd']*100:.1f}% | "
                  f"holdout Sortino={best['holdout']['sortino']:.2f} "
                  f"trades={best['holdout']['n_trades']}")

            history.append({
                "gen": gen,
                "best_fitness": best["fitness"],
                "mean_fitness": mean_fit,
                "viable_count": int(viable.size),
                "champ_train_sortino": best["train"]["sortino"],
                "champ_train_trades": best["train"]["n_trades"],
                "champ_train_pf": best["train"]["profit_factor"],
                "champ_train_dd": best["train"]["max_dd"],
                "champ_holdout_sortino": best["holdout"]["sortino"],
                "champ_holdout_trades": best["holdout"]["n_trades"],
                "champ_holdout_pf": best["holdout"]["profit_factor"],
            })

            # ---- breed the next generation ----
            if gen < gens - 1:
                elite_genomes = [r["genome"] for r in elites]
                next_pop = list(elite_genomes)  # elitism: carry survivors forward
                # Fresh random immigrants for diversity.
                for _ in range(immigrants):
                    next_pop.append(random_genome(rng))
                # Fill the rest with crossover + mutation of two random elites.
                while len(next_pop) < pop_size:
                    pa, pb = rng.sample(elite_genomes, 2)
                    child = mutate(crossover(pa, pb, rng), rng, mutation_rate)
                    next_pop.append(child)
                population = next_pop
            else:
                final_results = results
    finally:
        if pool is not None:
            pool.shutdown()

    return hall_of_fame, final_results, history


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def fmt_metrics(m):
    return (f"Sortino={m['sortino']:6.2f}  Sharpe={m['sharpe']:6.2f}  "
            f"ret={m['slice_ret']*100:7.2f}%  DD={m['max_dd']*100:6.2f}%  "
            f"trades={m['n_trades']:3d}  win={m['win_rate']*100:5.1f}%  "
            f"PF={m['profit_factor']:5.2f}  P&L=${m['net_pnl']:9.2f}")


def champion_cli(g, csv_path="QQQ_5m.csv"):
    """Reproduce the champion via smc_intraday.py CLI."""
    parts = [f"python smc_intraday.py --csv {csv_path}"]
    parts.append(f"--swing-n {g['swing_n']}")
    parts.append(f"--min-fvg-pct {g['min_fvg_pct']:.4f}")
    parts.append(f"--rr-partial {g['rr_partial']:.2f}")
    parts.append(f"--rr-runner {g['rr_runner']:.2f}")
    parts.append(f"--stop-buf {g['stop_buf']:.2f}")
    parts.append(f"--atr-n {g['atr_n']}")
    parts.append(f"--ema-n {g['ema_n']}")
    parts.append(f"--max-hold-bars {g['max_hold_bars']}")
    parts.append(f"--max-bars-after-bos {g['max_bars_after_bos']}")
    parts.append(f"--risk {g['risk_per_trade']:.4f}")
    parts.append(f"--no-entries-before-min {g['no_entries_before_min']}")
    parts.append(f"--no-new-entries-after-min {g['no_new_entries_after_min']}")
    if not g["allow_short"]:
        parts.append("--no-short")
    if not g["use_pd_filter"]:
        parts.append("--no-pd-filter")
    if not g["use_trend_filter"]:
        parts.append("--no-trend-filter")
    if not g["use_cooldown"]:
        parts.append("--no-cooldown")
    if not g["use_session_confluence"]:
        parts.append("--no-session-confluence")
    if g["flatten_at_session_close"]:
        parts.append("--flatten-eod")
    return " ".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="QQQ_5m.csv")
    p.add_argument("--pop", type=int, default=100)
    p.add_argument("--elite", type=int, default=10)
    p.add_argument("--gens", type=int, default=15)
    p.add_argument("--immigrants", type=int, default=5)
    p.add_argument("--mutation-rate", type=float, default=0.2)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--min-trades", type=int, default=20)
    p.add_argument("--objective", default="sortino",
                   choices=["sortino", "sharpe", "cagr", "pf"])
    p.add_argument("--account", type=float, default=10000.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--jobs", type=int, default=1,
                   help="Parallel worker processes (1 = serial).")
    p.add_argument("--folds", type=int, default=1,
                   help="Walk-forward train folds for robust fitness "
                        "(1 = single-window; 3+ selects for consistency).")
    p.add_argument("--spread-penalty", type=float, default=0.5,
                   help="Weight on the std of per-fold scores (higher = "
                        "demand more consistency across sub-periods).")
    p.add_argument("--out-prefix", default="")
    args = p.parse_args()

    champ, final_pop, history = evolve(
        args.csv, pop_size=args.pop, elite=args.elite, gens=args.gens,
        immigrants=args.immigrants, mutation_rate=args.mutation_rate,
        train_frac=args.train_frac, min_trades=args.min_trades,
        objective=args.objective, account=args.account, seed=args.seed,
        jobs=args.jobs, folds=args.folds, spread_penalty=args.spread_penalty,
    )

    # The genome that happened to do best on HOLDOUT in the final population —
    # used only to expose the selection-bias gap, never to pick the champion.
    best_holdout = max(final_pop, key=lambda r: r["holdout"]["sortino"])

    print("\n" + "=" * 92)
    print("  EVOLUTION COMPLETE — CHAMPION (selected by TRAIN fitness only)")
    print("=" * 92)
    print("  Champion genome:")
    g = champ["genome"]
    for k in GENE_NAMES:
        print(f"    {k:<26}: {g[k]}")
    print()
    print("  Champion performance (the honesty panel):")
    print(f"    TRAIN  : {fmt_metrics(champ['train'])}")
    print(f"    HOLDOUT: {fmt_metrics(champ['holdout'])}")
    print()
    print("  Overfitting check — gap between train and holdout Sortino:")
    decay = champ["train"]["sortino"] - champ["holdout"]["sortino"]
    print(f"    train Sortino {champ['train']['sortino']:.2f}  ->  "
          f"holdout {champ['holdout']['sortino']:.2f}   (decay {decay:+.2f})")
    print(f"    For reference, the final-population genome that fit the HOLDOUT")
    print(f"    best scored holdout Sortino {best_holdout['holdout']['sortino']:.2f} — the")
    print(f"    distance from the champion's holdout is the price of selection bias.")
    print("=" * 92)
    print("\n  Reproduce the champion:")
    print("    " + champion_cli(g, args.csv))
    print("=" * 92)

    # Persist artifacts.
    champ_path = f"{args.out_prefix}champion_genome.json"
    with open(champ_path, "w") as f:
        json.dump({
            "genome": g,
            "train": champ["train"],
            "holdout": champ["holdout"],
            "config": {
                "pop": args.pop, "elite": args.elite, "gens": args.gens,
                "objective": args.objective, "train_frac": args.train_frac,
                "min_trades": args.min_trades, "seed": args.seed,
            },
            "cli": champion_cli(g, args.csv),
        }, f, indent=2, default=str)
    hist_path = f"{args.out_prefix}evolution_history.csv"
    pd.DataFrame(history).to_csv(hist_path, index=False)
    print(f"\nSaved champion -> {champ_path}")
    print(f"Saved learning curve -> {hist_path}")


if __name__ == "__main__":
    main()
