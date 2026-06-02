# SMC trading research + live paper tracker

Research code exploring Smart-Money-Concepts (swings, break-of-structure, fair
value gaps) strategies on Nasdaq-100 data, with genetic-algorithm optimization,
walk-forward robustness testing, regime analysis, and a frozen live paper
tracker. **Research/education only — not financial advice.**

## The headline finding

After extensive testing across bulls and real bears (2008/2018/2020/2022), the
durable edge is **risk reduction, not return-beating**. The winning config:

> **SMC long-only, prudently levered.** Goes to cash in downtrends (sidestepped
> every major crash), and at ~8% risk/trade beats buy-and-hold on the full
> 20-year NDX (CAGR 17.1% vs 15.7%) with **-22% max drawdown vs -54%** and a
> higher Sharpe (0.87 vs 0.76). At a more conservative 5% risk it trades
> ~-15% drawdown for a slightly lower return. Main real risk: overnight gaps at
> leverage (daily bars understate them).

## Live paper tracking (runs in the cloud, no machine needed)

`paper_trader.py` freezes the winning config and runs it forward on fresh daily
data, accumulating a real out-of-sample track record.

```bash
python paper_trader.py            # fetch latest data, update paper account, report
python paper_trader.py --report   # show the journal without fetching
python paper_trader.py --reset    # start a fresh paper account today
```

State lives in `paper_state.json`; a readable journal is written to
`PAPER_LOG.md`. All performance is re-derived from data each run, so it can't
silently drift.

### Automated daily runs via GitHub Actions

`.github/workflows/paper-trade.yml` runs the tracker every US trading day at
~19:10 ET in GitHub's cloud and commits the updated state back — **your computer
does not need to be on.** To enable it:

1. Create an empty repo on GitHub (e.g. `smc-paper-trader`).
2. Push this project (see commands the assistant printed, or):
   ```bash
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
3. On GitHub: **Settings → Actions → General → Workflow permissions →
   Read and write permissions** (so the job can commit state back).
4. The schedule starts automatically; or trigger a run from the **Actions** tab
   (**paper-trade → Run workflow**). Read the daily result in `PAPER_LOG.md`.

## Key files

| File | What it does |
|------|--------------|
| `smc_strategy_v2.py` | Daily SMC long/short engine (swings, BOS, FVG, partial+trail) |
| `smc_intraday.py` | Intraday (5m/1h) SMC engine with session levels |
| `evolve_traders.py` / `evolve_robust.py` | Genetic optimizers (single-window / rotating walk-forward) |
| `ensemble.py` / `meta_ensemble.py` | Combine champions / diversify across seeds |
| `regime_test.py` | Performance attributed across real bull/bear regimes |
| `smc_crisis.py` / `crisis_hybrid.py` | Trend-following crisis-alpha shorts + hybrid |
| `leverage_long.py` / `leverage_test.py` | Prudent-leverage analysis |
| `paper_trader.py` | Frozen-config forward paper tracker |

Install deps: `pip install -r requirements.txt`
