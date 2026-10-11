# SMC trading research + live paper tracker

Research code exploring Smart-Money-Concepts (swings, break-of-structure, fair
value gaps) strategies on Nasdaq-100 data: genetic optimization, walk-forward
testing, regime analysis, and cloud-scheduled paper tracking.
**Research/education only — not financial advice.**

## Current strategy: `smc_v3.py` (honest rebuild, Oct 2026)

**Correction to earlier results.** The v1/v2 SMC engines (and the intraday
engine) had a look-ahead bias: an entry on bar *t* was decided using bar *t*'s
own close and required bar *t*'s low to have stayed above the FVG zone —
information only available after the bar closed. That silently removed losing
fills. With identical settings, fixing it cut the Sharpe from 0.88 to 0.38 and
the win rate from 80% to 58%. Earlier headline claims (Sharpe 0.85+, "8%
risk/trade beats buy & hold") came from that bias and are withdrawn.

What survived honest testing (decide at the close, trade at the next open,
gap-aware stops, 5 bps costs), ^NDX daily 2007–2026:

| | CAGR | Sharpe | Max drawdown |
|---|---|---|---|
| Buy & hold ^NDX (price) | 15.4% | 0.74 | −53.7% |
| SMC FVG entries, average of 72 settings | 0.1% | 0.05 | −7.6% |
| **v3 regime core** | **11.2%** | **0.82** | **−21.3%** |
| v3 core, 2× cap, futures-like financing | 15.7% | 0.77 | −33.5% |

**v3 regime core** = an ensemble of 8 standard trend rules (SMA and EMA at
100/150/200/250 days). Exposure = share of rules that are bullish, scaled toward
20% annualised volatility, capped at 1.0× (no leverage — tradable with a plain
ETF). Idle cash earns the 13-week T-bill rate. Nothing is optimised; nearby
settings give Sharpe 0.79–0.86.

- Real ≥15% declines: average −12.1% vs −28.0% for buy & hold (2008: −16.7% vs −53.7%).
- Trade-off: it lags in strong bull years and loses a little in whipsaw years
  (2011, 2015–16). It does not beat buy & hold on raw return unless levered,
  and leverage only helps with cheap (futures-like) financing — not CFD rates.
- The SMC entry engine is kept, honestly rebuilt, as an optional sleeve
  (weight 0 by default) and SMC structure is shown as live context.

```bash
python smc_v3.py report    # full validation
python smc_v3.py live      # today's target exposure + SMC structure context
python smc_v3.py track     # update the forward paper account (PAPER_LOG_V3.md)
python smc_v3.py smc       # the honest SMC sleeve alone, with trades
```

## Paper tracking (GitHub Actions, no machine needed)

`.github/workflows/paper-trade.yml` runs every US trading day at ~19:10 ET:

- `smc_v3.py track` → `PAPER_LOG_V3.md` (v3, live since 2026-10-12)

The legacy v2 tracker (`paper_trader.py`, `PAPER_LOG.md`) ran 2026-06-03 →
2026-10-09 and was **retired on 2026-10-12** because its engine had the
look-ahead bias above. Its journal is kept as a frozen record.

## Key files

| File | What it does |
|------|--------------|
| `smc_v3.py` | **Current strategy**: regime core + honest SMC sleeve + live/track |
| `smc_strategy_v2.py` | Legacy daily SMC engine (look-ahead bias — kept for comparison) |
| `smc_intraday.py` | Legacy intraday SMC engine (same bias) |
| `evolve_traders.py` / `evolve_robust.py` | Genetic optimizers (built on the legacy engine) |
| `ensemble.py` / `meta_ensemble.py` | Champion ensembles (legacy engine) |
| `regime_test.py` | Real bull/bear regime attribution helpers |
| `smc_crisis.py` / `crisis_hybrid.py` | Trend-following crisis shorts experiments |
| `paper_trader.py` | Retired v2 forward tracker |

Data: `NDX.csv` (^NDX daily), `IRX.csv` (13-week T-bill yield). Install:
`pip install -r requirements.txt`
