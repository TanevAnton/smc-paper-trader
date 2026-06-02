"""
paper_trader.py — forward paper-trading harness for the frozen winning config.

WHAT IT DOES
------------
Freezes the strategy that won the whole study — SMC long-only with prudent
leverage — and runs it FORWARD on freshly downloaded daily data. Every run it:

  1. Downloads the latest daily bars (yfinance; falls back to local NDX.csv).
  2. Runs the exact frozen strategy and reads off the LIVE state: am I in a
     position right now? what are my stop / target? did anything fill today?
  3. Keeps a paper account that started at go-live, so you accumulate a REAL
     out-of-sample track record — the only honest test left after backtesting.
  4. Prints today's action and compares the forward record to the backtest's
     expectation (win rate, expectancy, trade frequency) so drift is visible.

State lives in paper_state.json (just the go-live anchor + a daily journal);
all performance is RE-DERIVED from data each run, so it can never silently
drift out of sync with the strategy.

IMPORTANT HONESTY NOTES
  * This is PAPER trading — no broker, no real orders. It tells you what the
    strategy would do; you decide whether to mirror it.
  * Entries fill at the NEXT session's open (no look-ahead), so a brand-new
    signal shows up the day AFTER it triggers. Positions already on are shown
    live with their current stop/target.
  * The backtest's drawdowns assume stops fill at the stop price; real gaps can
    be worse, especially with leverage. Size accordingly.

USAGE
-----
    python paper_trader.py                      # update + status report
    python paper_trader.py --report             # show journal without fetching
    python paper_trader.py --reset              # start a fresh paper account today
    python paper_trader.py --ticker QQQ --risk 0.05
"""

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

from smc_strategy_v2 import backtest_v2

STATE_FILE = "paper_state.json"

# --- The frozen winning configuration (do not tune; that's the point) ---------
FROZEN = dict(
    swing_n=3,
    rr_runner=5.0,
    rr_partial=1.0,
    max_bars_after_bos=20,
    min_fvg_pct=0.001,
    allow_short=False,        # long-only: in CASH during bears (sidesteps crashes)
    use_trend_filter=True,    # EMA200 regime
    cost_bps=5.0,
)
ACCOUNT0 = 10000.0


# ----------------------------------------------------------------------------
def fetch_daily(ticker, years=4):
    """Download daily OHLC; return df[date,open,high,low,close]. Falls back to
    local NDX.csv if the download fails (offline / rate-limited)."""
    end = dt.date.today()
    start = dt.date(end.year - years, end.month, end.day)
    try:
        import yfinance as yf
        raw = yf.download(ticker, start=start.isoformat(), end=end.isoformat(),
                          auto_adjust=False, progress=False)
        if raw is None or raw.empty:
            raise RuntimeError("empty download")
        if hasattr(raw.columns, "nlevels") and raw.columns.nlevels > 1:
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.reset_index()
        df = pd.DataFrame({
            "date": pd.to_datetime(raw["Date"]),
            "open": raw["Open"].astype(float),
            "high": raw["High"].astype(float),
            "low": raw["Low"].astype(float),
            "close": raw["Close"].astype(float),
        }).dropna().sort_values("date").reset_index(drop=True)
        return df, f"yfinance:{ticker}"
    except Exception as e:
        if os.path.exists("NDX.csv"):
            from smc_strategy_v2 import load_ohlc
            df = load_ohlc("NDX.csv")
            return df, f"LOCAL NDX.csv (download failed: {e})"
        sys.exit(f"ERROR: could not download {ticker} and no local NDX.csv: {e}")


def load_state():
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE))
    return None


def save_state(state):
    json.dump(state, open(STATE_FILE, "w"), indent=2, default=str)


def write_markdown_log(state):
    """Human-readable journal committed to the repo so you can read the forward
    record straight from GitHub without running anything."""
    lines = [
        f"# Paper-trading journal — {state['ticker']}",
        "",
        f"- **Strategy:** frozen SMC long-only "
        f"(swing 3, RR 1/5, EMA200 filter, no shorts)",
        f"- **Go-live:** {state['go_live_date']}  "
        f"| **Risk/trade:** {state.get('risk', 0.05)*100:.0f}%",
        f"- **Latest update:** {state['journal'][-1]['date'] if state.get('journal') else 'n/a'}",
        "",
        "| Date | Last bar | Position | Paper equity | Action |",
        "|------|----------|----------|--------------|--------|",
    ]
    for row in state.get("journal", []):
        lines.append(f"| {row['date']} | {row.get('last_bar','')} | "
                     f"{row['position']} | ${row['paper_equity']:,.2f} | {row['action']} |")
    lines += ["", "_Auto-updated by the paper-trade workflow. Paper trading only — "
              "not financial advice._"]
    with open("PAPER_LOG.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def forward_metrics(trades, go_live):
    """Stats restricted to trades that CLOSED on/after the go-live date."""
    if not len(trades):
        return None
    t = trades.copy()
    t["exit_date"] = pd.to_datetime(t["exit_date"]).dt.date
    fwd = t[t["exit_date"] >= go_live]
    if not len(fwd):
        return {"n": 0}
    wins = (fwd["pnl_$"] > 0).sum()
    return {
        "n": len(fwd),
        "win_rate": wins / len(fwd),
        "pnl_sum": float(fwd["pnl_$"].sum()),
        "avg_r": float(fwd["r_total"].mean()) if "r_total" in fwd else float("nan"),
        "last_exits": fwd.tail(3).to_dict("records"),
    }


def backtest_expectation(trades):
    if not len(trades):
        return None
    wins = (trades["pnl_$"] > 0).sum()
    return {
        "win_rate": wins / len(trades),
        "expectancy": float(trades["pnl_$"].mean()),
        "n": len(trades),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="^NDX")
    p.add_argument("--risk", type=float, default=0.05,
                   help="Risk per trade (0.05 = prudent config, about -15 pct DD).")
    p.add_argument("--report", action="store_true",
                   help="Show the saved journal without downloading.")
    p.add_argument("--reset", action="store_true",
                   help="Start a fresh paper account dated today.")
    args = p.parse_args()

    today = dt.date.today()

    if args.report:
        state = load_state()
        if not state:
            sys.exit("No paper_state.json yet — run once without --report first.")
        print(f"PAPER JOURNAL — {state['ticker']}  (go-live {state['go_live_date']})")
        print("=" * 78)
        for row in state.get("journal", [])[-20:]:
            print(f"  {row['date']}  eq=${row['paper_equity']:>10,.2f}  "
                  f"{row['position']:<14}  {row['action']}")
        print("=" * 78)
        return

    df, src = fetch_daily(args.ticker)
    last_date = df["date"].iloc[-1].date()

    trades, eq, open_state = backtest_v2(
        df, risk_per_trade=args.risk, account=ACCOUNT0,
        return_open_state=True, **FROZEN)
    eq = np.asarray(eq, float)

    # Load / init state (anchor the go-live date).
    state = load_state()
    if state is None or args.reset:
        state = {
            "ticker": args.ticker,
            "go_live_date": str(today),
            "risk": args.risk,
            "config": FROZEN,
            "journal": [],
        }
        print(f"Initialized new paper account on {today} "
              f"({'reset' if args.reset else 'first run'}).")
    go_live = pd.to_datetime(state["go_live_date"]).date()

    # Forward paper equity = fresh $10k compounded by the strategy's return
    # since go-live (clean, normalized; can't drift).
    dates = df["date"].dt.date.to_numpy()
    gl_idx = int(np.argmax(dates >= go_live))
    if dates[gl_idx] < go_live:        # go-live is in the future of the data
        gl_idx = len(dates) - 1
    fwd_factor = eq[-1] / eq[gl_idx] if eq[gl_idx] > 0 else 1.0
    paper_equity = ACCOUNT0 * fwd_factor

    # Decide today's action from the trade log + open state.
    exits_today = []
    if len(trades):
        te = trades.copy()
        te["exit_date"] = pd.to_datetime(te["exit_date"]).dt.date
        exits_today = te[te["exit_date"] == last_date].to_dict("records")

    if open_state is not None:
        entered = pd.to_datetime(open_state["entry_date"]).date()
        if entered == last_date:
            action = f"ENTERED LONG @ {open_state['entry_price']:.2f}"
        else:
            action = (f"HOLDING LONG (day {open_state['bars_held']}, "
                      f"entered {open_state['entry_date']})")
        position_str = "LONG"
    elif exits_today:
        r = exits_today[-1]
        action = f"EXITED ({r['exit_reason']}) @ {r['exit']:.2f}, P&L ${r['pnl_$']:.2f}"
        position_str = "FLAT (just exited)"
    else:
        action = "no fill — flat, watching for setup"
        position_str = "FLAT"

    # ---- Report ----
    print("=" * 78)
    print(f"  PAPER TRADER — frozen SMC long-only  |  {args.ticker}  |  {today}")
    print("=" * 78)
    print(f"  Data source        : {src}")
    print(f"  Latest bar         : {last_date}  close {df['close'].iloc[-1]:,.2f}")
    print(f"  Go-live            : {state['go_live_date']}  "
          f"(risk {state.get('risk', args.risk)*100:.0f}%/trade)")
    print(f"  TODAY'S ACTION     : {action}")
    print("  " + "-" * 74)
    if open_state is not None:
        gain = open_state["unrealized_$"]
        print(f"  CURRENT POSITION   : LONG, entered {open_state['entry_date']} "
              f"@ {open_state['entry_price']:.2f}")
        print(f"    Stop  (trailing) : {open_state['stop']:.2f}   "
              f"Target: {open_state['target']:.2f}   "
              f"Partial taken: {open_state['partial_taken']}")
        print(f"    Unrealized       : ${gain:,.2f}   "
              f"({'+' if gain>=0 else ''}{gain/paper_equity*100:.2f}% of paper acct)")
    else:
        print(f"  CURRENT POSITION   : FLAT (in cash)")
    print("  " + "-" * 74)

    fwd = forward_metrics(trades, go_live)
    exp = backtest_expectation(trades)
    print(f"  PAPER ACCOUNT (since go-live)")
    print(f"    Equity           : ${paper_equity:,.2f}   "
          f"({(fwd_factor-1)*100:+.2f}% on $10,000)")
    if fwd and fwd.get("n", 0) > 0:
        print(f"    Forward trades   : {fwd['n']}   win rate {fwd['win_rate']*100:.0f}%   "
              f"net ${fwd['pnl_sum']:,.2f}   avg {fwd['avg_r']:+.2f}R")
    else:
        print(f"    Forward trades   : 0 closed yet (track record starts accumulating)")
    if exp:
        print(f"  BACKTEST EXPECTATION (full history, the bar to compare against)")
        print(f"    Win rate {exp['win_rate']*100:.0f}%   expectancy ${exp['expectancy']:.2f}/trade   "
              f"({exp['n']} trades over ~{len(df)/252:.0f}y)")
    print("=" * 78)
    print("  Entries fill at the NEXT session open; brand-new signals appear the")
    print("  following day. Mirror the CURRENT POSITION's stop/target if trading live.")
    print("=" * 78)

    # Append to journal (one row per calendar day, de-duplicated).
    journal = state.get("journal", [])
    journal = [r for r in journal if r["date"] != str(today)]
    journal.append({
        "date": str(today),
        "last_bar": str(last_date),
        "position": position_str,
        "action": action,
        "paper_equity": round(paper_equity, 2),
    })
    state["journal"] = journal
    state["risk"] = args.risk
    save_state(state)
    write_markdown_log(state)
    print(f"Logged to {STATE_FILE} and PAPER_LOG.md ({len(journal)} entries).")


if __name__ == "__main__":
    main()
