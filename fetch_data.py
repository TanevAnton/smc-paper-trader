"""
fetch_data.py — download historical Nasdaq data via yfinance and save it
in the same CSV shape backtester.py / cfd_simulator.py already expect
(Yahoo Finance: Date, Open, High, Low, Close, Adj Close, Volume).

Usage:
    python fetch_data.py                          # default: ^NDX, 20 years
    python fetch_data.py --ticker QQQ --years 15
    python fetch_data.py --ticker ^NDX --start 2005-01-01 --end 2026-05-22
"""

import argparse
import sys
from datetime import date

import yfinance as yf


def fetch(ticker: str, start: str, end: str) -> "pd.DataFrame":
    df = yf.download(ticker, start=start, end=end, auto_adjust=False,
                     progress=False)
    if df is None or df.empty:
        sys.exit(f"ERROR: no data returned for {ticker} {start}..{end}")

    # yfinance returns a MultiIndex when ticker is a list; flatten if so.
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    # Match Yahoo CSV column order exactly.
    keep = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in keep:
        if col not in df.columns:
            # auto_adjust=False should give us all of these; bail if not.
            sys.exit(f"ERROR: column {col!r} missing from yfinance response.")
    df = df[keep]
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="^NDX",
                   help="Yahoo ticker (default ^NDX = Nasdaq-100 index). "
                        "Use QQQ for the tradeable ETF.")
    p.add_argument("--years", type=int, default=20,
                   help="How many years back from today (ignored if --start set).")
    p.add_argument("--start", help="YYYY-MM-DD start date.")
    p.add_argument("--end", help="YYYY-MM-DD end date (default: today).")
    p.add_argument("--out", help="Output CSV path (default: <ticker>.csv).")
    args = p.parse_args()

    end = args.end or date.today().isoformat()
    if args.start:
        start = args.start
    else:
        start = f"{date.today().year - args.years}-{date.today().strftime('%m-%d')}"

    out = args.out or f"{args.ticker.lstrip('^')}.csv"
    print(f"Downloading {args.ticker}  {start} -> {end} ...")
    df = fetch(args.ticker, start, end)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    print(f"Range: {df['Date'].iloc[0]} -> {df['Date'].iloc[-1]}")
    print(f"First close: {df['Close'].iloc[0]:.2f}   "
          f"Last close: {df['Close'].iloc[-1]:.2f}")


if __name__ == "__main__":
    main()
