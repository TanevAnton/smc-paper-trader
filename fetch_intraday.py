"""
fetch_intraday.py — pull intraday OHLCV from yfinance, save in a shape the
intraday backtester can consume directly (Datetime + OHLCV columns).

yfinance window limits (free):
  - 1m  : last ~7 days
  - 5m  : last ~60 days
  - 15m : last ~60 days
  - 1h  : last ~730 days

Usage:
    python fetch_intraday.py                              # QQQ, 5m, 60d
    python fetch_intraday.py --ticker QQQ --interval 1m --period 7d
    python fetch_intraday.py --ticker NQ=F --interval 5m --period 60d
"""

import argparse
import sys

import yfinance as yf


def fetch(ticker: str, interval: str, period: str):
    # prepost=True keeps after-hours bars — useful for futures-like context;
    # for QQQ regular-hours analysis we'll filter to RTH in the backtester.
    df = yf.download(
        tickers=ticker,
        interval=interval,
        period=period,
        auto_adjust=False,
        prepost=False,
        progress=False,
    )
    if df is None or df.empty:
        sys.exit(f"ERROR: no data returned for {ticker} interval={interval} period={period}")

    # yfinance multi-index columns when given a single ticker as list — flatten.
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    # First column is either "Datetime" or "Date"; normalize to "Datetime".
    first_col = df.columns[0]
    if first_col != "Datetime":
        df = df.rename(columns={first_col: "Datetime"})

    keep = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
    for col in keep:
        if col not in df.columns:
            sys.exit(f"ERROR: column {col!r} missing from yfinance response. Have: {list(df.columns)}")
    df = df[keep]

    # Preserve timezone information (typically America/New_York for US tickers).
    df["Datetime"] = df["Datetime"].astype(str)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="QQQ")
    p.add_argument("--interval", default="5m",
                   choices=["1m", "2m", "5m", "15m", "30m", "60m", "1h"])
    p.add_argument("--period", default="60d",
                   help="yfinance period string, e.g. 7d, 30d, 60d, 1mo, 3mo, 730d")
    p.add_argument("--out", help="Output CSV path (default: <ticker>_<interval>.csv)")
    args = p.parse_args()

    out = args.out or f"{args.ticker.replace('^', '').replace('=', '_')}_{args.interval}.csv"
    print(f"Downloading {args.ticker}  interval={args.interval}  period={args.period} ...")
    df = fetch(args.ticker, args.interval, args.period)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} bars to {out}")
    print(f"Range: {df['Datetime'].iloc[0]}  ->  {df['Datetime'].iloc[-1]}")
    print(f"First close: {df['Close'].iloc[0]:.2f}   "
          f"Last close: {df['Close'].iloc[-1]:.2f}")


if __name__ == "__main__":
    main()
