"""Monthly total returns from Yahoo Finance (dividend-adjusted closes), cached."""

from __future__ import annotations

import time
from pathlib import Path

import warnings

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", module="yfinance")       # its pandas deprecation noise

CACHE = Path(__file__).resolve().parents[3] / "data" / "raw" / "prices"
MAX_AGE_HOURS = 24


def daily_prices(ticker: str, start: str = "1985-01-01", refresh: bool = False) -> pd.Series:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{ticker.replace('^', '_')}.csv"
    if not refresh and path.exists() and time.time() - path.stat().st_mtime < MAX_AGE_HOURS * 3600:
        s = pd.read_csv(path, index_col="date", parse_dates=["date"])["adj_close"]
        if s.index.min() <= pd.Timestamp(start) + pd.Timedelta(days=10) or len(s) > 100:
            return s
    hist = yf.Ticker(ticker).history(start=start, auto_adjust=False)
    if hist.empty:
        raise RuntimeError(f"No price history for {ticker}")
    s = pd.Series(hist["Adj Close"].to_numpy(), index=pd.DatetimeIndex(hist.index.tz_localize(None).normalize(),
                                                                         name="date"), name="adj_close")
    s.to_frame().to_csv(path)
    return s


def monthly_returns(prices: pd.Series) -> pd.Series:
    """Month-end to month-end returns, indexed by month-end (matching the French data)."""
    month_end = prices.resample("ME").last()
    return month_end.pct_change().dropna()


def monthly_total_returns(ticker: str, start: str = "1985-01-01", refresh: bool = False) -> pd.Series:
    r = monthly_returns(daily_prices(ticker, start, refresh))
    # The current month is unfinished: its month-end label lies in the future.
    return r[r.index <= pd.Timestamp.today().normalize()].rename(ticker)
