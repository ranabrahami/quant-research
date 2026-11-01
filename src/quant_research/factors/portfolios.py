"""Turn a cross-sectional signal into a long-short portfolio return series.

At each rebalance date the companies are sorted by the signal; the portfolio
holds the top fraction long and the bottom fraction short, equally weighted,
until the next rebalance. Monthly returns are the average member return
(i.e. equal weights reset monthly). The first holding month is the month
after the signal date, so no return before the signal is used.

With `holding_months`, each rebalance instead forms a cohort held for that
many months, and each month's return averages every live cohort
(Jegadeesh-Titman overlapping portfolios). This matches the portfolio to a
signal whose payoff arrives slowly.
"""

from __future__ import annotations

import pandas as pd


def holdings(signals: pd.DataFrame, signal: str, fraction: float = 1 / 3) -> pd.DataFrame:
    """Rows of (date, ticker, side) with side +1 long, -1 short."""
    rows = []
    for d, g in signals.dropna(subset=[signal]).groupby("date"):
        g = g.sort_values(signal)
        k = max(int(len(g) * fraction), 1)
        rows += [{"date": d, "ticker": t, "side": -1} for t in g["ticker"].iloc[:k]]
        rows += [{"date": d, "ticker": t, "side": 1} for t in g["ticker"].iloc[-k:]]
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    return out


def long_short_returns(signals: pd.DataFrame, returns: pd.DataFrame, signal: str,
                       fraction: float = 1 / 3, holding_months: int | None = None) -> pd.DataFrame:
    """Monthly long, short and long-minus-short returns. `returns`: month-end index x ticker columns.

    holding_months=None holds each portfolio until the next rebalance; an integer holds
    overlapping cohorts for that many months and averages them.
    """
    h = holdings(signals, signal, fraction)
    dates = sorted(h["date"].unique())
    rows = []
    for i, d in enumerate(dates):
        if holding_months is None:
            stop = dates[i + 1] if i + 1 < len(dates) else returns.index.max()
        else:
            stop = d + pd.offsets.MonthEnd(holding_months)
        months = returns.index[(returns.index > d) & (returns.index <= stop)]
        members = h[h["date"] == d]
        longs = members.loc[members["side"] == 1, "ticker"]
        shorts = members.loc[members["side"] == -1, "ticker"]
        for m in months:
            lr, sr = returns.loc[m, longs].mean(), returns.loc[m, shorts].mean()
            rows.append({"date": m, "cohort": d, "long": lr, "short": sr, "long_short": lr - sr,
                         "universe": returns.loc[m, members["ticker"]].mean()})
    out = pd.DataFrame(rows).groupby("date")[["long", "short", "long_short", "universe"]].mean()
    out["cohorts"] = pd.DataFrame(rows).groupby("date").size()
    return out


def event_time(signals: pd.DataFrame, returns: pd.DataFrame, signal: str, fraction: float = 1 / 3,
               max_months: int = 12) -> pd.DataFrame:
    """Long-minus-short return in each month k = 1..max_months after every signal date.

    Shows when a signal pays off: one row per (signal date, k) that has a return.
    """
    h = holdings(signals, signal, fraction)
    rows = []
    for d, g in h.groupby("date"):
        longs, shorts = g.loc[g["side"] == 1, "ticker"], g.loc[g["side"] == -1, "ticker"]
        for k in range(1, max_months + 1):
            m = d + pd.offsets.MonthEnd(k)
            if m in returns.index:
                rows.append({"date": d, "k": k, "month": m,
                             "long_short": returns.loc[m, longs].mean() - returns.loc[m, shorts].mean()})
    return pd.DataFrame(rows)
