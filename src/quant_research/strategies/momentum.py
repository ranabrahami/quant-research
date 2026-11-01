"""Cross-sectional momentum: buy recent winners, sell recent losers.

The signal at month-end t is the compounded return over months t-L+1 .. t-S,
i.e. the past L months skipping the most recent S. Skipping a month avoids the
short-term reversal that otherwise contaminates the signal (Jegadeesh 1990).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def trailing_return(returns: pd.DataFrame, lookback: int = 12, skip: int = 1) -> pd.DataFrame:
    window = lookback - skip
    logs = np.log1p(returns).rolling(window, min_periods=window).sum().shift(skip)
    return np.expm1(logs)


def long_short(signal: pd.DataFrame, fraction: float = 0.2, min_assets: int = 10) -> pd.DataFrame:
    """Equal weights: +1/k on the top `fraction` by signal, -1/k on the bottom; rows with too few assets dropped."""
    rows = {}
    for t, s in signal.iterrows():
        s = s.dropna()
        if len(s) < min_assets:
            continue
        k = max(int(round(len(s) * fraction)), 1)
        ranked = s.sort_values()
        w = pd.Series(0.0, index=signal.columns)
        w[ranked.index[-k:]] = 1 / k
        w[ranked.index[:k]] = -1 / k
        rows[t] = w
    return pd.DataFrame(rows).T
