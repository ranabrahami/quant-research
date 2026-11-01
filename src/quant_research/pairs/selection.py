"""Choosing pairs in a formation window.

Two methods, as compared by Gatev, Goetzmann and Rouwenhorst (2006) and later
work:

- distance: normalize each price to 1 at the start of the window and rank
  pairs by the sum of squared differences between the two normalized paths.
  No model; pairs are simply the ones that moved together.
- cointegration (Engle-Granger): regress log price A on log price B; if the
  residual (the spread) is stationary, the two share a long-run relation and
  deviations should revert. Tested with an ADF regression on the residual
  (one lag) and MacKinnon p-values for two variables.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.adfvalues import mackinnonp


def normalized(prices: pd.DataFrame) -> pd.DataFrame:
    return prices / prices.iloc[0]


def candidate_pairs(groups: dict[str, list[str]] | None, columns: list[str]) -> list[tuple[str, str]]:
    """All pairs within each group, or all pairs of `columns` when there are no groups."""
    if groups is None:
        return list(combinations(columns, 2))
    return [p for members in groups.values() for p in combinations([m for m in members if m in columns], 2)]


def distances(prices: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.Series:
    n = normalized(prices)
    return pd.Series({(a, b): float(((n[a] - n[b]) ** 2).sum()) for a, b in pairs}).sort_values()


def engle_granger(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float, float]:
    """(p-value, ADF t-stat, hedge ratio b, intercept a) for y = a + b x + e with e tested for a unit root."""
    X = np.column_stack([np.ones_like(x), x])
    (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - a - b * x
    de = np.diff(e)
    Z = np.column_stack([e[1:-1], de[:-1]])                 # gamma e_{t-1} + phi de_{t-1}
    target = de[1:]
    coef, *_ = np.linalg.lstsq(Z, target, rcond=None)
    resid = target - Z @ coef
    s2 = resid @ resid / (len(target) - Z.shape[1])
    se = np.sqrt(s2 * np.linalg.inv(Z.T @ Z)[0, 0])
    t = coef[0] / se
    return float(mackinnonp(t, regression="c", N=2)), float(t), float(b), float(a)


def cointegration(prices: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Engle-Granger on log prices for each pair (A regressed on B), sorted by p-value."""
    logs = np.log(prices)
    rows = []
    for a, b in pairs:
        p, t, beta, alpha = engle_granger(logs[a].to_numpy(), logs[b].to_numpy())
        rows.append({"pair": (a, b), "p": p, "t": t, "beta": beta, "alpha": alpha})
    return pd.DataFrame(rows).set_index("pair").sort_values("p")
