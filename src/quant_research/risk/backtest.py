"""Is a VaR model right? Count and test its exceptions (days the loss beat the VaR).

- Kupiec (1995) proportion of failures: is the exception rate equal to 1 - level?
- Christoffersen (1998) independence: are exceptions spread out, or do they
  cluster (an exception today makes one tomorrow more likely)?
- Basel traffic light: exceptions per 250 days at 99%; 0-4 green, 5-9 yellow,
  10+ red (the model is rejected and capital multipliers rise).
- Acerbi and Szekely (2014) Z2 for Expected Shortfall: 0 if ES is right,
  positive if the average tail loss is larger than forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def exceptions(r: pd.Series, var: pd.Series) -> pd.Series:
    data = pd.concat([r.rename("r"), var.rename("var")], axis=1).dropna()
    return (data["r"] < -data["var"]).rename("exception")


def _loglik(x: int, n: int, p: float) -> float:
    if p in (0.0, 1.0):
        return 0.0 if (x == 0 and p == 0.0) or (x == n and p == 1.0) else -np.inf
    return (n - x) * np.log(1 - p) + x * np.log(p)


def kupiec(exc: pd.Series, p: float = 0.01) -> tuple[float, float]:
    """Likelihood-ratio statistic and p-value for an exception rate of p."""
    n, x = len(exc), int(exc.sum())
    lr = -2 * (_loglik(x, n, p) - _loglik(x, n, x / n))
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen(exc: pd.Series) -> tuple[float, float]:
    """Likelihood-ratio statistic and p-value for exceptions being independent day to day."""
    e = exc.astype(int).to_numpy()
    prev, curr = e[:-1], e[1:]
    n00, n01 = int(((prev == 0) & (curr == 0)).sum()), int(((prev == 0) & (curr == 1)).sum())
    n10, n11 = int(((prev == 1) & (curr == 0)).sum()), int(((prev == 1) & (curr == 1)).sum())
    p01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    p11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    p = (n01 + n11) / len(curr)
    unrestricted = _loglik(n01, n00 + n01, p01) + _loglik(n11, n10 + n11, p11)
    lr = -2 * (_loglik(n01 + n11, len(curr), p) - unrestricted)
    return float(lr), float(stats.chi2.sf(lr, 1))


def basel_zones(exc: pd.Series, block: int = 250) -> pd.Series:
    """Traffic-light zone for each consecutive block of `block` days."""
    counts = [int(exc.iloc[i:i + block].sum()) for i in range(0, len(exc) - block + 1, block)]
    return pd.Series(pd.cut(counts, [-1, 4, 9, np.inf], labels=["green", "yellow", "red"]),
                     index=exc.index[::block][:len(counts)])


def es_z2(r: pd.Series, var: pd.Series, es: pd.Series, p: float = 0.01) -> float:
    """Acerbi-Szekely Z2: mean of loss x exception / (p ES) minus 1. Zero if ES is right."""
    data = pd.concat([r.rename("r"), var.rename("var"), es.rename("es")], axis=1).dropna()
    hit = data["r"] < -data["var"]
    return float((-data["r"] * hit / (p * data["es"])).mean() - 1)


def report(r: pd.Series, var: pd.Series, es: pd.Series | None = None, level: float = 0.99) -> pd.Series:
    p = 1 - level
    exc = exceptions(r, var)
    zones = basel_zones(exc)
    out = {"days": len(exc), "exceptions": int(exc.sum()), "rate": exc.mean(), "expected rate": p,
           "Kupiec p": kupiec(exc, p)[1], "independence p": christoffersen(exc)[1],
           "red years": int((zones == "red").sum()), "yellow years": int((zones == "yellow").sum())}
    if es is not None:
        out["ES Z2"] = es_z2(r, var, es, p)
    return pd.Series(out)
