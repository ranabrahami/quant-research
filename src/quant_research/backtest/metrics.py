"""Performance statistics, with uncertainty attached.

A Sharpe ratio from a few hundred months is an estimate with a wide error
band, and the best of many tried variants is biased upward. Alongside the
usual numbers this module gives a block-bootstrap confidence interval and the
deflated Sharpe ratio of Bailey and Lopez de Prado (2014), which asks whether
the chosen variant beats what the best of N unskilled variants would show.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649


def max_drawdown(r: pd.Series) -> float:
    wealth = (1 + r).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def summary(r: pd.Series, periods: int = 12) -> pd.Series:
    r = r.dropna()
    sr = r.mean() / r.std()
    return pd.Series({
        "start": r.index.min().strftime("%Y-%m"), "end": r.index.max().strftime("%Y-%m"), "months": len(r),
        "return / yr": r.mean() * periods, "volatility": r.std() * periods ** 0.5,
        "Sharpe": sr * periods ** 0.5, "t (mean)": sr * len(r) ** 0.5,
        "max drawdown": max_drawdown(r), "worst month": r.min(), "skew": stats.skew(r),
        "excess kurtosis": stats.kurtosis(r), "% months > 0": (r > 0).mean(),
    })


def information_ratio(r: pd.Series, benchmark: pd.Series, periods: int = 12) -> float:
    active = (r - benchmark).dropna()
    return float(active.mean() / active.std() * periods ** 0.5)


def bootstrap_sharpe(r: pd.Series, block: int = 12, n: int = 5000, periods: int = 12, seed: int = 0,
                     level: float = 0.95) -> tuple[float, float]:
    """Confidence interval for the annualized Sharpe from a circular block bootstrap.

    Blocks of `block` months keep autocorrelation and volatility clustering that
    an i.i.d. bootstrap would destroy.
    """
    x = r.dropna().to_numpy()
    T = len(x)
    rng = np.random.default_rng(seed)
    k = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=(n, k))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :T] % T
    samples = x[idx]
    sr = samples.mean(axis=1) / samples.std(axis=1, ddof=1) * periods ** 0.5
    lo, hi = np.quantile(sr, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def probabilistic_sharpe(r: pd.Series, benchmark_sr: float = 0.0) -> float:
    """P(true per-period Sharpe > benchmark_sr), allowing for skew and fat tails (Bailey and Lopez de Prado 2012)."""
    x = r.dropna()
    sr = x.mean() / x.std()
    g3, g4 = stats.skew(x), stats.kurtosis(x, fisher=False)
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return float(stats.norm.cdf((sr - benchmark_sr) * np.sqrt(len(x) - 1) / denom))


def expected_max_sharpe(trial_sharpes: np.ndarray) -> float:
    """Expected maximum per-period Sharpe among N unskilled trials with the observed spread of Sharpes."""
    n = len(trial_sharpes)
    sd = np.std(trial_sharpes, ddof=1)
    return float(sd * ((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n)
                       + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n * np.e))))


def deflated_sharpe(r: pd.Series, trial_sharpes: np.ndarray) -> float:
    """Probability that the chosen strategy's Sharpe beats the best of the trials by luck alone.

    `trial_sharpes` are the per-period (not annualized) Sharpes of every variant tried.
    """
    return probabilistic_sharpe(r, expected_max_sharpe(np.asarray(trial_sharpes)))


def break_even_cost_bps(gross: pd.Series, turnover: pd.Series) -> float:
    """One-way trading cost at which the average net return is zero."""
    return float(gross.mean() / turnover.mean() * 1e4)
