"""One-day Value at Risk and Expected Shortfall forecasts.

VaR at level 99% is the loss exceeded on 1% of days; Expected Shortfall (ES) is
the average loss on those days. Both are reported as positive numbers (a loss
of 3% is 0.03). Every forecast for day t uses returns up to day t-1 only.

Models, from simplest to richest:
- normal: rolling mean and standard deviation, normal quantile
- historical: the empirical quantile of the last `window` days
- ewma: RiskMetrics exponentially weighted volatility (lambda 0.94), normal quantile
- fhs: filtered historical simulation, the empirical quantile of returns scaled by
  their EWMA volatility, rescaled by today's volatility (fat tails + volatility clustering)
- garch_t: GARCH(1,1) with Student-t shocks, re-estimated once a year (see garch.py)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

LAMBDA = 0.94


def ewma_vol(r: pd.Series, lam: float = LAMBDA, seed_window: int = 250) -> pd.Series:
    """sigma_t from returns up to t-1: sigma2_t = lam sigma2_{t-1} + (1 - lam) r_{t-1}^2."""
    x = r.to_numpy()
    s2 = np.empty(len(x))
    s2[0] = np.nanvar(x[:seed_window])
    for t in range(1, len(x)):
        s2[t] = lam * s2[t - 1] + (1 - lam) * x[t - 1] ** 2
    return pd.Series(np.sqrt(s2), index=r.index)


def normal_var_es(mu, sigma, level: float = 0.99):
    z = stats.norm.ppf(1 - level)
    return -(mu + sigma * z), -(mu - sigma * stats.norm.pdf(z) / (1 - level))


def _empirical_es(window_values: np.ndarray, level: float) -> float:
    q = np.quantile(window_values, 1 - level)
    return -window_values[window_values <= q].mean()


def forecasts(r: pd.Series, window: int = 500, level: float = 0.99) -> pd.DataFrame:
    """VaR and ES forecasts for each day from the four non-GARCH models (columns var_<model>, es_<model>)."""
    p = 1 - level
    past = r.shift(1)
    out = pd.DataFrame(index=r.index)
    mu, sd = past.rolling(window).mean(), past.rolling(window).std()
    out["var_normal"], out["es_normal"] = normal_var_es(mu, sd, level)
    out["var_historical"] = -past.rolling(window).quantile(p)
    out["es_historical"] = past.rolling(window).apply(_empirical_es, raw=True, args=(level,))
    vol = ewma_vol(r)
    out["var_ewma"], out["es_ewma"] = normal_var_es(0.0, vol, level)
    z = (r / vol).shift(1)                              # standardized past returns
    out["var_fhs"] = -z.rolling(window).quantile(p) * vol
    out["es_fhs"] = z.rolling(window).apply(_empirical_es, raw=True, args=(level,)) * vol
    out.iloc[:window] = np.nan
    return out
