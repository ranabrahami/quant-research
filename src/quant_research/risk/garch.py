"""GARCH(1,1) with Student-t shocks: one-day VaR/ES and Monte Carlo multi-day losses.

    r_t = mu + sigma_t e_t,  e_t ~ standardized t(nu)
    sigma2_t = omega + alpha (r_{t-1} - mu)^2 + beta sigma2_{t-1}

Volatility clusters (a big move raises tomorrow's variance) and shocks have fat
tails. Parameters are re-estimated every `refit` days on the trailing
`window` days; between refits they are held fixed and the variance is filtered
forward with each new return, so forecasts use only past data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

SCALE = 100.0                    # arch fits better on percent returns


@dataclass
class GarchParams:
    mu: float
    omega: float
    alpha: float
    beta: float
    nu: float

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta


def t_quantile(p: float, nu: float) -> float:
    """Quantile of a t distribution scaled to unit variance."""
    return stats.t.ppf(p, nu) * np.sqrt((nu - 2) / nu)


def t_es(p: float, nu: float) -> float:
    """Expected value of a unit-variance t below its p-quantile (negative)."""
    q = stats.t.ppf(p, nu)
    tail = -(stats.t.pdf(q, nu) / p) * (nu + q ** 2) / (nu - 1)
    return tail * np.sqrt((nu - 2) / nu)


def rolling_forecasts(r: pd.Series, window: int = 2500, refit: int = 250, level: float = 0.99,
                      start: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """var_garch_t / es_garch_t / sigma for each day, plus the parameters of each refit."""
    p = 1 - level
    x = r * SCALE
    start = window if start is None else start
    rows, fits = [], []
    for s in range(start, len(x), refit):
        e = min(s + refit, len(x))
        fit = arch_model(x.iloc[s - window:s], mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
        par = fit.params
        fixed = arch_model(x.iloc[s - window:e], mean="Constant", vol="GARCH", p=1, q=1, dist="t").fix(par)
        sigma = fixed.conditional_volatility.iloc[window:] / SCALE          # sigma_t uses data to t-1
        nu, mu = par["nu"], par["mu"] / SCALE
        rows.append(pd.DataFrame({"sigma": sigma, "var_garch_t": -(mu + sigma * t_quantile(p, nu)),
                                  "es_garch_t": -(mu + sigma * t_es(p, nu))}))
        fits.append({"date": x.index[s], "mu": mu, "omega": par["omega"] / SCALE ** 2,
                     "alpha": par["alpha[1]"], "beta": par["beta[1]"], "nu": nu})
    return pd.concat(rows), pd.DataFrame(fits).set_index("date")


def simulate_losses(params: GarchParams, sigma2_next: float, horizon: int, n: int = 10_000,
                    seed: int | None = None) -> np.ndarray:
    """Monte Carlo: `n` simulated cumulative returns over `horizon` days from tomorrow's variance."""
    rng = np.random.default_rng(seed)
    shocks = rng.standard_t(params.nu, size=(n, horizon)) * np.sqrt((params.nu - 2) / params.nu)
    s2 = np.full(n, sigma2_next)
    wealth = np.ones(n)
    for h in range(horizon):
        eps = np.sqrt(s2) * shocks[:, h]
        wealth *= 1 + params.mu + eps
        s2 = params.omega + params.alpha * eps ** 2 + params.beta * s2
    return wealth - 1
