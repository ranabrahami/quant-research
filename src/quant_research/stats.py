"""Regression helpers shared by every study.

Returns are autocorrelated and heteroskedastic, so standard errors are
Newey-West (HAC). The lag length follows the common rule of thumb
floor(4 * (T / 100) ** (2/9)) unless given.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def newey_west_lags(n: int) -> int:
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def ols_hac(y: pd.Series, X: pd.DataFrame, lags: int | None = None):
    """OLS of y on X plus a constant, with Newey-West standard errors. Aligns and drops missing rows."""
    data = pd.concat([y.rename("y"), X], axis=1, join="inner").dropna()
    lags = newey_west_lags(len(data)) if lags is None else lags
    model = sm.OLS(data["y"], sm.add_constant(data.drop(columns="y")))
    return model.fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def rolling_ols(y: pd.Series, X: pd.DataFrame, window: int) -> pd.DataFrame:
    """Coefficients from a regression on each trailing window (point estimates only)."""
    data = pd.concat([y.rename("y"), X], axis=1, join="inner").dropna()
    Xc = sm.add_constant(data.drop(columns="y")).to_numpy()
    yv = data["y"].to_numpy()
    out = []
    for end in range(window, len(data) + 1):
        beta, *_ = np.linalg.lstsq(Xc[end - window:end], yv[end - window:end], rcond=None)
        out.append(beta)
    cols = ["const"] + list(X.columns)
    return pd.DataFrame(out, index=data.index[window - 1:], columns=cols)
