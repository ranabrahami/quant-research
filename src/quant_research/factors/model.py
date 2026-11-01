"""Time-series factor regressions: what explains an asset's returns?

    r_asset - r_f = alpha + sum_k beta_k * factor_k + error

alpha is the average return the factors don't explain (monthly; annualized x12
in reports). betas are exposures ("loadings"); a value investor loads
positively on HML, a quality investor on RMW, and so on.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_research.stats import newey_west_lags, ols_hac, rolling_ols

FF5_MOM = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
CAPM = ["Mkt-RF"]
NAMES = {"Mkt-RF": "Market", "SMB": "Size (small minus big)", "HML": "Value (high minus low B/M)",
         "RMW": "Profitability (robust minus weak)", "CMA": "Investment (conservative minus aggressive)",
         "Mom": "Momentum (up minus down)", "const": "Alpha"}


@dataclass
class FactorFit:
    table: pd.DataFrame          # coef, se, t, p per term (const = alpha), monthly units
    r2: float
    adj_r2: float
    n: int
    start: pd.Timestamp
    end: pd.Timestamp
    lags: int
    residual_vol: float          # annualized tracking error vs the factor model

    @property
    def alpha_annual(self) -> float:
        return float(self.table.at["const", "coef"] * 12)

    @property
    def alpha_t(self) -> float:
        return float(self.table.at["const", "t"])

    def summary(self) -> pd.DataFrame:
        t = self.table.copy()
        t.insert(0, "factor", [NAMES.get(i, i) for i in t.index])
        t.loc["const", ["coef", "se"]] *= 12               # alpha shown per year
        return t


def fit(excess_returns: pd.Series, factors: pd.DataFrame, terms: list[str] = FF5_MOM,
        lags: int | None = None) -> FactorFit:
    data = pd.concat([excess_returns.rename("y"), factors[terms]], axis=1, join="inner").dropna()
    lags = newey_west_lags(len(data)) if lags is None else lags
    res = ols_hac(data["y"], data[terms], lags)
    table = pd.DataFrame({"coef": res.params, "se": res.bse, "t": res.tvalues, "p": res.pvalues})
    return FactorFit(table, float(res.rsquared), float(res.rsquared_adj), int(res.nobs), data.index.min(),
                     data.index.max(), lags, float(res.resid.std() * 12 ** 0.5))


def attribution(excess_returns: pd.Series, factors: pd.DataFrame, result: FactorFit,
                terms: list[str] = FF5_MOM) -> pd.Series:
    """Average annual excess return split into beta x factor premium per factor, plus alpha."""
    data = pd.concat([excess_returns.rename("y"), factors[terms]], axis=1, join="inner").dropna()
    parts = {NAMES[k]: result.table.at[k, "coef"] * data[k].mean() * 12 for k in terms}
    parts["Alpha"] = result.alpha_annual
    out = pd.Series(parts)
    out["Total (actual average excess return)"] = data["y"].mean() * 12
    return out


def rolling_exposures(excess_returns: pd.Series, factors: pd.DataFrame, window: int = 60,
                      terms: list[str] = FF5_MOM) -> pd.DataFrame:
    """Loadings on each trailing `window` months, with alpha annualized."""
    out = rolling_ols(excess_returns, factors[terms], window)
    out["const"] = out["const"] * 12
    return out.rename(columns={"const": "alpha_annual"})


def excess(returns: pd.Series, factors: pd.DataFrame) -> pd.Series:
    """Asset returns minus the risk-free rate, on months both exist."""
    data = pd.concat([returns.rename("r"), factors["RF"]], axis=1, join="inner").dropna()
    return (data["r"] - data["RF"]).rename(returns.name)
