"""Weights in, net returns out: a monthly portfolio backtester with trading and borrow costs.

Timing. A row of target weights dated t is traded at the close of month t, so it
earns the returns of month t+1 onwards; nothing dated t can see month t+1.

Drift. Between rebalances positions move with their own returns: a weight w_i
becomes w_i (1 + r_i) / (1 + r_p), where r_p is the portfolio return on capital.
Turnover is measured against these drifted weights, not the previous targets,
so a strategy that "holds" still pays to restore its weights.

Costs. Trading cost is cost_bps x one-way turnover (sum of |trades| as a share
of capital), charged in the month after the trade. Borrow is borrow_bps a year
on the short exposure. Weights are fractions of capital: a long-short
portfolio with +1 long and -1 short is self-financing and its return is the
long leg minus the short leg.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Backtest:
    returns: pd.DataFrame     # gross, trading_cost, borrow_cost, net, turnover, long, short (exposures held)
    held: pd.DataFrame        # weights held during each month (after the prior close's trade)

    @property
    def net(self) -> pd.Series:
        return self.returns["net"]

    @property
    def gross(self) -> pd.Series:
        return self.returns["gross"]


def run(weights: pd.DataFrame, returns: pd.DataFrame, cost_bps: float = 0.0, borrow_bps: float = 0.0) -> Backtest:
    """Backtest target `weights` (rebalance dates x assets) on monthly `returns` (month-end x assets)."""
    weights = weights.reindex(columns=returns.columns).fillna(0.0)
    unknown = weights.index.difference(returns.index)
    if len(unknown):
        raise ValueError(f"Weight dates not in the return index, e.g. {unknown[0]:%Y-%m-%d}")
    dates = returns.index[returns.index >= weights.index.min()]
    w = pd.Series(0.0, index=returns.columns)
    traded = 0.0
    rows, held = [], []
    for i, t in enumerate(dates):
        if i > 0:
            r = returns.loc[t]
            active = w != 0
            if r[active].isna().any():
                raise ValueError(f"Missing return for a held asset in {t:%Y-%m}: {list(r[active & r.isna()].index)}")
            r = r.fillna(0.0)
            gross = float((w * r).sum())
            short = float(-w.clip(upper=0).sum())
            trading = traded * cost_bps / 1e4
            borrow = short * borrow_bps / 1e4 / 12
            rows.append({"date": t, "gross": gross, "trading_cost": trading, "borrow_cost": borrow,
                         "net": gross - trading - borrow, "turnover": traded,
                         "long": float(w.clip(lower=0).sum()), "short": short})
            held.append(w.rename(t))
            w = w * (1 + r) / (1 + gross)
        traded = 0.0
        if t in weights.index:
            target = weights.loc[t]
            traded = float((target - w).abs().sum())
            w = target.copy()
    return Backtest(pd.DataFrame(rows).set_index("date"), pd.DataFrame(held))


def overlapping(targets: pd.DataFrame, holding_months: int, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Monthly weights that average every target formed in the last `holding_months` months.

    The Jegadeesh-Titman construction: each target is a cohort held for
    `holding_months`, and the portfolio is the equal-weighted mix of live cohorts.
    """
    months = index[index >= targets.index.min()]
    out = []
    for t in months:
        live = targets[(targets.index <= t) & (targets.index > t - pd.offsets.MonthEnd(holding_months))]
        if len(live):
            out.append(live.mean().rename(t))
    return pd.DataFrame(out)


def daily_returns(held: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
    """Daily returns of a portfolio rebalanced monthly: `held` rows are month-ends, the weights held during that month.

    Within each month positions drift with their own daily returns, so the daily
    returns compound to the monthly backtest's return (up to differences between
    the data provider's daily and monthly files). Missing daily returns of held
    assets count as zero, the same convention as the monthly data gap.
    """
    out = []
    for m, w in held.iterrows():
        days = daily.loc[(daily.index > m - pd.offsets.MonthEnd(1)) & (daily.index <= m), w.index[w != 0]]
        if days.empty:
            continue
        growth = (1 + days.fillna(0.0)).cumprod()
        value = 1 + ((growth - 1) * w[days.columns]).sum(axis=1)
        out.append(value / value.shift(1, fill_value=1.0) - 1)
    return pd.concat(out).rename("portfolio")
