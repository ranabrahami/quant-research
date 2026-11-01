"""Trading one pair over a window, and a rolling strategy of overlapping portfolios.

Spread. For distance pairs, the difference of normalized prices; for
cointegrated pairs, log A - alpha - beta log B. Its mean and standard deviation
come from the formation window only.

Rules. At each close, if the pair is flat and |z| > entry, open: long the
cheap leg and short the rich leg, $1 each ("dollar" hedge) or, for
cointegrated pairs, in the ratio 1 : beta so the position matches the spread
being traded ("beta" hedge, scaled to the same $2 gross). Close when z crosses zero, or at the end
of the window. A decision at day t's close earns from day t+1.

Accounting. Each selected pair is allotted $1 of capital whether or not it
trades ("committed capital", as in Gatev et al.), so the portfolio return is
total pair P&L divided by the number of pairs. Positions drift with prices:
the P&L of an open trade is side x (wA (A_t/A_open - 1) - wB (B_t/B_open - 1)).
Trading cost is charged on the value of each leg traded; borrow accrues daily
on the short leg.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_research.pairs import selection as sel

TRADING_DAYS = 252


@dataclass
class Spec:
    method: str = "distance"            # or "cointegration"
    formation: int = 252
    trading: int = 126
    n_pairs: int = 20
    entry: float = 2.0
    max_p: float = 0.05                 # cointegration: only pairs with p below this
    hedge: str = "dollar"               # or "beta": legs sized 1 : beta (cointegration only)
    cost_bps: float = 0.0               # per leg traded
    borrow_bps: float = 0.0             # a year, on the short leg
    step: int = 21                      # a new portfolio every month


def spread_params(form: pd.DataFrame, a: str, b: str, method: str, coint_row=None):
    if method == "distance":
        n = sel.normalized(form)
        s = n[a] - n[b]
        return {"kind": "distance", "base_a": form[a].iloc[0], "base_b": form[b].iloc[0],
                "mean": s.mean(), "sd": s.std()}
    s = np.log(form[a]) - coint_row["alpha"] - coint_row["beta"] * np.log(form[b])
    return {"kind": "cointegration", "alpha": coint_row["alpha"], "beta": coint_row["beta"],
            "mean": s.mean(), "sd": s.std()}


def spread(prices_a: pd.Series, prices_b: pd.Series, par: dict) -> pd.Series:
    if par["kind"] == "distance":
        s = prices_a / par["base_a"] - prices_b / par["base_b"]
    else:
        s = np.log(prices_a) - par["alpha"] - par["beta"] * np.log(prices_b)
    return (s - par["mean"]) / par["sd"]


def trade_pair(pa: pd.Series, pb: pd.Series, z: pd.Series, entry: float, cost_bps: float = 0.0,
               borrow_bps: float = 0.0, wa: float = 1.0, wb: float = 1.0) -> tuple[pd.Series, list[dict]]:
    """Daily P&L on $1 committed capital, and a log of trades. `wa`, `wb`: dollars in each leg at entry."""
    A, B, Z = pa.to_numpy(), pb.to_numpy(), z.to_numpy()
    n = len(A)
    pnl = np.zeros(n)
    trades = []
    side, ia, ib, opened, value_prev = 0, 0.0, 0.0, 0, 0.0
    cost, borrow = cost_bps / 1e4, borrow_bps / 1e4 / TRADING_DAYS
    for t in range(n):
        if side != 0:
            leg_a, leg_b = wa * A[t] / ia, wb * B[t] / ib          # current dollar value of each leg
            value = side * ((leg_a - wa) - (leg_b - wb))
            short_leg = leg_b if side == 1 else leg_a
            pnl[t] += value - value_prev - borrow * short_leg
            value_prev = value
        last = t == n - 1
        crossed = side != 0 and t > opened and (np.sign(Z[t]) == side or Z[t] == 0)
        if side != 0 and (crossed or last):
            pnl[t] -= cost * (leg_a + leg_b)
            trades.append({"open": pa.index[opened], "close": pa.index[t], "days": t - opened,
                           "side": side, "converged": bool(crossed), "pnl": float(pnl[opened:t + 1].sum())})
            side = 0
        elif side == 0 and not last and abs(Z[t]) > entry:
            side = -1 if Z[t] > 0 else 1                     # z > 0: A rich -> short A, long B
            ia, ib, opened, value_prev = A[t], B[t], t, 0.0
            pnl[t] -= cost * (wa + wb)
    return pd.Series(pnl, index=pa.index), trades


def run(prices: pd.DataFrame, spec: Spec, groups: dict[str, list[str]] | None = None,
        start: str | None = None) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Daily returns of the overlapping-portfolio strategy, plus the selections and the trade log."""
    days = prices.index
    first = spec.formation if start is None else max(spec.formation, days.searchsorted(pd.Timestamp(start)))
    cohorts, picks, log = [], [], []
    for s in range(first, len(days) - 1, spec.step):
        form = prices.iloc[s - spec.formation:s].dropna(axis=1)
        # A price missing later in the window is carried forward (the position is held, not
        # dropped): requiring complete future data would select on information not yet known.
        trade = prices.iloc[s:s + spec.trading][form.columns].ffill()
        pairs = sel.candidate_pairs(groups, list(form.columns))
        if spec.method == "distance":
            chosen = [(p, None) for p in sel.distances(form, pairs).index[:spec.n_pairs]]
        else:
            ct = sel.cointegration(form, pairs)
            ct = ct[ct["p"] < spec.max_p].head(spec.n_pairs)
            chosen = list(ct.iterrows())
        total = pd.Series(0.0, index=trade.index)
        for (a, b), row in chosen:
            par = spread_params(form, a, b, spec.method, row)
            z = spread(trade[a], trade[b], par)
            wa, wb = 1.0, 1.0
            if spec.hedge == "beta" and row is not None:
                if row["beta"] <= 0:
                    continue                                 # not a spread trade; the slot stays in cash
                wa, wb = 2 / (1 + row["beta"]), 2 * row["beta"] / (1 + row["beta"])
            pnl, trades = trade_pair(trade[a], trade[b], z, spec.entry, spec.cost_bps, spec.borrow_bps, wa, wb)
            total += pnl
            log += [dict(t, cohort=days[s], a=a, b=b) for t in trades]
            picks.append({"cohort": days[s], "a": a, "b": b, **({"p": row["p"]} if row is not None else {})})
        cohorts.append(total / spec.n_pairs)                 # committed capital: unused slots earn zero
    daily = pd.concat(cohorts, axis=1).mean(axis=1)          # average of live cohorts each day
    return daily.rename(spec.method), pd.DataFrame(picks), pd.DataFrame(log)
