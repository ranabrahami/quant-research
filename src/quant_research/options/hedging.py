"""Delta hedging an option along a real price path.

Sell (or buy) an option at an implied volatility, then each day hold delta
units of the underlying (Black-Scholes delta at the same implied vol), financing
the hedge at the risk-free rate. At expiry the position's value is the hedged
P&L. For a short option the P&L is approximately

    sum over days of  1/2 Gamma_t S_t^2 (sigma_imp^2 dt - (dS_t / S_t)^2)

so the seller earns when realized variance is below implied variance and loses
when it is above, weighted by gamma (most when the price sits near the strike).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research.options import black_scholes as bs

DAYS_PER_YEAR = 252


def hedge(path: pd.Series, K: float, sigma_imp: float, r: float, q: float = 0.0, kinds=("call", "put"),
          position: int = -1) -> dict:
    """Delta-hedge `position` units of each option in `kinds` (a straddle by default) along daily closes `path`.

    The option expires on the last day of `path`. Returns the P&L, the premium, and the
    theta-gamma approximation of the P&L, all in currency per unit of underlying.
    """
    S = path.to_numpy(dtype=float)
    steps = len(S) - 1
    T = steps / DAYS_PER_YEAR
    dt = 1 / DAYS_PER_YEAR
    tau = T - np.arange(steps) * dt                               # time to expiry at each rebalance
    premium = sum(float(bs.price(S[0], K, T, r, sigma_imp, q, k)) for k in kinds)
    deltas = sum(bs.greeks(S[:-1], K, tau, r, sigma_imp, q, k)["delta"] for k in kinds)
    gammas = sum(bs.greeks(S[:-1], K, tau, r, sigma_imp, q, k)["gamma"] for k in kinds)
    hedge_units = -position * deltas                              # short option -> long delta
    cash = -position * premium - hedge_units[0] * S[0]
    for t in range(1, steps + 1):
        cash *= np.exp(r * dt)
        cash += hedge_units[t - 1] * S[t] * np.exp(q * dt) - hedge_units[t - 1] * S[t]   # dividends on the hedge
        if t < steps:
            cash -= (hedge_units[t] - hedge_units[t - 1]) * S[t]
    payoff = sum(max(S[-1] - K, 0) if k == "call" else max(K - S[-1], 0) for k in kinds)
    pnl = cash + hedge_units[-1] * S[-1] + position * payoff
    rets = np.diff(S) / S[:-1]
    approx = -position * 0.5 * gammas * S[:-1] ** 2 * (sigma_imp ** 2 * dt - rets ** 2)
    return {"pnl": float(pnl), "premium": premium, "approx_pnl": float(approx.sum()),
            "realized_vol": float(np.sqrt(np.mean(rets ** 2) * DAYS_PER_YEAR)), "spot": float(S[0])}
